"""P4-Pilot finite state runner. Manual-launch-only <=40 cuda:0 forwards; no CPU replay.
No retry/resume entry: any failure seals failure.json and stops future forward calls.
"""
import os,sys,time,copy,random,pickle,traceback,signal,resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pilot_common import *
import torch
from collector import Collector
from bounded_offset_cache_proposal import BoundedOffsetDataset

ACTIVE={}
STATE={'phase':'A','forward_attempted':0,'forward_completed':0,'PPU_used':False,'completed_samples':[],'cached_samples':[]}

def checkpoint_state():
 STATE['updated_epoch']=time.time()
 write_json(R/'STATE.json',STATE,exclusive=False)
def fail(exc):
 if ACTIVE.get('collector') is not None:
  c=ACTIVE['collector']
  try:
   target=R/'diagnostics'/('partial_forward_'+str(STATE['forward_attempted']))
   if not target.with_suffix('.npz').exists():write_npz(target.with_suffix('.npz'),**c.arrays)
   if not target.with_suffix('.json').exists():write_json(target.with_suffix('.json'),{'capture':c.meta,'trace_count':c.trace_count,'failure':str(exc)})
   if torch.cuda.is_initialized():write_json(R/'PPU_FAILURE_RESOURCE.json',{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'resource':gpu_status()})
  except Exception as save_error:event('FAILURE_EVIDENCE_WRITE_ERROR',message=str(save_error))
 STATE['status']='STOPPED';STATE['failure_type']=type(exc).__name__;STATE['failure']=str(exc)
 checkpoint_state();write_json(R/'FAILURE.json',{'state':STATE,'traceback':traceback.format_exc(),'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},exclusive=not (R/'FAILURE.json').exists())
 event('STOPPED',**STATE)

def alarm(signum,frame):raise TimeoutError('STAGE_HARD_TIMEOUT')
def set_deadline(seconds):signal.signal(signal.SIGALRM,alarm);signal.alarm(seconds)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.random.get_rng_state()}
def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.random.set_rng_state(s['torch_cpu'])
def torchsave(p,x,limit):
 p=Path(p)
 if p.exists():raise FileExistsError(p)
 tmp=p.with_suffix(p.suffix+'.tmp');torch.save(x,tmp)
 if tmp.stat().st_size>limit:raise RuntimeError('CACHE_ITEM_LIMIT')
 os.replace(tmp,p)

def buffer_hash(model):return digest({k:v for k,v in model.named_buffers()})
def normalize_output(result):
 pred,scores,gt=result
 if pred is None:pred=torch.empty((0,8,3),dtype=torch.float32,device=gt.device);scores=torch.empty(0,dtype=torch.float32,device=gt.device)
 return pred,scores,gt

def compare_cases(a,b,label):
 deltas={}
 for name in ('cls_preds','reg_preds','dir_preds','final_corners','final_scores','GT_runtime'):
  x=a[name];y=b[name]
  if x.shape!=y.shape:raise RuntimeError(label+' SHAPE '+name)
  error=float(np.max(np.abs(x-y),initial=0.));deltas[name]={'exact':bool(np.array_equal(x,y)),'max_abs_error':error}
  tol=1e-6 if name in ('cls_preds','reg_preds','dir_preds','final_scores') else 1e-5
  if error>tol:raise RuntimeError(label+' VALUE '+name+f' {error}')
 return deltas

def rawpair(first,second):
 for key in first:
  if key.startswith('offset_raw_slot') and not np.array_equal(first[key],second.get(key)):
   raise RuntimeError('PAIRED_RAW_METAFLOW_DIFFERENT '+key)

def run():
 global STATE
 # This entry is inaccessible through the CPU preparation path.
 claim=json.load((R/'LAUNCH.json').open())
 if os.environ.get('P4_LAUNCH_NONCE')!=claim['nonce']:raise RuntimeError('MANUAL_LAUNCH_HANDSHAKE_REQUIRED')
 if (R/'PPU_START.json').exists() or (R/'FAILURE.json').exists():raise RuntimeError('REFUSE_RETRY_OR_RESUME')
 if not json.load((R/'CPU_VALIDATION.json').open())['passed']:raise RuntimeError('CPU_GATE_FAILED')
 from launch_control import verify_ready
 verify_ready(full=True)
 random.seed(303);np.random.seed(303);torch.manual_seed(303)
 from opencood.hypes_yaml import yaml_utils
 from opencood.tools import train_utils
 from opencood.data_utils.datasets import build_dataset
 hypes=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)))
 hypes['train_params']['batch_size']=1;hypes['validate_dir']=hypes['test_dir']
 if hypes['time_delay']!=3 or hypes['history_frame']!=5:raise RuntimeError('CONFIG_CHANGED')
 model=train_utils.create_model(hypes);epoch,model=train_utils.load_saved_model(str(MODEL),model);model.eval()
 if epoch!=19 or not all(p.dtype==torch.float32 for p in model.parameters()):raise RuntimeError('MODEL_PROTOCOL_CHANGED')
 dataset=build_dataset(hypes,visualize=True,train=False)
 chosen=selections();batches=json.load((R/'INPUT_PREPARATION.json').open())['samples']
 if len(chosen)!=18 or len(batches)!=18:raise RuntimeError('SAMPLE_LIMIT')
 # C/D: finite PPU stage. No retry; counter persisted before each full model call.
 STATE['phase']='C';checkpoint_state();event('RESOURCE_BEFORE_PPU',**gpu_status())
 if not torch.cuda.is_available():raise RuntimeError('PPU_UNAVAILABLE')
 ppu_start=time.monotonic();set_deadline(900)
 device=torch.device('cuda:0');model=model.to(device);torch.cuda.synchronize()
 torch.cuda.reset_peak_memory_stats();initial_buffers=buffer_hash(model)
 write_json(R/'PPU_START.json',{'epoch':time.time(),'device':str(device),'device_name':torch.cuda.get_device_name(0),'pid':os.getpid(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()})
 perrun=[];smoke_baseline={};captured_smoke={};pairmeta={}
 def execute(sid,alpha,kind,capture):
  budget(ppu_start,900);gpu_status()
  if STATE['forward_attempted']>=40:raise RuntimeError('FORWARD_LIMIT')
  item=next(x for x in batches if x['sample_idx']==sid);path=Path(item['path'])
  if sha_file(path)!=item['sha256']:raise RuntimeError('FROZEN_ARCHIVE_HASH_CHANGED')
  payload=torch.load(path,map_location='cpu',weights_only=False,mmap=False);frozen=payload['batch']
  if digest(frozen)!=payload['input_hash']:raise RuntimeError('FROZEN_VALUE_HASH_CHANGED')
  restore_rng(payload['rng'])
  # load creates independent CPU storage; clone isolates forward's dictionary mutation.
  batch=train_utils.to_device(copy.deepcopy(frozen),device)
  if digest(batch)!=payload['input_hash']:raise RuntimeError('DEVICE_INPUT_VALUES_DIFFER')
  if buffer_hash(model)!=initial_buffers:raise RuntimeError('MODEL_BUFFERS_CHANGED_BEFORE_FORWARD')
  STATE.update(current_sample_idx=sid,current_alpha=alpha,current_kind=kind)
  STATE['forward_attempted']+=1;checkpoint_state();event('FORWARD_START',sample_idx=sid,alpha=alpha,kind=kind,counter=STATE['forward_attempted'])
  t=time.monotonic()
  with torch.no_grad():
   with Collector(model,dataset.post_processor,batch,alpha,capture) as col:
    ACTIVE['collector']=col
    outputs=model(batch['ego'])
    if not outputs['cls_preds'].is_cuda or str(outputs['cls_preds'].device)!='cuda:0':raise RuntimeError('MODEL_NOT_ON_APPROVED_PPU')
    STATE['forward_completed']+=1;STATE['PPU_used']=True;checkpoint_state()
    pred,score,gt=normalize_output(dataset.post_process(batch,{'ego':outputs}))
   col.finish(outputs,(pred,score),gt)
  torch.cuda.synchronize();elapsed=time.monotonic()-t
  if buffer_hash(model)!=initial_buffers:raise RuntimeError('MODEL_BUFFERS_CHANGED_AFTER_FORWARD')
  if digest(frozen)!=payload['input_hash']:raise RuntimeError('FROZEN_BATCH_MUTATED')
  try:
   canonical,mapping,hist=verify_historical(sid,alpha,col.arrays['final_corners'],col.arrays['final_scores'],col.arrays['GT_runtime'])
  except BaseException as exc:
   name=f'FAILED_{kind}_sample_{sid}_alpha_{alpha:.1f}'
   write_npz(R/'diagnostics'/(name+'.npz'),**col.arrays)
   write_json(R/'diagnostics'/(name+'.json'),{'failure':str(exc),'metadata':col.meta,'resource':gpu_status(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'PPU_elapsed_seconds':time.monotonic()-ppu_start})
   raise
  col.add('GT_canonical',canonical);col.meta.update(historical_validation=hist,input_hash=payload['input_hash'],forward_elapsed_seconds=elapsed,kind=kind,sample_idx=sid)
  if capture:
   from opencood.utils import common_utils
   # independently audited CPU NMS helper, copied to this new run directory.
   from contracts import replay_native_nms
   replay=replay_native_nms(col.arrays['nms_input_corners'],col.arrays['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
   if not np.array_equal(replay['keep'],col.arrays['native_nms_keep']):raise RuntimeError('NMS_CPU_REPLAY_KEEP_MISMATCH')
   col.meta['NMS_replay']={k:v for k,v in replay.items() if k!='keep'};col.meta['NMS_replay']['keep_exact']=True
  name=f'{kind}_sample_{sid}_alpha_{alpha:.1f}'
  # Persist before historical mismatch checks are raised: retained by outer failure handler below.
  write_npz(R/'diagnostics'/(name+'.npz'),**col.arrays);write_json(R/'diagnostics'/(name+'.json'),col.meta)
  row={'sample_idx':sid,'alpha':alpha,'kind':kind,'elapsed_seconds':elapsed,'historical':hist,'capture':capture,'output_device':str(outputs['cls_preds'].device),'npz':name+'.npz','gpu_allocated_bytes':torch.cuda.memory_allocated(),'gpu_reserved_bytes':torch.cuda.memory_reserved(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
  perrun.append(row);event('FORWARD_VALIDATED',**row)
  arrays=col.arrays;metadata=col.meta
  ACTIVE.clear()
  del batch,outputs,pred,score,gt,col,payload,frozen
  return arrays,metadata
 # Uninstrumented baseline: alpha scaling only. Heads copied from normal output.
 for alpha in [0.,1.]:
  a,m=execute(821,alpha,'smoke_native',False);smoke_baseline[alpha]=a
 # Required core order for first selected row is [1,0].
 for alpha in [1.,0.]:
  a,m=execute(821,alpha,'core',True);captured_smoke[alpha]=a
  pairmeta[str(alpha)]=m
  write_json(R/f'SMOKE_EQUIVALENCE_alpha_{alpha:.1f}.json',{'passed':True,'comparison':compare_cases(smoke_baseline[alpha],a,'INSTRUMENTATION_EQUIVALENCE')})
 rawpair(captured_smoke[0.],captured_smoke[1.])
 for name,info0 in pairmeta['0.0']['features'].items():
  if name.startswith(('history_','fusion_input_','pose_warp_','single_control')) and info0['sha256']!=pairmeta['1.0']['features'][name]['sha256']:raise RuntimeError('UPSTREAM_PAIR_FEATURES_DIFFER '+name)
 write_json(R/'SMOKE_TEST.json',{'passed':True,'sample_idx':821,'native_forwards':2,'instrumented_core_forwards':2,'historical_outputs_equal':True,'raw_offset_and_upstream_features_equal':True})
 STATE['phase']='D';STATE['completed_samples']=[821];checkpoint_state()
 # Reverse the first core order [1,0] and compare same frozen sample exactly.
 for alpha in [0.,1.]:
  a,m=execute(821,alpha,'reverse',True)
  write_json(R/f'REVERSE_EQUIVALENCE_alpha_{alpha:.1f}.json',{'passed':True,'comparison':compare_cases(captured_smoke[alpha],a,'REVERSE_EQUIVALENCE')})
  rawpair(captured_smoke[alpha],a)
 smoke_baseline.clear();captured_smoke.clear()
 for row in chosen[1:]:
  sid=int(row['sample_idx']);order=[1.,0.] if int(row['selection_order'])%2 else [0.,1.]
  pair={}
  for alpha in order:pair[alpha]=execute(sid,alpha,'core',True)
  rawpair(pair[0.][0],pair[1.][0])
  for name,m in pair[0.][1]['features'].items():
   if name.startswith(('history_','fusion_input_','pose_warp_','single_control')) and m['sha256']!=pair[1.][1]['features'][name]['sha256']:raise RuntimeError('UPSTREAM_PAIR_FEATURES_DIFFER '+name)
  STATE['completed_samples'].append(sid);checkpoint_state();del pair
 elapsed=time.monotonic()-ppu_start
 write_json(R/'PPU_EXECUTION.json',{'passed':True,'elapsed_seconds':elapsed,'full_forward_attempted':STATE['forward_attempted'],'full_forward_completed':STATE['forward_completed'],'actual_inference_on_PPU':STATE['PPU_used'],'samples':STATE['completed_samples'],'runs':perrun,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'resource_end':gpu_status()})
 STATE['phase']='E_READY';checkpoint_state();event('PPU_COLLECTION_COMPLETE',elapsed_seconds=elapsed,forwards=STATE['forward_completed'])
 signal.alarm(0)


 model.cpu();del model;torch.cuda.empty_cache()
if __name__=='__main__':
 claim=json.load((R/'LAUNCH.json').open()) if (R/'LAUNCH.json').exists() else {}
 if not claim or os.environ.get('P4_LAUNCH_NONCE')!=claim.get('nonce') or os.getppid()!=claim.get('worker',{}).get('pid'):
  print('INTERNAL_MANUAL_LAUNCH_HANDSHAKE_REQUIRED',file=sys.stderr);sys.exit(64)
 signal.signal(signal.SIGTERM,alarm);signal.signal(signal.SIGINT,alarm)
 try:run()
 except BaseException as exc:
  signal.alarm(0);fail(exc);sys.exit(1)
