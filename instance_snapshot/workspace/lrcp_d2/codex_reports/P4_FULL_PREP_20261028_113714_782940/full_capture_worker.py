"""Future manual PPU worker. Importing this file does not initialize CUDA or run a model."""
import os,sys,json,time,copy,random,traceback,resource,signal,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pilot_common import R,MODEL,BASE,STAGES,array,digest,sha_file,gt_mapping
from minimal_collector import collect_postprocess

P=R.parent/'P4_PILOT_RETRY_20261023_234457_384594'
MAX_FORWARD=1790;MAX_SECONDS=5400;MAX_OUTPUT=16*2**30
state={'phase':'PPU_RUNNING','attempted':0,'completed':0,'core_new':0,'guards':0,'completed_samples':0,'current_sample_idx':None,'alpha':None,'shards_complete':0,'shard_manifest_hashes':{},'started_epoch':time.time()}

def atomic(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def update(**kw):
 state.update(kw);state['updated_epoch']=time.time();atomic(R/'PPU_PROGRESS.json',state)
 print(json.dumps({'phase':state['phase'],'sample_idx':state['current_sample_idx'],'alpha':state['alpha'],'attempted':state['attempted'],'completed':state['completed'],'completed_samples':state['completed_samples'],'shards_complete':state['shards_complete'],'epoch':state['updated_epoch']},ensure_ascii=False),flush=True)
def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.random.set_rng_state(s['torch_cpu'])
def out_bytes():return sum(p.stat().st_size for p in (R/'capture').rglob('*') if p.is_file())
def stop_guard(start):
 if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('PPU_WALL_CLOCK_LIMIT')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_LIMIT')
 if out_bytes()>MAX_OUTPUT:raise RuntimeError('OUTPUT_LIMIT')
 if __import__('shutil').disk_usage(R).free<10*2**30:raise RuntimeError('DISK_RESERVE')
 if state['attempted']>=MAX_FORWARD:raise RuntimeError('FORWARD_BUDGET')
 if torch.cuda.is_initialized():
  free,total=torch.cuda.mem_get_info()
  if free<4*2**30 or torch.cuda.max_memory_allocated()>8*2**30 or torch.cuda.max_memory_reserved()>12*2**30:raise RuntimeError('PPU_MEMORY_RESOURCE_LIMIT')
def save_npz(path,a):
 if path.exists():raise FileExistsError(path)
 tmp=path.with_suffix('.npz.tmp')
 with tmp.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 if out_bytes()>MAX_OUTPUT:raise RuntimeError('OUTPUT_LIMIT')
 os.replace(tmp,path)
def equal_case(a,b,label):
 for k in ['cls_preds','reg_preds','dir_preds','final_corners','final_scores','GT_canonical']:
  if k not in a or k not in b or not np.array_equal(a[k],b[k]):raise RuntimeError(label+' EXACT_MISMATCH '+k)
def resource_status():
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_LIMIT_AFTER_FORWARD')
 return {'cpu_rss_bytes':int(next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')).split()[1])*1024,'cpu_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'PPU_allocated_bytes':torch.cuda.memory_allocated(),'PPU_reserved_bytes':torch.cuda.memory_reserved(),'PPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'PPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
def run():
 claim=json.loads((R/'PPU_LAUNCH.json').read_text());nonce=os.environ.get('P4_FULL_NONCE')
 if not nonce or nonce!=claim.get('nonce') or os.getppid()!=claim.get('worker_pid'):raise RuntimeError('MANUAL_WORKER_HANDSHAKE_REQUIRED')
 from full_launch_control import verify_ready
 verify_ready(full=True)
 manifest=json.loads((R/'INPUT_MANIFEST.json').read_text());input_by_id={x['sample_idx']:x for x in manifest['samples']}
 reuse=json.loads((R/'REUSE_18_VALIDATION.json').read_text());assert reuse['passed'];reuse_ids=set(reuse['sample_ids'])
 ids=sorted(input_by_id);assert len(ids)==910 and len(reuse_ids)==18
 random.seed(303);np.random.seed(303);torch.manual_seed(303)
 from opencood.hypes_yaml import yaml_utils
 from opencood.tools import train_utils
 from opencood.data_utils.datasets import build_dataset
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));h['train_params']['batch_size']=1;h['validate_dir']=h['test_dir']
 if h['time_delay']!=3 or h['history_frame']!=5:raise RuntimeError('MODEL_TIME_PROTOCOL_MISMATCH')
 model=train_utils.create_model(h);epoch,model=train_utils.load_saved_model(str(MODEL),model);model.eval()
 if epoch!=19 or not all(p.dtype==torch.float32 for p in model.parameters()):raise RuntimeError('MODEL_PROTOCOL_MISMATCH')
 dataset=build_dataset(h,visualize=True,train=False);pp=dataset.post_processor
 if pp.params['target_args']['score_threshold']!=.2 or pp.params['nms_thresh']!=.15:raise RuntimeError('THRESHOLD_CHANGED')
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('PPU_WALL_CLOCK_LIMIT')));signal.alarm(MAX_SECONDS)
 if not torch.cuda.is_available():raise RuntimeError('PPU_UNAVAILABLE')
 device=torch.device('cuda:0');model=model.to(device);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();initial_buffers=digest({k:v for k,v in model.named_buffers()})
 def execute(sid,alpha,kind,capture):
  stop_guard(start);item=input_by_id[sid];p=Path(item['path'])
  if sha_file(p)!=item['sha256']:raise RuntimeError('FROZEN_BATCH_HASH_CHANGED')
  payload=torch.load(p,map_location='cpu',weights_only=False,mmap=False);frozen=payload['batch']
  if digest(frozen)!=item['input_hash']:raise RuntimeError('FROZEN_VALUES_CHANGED')
  restore_rng(payload['rng']);batch=train_utils.to_device(copy.deepcopy(frozen),device)
  if digest(batch)!=item['input_hash']:raise RuntimeError('DEVICE_INPUT_MISMATCH')
  if digest({k:v for k,v in model.named_buffers()})!=initial_buffers:raise RuntimeError('MODEL_BUFFERS_CHANGED')
  offsets=[]
  def hook(module,args,out):
   if tuple(out.shape)!=(1,80,80,2):raise RuntimeError('METAFLOW_SHAPE')
   offsets.append(digest(out));return out*alpha
  handle=model.meta_flow.register_forward_hook(hook)
  try:
   update(current_sample_idx=sid,alpha=alpha,kind=kind,attempted=state['attempted']+1)
   with torch.no_grad():
    heads=model(batch['ego'])
    if not heads['cls_preds'].is_cuda:raise RuntimeError('NO_PPU_MODEL_OUTPUT')
    state['completed']+=1;update()
    if capture:
     arrays,metadata=collect_postprocess(pp,batch,heads)
     gt=pp.generate_gt_bbx(batch)
    else:
     pred,scores,gt=dataset.post_process(batch,{'ego':heads})
     if pred is None:pred=torch.empty((0,8,3),dtype=torch.float32,device=device);scores=torch.empty(0,dtype=torch.float32,device=device)
     arrays={k:array(heads[k]) for k in ['cls_preds','reg_preds','dir_preds']};arrays.update(final_corners=array(pred),final_scores=array(scores));metadata={}
   torch.cuda.synchronize()
   free,total=torch.cuda.mem_get_info()
   if free<4*2**30 or torch.cuda.max_memory_allocated()>8*2**30 or torch.cuda.max_memory_reserved()>12*2**30:raise RuntimeError('PPU_MEMORY_RESOURCE_LIMIT_AFTER_FORWARD')
  finally:handle.remove()
  if digest(frozen)!=item['input_hash'] or digest({k:v for k,v in model.named_buffers()})!=initial_buffers:raise RuntimeError('FROZEN_INPUT_OR_MODEL_BUFFERS_MUTATED')
  oldpath=BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz'
  try:
   with np.load(oldpath,allow_pickle=False) as old:
    gt_array=array(gt);mapping,gm=gt_mapping(old['gt_boxes'],gt_array);arrays['GT_canonical']=gt_array[mapping];arrays['GT_runtime']=gt_array
    if arrays['final_corners'].shape!=old['pred_boxes'].shape or arrays['final_scores'].shape!=old['pred_scores'].shape:raise RuntimeError('HISTORICAL_PRED_SHAPE_MISMATCH')
    if not np.allclose(arrays['final_corners'],old['pred_boxes'],rtol=0,atol=1e-5) or not np.allclose(arrays['final_scores'],old['pred_scores'],rtol=0,atol=1e-6):raise RuntimeError('HISTORICAL_PRED_VALUE_MISMATCH')
  except BaseException as exc:
   failure_dir=R/'capture';failure_dir.mkdir(exist_ok=True)
   partial=failure_dir/f'FAILED_attempt_{state["attempted"]}_sample_{sid}_alpha_{alpha:.1f}.npz'
   try:save_npz(partial,arrays)
   except BaseException as save_error:print('PARTIAL_CAPTURE_SAVE_FAILED '+str(save_error),file=sys.stderr,flush=True)
   atomic(failure_dir/f'FAILED_attempt_{state["attempted"]}.json',{'sample_idx':sid,'alpha':alpha,'kind':kind,'input_hash':item['input_hash'],'error':str(exc),'partial_path':str(partial),'raw_offset_hashes':offsets})
   raise
  metadata.update(sample_idx=sid,alpha=alpha,kind=kind,input_hash=item['input_hash'],GT_mapping=gm,raw_offset_hashes=offsets,resource=resource_status())
  if kind=='core':state['core_new']+=1
  else:state['guards']+=1
  update(**resource_status());del batch,payload,frozen,heads,gt
  return arrays,metadata
 # Guard checks count six new forwards, even though sample 821 is a reused core pair.
 native={};instrumented={};guard_meta={}
 for alpha in [0.,1.]:
  native[alpha],guard_meta[('native',alpha)]=execute(821,alpha,'guard_native',False)
 for alpha in [1.,0.]:
  a,m=execute(821,alpha,'guard_instrumented',True);equal_case(native[alpha],a,'SMOKE_NATIVE')
  with np.load(P/f'diagnostics/core_sample_821_alpha_{alpha:.1f}.npz',allow_pickle=False) as old:equal_case(a,old,'PILOT_CORE')
  instrumented[alpha]=a;guard_meta[('instrumented',alpha)]=m
  if m['raw_offset_hashes']!=guard_meta[('native',alpha)]['raw_offset_hashes']:raise RuntimeError('SMOKE_RAW_OFFSET_MISMATCH')
 if guard_meta[('instrumented',0.)]['raw_offset_hashes']!=guard_meta[('instrumented',1.)]['raw_offset_hashes']:raise RuntimeError('SMOKE_PAIRED_RAW_OFFSET_MISMATCH')
 for alpha in [0.,1.]:
  a,m=execute(821,alpha,'guard_reverse',True);equal_case(instrumented[alpha],a,'REVERSE')
  if m['raw_offset_hashes']!=guard_meta[('instrumented',alpha)]['raw_offset_hashes']:raise RuntimeError('REVERSE_RAW_OFFSET_MISMATCH')
 update(smoke_passed=True,guard_reverse_passed=True);del native,instrumented,guard_meta,a,m
 for shard in range(29):
  group=ids[shard*32:(shard+1)*32];shard_dir=R/'capture'/f'shard_{shard:03d}';shard_dir.mkdir(parents=True,exist_ok=False);records=[]
  for sid in group:
   if sid in reuse_ids:
    for alpha in [0.,1.]:
     f=P/f'diagnostics/core_sample_{sid}_alpha_{alpha:.1f}.npz';records.append({'sample_idx':sid,'alpha':alpha,'provenance':'pilot_reuse','path':str(f),'sha256':sha_file(f),'meta_path':str(f.with_suffix('.json')),'meta_sha256':sha_file(f.with_suffix('.json')),'input_hash':input_by_id[sid]['input_hash']})
   else:
    pair_offsets={}
    for alpha in [1.,0.]:
     a,meta=execute(sid,alpha,'core',True);pair_offsets[alpha]=meta['raw_offset_hashes'];f=shard_dir/f'sample_{sid}_alpha_{alpha:.1f}.npz';save_npz(f,a);atomic(f.with_suffix('.json'),meta)
     records.append({'sample_idx':sid,'alpha':alpha,'provenance':'new','path':str(f),'sha256':sha_file(f),'meta_path':str(f.with_suffix('.json')),'meta_sha256':sha_file(f.with_suffix('.json')),'input_hash':input_by_id[sid]['input_hash']})
     del a,meta
    if pair_offsets[0.]!=pair_offsets[1.]:raise RuntimeError('ALPHA_PAIR_RAW_OFFSET_MISMATCH')
   update(completed_samples=state['completed_samples']+1,current_sample_idx=sid,shard=shard)
  if len(records)!=2*len(group):raise RuntimeError('INCOMPLETE_SHARD')
  atomic(shard_dir/'COLLECTION_MANIFEST.json',{'shard':shard,'sample_ids':group,'records':records,'passed':True,'created_epoch':time.time()})
  state['shard_manifest_hashes'][str(shard)]=sha_file(shard_dir/'COLLECTION_MANIFEST.json');update(shards_complete=shard+1)
 if state['attempted']!=1790 or state['completed']!=1790 or state['core_new']!=1784 or state['guards']!=6 or state['completed_samples']!=910:raise RuntimeError('FINAL_FORWARD_OR_SAMPLE_COUNT')
 signal.alarm(0);update(phase='PPU_CAPTURE_COMPLETE');atomic(R/'PPU_CAPTURE_COMPLETE.json',{'passed':True,**state,'elapsed_seconds':time.monotonic()-start,'resource':resource_status()})
 model.cpu();del model;torch.cuda.empty_cache()
if __name__=='__main__':
 claim=json.loads((R/'PPU_LAUNCH.json').read_text()) if (R/'PPU_LAUNCH.json').exists() else {}
 if not claim or not os.environ.get('P4_FULL_NONCE') or os.environ.get('P4_FULL_NONCE')!=claim.get('nonce') or os.getppid()!=claim.get('worker_pid'):
  print('MANUAL_WORKER_HANDSHAKE_REQUIRED',file=sys.stderr);sys.exit(64)
 try:run()
 except BaseException as e:
  signal.alarm(0);update(phase='PPU_CAPTURE_FAILED',failure=str(e));atomic(R/'PPU_CAPTURE_FAILURE.json',{'passed':False,'error':str(e),'traceback':traceback.format_exc(),'state':state});raise
