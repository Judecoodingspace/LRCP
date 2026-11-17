"""One authorised CPU-only historical replay; fail closed, never invokes forward."""
import os,sys,time,copy,random,pickle,traceback,signal,resource,gc,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pilot_common import *
import torch
from bounded_offset_cache_proposal import BoundedOffsetDataset
from telemetry import snapshot,record,storage_inventory
P=R.parent/'P4_PILOT_RETRY_20261023_234457_384594'
LIMIT=35*2**30
START=time.monotonic()
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.random.get_rng_state()}
def alarm(signum,frame):raise TimeoutError('CPU_60_MINUTE_LIMIT')
def guard(extra=0):
 if time.monotonic()-START>3600:raise TimeoutError('CPU_60_MINUTE_LIMIT')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_8GiB_LIMIT')
 if bytes_in(R/'batches')+extra>LIMIT:raise RuntimeError('INPUT_CACHE_35GiB_LIMIT')
 if shutil.disk_usage(R).free<5*2**30:raise RuntimeError('DISK_RESERVE')
 if torch.cuda.is_initialized():raise RuntimeError('CUDA_INITIALIZED_UNEXPECTED')
def compare(a,b,path=''):
 if isinstance(a,torch.Tensor):
  if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):return path+' tensor mismatch'
 elif isinstance(a,np.ndarray):
  if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True):return path+' array mismatch'
 elif isinstance(a,dict):
  if not isinstance(b,dict) or a.keys()!=b.keys():return path+' keys mismatch'
  for k in a:
   d=compare(a[k],b[k],path+'/'+str(k))
   if d:return d
 elif isinstance(a,(tuple,list)):
  if type(a)!=type(b) or len(a)!=len(b):return path+' sequence mismatch'
  for i,(x,y) in enumerate(zip(a,b)):
   d=compare(x,y,path+'/'+str(i))
   if d:return d
 else:
  if type(a)!=type(b) or a!=b:return path+' value mismatch'
 return None

def run():
 if (R/'CPU_STARTED.json').exists():raise RuntimeError('REFUSE_CPU_RETRY')
 if shutil.disk_usage(R).free<60*2**30:raise RuntimeError('DISK_START_LIMIT')
 write_json(R/'CPU_STARTED.json',{'pid':os.getpid(),'time':time.time(),'forward_count':0,'cuda_initialized':False})
 signal.signal(signal.SIGALRM,alarm);signal.alarm(3600)
 random.seed(303);np.random.seed(303);torch.manual_seed(303)
 from opencood.hypes_yaml import yaml_utils
 from opencood.tools import train_utils
 from opencood.data_utils.datasets import build_dataset
 from torch.utils.data import DataLoader
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));h['train_params']['batch_size']=1;h['validate_dir']=h['test_dir']
 assert h['history_frame']==5 and h['time_delay']==3
 model=train_utils.create_model(h);epoch,model=train_utils.load_saved_model(str(MODEL),model);model.eval()
 if epoch!=19 or not all(p.dtype==torch.float32 for p in model.parameters()):raise RuntimeError('MODEL_SETUP_MISMATCH')
 expected=json.load((P/'MODEL_CPU_SETUP.json').open())
 if digest(rng())!=expected['CPU_RNG_hash_after_model_initialization']:raise RuntimeError('INITIAL_CPU_RNG_MISMATCH')
 if sha_file(MODEL/'net_epoch_bestval_at19.pth')!=expected['checkpoint_sha256']:raise RuntimeError('CHECKPOINT_CHANGED')
 del model;gc.collect();guard()
 common=set(json.load((BASE/'P1_common910_indices.json').open())['sample_ids']);assert len(common)==910
 pilot=json.load((P/'INPUT_PREPARATION.json').open());prior={int(x['sample_idx']):x for x in pilot['samples']};assert len(prior)==18
 info=pickle.load(open(h['test_dir'],'rb'));rows=[];equiv=[];progress={};guard()
 def collate(base,items):
  out=base(items)
  if out is None:return None
  if str(out['ego']['sample_idx']) not in common:return None
  return out
 for delay in [0,1,2,3]:
  h['time_delay']=delay;dataset=BoundedOffsetDataset(build_dataset(h,visualize=True,train=False))
  if hasattr(dataset.dataset,'__getitems__'):raise RuntimeError('PROXY_BYPASSED')
  def callback(when,index):
   guard()
   if when=='AFTER_GETITEM':
    for k,entry in dataset.dataset.scene_database[index].items():
     if k=='prev':continue
     offset=entry.get('params',{}).get('offset')
     if offset is not None and offset.untyped_storage().nbytes()>offset.numel()*offset.element_size():raise RuntimeError('PERSISTENT_OFFSET_STORAGE_RETAINED')
   if index%50==0 and when=='AFTER_GETITEM':record(when,delay=delay,sample_idx=index)
  dataset.audit_callback=callback
  loader=DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False,collate_fn=lambda items,base=dataset.collate_batch_test:collate(base,items),pin_memory=False,drop_last=False)
  event('CPU_DELAY_START',delay=delay);record('CPU_DELAY_START',delay=delay);count=0
  for batch in loader:
   guard()
   if batch is None:continue
   sid=int(batch['ego']['sample_idx']);count+=1
   if delay==3:
    ego=batch['ego'];current_gt=array(dataset.post_processor.generate_gt_bbx(batch));old=historical(sid,0.)['gt_boxes'];mapping,mmeta=gt_mapping(old,current_gt)
    slots={str(k):{'physical_cav_ids':v['physical_cav_ids'],'valid_mask':array(v['valid_mask']).tolist(),'time_delay':array(v['time_delay']).tolist()} for k,v in ego['calibrate_data'].items()}
    for v in slots.values():
     if v['valid_mask'][0] and v['time_delay'][0]!=300:raise RuntimeError('TIME_CODE_MISMATCH')
    clean=copy.deepcopy(batch);clean['ego'].pop('origin_lidar',None);storage=storage_inventory(clean)
    if storage['unique_storage_bytes']>2**30 or storage['shared_tensor_count']:raise RuntimeError('BATCH_STORAGE_LIMIT_OR_SHARED')
    current_rng=rng();dh=digest(clean);rh=digest(current_rng)
    if sid in prior:
     oldrec=prior[sid];op=Path(oldrec['path'])
     if sha_file(op)!=oldrec['sha256']:raise RuntimeError('PILOT_BATCH_HASH_CHANGED')
     oldpay=torch.load(op,map_location='cpu',weights_only=False,mmap=False)
     diff=compare(clean,oldpay['batch']);rngdiff=compare(current_rng,oldpay['rng'])
     erow={'sample_idx':sid,'content_equal':diff is None,'first_content_difference':diff,'rng_equal':rngdiff is None,'first_rng_difference':rngdiff,'new_value_hash':dh,'old_value_hash':oldrec['input_hash'],'new_rng_hash':rh,'old_rng_hash':oldrec['RNG_hash']}
     equiv.append(erow);write_json(R/'INPUT_EQUIVALENCE_PARTIAL.json',{'rows':equiv,'passed_so_far':diff is None and rngdiff is None},exclusive=False)
     if diff or rngdiff:raise RuntimeError('PILOT_INPUT_OR_RNG_MISMATCH sid='+str(sid)+' content='+str(diff)+' rng='+str(rngdiff))
     del oldpay
    payload={'batch':clean,'rng':current_rng,'input_hash':dh,'GT_mapping':mapping,'GT_mapping_validation':mmeta}
    guard(storage['unique_storage_bytes']+10*2**20)
    path=R/'batches'/f'sample_{sid}.pt';tmp=path.with_suffix('.pt.tmp')
    if path.exists() or tmp.exists():raise FileExistsError(path)
    torch.save(payload,tmp)
    if tmp.stat().st_size>2**30 or bytes_in(R/'batches')>LIMIT:raise RuntimeError('CACHE_CAP_AFTER_WRITE')
    os.replace(tmp,path);guard()
    row={'sample_idx':sid,'path':str(path),'sha256':sha_file(path),'input_hash':dh,'archive_bytes':path.stat().st_size,'RNG_hash':rh,'slots':slots,'GT_mapping':mmeta,'storage':storage,'sample_token':info[sid]['token'],'history_tokens':{str(k):info[sid]['prev_samples'][k] for k in [3,4,5,6,7]}}
    rows.append(row);write_json(R/'batches'/f'sample_{sid}.json',row)
    del clean,payload,current_rng,current_gt,old
   if count%50==0:event('CPU_REPLAY_PROGRESS',delay=delay,sample_idx=sid,common_count=count,cache_count=len(rows),cache_bytes=bytes_in(R/'batches'),elapsed=time.monotonic()-START)
  progress[str(delay)]=count;record('CPU_DELAY_COMPLETE',delay=delay,count=count,compacted_count=dataset.compacted_count);event('CPU_DELAY_COMPLETE',delay=delay,count=count)
  if count!=910:raise RuntimeError('DELAY_COMMON_SAMPLE_COUNT_MISMATCH')
  del loader,dataset,batch;gc.collect()
 assert set(str(x['sample_idx']) for x in rows)==common and len(rows)==910 and len(equiv)==18
 signal.alarm(0);guard();write_json(R/'INPUT_EQUIVALENCE.json',{'passed':True,'pilot_samples':equiv,'count':len(equiv),'content_and_rng_exact':True})
 write_json(R/'INPUT_MANIFEST.json',{'passed':True,'sample_count':len(rows),'samples':rows,'delay_counts':progress,'cache_bytes':bytes_in(R/'batches'),'elapsed_seconds':time.monotonic()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'seed':303,'model_forwards':0,'CUDA_initialized':torch.cuda.is_initialized()})
 event('CPU_PREPARATION_COMPLETE',samples=len(rows),elapsed=time.monotonic()-START)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  signal.alarm(0);meta={'passed':False,'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cached_count':len(list((R/'batches').glob('*.pt'))),'cache_bytes':bytes_in(R/'batches'),'model_forwards':0,'CUDA_initialized':torch.cuda.is_initialized()}
  write_json(R/'CPU_PREPARATION_FAILURE.json',meta);print(json.dumps(meta,ensure_ascii=False),flush=True);sys.exit(1)
