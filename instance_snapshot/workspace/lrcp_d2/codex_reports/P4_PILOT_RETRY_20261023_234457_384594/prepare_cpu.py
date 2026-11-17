"""CPU-only preparation. Never calls model forward or device initialization."""
import os,sys,time,copy,random,pickle,traceback,signal,resource,gc,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pilot_common import *
import torch
from bounded_offset_cache_proposal import BoundedOffsetDataset
from telemetry import snapshot,record,storage_inventory
STATE={'phase':'A','status':'PREPARING_CPU','forward_attempted':0,'forward_completed':0,'PPU_used':False,'completed_samples':[],'cached_samples':[]}
def checkpoint_state():write_json(R/'STATE.json',STATE,exclusive=False)
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

def run():
 global STATE
 if (R/'FAILURE.json').exists() or (R/'STATE.json').exists():raise RuntimeError('REFUSE_RESTART')
 if not json.load((R/'STATIC_CHECKS_FINAL.json').open())['passed']:raise RuntimeError('STATIC_GATE_FAILED')
 for p,h in json.load((R/'PROTECTED_HASHES_BEFORE.json').open()).items():
  if sha_file(p)!=h:raise RuntimeError('LOCKED_INPUT_CHANGED '+p)
 if shutil.disk_usage(R).free<25*2**30:raise RuntimeError('START_DISK_BELOW_25GiB')
 record('CPU_START');assert not torch.cuda.is_initialized()
 random.seed(303);np.random.seed(303);torch.manual_seed(303)
 from opencood.hypes_yaml import yaml_utils
 from opencood.tools import train_utils
 from opencood.data_utils.datasets import build_dataset
 from torch.utils.data import DataLoader
 hypes=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)))
 hypes['train_params']['batch_size']=1;hypes['validate_dir']=hypes['test_dir']
 if hypes['history_frame']!=5 or hypes['time_delay']!=3:raise RuntimeError('CONFIG_K_CHANGED')
 model=train_utils.create_model(hypes)
 epoch,model=train_utils.load_saved_model(str(MODEL),model)
 if epoch!=19:raise RuntimeError('CHECKPOINT_EPOCH_CHANGED')
 model.eval()
 write_json(R/'MODEL_CPU_SETUP.json',{'epoch':epoch,'checkpoint_sha256':sha_file(MODEL/'net_epoch_bestval_at19.pth'),'torch':torch.__version__,'numpy':np.__version__,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'CPU_RNG_hash_after_model_initialization':digest(rng()),'all_parameters_FP32':all(p.dtype==torch.float32 for p in model.parameters())})
 # B: Replay historical data path with same construction/collation order; no forward.
 STATE['phase']='B';checkpoint_state();start=time.monotonic();set_deadline(1800)
 chosen=selections();selected={int(x['sample_idx']) for x in chosen};common=set(json.load((BASE/'P1_common910_indices.json').open())['sample_ids'])
 info=pickle.load(open(hypes['test_dir'],'rb'));token_to_info={x['token']:x for x in info}
 batches=[];raw_hashes={};delay_counts={}
 def collate(base,items):
  out=base(items)
  if out is None:return None
  if str(out['ego']['sample_idx']) not in common:return None
  return out
 def hash_raw(sid):
  item=info[sid]
  for k in [0,3,4,5,6,7]:
   frame=item if k==0 else token_to_info[item['prev_samples'][k]]
   for agent in range(1,frame['agent_num']+1):
    p=frame[f'lidar_path_{agent}']
    if p not in raw_hashes:raw_hashes[p]=sha_file(p)
  p=str(Path(hypes['data_dir'])/'offset_maps'/(item['token']+'.npy'));raw_hashes[p]=sha_file(p)
 for delay in [0,1,2,3]:
  hypes['time_delay']=delay;dataset=BoundedOffsetDataset(build_dataset(hypes,visualize=True,train=False))
  if hasattr(dataset.dataset,'__getitems__'):raise RuntimeError('BATCH_FETCH_BYPASSES_PROXY')
  original_get=dataset.dataset.__class__.__getitem__
  event('CPU_DELAY_START',delay=delay)
  record('DELAY_START',delay=delay)
  loader=DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False,collate_fn=lambda items,base=dataset.collate_batch_test:collate(base,items),pin_memory=False,drop_last=False)
  dataset.audit_callback=lambda when,index:record(when,delay=delay,sample_idx=index,compacted_count=dataset.compacted_count,storage_detached_estimate=dataset.released_storage_bytes)
  count=0
  for batch in loader:
   budget(start,1800)
   if batch is None:continue
   sid=int(batch['ego']['sample_idx']);count+=1
   if delay==3 and sid in selected:
    hash_raw(sid)
    ego=batch['ego'];g=array(dataset.post_processor.generate_gt_bbx(batch))
    old=historical(sid,0.)['gt_boxes'];mapping,mmeta=gt_mapping(old,g)
    slot={str(k):{'physical_cav_ids':v['physical_cav_ids'],'valid_mask':array(v['valid_mask']).tolist(),'time_delay':array(v['time_delay']).tolist()} for k,v in ego['calibrate_data'].items()}
    for v in slot.values():
     if v['valid_mask'][0] and v['time_delay'][0]!=300:raise RuntimeError('TIME_CODE_NOT_300')
    # Original point cloud for visualization is not a model input and is not archived.
    clean=copy.deepcopy(batch);clean['ego'].pop('origin_lidar',None)
    inv=tensor_inventory(clean);nbytes=sum(x['nbytes'] for x in inv)
    if nbytes>2**30:raise RuntimeError('FROZEN_BATCH_GT1GiB')
    storages=storage_inventory(clean)
    if storages['unique_storage_bytes']>2**30:raise RuntimeError('FROZEN_STORAGE_GT1GiB')
    payload={'batch':clean,'rng':rng(),'input_hash':digest(clean),'GT_mapping':mapping,'GT_mapping_validation':mmeta}
    p=R/'batches'/f'sample_{sid}.pt';torchsave(p,payload,2**30)
    row={'sample_idx':sid,'path':str(p),'sha256':sha_file(p),'input_hash':payload['input_hash'],'uncompressed_tensor_bytes':nbytes,'archive_bytes':p.stat().st_size,'tensor_inventory':inv,'unique_storage':storages,'RNG_hash':digest(payload['rng']),'slots':slot,'GT_mapping':mmeta,'history_tokens':{str(k):info[sid]['prev_samples'][k] for k in [3,4,5,6,7]},'sample_token':info[sid]['token']}
    write_json(R/'batches'/f'sample_{sid}.json',row);batches.append(row)
    STATE['cached_samples'].append(sid);checkpoint_state();event('CPU_BATCH_CACHED',sample_idx=sid,archive_bytes=row['archive_bytes'],elapsed=time.monotonic()-start)
    del clean,payload,inv,storages,g
   if delay==3 and sid>=max(selected):break
   if count%100==0:event('CPU_REPLAY_PROGRESS',delay=delay,sample_idx=sid,valid_common_count=count,elapsed=time.monotonic()-start)
  delay_counts[str(delay)]=count;event('CPU_DELAY_COMPLETE',delay=delay,common_count=count,compacted_count=dataset.compacted_count,detached_storage_estimate=dataset.released_storage_bytes,elapsed=time.monotonic()-start)
  record('DELAY_COMPLETE',delay=delay,sample_idx=sid)
  del loader,dataset,batch
  gc.collect()
 if set(STATE['cached_samples'])!=selected:raise RuntimeError('MISSING_FROZEN_SAMPLES')
 write_json(R/'INPUT_PREPARATION.json',{'passed':True,'elapsed_seconds':time.monotonic()-start,'samples':batches,'delay_common_counts':delay_counts,'cache_bytes':bytes_in(R/'batches'),'seed':303,'CPU_model_forward_count':0,'resource_end':snapshot(),'CUDA_initialized':torch.cuda.is_initialized(),'physical_observation_interval_ms':600,'model_time_code':300})
 write_json(R/'RAW_INPUT_HASHES.json',raw_hashes)
 signal.alarm(0)

 assert not torch.cuda.is_initialized()
 STATE['phase']='CPU_CACHED';STATE['status']='CPU_VALIDATION_PENDING';checkpoint_state()
 event('CPU_REPLAY_FINISHED',samples=len(batches),resource=snapshot())
if __name__=='__main__':
 try:run()
 except BaseException as exc:
  signal.alarm(0);STATE.update(status='FAILED',failure=str(exc),failure_type=type(exc).__name__);checkpoint_state()
  write_json(R/'CPU_FAILURE.json',{'state':STATE,'traceback':traceback.format_exc(),'resource':snapshot()})
  record('CPU_FAILED',error=str(exc));sys.exit(1)
