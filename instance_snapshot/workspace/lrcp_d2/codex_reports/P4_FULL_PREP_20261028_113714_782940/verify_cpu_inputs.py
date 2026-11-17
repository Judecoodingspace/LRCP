"""Independent 910 archive readback; CPU only, no dataset replay or model."""
import os,sys,json,time,resource,shutil,gc,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pilot_common import R,BASE,sha_file,digest,array,gt_mapping,write_json
from telemetry import storage_inventory
C=R.parent/'P4_C1_20261027_162059_722622';sys.path.insert(0,str(C))
from c1_common import make_pp

def guard(start_epoch):
 if time.time()-start_epoch>3600:raise TimeoutError('CPU_PREPARATION_60_MINUTES')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_LIMIT')
 if shutil.disk_usage(R).free<5*2**30:raise RuntimeError('DISK_RESERVE')
 if torch.cuda.is_initialized():raise RuntimeError('CUDA_UNEXPECTED')

def run():
 manifest=json.loads((R/'INPUT_MANIFEST.json').read_text());equiv=json.loads((R/'INPUT_EQUIVALENCE.json').read_text());pre=json.loads((R/'REUSE_PRECHECK.json').read_text());started=json.loads((R/'CPU_STARTED.json').read_text());start=started['time']
 assert manifest['passed'] and manifest['sample_count']==910 and equiv['passed'] and equiv['count']==18 and pre['passed'] and pre['condition_count']==36
 common=json.loads((BASE/'P1_common910_indices.json').read_text())['sample_ids'];expected=[int(x) for x in common];samples=manifest['samples']
 if len(samples)!=910 or [x['sample_idx'] for x in samples]!=expected:raise RuntimeError('COMMON_SAMPLE_ORDER_MISMATCH')
 if sum(x['archive_bytes'] for x in samples)>35*2**30:raise RuntimeError('CACHE_LIMIT')
 pp=make_pp();totalgt=0;checks=[]
 for n,item in enumerate(samples):
  guard(start);sid=item['sample_idx'];p=Path(item['path'])
  if p.parent!=R/'batches' or p.stat().st_size!=item['archive_bytes'] or sha_file(p)!=item['sha256']:raise RuntimeError('ARCHIVE_HASH_MISMATCH '+str(sid))
  payload=torch.load(p,map_location='cpu',weights_only=False,mmap=False);batch=payload['batch']
  if digest(batch)!=item['input_hash'] or payload['input_hash']!=item['input_hash'] or digest(payload['rng'])!=item['RNG_hash']:raise RuntimeError('CONTENT_OR_RNG_HASH_MISMATCH '+str(sid))
  storage=storage_inventory(batch)
  if storage['shared_tensor_count'] or storage['unique_storage_bytes']>2**30:raise RuntimeError('INVALID_STORAGE')
  ego=batch['ego'];assert int(ego['sample_idx'])==sid and tuple(ego['anchor_box'].shape)==(80,80,2,7)
  slots={str(k):{'physical_cav_ids':v['physical_cav_ids'],'valid_mask':array(v['valid_mask']).tolist(),'time_delay':array(v['time_delay']).tolist()} for k,v in ego['calibrate_data'].items()}
  if slots!=item['slots']:raise RuntimeError('CAV_SLOT_MISMATCH')
  for v in slots.values():
   if v['valid_mask'][0] and v['time_delay'][0]!=300:raise RuntimeError('TIME_CODE_MISMATCH')
  gt=array(pp.generate_gt_bbx(batch));old=BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_0.0.npz'
  with np.load(old,allow_pickle=False) as z:mapping,meta=gt_mapping(z['gt_boxes'],gt)
  if meta!=item['GT_mapping'] or not np.array_equal(mapping,payload['GT_mapping']):raise RuntimeError('GT_MAPPING_CHANGED '+str(sid))
  totalgt+=len(gt);checks.append({'sample_idx':sid,'archive_sha256':item['sha256'],'input_hash':item['input_hash'],'RNG_hash':item['RNG_hash'],'GT_count':len(gt),'passed':True})
  del payload,batch,ego,gt;gc.collect()
  if n%50==0:print(json.dumps({'event':'INPUT_READBACK_PROGRESS','count':n+1,'sample_idx':sid,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}),flush=True)
 if totalgt!=8164:raise RuntimeError('HISTORICAL_GT_TOTAL_MISMATCH')
 guard(start)
 write_json(R/'CPU_INPUT_VALIDATION.json',{'passed':True,'sample_count':910,'GT_count':totalgt,'samples':checks,'exact_common_sample_order':True,'content_hash_readback':True,'all_input_file_hashes_checked':True,'CAV_and_time_codes_checked':True,'CPU_RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'total_elapsed_from_CPU_start':time.time()-start,'CUDA_initialized':False,'model_forwards':0})
 write_json(R/'REUSE_18_VALIDATION.json',{'passed':True,'sample_ids':sorted(x['sample_idx'] for x in equiv['pilot_samples']),'new_batch_content_and_rng_exact':True,'old_core_conditions_independently_replayed':36,'precheck_sha256':sha_file(R/'REUSE_PRECHECK.json'),'new_equivalence_sha256':sha_file(R/'INPUT_EQUIVALENCE.json'),'pilot_failed_worker_retained':True,'PPU_runtime_still_unverified':True,'model_forwards':0})
 write_json(R/'CPU_READY.json',{'passed':True,'phase':'CPU_READY','sample_count':910,'GT_count':totalgt,'input_manifest_sha256':sha_file(R/'INPUT_MANIFEST.json'),'input_validation_sha256':sha_file(R/'CPU_INPUT_VALIDATION.json'),'reuse_validation_sha256':sha_file(R/'REUSE_18_VALIDATION.json'),'model_forwards':0,'CUDA_initialized':False})
 print('CPU_READBACK_AND_REUSE_GATES_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  write_json(R/'CPU_READBACK_FAILURE.json',{'passed':False,'error':str(e),'traceback':traceback.format_exc(),'model_forwards':0,'CUDA_initialized':torch.cuda.is_initialized()});raise
