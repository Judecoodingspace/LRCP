"""Independent archive read-back, storage, RNG and historical GT validation; CPU only."""
import json,os,gc,ast,time,signal
from types import SimpleNamespace
import torch
import numpy as np
from pilot_common import *
from telemetry import snapshot,record,storage_inventory

def validate():
 prep=json.loads((R/'INPUT_PREPARATION.json').read_text());state=json.loads((R/'STATE.json').read_text())
 assert prep['passed'] and state['phase']=='CPU_CACHED' and not (R/'CPU_FAILURE.json').exists()
 assert prep['seed']==303 and prep['model_time_code']==300 and prep['CPU_model_forward_count']==0
 assert prep['elapsed_seconds']<=1800 and prep['resource_end']['peak_RSS_bytes']<=8*2**30
 common=json.loads((BASE/'P1_common910_indices.json').read_text())['sample_ids']
 expected_final=sum(int(s)<=max(int(x['sample_idx']) for x in selections()) for s in common)
 assert prep['delay_common_counts']=={'0':910,'1':910,'2':910,'3':expected_final}
 selected={int(x['sample_idx']) for x in selections()}
 assert len(selected)==18 and {x['sample_idx'] for x in prep['samples']}==selected
 from opencood.hypes_yaml import yaml_utils
 from opencood.data_utils.post_processor import build_postprocessor
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));pp=build_postprocessor(h['postprocess'],False)
 rows=[];gt_count=0
 for row in prep['samples']:
  sid=row['sample_idx'];p=Path(row['path']);assert p.parent==R/'batches';assert sha_file(p)==row['sha256'];assert p.stat().st_size==row['archive_bytes']
  payload=torch.load(p,map_location='cpu',weights_only=False,mmap=False);batch=payload['batch']
  assert digest(batch)==payload['input_hash']==row['input_hash'];assert digest(payload['rng'])==row['RNG_hash']
  assert int(batch['ego']['sample_idx'])==sid
  assert tensor_inventory(batch)==row['tensor_inventory']
  st=storage_inventory(batch);assert st['shared_tensor_count']==0 and st['unique_storage_bytes']<=2**30
  assert 'origin_lidar' not in batch['ego']
  slot={str(k):{'physical_cav_ids':v['physical_cav_ids'],'valid_mask':array(v['valid_mask']).tolist(),'time_delay':array(v['time_delay']).tolist()} for k,v in batch['ego']['calibrate_data'].items()}
  assert slot==row['slots']
  for v in slot.values():assert not v['valid_mask'][0] or v['time_delay'][0]==300
  current=array(pp.generate_gt_bbx(batch));old=historical(sid,0.)['gt_boxes'];mapping,meta=gt_mapping(old,current)
  assert meta==row['GT_mapping'] and np.array_equal(mapping,payload['GT_mapping'])
  assert np.array_equal(old,historical(sid,1.)['gt_boxes']);gt_count+=len(old)
  rows.append({'sample_idx':sid,'sha256':row['sha256'],'input_hash':row['input_hash'],'RNG_hash':row['RNG_hash'],'GT_count':len(old),'GT_mapping':meta,'storage':st,'passed':True})
  record('ARCHIVE_VALIDATED',sample_idx=sid);del payload,batch;gc.collect()
 assert gt_count==162
 for p,hsh in json.loads((R/'PROTECTED_HASHES_BEFORE.json').read_text()).items():assert sha_file(p)==hsh,p
 for p,hsh in json.loads((R/'RAW_INPUT_HASHES.json').read_text()).items():assert sha_file(p)==hsh,p
 assert not torch.cuda.is_initialized()
 resources=[json.loads(s) for s in (R/'cpu_resources.jsonl').read_text().splitlines()]
 out={'passed':True,'samples':rows,'sample_count':18,'GT_count':gt_count,'CPU_retry_count':1,'delay_common_counts':prep['delay_common_counts'],'replay_elapsed_seconds':prep['elapsed_seconds'],'replay_peak_RSS_bytes':prep['resource_end']['peak_RSS_bytes'],'all_observed_peak_RSS_bytes':max(x['peak_RSS_bytes'] for x in resources),'max_shm_used_bytes':max(x['shm_used_bytes'] for x in resources),'cache_archive_bytes':sum(x['archive_bytes'] for x in prep['samples']),'raw_input_files':len(json.loads((R/'RAW_INPUT_HASHES.json').read_text())),'CUDA_initialized':False,'model_forward_count':0,'resource':snapshot()}
 write_json(R/'CPU_VALIDATION.json',out);print(json.dumps({k:v for k,v in out.items() if k!='samples'}),flush=True)
if __name__=='__main__':
 try:validate()
 except BaseException:
  import traceback
  write_json(R/'CPU_VALIDATION_FAILURE.json',{'passed':False,'traceback':traceback.format_exc()});raise
