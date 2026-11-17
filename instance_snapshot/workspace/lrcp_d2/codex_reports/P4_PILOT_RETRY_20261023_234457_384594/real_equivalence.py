"""Small real-item equivalence test, not the full historical retry."""
import random,copy,json,gc,sys
from types import SimpleNamespace
import numpy as np
import torch
from pilot_common import *
from telemetry import record,storage_inventory
from bounded_offset_cache_proposal import BoundedOffsetDataset
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.random.get_rng_state()}
def restore(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.random.set_rng_state(x['torch_cpu'])
random.seed(303);np.random.seed(303);torch.manual_seed(303)
h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));h['train_params']['batch_size']=1;h['validate_dir']=h['test_dir'];h['time_delay']=3
native=build_dataset(h,visualize=True,train=False);fixed=BoundedOffsetDataset(build_dataset(h,visualize=True,train=False))
assert not hasattr(native,'__getitems__')
rows=[]
for sid in [821,683,776]:
 before=rng();a=native[sid];after=rng();ha=digest(a);ba=native.collate_batch_test([a]);hba=digest(ba);after_collate=rng()
 orig=[]
 for k,v in native.scene_database[sid].items():
  if k!='prev' and 'offset' in v.get('params',{}):
   t=v['params']['offset'];orig.append({'cav':str(k),'shape':list(t.shape),'dtype':str(t.dtype),'logical':t.numel()*t.element_size(),'storage':t.untyped_storage().nbytes(),'value_hash':digest(t)})
 restore(before);b=fixed[sid];assert digest(b)==ha;assert digest(rng())==digest(after)
 bb=fixed.collate_batch_test([b]);assert digest(bb)==hba;assert digest(rng())==digest(after_collate)
 compact=[]
 for k,v in fixed.scene_database[sid].items():
  if k!='prev' and 'offset' in v.get('params',{}):
   t=v['params']['offset'];compact.append({'cav':str(k),'shape':list(t.shape),'dtype':str(t.dtype),'logical':t.numel()*t.element_size(),'storage':t.untyped_storage().nbytes(),'value_hash':digest(t)})
 assert len(orig)==len(compact)
 for x,y in zip(orig,compact):
  assert all(x[k]==y[k] for k in ['cav','shape','dtype','logical','value_hash']);assert y['storage']==y['logical']
 assert storage_inventory(bb)['shared_tensor_count']==0
 rows.append({'sample_idx':sid,'item_hash':ha,'collated_hash':hba,'RNG_after_hash':digest(after_collate),'native':orig,'compacted':compact,'passed':True})
 record('REAL_EQUIVALENCE',sample_idx=sid)
 del a,b,ba,bb;gc.collect()
assert not torch.cuda.is_initialized()
write_json(R/'REAL_EQUIVALENCE.json',{'passed':True,'real_samples':rows,'same_item_collate_RNG':True,'CUDA_initialized':False,'model_forwards':0,'resource':record('REAL_EQUIVALENCE_COMPLETE')})
print('REAL_EQUIVALENCE_PASSED',len(rows),flush=True)
