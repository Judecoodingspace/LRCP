"""Read-only cross-condition checks. Usable by REAL worker; no CPU monkeypatch."""
from rd3_common import *

def full_record_path(sid,records,root):
 for rec in records:
  if rec['sample_idx']==sid and rec['condition']=='F':return Path(root)/rec['relative_path']
 index=load(R/'reused_records/index.json')
 for rec in index['records']:
  if rec['sample_idx']==sid and rec['condition']=='F':return Path(rec['data_path'])
 raise RuntimeError('FIRST_F_BASELINE_REQUIRED')

def compare_condition_identity(full,current):
 full=Path(full);current=Path(current);fm=load(full/'meta.json');cm=load(current/'meta.json')
 require(fm['sample_idx']==cm['sample_idx'] and fm['condition']=='F','CONDITION_FULL_IDENTITY')
 for key in ('feature_hashes','parameter_digest','buffer_digest','GT_mapping','input_leaf_before','input_leaf_after','history_integrity'):
  require(fm[key]==cm[key],'CONDITION_IDENTITY_CHANGED '+key)
 with np.load(full/'arrays.npz',allow_pickle=False) as f,np.load(current/'arrays.npz',allow_pickle=False) as c:
  for key in ('pred_offset','offset_GT','reference_full','GT_canonical','metaflow_slot_1','metaflow_slot_2'):
   require(np.array_equal(f[key],c[key]),'OTHER_AGENT_OR_OFFSET_CHANGED '+key)
 return True
