"""Normative schema is derived from actual frozen archive + preexisting R1
contract, never from a table of sample IDs, GT lengths or total leaf constants.
"""
from rd3_common import *
from input_integrity import snapshot,check,validate_label_offset,LABEL_PATH
import torch,copy

def frozen_contract(row):
 require(sha_file(FROZEN)=='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c','FROZEN_MANIFEST_IDENTITY')
 canonical=next(x for x in load(FROZEN)['samples'] if x['sample_idx']==row['sample_idx']);require(row==canonical,'SAMPLE_ROW_SUBSTITUTION')
 item=row['input'];require(sha_file(item['path'])==item['sha256'] and sha_file(CKPT)==CH,'FROZEN_INPUT_OR_CHECKPOINT_HASH')
 payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False)
 require(digest(payload['batch'])==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'FROZEN_INPUT_RNG_VALUES')
 contract=snapshot(payload['batch']);existing=next(x for x in load(REPAIR/'INPUT_LEAF_CONTRACT.json')['samples'] if x['sample_idx']==row['sample_idx'])
 require(contract==existing['all_original_fields_and_leaves'],'R1_CONTRACT_DIVERGED')
 return payload,contract

def device_contract(contract,device):
 result=copy.deepcopy(contract)
 for v in result['leaves'].values():
  if v['kind']=='tensor':v['device']=str(device)
 result['hash']=digest({'nodes':result['nodes'],'leaves':result['leaves']});return result

def verify_schema_before(row,batch,device,claimed_snapshot=None):
 _,cpu=frozen_contract(row);expected=device_contract(cpu,device)
 # R1 check preserves OTHER_INPUT_MUTATION and covers every original path,
 # actual value, scalar kind, container type, length and ordering.
 actual=check(expected,batch)
 if claimed_snapshot is not None:require(claimed_snapshot==actual,'CLAIMED_SCHEMA_SNAPSHOT_MISMATCH')
 return {'contract_hash':cpu['hash'],'runtime_contract_hash':expected['hash'],'leaf_paths':list(cpu['leaves']),'original_leaf_count':len(cpu['leaves']),'Tensor_leaf_count':sum(v['kind']=='tensor' for v in cpu['leaves'].values()),'phase':'before'}

def verify_schema_after(row,batch,device,claimed_before=None):
 _,cpu=frozen_contract(row);expected=device_contract(cpu,device)
 if claimed_before is not None:require(expected==claimed_before,'CLAIMED_ORIGINAL_SCHEMA_MISMATCH')
 validate_label_offset(batch);actual=check(expected,batch,(LABEL_PATH,))
 return {'contract_hash':cpu['hash'],'runtime_contract_hash':expected['hash'],'after_snapshot_hash':actual['hash'],'original_leaf_count':len(cpu['leaves']),'Tensor_leaf_count':sum(v['kind']=='tensor' for v in cpu['leaves'].values()),'phase':'after','all_original_paths_values_and_types_checked':True}
