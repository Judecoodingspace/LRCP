"""CPU reproduction of native history-container mutation, no LRCP model forward.
The actual get_batch_pillar_features method runs on deterministic data-preparation
stubs; derived-value insertion is tested separately from original tensor values.
This diagnoses the stopped run and does not change its code or resume inference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,copy,types,inspect
from opencood.models.point_pillar_deformable import PointPillarDeformable
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

class PillarStub:
 def __call__(self,b):b['pillar_features']=torch.zeros(1,64);return b
class ScatterStub:
 def __call__(self,b):b['spatial_features']=torch.zeros(5,64,1,1);return b
class BackboneStub:
 num_levels=3
 def get_projected_bev_features(self,b,batch_index,agent_index):
  require(b['pairwise_t_matrix'].shape==(1,5,5,4,4),'PAIRWISE_STUB_IDENTITY')
  return [torch.zeros(5,128,1,1) for _ in range(3)]

def original_equal(original,current):
 if isinstance(original,dict):return all(k in current and original_equal(v,current[k]) for k,v in original.items())
 if isinstance(original,list):return len(original)==len(current) and all(original_equal(a,b) for a,b in zip(original,current))
 if isinstance(original,torch.Tensor):return torch.equal(original,current)
 if isinstance(original,np.ndarray):return np.array_equal(original,current)
 return original==current

def run():
 row=load(R/'FROZEN_MANIFEST.json')['samples'][0];item=row['input'];require(sha_file(item['path'])==item['sha256'],'FROZEN_HASH')
 payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False);original=payload['batch'];work=copy.deepcopy(original)
 obj=types.SimpleNamespace(pillar_vfe=PillarStub(),scatter=ScatterStub(),backbone=BackboneStub())
 changes=[]
 for slot in (1,2):
  history=work['ego']['calibrate_data'][slot]['lidar_history'];before_keys=list(history[0]);before=digest(history[0])
  # Executes the ORIGINAL native orchestration method. Stubs avoid model ops.
  result=PointPillarDeformable.get_batch_pillar_features(obj,history,work['ego']['pairwise_t_matrix'],slot,[0])
  changes.append({'physical_slot':slot,'before_keys':before_keys,'after_keys':list(history[0]),'added_keys':[k for k in history[0] if k not in before_keys],'hash_before':before,'hash_after':digest(history[0]),'all_original_values_unchanged':original_equal(original['ego']['calibrate_data'][slot]['lidar_history'][0],history[0])})
 require(digest(work)!=digest(original),'REPRO_DID_NOT_TRIGGER_GUARD')
 require(original_equal(original,work),'ORIGINAL_INPUT_VALUES_CHANGED')
 for slot in (1,2):
  for key in ('pillar_features','spatial_features','pairwise_t_matrix'):work['ego']['calibrate_data'][slot]['lidar_history'][0].pop(key)
 require(digest(work)==digest(original),'REPRO_CLEANUP_NOT_EXACT')
 cpu_guard()
 result={'passed':True,'scope':'CPU_SOURCE_REPRODUCTION_ONLY_NOT_REAL_FORWARD_RECOVERY','sample_idx':row['sample_idx'],'native_method_file':inspect.getsourcefile(inspect.unwrap(PointPillarDeformable.get_batch_pillar_features)),'native_method_first_line':inspect.getsourcelines(inspect.unwrap(PointPillarDeformable.get_batch_pillar_features))[1],'changes':changes,'cause':'native get_batch_pillar_features reuses the mutable frozen-copy history dictionary; PillarVFE/Scatter add derived keys and the caller adds pairwise_t_matrix. Whole-dictionary digest changes although original tensors need not change.','certainty':'Reproduced exact original native orchestration with non-model stubs. Actual failed GPU process did not persist per-leaf before/after differences; no claim that these were its only mutations.','repair_not_applied':'Use throwaway per-history dictionaries at the native caller boundary, or independently compare every original leaf while separately recording derived additions; extend CPU Mock to exercise original history preparation. Verify unhooked Full equivalence before any separately authorized rerun.','CUDA_initialized':False,'real_model_forwards':0,'checkpoint_sha256':sha_file(CKPT)}
 write(R/'INPUT_MUTATION_CPU_REPRO.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
