"""Small synthetic CPU test of proposed cache compaction; no dataset replay/model."""
from pathlib import Path
import ast,difflib,json,hashlib
import numpy as np
import torch
from bounded_offset_cache_proposal import BoundedOffsetDataset
R=Path(__file__).resolve().parent
class Synthetic:
    def __init__(self):
        base=torch.arange(4*4*5*8*8*2,dtype=torch.float32).reshape(4,4,5,8,8,2)
        view=base[0,1,0][None]
        self.original=view
        self.scene_database={0:{'prev':{},1:{'params':{}},2:{'params':{'offset':view}}}}
        self.calls=0
    def __getitem__(self,index):
        self.calls+=1
        return {'offset':self.original}
    def __len__(self):return 1
x=Synthetic();wrapper=BoundedOffsetDataset(x)
np_before=np.random.get_state();torch_before=torch.random.get_rng_state().clone()
old_storage=x.original.untyped_storage().nbytes();item=wrapper[0]
compact=x.scene_database[0][2]['params']['offset']
checks={
 'native_getitem_exactly_once':x.calls==1,
 'returned_item_unchanged':item['offset'] is x.original,
 'persistent_values_exact':torch.equal(compact,x.original),
 'small_independent_storage':compact.untyped_storage().nbytes()==compact.numel()*compact.element_size() and compact.untyped_storage().data_ptr()!=x.original.untyped_storage().data_ptr(),
 'numpy_rng_unchanged':np.array_equal(np_before[1],np.random.get_state()[1]) and np_before[2:]==np.random.get_state()[2:],
 'torch_CPU_rng_unchanged':torch.equal(torch_before,torch.random.get_rng_state()),
 'CUDA_not_initialized':not torch.cuda.is_initialized(),
 'one_cache_compaction':wrapper.compacted_count==1,
}
assert all(checks.values())
for name in ['bounded_offset_cache_proposal.py','check_memory_proposal.py','analyze_pilot.py']:
 compile(ast.parse((R/name).read_text()),str(R/name),'exec')
old=(R/'runner.py').read_text()
new=old.replace('from collector import Collector','from collector import Collector\nfrom bounded_offset_cache_proposal import BoundedOffsetDataset')
new=new.replace('dataset=build_dataset(hypes,visualize=True,train=False)','dataset=BoundedOffsetDataset(build_dataset(hypes,visualize=True,train=False))')
# The self-created archive contains NumPy RNG state; PyTorch 2.6 defaults to weights_only=True.
new=new.replace("payload=torch.load(path,map_location='cpu')","payload=torch.load(path,map_location='cpu',weights_only=False)")
assert old!=new
compile(ast.parse(new),str(R/'runner_revision_candidate.py'),'exec')
(R/'runner_revision_candidate.py').write_text(new)
(R/'PROPOSED_MEMORY_FIX.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='executed_runner.py',tofile='proposed_next_run_runner.py')))
(R/'MEMORY_PROPOSAL_STATIC_CHECKS.json').write_text(json.dumps({'passed':True,'checks':checks,'synthetic_storage_before_bytes':old_storage,'synthetic_storage_after_bytes':compact.untyped_storage().nbytes(),'real_dataset_replay_executed':False,'model_forward_executed':False,'runtime_collector_validated':False,'executed_runner_modified':False},indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'storage_before':old_storage,'storage_after':compact.untyped_storage().nbytes(),'real_dataset_replay_executed':False}))
