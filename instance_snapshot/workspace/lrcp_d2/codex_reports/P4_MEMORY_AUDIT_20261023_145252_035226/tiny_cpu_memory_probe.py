"""Tiny synthetic CPU path check only. Never imports LRCP or runs historical items/model.
Total test array payload <128 KiB. Saves only to this audit directory's overlay path.
"""
from pathlib import Path
import os,json,resource,hashlib,importlib.util,shutil
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
R=Path(__file__).resolve().parent
P=R.parent/'P4_PILOT_20261022_193114_346841'
def shm_snapshot():
 u=shutil.disk_usage('/dev/shm')
 files={p.name:{'size':p.stat().st_size,'blocks':p.stat().st_blocks} for p in Path('/dev/shm').iterdir()}
 return {'total_bytes':u.total,'used_bytes':u.used,'files':files}
def children():return Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip()
def mappings():
 return [s for s in Path('/proc/self/maps').read_text().splitlines() if any(k in s for k in ['/dev/shm/','memfd:','/SYSV'])]
start=shm_snapshot();before_children=children()
spec=importlib.util.spec_from_file_location('proposal',P/'bounded_offset_cache_proposal.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class SmallSyntheticDataset:
 def __init__(self):
  parent=torch.arange(4*4*5*8*8*2,dtype=torch.float32).reshape(4,4,5,8,8,2)
  self.view=parent[0,1,0][None]
  self.scene_database={0:{'prev':{},1:{'params':{'offset':self.view}}}}
  self.calls=0
 def __len__(self):return 1
 def __getitem__(self,index):self.calls+=1;return {'offset':self.view}
dataset=SmallSyntheticDataset();proxy=mod.BoundedOffsetDataset(dataset)
rng=torch.random.get_rng_state().clone();np_state=np.random.get_state()
returned=proxy[0];compact=dataset.scene_database[0][1]['params']['offset']
checks={'cache_clone_values_exact':bool(torch.equal(compact,dataset.view)),'cache_clone_not_shared':not compact.is_shared(),'original_tensor_not_shared':not dataset.view.is_shared(),'clone_storage_small':compact.untyped_storage().nbytes()==compact.numel()*compact.element_size(),'native_getitem_once':dataset.calls==1,'returned_item_unchanged':returned['offset'] is dataset.view,'clone_torch_rng_unchanged':bool(torch.equal(rng,torch.random.get_rng_state())),'clone_numpy_rng_unchanged':np.array_equal(np_state[1],np.random.get_state()[1]) and np_state[2:]==np.random.get_state()[2:]}
loader=DataLoader(TensorDataset(torch.arange(8,dtype=torch.float32).reshape(4,2)),batch_size=1,num_workers=0,pin_memory=False,shuffle=False)
it=iter(loader);batch=next(it)
checks['single_process_iterator']=type(it).__name__=='_SingleProcessDataLoaderIter'
checks['worker_info_none']=torch.utils.data.get_worker_info() is None
checks['dataloader_output_not_shared']=not batch[0].is_shared()
checks['no_new_child_process']=children()==before_children
p=R/'tiny_frozen_batch.pt';tmp=p.with_suffix('.pt.tmp')
assert not p.exists() and not tmp.exists()
torch.save({'batch':batch,'numpy_rng':np.random.get_state()},tmp);os.replace(tmp,p)
loaded=torch.load(p,map_location='cpu',weights_only=False,mmap=False)
checks['disk_batch_values_exact']=torch.equal(loaded['batch'][0],batch[0])
checks['disk_loaded_tensor_not_shared']=not loaded['batch'][0].is_shared()
npy=R/'tiny_array.npy';np.save(npy,np.arange(16,dtype=np.float32),allow_pickle=False)
a=np.load(npy,allow_pickle=False,mmap_mode=None)
checks['numpy_normal_load_not_memmap']=not isinstance(a,np.memmap)
checks['CUDA_not_initialized']=not torch.cuda.is_initialized()
end=shm_snapshot()
checks['shm_used_bytes_unchanged']=start['used_bytes']==end['used_bytes']
checks['shm_no_new_files']=set(start['files'])==set(end['files'])
checks['no_shm_or_memfd_mapping']=not mappings()
checks['RSS_below_8GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30
result={'passed':all(checks.values()),'checks':checks,'torch_version':torch.__version__,'numpy_version':np.__version__,'torch_file':torch.__file__,'torch_CPU_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'OS_threads':len(list(Path('/proc/self/task').iterdir())),'child_processes_before':before_children,'child_processes_after':children(),'tensor_data_device':str(compact.device),'dataloader_iterator_type':type(it).__name__,'torch_multiprocessing_module_imported':'torch.multiprocessing' in __import__('sys').modules,'explicit_multiprocessing_started':False,'cache_storage_original_bytes':dataset.view.untyped_storage().nbytes(),'cache_storage_compact_bytes':compact.untyped_storage().nbytes(),'serialized_file_bytes':p.stat().st_size,'numpy_file_bytes':npy.stat().st_size,'shm_before':start,'shm_after':end,'relevant_process_memory_maps':mappings(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'real_dataset_replay':False,'real_model_forward':False,'PPU_inference':False,'scope_limit':'Synthetic only; does not establish real 18-sample equivalence or prove every native dependency never uses shmem.'}
(R/'TINY_CPU_PROBE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','dataloader_iterator_type','cache_storage_original_bytes','cache_storage_compact_bytes','serialized_file_bytes','shm_before','shm_after','peak_RSS_bytes']},ensure_ascii=False))
assert all(checks.values()),checks
