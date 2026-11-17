"""CPU resource snapshots; no accelerator queries or initialization."""
import os,json,time,resource,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
START=time.monotonic()
def snapshot():
 status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
 cg=Path('/sys/fs/cgroup/memory'); st={}
 if (cg/'memory.stat').exists():st=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
 shm=os.statvfs('/dev/shm');files=list((R/'batches').glob('*.pt'))
 return {'epoch':time.time(),'elapsed_seconds':time.monotonic()-START,'pid':os.getpid(),'RSS_bytes':int(status['VmRSS'].split()[0])*1024,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cgroup_version':1,'cgroup_limit_bytes':int((cg/'memory.limit_in_bytes').read_text()),'cgroup_current_bytes':int((cg/'memory.usage_in_bytes').read_text()),'cgroup_cache_bytes':int(st.get('cache',0)),'cgroup_rss_bytes':int(st.get('rss',0)),'cgroup_shmem_bytes':int(st.get('shmem',0)),'shm_total_bytes':shm.f_blocks*shm.f_frsize,'shm_used_bytes':(shm.f_blocks-shm.f_bfree)*shm.f_frsize,'disk_free_bytes':shutil.disk_usage(R).free,'cache_count':len(files),'cache_bytes':sum(f.stat().st_size for f in files)}
def record(event,**kw):
 x={**snapshot(),'event':event,**kw}
 with (R/'cpu_resources.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 if x['peak_RSS_bytes']>8*2**30:raise RuntimeError('CPU_RSS_LIMIT_EXCEEDED')
 return x
def storage_inventory(value):
 import torch
 import numpy as np
 sizes={};logical=0;shared=0
 def walk(v):
  nonlocal logical,shared
  if torch.is_tensor(v):
   logical+=v.numel()*v.element_size();s=v.untyped_storage();sizes[(s.data_ptr(),s.nbytes())]=s.nbytes();shared+=int(v.is_shared())
  elif isinstance(v,np.ndarray):logical+=v.nbytes
  elif isinstance(v,dict):
   for x in v.values():walk(x)
  elif isinstance(v,(list,tuple)):
   for x in v:walk(x)
 walk(value)
 return {'logical_bytes':logical,'unique_torch_storage_count':len(sizes),'unique_storage_bytes':sum(sizes.values()),'shared_tensor_count':shared}
