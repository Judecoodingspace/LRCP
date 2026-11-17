"""RD3 identities and durable writes. Import never creates a model/device."""
import os,sys,json,hashlib,csv,time,resource,shutil
from pathlib import Path
import cpu_only
cpu_only.install()
R=Path(__file__).resolve().parent
R2=R.parent/'RD3_REAL_FOUR_GROUP_R2'
REPAIR=R.parent/'RD3_INPUT_INTEGRITY_REPAIR'
OLD=R.parent/'RD3_MESSAGE_UTILITY_PILOT'
FROZEN=OLD/'FROZEN_MANIFEST.json'
REPORTS=R.parent;SRC=Path('/workspace/LRCP');BASE=Path('/workspace/lrcp_d2')
C2=REPORTS/'P5_B0_C2_20261031_cpu';C3=REPORTS/'P5_B0_C3_20261031_cpu';FULL=REPORTS/'P4_FULL_PREP_20261028_113714_782940'
sys.path.insert(1,str(C2));sys.path.insert(2,str(SRC))
from pilot_common import sha_file,digest,array,iou_matrix,gt_mapping,STAGES,MODEL,jsonable
import numpy as np
CKPT=MODEL/'net_epoch_bestval_at19.pth';CH='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'
CONDITIONS={'F':[True,True,True],'G-B':[True,False,True],'G-C':[True,True,False],'E':[True,False,False]}
QUERIES=[0,1620,3240,4860,6399]

def require(ok,reason):
 if not bool(ok):raise RuntimeError(reason)
def load(p):return json.loads(Path(p).read_text())
def write(p,x,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if exclusive and p.exists():raise FileExistsError(p)
 tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('x') as f:json.dump(jsonable(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def csvwrite(p,rows,fields=None):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def cpu_guard():
 import torch
 require(not torch.cuda.is_initialized(),'CPU_CUDA_INITIALIZED')
 require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30,'CPU_RSS_LIMIT')
 require(shutil.disk_usage(R).free>10*2**30,'DISK_RESERVE')
def full_records():
 out={}
 for p in sorted((FULL/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
  for row in load(p)['records']:
   if float(row['alpha'])==1:out[int(row['sample_idx'])]=row
 require(len(out)==910,'FULL_HISTORY_COUNT');return out
