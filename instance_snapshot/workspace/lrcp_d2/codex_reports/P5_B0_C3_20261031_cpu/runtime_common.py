"""C3 process/status helpers. No model imports or computation contexts."""
import os,json,time,hashlib,uuid,resource,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
C2=ROOT.parent/'P5_B0_C2_20261031_cpu'
C1=ROOT.parent/'P5_B0_BACKEND_20261031_cpu'
P5=C2.parent/'P5_A_CPU_FIX_20261028_storage'
FULL=C2.parent/'P4_FULL_PREP_20261028_113714_782940'
PYTHON='/workspace/venvs/lrcp/bin/python'
UUID='GPU-019e1215-c270-091c-0000-0000c0610973'
BUDGET={'max_attempts':22,'controls':4,'core':18,'cpu_rss_bytes':8*2**30,'min_free_device_mib':4096,'min_disk_bytes':10*2**30,'wall_seconds':1800,'capture_max_bytes':2*2**30,'device_reserved_max_bytes':12*2**30,'device_allocated_max_bytes':8*2**30}
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 p=Path(path);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('x') as f:json.dump(value,f,allow_nan=False,sort_keys=True);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if tmp.exists():tmp.unlink()
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();fields=s[s.rfind(')')+2:].split()
  return {'pid':int(pid),'start_ticks':int(fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'process_state':fields[0]}
 except (OSError,ValueError,IndexError):return None
def alive(who):
 if not who:return False
 now=identity(who['pid'])
 return bool(now and now['process_state'] not in ('Z','X') and all(now[k]==who[k] for k in ('pid','start_ticks','boot_id')))
def rss(pid):
 try:
  lines=(Path('/proc')/str(pid)/'status').read_text().splitlines()
  found=[x for x in lines if x.startswith('VmRSS:')]
  if found:return int(found[0].split()[1])*1024
  who=identity(pid)
  if who is None or who['process_state'] in ('Z','X'):return 0
  # Linux may remove VmRSS before waitpid can observe exit. statm still
  # reports resident pages (zero once the mm is detached); do not mistake
  # this normal exit window for a missing watchdog.
  try:
   pages=(Path('/proc')/str(pid)/'statm').read_text().split()
   if len(pages)>=2:return int(pages[1])*os.sysconf('SC_PAGE_SIZE')
  except FileNotFoundError:return 0
  raise RuntimeError('RSS_TELEMETRY_UNAVAILABLE')
 except FileNotFoundError:return 0

def tree_rss(pid):
 total=rss(pid)
 try:children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
 except FileNotFoundError:children=[]
 for child in children:total+=tree_rss(int(child))
 return total

def status(root):
 root=Path(root);p=read(root/'progress.json',{})
 if not p:p={'state':read(ROOT/'P5_B0_READY.json',{}).get('state','NOT_READY'),'mock_only':root.parent==ROOT/'mock_runs','started':0,'completed':0,'core_completed':0}
 owner=read(root/'owner.json',{});worker=read(root/'worker_identity.json',{})
 if p.get('state') in ('PPU_RUNNING_CONTROLS','PPU_RUNNING_CORE','MOCK_RUNNING_CONTROLS','MOCK_RUNNING_CORE') and worker and not alive(worker):
  p={**p,'state':'PPU_CAPTURE_FAILED' if not p.get('mock_only') else 'MOCK_FAILED','error':'WORKER_NOT_ALIVE_NO_RETRY'}
 b=read(root/'budget.json',{'attempts':[]});p.update(reserved_forwards=len(b['attempts']),max_forwards=22,worker_alive=alive(worker),run_root=str(root),device_uuid=UUID)
 p['resources']=read(root/'supervisor_resources.json',{})
 return p

def child_environment(mock=False):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='/workspace/LRCP:'+str(C2))
 env['CUDA_VISIBLE_DEVICES']='' if mock else '0'
 return env


def capture_bytes(root):
 """Snapshot bytes while atomic directories are being renamed; ENOENT is benign."""
 total=0
 for base,dirs,files in os.walk(Path(root)/'capture'):
  for name in files:
   try:total+=(Path(base)/name).stat().st_size
   except FileNotFoundError:pass
 return total
