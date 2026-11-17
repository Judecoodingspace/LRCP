"""Manual-only launch controller; check/status import no torch or CUDA."""
import os,sys,json,time,hashlib,shutil,subprocess,signal,fcntl,uuid,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
PYTHON='/workspace/venvs/lrcp/bin/python'
BUDGET={'samples':910,'reused_samples':18,'core_new_forwards':1784,'guard_forwards':6,'max_forward_attempts':1790,'CPU_RSS_bytes':8*2**30,'cache_bytes':35*2**30,'PPU_wall_seconds':5400,'capture_bytes':16*2**30,'PPU_allocated_peak_bytes':8*2**30,'PPU_reserved_peak_bytes':12*2**30,'PPU_min_free_bytes':4*2**30}
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();t=s[s.rfind(')')+2:].split()
  return {'pid':int(pid),'start_ticks':int(t[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'process_state':t[0],'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def alive(exp):
 if not exp:return False
 now=identity(exp['pid'])
 return bool(now and now['process_state']!='Z' and all(now[k]==exp[k] for k in ['pid','start_ticks','boot_id']) and str(R) in now['cmdline'] and 'worker' in now['cmdline'])
def verify_ready(full=True,ready_override=None):
 ready=ready_override if ready_override is not None else read(R/'PPU_READY.json',{});prep=read(R/'INPUT_MANIFEST.json',{});cpu=read(R/'CPU_READY.json',{});equiv=read(R/'INPUT_EQUIVALENCE.json',{});reuse=read(R/'REUSE_18_VALIDATION.json',{});static=read(R/'STATIC_VALIDATION.json',{})
 if ready.get('run_directory')!=str(R) or ready.get('state')!='PPU_READY_NOT_STARTED' or ready.get('budget')!=BUDGET:raise RuntimeError('READY_INVALID')
 if not all(x.get('passed') for x in [prep,cpu,equiv,reuse,static]):raise RuntimeError('READY_GATE_MISSING')
 samples=prep['samples'];ids=[x['sample_idx'] for x in samples]
 if ready.get('input_manifest_sha256')!=sha(R/'INPUT_MANIFEST.json') or ready.get('frozen_batch_hashes')!={str(x['sample_idx']):x['sha256'] for x in samples}:raise RuntimeError('READY_INPUT_MANIFEST_MISMATCH')
 if cpu.get('input_manifest_sha256')!=sha(R/'INPUT_MANIFEST.json') or cpu.get('input_validation_sha256')!=sha(R/'CPU_INPUT_VALIDATION.json') or cpu.get('reuse_validation_sha256')!=sha(R/'REUSE_18_VALIDATION.json'):raise RuntimeError('CPU_READY_EVIDENCE_MISMATCH')
 if len(ids)!=910 or len(set(ids))!=910 or len(equiv['pilot_samples'])!=18 or len(reuse['sample_ids'])!=18:raise RuntimeError('READY_SAMPLE_COUNTS')
 if sum(x['archive_bytes'] for x in samples)>BUDGET['cache_bytes']:raise RuntimeError('READY_CACHE_LIMIT')
 for x in samples:
  p=Path(x['path'])
  if p.parent!=R/'batches' or not p.is_file() or p.stat().st_size!=x['archive_bytes']:raise RuntimeError('INPUT_MISSING_OR_WRONG_SIZE '+str(p))
  if full and sha(p)!=x['sha256']:raise RuntimeError('INPUT_HASH_CHANGED '+str(p))
 for path,h in ready['locked_files'].items():
  if not Path(path).is_file() or sha(path)!=h:raise RuntimeError('LOCKED_FILE_CHANGED '+path)
 if full:
  for path,h in ready['protected_files'].items():
   if not Path(path).is_file() or sha(path)!=h:raise RuntimeError('PROTECTED_FILE_CHANGED '+path)
 disk=shutil.disk_usage(R).free
 if disk<30*2**30:raise RuntimeError('INSUFFICIENT_DISK')
 cg=Path('/sys/fs/cgroup/memory');v2=Path('/sys/fs/cgroup')
 if (cg/'memory.limit_in_bytes').exists():limit=int((cg/'memory.limit_in_bytes').read_text());current=int((cg/'memory.usage_in_bytes').read_text())
 elif (v2/'memory.max').exists() and (v2/'memory.max').read_text().strip()!='max':limit=int((v2/'memory.max').read_text());current=int((v2/'memory.current').read_text())
 else:raise RuntimeError('CGROUP_MEMORY_LIMIT_UNAVAILABLE')
 if limit-current<8*2**30:raise RuntimeError('CGROUP_HEADROOM_BELOW_8GiB')
 return {'passed':True,'count':910,'full_hash_check':full,'free_bytes':disk,'cgroup_limit_bytes':limit,'cgroup_current_bytes':current,'device_check':'not performed in dry-run','CUDA_initialized':False}
def conflicts():
 out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:continue
  if '/codex_reports/P4_' in cmd and ('/full_capture_worker.py' in cmd or '/ppu_runner.py' in cmd or 'full_launch_control.py worker' in cmd):out.append({'pid':int(p.name),'cmdline':cmd})
 return out
def launch(dry=False,approve=False):
 if dry:return {**verify_ready(True),'dry_run':True,'started':False}
 if not approve or not sys.stdin.isatty():raise RuntimeError('MANUAL_TERMINAL_AND_EXPLICIT_1790_BUDGET_APPROVAL_REQUIRED')
 with (R/'.launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_ready(True)
  if (R/'PPU_LAUNCH.json').exists() or (R/'PPU_CAPTURE_COMPLETE.json').exists() or (R/'PPU_CAPTURE_FAILURE.json').exists():raise RuntimeError('REFUSE_DUPLICATE_LAUNCH_OR_RETRY')
  c=conflicts()
  if c:raise RuntimeError('P4_PROCESS_CONFLICT '+repr(c))
  if os.environ.get('CUDA_VISIBLE_DEVICES') not in [None,'0']:raise RuntimeError('CUDA_MAPPING_UNSUPPORTED')
  from pilot_common import gpu_status
  import torch,numpy as np
  pilot=R.parent/'P4_PILOT_RETRY_20261023_234457_384594'
  setup=read(pilot/'MODEL_CPU_SETUP.json',{});execution=read(pilot/'PPU_EXECUTION.json',{})
  if torch.__version__!=setup.get('torch') or np.__version__!=setup.get('numpy'):raise RuntimeError('PILOT_RUNTIME_VERSION_CHANGED')
  device=gpu_status()
  old_name=execution.get('resource_end',{}).get('framework',{}).get('name')
  if not old_name or device['framework']['name']!=old_name:raise RuntimeError('PILOT_PPU_DEVICE_NAME_MISMATCH')
  if device['framework']['free_total_bytes'][0]<BUDGET['PPU_min_free_bytes']:raise RuntimeError('DEVICE_FREE_MEMORY_BELOW_LIMIT')
  nonce=uuid.uuid4().hex;claim={'run_directory':str(R),'nonce':nonce,'launch_epoch':time.time(),'ready_sha256':sha(R/'PPU_READY.json'),'budget':BUDGET,'device_preflight':device,'worker_pid':None,'worker_identity':None}
  atomic(R/'PPU_LAUNCH.json',claim)
  try:
   log=(R/'ppu.log').open('xb',buffering=0);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',P4_FULL_NONCE=nonce)
   process=subprocess.Popen(['bash',str(R/'ppu_worker.sh'),nonce],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True);who=identity(process.pid)
   if not who:raise RuntimeError('WORKER_IDENTITY_UNAVAILABLE')
   claim.update(worker_pid=process.pid,worker_identity=who);atomic(R/'PPU_LAUNCH.json',claim);log.close()
  except BaseException as e:atomic(R/'PPU_CAPTURE_FAILURE.json',{'passed':False,'error':str(e)});raise
 return {'started':True,'pid':claim['worker_pid'],'log':str(R/'ppu.log'),'budget':BUDGET}
def worker(nonce):
 claim=read(R/'PPU_LAUNCH.json',{});deadline=time.monotonic()+10
 while not claim.get('worker_pid'):
  if time.monotonic()>deadline:raise RuntimeError('CLAIM_HANDSHAKE_TIMEOUT')
  time.sleep(.05);claim=read(R/'PPU_LAUNCH.json',{})
 if nonce!=claim.get('nonce') or os.environ.get('P4_FULL_NONCE')!=nonce or os.getpid()!=claim['worker_pid'] or not alive(claim['worker_identity']):raise RuntimeError('WORKER_IDENTITY_MISMATCH')
 if (R/'PPU_WORKER_STARTED.json').exists():raise RuntimeError('REFUSE_WORKER_REENTRY')
 atomic(R/'PPU_WORKER_STARTED.json',{'epoch':time.time(),'identity':identity(os.getpid())});code=1;reason='NOT_FINISHED';child=None
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda s,f:(_ for _ in ()).throw(InterruptedError('WORKER_SIGNAL '+str(s))))
 parent_pid=os.getpid()
 def parent_death_guard():
  import ctypes
  if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:os._exit(70)
  if os.getppid()!=parent_pid:os.kill(os.getpid(),signal.SIGTERM)
 try:
  verify_ready(True)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',P4_FULL_NONCE=nonce)
  child=subprocess.Popen([PYTHON,'-B',str(R/'full_capture_worker.py')],env=env,preexec_fn=parent_death_guard)
  child_identity=identity(child.pid)
  if not child_identity:raise RuntimeError('CAPTURE_CHILD_IDENTITY_UNAVAILABLE')
  atomic(R/'PPU_CHILD_IDENTITY.json',child_identity)
  deadline=time.monotonic()+BUDGET['PPU_wall_seconds']
  try:code=child.wait(timeout=BUDGET['PPU_wall_seconds']-5)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=max(.1,deadline-time.monotonic()-.25))
   except subprocess.TimeoutExpired:child.kill();child.wait()
   raise RuntimeError('WORKER_TIMEOUT')
  cap=read(R/'PPU_CAPTURE_COMPLETE.json',{})
  if code!=0 or not cap.get('passed') or cap.get('completed')!=1790:raise RuntimeError('CAPTURE_FAILED_OR_INCOMPLETE')
  reason='PPU_CAPTURE_COMPLETE';code=0
 except BaseException as e:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  traceback.print_exc();reason=str(e);code=1
  if not (R/'PPU_CAPTURE_FAILURE.json').exists():atomic(R/'PPU_CAPTURE_FAILURE.json',{'passed':False,'error':reason})
 finally:atomic(R/'PPU_WORKER_EXIT.json',{'exit_code':code,'reason':reason,'identity':identity(os.getpid()),'epoch':time.time()})
 return code
def status():
 claim=read(R/'PPU_LAUNCH.json');progress=read(R/'PPU_PROGRESS.json',{});cap=read(R/'PPU_CAPTURE_COMPLETE.json',{});err=read(R/'PPU_CAPTURE_FAILURE.json',{});cpu=read(R/'CPU_VALIDATION_COMPLETE.json',{});accept=read(R/'RESEARCH_ACCEPTED.json',{});exit_=read(R/'PPU_WORKER_EXIT.json',{});live=alive(claim.get('worker_identity')) if claim else False
 if accept.get('passed') and cpu.get('passed') and cap.get('passed') and exit_.get('exit_code')==0 and sha(R/'CPU_VALIDATION_COMPLETE.json')==accept.get('CPU_validation_sha256') and sha(R/'PPU_CAPTURE_COMPLETE.json')==accept.get('PPU_capture_sha256'):phase='RESEARCH_ACCEPTED'
 elif cpu.get('passed') and cap.get('passed') and exit_.get('exit_code')==0:phase='CPU_VALIDATION_COMPLETE'
 elif cap.get('passed') and exit_.get('exit_code')==0:phase='PPU_CAPTURE_COMPLETE'
 elif live:phase='PPU_RUNNING'
 elif claim or err:phase='PPU_CAPTURE_FAILED'
 else:
  try:verify_ready(False);phase='PPU_READY_NOT_STARTED'
  except Exception:phase='CPU_READY' if read(R/'CPU_READY.json',{}).get('passed') else 'CPU_NOT_READY'
 child=read(R/'PPU_CHILD_IDENTITY.json',{});now=identity(child.get('pid',-1));live_child=bool(now and all(now.get(k)==child.get(k) for k in ['pid','start_ticks','boot_id']))
 rss=None
 if live_child:
  try:rss=int(next(line for line in (Path('/proc')/str(child['pid'])/'status').read_text().splitlines() if line.startswith('VmRSS:')).split()[1])*1024
  except (OSError,StopIteration,ValueError):pass
 return {'state':phase,'CPU_READY':bool(read(R/'CPU_READY.json',{}).get('passed')),'worker_alive':live,'current_sample_idx':progress.get('current_sample_idx'),'alpha':progress.get('alpha'),'completed_samples':progress.get('completed_samples',0),'expected_samples':910,'attempted_forwards':progress.get('attempted',0),'completed_forwards':progress.get('completed',0),'max_forwards':1790,'shards_complete':progress.get('shards_complete',0),'CPU_RSS_bytes':rss if rss is not None else progress.get('cpu_rss_bytes'),'CPU_peak_RSS_bytes':progress.get('cpu_peak_rss_bytes'),'PPU_allocated_bytes':progress.get('PPU_allocated_bytes'),'PPU_reserved_bytes':progress.get('PPU_reserved_bytes'),'last_progress_epoch':progress.get('updated_epoch'),'CPU_validation':read(R/'CPU_VALIDATION_PROGRESS.json',{}),'CPU_validation_failure':read(R/'CPU_VALIDATION_FAILURE.json',{}).get('error'),'exit_code':exit_.get('exit_code'),'error':err.get('error') or exit_.get('reason')}
def monitor():
 while True:
  z=status();print(json.dumps(z,indent=2,ensure_ascii=False),flush=True);p=R/'ppu.log'
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));print('\n'.join(f.read().decode(errors='replace').splitlines()[-8:]),flush=True)
  vendor=shutil.which('nvidia-smi')
  if vendor:
   try:
    q=subprocess.run([vendor,'--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],capture_output=True,text=True,timeout=5);print(q.stdout if q.returncode==0 else 'Device query unavailable',flush=True)
   except Exception:print('Device query unavailable',flush=True)
  if z['state']!='PPU_RUNNING':break
  time.sleep(5)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','worker','status','monitor','check']);p.add_argument('--nonce');p.add_argument('--dry-run',action='store_true');p.add_argument('--approve-1790',action='store_true');a=p.parse_args()
 try:
  if a.command=='launch':print(json.dumps(launch(a.dry_run,a.approve_1790),indent=2,ensure_ascii=False))
  elif a.command=='worker':sys.exit(worker(a.nonce))
  elif a.command=='status':print(json.dumps(status(),indent=2,ensure_ascii=False))
  elif a.command=='monitor':monitor()
  else:print(json.dumps(verify_ready(True),indent=2))
 except BaseException as e:
  if isinstance(e,SystemExit):raise
  print(json.dumps({'passed':False,'error':str(e)},ensure_ascii=False),file=sys.stderr);sys.exit(1)
