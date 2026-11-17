"""Manual launch and read-only monitoring. Import/dry-run use ONLY standard library.
No preparation function calls launch(). Model work requires launch handshake.
"""
import os,sys,json,time,hashlib,shutil,subprocess,signal,uuid,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
PYTHON='/workspace/venvs/lrcp/bin/python'
POINTER=R.parent/'P4_PILOT_READY_CURRENT.txt'
LIMITS={'sample_count':18,'max_full_forwards':40,'CPU_RSS_bytes':8*2**30,'PPU_seconds':900,'CPU_analysis_seconds':1800,'cache_bytes':18*2**30,'diagnostic_bytes':2**30}
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default

def put(p,obj):
 p=Path(p);temp=p.with_name(p.name+'.control.tmp')
 with temp.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split()
  return {'pid':pid,'start_ticks':int(tail[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'state':tail[0],'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def alive(expected):
 if not expected:return False
 now=identity(expected['pid'])
 return bool(now and now['state']!='Z' and all(now[k]==expected[k] for k in ('pid','start_ticks','boot_id')) and str(R) in now['cmdline'] and 'worker' in now['cmdline'])
def resources():
 c=Path('/sys/fs/cgroup/memory');shm=os.statvfs('/dev/shm')
 return {'disk_free_bytes':shutil.disk_usage(R).free,'cgroup_limit_bytes':int((c/'memory.limit_in_bytes').read_text()),'cgroup_current_bytes':int((c/'memory.usage_in_bytes').read_text()),'shm_total_bytes':shm.f_blocks*shm.f_frsize,'shm_used_bytes':(shm.f_blocks-shm.f_bfree)*shm.f_frsize}
def verify_ready(full=True,require_pointer=True):
 ready=read(R/'PPU_READY.json')
 if not ready or ready.get('run_directory')!=str(R) or ready.get('state')!='READY_NOT_STARTED':raise RuntimeError('INVALID_READY_DIRECTORY_OR_STATE')
 if ready['limits']!=LIMITS:raise RuntimeError('READY_BUDGET_CHANGED')
 if require_pointer and (not POINTER.exists() or POINTER.read_text().strip()!=str(R)):raise RuntimeError('STALE_READY_POINTER')
 if not ready['CPU_passed'] or not ready['static_passed']:raise RuntimeError('READY_GATES_NOT_PASSED')
 prep=read(R/'INPUT_PREPARATION.json');validation=read(R/'CPU_VALIDATION.json');static=read(R/'SCRIPT_STATIC_CHECKS.json')
 if not prep or not validation or not static or not all(x.get('passed') for x in (prep,validation,static)):raise RuntimeError('CPU_OR_STATIC_EVIDENCE_MISSING')
 projected=[{k:x[k] for k in ['sample_idx','path','sha256','input_hash','archive_bytes']} for x in prep['samples']]
 if ready['batches']!=projected:raise RuntimeError('READY_BATCHES_NOT_EQUAL_TO_ACCEPTED_PREPARATION')
 if ready['protected_input_hashes']!=read(R/'PROTECTED_HASHES_BEFORE.json'):raise RuntimeError('READY_PROTECTED_MANIFEST_MISMATCH')
 required={str(p) for p in R.glob('*.py')}|{str(R/n) for n in ['launch_ppu.sh','ppu_worker.sh','monitor_ppu.sh','status_ppu.sh','CPU_VALIDATION.json','INPUT_PREPARATION.json','RAW_INPUT_HASHES.json','PROTECTED_HASHES_BEFORE.json','SCRIPT_STATIC_CHECKS.json','SAMPLE_SELECTION.csv','SELECTED_SAMPLES_ALL_GT.csv','CPU_READY.json','REAL_EQUIVALENCE.json','MODEL_CPU_SETUP.json','CPU_COLLECTOR_FINAL_CHECKS.json','LAUNCH_GUARD_TESTS.json','BASH_SYNTAX_CHECKS.json']}
 if not required.issubset(set(ready['preparation_file_hashes'])):raise RuntimeError('INCOMPLETE_PREPARATION_HASH_COVERAGE')
 if len(ready['batches'])!=18 or len({x['sample_idx'] for x in ready['batches']})!=18:raise RuntimeError('READY_SAMPLE_COUNT')
 for item in ready['batches']:
  p=Path(item['path'])
  if p.parent!=R/'batches' or not p.exists() or p.stat().st_size!=item['archive_bytes']:raise RuntimeError('BATCH_MISSING_OR_WRONG_SIZE '+str(p))
  if full and sha(p)!=item['sha256']:raise RuntimeError('BATCH_HASH_CHANGED '+str(p))
 for p,h in ready['preparation_file_hashes'].items():
  if sha(p)!=h:raise RuntimeError('PREPARATION_CODE_OR_GATE_CHANGED '+p)
 if full:
  for p,h in ready['protected_input_hashes'].items():
   if sha(p)!=h:raise RuntimeError('SOURCE_CHECKPOINT_CONFIG_OR_HISTORY_CHANGED '+p)
  for p,h in read(R/'RAW_INPUT_HASHES.json').items():
   if sha(p)!=h:raise RuntimeError('RAW_INPUT_CHANGED '+p)
 res=resources()
 if res['disk_free_bytes']<25*2**30:raise RuntimeError('DISK_BELOW_25GiB')
 if res['cgroup_current_bytes']+8*2**30>res['cgroup_limit_bytes']:raise RuntimeError('CGROUP_NOMINAL_HEADROOM_BELOW_8GiB')
 if sum(x['archive_bytes'] for x in ready['batches'])>LIMITS['cache_bytes']:raise RuntimeError('CACHE_OVER_BUDGET')
 return {'passed':True,'run_directory':str(R),'batch_count':18,'full_hash_check':full,'resources':res,'device_preflight':'not invoked by dry-run','CUDA_initialized_by_this_check':False}
def other_p4_workers():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except (OSError,PermissionError):continue
  if '/codex_reports/P4_PILOT' in cmd and ('/ppu_runner.py' in cmd or '/launch_control.py worker' in cmd):found.append({'pid':int(p.name),'cmdline':cmd})
 return found

def launch(dry=False):
 if dry:
  result=verify_ready(full=True);result.update(dry_run=True,started=False);return result
 with (R/'.launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  verify_ready(full=True)
  if (R/'LAUNCH.json').exists() or (R/'WORKER_FINAL.json').exists() or (R/'PPU_START.json').exists():raise RuntimeError('REFUSE_DUPLICATE_LAUNCH_OR_RETRY')
  conflicts=other_p4_workers()
  if conflicts:raise RuntimeError('ANOTHER_P4_WORKER '+repr(conflicts))
  if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0'):raise RuntimeError('CUDA_VISIBLE_DEVICES_MUST_BE_UNSET_OR_0')
  if os.environ.get('CUDA_DEVICE_ORDER') not in (None,'PCI_BUS_ID'):raise RuntimeError('UNSUPPORTED_DEVICE_MAPPING')
  # Available only when the user explicitly executes this manual entry.
  import pilot_common
  import numpy as np
  import torch
  setup=read(R/'MODEL_CPU_SETUP.json')
  if torch.__version__!=setup['torch'] or np.__version__!=setup['numpy']:raise RuntimeError('FRAMEWORK_VERSION_CHANGED')
  detected=pilot_common.gpu_status()
  nonce=uuid.uuid4().hex
  claim={'run_directory':str(R),'nonce':nonce,'launch_epoch':time.time(),'ready_sha256':sha(R/'PPU_READY.json'),'limits':LIMITS,'device_preflight':detected,'resource_preflight':resources(),'worker':None,'manual_entry':str(R/'launch_ppu.sh')}
  put(R/'LAUNCH.json',claim)
  try:
   log=(R/'ppu.log').open('xb',buffering=0)
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',P4_LAUNCH_NONCE=nonce)
   signal.signal(signal.SIGHUP,signal.SIG_IGN)
   proc=subprocess.Popen(['bash',str(R/'ppu_worker.sh'),nonce],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
   ident=identity(proc.pid)
   if ident is None:raise RuntimeError('WORKER_DIED_BEFORE_IDENTITY_RECORD')
   claim['worker']=ident;put(R/'LAUNCH.json',claim);log.close()
  except BaseException as exc:
   put(R/'WORKER_FINAL.json',{'state':'FAILED','reason':str(exc),'exit_code':1,'end_epoch':time.time()});raise
 return {'started':True,'worker':claim['worker'],'log':str(R/'ppu.log'),'run_directory':str(R)}
def worker_event(phase,**extra):
 put(R/'WORKER_PROGRESS.json',{'phase':phase,'updated_epoch':time.time(),**extra})
 print(json.dumps({'worker_phase':phase,'epoch':time.time(),**extra},ensure_ascii=False),flush=True)
def child(script,seconds):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 parent_pid=os.getpid()
 def parent_death_guard():
  import ctypes
  if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:os._exit(70)
  if os.getppid()!=parent_pid:os.kill(os.getpid(),signal.SIGTERM)
 proc=subprocess.Popen([PYTHON,'-B',str(R/script)],env=env,start_new_session=True,preexec_fn=parent_death_guard)
 worker_event('PPU' if script=='ppu_runner.py' else 'CPU_VALIDATION',child_pid=proc.pid,script=script,timeout_seconds=seconds)
 try:return proc.wait(timeout=seconds)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise TimeoutError(script+' HARD_TIMEOUT '+str(seconds))
 except BaseException:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise

def worker(nonce):
 # Reject accidental direct entry BEFORE writing any run or failure state.
 claim=read(R/'LAUNCH.json')
 if not claim or nonce!=claim.get('nonce') or os.environ.get('P4_LAUNCH_NONCE')!=nonce:raise RuntimeError('INTERNAL_WORKER_REQUIRES_LAUNCH_HANDSHAKE')
 deadline=time.monotonic()+10
 while not claim.get('worker'):
  if time.monotonic()>deadline:raise RuntimeError('NO_LAUNCH_HANDSHAKE')
  time.sleep(.05);claim=read(R/'LAUNCH.json',{})
 if claim['worker']['pid']!=os.getpid() or not alive(claim['worker']):raise RuntimeError('WORKER_IDENTITY_MISMATCH')
 if (R/'WORKER_STARTED.json').exists():raise RuntimeError('REFUSE_WORKER_REENTRY')
 code=1;final='FAILED';reason='WORKER_NOT_COMPLETED'
 try:
  signal.signal(signal.SIGHUP,signal.SIG_IGN)
  for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda s,f:(_ for _ in ()).throw(InterruptedError('WORKER_SIGNAL '+str(s))))
  if sha(R/'PPU_READY.json')!=claim['ready_sha256']:raise RuntimeError('READY_CHANGED_AFTER_LAUNCH')
  if (R/'WORKER_STARTED.json').exists():raise RuntimeError('REFUSE_WORKER_REENTRY')
  put(R/'WORKER_STARTED.json',{'epoch':time.time(),'identity':identity(os.getpid()),'nonce':nonce})
  verify_ready(full=True)
  worker_event('PPU_STARTING')
  rc=child('ppu_runner.py',900)
  if rc!=0:raise RuntimeError('PPU_CHILD_EXIT '+str(rc))
  execution=read(R/'PPU_EXECUTION.json',{})
  if not execution.get('passed') or execution['full_forward_completed']!=40 or len(execution['samples'])!=18:raise RuntimeError('COLLECTION_COUNTERS_FAILED')
  worker_event('COMPLETED_PENDING_VALIDATION')
  start=time.monotonic()
  for script in ['analyze_pilot.py','verify_pilot.py','write_results_report.py']:
   remaining=1800-(time.monotonic()-start)
   if remaining<=0:raise TimeoutError('TOTAL_CPU_VALIDATION_TIMEOUT')
   rc=child(script,remaining)
   if rc!=0:raise RuntimeError(script+' EXIT '+str(rc))
  acceptance=read(R/'ACCEPTANCE.json',{});independent=read(R/'INDEPENDENT_VALIDATION.json',{})
  if not acceptance.get('passed') or not independent.get('passed'):raise RuntimeError('FINAL_ACCEPTANCE_FAILED')
  final='VALIDATED';reason='ALL_CPU_PPU_AND_INDEPENDENT_GATES_PASSED';code=0
 except BaseException as exc:
  reason=str(exc);traceback.print_exc()
  s=read(R/'STATE.json',{})
  if s.get('phase')=='C' and not (R/'SMOKE_TEST.json').exists():final='SMOKE_FAILED'
 finally:
  # Finish all fallible log/sealing work BEFORE writing the terminal exit record.
  manifest_hash=None
  try:
   worker_event(final,exit_code=code,reason=reason)
   with (R/'PPU_FINAL_ARTIFACTS.sha256').open('x') as f:
    for p in sorted(R.rglob('*')):
     if p.is_file() and p.name not in ['PPU_FINAL_ARTIFACTS.sha256','WORKER_FINAL.json','EXIT_CODE.json'] and not p.is_relative_to(R/'batches') and p.suffix!='.tmp':f.write(sha(p)+'  '+str(p.relative_to(R))+'\n')
   manifest_hash=sha(R/'PPU_FINAL_ARTIFACTS.sha256')
  except BaseException as sealing_error:
   final='FAILED';code=1;reason='FINALIZATION_FAILED: '+str(sealing_error)
  out={'state':final,'exit_code':code,'reason':reason,'end_epoch':time.time(),'identity':identity(os.getpid()),'final_artifact_manifest_sha256':manifest_hash}
  put(R/'WORKER_FINAL.json',out)
  put(R/'EXIT_CODE.json',{'exit_code':code,'state':final,'epoch':time.time()})
 return code

def status():
 claim=read(R/'LAUNCH.json');s=read(R/'STATE.json',{});prog=read(R/'WORKER_PROGRESS.json',{});final=read(R/'WORKER_FINAL.json');ex=read(R/'EXIT_CODE.json')
 live=alive(claim.get('worker')) if claim else False
 reason=None
 if not claim:
  try:verify_ready(full=False);state='READY_NOT_STARTED'
  except BaseException as exc:state='FAILED';reason=str(exc)
 elif final:
  state=final['state'];reason=final.get('reason')
  if state=='VALIDATED' and live and ex is None:
   state='COMPLETED_PENDING_VALIDATION';reason='FINAL_EXIT_RECORD_PENDING'
  elif state=='VALIDATED':
   a=read(R/'ACCEPTANCE.json',{});v=read(R/'INDEPENDENT_VALIDATION.json',{});e=read(R/'PPU_EXECUTION.json',{})
   recorded=final.get('identity',{})
   identity_ok=bool(claim.get('worker') and all(recorded.get(k)==claim['worker'].get(k) for k in ['pid','start_ticks','boot_id']))
   sealed=(R/'PPU_FINAL_ARTIFACTS.sha256').exists() and final.get('final_artifact_manifest_sha256')==sha(R/'PPU_FINAL_ARTIFACTS.sha256')
   if not (sealed and identity_ok and ex and ex['exit_code']==0 and ex.get('state')=='VALIDATED' and a.get('passed') and v.get('passed') and e.get('passed') and e.get('full_forward_completed')==40 and len(e.get('samples',[]))==18 and v.get('protected_hashes_unchanged')):state='FAILED';reason='FINAL_MARKERS_NOT_BACKED_BY_ACCEPTANCE'
   elif live:state='COMPLETED_PENDING_VALIDATION';reason='ACCEPTANCE_PASSED_WAITING_FOR_WORKER_EXIT'
 elif live:
  state='COMPLETED_PENDING_VALIDATION' if (R/'PPU_EXECUTION.json').exists() else 'RUNNING'
 else:state='FAILED';reason='WORKER_IDENTITY_NOT_ALIVE_OR_EXIT_CODE_MISSING'
 return {'state':state,'run_directory':str(R),'worker_alive_with_identity':live,'worker':claim.get('worker') if claim else None,'current_sample_idx':s.get('current_sample_idx'),'completed_samples':len(s.get('completed_samples',[])),'expected_samples':18,'forward_attempted':s.get('forward_attempted',0),'forward_completed':s.get('forward_completed',0),'max_forwards':40,'phase':prog.get('phase',s.get('phase')),'last_progress_epoch':max(prog.get('updated_epoch',0),s.get('updated_epoch',0)) or None,'exit_code':ex.get('exit_code') if ex else None,'error_or_stop_reason':reason or s.get('failure')}
def monitor(interval):
 if not 5<=interval<=10:raise ValueError('interval must be 5..10 seconds')
 while True:
  z=status();print(json.dumps(z,ensure_ascii=False,indent=2),flush=True)
  log=R/'ppu.log'
  if log.exists():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-5000));print('\n'.join(f.read().decode(errors='replace').splitlines()[-8:]),flush=True)
  vendor=shutil.which('nvidia-smi')
  if vendor:
   q=subprocess.run([vendor,'--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],capture_output=True,text=True,timeout=10)
   print(q.stdout if q.returncode==0 else 'Device telemetry unavailable: '+q.stderr,flush=True)
  if z['state'] not in ('RUNNING','COMPLETED_PENDING_VALIDATION'):break
  time.sleep(interval)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','worker','status','monitor','check']);p.add_argument('--nonce');p.add_argument('--dry-run',action='store_true');p.add_argument('--interval',type=float,default=5);a=p.parse_args()
 try:
  if a.command=='launch':print(json.dumps(launch(a.dry_run),ensure_ascii=False,indent=2))
  elif a.command=='worker':sys.exit(worker(a.nonce))
  elif a.command=='status':print(json.dumps(status(),ensure_ascii=False,indent=2))
  elif a.command=='monitor':monitor(a.interval)
  elif a.command=='check':print(json.dumps(verify_ready(full=True),ensure_ascii=False,indent=2))
 except BaseException as exc:
  if isinstance(exc,KeyboardInterrupt) and a.command=='monitor':
   print('监控已停止；后台作业不受影响。');sys.exit(0)
  if isinstance(exc,SystemExit):raise
  print(json.dumps({'passed':False,'state':'FAILED','error':str(exc),'run_directory':str(R)},ensure_ascii=False),file=sys.stderr);sys.exit(1)
