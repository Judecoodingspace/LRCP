#!/usr/bin/env python3
"""Owner entry and process supervisor. Status/monitor never import a model."""
import os,sys,json,time,uuid,fcntl,argparse,subprocess,signal,shutil
from pathlib import Path
from runtime_common import ROOT,C2,PYTHON,BUDGET,read,atomic,sha,identity,alive,tree_rss,status,child_environment,capture_bytes

def assert_ready():
 ready=read(ROOT/'P5_B0_READY.json',{})
 if not ready.get('passed') or ready.get('state')!='PPU_READY_NOT_STARTED' or ready.get('budget')!=BUDGET:raise RuntimeError('NOT_READY')
 for p,h in ready['locked_files'].items():
  if sha(p)!=h:raise RuntimeError('READY_SOURCE_CHANGED '+p)
 return ready

def preflight():
 p=subprocess.run([PYTHON,'-B',str(ROOT/'preflight_cpu.py')],env=child_environment(True),capture_output=True,text=True,timeout=300)
 if p.returncode:raise RuntimeError('PREFLIGHT_FAILED '+p.stdout[-1200:]+p.stderr[-600:])
 return json.loads(p.stdout)

def run_validation(root,mock=False):
 root=Path(root).resolve();kind=read(root/'manifest.json',{}).get('kind')
 if kind!=('mock' if mock else 'real'):raise RuntimeError('VALIDATOR_MODE_MISMATCH')
 if not mock and (root!=ROOT/'real_run' or status(root).get('state') not in ('PPU_CAPTURE_COMPLETE','CPU_VALIDATION_COMPLETE')):raise RuntimeError('CAPTURE_NOT_COMPLETE')
 cmd=[PYTHON,'-B',str(C2/'INDEPENDENT_VALIDATOR.py'),str(root)]+(['--mock'] if mock else [])
 with (root/'independent_validation.log').open('w') as log:
  err=root/'independent_validation.stderr.log'
  with err.open('w') as errors:
   p=subprocess.Popen(cmd,stdout=log,stderr=errors,env=child_environment(True),start_new_session=True)
   deadline=time.monotonic()+600
   try:
    while p.poll() is None:
     if tree_rss(p.pid)>BUDGET['cpu_rss_bytes']:raise RuntimeError('VALIDATOR_CPU_RSS_LIMIT')
     if time.monotonic()>deadline:raise RuntimeError('VALIDATOR_TIMEOUT')
     time.sleep(.25)
   except BaseException:
    if p.poll() is None:
     os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
  p.stderr=err.read_text()
 if p.returncode:raise RuntimeError('INDEPENDENT_VALIDATION_FAILED '+(root/'independent_validation.log').read_text()[-1000:]+p.stderr[-300:])
 result=json.loads((root/'independent_validation.log').read_text())
 if not result.get('passed') or result.get('mock_only')!=mock:raise RuntimeError('INVALID_VALIDATION_RESULT')
 atomic(root/'CPU_VALIDATION_COMPLETE.json',{'passed':True,'mock_only':mock,'manifest_sha256':sha(root/'manifest.json'),'result':result})
 progress=read(root/'progress.json',{});progress.update(state='MOCK_VALIDATION_COMPLETE' if mock else 'CPU_VALIDATION_COMPLETE');atomic(root/'progress.json',progress)
 return {'passed':True,'mock_only':mock,'validated_records':result['validated_records']}

def launch(root,mock=False,approve=False,fault=''):
 root=Path(root).resolve()
 if mock:
  if root.parent!=ROOT/'mock_runs':raise RuntimeError('MOCK_DIRECTORY_REQUIRED')
 else:
  if root!=ROOT/'real_run' or not approve or not sys.stdin.isatty() or fault:raise RuntimeError('OWNER_TERMINAL_AND_EXPLICIT_APPROVE_22_REQUIRED')
  assert_ready()
 root.mkdir(parents=True,exist_ok=True)
 with (root/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'owner.json').exists() or (root/'run_claim.json').exists() or (root/'budget.json').exists():raise RuntimeError('REFUSE_RESTART_OR_DUPLICATE')
  from engine_adapter import resource_check
  try:
   resource_check(mock,root,fault)
   preflight()
  except BaseException as e:
   atomic(root/'preflight_failure.json',{'error':str(e),'epoch':time.time(),'device_busy':str(e)=='DEVICE_BUSY','forward_started':False})
   raise
  atomic(root/'execution_sources.json',{str(p):sha(p) for p in ROOT.glob('*.py')})
  token=uuid.uuid4().hex;owner={'run_id':root.name,'token':token,'mock_only':mock,'supervisor':identity(os.getpid()),'created_epoch':time.time(),'max_forwards':22}
  atomic(root/'owner.json',owner)
  atomic(root/'progress.json',{'state':'MOCK_RUNNING_CONTROLS' if mock else 'PPU_RUNNING_CONTROLS','mock_only':mock,'started':0,'completed':0,'core_completed':0,'controls_verified':0})
  child=None;started=time.monotonic()
  def interrupted(sig,frame):raise InterruptedError('SUPERVISOR_SIGNAL '+str(sig))
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
  parent=os.getpid()
  def guard_parent():
   import ctypes
   if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0:os._exit(70)
   if os.getppid()!=parent:os._exit(71)
  try:
   env=child_environment(mock);env['P5_B0_OWNER_TOKEN']=token
   cmd=['bash',str(ROOT/'ppu_worker.sh'),'--root',str(root),'--token',token]
   if mock:cmd+=['--mock']
   if fault:cmd+=['--fault',fault]
   with (root/'ppu.log').open('xb') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True,preexec_fn=guard_parent)
    atomic(root/'supervised_child.json',identity(child.pid))
    last_device=0;max_rss=0
    while True:
     code=child.poll()
     if code is not None:break
     mem=tree_rss(child.pid);max_rss=max(max_rss,mem)
     if mem>BUDGET['cpu_rss_bytes']:raise RuntimeError('CPU_TREE_RSS_LIMIT')
     if time.monotonic()-started>BUDGET['wall_seconds']:raise RuntimeError('WALL_TIME_LIMIT')
     if shutil.disk_usage(root).free<BUDGET['min_disk_bytes']:raise RuntimeError('DISK_LOW')
     if time.monotonic()-last_device>5:
      device=resource_check(mock,root,'',child.pid if not mock else None)
      cap_bytes=capture_bytes(root)
      if cap_bytes>BUDGET['capture_max_bytes']:raise RuntimeError('CAPTURE_BYTES_LIMIT')
      atomic(root/'supervisor_resources.json',{'device':device,'CPU_tree_RSS_bytes':mem,'CPU_tree_peak_RSS_bytes':max_rss,'disk_free_bytes':shutil.disk_usage(root).free,'capture_bytes':cap_bytes,'updated_epoch':time.time()});last_device=time.monotonic()
     time.sleep(.25)
   if code!=0 or read(root/'state.json',{}).get('state')!=('MOCK_COMPLETE' if mock else 'CAPTURE_COMPLETE'):raise RuntimeError('WORKER_EXIT '+str(code))
   if mock:
    if fault=='validation_fail':
     # Explicit MOCK_ONLY corruption after collection, before independent replay.
     mf=read(root/'manifest.json');first=Path(mf['records'][0]['path'])/'payload.bin'
     first.write_bytes(first.read_bytes()+b'corrupt')
    run_validation(root,True)
   return {'passed':True,'state':status(root)['state'],'mock_only':mock,'root':str(root)}
  except BaseException as e:
   if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   progress=read(root/'progress.json',{});prev=progress.get('state','')
   phase='MOCK_FAILED' if mock else ('PPU_STOPPED_CONTROL_FAILED' if progress.get('controls_verified',0)<4 else 'PPU_CAPTURE_FAILED')
   progress.update(state=phase,error=str(e),previous_state=prev);atomic(root/'progress.json',progress)
   atomic(root/'failure.json',{'error':str(e),'epoch':time.time(),'reserved':len(read(root/'budget.json',{'attempts':[]})['attempts']),'no_retry':True})
   raise

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','status','monitor','validate','preflight']);p.add_argument('--mock',action='store_true');p.add_argument('--root',default=str(ROOT/'real_run'));p.add_argument('--approve-22',action='store_true');p.add_argument('--fault',default='');p.add_argument('--once',action='store_true');args=p.parse_args()
 if args.command=='launch':return launch(args.root,args.mock,args.approve_22,args.fault)
 if args.command=='preflight':return preflight()
 if args.command=='validate':
  try:return run_validation(args.root,args.mock)
  except BaseException as e:
   root=Path(args.root).resolve()
   if root.exists() and (root==ROOT/'real_run' or root.parent==ROOT/'mock_runs'):
    atomic(root/'CPU_VALIDATION_FAILURE.json',{'passed':False,'error':str(e),'mock_only':args.mock})
    progress=read(root/'progress.json',{});progress.update(state='MOCK_VALIDATION_FAILED' if args.mock else 'CPU_VALIDATION_FAILED',error=str(e));atomic(root/'progress.json',progress)
   raise
 if args.command=='status':return status(args.root)
 while True:
  print(json.dumps(status(args.root),indent=2),flush=True)
  if args.once:return {'read_only':True}
  time.sleep(3)
if __name__=='__main__':
 try:print(json.dumps(main(),indent=2,ensure_ascii=False))
 except BaseException as e:
  if isinstance(e,SystemExit):raise
  print(json.dumps({'passed':False,'error':type(e).__name__+': '+str(e)}),file=sys.stderr);sys.exit(1)
