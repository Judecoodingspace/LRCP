"""Foreground R4 supervisor. CPU_PREFLIGHT never imports the real backend.
REAL_FORWARD requires shell confirmation and a durable parent/worker token.
"""
from control import *
import argparse,signal,traceback,ctypes,contextlib
from device_guard import query_live
from runtime_helpers import tree_rss

class Tee:
 def __init__(self,stream,file):self.stream=stream;self.file=file
 def write(self,s):self.stream.write(s);self.stream.flush();self.file.write(s);self.file.flush();return len(s)
 def flush(self):self.stream.flush();self.file.flush()

def resource_guard(root,*,real=True,own=None):
 rss=tree_rss(os.getpid());free=shutil.disk_usage(ROOT).free;size=used_bytes(root)
 require(rss<LIMITS['cpu_rss_bytes'],'CPU_TREE_RSS_LIMIT');require(free>LIMITS['disk_free_bytes'],'DISK_RESERVE');require(size<LIMITS['output_bytes'],'OUTPUT_PHYSICAL_BYTES_LIMIT')
 device=query_live(free,rss,own) if real else None
 return {'rss':rss,'disk_free':free,'unique_inode_output_bytes':size,'device':device}

def manifest_save(root,path,records,state,error=None):
 ats=load(path)['attempts'];m={'schema':'RD3_R4_RUN_V1','backend':'REAL' if Path(path)==BUDGET else 'CPU_TEST','state':state,'plan_sha256':sha(PLAN),'records':records,'reserved':len(ats),'entered':sum(a['entered_epoch'] is not None for a in ats),'returned':sum(a['returned_epoch'] is not None for a in ats),'committed':sum(a['committed_epoch'] is not None for a in ats),'scientific_record_count':5+len(records),'error':error,'updated_epoch':time.time()}
 atomic(Path(root)/'manifest.json',m);return m

def independent(root,candidate,ordinal,kind):
 out=Path(root)/f'validation_{ordinal:03d}.json'
 cmd=[PYTHON,'-B',str(ROOT/'validate_cpu.py'),str(root),'--candidate',str(candidate),'--output',str(out)]
 if kind!='REAL':cmd.append('--cpu-test')
 with (Path(root)/f'validation_{ordinal:03d}.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=cpu_env(),timeout=180)
 require(proc.returncode==0,'INDEPENDENT_CANDIDATE_REJECTED '+str(out));result=load(out)
 require(result['passed'] and result['sample_idx']==load(Path(candidate)/'meta.json')['sample_idx'],'INDEPENDENT_RESULT_IDENTITY')
 return out

def verify_reused_runtime():
 out=REAL_ROOT/'reused_start_validation.json'
 cmd=[PYTHON,'-B',str(ROOT/'dataset_cpu.py'),'--revalidate-five','--output',str(out)]
 with (REAL_ROOT/'reused_start_validation.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=cpu_env(),timeout=360)
 require(proc.returncode==0 and load(out)['passed'],'REUSED_FIVE_STARTUP_REJECTED')

def condition_identity(root,p,meta,records):
 if p['condition']=='F':return
 from dataset_cpu_shared import full_record_path,compare_condition_identity
 baseline=full_record_path(p['row']['sample_idx'],records,root)
 compare_condition_identity(baseline,Path(p['candidate_path']))

def run_sequence(backend,root,path,kind,tasks,*,inject_fault=None):
 from candidate_store import commit_candidate
 import gc
 records=[];manifest_save(root,path,records,'RUNNING');p=None
 try:
  for task in tasks:
   row=next(x for x in load(FROZEN)['samples'] if x['sample_idx']==task['sample_idx']);condition=task['condition']
   resource_guard(root,real=kind=='REAL',own=os.getpid() if kind=='REAL' else None)
   if kind=='REAL':
    import torch
    require(torch.cuda.memory_allocated(0)<LIMITS['allocated_bytes'] and torch.cuda.memory_reserved(0)<LIMITS['reserved_bytes'],'DEVICE_MEMORY_LIMIT')
   require(not (Path(root)/'capture'/f"sample_{row['sample_idx']}_{condition}").exists(),'ACCEPTED_RECORD_DUPLICATE')
   p=backend.prepare(row);attempt=reserve(path,task,kind);p.update(run_root=root,attempt=attempt,condition=condition)
   print(kind,'sample',row['sample_idx'],'condition',condition,'attempt',attempt['ordinal'],'RESERVED',flush=True)
   def model_event(state):
    event(path,attempt['ordinal'],state);manifest_save(root,path,records,'RUNNING');print(kind,row['sample_idx'],condition,attempt['ordinal'],state,flush=True)
   p['on_model_event']=model_event
   if inject_fault=='reserved_crash':raise RuntimeError('TEST_RESERVED_CRASH')
   a,meta=backend.forward_collect(p,condition)
   if inject_fault=='post_return':raise RuntimeError('TEST_AFTER_MODEL_RETURN')
   if kind=='REAL':condition_identity(root,p,meta,records)
   validation=independent(root,p['candidate_path'],attempt['ordinal'],kind)
   if condition=='F' and kind=='REAL':require(load(validation)['baseline_control']['native_alpha1_agrees'],'EVERY_NEW_SAMPLE_FIRST_F_BASELINE')
   event(path,attempt['ordinal'],'VALIDATED');rec=commit_candidate(root,p,validation);records.append(rec)
   event(path,attempt['ordinal'],'COMMITTED');manifest_save(root,path,records,'RUNNING')
   print(kind,'sample',row['sample_idx'],'condition',condition,'attempt',attempt['ordinal'],'COMMITTED',flush=True)
   del a,meta,p;p=None;gc.collect()
  b=load(path);b['state']='COMPLETED';atomic(path,b);return manifest_save(root,path,records,'COMPLETED')
 except BaseException as e:
  if p is not None:
   from diagnostics import save_unaccepted
   p['diagnostic_path']=save_unaccepted(root,p['attempt'],load(path)['attempts'][-1]['returned_epoch'] is not None,p['leaf_before'],p['batch'],e,p.get('returned_heads',{}),extra={'runner_failure':True})
  b=load(path);b['state']='FAILED';b['error']=str(e);atomic(path,b)
  m=manifest_save(root,path,records,'FAILED',str(e));atomic(Path(root)/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'counts':{k:m[k] for k in ('reserved','entered','returned','committed')},'candidate_path':p.get('candidate_path') if p else None,'diagnostic_path':p.get('diagnostic_path') if p else None})
  raise

def worker(token):
 claim=authorize_worker(token);ctypes.CDLL(None).prctl(1,signal.SIGTERM);require(os.getppid()==claim['owner']['pid'],'OWNER_DIED')
 atomic(REAL_ROOT/'worker_claim.json',identity(os.getpid()),exclusive=True)
 def killed(signum,frame):raise RuntimeError('SUPERVISOR_TERMINATION')
 signal.signal(signal.SIGTERM,killed);verify_prepared();resource_guard(REAL_ROOT,real=True,own=os.getpid())
 from real_backend import RealBackend
 backend=RealBackend.build();atomic(REAL_ROOT/'NATIVE_ENTRY_RUNTIME.json',backend.native_entry,exclusive=True)
 return run_sequence(backend,REAL_ROOT,BUDGET,'REAL',plan_check()['tasks'])

def launch(confirm):
 require_manual_confirm(confirm)
 with (REAL_ROOT/'run.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('R4_ALREADY_RUNNING '+json.dumps(status()))
  require(not (REAL_ROOT/'claim.json').exists(),'PERSISTENT_CLAIM_NO_REENTRY '+json.dumps(status()))
  verify_prepared();b=load(BUDGET);require(b['state']=='UNSTARTED' and not b['attempts'],'BUDGET_NOT_UNSTARTED_NO_RESUME')
  resource_guard(REAL_ROOT,real=True);verify_reused_runtime()
  token=uuid.uuid4().hex;atomic(REAL_ROOT/'claim.json',{'token':token,'mode':'REAL_FORWARD','manual_confirmation':True,'owner':identity(os.getpid()),'time':time.time(),'plan_sha256':sha(PLAN)},exclusive=True)
  env=dict(cpu_env(),CUDA_VISIBLE_DEVICES='0',RD3_R4_WORKER_TOKEN=token)
  # Sole creation of a real model worker. Reached only after manual confirmation.
  cmd=[PYTHON,'-B',str(ROOT/'runner.py'),'--mode','REAL_FORWARD','--worker-token',token]
  started=time.monotonic();peak=0;records=[]
  def terminate(signum,frame):raise RuntimeError('MANUAL_SUPERVISOR_INTERRUPTED')
  previous={s:signal.signal(s,terminate) for s in (signal.SIGTERM,signal.SIGINT)}
  try:
   with (REAL_ROOT/'worker.log').open('x') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:
     last=None
     while proc.poll() is None:
      require(time.monotonic()-started<LIMITS['wall_seconds'],'REAL_WALL_TIME_LIMIT')
      obs=resource_guard(REAL_ROOT,real=True,own=proc.pid);peak=max(peak,obs['rss'])
      atomic(REAL_ROOT/'resources.json',{**obs,'peak_cpu_tree_rss':peak,'elapsed_seconds':time.monotonic()-started,'worker_pid':proc.pid})
      st=status();mark=(st['entered'],st['returned'],st['committed'])
      if mark!=last:print('PROGRESS',json.dumps({k:st[k] for k in ('state','current_task','entered','returned','committed','remaining_budget')},ensure_ascii=False),flush=True);last=mark
      time.sleep(2)
    except BaseException:
     if proc.poll() is None:
      os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     raise
   require(proc.returncode==0,'R4_WORKER_FAILED_SEE_WORKER_LOG')
   require(status()['committed']==27,'INCOMPLETE_NEW_RECORDS');return status()
  except BaseException as e:
   b=load(BUDGET);b['state']='FAILED';b['error']=str(e);atomic(BUDGET,b)
   existing=load(REAL_ROOT/'manifest.json') if (REAL_ROOT/'manifest.json').exists() else {}
   manifest_save(REAL_ROOT,BUDGET,existing.get('records',[]),'FAILED',str(e));atomic(REAL_ROOT/'supervisor_failure.json',{'error':str(e),'traceback':traceback.format_exc(),'status':status()});raise
  finally:
   for s,h in previous.items():signal.signal(s,h)
   print('FINAL_STATUS',json.dumps(status(),ensure_ascii=False),flush=True)

def dry_run():
 verify_seal();plan=plan_check();b=load(BUDGET)
 require(b['max_new_real_forward_attempts']==27,'BUDGET_MAX')
 return {'passed':True,'mode':'CPU_PREFLIGHT','dry_run':True,'tasks':plan['tasks'],'reused_count':5,'budget_state':b['state'],'reserved':len(b['attempts']),'real_backend_imported':'real_backend' in sys.modules,'new_real_forwards':0,'model_or_torch_imported':'torch' in sys.modules}

def main():
 p=argparse.ArgumentParser(description='R4 manual foreground launcher; default CPU_PREFLIGHT is read-only dry-run.')
 p.add_argument('--mode',choices=['CPU_PREFLIGHT','REAL_FORWARD'],default='CPU_PREFLIGHT');p.add_argument('--confirm-real-forwards',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--worker-token',help=argparse.SUPPRESS);args=p.parse_args()
 if args.mode=='CPU_PREFLIGHT':require(not args.worker_token and not args.confirm_real_forwards,'CPU_MODE_NO_REAL_ARGUMENTS');print(json.dumps(dry_run(),ensure_ascii=False));return
 require(not args.dry_run,'REAL_MODE_CANNOT_BE_DRY_RUN')
 if args.worker_token:worker(args.worker_token);return
 require_manual_confirm(args.confirm_real_forwards)
 with (ROOT/'logs/manual_run.log').open('a',buffering=1) as log:
  with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):launch(args.confirm_real_forwards)
if __name__=='__main__':
 try:main()
 except BaseException as e:print('R4_LAUNCH_REJECTED',str(e),file=sys.stderr,flush=True);sys.exit(2)
