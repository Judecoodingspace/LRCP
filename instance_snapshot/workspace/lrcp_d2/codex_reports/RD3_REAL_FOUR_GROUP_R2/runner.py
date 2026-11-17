"""R2 real supervisor. Real worker imports ONLY real_backend; no Mock branch.
CPU orchestration tests inject a collector from a separate test-only process.
"""
from rd3_common import *
import argparse,subprocess,fcntl,signal,uuid,traceback,gc,ctypes
sys.path.insert(2,str(C3))
from device_guard import query_live
from runtime_common import tree_rss,identity,alive
from candidate_store import commit_candidate
PYTHON='/workspace/venvs/lrcp/bin/python'
LIMITS={'max_attempts':32,'wall_seconds':1800,'cpu_rss_bytes':8*2**30,'disk_free_bytes':10*2**30,'output_bytes':2*2**30,'allocated_bytes':8*2**30,'reserved_bytes':12*2**30}
BUDGET=R/'R2_RUN_BUDGET.json'

def used_bytes(root):
 total=0;seen=set()
 for base,dirs,files in os.walk(root):
  for name in files:
   try:s=(Path(base)/name).stat()
   except FileNotFoundError:continue
   key=(s.st_dev,s.st_ino)
   if key not in seen:total+=s.st_size;seen.add(key)
 return total

def resource_guard(root,*,real=True,own=None):
 rss=tree_rss(os.getpid());free=shutil.disk_usage(R).free;size=used_bytes(root)
 require(rss<LIMITS['cpu_rss_bytes'],'CPU_TREE_RSS_LIMIT');require(free>LIMITS['disk_free_bytes'],'DISK_RESERVE');require(size<LIMITS['output_bytes'],'OUTPUT_PHYSICAL_BYTES_LIMIT')
 device=query_live(free,rss,own) if real else None
 return {'rss':rss,'disk_free':free,'unique_inode_output_bytes':size,'device':device}

def reserve(path,sid,condition,kind):
 path=Path(path)
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b=load(path)
  require(b['max_attempts']==32 and len(b['attempts'])<32,'R2_FORWARD_BUDGET_EXHAUSTED')
  require(not any(a['sample_idx']==sid and a['condition']==condition for a in b['attempts']),'NO_AUTOMATIC_RETRY')
  attempt={'ordinal':len(b['attempts'])+1,'sample_idx':sid,'condition':condition,'kind':kind,'status':'RESERVED','reserved_epoch':time.time(),'entered_epoch':None,'returned_epoch':None,'validated_epoch':None,'committed_epoch':None}
  b['attempts'].append(attempt);write(path,b);return attempt

def event(path,ordinal,status):
 path=Path(path)
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b=load(path);a=b['attempts'][ordinal-1]
  require(a['ordinal']==ordinal,'ATTEMPT_IDENTITY');a['status']=status
  key={'MODEL_ENTERED':'entered_epoch','MODEL_RETURNED':'returned_epoch','VALIDATED':'validated_epoch','COMMITTED':'committed_epoch'}.get(status)
  if key:require(a[key] is None,'DUPLICATE_FORWARD_EVENT');a[key]=time.time()
  write(path,b)

def manifest_save(root,path,kind,records,state,error=None):
 b=load(path);attempts=b['attempts']
 m={'schema':'RD3_R2_RUN_V1','backend':kind,'state':state,'frozen_sha256':sha_file(FROZEN),'records':records,'reserved':len(attempts),'entered':sum(a['entered_epoch'] is not None for a in attempts),'returned':sum(a['returned_epoch'] is not None for a in attempts),'committed':len(records),'real_forwards':sum(a['entered_epoch'] is not None for a in attempts) if kind=='REAL' else 0}
 if error:m['error']=error
 write(Path(root)/'manifest.json',m)
 if kind=='REAL':write(R/'REAL_RUN_MANIFEST.json',m)
 return m

def independent(root,candidate,ordinal,kind):
 path=Path(root)/f'validation_{ordinal:03d}.json';cmd=[PYTHON,'-B',str(R/'validate_cpu.py'),str(root),'--candidate',str(candidate),'--output',str(path)]
 if kind!='REAL':cmd.append('--cpu-test')
 with (Path(root)/f'validation_{ordinal:03d}.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=180)
 require(proc.returncode==0,'INDEPENDENT_CANDIDATE_REJECTED '+str(path))
 require(load(path)['passed'],'INDEPENDENT_CANDIDATE_REJECTED');return path

def run_sequence(backend,root,path,kind,rows,*,inject_fault=None):
 root=Path(root);records=[];manifest_save(root,path,kind,records,'RUNNING')
 try:
  for row in rows:
   for condition in CONDITIONS:
    resource_guard(root,real=kind=='REAL',own=os.getpid() if kind=='REAL' else None)
    if kind=='REAL':
     import torch
     require(torch.cuda.memory_allocated(0)<LIMITS['allocated_bytes'] and torch.cuda.memory_reserved(0)<LIMITS['reserved_bytes'],'DEVICE_MEMORY_LIMIT')
    p=backend.prepare(row);attempt=reserve(path,row['sample_idx'],condition,kind);p.update(run_root=root,attempt=attempt,condition=condition)
    def model_event(status):event(path,attempt['ordinal'],status);manifest_save(root,path,kind,records,'RUNNING')
    p['on_model_event']=model_event
    if inject_fault=='reserved_crash':raise RuntimeError('TEST_RESERVED_CRASH')
    a,meta=backend.forward_collect(p,condition)
    if inject_fault=='scientific_control':
     from candidate_store import complete_candidate
     a['gt_tp70'][0]=1-a['gt_tp70'][0];complete_candidate(p,a,meta)
    result=independent(root,p['candidate_path'],attempt['ordinal'],kind)
    event(path,attempt['ordinal'],'VALIDATED')
    if kind=='REAL' and attempt['ordinal']==1:
     r=load(result);require(row['sample_idx']==324 and condition=='F' and r['baseline_control']['native_alpha1_agrees'],'MANDATORY_FIRST_F_BASELINE')
     write(R/'FIRST_FULL_BASELINE_CONTROL.json',{**r,'validation_path':str(result),'validation_sha256':sha_file(result)},exclusive=True)
    rec=commit_candidate(root,p,result);records.append(rec);event(path,attempt['ordinal'],'COMMITTED');manifest_save(root,path,kind,records,'RUNNING')
    print('REAL_COMMITTED' if kind=='REAL' else 'CPU_TEST_COMMITTED',row['sample_idx'],condition,'attempt',attempt['ordinal'],flush=True)
    del a,meta,p;gc.collect()
  return manifest_save(root,path,kind,records,'REAL_CAPTURE_COMPLETE' if kind=='REAL' else 'CPU_TEST_CAPTURE_COMPLETE')
 except BaseException as e:
  m=manifest_save(root,path,kind,records,'RD3_R2_REAL_REJECTED' if kind=='REAL' else 'CPU_TEST_REJECTED',str(e))
  write(root/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'reserved':m['reserved'],'entered':m['entered'],'returned':m['returned'],'committed':m['committed'],'candidate_path':p.get('candidate_path') if 'p' in locals() else None,'diagnostic_path':p.get('diagnostic_path') if 'p' in locals() else None})
  if kind=='REAL' and not (R/'FIRST_FULL_BASELINE_CONTROL.json').exists():write(R/'FIRST_FULL_BASELINE_CONTROL.json',{'passed':False,'state':'RD3_R2_REAL_REJECTED','error':str(e),'attempts':m['reserved'],'entered':m['entered'],'returned':m['returned'],'committed':m['committed']})
  raise

def gates():
 ready=load(R/'CPU_ENTRY_READY.json');require(ready['passed'],'CPU_ENTRY_NOT_READY')
 seal=load(R/'SOURCE_AND_CHECKPOINT_SEAL.json')
 require(sha_file(FROZEN)=='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c' and sha_file(CKPT)==CH,'FROZEN_MODEL_SEAL')
 for p,h in seal['execution_sources'].items():require(sha_file(p)==h,'ENTRY_SOURCE_CHANGED '+p)
 for p,h in seal['historical_sources'].items():require(sha_file(p)==h,'HISTORICAL_SOURCE_CHANGED '+p)
 for p,h in ready['evidence_hashes'].items():require(sha_file(p)==h,'CPU_EVIDENCE_CHANGED')
 require(load(R/'HISTORY_BASELINE_AUDIT.json')['baseline_evidence_complete'],'BASELINE_EVIDENCE_MISSING')
 return load(FROZEN)

def worker(root,token):
 root=Path(root);claim=load(root/'claim.json');require(claim['token']==token and claim['owner']['pid']==os.getppid() and alive(claim['owner']),'WORKER_OWNER_IDENTITY')
 ctypes.CDLL(None).prctl(1,signal.SIGTERM);require(os.getppid()==claim['owner']['pid'],'OWNER_DIED')
 with (root/'worker_claim.json').open('x') as f:json.dump(identity(os.getpid()),f);f.flush();os.fsync(f.fileno())
 def killed(signum,frame):raise RuntimeError('SUPERVISOR_TERMINATION')
 signal.signal(signal.SIGTERM,killed)
 frozen=gates();resource_guard(root,real=True,own=os.getpid())
 try:
  from real_backend import RealBackend
  backend=RealBackend.build();write(root/'NATIVE_ENTRY_RUNTIME.json',backend.native_entry,exclusive=True)
  return run_sequence(backend,root,BUDGET,'REAL',frozen['samples'])
 except BaseException as e:
  if not (root/'failure.json').exists():manifest_save(root,BUDGET,'REAL',[],'RD3_R2_REAL_REJECTED',str(e));write(root/'failure.json',{'error':str(e),'traceback':traceback.format_exc()})
  raise

def launch(root):
 root=Path(root).resolve();require(root==R/'real_run','FIXED_REAL_RUN_DIRECTORY');root.mkdir(exist_ok=True)
 with (root/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);require(not (root/'claim.json').exists(),'PERSISTENT_R2_CLAIM_NO_REENTRY')
  gates();resource_guard(root,real=True);require(load(BUDGET)['attempts']==[],'R2_BUDGET_ALREADY_USED')
  token=uuid.uuid4().hex;write(root/'claim.json',{'token':token,'owner':identity(os.getpid()),'backend':'REAL','max_attempts':32,'time':time.time()},exclusive=True)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  cmd=[PYTHON,'-B',str(R/'runner.py'),'--root',str(root),'--worker',token]
  started=time.monotonic();peak=0
  with (root/'worker.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   try:
    while proc.poll() is None:
     require(time.monotonic()-started<LIMITS['wall_seconds'],'REAL_WALL_TIME_LIMIT')
     obs=resource_guard(root,real=True,own=proc.pid);peak=max(peak,obs['rss']);write(root/'resources.json',{**obs,'peak_cpu_tree_rss':peak,'elapsed_seconds':time.monotonic()-started,'worker_pid':proc.pid})
     time.sleep(2)
   except BaseException as e:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    write(root/'supervisor_failure.json',{'error':str(e)});raise
  require(proc.returncode==0,'R2_WORKER_FAILED_SEE_LOG')
 return load(root/'manifest.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker');args=p.parse_args()
 if args.worker:worker(args.root,args.worker)
 else:launch(args.root)
