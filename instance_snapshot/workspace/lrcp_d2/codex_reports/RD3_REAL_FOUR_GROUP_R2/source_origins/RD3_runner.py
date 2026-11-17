"""RD3 single entry. Durable attempt reservations; failed forwards never retried.
Supervisor owns flock and persistent claim; only its authenticated child builds model.
"""
from rd3_common import *
import argparse,subprocess,fcntl,signal,uuid,traceback
sys.path.insert(2,str(C3))
from device_guard import query_live
from runtime_common import tree_rss,identity,alive
PYTHON='/workspace/venvs/lrcp/bin/python'
LIMITS={'max_attempts':32,'wall_seconds':1800,'cpu_rss_bytes':8*2**30,'disk_free_bytes':10*2**30,'output_bytes':2*2**30,'allocated_bytes':8*2**30,'reserved_bytes':12*2**30}

def budget_path(root,mock):return Path(root)/'budget.json' if mock else R/'RUN_BUDGET.json'
def manifest_save(root,value,mock):
 write(Path(root)/'manifest.json',value)
 if not mock:write(R/'REAL_RUN_MANIFEST.json',value)
def reserve(root,mock,sid,condition,maximum=32):
 path=budget_path(root,mock)
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b=load(path)
  require(b['max_attempts']==maximum and len(b['attempts'])<maximum,'FORWARD_BUDGET_EXHAUSTED')
  require(not any(x['sample_idx']==sid and x['condition']==condition for x in b['attempts']),'NO_AUTOMATIC_RETRY')
  attempt={'ordinal':len(b['attempts'])+1,'sample_idx':sid,'condition':condition,'reserved_at':time.time(),'pid':os.getpid(),'status':'RESERVED'}
  b['attempts'].append(attempt);write(path,b);return attempt

def resource_guard(root,mock,own=None,fault=None):
 rss=tree_rss(os.getpid());free=shutil.disk_usage(R).free
 require(rss<LIMITS['cpu_rss_bytes'] and fault!='rss','CPU_RSS_LIMIT')
 require(free>LIMITS['disk_free_bytes'] and fault!='disk','DISK_RESERVE')
 size=0
 for base,dirs,files in os.walk(root):
  for name in files:
   try:size+=(Path(base)/name).stat().st_size
   except FileNotFoundError:pass  # atomic shard directory rename
 require(size<LIMITS['output_bytes'] and fault!='output','OUTPUT_LIMIT')
 require(fault!='device','DEVICE_IDENTITY_MISMATCH')
 return {'rss':rss,'disk_free':free,'output_bytes':size,'device':None if mock else query_live(free,rss,own)}

def gates():
 audit=load(R/'CPU_DATA_AUDIT.json');v=load(R/'INTERVENTION_VALIDATION.json');ready=load(R/'CPU_READY.json')
 require(audit['passed'] and v['passed'] and ready['passed'],'CPU_SCIENTIFIC_GATE')
 f=load(R/'FROZEN_MANIFEST.json');require(f['sufficient'] and 4<=len(f['samples'])<=8 and len({x['scene_id'] for x in f['samples']})>=2,'ELIGIBILITY_GATE')
 require(sha_file(R/'FROZEN_MANIFEST.json')==ready['frozen_sha256'],'FROZEN_MANIFEST_CHANGED')
 for p,h in ready['locked_files'].items():require(sha_file(p)==h,'DEPENDENCY_OR_SOURCE_CHANGED '+p)
 require(sha_file(CKPT)==CH,'CHECKPOINT_CHANGED')
 return f

def commit(root,a,meta,rec,fault=None):
 root=Path(root);target=root/'capture'/f"sample_{rec['sample_idx']}_{rec['condition']}";target.parent.mkdir(exist_ok=True)
 tmp=target.with_name(target.name+'.tmp');require(not target.exists() and not tmp.exists(),'DUPLICATE_SHARD')
 tmp.mkdir()
 with (tmp/'arrays.npz').open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 if fault=='write':raise RuntimeError('INJECTED_WRITE_FAILURE')
 write(tmp/'meta.json',meta)
 write(tmp/'manifest.json',{'schema':'RD3_ATOMIC_SHARD_V1','files':{n:sha_file(tmp/n) for n in ('arrays.npz','meta.json')}})
 os.rename(tmp,target);fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 rec.update(relative_path=str(target.relative_to(root)),manifest_sha256=sha_file(target/'manifest.json'));return rec

def independent(root,index,mock):
 cmd=[PYTHON,'-B',str(R/'validate_cpu.py'),str(root),'--record',str(index)]+(['--mock'] if mock else [])
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 with (Path(root)/f'control_{index}.log').open('w') as log:
  proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
 require(proc.returncode==0,'INDEPENDENT_BASELINE_CONTROL_FAILED')
 require(load(Path(root)/f'control_{index}.json')['passed'],'INDEPENDENT_BASELINE_CONTROL_FAILED')

def worker(root,mock,token,fault=None,count=1):
 root=Path(root);claim=load(root/'claim.json')
 require(claim['token']==token and claim['owner']['pid']==os.getppid() and alive(claim['owner']),'WORKER_OWNER_IDENTITY')
 with (root/'worker_claim.json').open('x') as f:json.dump(identity(os.getpid()),f);f.flush();os.fsync(f.fileno())
 frozen=load(R/'FROZEN_MANIFEST.json') if mock else gates()
 rows=frozen['samples'][:count] if mock else frozen['samples'];require(rows,'NO_FROZEN_SAMPLES')
 manifest={'schema':'RD3_RUN_V1','backend':'mock' if mock else 'real','state':'RUNNING','frozen_sha256':sha_file(R/'FROZEN_MANIFEST.json'),'records':[],'real_forwards':0};manifest_save(root,manifest,mock)
 try:
  resource_guard(root,mock,os.getpid() if not mock else None,fault)
  if mock:
   from mock_backend import build
   backend=build()
  else:
   from backend import Backend
   backend=Backend.real()
  for row in rows:
   for condition in CONDITIONS:
    resource_guard(root,mock,os.getpid() if not mock else None)
    if not mock:
     import torch
     require(torch.cuda.memory_allocated(0)<LIMITS['allocated_bytes'] and torch.cuda.memory_reserved(0)<LIMITS['reserved_bytes'],'DEVICE_MEMORY_LIMIT')
    p=backend.prepare(row)
    reserve(root,mock,row['sample_idx'],condition)
    if fault=='reserved_crash':raise RuntimeError('INJECTED_CRASH_AFTER_RESERVE')
    if fault=='forward':raise RuntimeError('INJECTED_FORWARD_FAILURE')
    def model_event(status):
     bp=budget_path(root,mock)
     with bp.with_suffix('.lock').open('a') as lk:
      fcntl.flock(lk,fcntl.LOCK_EX);budget=load(bp);budget['attempts'][-1]['status']=status;write(bp,budget)
    p['on_model_event']=model_event
    a,meta=backend.forward_collect(p,condition)
    if not mock:manifest['real_forwards']+=1
    if fault=='control' and condition=='F':a['gt_tp70'][0]=1-a['gt_tp70'][0]
    rec=commit(root,a,meta,{'sample_idx':row['sample_idx'],'condition':condition},fault)
    manifest['records'].append(rec);manifest_save(root,manifest,mock)
    write(root/'progress.json',{'state':'RUNNING','completed':len(manifest['records']),'sample_idx':row['sample_idx'],'condition':condition})
    print('COMMITTED',row['sample_idx'],condition,'attempts',len(load(budget_path(root,mock))['attempts']),flush=True)
    del a,meta,p
    import gc;gc.collect()
    if condition=='F':independent(root,len(manifest['records'])-1,mock)
    if fault=='after_control':raise RuntimeError('INJECTED_POST_CONTROL_CRASH')
  manifest['state']='MOCK_CAPTURE_COMPLETE' if mock else 'REAL_CAPTURE_COMPLETE';manifest_save(root,manifest,mock)
  write(root/'progress.json',{'state':manifest['state'],'completed':len(manifest['records'])})
 except BaseException as e:
  manifest['state']='FAILED';manifest['error']=str(e);manifest_save(root,manifest,mock)
  write(root/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'attempts':len(load(budget_path(root,mock))['attempts']),'records':len(manifest['records'])});raise

def launch(root,mock=False,fault=None,count=1):
 root=Path(root).resolve();root.mkdir(exist_ok=True,parents=True)
 with (root/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  require(not (root/'claim.json').exists(),'PERSISTENT_RUN_CLAIM_NO_REENTRY')
  if not mock:gates()
  resource_guard(root,mock,None,fault if fault in ('rss','disk','device','output') else None)
  token=uuid.uuid4().hex;write(root/'claim.json',{'token':token,'owner':identity(os.getpid()),'mock':mock,'started_at':time.time()},exclusive=True)
  bp=budget_path(root,mock)
  if mock:write(bp,{'schema':'RD3_BUDGET_V1','max_attempts':32,'attempts':[],'mock_only':True},exclusive=True)
  else:require(load(bp)['attempts']==[] and load(bp)['max_attempts']==32,'REAL_BUDGET_ALREADY_USED')
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='' if mock else '0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  cmd=[PYTHON,'-B',str(R/'runner.py'),'--worker',token,'--root',str(root),'--count',str(count)]+(['--mock'] if mock else ['--real'])+(['--fault',fault] if fault else [])
  start=time.monotonic();peak=0
  with (root/'worker.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   try:
    while proc.poll() is None:
     require(time.monotonic()-start<LIMITS['wall_seconds'],'WALL_TIME_LIMIT')
     rr=resource_guard(root,mock,proc.pid if not mock else None);peak=max(peak,rr['rss'])
     write(root/'resources.json',{**rr,'elapsed':time.monotonic()-start,'peak_tree_rss':peak,'worker_pid':proc.pid})
     time.sleep(2)
   except BaseException as e:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    write(root/'supervisor_failure.json',{'error':str(e)});raise
  require(proc.returncode==0,'WORKER_FAILED_SEE_WORKER_LOG')
 return load(root/'manifest.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--mock',action='store_true');g.add_argument('--real',action='store_true');p.add_argument('--root',required=True);p.add_argument('--worker');p.add_argument('--fault');p.add_argument('--count',type=int,default=1);args=p.parse_args()
 require(not args.fault or args.mock,'FAULT_INJECTION_MOCK_ONLY')
 if args.mock:os.environ['CUDA_VISIBLE_DEVICES']=''
 if args.worker:worker(args.root,args.mock,args.worker,args.fault,args.count)
 else:launch(args.root,args.mock,args.fault,args.count)
