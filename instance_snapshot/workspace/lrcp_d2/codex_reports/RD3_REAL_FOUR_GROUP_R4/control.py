"""Standard-library-only R4 permissions, immutable plan, durable ledger, status.
Import/help/status/monitor cannot import a model or initialize CUDA.
"""
import os,sys,json,time,hashlib,fcntl,shutil,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPORTS=ROOT.parent
R2=REPORTS/'RD3_REAL_FOUR_GROUP_R2';R3=REPORTS/'RD3_SCHEMA_RECOVERY_R3'
OLD=REPORTS/'RD3_MESSAGE_UTILITY_PILOT';FROZEN=OLD/'FROZEN_MANIFEST.json'
CKPT=Path('/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth')
FROZEN_HASH='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c'
CHECKPOINT_HASH='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'
PYTHON='/workspace/venvs/lrcp/bin/python'
REAL_ROOT=ROOT/'real_run';BUDGET=ROOT/'R4_NEW_RUN_BUDGET.json';PLAN=ROOT/'R4_EXECUTION_PLAN_27.json'
SEAL=ROOT/'R4_SOURCE_AND_CHECKPOINT_SEAL.json'
CONDITIONS=('F','G-B','G-C','E')
LIMITS={'max_attempts':27,'wall_seconds':1800,'cpu_rss_bytes':8*2**30,'disk_free_bytes':10*2**30,'output_bytes':2*2**30,'allocated_bytes':8*2**30,'reserved_bytes':12*2**30}

def require(ok,reason):
 if not ok:raise RuntimeError(reason)
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def atomic(p,value,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if exclusive:require(not p.exists(),'EXCLUSIVE_FILE_EXISTS '+str(p))
 tmp=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
 with tmp.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def plan_check():
 p=load(PLAN);require(sha(PLAN)==(ROOT/'R4_EXECUTION_PLAN_27.sha256').read_text().split()[0],'PLAN_HASH')
 require(p['frozen_sha256']==FROZEN_HASH and p['checkpoint_sha256']==CHECKPOINT_HASH,'PLAN_IDENTITY')
 expected=[(r['sample_idx'],c) for r in load(FROZEN)['samples'] for c in CONDITIONS]
 reused=[(x['sample_idx'],x['condition']) for x in p['reused_pairs']]
 tasks=[(x['sample_idx'],x['condition']) for x in p['tasks']]
 require(len(expected)==32 and len(set(expected))==32 and len(reused)==len(set(reused))==5 and len(tasks)==len(set(tasks))==27,'EXACT_32_MINUS_5_EQUALS_27')
 require(tasks==[x for x in expected if x not in set(reused)] and set(tasks).isdisjoint(reused) and set(expected)==set(tasks)|set(reused),'EXACT_PLAN_SET_DIFFERENCE')
 require(tasks[:3]==[(424,c) for c in ('G-B','G-C','E')] and [x['ordinal'] for x in p['tasks']]==list(range(1,28)),'PLAN_ORDER')
 for task in p['tasks']:
  row=next(r for r in load(FROZEN)['samples'] if r['sample_idx']==task['sample_idx'])
  require(task['input_archive_sha256']==row['input']['sha256'] and task['scene_id']==row['scene_id'],'PLAN_ARCHIVE_IDENTITY')
 return p

def verify_seal():
 require(sha(FROZEN)==FROZEN_HASH and sha(CKPT)==CHECKPOINT_HASH,'FROZEN_CHECKPOINT_HASH')
 s=load(SEAL)
 for group in ('execution_sources','historical_sources'):
  for path,h in s[group].items():require(sha(path)==h,'SEALED_SOURCE_CHANGED '+path)
 require(s['plan_sha256']==sha(PLAN),'SEALED_PLAN_CHANGED');plan_check();return s

def cpu_env():return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
def verify_prepared():
 s=verify_seal();prep=load(ROOT/'R4_PREP_INDEPENDENT_VALIDATION.json')
 require(prep['passed'] and prep['state']=='RD3_R4_READY_FOR_MANUAL_RUN','PREPARATION_NOT_READY')
 for p,h in prep['sealed_evidence'].items():require(sha(p)==h,'PREPARATION_EVIDENCE_CHANGED '+p)
 require(prep['source_seal_sha256']==sha(SEAL),'PREPARATION_SOURCE_SEAL_CHANGED')
 return prep

def safe_budget_path(path):
 path=Path(path).resolve()
 require(path==BUDGET or path.is_relative_to(ROOT/'cpu_tests'),'BUDGET_SCOPE')
 return path

def reserve(path,task,kind):
 path=safe_budget_path(path);require((path==BUDGET and kind=='REAL') or (path!=BUDGET and kind=='CPU_TEST'),'BUDGET_KIND')
 if kind=='REAL':authorize_worker()
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b=load(path);plan=plan_check();attempts=b['attempts']
  require(b['max_new_real_forward_attempts']==27 and len(attempts)<27,'R4_FORWARD_BUDGET_EXHAUSTED')
  require(b['plan_sha256']==sha(PLAN),'BUDGET_PLAN_HASH')
  require(not any((x['sample_idx'],x['condition'])==(task['sample_idx'],task['condition']) for x in attempts),'NO_AUTOMATIC_RETRY')
  require(task==plan['tasks'][len(attempts)],'ORDERED_TASK_OR_REUSED_RECORD_REJECTED')
  require(all(x['committed_epoch'] is not None for x in attempts),'PREVIOUS_ATTEMPT_NOT_COMMITTED_NO_RESUME')
  a={**task,'kind':kind,'status':'RESERVED','reserved_epoch':time.time(),'entered_epoch':None,'returned_epoch':None,'validated_epoch':None,'committed_epoch':None}
  attempts.append(a);b['state']='RUNNING';atomic(path,b);return a

def event(path,ordinal,status):
 path=safe_budget_path(path)
 if path==BUDGET:authorize_worker()
 transition={'MODEL_ENTERED':('RESERVED','entered_epoch'),'MODEL_RETURNED':('MODEL_ENTERED','returned_epoch'),'VALIDATED':('MODEL_RETURNED','validated_epoch'),'COMMITTED':('VALIDATED','committed_epoch')}
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);b=load(path);a=b['attempts'][ordinal-1];previous,key=transition[status]
  require(a['ordinal']==ordinal and a['status']==previous and a[key] is None,'INVALID_FORWARD_EVENT_ORDER')
  a['status']=status;a[key]=time.time();atomic(path,b)

def identity(pid):
 try:
  fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':int(pid),'start_ticks':int(fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'process_state':fields[0]}
 except (OSError,ValueError,IndexError):return None

def alive(who):
 if not who:return False
 now=identity(who['pid'])
 return bool(now and now['process_state'] not in ('Z','X') and all(now[k]==who[k] for k in ('pid','start_ticks','boot_id')))

def authorize_worker(token=None):
 claim=load(REAL_ROOT/'claim.json') if (REAL_ROOT/'claim.json').exists() else {}
 require(claim.get('mode')=='REAL_FORWARD' and claim.get('manual_confirmation') is True,'MANUAL_REAL_PERMISSION_REQUIRED')
 require(os.environ.get('RD3_R4_WORKER_TOKEN')==claim.get('token') and (token is None or token==claim['token']),'REAL_WORKER_TOKEN_REQUIRED')
 require(claim['owner']['pid']==os.getppid() and alive(claim['owner']),'REAL_WORKER_PARENT_REQUIRED')
 return claim

def require_manual_confirm(confirm):
 require(confirm and os.environ.get('RD3_R4_MANUAL_ENTRY')=='1','MANUAL_CONFIRMATION_REQUIRED')

def used_bytes(root):
 total=0;seen=set()
 for base,dirs,files in os.walk(root):
  for name in files:
   try:s=(Path(base)/name).stat()
   except FileNotFoundError:continue
   if (s.st_dev,s.st_ino) not in seen:total+=s.st_size;seen.add((s.st_dev,s.st_ino))
 return total

def lock_status():
 p=REAL_ROOT/'run.lock'
 if not p.exists():return {'exists':False,'kernel_lock_observed':False}
 inode=p.stat().st_ino
 try:lines=Path('/proc/locks').read_text().splitlines();held=any(str(inode)==x.split()[5].split(':')[-1] for x in lines if len(x.split())>5)
 except (OSError,IndexError):held=None
 return {'exists':True,'kernel_lock_observed':held,'claim_exists':(REAL_ROOT/'claim.json').exists()}

def status():
 b=load(BUDGET);ats=b['attempts'];m=load(REAL_ROOT/'manifest.json') if (REAL_ROOT/'manifest.json').exists() else {}
 claim=load(REAL_ROOT/'claim.json') if (REAL_ROOT/'claim.json').exists() else {};worker=load(REAL_ROOT/'worker_claim.json') if (REAL_ROOT/'worker_claim.json').exists() else {}
 counts={key:sum(a[key+'_epoch'] is not None for a in ats) for key in ('entered','returned','committed')}
 state=m.get('state','PREPARED');error=m.get('error')
 if state=='RUNNING' and not alive(claim.get('owner')):state='STOPPED';error=error or 'OWNER_EXITED_NO_AUTOMATIC_RESUME'
 current=next((a for a in reversed(ats) if a['committed_epoch'] is None),ats[-1] if ats else None)
 last=next((a for a in reversed(ats) if a['committed_epoch'] is not None),None)
 return {'state':state,'reserved':len(ats),**counts,'remaining_budget':27-len(ats),'real_forwards':counts['entered'],'scientific_record_count':5+counts['committed'],'current_task':current,'last_commit':last,'latest_error':error,'worker_pid':worker.get('pid'),'owner_pid':claim.get('owner',{}).get('pid'),'worker_alive':alive(worker),'lock':lock_status(),'tasks':[{**t,'status':next((a['status'] for a in ats if a['ordinal']==t['ordinal']),'PENDING')} for t in plan_check()['tasks']],'log_directory':str(REAL_ROOT),'reports':str(ROOT/'analysis')}
