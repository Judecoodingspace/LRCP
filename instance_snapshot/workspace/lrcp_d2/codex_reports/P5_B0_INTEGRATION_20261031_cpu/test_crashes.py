import os,tempfile,shutil,json,subprocess,sys
from pathlib import Path
R=Path(__file__).parent;roots=[];stats={}
script='''from pathlib import Path
from runner_engine import run
from mock_backend import MockBackend
import os,sys
class Crash(MockBackend):
 def fault(self,stage):
  if stage==sys.argv[2]:os._exit(71)
run(Path(sys.argv[1]),Path(sys.argv[3]),Crash())
'''
try:
 for stage in ('after_reserve','after_forward','during_write','after_commit'):
  root=Path(tempfile.mkdtemp(prefix='b0_crash_'));roots.append(root)
  env=dict(os.environ,PYTHONPATH=str(R),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
  proc=subprocess.run([sys.executable,'-B','-c',script,str(root),stage,str(R/'P5_SAMPLE_PROPOSAL_V2.csv')],env=env,capture_output=True,timeout=120)
  assert proc.returncode==71,(stage,proc.stderr.decode())
  b=json.loads((root/'budget.json').read_text());assert len(b['attempts'])==1 and (root/'run_claim.json').exists()
  again=subprocess.run([sys.executable,'-B','-c',script,str(root),'none',str(R/'P5_SAMPLE_PROPOSAL_V2.csv')],env=env,capture_output=True,timeout=120)
  assert again.returncode!=0 and b'FileExistsError' in again.stderr and len(json.loads((root/'budget.json').read_text())['attempts'])==1
  committed=list((root/'capture').glob('sample_*')) if (root/'capture').exists() else []
  assert len(committed)==(1 if stage=='after_commit' else 0)
  stats[stage]={'persisted_attempts':1,'rerun_refused':True,'committed_shards':len(committed)}
 (R/'CRASH_TEST.json').write_text(json.dumps({'passed':True,'mock_only':True,'stages':stats,'model_forwards':0,'CUDA_initialized':False},indent=2))
finally:
 for root in roots:shutil.rmtree(root)
