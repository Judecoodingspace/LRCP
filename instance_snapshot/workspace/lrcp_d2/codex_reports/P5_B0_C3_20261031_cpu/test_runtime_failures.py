"""Faults through the SAME controller/worker/C2 engine as Owner real mode."""
import os,sys,json,subprocess,time
from pathlib import Path
from runtime_common import ROOT,PYTHON,read,atomic,child_environment,sha
cases={}
def test_fault(fault,needle,started=None,core=0):
 root=ROOT/'mock_runs'/('fault_'+fault)
 cmd=[PYTHON,'-B',str(ROOT/'controller.py'),'launch','--mock','--root',str(root),'--fault',fault]
 p=subprocess.run(cmd,capture_output=True,text=True,env=child_environment(True),timeout=600)
 (ROOT/('fault_'+fault+'.log')).write_text(p.stdout+'\n'+p.stderr)
 progress=read(root/'progress.json',{});budget=read(root/'budget.json',{'attempts':[]})
 error=p.stderr+' '+p.stdout+' '+json.dumps(read(root/'failure.json',{}))
 if (root/'ppu.log').exists():error+=' '+(root/'ppu.log').read_text()[-5000:]
 ok=p.returncode!=0 and needle in error and progress.get('core_completed',0)==core
 if started is not None:ok=ok and progress.get('started',0)==started
 reserved=len(budget['attempts'])
 # Restart the exact same root and demonstrate reservations never refunded.
 if (root/'owner.json').exists():
  q=subprocess.run(cmd,capture_output=True,text=True,env=child_environment(True),timeout=30)
  ok=ok and q.returncode!=0 and 'REFUSE_RESTART_OR_DUPLICATE' in q.stderr and len(read(root/'budget.json',{'attempts':[]})['attempts'])==reserved
 cases[fault]={'passed':bool(ok),'error_expected':needle,'reserved':reserved,'started':progress.get('started',0),'core_completed':progress.get('core_completed',0),'state':progress.get('state'),'restart_rejected':(root/'owner.json').exists()}
 atomic(ROOT/'C3_FAILURE_INJECTION.json',{'passed':all(x['passed'] for x in cases.values()),'complete':False,'same_formal_entry':True,'tests':cases,'MOCK_ONLY':True})
 print(fault,json.dumps(cases[fault]),flush=True)
 if not ok:raise AssertionError(error[-6000:])
for args in [('device_identity','DEVICE_IDENTITY_MISMATCH',0),('device_busy','DEVICE_BUSY',0),('device_query','DEVICE_QUERY_FAILED',0),('disk','DISK_LOW',0),('rss','RSS_HIGH',0),('forward_crash','WORKER_EXIT 83',1),('write_crash','WORKER_EXIT 85',1),('commit_crash','WORKER_EXIT 86',1),('control_mismatch','CONTROL_INDEPENDENT_FAILED',1),('budget_exhausted','FORWARD_LIMIT',0),('control_crash','WORKER_EXIT 87',4),('core_reserved_crash','WORKER_EXIT 84',4),('core_disk','DISK_LOW',4),('core_rss','RSS_HIGH',4)]:test_fault(*args)
atomic(ROOT/'C3_FAILURE_INJECTION.json',{'passed':all(x['passed'] for x in cases.values()),'complete':True,'same_formal_entry':True,'tests':cases,'MOCK_ONLY':True,'real_model_forward':0,'CUDA_initialized':False})
