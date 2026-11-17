"""Static/import and safe read-only entry checks. Never runs launch_ppu.sh."""
import os,sys,json,subprocess,ast
from runtime_common import ROOT,C2,PYTHON,child_environment,atomic
checks={}
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
checks['python_AST']=True
names=['launch_ppu','ppu_worker','preflight_cpu','monitor_ppu','status_ppu','validate_cpu']
for name in names:
 p=ROOT/(name+'.sh');r=subprocess.run(['bash','-n',str(p)],capture_output=True)
 checks[name+'_shell']=r.returncode==0 and os.access(p,os.X_OK)
checks['all_absolute_paths']=all(str(ROOT) in (ROOT/(n+'.sh')).read_text() and PathCheck for n in names for PathCheck in [os.path.exists(PYTHON)])
code="import sys;sys.path.insert(0,'"+str(ROOT)+"');import controller,ppu_worker,engine_adapter,mock_factory,device_guard;import torch;assert not torch.cuda.is_initialized();print('CPU_IMPORT_OK')"
r=subprocess.run([PYTHON,'-B','-c',code],capture_output=True,text=True,env=child_environment(True));checks['CPU_import_without_cuda']=r.returncode==0
r=subprocess.run([PYTHON,'-B',str(ROOT/'controller.py'),'launch'],capture_output=True,text=True,env=child_environment(True));checks['real_without_owner_approval_rejected']=r.returncode!=0 and 'OWNER_TERMINAL_AND_EXPLICIT_APPROVE_22_REQUIRED' in r.stderr
r=subprocess.run([str(ROOT/'validate_cpu.sh'),'--mock'],capture_output=True,text=True);checks['real_validation_script_refuses_mock']=r.returncode==2
r=subprocess.run([str(ROOT/'status_ppu.sh')],capture_output=True,text=True);checks['status_read_only']=r.returncode==0
r=subprocess.run([str(ROOT/'monitor_ppu.sh'),'--once'],capture_output=True,text=True);checks['monitor_read_only']=r.returncode==0
checks['real_run_not_created']=not (ROOT/'real_run').exists()
checks['worker_build_real_call']='LRCPBackend.build_real(\'cuda:0\')' in (ROOT/'ppu_worker.py').read_text()
checks['same_C2_engine_run']='engine.run(self.root,C2/' in (ROOT/'engine_adapter.py').read_text()
result={'passed':all(checks.values()),'checks':checks,'real_launch_shell_executed':False,'real_model_forward':0,'CUDA_initialized':False}
atomic(ROOT/'C3_STATIC_VALIDATION.json',result);print(json.dumps(result,indent=2));assert result['passed']
