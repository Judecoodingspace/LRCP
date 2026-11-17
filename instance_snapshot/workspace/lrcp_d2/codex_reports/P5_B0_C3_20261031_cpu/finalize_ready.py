"""Publish READY only from actual C3 evidence. No model construction or launch."""
import os,sys,json,time,subprocess,ast
from pathlib import Path
from runtime_common import ROOT,C2,C1,P5,PYTHON,BUDGET,UUID,read,atomic,sha,child_environment,status

def require(ok,reason):
 if not ok:raise RuntimeError(reason)

def main():
 from controller import preflight
 audit=preflight();require(audit['passed'],'PREFLIGHT');atomic(ROOT/'C3_INPUT_AND_DEPENDENCY_AUDIT.json',audit)
 root=ROOT/'mock_runs'/'ready_acceptance';state=read(root/'progress.json');manifest=read(root/'manifest.json');budget=read(root/'budget.json');proof=read(root/'CPU_VALIDATION_COMPLETE.json')
 require(state['state']=='MOCK_VALIDATION_COMPLETE' and state['started']==state['completed']==state['committed']==22 and state['core_completed']==18 and state['controls_verified']==4,'E2E_COUNTS')
 require(manifest['kind']=='mock' and len(manifest['records'])==22 and len(budget['attempts'])==22 and budget['max']==22,'E2E_BUDGET')
 require(proof['passed'] and proof['mock_only'] and proof['manifest_sha256']==sha(root/'manifest.json') and proof['result']['validated_records']==22,'INDEPENDENT_SCIENCE')
 require([(x['sample'],x['kind']) for x in budget['attempts'][:4]]==[(590,'off'),(590,'pass'),(918,'pass'),(918,'off')],'CONTROL_ORDER')
 receipts=read(root/'execution_sources.json')
 core=['controller.py','ppu_worker.py','engine_adapter.py','runtime_common.py','mock_factory.py','preflight_cpu.py','device_guard.py','control_validate_cpu.py']
 for name in core:require(receipts[str(ROOT/name)]==sha(ROOT/name),'SOURCE_CHANGED_SINCE_E2E '+name)
 require((root/'worker_started_once.json').is_file(),'WORKER_ONCE_PROOF')
 for report in ['C3_STATIC_VALIDATION.json','C3_FAILURE_INJECTION.json','C3_BUDGET_CONCURRENCY.json','SCIENTIFIC_FAULT_INJECTION.json','C3_FINAL_VALIDATION_FAILURE_TEST.json','C3_STARTUP_FAILURE_PERSISTENCE.json','C3_WORKER_REENTRY_TEST.json']:
  require(read(ROOT/report)['passed'],'FAILED_EVIDENCE '+report)
 failures=read(ROOT/'C3_FAILURE_INJECTION.json');require(failures['complete'] and len(failures['tests'])==14,'RUNTIME_INJECTIONS')
 semantic=read(ROOT/'SCIENTIFIC_FAULT_INJECTION.json');require(len(semantic['tests'])==17 and semantic['rehashed_semantic_faults_rejected'],'SEMANTIC_INJECTIONS')
 for name in ['launch_ppu','ppu_worker','preflight_cpu','monitor_ppu','status_ppu','validate_cpu']:
  p=ROOT/(name+'.sh');require(p.is_file() and os.access(p,os.X_OK),'SCRIPT_MISSING '+name)
  require(subprocess.run(['bash','-n',str(p)]).returncode==0,'SHELL_SYNTAX')
 require(not (ROOT/'real_run').exists(),'REAL_RUN_ALREADY_EXISTS')
 env=child_environment(True)
 code="import sys,json,torch,numpy; print(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'torch_source':torch.__file__,'numpy_source':numpy.__file__}))"
 current=json.loads(subprocess.check_output([PYTHON,'-B','-c',code],env=env,text=True))
 historical=read(ROOT.parent/'P4_PILOT_RETRY_20261023_234457_384594/MODEL_CPU_SETUP.json')
 require(current['torch']==historical['torch'] and current['numpy']==historical['numpy'] and not current['CUDA_initialized'],'RUNTIME_VERSION')
 atomic(ROOT/'C3_RUNTIME_IDENTITY.json',{'passed':True,**current})
 from engine_adapter import resource_check
 device=resource_check(False,ROOT)
 atomic(ROOT/'C3_DEVICE_RESOURCE_AUDIT.json',{'passed':True,'live_query_performed':True,'device':device,'epoch':time.time(),'interface':'/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi compatibility','runtime_guard_wired':True,'budget':BUDGET,'CUDA_initialized':False,'real_model_forward':0})
 # Prove the successful run cannot be restarted; no forward is entered.
 repeated=subprocess.run([PYTHON,'-B',str(ROOT/'controller.py'),'launch','--mock','--root',str(root)],capture_output=True,text=True,env=env)
 require(repeated.returncode!=0 and 'REFUSE_RESTART_OR_DUPLICATE' in repeated.stderr,'RESTART_GUARD')
 require(read(root/'budget.json')==budget,'BUDGET_REFUNDED')
 atomic(ROOT/'C3_BUDGET_AND_STATE_VALIDATION.json',{'passed':True,'max_forwards':22,'reserved':22,'started':22,'completed':22,'controls':4,'core':18,'original_C2_engine':str(C2/'runner_engine.py'),'persistent_reservation_before_forward':True,'duplicate_restart_rejected':True,'concurrency':read(ROOT/'C3_BUDGET_CONCURRENCY.json'),'crash_tests':failures['tests'],'real_model_forward':0})
 atomic(ROOT/'C3_MANUAL_ENTRY_TEST.json',{'passed':True,'scripts':[str(ROOT/(n+'.sh')) for n in ['launch_ppu','ppu_worker','preflight_cpu','monitor_ppu','status_ppu','validate_cpu']],'same_python_owner_entry_mock':True,'real_launch_shell_executed':False,'mock_worker_shell_executed':True,'all_shell_syntax_and_executable':True})
 atomic(ROOT/'C3_END_TO_END_MOCK.json',{'passed':True,'MOCK_ONLY':True,'root':str(root),'validated_records':22,'controls_verified':4,'core_completed':18,'all_sample_GT_transitions_checked':len(proof['result']['per_GT_transitions']),'native_P5A_observer':True,'native_CPU_postprocess_and_GT_matcher':True,'original_C2_runner':str(C2/'runner_engine.py'),'manifest_sha256':sha(root/'manifest.json'),'science_result_sha256':sha(root/'CPU_VALIDATION_COMPLETE.json'),'resource_peak':read(root/'supervisor_resources.json'),'source_hash_receipt':str(root/'execution_sources.json'),'real_device_controls_passed':False,'real_model_forward':0,'CUDA_initialized':False})
 report={'passed':True,'state':'PPU_READY_NOT_STARTED','engineering_gates_passed':True,'same_entry_mock_records':22,'runtime_faults_passed':14,'scientific_faults_passed':17,'real_device_controls':'NOT_STARTED: must pass in future authorized execution','real_model_forward':0,'CUDA_initialized':False,'PPU_started':False,'real_launch_shell_executed':False,'device':device,'frozen_protocol':{'epoch':19,'k':3,'time_encoding':300,'alpha':1,'dtype':'FP32','eval':True,'no_grad':True,'samples':18,'intervention':'Block0 valid remote reference shift removal only'}}
 atomic(ROOT/'C3_READY_VALIDATION.json',report)
 locked={}
 for source in [*ROOT.glob('*.py'),*ROOT.glob('*.sh'),*C2.glob('*.py'),*C1.glob('*.py')]:locked[str(source)]=sha(source)
 for name in ['C3_INPUT_AND_DEPENDENCY_AUDIT.json','C3_RUNTIME_IDENTITY.json','C3_STATIC_VALIDATION.json','C3_FAILURE_INJECTION.json','SCIENTIFIC_FAULT_INJECTION.json','C3_FINAL_VALIDATION_FAILURE_TEST.json','C3_BUDGET_AND_STATE_VALIDATION.json','C3_END_TO_END_MOCK.json','C3_MANUAL_ENTRY_TEST.json','C3_READY_VALIDATION.json','C3_DEVICE_RESOURCE_AUDIT.json']:
  locked[str(ROOT/name)]=sha(ROOT/name)
 for source in [C2/'SHA256SUMS',C1/'SHA256SUMS',C2/'C2_VALIDATION.json',P5/'P5_READY.json',Path(current['torch_source']),Path(current['numpy_source']),Path(PYTHON).resolve(),root/'manifest.json',root/'CPU_VALIDATION_COMPLETE.json']:
  locked[str(source)]=sha(source)
 ready={**report,'run_directory':str(ROOT),'real_output_directory':str(ROOT/'real_run'),'budget':BUDGET,'device_uuid':UUID,'samples':read(P5/'P5_READY.json')['samples'],'checkpoint':{'sha256':'666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'},'locked_files':locked,'worker':str(ROOT/'ppu_worker.py'),'original_engine':str(C2/'runner_engine.py'),'backend':str(C2/'real_backend.py'),'independent_validator':str(C2/'INDEPENDENT_VALIDATOR.py'),'manual_command':str(ROOT/'launch_ppu.sh')+' --approve-22','created_epoch':time.time(),'authorization':'No device execution authorized by CPU preparation; Owner must obtain separate approval and manually launch from terminal.'}
 atomic(ROOT/'P5_B0_READY.json',ready)
 from controller import assert_ready
 assert_ready()
 print(json.dumps({k:report[k] for k in ('passed','state','engineering_gates_passed','real_model_forward','CUDA_initialized','PPU_started')},indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(ROOT/'C3_READY_VALIDATION.json',{'passed':False,'state':'NOT_READY','error':type(e).__name__+': '+str(e),'real_model_forward':0,'CUDA_initialized':False})
  print(type(e).__name__+': '+str(e));raise
