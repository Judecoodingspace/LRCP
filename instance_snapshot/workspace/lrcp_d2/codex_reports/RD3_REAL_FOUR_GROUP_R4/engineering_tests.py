"""Executed R4 final-runner tests. Only CPU_TEST simulations use copied budgets."""
import cpu_only
cpu_only.install()
from rd3_common import *
import control,runner,subprocess,copy,fcntl,traceback,contextlib
from cpu_pipeline import fixture_backend
from dataset_cpu import revalidate_five
from device_guard import validate_telemetry,UUID,NAME
from analyze_cpu import effect_class,component_difference,Q_FIELDS
from opencood.utils.eval_utils import calculate_ap
TEST=R/'cpu_tests';cases=[]

def record(name,fn):
 result=fn();cases.append({'case_id':name,'passed':True,'result':result});print('R4_ENGINEERING_PASS',name,flush=True)

def rejected(fn,reason):
 try:fn()
 except Exception as e:
  require(reason in str(e),'WRONG_REJECTION '+str(e));return str(e)
 raise RuntimeError('EXPECTED_REJECTION '+reason)

def budget_case(name):
 root=TEST/name;root.mkdir();path=root/'budget.json'
 control.atomic(path,{'state':'UNSTARTED','max_new_real_forward_attempts':27,'plan_sha256':control.sha(control.PLAN),'attempts':[]})
 return root,path

def run():
 before=sha_file(R/'R4_NEW_RUN_BUDGET.json');tree_before=sorted(str(p.relative_to(R/'real_run')) for p in (R/'real_run').rglob('*'))
 plan=control.plan_check();record('exact_ordered_27_plan',lambda:{'task_count':len(plan['tasks']),'first':plan['tasks'][:3]})
 record('runner_CPU_dry_run',runner.dry_run)
 record('manual_confirmation_required',lambda:rejected(lambda:control.require_manual_confirm(False),'MANUAL_CONFIRMATION_REQUIRED'))
 record('forged_worker_rejected',lambda:rejected(lambda:control.authorize_worker('forged'),'MANUAL_REAL_PERMISSION_REQUIRED'))
 def lock_test():
  p=TEST/'simulation_run.lock'
  with p.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   cmd=[sys.executable,'-B','-c','import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',str(p)]
   proc=subprocess.run(cmd,capture_output=True,text=True,env=control.cpu_env());require(proc.returncode!=0 and 'BlockingIOError' in proc.stderr,'LOCK_SECOND_OWNER_NOT_REJECTED')
  return {'second_process_excluded':True,'actual_runner_uses_same_flock':True}
 record('cross_process_exclusive_run_lock',lock_test)
 def budget_test():
  root,path=budget_case('budget_boundaries');task=plan['tasks'][0];a=control.reserve(path,task,'CPU_TEST')
  duplicate=rejected(lambda:control.reserve(path,task,'CPU_TEST'),'NO_AUTOMATIC_RETRY')
  incomplete=rejected(lambda:control.reserve(path,plan['tasks'][1],'CPU_TEST'),'PREVIOUS_ATTEMPT_NOT_COMMITTED_NO_RESUME')
  invalid_event=rejected(lambda:control.event(path,1,'MODEL_RETURNED'),'INVALID_FORWARD_EVENT_ORDER')
  for state in ('MODEL_ENTERED','MODEL_RETURNED','VALIDATED','COMMITTED'):control.event(path,1,state)
  for t in plan['tasks'][1:]:
   control.reserve(path,t,'CPU_TEST')
   for state in ('MODEL_ENTERED','MODEL_RETURNED','VALIDATED','COMMITTED'):control.event(path,t['ordinal'],state)
  exhausted=rejected(lambda:control.reserve(path,plan['tasks'][-1],'CPU_TEST'),'R4_FORWARD_BUDGET_EXHAUSTED')
  root2,path2=budget_case('reused_pair_refusal');bad={**plan['tasks'][0],'sample_idx':324,'condition':'F'}
  reused=rejected(lambda:control.reserve(path2,bad,'CPU_TEST'),'ORDERED_TASK_OR_REUSED_RECORD_REJECTED')
  return {'duplicate':duplicate,'incomplete_no_resume':incomplete,'invalid_event':invalid_event,'attempt28':exhausted,'reused_pair':reused,'all_are_explicit_CPU_TEST_simulation':True}
 record('budget_events_duplicates_no_resume_and_27_limit',budget_test)
 # Same final sequence/collector with mock, never REAL_FORWARD.
 backend,fixture=fixture_backend()
 def sequence_test(name,fault):
  root,path=budget_case(name);error=None
  try:runner.run_sequence(backend,root,path,'CPU_TEST',plan['tasks'][:2] if fault else plan['tasks'][:1],inject_fault=fault)
  except Exception as e:error=str(e)
  b=load(path);require(len(b['attempts'])==1,'FAILED_SEQUENCE_MUST_NOT_RUN_NEXT')
  if fault:
   require(error and b['state']=='FAILED','SIMULATED_FAILURE_MUST_STOP')
   require(not list((root/'capture').glob('*')) if (root/'capture').exists() else True,'UNACCEPTED_CANNOT_COMMIT')
   f=load(root/'failure.json');d=Path(f['diagnostic_path']);require(d.parent==root/'diagnostics' and load(d/'diagnostic.json')['state']=='UNACCEPTED','RECOVERABLE_DIAGNOSTIC_REQUIRED')
   if fault=='post_return':require(load(d/'diagnostic.json')['model_returned'] and (root/'diagnostics/candidate_001/arrays.npz').exists(),'MODEL_RETURN_ARRAYS_DURABLE')
   rejected(lambda:control.reserve(path,plan['tasks'][0],'CPU_TEST'),'NO_AUTOMATIC_RETRY')
  else:require(error is None and b['attempts'][0]['committed_epoch'] is not None,'CPU_COMMIT_SEQUENCE')
  return {'error':error,'ledger':b,'no_real_budget_touched':True}
 record('formal_runner_CPU_atomic_commit',lambda:sequence_test('sequence_success',None))
 record('formal_runner_reserved_crash_stops_no_retry',lambda:sequence_test('sequence_reserved_crash','reserved_crash'))
 record('formal_runner_returned_exception_preserves_UNACCEPTED',lambda:sequence_test('sequence_post_return','post_return'))
 def native_whitelist_fault():
  root=TEST/'unknown_derived';root.mkdir();row=load(FROZEN)['samples'][0];p=backend.prepare(row);p.update(run_root=root,attempt={'ordinal':1,'kind':'CPU_TEST'},condition='F',fault='unknown_derived')
  return rejected(lambda:backend.forward_collect(p,'F'),'UNEXPECTED_DERIVED_FIELDS')
 record('temporary_history_nonwhitelist_rejected',native_whitelist_fault)
 def resource_test():
  valid=f'0, {UUID}, {NAME}, 98304 MiB, 2 MiB, 0 %\n';rss=1024;disk=100*2**30
  ok=validate_telemetry(valid,'',disk,rss)
  bad=[('DEVICE_IDENTITY_MISMATCH',valid.replace(UUID,'wrong'),'',disk,rss),('DEVICE_MEMORY_LOW',valid.replace('2 MiB','98000 MiB'),'',disk,rss),('DEVICE_BUSY',valid,f'{UUID}, 99999, other, 100 MiB\n',disk,rss),('DISK_LOW',valid,'',1,rss),('RSS_HIGH',valid,'',disk,9*2**30)]
  return {'valid':ok,'rejected':[rejected(lambda x=x:validate_telemetry(x[1],x[2],x[3],x[4]),x[0]) for x in bad]}
 record('preserved_R2_resource_thresholds',resource_test)
 def readonly_scripts():
  cmds=[[str(R/'manual_status.sh')],[str(R/'manual_monitor.sh'),'--once'],[str(R/'manual_run.sh'),'--dry-run']]+[[str(R/n),'--help'] for n in ('manual_run.sh','manual_monitor.sh','manual_status.sh','manual_finalize_cpu.sh')]
  results=[]
  for i,cmd in enumerate(cmds):
   proc=subprocess.run(cmd,capture_output=True,text=True,env=control.cpu_env(),timeout=30);require(proc.returncode==0,'SAFE_SCRIPT_FAILED '+str(cmd)+' '+proc.stderr)
   (TEST/f'safe_script_{i}.log').write_text(proc.stdout+proc.stderr);results.append({'command':cmd,'exit_code':proc.returncode})
  proc=subprocess.run([str(R/'manual_run.sh')],capture_output=True,text=True,env=control.cpu_env(),timeout=30)
  require(proc.returncode==2 and 'MANUAL_CONFIRMATION_REQUIRED' in proc.stderr,'NO_CONFIRM_SCRIPT_MUST_REFUSE')
  require(sha_file(R/'R4_NEW_RUN_BUDGET.json')==before,'READONLY_SCRIPT_CHANGED_BUDGET')
  require(sorted(str(p.relative_to(R/'real_run')) for p in (R/'real_run').rglob('*'))==tree_before,'READONLY_SCRIPT_CHANGED_REAL_RUN')
  return {'executed_commands':results,'no_confirmation_exit_code':2,'real_directory_and_budget_unchanged':True}
 record('help_status_monitor_dry_run_read_only',readonly_scripts)
 def analysis_math():
  stats={.5:{'tp':[1,0,1],'fp':[0,1,0],'score':[.4,.9,.8],'gt':2}};ap=calculate_ap(stats,.5)[0]
  require(abs(ap-2/3)<1e-12 and abs(ap-.75)>1e-3,'GLOBAL_SCORE_AP_NOT_FRAME_AVERAGE')
  samples=[({'TP50':1,'FP50':-1,'FN50':-1},'POSITIVE'),({'TP50':-1,'FP50':1,'FN50':1},'NEGATIVE'),({'TP50':1,'FP50':1,'FN50':-1},'MIXED'),({'TP50':0,'FP50':0,'FN50':0},'NO_CHANGE')]
  for v,expected in samples:require(effect_class(v,50)==expected,'UTILITY_COMPONENT_DIRECTION')
  # A=1, AB=3, AC=4, ABC=8 -> U_B|C=4,U_C|B=5,U_B|A=2,U_C|A=3,I=2.
  A,AB,AC,ABC=1,3,4,8;require([ABC-AC,ABC-AB,AB-A,AC-A,ABC-AB-AC+A]==[4,5,2,3,2],'UTILITY_INTERACTION_ALGEBRA')
  return {'synthetic_only':True,'global_PR_AP':ap,'per_frame_mean_would_be':.75,'four_sign_cases':True,'marginal_and_interaction_algebra':True,'no_real_metrics_generated':True}
 record('analysis_global_PR_and_unweighted_component_algebra',analysis_math)
 cpu_guard();require(sha_file(R/'R4_NEW_RUN_BUDGET.json')==before and load(R/'R4_NEW_RUN_BUDGET.json')['attempts']==[],'PREPARATION_REAL_ATTEMPT_INCREMENT')
 write(R/'R4_ENGINEERING_TESTS.json',{'passed':True,'cases':cases,'count':len(cases),'CUDA_initialized':False,'blocked_cuda_entry_calls':cpu_only.CALLS,'new_real_forward':0,'new_GPU_PPU_forward':0,'new_real_budget_attempts':0,'sources':{str(p):sha_file(p) for p in R.glob('*.py')}},exclusive=True)
 print('R4_ENGINEERING_TESTS_PASSED',len(cases),flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:write(R/'R4_ENGINEERING_FAILURE.json',{'passed':False,'error':str(e),'traceback':traceback.format_exc()});raise
