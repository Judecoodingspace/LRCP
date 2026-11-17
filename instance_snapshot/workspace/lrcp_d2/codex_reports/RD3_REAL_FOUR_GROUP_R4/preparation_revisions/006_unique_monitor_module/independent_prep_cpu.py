"""Final independent CPU prep acceptance. Reopens original science and inputs;
replays tests, subprocess CLI guards and actual final native backend CPU binding.
"""
import cpu_only
cpu_only.install()
from rd3_common import *
import torch
import control,subprocess,traceback,gc,copy
from validate_cpu import validate_candidate
from dataset_cpu import revalidate_five
from per_sample_schema import frozen_contract,verify_schema_before
from cpu_pipeline import fixture_backend
from input_integrity import snapshot
from telemetry import observe


def run():
 seal=control.verify_seal();protected=load(R/'R4_HISTORY_READ_ONLY_SEAL.json')
 for p,h in protected['files'].items():require(sha_file(p)==h,'HISTORY_CHANGED '+p)
 require(subprocess.check_output(['git','status','--porcelain'],cwd=SRC,text=True)==protected['git_status'],'MODEL_GIT_STATUS_CHANGED')
 plan=control.plan_check();budget=load(control.BUDGET);require(budget['state']=='UNSTARTED' and budget['max_new_real_forward_attempts']==27 and not budget['attempts'],'NEW_BUDGET_MUST_REMAIN_UNSTARTED_ZERO')
 require(not (control.REAL_ROOT/'claim.json').exists() and not (control.REAL_ROOT/'manifest.json').exists() and not list((control.REAL_ROOT/'capture').iterdir()) and not list((control.REAL_ROOT/'diagnostics').iterdir()),'NO_REAL_ACTIVITY_IN_PREPARATION')
 pre=load(R/'R4_ALL_8_CPU_PREFLIGHT.json');require(len(pre['samples'])==8,'ALL8_COVERAGE')
 for p,h in pre['source_hashes'].items():require(sha_file(p)==h,'FINAL_CPU_SOURCE_CHANGED '+p)
 fresh=[]
 for row,rec in zip(load(FROZEN)['samples'],pre['samples']):
  payload,contract=frozen_contract(row);require(rec['sample_idx']==row['sample_idx'] and rec['contract_hash']==contract['hash'],'ACTUAL_FROZEN_CONTRACT')
  result=validate_candidate(Path(rec['candidate_path']).parent.parent,rec['candidate_path'],True)
  require(result['input_contract_hash']==contract['hash'] and result['input_leaves']==len(contract['leaves']),'ACTUAL_ALL8_SCHEMA')
  fresh.append(result);print('R4_INDEPENDENT_ALL8',row['sample_idx'],flush=True);del payload;gc.collect()
 masks=[]
 for rec in pre['additional_mask_conditions']:masks.append(validate_candidate(Path(rec['candidate_path']).parent.parent,rec['candidate_path'],True))
 full=load(R/'R4_FULL_UNHOOKED_CONTROL.json')
 for name in ('hooked_candidate','unhooked_candidate'):validate_candidate(Path(full[name]).parent.parent,full[name],True)
 with np.load(Path(full['hooked_candidate'])/'arrays.npz',allow_pickle=False) as a,np.load(Path(full['unhooked_candidate'])/'arrays.npz',allow_pickle=False) as b:
  require(set(a.files)==set(b.files),'FULL_SET')
  for key in a.files:require(np.array_equal(a[key],b[key]),'FULL_WITH_WITHOUT_MASK '+key)
 # Fresh actual original-native historical preparation for every frozen sample.
 backend,fixture=fixture_backend();native=[]
 for row,rec in zip(load(FROZEN)['samples'],pre['samples']):
  p=backend.prepare(row)
  from input_integrity import DisposableHistory,check
  from evidence_collector import snapshot as snap
  before=snap(p['batch'])
  with DisposableHistory(backend.model,p['batch']) as history:
   for slot in (1,2):
    e=p['batch']['ego'];v=e['calibrate_data'][slot]
    features=backend.model.get_batch_pillar_features(v['lidar_history'],e['pairwise_t_matrix'],slot,[0]);backend.model.meta_flow(features,v['time_delay'])
  check(before,p['batch']);receipts=history.validate()
  original=load(Path(rec['candidate_path'])/'meta.json')['history_integrity']
  require(receipts==original,'INDEPENDENT_NATIVE_HISTORY_REPLAY')
  native.append({'sample_idx':row['sample_idx'],'two_native_histories_unchanged':True});print('R4_INDEPENDENT_HISTORY',row['sample_idx'],flush=True)
  del p,features,history;gc.collect()
 faults=load(R/'FAULT_INJECTION_RESULTS.json');replayed=[]
 for case in faults['executed_cases']:
  name=case['case_id'];expected=case['expected_error'];error=None
  if 'injected_input_path' in case:
   bad=torch.load(case['injected_input_path'],map_location='cpu',weights_only=False);row=load(FROZEN)['samples'][1]
   require(sha_file(case['injected_input_path'])==case['injected_input_sha256'],'INJECTED_CASE_HASH')
   try:verify_schema_before(row,bad,'cpu',snapshot(bad))
   except Exception as e:error=str(e)
   require(error and 'OTHER_INPUT_MUTATION' in error,'INDEPENDENT_SCHEMA_FAULT_REJECTION')
  elif name in ('B_C_history_exchange','B_C_offset_exchange'):
   root=R/'cpu_tests'/('independent_'+name);root.mkdir();row=load(FROZEN)['samples'][0];p=backend.prepare(row);backend.model.fault='swap_histories' if name=='B_C_history_exchange' else 'swap_offsets';p.update(run_root=root,attempt={'ordinal':1,'kind':'CPU_TEST'},condition='F')
   try:backend.forward_collect(p,'F')
   except Exception as e:error=str(e)
   backend.model.fault=None;require(error and expected in error,'INDEPENDENT_MAPPING_REJECTION')
  else:
   candidate=Path(case['candidate_path'])
   try:validate_candidate(candidate.parent.parent,candidate,True)
   except Exception as e:error=str(e)
   require(error and ('ORIGINAL_SCALAR' in error or 'ORIGINAL_TENSOR_VALUES_METADATA' in error),'INDEPENDENT_FORGED_RECORD_REJECTION')
  replayed.append({'case_id':name,'actual_error':error,'correctly_rejected':True});print('R4_INDEPENDENT_FAULT',name,flush=True)
 require(len(replayed)==20,'ALL_20_FAULTS_REQUIRED')
 five=revalidate_five();require(len(five['records'])==5,'FIVE_REAL_INDEPENDENTLY_ACCEPTED')
 index=load(R/'reused_records/index.json');require(index['records']==[{k:v for k,v in x.items() if k!='validation'} for x in five['records']],'FIVE_INDEX_BYTES_AND_IDENTITY')
 require({(x['sample_idx'],x['condition']) for x in five['records']}=={(x['sample_idx'],x['condition']) for x in plan['reused_pairs']},'ACTUAL_FIVE_PLAN_SET')
 # Repeat the actual real backend binding audit in its own hard-guarded CPU process.
 out=R/'cpu_tests/independent_native_binding.json'
 with (R/'logs/independent_native_binding.log').open('x') as log:proc=subprocess.run([control.PYTHON,'-B',str(R/'native_entry_cpu_audit.py'),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,env=control.cpu_env(),timeout=180)
 require(proc.returncode==0,'INDEPENDENT_NATIVE_BINDING_FAILED');binding=load(out);require(binding['actual_native_model_constructed_on_CPU'] and not binding['native_forward_calls'] and not binding['CUDA_entry_calls'],'ACTUAL_REAL_BACKEND_BINDING')
 # CLI routing/import checks in fresh stdlib-only processes, never REAL_FORWARD.
 cli=[];before_budget=sha_file(control.BUDGET)
 for cmd,expected in [([str(R/'manual_status.sh')],0),([str(R/'manual_monitor.sh'),'--once'],0),([str(R/'manual_run.sh'),'--dry-run'],0),([str(R/'manual_run.sh')],2)]:
  proc=subprocess.run(cmd,capture_output=True,text=True,env=control.cpu_env(),timeout=30);require(proc.returncode==expected,'INDEPENDENT_CLI_GATE '+str(cmd));cli.append({'command':cmd,'actual_exit':proc.returncode})
 for script in R.glob('manual_*.sh'):require(subprocess.run(['bash','-n',str(script)],capture_output=True).returncode==0,'BASH_SYNTAX')
 cmd=[control.PYTHON,'-B','-c','import sys; import runner,status,monitor; assert "torch" not in sys.modules; assert "real_backend" not in sys.modules; print("NO_MODEL_IMPORT")']
 proc=subprocess.run(cmd,capture_output=True,text=True,cwd=R,env=control.cpu_env(),timeout=30);require(proc.returncode==0,'SAFE_IMPORTS '+proc.stderr)
 require(sha_file(control.BUDGET)==before_budget,'READONLY_CHANGED_BUDGET')
 engineering=load(R/'R4_ENGINEERING_TESTS.json')
 for path,h in engineering['sources'].items():require(sha_file(path)==h,'ENGINEERING_SOURCE_CHANGED')
 require(engineering['count']>=12 and all(c['passed'] for c in engineering['cases']),'ENGINEERING_ACTUAL_CASES')
 # Independently replay ledger transitions/exhaustion on disposable CPU_TEST ledger.
 root=R/'cpu_tests/independent_budget';root.mkdir();path=root/'budget.json';control.atomic(path,{'state':'UNSTARTED','max_new_real_forward_attempts':27,'plan_sha256':sha_file(control.PLAN),'attempts':[]})
 for task in plan['tasks']:
  control.reserve(path,task,'CPU_TEST')
  for state in ('MODEL_ENTERED','MODEL_RETURNED','VALIDATED','COMMITTED'):control.event(path,task['ordinal'],state)
 try:control.reserve(path,plan['tasks'][-1],'CPU_TEST');raise AssertionError('BUDGET_LIMIT_NOT_REJECTED')
 except RuntimeError as e:require('R4_FORWARD_BUDGET_EXHAUSTED' in str(e),'BUDGET_LIMIT_ERROR')
 # Original historical budget values, and all protected bytes unchanged at END.
 old=load(OLD/'RUN_BUDGET.json');r2=load(R2/'R2_RUN_BUDGET.json')
 def counts(b):return {k:sum(a.get(k+'_epoch') is not None for a in b['attempts']) for k in ('entered','returned','committed')}
 old_manifest=load(OLD/'REAL_RUN_MANIFEST.json');old_counts={'entered':old_manifest['forward_entered_count'],'returned':old_manifest['forward_returned_count'],'committed':old_manifest['committed_conditions']}
 require(len(old['attempts'])==1 and old['attempts'][0]['status']=='MODEL_RETURNED' and old_counts=={'entered':1,'returned':1,'committed':0} and counts(r2)=={'entered':5,'returned':5,'committed':4},'HISTORICAL_BUDGET_PRESERVATION')
 for p,h in protected['files'].items():require(sha_file(p)==h,'HISTORY_END_CHANGED '+p)
 require(load(control.BUDGET)==budget,'NEW_BUDGET_MUTATED_DURING_PREP')
 require(not torch.cuda.is_initialized() and not cpu_only.CALLS,'CPU_DEVICE_GUARD')
 obs=observe();require(obs['display_state']=='WAITING_FOR_MANUAL_START','MONITOR_PRESTART_STATE')
 evidence_names=['R4_REUSED_FIVE_RECORDS_VALIDATION.json','R4_ALL_8_CPU_PREFLIGHT.json','R4_FULL_UNHOOKED_CONTROL.json','R4_ENGINEERING_TESTS.json','FAULT_INJECTION_RESULTS.json','R4_NATIVE_ENTRY_CPU_AUDIT.json','R4_MONITOR_VALIDATION.json','R4_REAL_ENTRY_AUDIT.md','R4_MANUAL_COMMANDS.md','reused_records/index.json']
 evidence={str(R/n):sha_file(R/n) for n in evidence_names}
 result={'passed':True,'state':'RD3_R4_READY_FOR_MANUAL_RUN','source_seal_sha256':sha_file(control.SEAL),'sealed_evidence':evidence,'all8_fresh_independent_validations':fresh,'mask_condition_validations':masks,'native_histories_independently_reexecuted':native,'all20_faults_independently_replayed':replayed,'five_saved_real_records_freshly_revalidated':five,'actual_native_backend_binding_fresh_process':binding,'safe_CLI_actual_executions':cli,'historical_hashes_unchanged':len(protected['files']),'historical_RD3_counts':old_counts,'historical_R2_counts':counts(r2),'new_budget_state':'UNSTARTED','new_budget_attempts':0,'new_real_forward':0,'GPU_PPU_forward_increment':0,'CUDA_initialized':False,'blocked_CUDA_entry_calls':[],'execution_plan_count':27,'manual_confirmation_and_persistent_no_resume':True,'resources_internal_guard_retained':control.LIMITS,'monitor_prestart_device':obs['device'],'formal_science_estimated':False,'forward_execution_is_user_manual_only':True}
 write(R/'R4_PREP_INDEPENDENT_VALIDATION.json',result,exclusive=True);print('RD3_R4_READY_FOR_MANUAL_RUN',flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:write(R/'R4_PREP_INDEPENDENT_VALIDATION.json',{'passed':False,'state':'RD3_R4_PREPARATION_FAILED','error':str(e),'traceback':traceback.format_exc(),'new_real_forward':0,'CUDA_initialized':torch.cuda.is_initialized()},exclusive=True);raise
