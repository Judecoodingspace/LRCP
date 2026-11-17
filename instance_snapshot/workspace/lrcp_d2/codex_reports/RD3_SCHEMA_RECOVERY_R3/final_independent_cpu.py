"""Final separate CPU process: recomputes positive and negative acceptance,
native-history fingerprints, all five real records and immutable budgets.
No producer passed field is sufficient for readiness or scientific recovery.
"""
from rd3_common import *
from validate_cpu import validate_candidate
from per_sample_schema import frozen_contract
from input_integrity import snapshot
from cpu_pipeline import fixture_backend
from INDEPENDENT_VALIDATOR import same
import torch,copy,subprocess,gc,traceback,importlib.util

def run():
 all8=load(R/'ALL_8_CPU_PREFLIGHT.json');faults=load(R/'FAULT_INJECTION_RESULTS.json');recovery=load(R/'CANDIDATE_005_RECOVERY_VALIDATION.json');reuse=load(R/'ACCEPTED_324_REUSE_VALIDATION.json')
 expected_rows=load(FROZEN)['samples'];require([r['sample_idx'] for r in all8['samples']]==[r['sample_idx'] for r in expected_rows],'EXACT_ALL8_TESTED')
 checks=[];source_hashes={str(p):sha_file(p) for p in sorted(R.glob('*.py'))}
 for record,row in zip(all8['samples'],expected_rows):
  require(record['producer_source_sha256']==sha_file(R/'evidence_collector.py') and record['validator_source_sha256']==sha_file(R/'validate_cpu.py'),'FINAL_CODE_CHANGED_AFTER_TESTS')
  path=Path(record['candidate_path']);meta=load(path/'meta.json')
  for p,h in meta['r3_source_hashes'].items():require(sha_file(p)==h,'EXECUTED_CORE_SOURCE_CHANGED')
  payload,contract=frozen_contract(row);require(contract['hash']==record['contract_hash'],'ORIGINAL_SAMPLE_CONTRACT')
  require(record['before']['original_leaf_count']==record['after']['original_leaf_count']==len(contract['leaves']),'PER_SAMPLE_COUNT_ONLY_DERIVED')
  require(sha_file(path/'manifest.json')==record['candidate_manifest_sha256'] and sha_file(record['validation_path'])==record['validation_sha256'],'PREFLIGHT_EVIDENCE_HASH')
  fresh=validate_candidate(R/'cpu_preflight',path,cpu_test=True);require(fresh['input_contract_hash']==contract['hash'],'INDEPENDENT_ACTUAL_SAMPLE_SCHEMA')
  checks.append({'sample_idx':row['sample_idx'],'actual_leaf_count':len(contract['leaves']),'GT_list_length':len(payload['batch']['ego']['object_ids']),'independent_fresh_validation':fresh});print('FINAL_INDEPENDENT_SAMPLE',row['sample_idx'],flush=True)
  del payload;gc.collect()
 for extra in all8['additional_mask_conditions']:require(validate_candidate(R/'cpu_preflight',extra['candidate_path'],cpu_test=True)['passed'],'ADDITIONAL_MASK_REGRESSION')
 # Independent native-history computation on all original frozen inputs. No
 # full LRCP or neural response/head forward is invoked by this section.
 spec=importlib.util.spec_from_file_location('r3_independent_native',REPAIR/'mock_backend.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 state=native.checkpoint();owner=native.HistoryOwner(state,fusion=False);native_checks=[]
 for record,row in zip(all8['samples'],expected_rows):
  payload,_=frozen_contract(row);batch=payload['batch'];meta=load(Path(record['candidate_path'])/'meta.json');features=[]
  for slot in (1,2):
   h=batch['ego']['calibrate_data'][slot]['lidar_history'][0];before=snapshot(h);temp=dict(h)
   with torch.no_grad():out=owner.get_batch_pillar_features([temp],batch['ego']['pairwise_t_matrix'],slot,[0])
   require(snapshot(h)==before,'INDEPENDENT_NATIVE_HISTORY_SIDE_EFFECT')
   receipt=meta['history_integrity']['receipts'][slot-1];require(digest(out)==receipt['feature_hash'],'ACTUAL_NATIVE_HISTORY_FEATURE_VALUES')
   require(list(temp)==list(h)+['pillar_features','spatial_features','pairwise_t_matrix'],'INDEPENDENT_NATIVE_DERIVED_SCOPE');features.append(digest(out))
  native_checks.append({'sample_idx':row['sample_idx'],'original_history_preserved':True,'native_feature_hashes_match':features});del payload,batch,temp,out;gc.collect()
 # Re-execute the real fixed Collector against saved injected values. Self
 # snapshots are again forged so this tests the normative frozen comparison.
 backend,_=fixture_backend();fault_checks=[];case_root=R/'independent_fault_replay';case_root.mkdir()
 for case in faults['executed_cases']:
  for p,h in case['source_hashes'].items():require(sha_file(p)==h,'FAULT_TEST_NOT_FINAL_SOURCE')
  name=case['case_id'];error=None
  if 'injected_input_path' in case:
   require(sha_file(case['injected_input_path'])==case['injected_input_sha256'],'INJECTED_INPUT_HASH')
   row=next(r for r in expected_rows if r['sample_idx']==424);p=backend.prepare(row);p['batch']=torch.load(case['injected_input_path'],map_location='cpu',weights_only=False);p['leaf_before']=snapshot(p['batch']);root=case_root/name;root.mkdir()
   p.update(run_root=root,attempt={'ordinal':1,'kind':'CPU_TEST','sample_idx':424,'condition':'F'},condition='F')
   try:backend.forward_collect(p,'F')
   except Exception as e:error=str(e)
   require(error is not None and 'OTHER_INPUT_MUTATION' in error,'INDEPENDENT_PRODUCER_FAULT_ACCEPTED '+name)
  elif name in ('B_C_history_exchange','B_C_offset_exchange'):
   row=expected_rows[0];p=backend.prepare(row);root=case_root/name;root.mkdir();p.update(run_root=root,attempt={'ordinal':1,'kind':'CPU_TEST','sample_idx':row['sample_idx'],'condition':'F'},condition='F');backend.model.fault='swap_histories' if name=='B_C_history_exchange' else 'swap_offsets'
   try:backend.forward_collect(p,'F')
   except Exception as e:error=str(e)
   finally:backend.model.fault=None
   require(error is not None and case['expected_error'] in error,'INDEPENDENT_IDENTITY_FAULT_ACCEPTED')
  else:
   path=Path(case['candidate_path']);require(sha_file(path/'manifest.json')==case['candidate_manifest_sha256'],'FORGED_CASE_HASH')
   # File hashes are all valid; independently rejecting the actual content is
   # required even though producer passed=true exists in both envelopes.
   require(load(path/'meta.json')['passed'] is True,'FORGED_TRUE_CASE')
   try:validate_candidate(path.parent.parent,path,cpu_test=True)
   except Exception as e:error=str(e)
   require(error is not None and ('ORIGINAL_SCALAR' in error or 'ORIGINAL_TENSOR_VALUES_METADATA' in error),'REHASHED_FORGERY_ACCEPTED')
  fault_checks.append({'case_id':name,'actual_error':error,'independently_rejected':True});print('FINAL_INDEPENDENT_FAULT',name,flush=True);gc.collect()
 # Scientific recovery is independently re-executed, not inferred from a report.
 source=R2/'real_run/diagnostics/candidate_005';actual_recovery=None;recovery_error=None
 try:actual_recovery=validate_candidate(R2/'real_run',source)
 except Exception as e:recovery_error=str(e)
 recoverable=actual_recovery is not None and actual_recovery['passed']
 require(recoverable==recovery['CANDIDATE_005_RECOVERABLE'],'RECOVERY_RESULT_NOT_REPRODUCIBLE')
 available=set();real_reuse=[];r2=load(R2/'real_run/manifest.json')
 for rec in r2['records']:
  path=R2/'real_run'/rec['relative_path'];m=load(path/'manifest.json');require(sha_file(path/'manifest.json')==rec['manifest_sha256'],'REUSE_MANIFEST')
  for name,h in m['files'].items():require(sha_file(path/name)==h,'REUSE_PAYLOAD')
  result=validate_candidate(R2/'real_run',R2/'real_run'/m['candidate']);available.add((rec['sample_idx'],rec['condition']));real_reuse.append(result)
 reusable_324={(x['sample_idx'],x['condition']) for x in real_reuse}=={(324,c) for c in CONDITIONS};require(reusable_324==reuse['ACCEPTED_324_REUSABLE'],'REUSE_RESULT_NOT_REPRODUCIBLE')
 if recoverable:
  available.add((424,'F'));target=R/'recovered_candidates/sample_424_F';manifest=load(target/'RECOVERY_MANIFEST.json')
  require(manifest['source_R2_state_still']=='UNACCEPTED' and manifest['source_R2_committed_for_attempt_still']==0,'RECOVERY_REWRITES_SOURCE_STATE')
  for name,h in manifest['files_verified_copied'].items():require(sha_file(target/name)==sha_file(source/name)==h,'RECOVERY_COPY_NOT_EXACT_SOURCE')
  require(sha_file(target/'original_manifest.json')==sha_file(source/'manifest.json'),'ORIGINAL_MANIFEST_COPY')
 old=load(OLD/'RUN_BUDGET.json');oldmanifest=load(OLD/'REAL_RUN_MANIFEST.json');budget=load(R2/'R2_RUN_BUDGET.json');fifth=budget['attempts'][4]
 require(len(old['attempts'])==1 and oldmanifest['forward_entered_count']==oldmanifest['forward_returned_count']==1 and oldmanifest['committed_conditions']==0,'ORIGINAL_RD3_LEDGER')
 require(len(budget['attempts'])==5 and sum(a['entered_epoch'] is not None for a in budget['attempts'])==sum(a['returned_epoch'] is not None for a in budget['attempts'])==5 and sum(a['committed_epoch'] is not None for a in budget['attempts'])==4 and r2['committed']==4 and fifth['status']=='MODEL_RETURNED' and fifth['committed_epoch'] is None,'ORIGINAL_R2_LEDGER')
 require(load(source/'manifest.json')['state']=='UNACCEPTED' and not (R2/'real_run/capture/sample_424_F').exists(),'OLD_CANDIDATE_PROMOTED')
 seal=load(R/'HISTORY_READ_ONLY_SEAL.json')
 for path,h in seal['files'].items():require(sha_file(path)==h,'HISTORICAL_OR_MODEL_FILE_CHANGED '+path)
 require(subprocess.check_output(['git','-C',str(SRC),'status','--short'],text=True)==seal['git_status'],'GIT_STATE_CHANGED')
 expected={(row['sample_idx'],c) for row in expected_rows for c in CONDITIONS};missing=[{'sample_idx':row['sample_idx'],'scene_id':row['scene_id'],'condition':c} for row in expected_rows for c in CONDITIONS if (row['sample_idx'],c) not in available]
 require(len(expected-available)==len(missing),'MISSING_COMBINATIONS')
 write(R/'BUDGET_AND_PROVENANCE_AUDIT.json',{'passed':True,'old_RD3_state':{'entered':1,'returned':1,'committed':0},'R2_state':{'entered':5,'returned':5,'committed':4,'attempt5':'MODEL_RETURNED','candidate005':'UNACCEPTED','candidate005_committed':0},'new_R3_real_forward':0,'new_GPU_PPU_calls':0,'R2_remaining_budget_untouched':budget['max_attempts']-len(budget['attempts']),'CANDIDATE_005_RECOVERABLE':recoverable,'ACCEPTED_324_REUSABLE':reusable_324,'future_usable_source_sample_conditions':[{'sample_idx':s,'condition':c} for s,c in sorted(available)],'remaining_real_forward_count_for_original_8x4':len(missing),'missing_sample_conditions':missing,'future_real_execution_requires_separate_authorization':True,'automatically_launchable_real_job_created':False,'all_history_and_model_hashes_unchanged':len(seal['files']),'CUDA_initialized':torch.cuda.is_initialized()},exclusive=True)
 cpu_guard();require(not cpu_only.CALLS,'CUDA_ENTRY_WAS_ATTEMPTED')
 output={'passed':True,'state':'RD3_R3_CPU_READY','CANDIDATE_005_RECOVERABLE':recoverable,'ACCEPTED_324_REUSABLE':reusable_324,'all8_independent_checks':checks,'independent_native_history_checks':native_checks,'all_negative_cases_independently_replayed':fault_checks,'candidate005_fresh_independent_validation':actual_recovery,'candidate005_error':recovery_error,'accepted324_fresh_independent_validations':real_reuse,'remaining_real_forward_count':len(missing),'new_real_forward':0,'new_GPU_PPU_calls':0,'CUDA_initialized':torch.cuda.is_initialized(),'blocked_cuda_entry_calls':cpu_only.CALLS,'all_history_hashes_verified':len(seal['files']),'R3_actual_source_hashes':source_hashes,'overall_8x4_completed':False,'new_AP_or_message_utility_conclusion':False,'no_future_real_run_authorized':True}
 write(R/'INDEPENDENT_CPU_VALIDATION.json',output,exclusive=True);print('RD3_R3_CPU_READY','CANDIDATE_005_RECOVERABLE='+str(recoverable).lower(),'ACCEPTED_324_REUSABLE='+str(reusable_324).lower(),'remaining='+str(len(missing)),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'INDEPENDENT_CPU_VALIDATION.json',{'passed':False,'state':'RD3_R3_CPU_REJECTED','error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'new_real_forward':0},exclusive=True);raise
