"""CPU-only, independent scientific evidence reuse. No source promotion, no
inference, no AP/utility output, and no changes to R2's UNACCEPTED state/ledger.
"""
from rd3_common import *
from validate_cpu import validate_candidate
from INDEPENDENT_VALIDATOR import same
import torch,traceback

def run():
 preflight=load(R/'ALL_8_CPU_PREFLIGHT.json');require(preflight['passed'] and len(preflight['samples'])==len(load(FROZEN)['samples']),'ALL_8_CPU_GATE_REQUIRED')
 require(load(R/'FAULT_INJECTION_RESULTS.json')['passed'],'FAULT_GATE_REQUIRED')
 for record in preflight['samples']:
  require(record['producer_source_sha256']==sha_file(R/'evidence_collector.py') and record['validator_source_sha256']==sha_file(R/'validate_cpu.py'),'FINAL_CODE_NOT_TESTED_ALL8')
 source_root=R2/'real_run';r2_manifest=load(source_root/'manifest.json');budget=load(R2/'R2_RUN_BUDGET.json');require(len(budget['attempts'])==5 and r2_manifest['committed']==4,'SOURCE_BUDGET_NOT_EXPECTED')
 source=source_root/'diagnostics/candidate_005';raw=load(source/'raw_meta.json');actual=budget['attempts'][4]
 require(actual['ordinal']==5 and actual['sample_idx']==424 and actual['condition']=='F' and actual['status']=='MODEL_RETURNED' and actual['entered_epoch'] and actual['returned_epoch'] and actual['committed_epoch'] is None,'SOURCE_ATTEMPT5_IDENTITY')
 require(load(source/'manifest.json')['state']=='UNACCEPTED','SOURCE_CANDIDATE_STATE')
 result=None;error=None
 try:result=validate_candidate(source_root,source)
 except Exception as e:error=str(e)
 recoverable=result is not None and result['passed'] and result['input_leaves']==len(next(x for x in load(R/'PER_SAMPLE_LEAF_CONTRACT_VALIDATION.json')['samples'] if x['sample_idx']==424)['original_paths'])
 recovery={'CANDIDATE_005_RECOVERABLE':bool(recoverable),'passed':bool(recoverable),'source_path':str(source),'R2_attempt_id':5,'source_state':'UNACCEPTED','R2_attempt_status':'MODEL_RETURNED','R2_committed_for_this_attempt':0,'source_original_manifest_sha256':sha_file(source/'manifest.json'),'source_file_hashes':load(source/'manifest.json')['files'],'independent_validation':result,'error':error,'validator_source_sha256':sha_file(R/'validate_cpu.py'),'schema_change':'global leaf==60 replaced by exact THIS sample frozen path/type/value/container/device contract, derived from archive and preexisting R1 contract','head_or_MetaFlow_outputs_recomputed_or_backfilled':False,'real_forward_increment':0,'new_AP_or_utility_computed':False,'CUDA_initialized':torch.cuda.is_initialized()}
 write(R/'CANDIDATE_005_RECOVERY_VALIDATION.json',recovery,exclusive=True)
 if recoverable:
  target=R/'recovered_candidates/sample_424_F';target.mkdir(parents=True)
  for name,h in recovery['source_file_hashes'].items():
   shutil.copyfile(source/name,target/name);require(sha_file(target/name)==h,'RECOVERY_COPY_HASH');(target/name).chmod(0o444)
  shutil.copyfile(source/'manifest.json',target/'original_manifest.json');(target/'original_manifest.json').chmod(0o444)
  write(target/'RECOVERY_MANIFEST.json',{'state':'INDEPENDENTLY_VERIFIED_RECOVERY_REFERENCE','source_path':str(source),'source_R2_state_still':'UNACCEPTED','source_attempt_id':5,'source_ledger_status_still':'MODEL_RETURNED','source_R2_committed_for_attempt_still':0,'files_verified_copied':recovery['source_file_hashes'],'original_manifest_sha256':recovery['source_original_manifest_sha256'],'R3_validation_path':str(R/'CANDIDATE_005_RECOVERY_VALIDATION.json'),'R3_validation_sha256':sha_file(R/'CANDIDATE_005_RECOVERY_VALIDATION.json'),'validator_source_sha256':recovery['validator_source_sha256'],'usable_only_by_future_explicitly_provenanced_dataset':True,'not_a_completed_8x4_experiment':True,'schema_change':recovery['schema_change']},exclusive=True)
 # Existing ACCEPTED flags are only provenance. Re-run every actual native
 # scientific/leaf/mask/postprocess control using the fixed independent validator.
 reusable=[];base=None
 for rec in r2_manifest['records']:
  path=source_root/rec['relative_path'];m=load(path/'manifest.json');require(sha_file(path/'manifest.json')==rec['manifest_sha256'] and m['state']=='ACCEPTED' and m['backend']=='REAL','ACCEPTED_ORIGINAL_HASH')
  for name,h in m['files'].items():require(sha_file(path/name)==h,'ACCEPTED_FILE_HASH')
  require(sha_file(m['validation_path'])==m['validation_sha256'],'ORIGINAL_CONTROL_CERT_HASH')
  candidate=source_root/m['candidate'];require(sha_file(candidate/'manifest.json')==m['candidate_manifest_sha256'],'SOURCE_CANDIDATE_ORIGINAL_HASH')
  check=validate_candidate(source_root,candidate)
  meta=load(path/'meta.json')
  with np.load(path/'arrays.npz',allow_pickle=False) as z:stable={k:z[k].copy() for k in ('reference_full','metaflow_slot_1','metaflow_slot_2','pred_offset','offset_GT','GT_canonical')}
  if base is None:base=(meta,stable)
  else:
   require(meta['feature_hashes']==base[0]['feature_hashes'],'ORIGINAL_AGENT_FEATURES_CHANGED')
   for key in stable:same(stable[key],base[1][key],'REUSE_CROSS_CONDITION_'+key)
  reusable.append({'sample_idx':rec['sample_idx'],'condition':rec['condition'],'source_path':str(path),'source_manifest_sha256':rec['manifest_sha256'],'source_file_hashes':m['files'],'independent_validation':check})
  print('REAL_SOURCE_REVALIDATED',rec['sample_idx'],rec['condition'],flush=True)
 all_reusable=len(reusable)==4 and {(x['sample_idx'],x['condition']) for x in reusable}=={(324,c) for c in CONDITIONS}
 write(R/'ACCEPTED_324_REUSE_VALIDATION.json',{'ACCEPTED_324_REUSABLE':all_reusable,'passed':all_reusable,'records':reusable,'existing_acceptance_not_used_as_scientific_proof':True,'all_original_model_math_and_threshold_sources_unchanged':True,'validator_source_sha256':sha_file(R/'validate_cpu.py'),'real_forward_increment':0,'CUDA_initialized':torch.cuda.is_initialized()},exclusive=True)
 print('CANDIDATE_005_RECOVERABLE='+str(bool(recoverable)).lower(),'ACCEPTED_324_REUSABLE='+str(all_reusable).lower(),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'RECOVERY_AUDIT_FAILURE.json',{'state':'RD3_R3_CPU_REJECTED','error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
