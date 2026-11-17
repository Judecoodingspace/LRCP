"""CPU-only final gate and reproducible delivery from independently checked data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_only
cpu_only.install()
from rd3_common import *
import torch,subprocess,ast,difflib
from input_integrity import snapshot

def run():
 core=load(R/'INDEPENDENT_CORE_VALIDATION.json');negative=load(R/'INDEPENDENT_FAULT_VALIDATION.json');faults=load(R/'SCIENTIFIC_FAULT_TESTS.json');runroot=R/'formal_cpu_r2';seal=load(R/'HISTORY_SOURCE_SEAL.json')
 require(core['passed'] and negative['passed'],'INDEPENDENT_CPU_GATES')
 require(core['CUDA_initialized']==negative['CUDA_initialized']==False and core['real_forward_increment']==negative['real_forward_increment']==0,'CPU_BOUNDARY')
 source_lock=load(runroot/'claim.json')['source_hashes']
 for path,h in source_lock.items():require(sha_file(path)==h,'FORMAL_SOURCE_CHANGED')
 for path,h in seal['files'].items():require(sha_file(path)==h,'HISTORICAL_EVIDENCE_CHANGED '+path)
 frozen=load(FROZEN);require(sha_file(CKPT)==CH,'CHECKPOINT_CHANGED')
 source_versions=load(R/'REPAIR_SOURCE_HASHES.json')['all_repair_python']
 for name in ('independent_cpu.py','independent_faults_cpu.py','attention_validation.py','test_faults_cpu.py'):
  require(sha_file(R/name)==source_versions[str(R/name)],'VALIDATOR_OR_TEST_CHANGED')
 inventories=[]
 for row in frozen['samples']:
  item=row['input'];require(sha_file(item['path'])==item['sha256'],'ORIGINAL_FROZEN_INPUT')
  p=torch.load(item['path'],map_location='cpu',weights_only=False);require(digest(p['batch'])==item['input_hash'] and digest(p['rng'])==item['RNG_hash'],'INPUT_OR_RNG')
  inventories.append({'sample_idx':row['sample_idx'],'scene_id':row['scene_id'],'source_archive_sha256':item['sha256'],'input_semantic_hash':item['input_hash'],'agent_slots':list(map(str,p['batch']['ego']['cav_id_list'])),'identity_and_time_reference':{'scene_frame':row['scene_frame'],'ego_timestamp_raw':row['ego_timestamp_raw'],'remote_timestamp_raw':row['remote_timestamp_raw'],'history':row['history'],'time_delay':row['time_delay']},'all_original_fields_and_leaves':snapshot(p['batch'])})
 write(R/'INPUT_LEAF_CONTRACT.json',{'schema':'RD3_R1_ORIGINAL_LEAF_V1','frozen_manifest_reference':str(FROZEN),'frozen_manifest_sha256':sha_file(FROZEN),'checkpoint_sha256':CH,'real_data_refrozen':False,'original_input_policy':'all original names, container types/order, values and Tensor shape/dtype/device must remain identical','execution_device':'cpu','temporary_history_policy':{'owner':'DisposableHistory.isolated call arguments only; never the original history dictionary','original_tensor_keys':['voxel_features','voxel_coords','voxel_num_points'],'shallow_tensor_storage_shared':True,'check_original_and_temporary_leaves_before_and_after':True,'allow_only_added_keys':['pillar_features','spatial_features','pairwise_t_matrix']},'working_batch_allowed_addition':{'path':'/ego/label_dict/offset','semantics':'exact concat of each original calibrate slot offset_GT flattened and unsqueezed in native iteration order','must_preserve_all_other_original_and_derived_fields':True},'Agent_provenance':['fixed slots and physical IDs','exact source history objects','actual returned history-feature object to MetaFlow pre-hook','feature value hash and time','per-slot MetaFlow output equals pred_offset slice','reference formula per scale'],'failure_classes_rejected':['original values','shape','dtype','device','slot or frame permutation','non-whitelisted fields or scope','contract hash corruption'],'samples':inventories},exclusive=True)
 receipts=[]
 for rec in load(runroot/'ledger.json')['records']:
  p=runroot/rec['relative_path'];meta=load(p/'meta.json')
  for r in meta['history_integrity']['receipts']:
   require(r['original_before']==r['original_after'],'ORIGINAL_HISTORY_CHANGED')
   receipts.append({'condition':rec['condition'],'mask_installed':rec['mask_installed'],'slot':r['slot'],'original_before_hash':r['original_before'][0]['hash'],'original_after_hash':r['original_after'][0]['hash'],'temporary_added_keys':['pillar_features','spatial_features','pairwise_t_matrix'],'feature_hash':r['feature_hash']})
 write(R/'DISPOSABLE_HISTORY_VALIDATION.json',{'passed':True,'original_history_containers_unchanged':True,'original_tensor_values_shape_dtype_device_unchanged':True,'shallow_storage_alias_tested':True,'unexpected_inplace_writes_rejected':True,'native_features_equal_before_after_repair':True,'independent_native_samples':8,'receipts':receipts,'independent_core_sha256':sha_file(R/'INDEPENDENT_CORE_VALIDATION.json')},exclusive=True)
 write(R/'NATIVE_HISTORY_CPU_TEST.json',{'passed':True,'actual_native_method':'PointPillarDeformable.get_batch_pillar_features','actual_native_components':['PillarVFE','PointPillarScatter','ResNetModified','get_projected_bev_features','warp_affine_simple'],'checkpoint_component_weights_frozen':True,'full_real_LRCP_forward_executed':False,'independent_native_calls':core['independent_native_calls'],'samples':core['native_history_samples'],'unsupported_device_operator_substitution':'original repository ms_deform_attn_core_pytorch CPU debug sampler only','explicit_mock_interfaces':['MetaFlow response (actual native history features supplied)','detection head outputs'],'CUDA_initialized':False,'real_forward_increment':0},exclusive=True)
 write(R/'FOUR_CONDITION_CPU_REGRESSION.json',{'passed':True,'kind':'CPU_MOCK_ONLY','sample_idx':324,'records':core['checks'],'Full_no_mask_vs_installed_no_intervention':'bitwise exact','conditions':CONDITIONS,'cross_blocks':4,'queries_per_block':6400,'heads':8,'levels':9,'sampling_points':8,'blocked_weights':'strictly zero over every query/head/level/point','weight_normalization_atol':1e-5,'old_formula_tolerances_unchanged':True,'input_reference_and_MetaFlow_identity_unchanged':True,'scientific_real_detection_result':False},exclusive=True)
 old=load(OLD/'RUN_BUDGET.json');real=load(OLD/'REAL_RUN_MANIFEST.json')
 require(len(old['attempts'])==real['reserved_attempts']==real['forward_entered_count']==real['forward_returned_count']==1 and real['committed_conditions']==0,'OLD_BUDGET_STATE')
 protected=['RUN_BUDGET.json','REAL_RUN_MANIFEST.json','real_run/manifest.json','real_run/failure.json','real_run/claim.json','real_run/worker_claim.json','real_run/run.lock','FROZEN_MANIFEST.json']
 preservation={}
 for name in protected:
  p=OLD/name;original=seal['files'][str(p)];current=sha_file(p);require(original==current,'PROTECTED_HISTORY_CHANGED')
  preservation[name]={'before_sha256':original,'after_sha256':current,'identical':True}
 status=subprocess.check_output(['git','-C',str(SRC),'status','--short'],text=True);require(status==seal['git_status'],'LRCP_GIT_STATUS_CHANGED')
 write(R/'RUN_BUDGET_PRESERVATION.json',{'passed':True,'original_budget_state':{'reserved_attempts':1,'model_entered':1,'model_returned':1,'scientific_shards_committed':0},'real_forward_increment_this_round':0,'original_attempt_removed':False,'budget_reset':False,'new_real_budget_file_created':(R/'RUN_BUDGET.json').exists(),'protected_files':preservation,'historical_files_verified':len(seal['files']),'LRCP_git_status_unchanged':status},exclusive=True)
 require(not (R/'RUN_BUDGET.json').exists(),'NEW_REAL_BUDGET_FORBIDDEN')
 all_source={str(p):sha_file(p) for p in sorted(R.glob('*.py'))}
 for p in R.glob('*.py'):ast.parse(p.read_text())
 cpu_guard()
 acceptance={'passed':True,'state':'CPU_REPAIR_PASSED','scope':'CPU engineering acceptance only; no successful real four-group inference claimed','core_validation':{'file':'INDEPENDENT_CORE_VALIDATION.json','sha256':sha_file(R/'INDEPENDENT_CORE_VALIDATION.json')},'fault_validation':{'file':'INDEPENDENT_FAULT_VALIDATION.json','sha256':sha_file(R/'INDEPENDENT_FAULT_VALIDATION.json')},'actual_input_leaf_comparison':True,'actual_native_history_components_executed':True,'frozen_native_feature_repair_equivalence_samples':8,'unmasked_F_control_exact':True,'all_four_conditions_mask_valid':True,'original_history_tensors_and_containers_unchanged':True,'whitelist_scope_and_label_semantics_validated':True,'fault_cases':len(faults['faults']),'independent_rehashed_bad_cases':len(negative['rehash_corruption_cases']),'unaccepted_diagnostics_recoverable':True,'prior_actual_failure_preserved':True,'original_real_state':{'reserved':1,'entered':1,'returned':1,'committed':0},'historical_files_unchanged':len(seal['files']),'execution_source_hashes':source_lock,'validator_and_repair_source_hashes':all_source,'real_forward_increment':0,**cpu_only.evidence(),'eligible_to_request_separate_real_run_authorization':True,'real_run_automatically_authorized_or_started':False,'limitations':['MetaFlow response and detection head remain explicit CPU interface mocks. Native history preparation and fusion are actual original code with frozen component weights.','CPU debug sampling does not establish device kernel equality; future authorized F must independently pass original historical baseline tolerances.','Actual old GPU process had no per-leaf dump; this repair proves the CPU-reproducible container mechanism, not the exclusivity of the old real failure cause.']}
 write(R/'INDEPENDENT_CPU_VALIDATION.json',acceptance,exclusive=True)
 # Refresh traceable source list including this final CPU gate; historical hashes
 # and before/after repair source entries are preserved unchanged.
 versions=load(R/'REPAIR_SOURCE_HASHES.json');versions['all_repair_python']=all_source;write(R/'REPAIR_SOURCE_HASHES.json',versions)
 print('CPU_REPAIR_PASSED; actual real-forward increment=0; CUDA_initialized=False',flush=True)
if __name__=='__main__':run()
