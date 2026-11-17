"""Independent CPU-only stopped-run forensics. Never promotes candidate_005,
never edits the sealed 60-leaf gate, and never emits partial scientific metrics.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,traceback
from validate_cpu import load_runtime_inputs,compare_original,expected_device_snapshot,validate_candidate
from input_integrity import snapshot
sys.path.insert(2,str(C3))
from runtime_common import alive
from device_guard import query_live

def run():
 root=R/'real_run';budget=load(R/'R2_RUN_BUDGET.json');m=load(root/'manifest.json');failure=load(root/'failure.json')
 expected=[(324,'F'),(324,'G-B'),(324,'G-C'),(324,'E'),(424,'F')]
 require([(a['sample_idx'],a['condition']) for a in budget['attempts']]==expected,'EXACT_STOPPED_ATTEMPTS')
 require(len(budget['attempts'])==5 and all(a['entered_epoch'] and a['returned_epoch'] for a in budget['attempts']),'ACTUAL_ENTERED_RETURNED_COUNTS')
 require(sum(a['committed_epoch'] is not None for a in budget['attempts'])==4 and not budget['attempts'][-1]['validated_epoch'] and not budget['attempts'][-1]['committed_epoch'],'LAST_ATTEMPT_UNACCEPTED')
 require(m['reserved']==m['entered']==m['returned']==5 and m['committed']==4 and m['state']=='RD3_R2_REAL_REJECTED' and failure['error']=='ORIGINAL_60_LEAF_CONTRACT','STOPPED_STATE')
 require(not alive(load(root/'claim.json')['owner']) and not alive(load(root/'worker_claim.json')),'WORKER_OR_OWNER_STILL_ALIVE')
 seal=load(R/'SOURCE_AND_CHECKPOINT_SEAL.json')
 for path,h in seal['execution_sources'].items():require(sha_file(path)==h,'EXECUTED_SOURCE_MODIFIED')
 for path,h in seal['historical_sources'].items():require(sha_file(path)==h,'OLD_EVIDENCE_MODIFIED')
 frozen=load(FROZEN);inventory=[];old_contract=load(REPAIR/'INPUT_LEAF_CONTRACT.json')
 for row in frozen['samples']:
  item=row['input'];require(sha_file(item['path'])==item['sha256'],'FROZEN_FILE_CHANGED');payload=torch.load(item['path'],map_location='cpu',weights_only=False);snap=snapshot(payload['batch'])
  require(digest(payload['batch'])==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'FROZEN_INPUT_RNG_CHANGED')
  r1=next(x for x in old_contract['samples'] if x['sample_idx']==row['sample_idx']);require(snap==r1['all_original_fields_and_leaves'],'EXISTING_R1_PER_SAMPLE_CONTRACT')
  inventory.append({'sample_idx':row['sample_idx'],'original_total_leaves':len(snap['leaves']),'original_Tensor_leaves':sum(x['kind']=='tensor' for x in snap['leaves'].values()),'GT_object_id_scalar_leaves':len(payload['batch']['ego']['object_ids']),'R1_original_contract_unchanged':True})
 accepted_checks=[]
 for rec in m['records']:
  path=root/rec['relative_path'];require(sha_file(path/'manifest.json')==rec['manifest_sha256'],'ACCEPTED_MANIFEST_HASH')
  accepted=load(path/'manifest.json');require(accepted['state']=='ACCEPTED' and accepted['backend']=='REAL','REAL_SCIENCE_KIND')
  for name,h in accepted['files'].items():require(sha_file(path/name)==h,'ACCEPTED_FILE_HASH')
  check=validate_candidate(root,root/accepted['candidate']);accepted_checks.append(check)
 inputs=[]
 for ordinal in range(1,6):
  path=root/'diagnostics'/f'candidate_{ordinal:03d}';manifest=load(path/'manifest.json');meta=load(path/'meta.json')
  require(manifest['state']=='UNACCEPTED' and manifest['stage']=='COLLECTED','UNACCEPTED_CANDIDATE_STATE')
  for name,h in manifest['files'].items():require(sha_file(path/name)==h,'UNACCEPTED_COMPLETE_FILE_HASH')
  row=next(x for x in frozen['samples'] if x['sample_idx']==meta['sample_idx']);payload=torch.load(row['input']['path'],map_location='cpu',weights_only=False);before=payload['batch'];after,runtime=load_runtime_inputs(path/'inputs_after.pt')
  compare_original(before,after)
  require(meta['input_leaf_before']==expected_device_snapshot(before,'cuda:0') and meta['input_leaf_after']==runtime,'ACTUAL_INPUT_LEAF_IDENTITY')
  require(all(v['device']=='cuda:0' for v in runtime['leaves'].values() if v['kind']=='tensor'),'INPUT_DEVICE_METADATA')
  offset=torch.cat([v['offset'].flatten(1,2).unsqueeze(2) for v in before['ego']['calibrate_data'].values()],2)
  require(torch.equal(after['ego']['label_dict']['offset'],offset),'NATIVE_LABEL_OFFSET_SEMANTICS')
  for slot in (1,2):require(list(after['ego']['calibrate_data'][slot]['lidar_history'][0])==['voxel_features','voxel_coords','voxel_num_points'],'ORIGINAL_HISTORY_CONTAINER_LEAK')
  inputs.append({'ordinal':ordinal,'sample_idx':meta['sample_idx'],'condition':meta['condition'],'original_total_leaves':len(meta['input_leaf_before']['leaves']),'input_values_shapes_dtypes_devices_and_container_order_unchanged':True,'original_history_container_unchanged':True,'formal_scientific_committed':ordinal<=4,'scope':'input forensics only; no promotion or science acceptance for ordinal5'})
 # Unmodified independent gate also rejects ordinal5's true 61-leaf schema.
 last_reason=None
 try:validate_candidate(root,root/'diagnostics/candidate_005')
 except Exception as e:last_reason=str(e)
 require(last_reason=='ORIGINAL_60_INPUT_LEAVES','UNCHANGED_INDEPENDENT_LEAF_GATE')
 require(not (root/'capture/sample_424_F').exists(),'FAILED_RECORD_PROMOTED')
 form=load(root/'final_validation.json');require(not form['passed'] and form['error']=='EXACT_8_BY_4_REAL_COMPLETENESS','FINAL_COMPLETENESS_REJECTION')
 first=load(R/'FIRST_FULL_BASELINE_CONTROL.json');require(first['passed'] and first['baseline_control']['native_alpha1_agrees'],'FIRST_F_FACT')
 write(R/'REAL_INPUT_INTEGRITY.json',{'input_forensics_passed':True,'all_5_executed_inputs_preserved':True,'overall_experiment_accepted':False,'gate_that_stopped_run':'ORIGINAL_60_LEAF_CONTRACT','independent_unmodified_leaf_gate_error':last_reason,'per_sample_original_leaf_inventory':inventory,'executed_candidates':inputs,'finding':'Total leaves are 54 plus GT object-id scalar count. All eight samples have 44 Tensor leaves. Fixed 60 applies to sample324 but sample424 has61; this is an R2-introduced schema-count gate error, not evidence of Tensor or input mutation.','no_gate_changed':True,'no_candidate_promoted_after_failure':True,'CUDA_initialized':False,'model_forwards':0},exclusive=True)
 try:live={'passed':True,**query_live(shutil.disk_usage(R).free,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
 except Exception as e:live={'passed':False,'error':str(e)}
 out={'passed':False,'state':'RD3_R2_REAL_REJECTED','scientific_results':'NOT_ESTIMABLE','reason':'sample424/F returned, then R2 fixed original-total-leaf==60 gate rejected its unchanged frozen61-leaf schema','failure_code':'ORIGINAL_60_LEAF_CONTRACT','final_completeness_rejection':form,'new_reserved':5,'new_entered':5,'new_returned':5,'committed_real_shards':4,'complete_8x4_real_records':False,'first_324_F_native_baseline_passed':True,'independently_revalidated_accepted_records':accepted_checks,'all_5_original_inputs_preserved':True,'per_sample_schema_variation':inventory,'unaccepted_424_F_preserved':True,'remaining_budget_not_used':27,'automatic_retry':False,'post_failure_forward_increment':0,'prior_RD3_budget_unchanged':{'reserved':1,'entered':1,'returned':1,'committed':0},'worker_and_owner_exited':True,'post_run_device_query':live,'CUDA_initialized':False,'model_forwards_during_validation':0}
 write(R/'INDEPENDENT_REAL_VALIDATION.json',out,exclusive=True)
 write(R/'POST_RUN_STATE.json',{'state':'RD3_R2_REAL_REJECTED','allow_automatic_resume':False,'reason':failure['error'],'reserved':5,'entered':5,'returned':5,'committed':4,'science':'NOT_ESTIMABLE'})
 counts={'F':2,'G-B':1,'G-C':1,'E':1};rows=[]
 for c in CONDITIONS:rows.append({'condition':c,'frozen_samples':8,'entered_forwards':counts[c],'accepted_sample_conditions':1,'TP50':'','FP50':'','FN50':'','TP70':'','FP70':'','FN70':'','Precision50':'','Recall50':'','Precision70':'','Recall70':'','local_AP50':'','local_AP70':'','final_prediction_count':'','NMS_removed_count':'','status':'NOT_ESTIMABLE_INCOMPLETE_8x4'})
 csvwrite(R/'FOUR_GROUP_REAL_METRICS.csv',rows)
 csvwrite(R/'PER_GT_TRANSITIONS.csv',[],['sample_idx','scene_id','condition_from','condition_to','gt_idx','iou_threshold','transition','from_matched_anchor','to_matched_anchor','status'])
 u=[];inter=[]
 for row in frozen['samples']:
  for name in ('B|C','C|B','B|A','C|A'):u.append({'sample_idx':row['sample_idx'],'scene_id':row['scene_id'],'utility':name,**{k:'' for k in ('TP50','FP50','FN50','TP70','FP70','FN70','detection_quality')},'status':'NOT_ESTIMABLE'})
  inter.append({'scope':'sample','sample_idx':row['sample_idx'],'scene_id':row['scene_id'],**{k:'' for k in ('TP50','FP50','FN50','TP70','FP70','FN70','detection_quality')},'status':'NOT_ESTIMABLE'})
 for scene in (28,29,93):inter.append({'scope':'scene','sample_idx':'','scene_id':scene,**{k:'' for k in ('TP50','FP50','FN50','TP70','FP70','FN70','detection_quality')},'status':'NOT_ESTIMABLE'})
 csvwrite(R/'AGENT_MARGINAL_UTILITY.csv',u);csvwrite(R/'AGENT_INTERACTION.csv',inter)
 csvwrite(R/'NMS_STAGE_ANALYSIS.csv',[],['sample_idx','scene_id','condition','gt_idx','stage','best_IoU','matched_anchor','suppressor_anchor','status'])
 write(R/'ANALYSIS_NOT_ESTIMABLE.json',{'state':'NOT_ESTIMABLE','reason':'4/32 accepted real records; whole frozen8-cohort analyses forbidden before complete independent acceptance','partial_324_science_not_substituted':True,'numeric_empty_cells_mean':'unavailable, not zero','header_only_tables_mean':'no allowed aggregate scientific analysis, not no transitions','new_entered':5,'new_returned':5,'new_committed':4,'post_failure_forward_increment':0})
 cpu_guard();print('STOPPED_R2_CPU_AUDIT_COMPLETE; REAL_REJECTED; new=5 entered/5 returned/4 committed; no retry',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'STOPPED_CPU_AUDIT_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
