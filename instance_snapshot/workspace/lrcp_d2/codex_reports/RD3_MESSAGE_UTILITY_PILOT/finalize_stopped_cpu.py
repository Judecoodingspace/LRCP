"""Independent CPU audit of a failed run; emits missing results, never predictions.
Checks actual budget and immutable files rather than trusting producer passed flags.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,pickle,collections,subprocess,traceback
sys.path.insert(2,str(C3))
from runtime_common import alive
from device_guard import query_live

def run():
 started=time.monotonic();root=R/'real_run';ready=load(R/'CPU_READY.json');frozen=load(R/'FROZEN_MANIFEST.json')
 require(sha_file(R/'FROZEN_MANIFEST.json')==ready['frozen_sha256'],'FROZEN_CHANGED')
 for p,h in ready['locked_files'].items():require(sha_file(p)==h,'RUN_SOURCE_CHANGED '+p)
 require(sha_file(CKPT)==CH,'CHECKPOINT_CHANGED')
 original=load(root/'manifest.json');budget=load(R/'RUN_BUDGET.json');attempts=budget['attempts']
 require(budget['max_attempts']==32 and len(attempts)==1,'BUDGET_MISMATCH')
 require(attempts[0]['sample_idx']==324 and attempts[0]['condition']=='F' and attempts[0]['status']=='MODEL_RETURNED','FORWARD_EVIDENCE')
 require(original['state']=='FAILED' and original['error']=='OTHER_INPUT_MUTATION' and original['records']==[],'RAW_RUN_STATE')
 require(load(root/'failure.json')['attempts']==1 and load(root/'failure.json')['records']==0,'FAILURE_STATE')
 require(not any((root/'capture').glob('sample_*')),'UNEXPECTED_SHARD')
 require(not alive(load(root/'worker_claim.json')) and not alive(load(root/'claim.json')['owner']),'RUN_STILL_ACTIVE')
 # Reconstruct population counts from the original index, independently of audit
 # producer's flags, and rehash every real frozen input file after the stopped run.
 source=Path('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl');require(sha_file(source)==frozen['raw_index_sha256'],'RAW_INDEX_CHANGED')
 info=pickle.load(source.open('rb'));source_manifest=load(FULL/'INPUT_MANIFEST.json');common=list(map(int,load(BASE/'P1_common910_indices.json')['sample_ids']))
 require([x['sample_idx'] for x in source_manifest['samples']]==common and len(common)==910,'COMMON910_ORDER')
 counts=collections.Counter();scenes=set()
 import re
 for n,item in enumerate(source_manifest['samples']):
  require(sha_file(item['path'])==item['sha256'],'HISTORICAL_INPUT_CHANGED '+str(item['sample_idx']))
  raw=info[item['sample_idx']];count=raw['agent_num'];counts[count]+=1
  require(raw['token']==item['sample_token'],'TOKEN_CHANGED')
  if count==3:scenes.add(int(re.search(r'scene_(\d+)_',raw['lidar_path_1'])[1]))
  if n%100==0:cpu_guard();print('INDEPENDENT_POST_RUN_INPUT_HASH',n+1,'/910',flush=True)
 for row in frozen['samples']:
  item=row['input'];payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False);e=payload['batch']['ego']
  require(digest(payload['batch'])==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'FROZEN_SEMANTICS_CHANGED')
  require(int(e['record_len'][0])==3 and list(map(str,e['cav_id_list']))==['1','2','3'],'FROZEN_THREE_AGENT_IDENTITY')
  require(set(e['calibrate_data'][1]['lidar_history'][0])=={'voxel_features','voxel_coords','voxel_num_points'},'FROZEN_HISTORY_POLLUTION')
  require(set(e['calibrate_data'][2]['lidar_history'][0])=={'voxel_features','voxel_coords','voxel_num_points'},'FROZEN_HISTORY_POLLUTION')
  del payload,e
 diagnosis=load(R/'INPUT_MUTATION_CPU_REPRO.json');require(diagnosis['passed'] and diagnosis['real_model_forwards']==0,'CPU_REPRO')
 # Preserve producer's original completed-collection counter (0). Report the
 # independently evidenced MODEL_RETURNED counter (1) in the delivery manifest.
 corrected={**original,'real_forwards':1,'forward_entered_count':1,'forward_returned_count':1,'committed_conditions':0,'reserved_attempts':1,'producer_completed_collection_count':original['real_forwards'],'raw_producer_manifest':str(root/'manifest.json'),'raw_producer_manifest_sha256':sha_file(root/'manifest.json'),'correction_basis':'persistent attempt status MODEL_RETURNED; producer increments its counter after collection, which failed','automatic_retry':False,'scientific_results_accepted':False}
 write(R/'REAL_RUN_MANIFEST.json',corrected)
 resource_snapshot=load(root/'resources.json')
 try:live={'query_passed':True,**query_live(shutil.disk_usage(R).free,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
 except Exception as e:live={'query_passed':False,'error':str(e)}
 validation={'passed':False,'audit_passed':True,'state':'REAL_REJECTED_INPUT_INTEGRITY','scientific_results_accepted':False,'reason':'OTHER_INPUT_MUTATION at backend.py:99; no atomic scientific shard committed','independent_completeness_gate':load(root/'validation.json'),'source_files_unchanged':len(ready['locked_files']),'checkpoint_unchanged':True,'all_910_input_files_unchanged':True,'frozen_8_input_and_RNG_values_unchanged':True,'raw_agent_count_distribution':dict(counts),'three_agent_scenes':sorted(scenes),'reserved_attempts':1,'model_calls_entered':1,'model_calls_returned':1,'committed_shards':0,'nonfull_model_calls':0,'worker_and_owner_exited':True,'no_retry':True,'native_input_mutation_cpu_reproduction':diagnosis,'last_real_supervisor_resources':resource_snapshot,'post_run_device_query':live,'CUDA_initialized_in_this_CPU_process':torch.cuda.is_initialized(),'real_model_forwards_in_this_CPU_process':0,'elapsed_seconds':time.monotonic()-started}
 write(R/'INDEPENDENT_VALIDATION.json',validation,exclusive=True)
 write(R/'POST_REAL_GATE_STATUS.json',{'allowed_to_resume':False,'prelaunch_CPU_READY_is_historical_only':True,'failure_gate':'OTHER_INPUT_MUTATION','reserved_forwards':1,'remaining_budget_does_not_authorize_automatic_retry':True,'scientific_results_accepted':False})
 # Required deliverables have schema plus explicit missing statuses. Missing is
 # not zero and header-only transitions do not imply zero scientific changes.
 fields=['condition','sample_count','TP50','FP50','FN50','TP70','FP70','FN70','Precision50','Recall50','Precision70','Recall70','local_AP50','local_AP70','status','reason'];rows=[]
 for c in CONDITIONS:
  row={k:'' for k in fields};row.update(condition=c,sample_count=8,status='FORWARD_RETURNED_NO_COMMITTED_EVIDENCE' if c=='F' else 'NOT_EXECUTED',reason='INPUT_INTEGRITY_GATE_FAILED; all scientific metrics unavailable');rows.append(row)
 csvwrite(R/'FOUR_CONDITION_METRICS.csv',rows,fields)
 csvwrite(R/'PER_GT_TRANSITIONS.csv',[],['sample_idx','scene_id','condition','gt_idx','iou_threshold','transition','full_matched_anchor','removed_matched_anchor','status'])
 utility=[];interaction=[]
 for sample in frozen['samples']:
  for name in ('B|C','C|B','B|A','C|A'):utility.append({'sample_idx':sample['sample_idx'],'scene_id':sample['scene_id'],'utility':name,**{k:'' for k in ('TP50','FP50','FN50','TP70','FP70','FN70')},'status':'NOT_ESTIMABLE_NO_FOUR_GROUP_EVIDENCE'})
  interaction.append({'scope':'sample','sample_idx':sample['sample_idx'],'scene_id':sample['scene_id'],**{k:'' for k in ('TP50','FP50','FN50','TP70','FP70','FN70')},'status':'NOT_ESTIMABLE_NO_FOUR_GROUP_EVIDENCE'})
 for scene in sorted(scenes):interaction.append({'scope':'scene','sample_idx':'','scene_id':scene,**{k:'' for k in ('TP50','FP50','FN50','TP70','FP70','FN70')},'status':'NOT_ESTIMABLE_NO_FOUR_GROUP_EVIDENCE'})
 csvwrite(R/'AGENT_UTILITY.csv',utility);csvwrite(R/'AGENT_INTERACTION.csv',interaction)
 write(R/'ANALYSIS_NOT_ESTIMABLE.json',{'state':'NOT_ESTIMABLE','four_group_metrics_available':False,'AP_available':False,'GT_transition_count_available':False,'utility_available':False,'interaction_available':False,'CSV_empty_numeric_cells_mean':'unavailable, never zero','PER_GT_TRANSITIONS_header_only_means':'no accepted paired records, not absence of GT transitions','forward_count':1,'atomic_shards':0,'reason':'formal input-integrity failure before first F scientific commit','no_model_forward_during_analysis':True})
 cpu_guard();print('STOPPED_RUN_CPU_AUDIT_COMPLETE; SCIENCE_REJECTED; real forwards=1; committed=0',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'FINAL_CPU_AUDIT_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()});raise
