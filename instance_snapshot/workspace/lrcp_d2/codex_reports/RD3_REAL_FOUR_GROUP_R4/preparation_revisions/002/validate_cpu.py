"""Independent CPU candidate/final validator: never constructs or forwards model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_only
cpu_only.install()
from rd3_common import *
import torch,argparse,traceback,copy
from input_integrity import snapshot
from attention_validation import verify_attention
from INDEPENDENT_VALIDATOR import verify_heads,same
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
from types import SimpleNamespace
_import_path=sys.path.copy();sys.path.insert(1,str(REPORTS/'P4_C2_20261027_233036_301795'))
from c1_reused import eval_native
sys.path[:]=_import_path

def load_runtime_inputs(path):
 locations={}
 def cpu_map(storage,location):locations[storage.data_ptr()]=location;return storage
 value=torch.load(path,map_location=cpu_map,weights_only=False)
 runtime=snapshot(value)
 def visit(x,p=''):
  if isinstance(x,dict):
   for k,v in x.items():visit(v,p+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   for k,v in enumerate(x):visit(v,p+'/'+str(k))
  elif isinstance(x,torch.Tensor):runtime['leaves'][p]['device']=locations[x.untyped_storage().data_ptr()]
 visit(value);runtime['hash']=digest({'nodes':runtime['nodes'],'leaves':runtime['leaves']});return value,runtime

def expected_device_snapshot(value,device):
 result=snapshot(value)
 for v in result['leaves'].values():
  if v['kind']=='tensor':v['device']=device
 result['hash']=digest({'nodes':result['nodes'],'leaves':result['leaves']});return result

def compare_original(before,after,path=''):
 require(type(before) is type(after),'ORIGINAL_CONTAINER_TYPE '+path)
 if isinstance(before,dict):
  additions=['offset'] if path=='/ego/label_dict' else []
  require(list(after)==list(before)+additions,'ORIGINAL_NAMES_OR_ORDER '+path)
  for k in before:compare_original(before[k],after[k],path+'/'+str(k))
 elif isinstance(before,(list,tuple)):
  require(len(before)==len(after),'ORIGINAL_SEQUENCE '+path)
  for i,(a,b) in enumerate(zip(before,after)):compare_original(a,b,path+'/'+str(i))
 elif isinstance(before,torch.Tensor):require(before.shape==after.shape and before.dtype==after.dtype and torch.equal(before,after),'ORIGINAL_TENSOR_VALUES_METADATA '+path)
 elif isinstance(before,np.ndarray):require(before.shape==after.shape and before.dtype==after.dtype and np.array_equal(before,after),'ORIGINAL_ARRAY '+path)
 else:require(before==after,'ORIGINAL_SCALAR '+path)

def historical(a,row):
 evidence=load(R2/'HISTORY_BASELINE_AUDIT.json');entry=next(x for x in evidence['samples'] if x['sample_idx']==row['sample_idx']);errors={}
 for kind in ('P3A','P4_native_alpha1'):require(sha_file(entry[kind]['path'])==entry[kind]['sha256'],'HISTORY_REFERENCE_SHA')
 with np.load(entry['P4_native_alpha1']['path'],allow_pickle=False) as old:
  for key,tol in [('cls_preds',1e-6),('reg_preds',1e-5),('dir_preds',1e-5),('final_corners',1e-5),('final_scores',1e-6),('native_nms_keep',0)]:
   same(a[key],old[key],'P4_NATIVE_'+key,tol);errors[key]=float(np.max(np.abs(a[key].astype(float)-old[key].astype(float)),initial=0.))
  for key in ('score_mask','quality_mask','ids_score_pass','ids_quality_pass','ids_nms_top1000','ids_nms_keep','ids_final_range','range_mask'):same(a[key],old[key],'P4_STAGE_'+key)
 with np.load(entry['P3A']['path'],allow_pickle=False) as old:
  for new,key,tol in [('final_corners','pred_boxes',1e-5),('final_scores','pred_scores',1e-6),('GT_canonical','gt_boxes',1e-5)]:same(a[new],old[key],'P3A_'+key,tol)
  for t in (.5,.7):
   _,matched,flags=eval_native(old['pred_boxes'],old['pred_scores'],old['gt_boxes'],t);same(a[f'gt_tp{int(t*100)}'],flags,'P3A_GT_MATCH');same(a[f'prediction_gt_idx{int(t*100)}'],matched,'P3A_UNIQUE_PRED_MATCH')
 return {'native_alpha1_agrees':True,'P3A_fields_compared':['pred_boxes','pred_scores','gt_boxes','recomputed_unique_GT_matches'],'P4_supplemental_fields':['cls_preds','reg_preds','dir_preds','native_nms_keep','native_stage_lineage'],'maximum_absolute_errors':errors,'tolerances_unchanged':True}

def validate_candidate(root,candidate,cpu_test=False):
 root=Path(root).resolve();path=Path(candidate).resolve();require(path.parent==root/'diagnostics','CANDIDATE_PATH_SCOPE')
 manifest=load(path/'manifest.json');require(manifest['state']=='UNACCEPTED' and manifest['stage']=='COLLECTED','CANDIDATE_NOT_COMPLETE')
 require(set(manifest['files'])=={'raw_arrays.npz','raw_meta.json','inputs_after.pt','arrays.npz','meta.json'},'COMPLETE_CANDIDATE_FILE_SET')
 for name,h in manifest['files'].items():require(sha_file(path/name)==h,'CANDIDATE_FILE_HASH '+name)
 meta=load(path/'meta.json');rawmeta=load(path/'raw_meta.json')
 require(rawmeta['state']=='UNACCEPTED' and rawmeta['stage']=='MODEL_RETURNED' and rawmeta['model_returned'] is True,'DURABLE_MODEL_RETURN')
 require(rawmeta['sample_idx']==meta['sample_idx'] and rawmeta['input_leaf_before']==meta['input_leaf_before'] and rawmeta['input_leaf_after']==meta['input_leaf_after'],'RAW_COLLECTED_INPUT_EVIDENCE')
 expected_kind='CPU_TEST' if cpu_test else 'REAL';require(meta['backend']==expected_kind,'MOCK_IN_REAL_EXPERIMENT')
 frozen=load(FROZEN);row=next(x for x in frozen['samples'] if x['sample_idx']==meta['sample_idx']);item=row['input']
 require(meta['input_manifest']==row and meta['frozen_sha256']==sha_file(FROZEN) and meta['checkpoint_sha256']==CH,'INPUT_MODEL_IDENTITIES')
 require(meta['condition'] in CONDITIONS and meta['keep']==CONDITIONS[meta['condition']] and meta['slot_physical_ids']==['1','2','3'],'FIXED_AGENT_SLOTS')
 if not cpu_test:
  from control import SEAL,BUDGET,REAL_ROOT
  is_r4=meta.get('schema')=='RD3_R4_REAL_V1'
  sealpath=SEAL if is_r4 else R2/'SOURCE_AND_CHECKPOINT_SEAL.json'
  native_root=REAL_ROOT if is_r4 else R2/'real_run';ledgerpath=BUDGET if is_r4 else R2/'R2_RUN_BUDGET.json'
  if is_r4:require(root==REAL_ROOT,'R4_REAL_ROOT_SCOPE')
  else:require(root==R2/'real_run','HISTORICAL_REAL_ROOT_SCOPE')
  seal=load(sealpath);require(meta['source_seal_sha256']==sha_file(sealpath),'SOURCE_SEAL_IDENTITY')
  for source,h in seal['execution_sources'].items():require(sha_file(source)==h,'EXECUTION_SOURCE_CHANGED')
  require(sha_file(CKPT)==CH,'CHECKPOINT_CHANGED');identity=load(C2/'CHECKPOINT_IDENTITY.json');require(meta['parameter_digest']==identity['parameter_digest'] and meta['buffer_digest']==identity['buffer_digest'],'NATIVE_PARAMETERS_BUFFERS')
  n=meta['native_entry'];require(n['model_class']=='opencood.models.point_pillar_deformable.PointPillarDeformable' and n['MetaFlow_class']=='opencood.models.meta_flow.MetaFlow' and n['head_classes']=={k:'Conv2d' for k in ('cls_head','reg_head','dir_head')} and n['sampler']=='opencood.utils.ms_deform_attn_ops.functions.ms_deform_attn_func.MSDeformAttnFunction' and n['mock_modules_present'] is False,'REAL_ENTRY_PROVENANCE')
  require(sha_file(n['extension_path'])==n['extension_sha256'],'COMPILED_DEVICE_KERNEL_SHA')
  require(n==load(native_root/'NATIVE_ENTRY_RUNTIME.json'),'ACTUAL_R2_NATIVE_RUNTIME_IDENTITY')
  attempt=rawmeta['attempt'];ledger=load(ledgerpath)['attempts'];actual=ledger[attempt['ordinal']-1]
  require(actual['ordinal']==attempt['ordinal'] and actual['sample_idx']==meta['sample_idx'] and actual['condition']==meta['condition'] and actual['kind']=='REAL' and actual['entered_epoch'] is not None and actual['returned_epoch']>=actual['entered_epoch'],'PERSISTENT_REAL_ENTERED_RETURNED')
  require(rawmeta['checkpoint_sha256']==CH and rawmeta['source_seal_sha256']==meta['source_seal_sha256'],'RAW_REAL_MODEL_SOURCE_HASH')
  resources=load(native_root/'resources.json');worker=load(native_root/'worker_claim.json');require(resources['device']['uuid']=='GPU-019e1215-c270-091c-0000-0000c0610973' and resources['device']['name']=='PPU-ZW810E' and resources['worker_pid']==worker['pid'],'DEVICE_WORKER_PROVENANCE')
 require(sha_file(item['path'])==item['sha256'],'FROZEN_ARCHIVE_SHA');payload=torch.load(item['path'],map_location='cpu',weights_only=False);before=payload['batch']
 require(digest(before)==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'FROZEN_INPUT_AND_RNG')
 after,runtime=load_runtime_inputs(path/'inputs_after.pt');compare_original(before,after)
 expected_device='cpu' if cpu_test else 'cuda:0';require(meta['input_leaf_before']==expected_device_snapshot(before,expected_device),'ORIGINAL_RUNTIME_INPUT_LEAVES')
 require(meta['input_leaf_after']==runtime,'ACTUAL_SERIALIZED_AFTER_LEAVES')
 # Explicitly derive path/type/value schema for THIS actual frozen archive.
 expected_original=expected_device_snapshot(before,expected_device)
 existing=next(x for x in load(REPAIR/'INPUT_LEAF_CONTRACT.json')['samples'] if x['sample_idx']==row['sample_idx'])
 require(snapshot(before)==existing['all_original_fields_and_leaves'],'INDEPENDENT_R1_PER_SAMPLE_CONTRACT')
 require(set(meta['input_leaf_before']['leaves'])==set(expected_original['leaves']) and meta['input_leaf_before']['nodes']==expected_original['nodes'] and meta['input_leaf_before']['leaves']==expected_original['leaves'],'EXACT_PER_SAMPLE_SCHEMA')
 for p,v in runtime['leaves'].items():
  if v['kind']=='tensor':require(v['device']==expected_device,'RUNTIME_DEVICE_CHANGE '+p)
 ego=after['ego'];require(int(ego['record_len'][0])==3 and list(map(str,ego['cav_id_list']))==['1','2','3'],'ORIGINAL_PHYSICAL_MAPPING')
 label_expected=torch.cat([v['offset'].flatten(1,2).unsqueeze(2) for v in ego['calibrate_data'].values()],2)
 require(torch.equal(ego['label_dict']['offset'],label_expected) and ego['label_dict']['offset'].shape==label_expected.shape and ego['label_dict']['offset'].dtype==label_expected.dtype,'LABEL_OFFSET_DERIVED_SEMANTICS')
 history=meta['history_integrity'];require([x['slot'] for x in history['receipts']]==[1,2] and [x['slot'] for x in history['metaflow_events']]==[1,2],'METAFLOW_PHYSICAL_ORDER')
 for receipt,event in zip(history['receipts'],history['metaflow_events']):
  slot=receipt['slot'];source=before['ego']['calibrate_data'][slot]['lidar_history'][0];expected=expected_device_snapshot(source,expected_device)
  require(receipt['original_before']==receipt['original_after']==[expected],'ORIGINAL_HISTORY_LEAVES')
  temp=receipt['temporary_after'][0];require(temp['hash']==digest({'nodes':temp['nodes'],'leaves':temp['leaves']}),'TEMP_HISTORY_HASH')
  require(temp['nodes']['']['keys']==expected['nodes']['']['keys']+[['str',repr(k)] for k in ('pillar_features','spatial_features','pairwise_t_matrix')],'DERIVED_WHITELIST_SCOPE')
  for p,v in expected['leaves'].items():require(temp['leaves'][p]==v,'TEMPORARY_ORIGINAL_TENSOR_CHANGE')
  require(set(temp['leaves'])==set(expected['leaves'])|{'/pillar_features','/spatial_features','/pairwise_t_matrix'},'TEMP_FIELD_SET')
  pair=expected_device_snapshot(before['ego']['pairwise_t_matrix'],expected_device)['leaves'][''];require(temp['leaves']['/pairwise_t_matrix']==pair and receipt['pairwise_hash']==digest(before['ego']['pairwise_t_matrix']),'ORIGINAL_PAIRWISE_VALUES')
  time=expected_device_snapshot(before['ego']['calibrate_data'][slot]['time_delay'],expected_device)['leaves'][''];require(event['time']==time and event['input_feature_hash']==receipt['feature_hash'] and event['physical_id']==receipt['physical_id']==str(slot+1),'AGENT_TIME_FEATURE_PROVENANCE')
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));pp=build_postprocessor(h['postprocess'],False)
 baseline=None
 with np.load(path/'arrays.npz',allow_pickle=False) as z:
  a={k:z[k] for k in z.files}
  with np.load(path/'raw_arrays.npz',allow_pickle=False) as raw:
   require(all(k in raw and k in a for k in ('cls_preds','reg_preds','dir_preds','pred_offset','metaflow_slot_1','metaflow_slot_2')),'RAW_HEAD_METAFLOW_COMPLETENESS')
   for key in raw.files:
    if key in a:same(a[key],raw[key],'RAW_FINAL_ARRAY_CONSISTENCY_'+key)
    if raw[key].dtype.kind=='f':require(np.isfinite(raw[key]).all(),'RAW_OUTPUT_NONFINITE_'+key)
  for key,shape in [('cls_preds',(1,2,80,80)),('reg_preds',(1,14,80,80)),('dir_preds',(1,4,80,80))]:require(a[key].shape==shape and a[key].dtype==np.float32,'REAL_HEAD_SHAPE_DTYPE')
  require(a['raw_corners'].shape==(12800,8,3) and a['scores_all'].shape==(12800,),'COMPLETE_RAW_CANDIDATES')
  verify_attention(a,meta);verify_heads(a,meta,before,pp)
  critical=['cls_preds','reg_preds','dir_preds','pred_offset','raw_corners','scores_all','final_corners','final_scores','IoU_GT_raw','metaflow_slot_1','metaflow_slot_2']
  require(all(np.isfinite(a[k]).all() for k in critical),'NONFINITE_SCIENTIFIC_OUTPUT')
  for event in history['metaflow_events']:require(digest(a[f"metaflow_slot_{event['slot']}"])==event['output_hash'],'METAFLOW_OUTPUT_AGENT_HASH')
  with np.load(BASE/f"P3A_20261014_120929_boxes/sample_{row['sample_idx']}_alpha_1.0.npz",allow_pickle=False) as old:
   current=array(pp.generate_gt_bbx(before));mapping,_=gt_mapping(old['gt_boxes'],current);same(a['GT_canonical'],current[mapping],'NATIVE_CURRENT_GT_IDENTITY',1e-5)
  if meta['condition']=='F' and not cpu_test:baseline=historical(a,row)
 cpu_guard()
 return {'passed':True,'backend':expected_kind,'sample_idx':meta['sample_idx'],'condition':meta['condition'],'candidate_manifest_sha256':sha_file(path/'manifest.json'),'input_leaves':len(expected_original['leaves']),'input_contract_hash':snapshot(before)['hash'],'validator_source_sha256':sha_file(Path(__file__)),'source_state_preserved':'UNACCEPTED','R2_attempt_id':rawmeta['attempt']['ordinal'],'source_manifest_state':manifest['state'],'source_execution_identity_checked':not cpu_test,'input_integrity_independently_verified':True,'all_original_values_and_devices_preserved':True,'attention_independently_verified':True,'native_postprocess_and_unique_GT_recomputed':True,'baseline_control':baseline,'CUDA_initialized':False,'model_forwards':0}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--candidate');p.add_argument('--output',required=True);p.add_argument('--cpu-test',action='store_true');args=p.parse_args()
 try:
  require(args.candidate is not None,'R4_EVIDENCE_VALIDATION_ONLY')
  result=validate_candidate(args.root,args.candidate,args.cpu_test)
  write(args.output,result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)
 except BaseException as e:
  write(args.output,{'passed':False,'state':'R4_CANDIDATE_REJECTED','error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0},exclusive=True);raise
