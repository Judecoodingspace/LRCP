"""Separate CPU validator: recursively compares REAL input tensors/metadata;
reruns native history components from frozen data; does not forward the LRCP
model or trust producer passed flags. No TP/FP/AP conclusions are generated.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_only
cpu_only.install()
from rd3_common import *
import torch,copy,argparse,gc,traceback,fcntl
from attention_validation import verify_attention
from INDEPENDENT_VALIDATOR import verify_heads,same
from opencood.data_utils.post_processor import build_postprocessor
from mock_backend import HistoryOwner,checkpoint,NativeCallTrace,config
from opencood.models.point_pillar_deformable import PointPillarDeformable

ALLOWED_EXTRA=['pillar_features','spatial_features','pairwise_t_matrix']

def compare_tree(frozen,actual,path=''):
 require(type(frozen) is type(actual),'INPUT_TYPE '+path)
 if isinstance(frozen,dict):
  allowed=['offset'] if path=='/ego/label_dict' else []
  expected=list(frozen)
  require(list(actual)==expected+allowed,'ORIGINAL_FIELD_NAMES_ORDER '+path)
  for key in frozen:compare_tree(frozen[key],actual[key],path+'/'+str(key))
 elif isinstance(frozen,(list,tuple)):
  require(len(frozen)==len(actual),'INPUT_SEQUENCE_LENGTH '+path)
  for i,(a,b) in enumerate(zip(frozen,actual)):compare_tree(a,b,path+'/'+str(i))
 elif isinstance(frozen,torch.Tensor):
  require(frozen.shape==actual.shape,'INPUT_SHAPE '+path);require(frozen.dtype==actual.dtype,'INPUT_DTYPE '+path);require(frozen.device==actual.device,'INPUT_DEVICE '+path);require(torch.equal(frozen,actual),'INPUT_VALUES '+path)
 elif isinstance(frozen,np.ndarray):require(frozen.shape==actual.shape and frozen.dtype==actual.dtype and np.array_equal(frozen,actual),'INPUT_ARRAY '+path)
 else:require(frozen==actual,'INPUT_SCALAR '+path)

def descriptor(t):
 require(isinstance(t,torch.Tensor),'TENSOR_EXPECTED')
 return {'kind':'tensor','shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'layout':str(t.layout),'value_hash':digest(t)}

def validate_snapshot_against_actual(record,value):
 # Inventory is independently enumerated; producer before/after hashes alone
 # cannot certify input integrity. JSON-safe metadata hash is also recomputed.
 require(record['hash']==digest({'nodes':record['nodes'],'leaves':record['leaves']}),'LEAF_CONTRACT_HASH')
 leaves={};nodes={}
 def visit(x,p):
  if isinstance(x,dict):
   nodes[p]={'kind':type(x).__name__,'keys':[[type(k).__name__,repr(k)] for k in x]}
   for k,v in x.items():visit(v,p+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   nodes[p]={'kind':type(x).__name__,'length':len(x)}
   for k,v in enumerate(x):visit(v,p+'/'+str(k))
  elif isinstance(x,torch.Tensor):leaves[p]=descriptor(x)
  elif isinstance(x,np.ndarray):leaves[p]={'kind':'ndarray','shape':list(x.shape),'dtype':str(x.dtype),'value_hash':digest(x)}
  else:leaves[p]={'kind':type(x).__name__,'value_hash':digest(x),'repr':repr(x)}
 visit(value,'');require(leaves==record['leaves'] and nodes==record['nodes'],'ACTUAL_LEAF_INVENTORY')

def native_history_reference(owner,row):
 item=row['input'];require(sha_file(item['path'])==item['sha256'],'FROZEN_ARCHIVE_SHA')
 payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False);batch=payload['batch'];e=batch['ego']
 require(digest(batch)==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'FROZEN_SEMANTIC_HASH')
 require(int(e['record_len'][0])==3 and list(map(str,e['cav_id_list']))==['1','2','3'],'SLOT_IDENTITIES')
 expected=[]
 for slot in (1,2):
  history=e['calibrate_data'][slot]['lidar_history'][0];require(tuple(history)==('voxel_features','voxel_coords','voxel_num_points'),'ORIGINAL_HISTORY_KEYS')
  require(set(history['voxel_coords'][:,0].tolist())==set(range(5)),'FIVE_FRAME_ORDER')
  original=copy.deepcopy(history);reference=copy.deepcopy(history)
  with torch.no_grad():raw=PointPillarDeformable.get_batch_pillar_features(owner,[reference],e['pairwise_t_matrix'],slot,[0])
  # Repaired boundary semantics independently implemented here, not via producer
  # wrapper: a shallow container around the ORIGINAL shared tensor objects.
  temp=dict(history)
  require(all(temp[k] is history[k] for k in history),'SHALLOW_ALIAS')
  with torch.no_grad():repaired=PointPillarDeformable.get_batch_pillar_features(owner,[temp],e['pairwise_t_matrix'],slot,[0])
  compare_tree(original,history,'/native_history')
  require(list(temp)==list(original)+ALLOWED_EXTRA and list(reference)==list(temp),'TEMPORARY_WHITELIST_SCOPE')
  for key in original:
   require(descriptor(original[key])==descriptor(temp[key]),'TEMP_ORIGINAL_LEAF_MODIFIED')
  for a,b in zip(raw,repaired):require(torch.equal(a,b),'NATIVE_HISTORY_NUMERIC_CHANGE')
  expected.append({'slot':slot,'physical_id':e['calibrate_data'][slot]['physical_cav_ids'][0],'feature_hash':digest(repaired),'feature_shapes':[list(t.shape) for t in repaired],'original':original,'temporary':temp,'metaflow_input':repaired,'pairwise_hash':digest(e['pairwise_t_matrix'])})
 return batch,expected

def validate_case(root,rec,state,reference):
 path=Path(root)/rec['relative_path'];require(path.parent==Path(root)/'capture','ONLY_CPU_CAPTURE_ALLOWED')
 require(sha_file(path/'manifest.json')==rec['manifest_sha256'],'CASE_MANIFEST_SHA');m=load(path/'manifest.json');require(m['kind']=='MOCK_ONLY_CPU_R1','MOCK_CASE_KIND')
 require(set(m['files'])=={'arrays.npz','inputs_after.pt','meta.json'},'CASE_FIELDS')
 for file,h in m['files'].items():require(sha_file(path/file)==h,'CASE_FILE_SHA')
 meta=load(path/'meta.json');sources=load(Path(root)/'claim.json')['source_hashes']
 require(meta['run_source_hashes']==sources,'RUN_SOURCE_VERSION')
 for source,h in sources.items():require(sha_file(source)==h,'RUN_SOURCE_HASH')
 row=load(FROZEN)['samples'][0];require(meta['sample_idx']==row['sample_idx'] and meta['input_manifest']==row and meta['frozen_sha256']==sha_file(FROZEN),'FROZEN_AGENT_MAPPING')
 after=torch.load(path/'inputs_after.pt',map_location='cpu',weights_only=False);before,expected=reference
 compare_tree(before,after);e=after['ego'];off=torch.cat([v['offset'].flatten(1,2).unsqueeze(2) for v in e['calibrate_data'].values()],2)
 require(torch.equal(e['label_dict']['offset'],off) and e['label_dict']['offset'].dtype==off.dtype and e['label_dict']['offset'].device==off.device,'LABEL_OFFSET_ONLY_SEMANTICS')
 validate_snapshot_against_actual(meta['input_leaf_before'],before);validate_snapshot_against_actual(meta['input_leaf_after'],after)
 require(meta['checkpoint_sha256']==CH and meta['slot_physical_ids']==['1','2','3'],'MODEL_SLOT_IDENTITY')
 for component in meta['component_identity']:
  subset={k[len(component['prefix']):]:v for k,v in state.items() if k.startswith(component['prefix'])}
  require(component['state_hash']==digest(subset) and component['entries']==len(subset),'FROZEN_COMPONENT_IDENTITY')
 evidence=meta['history_integrity'];require(len(evidence['receipts'])==len(evidence['metaflow_events'])==2,'NATIVE_HISTORY_CALLS')
 for item,ev,truth in zip(evidence['receipts'],evidence['metaflow_events'],expected):
  require(item['slot']==truth['slot'] and item['physical_id']==truth['physical_id'] and ev['slot']==truth['slot'],'AGENT_HISTORY_MAPPING')
  require(item['feature_hash']==ev['input_feature_hash']==truth['feature_hash'] and item['feature_shapes']==truth['feature_shapes'],'METAFLOW_ACTUAL_INPUT')
  require(item['pairwise_hash']==truth['pairwise_hash'],'PAIRWISE_ORIGINAL_VALUES')
  validate_snapshot_against_actual(item['original_before'][0],truth['original']);validate_snapshot_against_actual(item['original_after'][0],truth['original']);validate_snapshot_against_actual(item['temporary_after'][0],truth['temporary'])
  expected_time=before['ego']['calibrate_data'][truth['slot']]['time_delay'];require(ev['time']==descriptor(expected_time),'TIME_DELAY_MAPPING')
 with np.load(path/'arrays.npz',allow_pickle=False) as z:
  a={k:z[k] for k in z.files};verify_attention(a,meta)
  for item,truth in zip(evidence['metaflow_events'],expected):require(digest(a[f"metaflow_slot_{truth['slot']}"])==item['output_hash'],'OFFSET_AGENT_OUTPUT_IDENTITY')
  pp=build_postprocessor(config()['postprocess'],False);verify_heads(a,meta,before,pp)
  slim={k:a[k].copy() for k in ('cls_preds','reg_preds','dir_preds','pred_offset','reference_full','final_corners','final_scores','native_nms_keep','gt_tp50','gt_tp70')}
 return {'condition':meta['condition'],'mask_installed':meta['mask_installed'],'input_original_leaves_verified':len(meta['input_leaf_before']['leaves']),'actual_native_history_verified':True,'attention_verified':True},slim

def validate(root):
 root=Path(root);seal=load(R/'HISTORY_SOURCE_SEAL.json')
 for path,h in seal['files'].items():require(sha_file(path)==h,'HISTORICAL_FILE_MODIFIED '+path)
 require(sha_file(CKPT)==CH,'CHECKPOINT_SHA');state=checkpoint();owner=HistoryOwner(state,fusion=False)
 references={};native_tests=[]
 with NativeCallTrace() as trace:
  for row in load(FROZEN)['samples']:
   cpu_guard();batch,truth=native_history_reference(owner,row);native_tests.append({'sample_idx':row['sample_idx'],'two_remote_inputs_actual':True,'unrepaired_and_disposable_features_exact':True,'feature_hashes':[x['feature_hash'] for x in truth]})
   if row['sample_idx']==324:references[324]=(batch,truth)
   else:del batch,truth
   gc.collect()
 ledger=load(root/'ledger.json');require(ledger['real_forward_increment']==0 and len(ledger['records'])==5,'CPU_RECORD_COMPLETENESS')
 checks=[];control=None;full=None
 for rec in ledger['records']:
  result,heads=validate_case(root,rec,state,references[324]);checks.append(result)
  if rec['condition']=='F' and not rec['mask_installed']:control=heads
  if rec['condition']=='F' and rec['mask_installed']:full=heads
 require(control is not None and full is not None,'FULL_CONTROL_MISSING')
 for key in control:require(np.array_equal(control[key],full[key]),'FULL_UNHOOKED_EXACT_'+key)
 old=load(OLD/'RUN_BUDGET.json');real=load(OLD/'REAL_RUN_MANIFEST.json')
 require(len(old['attempts'])==1 and old['attempts'][0]['status']=='MODEL_RETURNED' and real['real_forwards']==real['forward_entered_count']==real['forward_returned_count']==1 and real['committed_conditions']==0,'ORIGINAL_BUDGET_STATE')
 # New run is not still holding a lock. This opens the existing file read-only.
 with (root/'run.lock').open('rb') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
 cpu_guard()
 return {'passed':True,'state':'CPU_REPAIR_CORE_VALIDATED','science_result_scope':'CPU_MOCK_ONLY_NO_REAL_DETECTION_CONCLUSION','checks':checks,'native_history_samples':native_tests,'independent_native_calls':dict(trace.counts),'Full_exact':True,'history_original_values_unchanged':True,'all_original_fields_checked':True,'historical_files_rehashed':len(seal['files']),'real_forward_increment':0,**cpu_only.evidence()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);args=p.parse_args()
 try:out=validate(args.root);write(args.output,out,exclusive=True);print(json.dumps(out,ensure_ascii=False),flush=True)
 except BaseException as e:write(args.output,{'passed':False,'error':str(e),'traceback':traceback.format_exc(),**cpu_only.evidence()},exclusive=True);raise
