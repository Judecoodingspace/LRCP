"""Independent CPU negative acceptance: verify actual UNACCEPTED evidence and
rehash corrupted copies before sending them through real independent validators.
Never trusts producer's fault passed/rejected booleans.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_only
cpu_only.install()
from rd3_common import *
import torch,copy,fcntl,subprocess,traceback,gc
from independent_cpu import validate_snapshot_against_actual,native_history_reference,validate_case
from mock_backend import HistoryOwner,checkpoint
from input_integrity import DisposableHistory
from attention_validation import verify_attention
from diagnostics import accepted_record

def expect_reject(fn,fragment):
 try:fn()
 except Exception as e:
  require(fragment in str(e),'WRONG_INDEPENDENT_REJECTION '+str(e));return str(e)
 raise AssertionError('NEGATIVE_ACCEPTED '+fragment)

def run():
 main=R/'formal_cpu_r2';report=load(R/'SCIENTIFIC_FAULT_TESTS.json');row=load(FROZEN)['samples'][0]
 payload=torch.load(row['input']['path'],map_location='cpu',weights_only=False);frozen=payload['batch']
 required={'voxel_features','voxel_coords','voxel_num_points','swap_histories','swap_offsets','pairwise','unknown_derived','shape','dtype','device_meta','contract_hash','post_return_values','unexpected_label','invalid_label_offset','nonzero_blocked_attention'}
 require({x['fault'] for x in report['faults']}==required,'FAULT_SET')
 diagnostics=[]
 for item in report['faults']:
  name=item['fault']
  if name=='nonzero_blocked_attention':continue
  p=R/item['diagnostic_relative'];m=load(p/'manifest.json');d=load(p/'diagnostic.json')
  require(p.parent.name=='diagnostics' and d['state']==m['state']=='UNACCEPTED' and d['science_accepted'] is False,'DIAGNOSTIC_SCOPE')
  for file,h in m['files'].items():require(sha_file(p/file)==h,'DIAGNOSTIC_FILE_HASH')
  require(d['attempt']['sample_idx']==324 and d['attempt']['kind']=='CPU_MOCK_FAULT','ATTEMPT_IDENTITY')
  original=copy.deepcopy(d['before'])
  if name=='contract_hash':
   good=digest({'nodes':original['nodes'],'leaves':original['leaves']});require(original['hash']!=good,'HASH_FAULT_NOT_ACTUAL');original['hash']=good
  validate_snapshot_against_actual(original,frozen)
  after=d['after']['leaves'];base=d['before']['leaves'];prefix='/ego/calibrate_data/1/lidar_history/0/'
  if name in ('voxel_features','voxel_coords','voxel_num_points'):
   path=prefix+name;require(base[path]['value_hash']!=after[path]['value_hash'] and base[path]['shape']==after[path]['shape'],'ACTUAL_VALUE_FAULT')
  if name=='pairwise':require(base['/ego/pairwise_t_matrix']['value_hash']!=after['/ego/pairwise_t_matrix']['value_hash'],'ACTUAL_PAIRWISE_FAULT')
  if name in ('shape','dtype','device_meta'):
   field={'shape':'shape','dtype':'dtype','device_meta':'device'}[name];require(base[prefix+'voxel_features'][field]!=after[prefix+'voxel_features'][field],'ACTUAL_METADATA_FAULT')
  if name=='unknown_derived':require('/unapproved_cache' in d['extra']['temporary_histories'][0]['leaves'],'ACTUAL_UNKNOWN_FIELD')
  if name=='post_return_values':require(base[prefix+'voxel_features']['value_hash']!=after[prefix+'voxel_features']['value_hash'] and d['model_returned'],'RETURNED_INPUT_FAULT')
  if name=='unexpected_label':require('/ego/label_dict/surprise' in after and d['model_returned'],'UNEXPECTED_GLOBAL_FIELD')
  with np.load(p/'arrays.npz',allow_pickle=False) as z:
   for key,info in d['array_inventory'].items():require(key in z and digest(z[key])==info['value_hash'] and list(z[key].shape)==info['shape'] and str(z[key].dtype)==info['dtype'],'DIAGNOSTIC_ARRAY_HASH')
   if d['model_returned']:require(all(k in z for k in ('cls_preds','reg_preds','dir_preds','pred_offset')),'RETURNED_ARRAYS_NOT_RECOVERABLE')
   if name=='swap_offsets':
    events=d['extra']['metaflow_events'];pred=z['pred_offset'];require(digest(pred[:,:,0,:].reshape(1,80,80,2))!=events[0]['output_hash'] and digest(pred[:,:,1,:].reshape(1,80,80,2))!=events[1]['output_hash'],'OFFSET_SWAP_NOT_ACTUAL')
  require(not list((p.parent.parent/'capture').glob('*')),'DIAGNOSTIC_BECAME_CAPTURE')
  expect_reject(lambda:accepted_record(p.parent.parent,str(p.relative_to(p.parent.parent))),'DIAGNOSTIC_NOT_ACCEPTED_SCIENCE')
  diagnostics.append({'fault':name,'unaccepted_files_and_arrays_verified':True,'model_returned':d['model_returned']})
 # Full checks on independently rehashed corrupt accepted-looking copies.
 state=checkpoint();owner=HistoryOwner(state,fusion=False);reference=native_history_reference(owner,row)
 source=main/'capture/G-B';source_rec=next(x for x in load(main/'ledger.json')['records'] if x['condition']=='G-B')
 badroot=R/'independent_corruptions';(badroot/'capture').mkdir(parents=True)
 write(badroot/'claim.json',load(main/'claim.json'))
 corrupted=[]
 for name in ('leaf_values','wrong_scope_whitelist','blocked_weight','contract_hash'):
  p=badroot/'capture'/name;p.mkdir();meta=load(source/'meta.json');after=torch.load(source/'inputs_after.pt',map_location='cpu',weights_only=False);meta['passed']=True
  if name=='leaf_values':after['ego']['calibrate_data'][1]['lidar_history'][0]['voxel_features'].view(-1)[0]+=1
  if name=='wrong_scope_whitelist':after['ego']['calibrate_data'][1]['pillar_features']=torch.zeros(1)
  if name=='contract_hash':meta['input_leaf_before']['hash']='broken'
  if name=='blocked_weight':
   with np.load(source/'arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
   w=a['cross_0_full_weights'];idx=np.unravel_index(np.argmax(w[0,0,0,:3]),w[0,0,0,:3].shape);amount=float(w[0,0,0,idx[0],idx[1]])*.5;w[0,0,0,idx[0],idx[1]]-=amount;w[0,0,0,3,0]=amount
   with (p/'arrays.npz').open('xb') as f:np.savez_compressed(f,**a)
   del a
  else:os.link(source/'arrays.npz',p/'arrays.npz')
  torch.save(after,p/'inputs_after.pt');write(p/'meta.json',meta)
  write(p/'manifest.json',{'kind':'MOCK_ONLY_CPU_R1','files':{file:sha_file(p/file) for file in ('arrays.npz','inputs_after.pt','meta.json')}})
  rec={**source_rec,'relative_path':str(p.relative_to(badroot)),'manifest_sha256':sha_file(p/'manifest.json')}
  reason={'leaf_values':'INPUT_VALUES','wrong_scope_whitelist':'ORIGINAL_FIELD_NAMES_ORDER','blocked_weight':'BLOCKED_WEIGHT_NONZERO','contract_hash':'LEAF_CONTRACT_HASH'}[name]
  err=expect_reject(lambda:validate_case(badroot,rec,state,reference),reason);corrupted.append({'fault':name,'all_file_hashes_recomputed':True,'producer_passed_true_ignored':True,'independent_error':err})
  del after;gc.collect()
 # Explicit same-value/provenance case: native caller must use the exact history
 # output object for the slot, even if another feature list has equal values.
 batch=copy.deepcopy(frozen)
 with DisposableHistory(owner,batch) as ctx:
  features=owner.get_batch_pillar_features(batch['ego']['calibrate_data'][1]['lidar_history'],batch['ego']['pairwise_t_matrix'],1,[0])
  identical=[x.clone() for x in features];require(digest(identical)==digest(features),'IDENTICAL_TEST_SETUP')
  provenance_error=expect_reject(lambda:owner.meta_flow(identical,batch['ego']['calibrate_data'][1]['time_delay']),'METAFLOW_AGENT_INPUT_MAPPING')
 # Physical B/C exchange independently rejected at original method boundary.
 batch=copy.deepcopy(frozen)
 with DisposableHistory(owner,batch):
  swap_error=expect_reject(lambda:owner.get_batch_pillar_features(batch['ego']['calibrate_data'][2]['lidar_history'],batch['ego']['pairwise_t_matrix'],1,[0]),'HISTORY_AGENT_INPUT_MISMATCH')
 # Verify old budget and historical lock bytes, no mutation/lock creation there.
 seal=load(R/'HISTORY_SOURCE_SEAL.json')
 for path,h in seal['files'].items():require(sha_file(path)==h,'HISTORY_CHANGED '+path)
 old=load(OLD/'RUN_BUDGET.json');real=load(OLD/'REAL_RUN_MANIFEST.json');require(len(old['attempts'])==1 and real['forward_entered_count']==real['forward_returned_count']==1 and real['committed_conditions']==0,'PRIOR_BUDGET')
 fr=load(FROZEN);require(sha_file(MODEL/'config.yaml')==fr['configuration_sha256'] and sha_file('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl')==fr['raw_index_sha256'],'FROZEN_CONFIG_OR_INDEX')
 with (main/'run.lock').open('rb') as file:
  fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
  code="import fcntl,sys; f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
  proc=subprocess.run(['/workspace/venvs/lrcp/bin/python','-B','-c',code,str(main/'run.lock')],capture_output=True,text=True)
  require(proc.returncode!=0 and 'BlockingIOError' in proc.stderr,'INDEPENDENT_LOCK_FAILURE')
 # Immutable old lock/claim/budget files are all included in the historical seal.
 cpu_guard();out={'passed':True,'diagnostic_cases':diagnostics,'rehash_corruption_cases':corrupted,'identical_values_wrong_provenance_rejected':provenance_error,'B_C_history_exchange_rejected':swap_error,'cross_process_lock_independently_rejected':True,'historical_hashes_preserved':len(seal['files']),'original_real_budget':{'reserved':1,'entered':1,'returned':1,'committed':0},'real_forward_increment':0,**cpu_only.evidence()}
 write(R/'INDEPENDENT_FAULT_VALIDATION.json',out,exclusive=True);print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'INDEPENDENT_FAULT_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),**cpu_only.evidence()});raise
