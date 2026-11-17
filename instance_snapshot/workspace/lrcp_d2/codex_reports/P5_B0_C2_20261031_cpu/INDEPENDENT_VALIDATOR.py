"""CPU validator, independently recomputes saved evidence; never runs model.

Current C1 schema lacks independently persisted parameter/online instrumentation
proof. The command reports INCOMPLETE instead of granting scientific acceptance.
"""
import os,sys,json,csv,hashlib,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
from pilot_common import MODEL,sha_file,digest,iou_matrix,STAGES
from contracts import attribute_gt
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
from types import SimpleNamespace
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P4_C2_20261027_233036_301795')
from c1_reused import postprocess,eval_native

R=Path(__file__).resolve().parent;P5=R.parent/'P5_A_CPU_FIX_20261028_storage';FULL=R.parent/'P4_FULL_PREP_20261028_113714_782940'
CHECKPOINT=MODEL/'net_epoch_bestval_at19.pth';CH='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'
class ValidationError(RuntimeError):pass

def require(ok,reason):
 if not bool(ok):raise ValidationError(reason)
def same(a,b,reason,atol=0):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape,reason+'_SHAPE')
 require(np.array_equal(a,b) if atol==0 else np.allclose(a,b,rtol=0,atol=atol,equal_nan=False),reason)
def sha(p):return sha_file(p)

def verify_reference(a,meta):
 orig=a['reference_original_full'];offset=a['effective_offset_full'];active=meta['active_slots'];valid=meta['valid_slots'];L=len(active)*3+3
 require(orig.shape==(1,6400,L,2) and offset.shape==(1,6400,len(active),2),'REFERENCE_SHAPE')
 require(np.isfinite(orig).all() and np.isfinite(offset).all(),'REFERENCE_NONFINITE')
 base=orig[:,:,:1,:];same(orig[:,:,:3,:],np.broadcast_to(base,orig[:,:,:3,:].shape),'EGO_BASE')
 expected=orig.copy()
 for j,slot in enumerate(active):
  sl=slice(3*(j+1),3*(j+2))
  if slot in valid:
   shifted=base+offset[:,:,j:j+1,:]/np.array([80,80],np.float32)
   same(orig[:,:,sl,:],np.broadcast_to(shifted,orig[:,:,sl,:].shape),'SHIFT_FORMULA',1e-6)
   if meta['condition']=='core':expected[:,:,sl,:]=base
 same(a['cross_0_direct_reference'],expected,'BLOCK0_REFERENCE')
 for i in (1,2,3):same(a[f'cross_{i}_direct_reference'],orig,'LATER_REFERENCE')
 return True

def verify_attention(a,meta):
 q=meta['query_ids'];active=meta['active_slots'];n=3*(len(active)+1);require(len(q)>0 and len(q)==len(set(q)) and all(0<=x<6400 for x in q),'QUERY_IDS')
 summaries=[]
 for block in range(4):
  ref=a[f'cross_{block}_reference'];direct=a[f'cross_{block}_direct_reference'];off=a[f'cross_{block}_offset'];loc=a[f'cross_{block}_location'];w=a[f'cross_{block}_weights'];sp=a[f'cross_{block}_spatial_shapes']
  same(sp,np.tile(np.array([[80,80],[40,40],[20,20]],np.int64),(len(active)+1,1)),'SPATIAL_SHAPES')
  require(ref.shape==(1,len(q),n,2) and off.shape[:2]==(1,len(q)) and off.ndim==6 and off.shape[3]==n and off.shape[-1]==2 and w.shape==off.shape[:-1] and loc.shape==off.shape,'ATTENTION_SHAPE')
  same(ref,direct[:,q,:,:],'LOCAL_REFERENCE');require(all(np.isfinite(x).all() for x in (ref,off,loc,w)),'ATTENTION_NONFINITE')
  norm=sp[:,::-1].astype(np.float32)
  same(loc,ref[:,:,None,:,None,:]+off/norm[None,None,None,:,None,:],'LOCATION_FORMULA',1e-6)
  same(w.sum(axis=(-1,-2)),np.ones_like(w.sum(axis=(-1,-2))),'WEIGHT_SUM',1e-5)
  projected=a[f'cross_{block}_sampling_offsets_output'];heads=off.shape[2];points=off.shape[4]
  require(projected.shape==(1,len(q),heads*15*points*2),'OFFSET_PROJECTION_SHAPE')
  same(off,projected.reshape(1,len(q),heads,15,points,2)[:,:,:,:n,:,:],'OFFSET_PROJECTION')
  logits=a[f'cross_{block}_attention_weights_output'];require(logits.shape==(1,len(q),heads*15*points),'WEIGHT_PROJECTION_SHAPE')
  v=logits.reshape(1,len(q),heads,15,points)[:,:,:,:n,:].reshape(1,len(q),heads,n*points);v=v-v.max(-1,keepdims=True);s=np.exp(v);s=(s/s.sum(-1,keepdims=True)).reshape(w.shape)
  same(w,s,'WEIGHT_SOFTMAX',1e-6)
  for level,(height,width) in enumerate(sp):
   v=loc[:,:,:,level,:,:];outside=((v<0)|(v>1)).any(-1);xx=v[...,0]*width-.5;yy=v[...,1]*height-.5;unsupported=(xx<=-1)|(xx>=width)|(yy<=-1)|(yy>=height)
   summaries.append({'block':block,'level':level,'recorded_query_points':int(outside.size),'normalized_outside':int(outside.sum()),'kernel_unsupported':int(unsupported.sum())})
 out=a['block0_output_selected'];require(out.shape==(1,len(q),128) and np.isfinite(out).all(),'BLOCK0_OUTPUT')
 return summaries

def verify_identity(meta,sid,selected,items):
 item=items[sid];row=selected[sid]
 require(int(meta['sample_idx'])==sid and str(meta['scene_id'])==str(row['scene_id']),'SAMPLE_IDENTITY')
 require(meta['input_hash']==item['input_hash'] and meta['checkpoint_sha256']==CH,'IDENTITY_HASH')
 require(meta['slots']==item['slots'],'PHYSICAL_SLOT_MANIFEST')
 keys=list(item['slots']);require(meta['active_slots']==keys[:len(meta['active_slots'])],'ACTIVE_SLOT_ORDER')
 valid=[k for k in keys if any(item['slots'][k]['valid_mask'])];require(meta['valid_slots']==valid,'VALID_SLOT_SET')
 for k in valid:
  slot=item['slots'][k];require(all(t==300 for t in slot['time_delay']) and all(x is not None for x in slot['physical_cav_ids']),'TIME_OR_CAV')

def verify_heads(a,meta,frozen,pp):
 geom={k:a[k].copy() for k in ('cls_preds','reg_preds','dir_preds','GT_canonical')};inp={'ego':{'anchor_box':frozen['ego']['anchor_box'].clone(),'transformation_matrix':frozen['ego']['transformation_matrix'].clone()}}
 replay,_=postprocess(pp,inp,geom,geom)
 exact=['score_mask','large_mask','z_mask','quality_mask','native_nms_keep','range_mask','ids_raw_decoded','ids_score_pass','ids_quality_pass','ids_nms_top1000','ids_nms_keep','ids_final_range','suppressor','first_deletion_code']
 for key in exact:same(a[key],replay[key],'POSTPROCESS_'+key)
 for key,tol in [('decoded_before_dir',1e-5),('scores_all',1e-6),('raw_corners',1e-5),('final_corners',1e-5),('final_scores',1e-6),('suppressor_overlap',1e-6)]:same(a[key],replay[key],'POSTPROCESS_'+key,tol)
 mat=iou_matrix(a['GT_canonical'],replay['raw_corners']);stored_mat=iou_matrix(a['GT_canonical'],a['raw_corners'])
 same(a['IoU_GT_raw'],stored_mat,'GT_RAW_IOU',1e-6);same(stored_mat,mat,'GT_RAW_REPLAY_IOU',1e-5)
 for t in (.5,.7):same(stored_mat>=t,mat>=t,'GT_RAW_THRESHOLD')
 stages=attribute_gt(stored_mat,{k:replay['ids_'+k] for k in STAGES});require(meta['gt_candidate_stages']==stages,'GT_STAGES')
 for t in (.5,.7):
  _,matched,flags=eval_native(replay['final_corners'],replay['final_scores'],a['GT_canonical'],t);i=int(t*100)
  same(a[f'gt_tp{i}'],flags,f'GT_TP{i}');same(a[f'prediction_gt_idx{i}'],matched,f'GT_MATCH{i}');same(a[f'prediction_fp{i}'],(matched<0).astype(np.int8),f'PRED_FP{i}')
  used=matched[matched>=0];require(len(used)==len(set(used.tolist())),'DUPLICATE_GT_MATCH')
 return True

def verify_historical_control(a,sid):
 with np.load(P5/f'capture/sample_{sid}/alpha_1.0.npz',allow_pickle=False) as old:
  for k,tol in [('cls_preds',1e-6),('reg_preds',1e-5),('dir_preds',1e-5),('final_corners',1e-5),('final_scores',1e-6),('native_nms_keep',0)]:same(a[k],old[k],'CONTROL_'+k,tol)
  for k in ('score_mask','quality_mask','ids_score_pass','ids_quality_pass','ids_nms_top1000','ids_nms_keep','ids_final_range','range_mask'):same(a[k],old[k],'CONTROL_STAGE_'+k)
  old_iou=iou_matrix(old['GT_canonical'],old['raw_corners'])
  for threshold in (.5,.7):same(a['IoU_GT_raw']>=threshold,old_iou>=threshold,'CONTROL_IOU_THRESHOLD')
  for i in range(4):
   for k in ('reference','offset','location','weights'):same(a[f'cross_{i}_{k}'],old[f'cross_{i}_{k}'],'CONTROL_ATTENTION',1e-6)
 return True

def validate(root,mock=False):
 root=Path(root).resolve();selection={int(x['sample_idx']):x for x in csv.DictReader((R/'P5_SAMPLE_PROPOSAL_V2.csv').open())}
 launch=json.loads((P5/'PPU_LAUNCH.json').read_text());require(sha(P5/'P5_READY.json')==launch['ready_sha256'],'HISTORICAL_READY_HASH')
 ready=json.loads((P5/'P5_READY.json').read_text());require(sha(FULL/'INPUT_MANIFEST.json')==ready['locked_files'][str(FULL/'INPUT_MANIFEST.json')],'INPUT_MANIFEST_HASH')
 items={int(x['sample_idx']):x for x in json.loads((FULL/'INPUT_MANIFEST.json').read_text())['samples']}
 require(len(selection)==18 and sha(CHECKPOINT)==CH and sha(R/'P5_SAMPLE_PROPOSAL_V2.csv')==sha(P5/'P5_SAMPLE_PROPOSAL_V2.csv'),'INPUT_OR_CHECKPOINT')
 checkpoint_identity=json.loads((R/'CHECKPOINT_IDENTITY.json').read_text());require(checkpoint_identity['checkpoint_sha256']==CH and checkpoint_identity['parameter_digest']=='469b88ac6f0d888454afa6dee14a6fea46c60628fa804d1a08171963365cb80b' and checkpoint_identity['buffer_digest']=='d008d45df0a2998d42974132e3bb84c0b15dd980f98452964ad878479ea2b11e','CHECKPOINT_IDENTITY')
 acceptance=json.loads((P5/'RESEARCH_ACCEPTED.json').read_text());require(acceptance['passed'] and sha(P5/'CPU_VALIDATION_COMPLETE.json')==acceptance['CPU_validation_sha256'] and sha(P5/'PPU_CAPTURE_COMPLETE.json')==acceptance['PPU_capture_sha256'],'HISTORICAL_ACCEPTANCE')
 cap=json.loads((P5/'PPU_CAPTURE_COMPLETE.json').read_text());require(cap['passed'] and sha(P5/'CAPTURE_MANIFEST.json')==cap['manifest_sha256'],'HISTORICAL_MANIFEST')
 history_records={(int(x['sample_idx']),int(x['alpha'])):x for x in json.loads((P5/'CAPTURE_MANIFEST.json').read_text())['records']}
 manifest=json.loads((root/'manifest.json').read_text());require(manifest['kind']==('mock' if mock else 'real'),'RUN_KIND');records=manifest['records'];budget=json.loads((root/'budget.json').read_text());expected={(s,'core') for s in selection}|{(590,'off'),(590,'pass'),(918,'pass'),(918,'off')}
 keys=[(int(x['sample_idx']),x['condition']) for x in records];require(len(keys)==22 and len(set(keys))==22 and set(keys)==expected,'RECORD_IDENTITY')
 require(budget['max']==22 and len(budget['attempts'])==22 and {(int(x['sample']),x['kind']) for x in budget['attempts']}==expected,'BUDGET')
 require([(int(x['sample']),x['kind']) for x in budget['attempts'][:4]]==[(590,'off'),(590,'pass'),(918,'pass'),(918,'off')],'CONTROL_ORDER')
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));pp=build_postprocessor(h['postprocess'],False);checks=[];controls={};transitions=[];parameter_digest=None;buffer_digest=None
 for rec in records:
  sid=int(rec['sample_idx']);kind=rec['condition'];p=Path(rec['path']).resolve();require(p.parent==root/'capture' and p.name==f"sample_{sid*3+{'off':0,'pass':1,'core':2}[kind]}",'SHARD_PATH')
  require(sha(p/'manifest.json')==rec['manifest_sha256'],'MANIFEST_HASH');m=json.loads((p/'manifest.json').read_text());require(m['sample']==sid*3+{'off':0,'pass':1,'core':2}[kind],'SHARD_ID')
  for n in ('payload.bin','meta.json'):require(sha(p/n)==m['files'][n],'FILE_HASH')
  meta=json.loads((p/'meta.json').read_text());require(meta['condition']==kind and meta['backend']==('mock' if mock else 'real'),'CONDITION');verify_identity(meta,sid,selection,items)
  item=items[sid];require(sha(item['path'])==item['sha256'],'FROZEN_FILE_HASH');payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False)
  require(digest(payload['batch'])==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'FROZEN_SEMANTICS')
  ego=payload['batch']['ego'];record_len=int(ego['record_len'][0]);require(meta['active_slots']==list(item['slots'])[:record_len-1],'ACTIVE_SLOT_COUNT')
  for key,v in ego['calibrate_data'].items():same(np.asarray(v['valid_mask'].cpu()),np.asarray(item['slots'][str(key)]['valid_mask']),'VALID_MASK')
  history_rec=history_records[(sid,1)];hp=P5/f'capture/sample_{sid}/alpha_1.0.npz';require(Path(history_rec['path'])==hp and sha(hp)==history_rec['sha256'] and sha(hp.with_suffix('.json'))==history_rec['meta_sha256'],'HISTORICAL_RECORD_HASH')
  history_meta=json.loads(hp.with_suffix('.json').read_text());require(meta['query_ids']==history_meta['query_ids'],'QUERY_ALIGNMENT')
  with np.load(p/'payload.bin',allow_pickle=False) as z:
   a={k:z[k] for k in z.files};verify_reference(a,meta);summary=verify_attention(a,meta)
   with np.load(P5/f'capture/sample_{sid}/alpha_1.0.npz',allow_pickle=False) as history:
    same(a['GT_canonical'],history['GT_canonical'],'GT_IDENTITY',1e-5)
    if kind=='core':
     _,_,old70=eval_native(history['final_corners'],history['final_scores'],history['GT_canonical'],.7)
     require(len(old70)==len(a['gt_tp70']),'GT_COUNT')
     transitions.extend({'sample_idx':sid,'gt_idx':i,'native_tp70':int(x),'intervened_tp70':int(y)} for i,(x,y) in enumerate(zip(old70,a['gt_tp70'])))
   verify_heads(a,meta,payload['batch'],pp)
   integrity=meta.get('runtime_integrity',{});require(integrity.get('parameter_before') and integrity.get('parameter_before')==integrity.get('parameter_after') and integrity.get('buffer_before')==integrity.get('buffer_after') and integrity.get('frozen_before')==item['input_hash']==integrity.get('frozen_after'),'RUNTIME_INTEGRITY')
   if parameter_digest is None:parameter_digest=integrity['parameter_before'];buffer_digest=integrity['buffer_before']
   require(integrity['parameter_before']==parameter_digest and integrity['buffer_before']==buffer_digest,'CROSS_CASE_MODEL_IDENTITY')
   if not mock:require(parameter_digest==checkpoint_identity['parameter_digest'] and buffer_digest==checkpoint_identity['buffer_digest'],'CHECKPOINT_PARAMETERS')
   require(integrity.get('online_reference')=={'all_6400_reference_formula':True,'ego_and_invalid_unchanged':True,'later_reference_unchanged':True,'cross_call_count':4},'ONLINE_EVIDENCE')
   if kind in ('off','pass'):
    if not mock:verify_historical_control(a,sid)
    controls[(sid,kind)]={k:a[k].copy() for k in ('cls_preds','reg_preds','dir_preds','pred_offset','final_corners','final_scores','gt_tp50','gt_tp70','prediction_gt_idx50','prediction_gt_idx70','native_nms_keep')}
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise ValidationError('CPU_RSS_LIMIT')
  checks.append({'sample_idx':sid,'condition':kind,'local_boundary':summary})
 for sid in (590,918):
  for k in controls[sid,'off']:same(controls[sid,'off'][k],controls[sid,'pass'][k],'CONTROL_PAIR_'+k,1e-5 if k in ('final_corners','reg_preds','dir_preds','pred_offset') else (1e-6 if k in ('final_scores','cls_preds') else 0))
 require(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')

 return {'passed':True,'state':'MOCK_VALIDATION_COMPLETE' if mock else 'CPU_VALIDATION_COMPLETE','mock_only':mock,'validated_records':len(checks),'checks':checks,'per_GT_transitions':transitions,'limitations':['sampling location and boundary independently recomputed only for recorded queries; global summaries are not independently recomputable','runtime integrity evidence captured by producer and compared with independently loaded checkpoint'],'CUDA_initialized':False,'real_model_forward':0}
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('run_directory');parser.add_argument('--mock',action='store_true');args=parser.parse_args()
 try:
  result=validate(args.run_directory,mock=args.mock);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
 except Exception as e:print(json.dumps({'passed':False,'error':type(e).__name__+': '+str(e)}));raise SystemExit(1)
