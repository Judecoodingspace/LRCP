"""Independent CPU evidence verifier. No model construction or forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,argparse,traceback
# Independently recompute native decode, NMS suppression and unique GT matching.
from INDEPENDENT_VALIDATOR import verify_heads,same
from backend import configuration
from opencood.data_utils.post_processor import build_postprocessor

def verify_attention(a,meta):
 keep=CONDITIONS[meta['condition']];require(meta['keep']==keep,'MASK_IDENTITY')
 q=QUERIES;ref=a['reference_full'];require(ref.shape==(1,6400,9,2),'FULL_REFERENCE_SHAPE')
 for slot in (1,2):
  off=a[f'metaflow_slot_{slot}'];require(off.shape==(1,80,80,2),'OFFSET_SHAPE')
  same(ref[:,:,slot*3:(slot+1)*3,:],np.broadcast_to(ref[:,:,:1,:]+off.reshape(1,6400,1,2)/np.array([80,80],np.float32),(1,6400,3,2)),'METAFLOW_SLOT_FORMULA',1e-6)
  same(a['pred_offset'][:,:,slot-1,:],off.reshape(1,6400,2),'OFFSET_SLOT_IDENTITY')
 for i in range(4):
  w=a[f'cross_{i}_full_weights'];require(w.ndim==5 and w.shape[:2]==(1,6400) and w.shape[3]==9,'FULL_WEIGHTS_SHAPE')
  require(np.isfinite(w).all() and (w>=0).all(),'NONFINITE_OR_NEGATIVE_WEIGHT')
  same(w.sum((-1,-2)),np.ones(w.shape[:3],np.float32),'GLOBAL_SUM',1e-5)
  for slot,enabled in enumerate(keep):
   if not enabled:require(np.count_nonzero(w[:,:,:,slot*3:(slot+1)*3,:])==0,'BLOCKED_WEIGHT_NONZERO')
  heads,points=w.shape[2],w.shape[-1]
  raw=a[f'cross_{i}_raw_logits_selected'];require(np.isfinite(raw).all(),'RAW_LOGITS_NONFINITE')
  v=raw.reshape(1,len(q),heads,5,3,points)[:,:,:,:3,:,:].copy()
  for slot,enabled in enumerate(keep):
   if not enabled:v[:,:,:,slot,:,:]=-np.inf
  flat=v.reshape(1,len(q),heads,-1);ex=np.exp(flat-flat.max(-1,keepdims=True));expect=(ex/ex.sum(-1,keepdims=True)).reshape(1,len(q),heads,9,points)
  same(w[:,q],expect,'MASKED_SOFTMAX_RECOMPUTE',1e-6);same(a[f'cross_{i}_weights'],w[:,q],'SELECTED_WEIGHTS')
  sp=a[f'cross_{i}_spatial_shapes'];same(sp,np.tile(np.array([[80,80],[40,40],[20,20]]),(3,1)),'SPATIAL_ORDER')
  r=a[f'cross_{i}_reference'];same(r,ref[:,q],'REFERENCE_UNCHANGED')
  off=a[f'cross_{i}_offset'];proj=a[f'cross_{i}_sampling_offsets_output'];same(off,proj.reshape(1,len(q),heads,15,points,2)[:,:,:,:9],'OFFSET_PROJECTION')
  same(a[f'cross_{i}_location'],r[:,:,None,:,None,:]+off/sp[:,::-1].astype(np.float32)[None,None,None,:,None,:],'LOCATION_RECOMPUTE',1e-6)
 return True

def historical(a,row):
 rec=row['historical_full'];require(sha_file(rec['path'])==rec['sha256'] and sha_file(rec['meta_path'])==rec['meta_sha256'],'P4_HISTORY_HASH')
 with np.load(rec['path'],allow_pickle=False) as old:
  for k,tol in [('cls_preds',1e-6),('reg_preds',1e-5),('dir_preds',1e-5),('final_corners',1e-5),('final_scores',1e-6),('native_nms_keep',0)]:same(a[k],old[k],'NATIVE_FULL_'+k,tol)
  for k in ('score_mask','quality_mask','ids_score_pass','ids_quality_pass','ids_nms_top1000','ids_nms_keep','ids_final_range','range_mask'):same(a[k],old[k],'NATIVE_STAGE_'+k)
 path=BASE/f"P3A_20261014_120929_boxes/sample_{row['sample_idx']}_alpha_1.0.npz"
 with np.load(path,allow_pickle=False) as old:
  same(a['final_corners'],old['pred_boxes'],'P3A_BOX',1e-5);same(a['final_scores'],old['pred_scores'],'P3A_SCORE',1e-6);same(a['GT_canonical'],old['gt_boxes'],'P3A_GT',1e-5)
  from c1_reused import eval_native
  for t in (.5,.7):
   _,match,flags=eval_native(old['pred_boxes'],old['pred_scores'],old['gt_boxes'],t);same(a[f'gt_tp{int(t*100)}'],flags,'P3A_MATCH');same(a[f'prediction_gt_idx{int(t*100)}'],match,'P3A_PRED_MATCH')
 return True

def record(root,rec,mock=False,control=False):
 root=Path(root);p=root/rec['relative_path'];require(p.parent==root/'capture','SHARD_PATH')
 require(sha_file(p/'manifest.json')==rec['manifest_sha256'],'SHARD_MANIFEST_HASH');m=load(p/'manifest.json')
 for name,hash_ in m['files'].items():require(name in ('arrays.npz','meta.json') and sha_file(p/name)==hash_,'SHARD_FILE_HASH')
 meta=load(p/'meta.json');require(meta['backend']==('mock' if mock else 'real') and meta['schema']=='RD3_CAPTURE_V1','BACKEND_SCHEMA')
 frozen=load(R/'FROZEN_MANIFEST.json');row=next(x for x in frozen['samples'] if x['sample_idx']==rec['sample_idx'])
 require(meta['sample_idx']==rec['sample_idx'] and meta['condition']==rec['condition'],'RECORD_IDENTITY')
 require(meta['frozen_sha256']==sha_file(R/'FROZEN_MANIFEST.json') and meta['input_manifest']==row and meta['input_hash']==row['input']['input_hash'] and meta['slot_physical_ids']==['1','2','3'],'INPUT_IDENTITY')
 require(meta['checkpoint_sha256']==CH,'CHECKPOINT_IDENTITY')
 if not mock:
  identity=load(C2/'CHECKPOINT_IDENTITY.json');require(meta['parameter_digest']==identity['parameter_digest'] and meta['buffer_digest']==identity['buffer_digest'],'MODEL_IDENTITY')
 item=row['input'];require(sha_file(item['path'])==item['sha256'],'BATCH_SHA');payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False)
 require(digest(payload['batch'])==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'BATCH_SEMANTICS')
 pp=build_postprocessor(configuration()['postprocess'],False)
 with np.load(p/'arrays.npz',allow_pickle=False) as z:
  a={k:z[k] for k in z.files};verify_attention(a,meta);verify_heads(a,meta,payload['batch'],pp)
  if control and not mock:historical(a,row)
  with np.load(BASE/f"P3A_20261014_120929_boxes/sample_{row['sample_idx']}_alpha_1.0.npz",allow_pickle=False) as old:same(a['GT_canonical'],old['gt_boxes'],'SAME_GT',1e-5)
 cpu_guard();return {'sample_idx':rec['sample_idx'],'condition':rec['condition'],'independently_recomputed':True,'historical_control':control and not mock}

def validate(root,mock=False):
 root=Path(root);manifest=load(root/'manifest.json');require(manifest['backend']==('mock' if mock else 'real'),'RUN_KIND')
 records=manifest['records'];frozen=load(R/'FROZEN_MANIFEST.json');expected=[(x['sample_idx'],c) for x in frozen['samples'] for c in CONDITIONS]
 if not mock:require([(r['sample_idx'],r['condition']) for r in records]==expected,'FOUR_GROUP_COMPLETENESS')
 checks=[record(root,r,mock,r['condition']=='F') for r in records]
 for sid in {r['sample_idx'] for r in records}:
  group=[r for r in records if r['sample_idx']==sid];base=None
  for r in group:
   p=root/r['relative_path'];meta=load(p/'meta.json')
   with np.load(p/'arrays.npz',allow_pickle=False) as z:
    val={k:z[k].copy() for k in ('reference_full','metaflow_slot_1','metaflow_slot_2','pred_offset','offset_GT','GT_canonical')}
   if base is None:base=(meta,val)
   else:
    require(meta['feature_hashes']==base[0]['feature_hashes'],'CROSS_CONDITION_FEATURE_MUTATION')
    for k in val:same(val[k],base[1][k],'CROSS_CONDITION_'+k)
 return {'passed':True,'kind':'MOCK_ONLY' if mock else 'REAL_CPU_SCIENTIFIC_ACCEPTED','validated_records':len(checks),'checks':checks,'CUDA_initialized':False,'real_forwards':0,'limitations':['Full weights independently checked for every query/head/level/point; softmax and location recomputed on five prespecified spatial queries.','Input/feature and runtime model hashes are producer evidence checked against frozen/checkpoint identities; no second model forward used.']}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--mock',action='store_true');parser.add_argument('--record',type=int);args=parser.parse_args();root=Path(args.root)
 try:
  if args.record is not None:
   rec=load(root/'manifest.json')['records'][args.record];out={'passed':True,**record(root,rec,args.mock,True)}
  else:out=validate(root,args.mock)
  print(json.dumps(out,ensure_ascii=False));write(root/('control_'+str(args.record)+'.json' if args.record is not None else 'validation.json'),out)
 except BaseException as e:
  write(root/('control_'+str(args.record)+'.json' if args.record is not None else 'validation.json'),{'passed':False,'error':str(e),'traceback':traceback.format_exc()});raise
