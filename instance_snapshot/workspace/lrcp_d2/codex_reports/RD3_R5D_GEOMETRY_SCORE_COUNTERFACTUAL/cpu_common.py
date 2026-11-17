"""R5D common safety, identity and pure arithmetic primitives. No model imports.
The producer and validator implement separate postprocessing loops.
"""
import os,sys,json,csv,hashlib,importlib.abc,traceback,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPORTS=ROOT.parent
CROOT=REPORTS/'RD3_R5C_STAGE_MECHANISM_AUDIT';AROOT=REPORTS/'RD3_R5A_EVIDENCE_AUDIT';BROOT=REPORTS/'RD3_R5B_SAMPLE_UTILITY'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
DENIED=[]
class NoDeviceModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):DENIED.append(fullname);raise ImportError('CPU_OFFLINE_ONLY '+fullname)
sys.meta_path.insert(0,NoDeviceModel())
def allowed(p):
 if isinstance(p,int):return True
 try:return Path(os.fsdecode(p)).resolve().is_relative_to(ROOT)
 except (ValueError,TypeError):return False
def guard(event,args):
 if event=='subprocess.Popen':DENIED.append(event);raise RuntimeError('NO_SUBPROCESS_MODEL_BYPASS')
 if event=='open':
  p,m,f=args;wr=(isinstance(m,str) and any(x in m for x in 'wax+')) or isinstance(f,int) and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
  if wr and not allowed(p):DENIED.append(str(p));raise PermissionError('R5D_HISTORY_WRITE_FORBIDDEN')
 if event in ('os.remove','os.mkdir','os.rmdir','os.chmod','os.utime') and not allowed(args[0]):DENIED.append(event);raise PermissionError('R5D_OUTSIDE_WRITE')
 if event=='ctypes.dlopen' and args[0] and any(x in str(args[0]).lower() for x in ['libcuda','libcudart','libtorch','ppu_sdk','libalinpu']):DENIED.append(event);raise RuntimeError('NO_DEVICE_LIBRARY')
sys.addaudithook(guard)
import numpy as np
from source_reference.offline_science import matrix,polys,overlap,nms_replay,unique_match,STAGES,canonical_gt_ids
from source_reference.offline_storage_reader import digest,load_storage_archive

def require(ok,n):
 if not bool(ok):raise RuntimeError(n)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def j(x):return json.dumps(x,ensure_ascii=False,allow_nan=False)
def table(n,rr):
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with (ROOT/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def same(a,b,n,tol=0):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape,n+' SHAPE');require(np.isfinite(a).all() and np.isfinite(b).all(),n+' FINITE');error=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0));require(error<=tol,n+' DIFF '+str(error));return error
PAIRS=load(ROOT/'R5D_DESIGN_FROZEN.json')['pairs'];COMBOS=['G0_S0','G0_S1','G1_S0','G1_S1']
CONFIG=load(CROOT/'R5C_C0_REPLAY_VALIDATION.json')['frozen_postprocess_config']
GM={(int(r['sample_idx']),int(r['gt_idx'])):r for r in rows(AROOT/'R5A_GT_IDENTITY_MAP.csv')}
LOC={(int(r['sample_idx']),r['condition']):r for r in rows(BROOT/'R5B_RAW_SHARD_LOCATOR.csv')}
FOCAL={725:['14014','4071'],324:['29118']}
SOURCE_FIELDS=['raw_corners','scores_all','decoded_before_dir','reg_preds','dir_preds','cls_preds','GT_canonical','IoU_GT_raw','score_mask','large_mask','z_mask','quality_mask','native_nms_keep','range_mask','final_corners','final_scores','suppressor','suppressor_overlap','first_deletion_code']+['ids_'+s for s in STAGES]+[x+str(t) for x in ['gt_tp','prediction_gt_idx','prediction_fp'] for t in (50,70)]
def source_keys():return sorted({(p['sample_idx'],p[c]) for p in PAIRS for c in ['condition0','condition1']})
def source_path(sid,c):return ROOT/'evidence/source'/f'sample_{sid}_{c}.npz'
def run_path(run_id):return ROOT/'evidence/runs'/(run_id+'.npz')
def load_sources(export=False):
 out={};meta={};frozen=load(REPORTS/'RD3_MESSAGE_UTILITY_PILOT/FROZEN_MANIFEST.json');batch={};proof={};initial=load(ROOT/'R5D_INITIAL_SOURCE_VALIDATION.json')
 for p,h in initial['source_files'].items():require(sha(p)==h,'SOURCE_SHA '+p)
 require(sha(ROOT/'R5D_DESIGN_FROZEN.json')==(ROOT/'R5D_DESIGN_SHA256.txt').read_text().split()[0],'FROZEN_DESIGN_SHA')
 import yaml
 actual=next(x for x in load(CROOT/'R5C_NATIVE_PIPELINE_CONTRACT.json')['actual_sources'] if x['kind']=='frozen_config');tree=yaml.compose(Path(actual['absolute_path']).read_text());post=next(v for k,v in tree.value if k.value=='postprocess');require(CONFIG==yaml.safe_load(yaml.serialize(post)),'ACTUAL_FROZEN_CONFIG')
 for sid,c in source_keys():
  loc=LOC[sid,c];require(sha(loc['absolute_path'])==loc['sha256'],'NPZ_SHA')
  with np.load(loc['absolute_path'],allow_pickle=False) as z:a={k:z[k].copy() for k in SOURCE_FIELDS};require('iou_preds' not in z.files,'NO_LEARNED_QUALITY_BRANCH')
  parent=Path(loc['absolute_path']).parent;m=load(parent/'meta.json');mf=load(parent/'manifest.json');require(mf['state']=='ACCEPTED' and sha(parent/'meta.json')==mf['files']['meta.json'] and loc['sha256']==mf['files']['arrays.npz'],'ORIGINAL_ACCEPTED_META_SHA');meta[sid,c]=m
  require(m['sample_idx']==sid and m['condition']==c and m['frozen_sha256']=='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c' and m['checkpoint_sha256']=='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724','SOURCE_IDENTITY')
  rec=next(x for x in frozen['samples'] if x['sample_idx']==sid)
  if sid not in batch:
   require(sha(rec['input']['path'])==rec['input']['sha256'],'FROZEN_BATCH_SHA');payload,_=load_storage_archive(rec['input']['path']);batch[sid]=payload['batch'];anchor=np.asarray(batch[sid]['ego']['anchor_box']);trans=np.asarray(batch[sid]['ego']['transformation_matrix']);require(anchor.shape==(80,80,2,7) and trans.shape==(4,4),'FROZEN_ANCHOR_COORD_SHAPES');proof[str(sid)]={'frozen_input_path':rec['input']['path'],'frozen_input_sha256':rec['input']['sha256'],'anchor_shape':list(anchor.shape),'anchor_sha256':digest(anchor),'coordinate_transformation_sha256':digest(trans),'pairwise_t_matrix_frozen_sha':rec['pairwise_hash'],'ego_physical_id':rec['ego_physical_id'],'remote_physical_ids':rec['remote_physical_ids'],'GT_count':rec['GT_count']}
  require(a['raw_corners'].shape==(12800,8,3) and a['raw_corners'].dtype==np.float32 and a['scores_all'].shape==(12800,) and a['scores_all'].dtype==np.float32,'FULL_FLOAT32_G_S');same(a['ids_raw_decoded'],np.arange(12800),'ANCHOR_IDS_ONE_TO_ONE');require(a['cls_preds'].shape==(1,2,80,80) and a['reg_preds'].shape==(1,14,80,80) and a['dir_preds'].shape==(1,4,80,80),'HEAD_GRID_SHAPES')
  physical,err=canonical_gt_ids(a['GT_canonical'],batch[sid],CONFIG['order']);require([str(x) for x in physical]==[GM[sid,gi]['physical_object_id'] for gi in range(len(a['GT_canonical']))],'FROZEN_GT_OBJECT_ID_BIJECTION')
  for gi,g in enumerate(a['GT_canonical']):require(digest(g)==GM[sid,gi]['GT_hash'],'GT_GEOMETRY_HASH')
  require(np.all(np.isfinite(a['raw_corners'])) and np.all(np.isfinite(a['scores_all'])),'FINITE_SOURCES')
  if export:
   with source_path(sid,c).open('wb') as f:np.savez_compressed(f,**a)
  for v in a.values():v.setflags(write=False)
  out[sid,c]=a
 for p in PAIRS:
  sid=p['sample_idx'];b=out[sid,p['condition0']];a=out[sid,p['condition1']];bm=meta[sid,p['condition0']];am=meta[sid,p['condition1']]
  same(b['ids_raw_decoded'],a['ids_raw_decoded'],'PAIR_CANDIDATE_ORDER');same(b['GT_canonical'],a['GT_canonical'],'PAIR_GT_COORDINATES');require(bm['input_manifest']==am['input_manifest'] and bm['input_hash']==am['input_hash'] and bm['slot_physical_ids']==am['slot_physical_ids'],'PAIR_INPUT_SLOT_COORD_IDENTITY')
  proof[p['pair_id']]={'all_candidate_IDs_equal':True,'head_grid':[80,80,2],'source_input_hash':bm['input_hash'],'fixed_slot_physical_ids':bm['slot_physical_ids'],'same_GT':True,'same_coordinate_frame':True,'same_frozen_config_sha256':actual['sha256']}
 if export:write('R5D_INPUT_ALIGNMENT_VALIDATION.json',proof)
 return out,meta,proof

def counts(a):
 result={'final_prediction_count':len(a['final_scores']),'GT_count':len(a['GT_canonical'])}
 for t in (50,70):
  tp=int(a[f'gt_tp{t}'].sum());result.update({f'TP{t}':tp,f'FP{t}':len(a['final_scores'])-tp,f'FN{t}':len(a['GT_canonical'])-tp})
 return result

def ci_state(a,cid,t):
 pp=np.flatnonzero(a['ids_final_range']==cid)
 if not len(pp):return 'ABSENT_FINAL',None,None
 pi=int(pp[0]);gi=int(a[f'prediction_gt_idx{t}'][pi]);return ('TP' if gi>=0 else 'FP'),pi,gi if gi>=0 else None

def GT_info(a,sid,gi,t):
 pp=np.flatnonzero(a[f'prediction_gt_idx{t}']==gi);iv=a['IoU_GT_raw'][gi];high=iv>=t/100;members={s:a['ids_'+s][high[a['ids_'+s]]].astype(int).tolist() for s in STAGES}
 if len(pp):pi=int(pp[0]);cid=int(a['ids_final_range'][pi]);match={'TP_FN':'TP','matched_prediction_index':pi,'matched_candidate_id':cid,'matched_IoU':float(iv[cid]),'matched_score':float(a['scores_all'][cid])}
 else:match={'TP_FN':'FN','matched_prediction_index':None,'matched_candidate_id':None,'matched_IoU':None,'matched_score':None}
 first=next((s for s in STAGES if not members[s]),None);bottleneck='DETECTED' if len(pp) else 'GT_MATCH_COMPETITION' if members['final_range'] else {'raw_decoded':'RAW_GEOMETRY','score_pass':'SCORE_FILTER','quality_pass':'LARGE_Z_FILTER','nms_top1000':'TOP1000','nms_keep':'NMS','final_range':'FINAL_RANGE'}[first]
 return {**match,**{'high_count_'+s:len(v) for s,v in members.items()},**{'high_ids_'+s:j(v) for s,v in members.items()},'bottleneck':bottleneck,'first_empty_stage':first,'raw_max_IoU':float(iv.max()),'raw_max_IoU_id':int(np.argmax(iv)),'final_max_IoU':float(iv[a['ids_final_range']].max()) if len(a['ids_final_range']) else None,'all_preNMS_high_suppressed':bool(members['nms_top1000']) and not members['nms_keep']}

def sealed_history_unchanged():
 protected=load(ROOT/'R5D_HISTORY_READ_ONLY_SEAL.json')['files'];bad=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];require(not bad,'HISTORY_CHANGED '+j(bad));return len(protected)
