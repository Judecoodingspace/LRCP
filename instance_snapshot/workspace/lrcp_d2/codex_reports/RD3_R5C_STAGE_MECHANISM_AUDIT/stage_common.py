"""R5C read-only NumPy/Shapely native-stage replay. No torch/opencood imports.
Original model outputs remain immutable; we only decode/replay saved outputs.
"""
import os,sys,json,csv,hashlib,ast,collections,itertools,importlib.abc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPORTS=ROOT.parent;A=REPORTS/'RD3_R5A_EVIDENCE_AUDIT';B=REPORTS/'RD3_R5B_SAMPLE_UTILITY'
DENIED=[]
class OfflineOnly(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):DENIED.append(fullname);raise ImportError('R5C_NO_MODEL_DEVICE_IMPORT '+fullname)
sys.meta_path.insert(0,OfflineOnly());os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True

def allowed(p):
 if isinstance(p,int):return True
 try:return Path(os.fsdecode(p)).resolve().is_relative_to(ROOT)
 except (TypeError,ValueError):return False

def audit_guard(event,args):
 if event=='subprocess.Popen':DENIED.append(event);raise RuntimeError('R5C_NO_SUBPROCESS')
 if event=='open':
  p,mode,flags=args;writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
  if writing and not allowed(p):DENIED.append(str(p));raise PermissionError('R5C_WRITE_OUTSIDE_NEW_DIRECTORY')
 if event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime') and not allowed(args[0]):DENIED.append(event);raise PermissionError('R5C_MUTATION_OUTSIDE_NEW_DIRECTORY')
 if event=='ctypes.dlopen' and args[0] and any(x in str(args[0]).lower() for x in ('libcuda','libcudart','libtorch','libalinpu','ppu_sdk')):DENIED.append(event);raise RuntimeError('R5C_NO_DEVICE_LIBRARY')
sys.addaudithook(audit_guard)
import numpy as np
import yaml
from source_reference.offline_science import matrix,polys,overlap,nms_replay,unique_match,STAGES
from source_reference.offline_storage_reader import digest,load_storage_archive
CONDITIONS=('F','G-B','G-C','E');STATE_ORDER=('E','G-B','G-C','F')
CONTRASTS={'U_B_given_C':('G-B','F'),'U_C_given_B':('G-C','F'),'U_B_given_A':('E','G-C'),'U_C_given_A':('E','G-B')}
CORE=((725,'14014'),(725,'4071'),(324,'29118'))
INTERACTION_SAMPLES=(324,725,384,424)
DATA={};COUNTS={};MATCH={};GEOMETRY_IOU={};CONFIG=None;LOCS={};GT_MAP={};REPLAY=[];SOURCES=[]

def require(ok,name):
 if not bool(ok):raise RuntimeError(name)
def load(p):return json.loads(Path(p).read_text())
def csvrows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(n,d):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for x in rows for k in x))
 with (ROOT/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def same(a,b,name,tol=0):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape,'SHAPE '+name);require(np.isfinite(a).all() and np.isfinite(b).all(),'NONFINITE '+name);error=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.));require(error<=tol,'DIFFERENCE '+name+' '+str(error));return error

def native_stage_replay(a,config):
 raw=a['raw_corners'];score=a['scores_all'];require(raw.shape==(12800,8,3) and score.shape==(12800,),'ALL_RAW_HEAD_ANCHORS')
 scoremask=score>np.float32(config['target_args']['score_threshold']);scoreids=np.flatnonzero(scoremask).astype(np.int32);same(a['score_mask'],scoremask,'score_mask');same(a['ids_score_pass'],scoreids,'score_ids')
 boxes=raw[scoreids];span=boxes.max(1)-boxes.min(1);large=(span[:,0]<=6)&(span[:,1]<=6)&(span[:,1]!=0);zm=(boxes[:,:,2].min(1)>=-3)&(boxes[:,:,2].max(1)<=1);quality=large&zm
 same(a['large_mask'],large,'original_large_y_span_quirk');same(a['z_mask'],zm,'z_mask');same(a['quality_mask'],quality,'quality_mask');q=scoreids[quality];same(a['ids_quality_pass'],q,'quality_ids');same(a['nms_input_corners'],raw[q],'NMS_input_corners');same(a['nms_input_scores'],score[q],'NMS_input_scores_no_quality_rescore')
 top=q[score[q].argsort()[::-1][:1000]];same(a['ids_nms_top1000'],top,'native_top1000_order')
 keep,sup,ovs=nms_replay(raw[q],score[q],config['nms_thresh']);same(a['native_nms_keep'],keep,'native_NMS_keep');nativeids=q[keep];same(a['ids_nms_keep'],nativeids,'NMS_keep_raw_IDs')
 suppressor=np.full(12800,-1,np.int32);overlaps=np.zeros(12800,np.float32)
 for removed,owner in sup.items():suppressor[q[removed]]=q[owner];overlaps[q[removed]]=ovs[removed]
 same(a['suppressor'],suppressor,'all_suppressor_relations');same(a['suppressor_overlap'],overlaps,'all_suppressor_IoU')
 nr=raw[nativeids];limits=np.asarray(config['gt_range']);rangemask=((nr>=limits[:3])&(nr<=limits[3:])).all(2).sum(1)>=8;same(a['range_mask'],rangemask,'native_final_range_mask');final=nativeids[rangemask];same(a['ids_final_range'],final,'final_raw_IDs');same(a['final_corners'],raw[final],'exact_final_corners');same(a['final_scores'],score[final],'exact_final_scores')
 same(a['ids_raw_decoded'],np.arange(12800),'all_raw_IDs');code=np.ones(12800,np.int8)
 for stage,num in zip(STAGES[1:],range(2,7)):code[a['ids_'+stage]]=num
 same(a['first_deletion_code'],code,'deletion_code')
 return {'score_removed':12800-len(scoreids),'large_or_z_removed':len(scoreids)-len(q),'top1000_removed':len(q)-len(top),'NMS_suppressed':len(sup),'final_range_removed':len(nativeids)-len(final),'final_count':len(final),'all_stage_IDs_exact':True,'all_suppressors_exact':True}

def match_trace(a,t):
 m=unique_match(a['final_corners'],a['final_scores'],a['GT_canonical'],t/100);same(m['matched'],a[f'prediction_gt_idx{t}'],'native_unique_match_'+str(t));same(m['flags'],a[f'gt_tp{t}'],'GT_flags_'+str(t));same(m['matched']<0,a[f'prediction_fp{t}'],'FP_flags_'+str(t));return m

def head_decode_check(a,batch,config):
 logits=a['cls_preds'].transpose(0,2,3,1).reshape(-1);sigmoid=(1/(1+np.exp(-logits))).astype(np.float32);score_error=same(sigmoid,a['scores_all'],'saved_cls_sigmoid_score',1e-6)
 anchor=np.asarray(batch['ego']['anchor_box']).reshape(-1,7).astype(np.float32);deltas=a['reg_preds'].transpose(0,2,3,1).reshape(-1,7);diagonal=np.sqrt(anchor[:,4]**2+anchor[:,5]**2);decoded=np.zeros_like(deltas);decoded[:,:2]=deltas[:,:2]*diagonal[:,None]+anchor[:,:2];decoded[:,2]=deltas[:,2]*anchor[:,3]+anchor[:,2];decoded[:,3:6]=np.exp(deltas[:,3:6])*anchor[:,3:6];decoded[:,6]=deltas[:,6]+anchor[:,6]
 decode_error=same(decoded,a['decoded_before_dir'].reshape(-1,7),'saved_reg_anchor_decode',1e-5)
 direction=a['dir_preds'].transpose(0,2,3,1).reshape(-1,config['dir_args']['num_bins']).argmax(-1);period=2*np.pi/config['dir_args']['num_bins'];offset=config['dir_args']['dir_offset'];yaw=a['decoded_before_dir'].reshape(-1,7)[:,6].copy();yaw=yaw-offset-np.floor((yaw-offset)/period)*period+offset+period*direction.astype(np.float32);yaw=yaw-np.floor(yaw/(2*np.pi)+.5)*(2*np.pi)
 # The saved all-anchor raw corners are actual native direction-corrected/projected
 # outputs. We do not replace them with CPU-regenerated predictions.
 ids=a['ids_score_pass'];dir_error=same(yaw[ids],a['score_pass_centers_after_dir'][:,6],'direction_bin_native_yaw',1e-5)
 return {'cls_sigmoid_max_abs_error':score_error,'reg_decode_max_abs_error':decode_error,'direction_yaw_max_abs_error':dir_error,'quality_rescore_branch_present':False,'geometric_quality_masks_applied':True}

def read_sources_and_replay():
 global CONFIG
 av=load(A/'R5A_RESULT.json');bv=load(B/'R5B_INDEPENDENT_VALIDATION.json');require(av['state']=='R5A_PASS' and bv['state']=='R5B_PASS','PREVIOUS_AUDITS_MUST_PASS')
 for p,h in load(ROOT/'R5C_INITIAL_SOURCE_VALIDATION.json')['source_files'].items():require(sha(p)==h,'SOURCE_CHANGED '+p)
 sumsA={x.split('  ',1)[1]:x.split('  ',1)[0] for x in (A/'SHA256SUMS.txt').read_text().splitlines()};sumsB={x.split('  ',1)[1]:x.split('  ',1)[0] for x in (B/'SHA256SUMS.txt').read_text().splitlines()}
 for n in ['RD3_R5B_SCIENTIFIC_ASSESSMENT.md','R5B_SAMPLE_AGENT_UTILITY.csv','R5B_GT_TRANSITION_AUDIT.csv','R5B_GT_FOUR_CONDITION_PATTERNS.csv','R5B_R5C_CANDIDATE_CASES.csv','R5B_RAW_SHARD_LOCATOR.csv']:require(sha(B/n)==sumsB[n],'R5B_SEALED_INPUT '+n)
 for n in ['R5A_GT_IDENTITY_MAP.csv','R5A_GT_NMS_EVIDENCE_READINESS.md','R5A_GT_NMS_FIELD_AVAILABILITY.json']:require(sha(A/n)==sumsA[n],'R5A_SEALED_INPUT '+n)
 inv=csvrows(A/'R5A_SOURCE_INVENTORY.csv');configpaths=[x['absolute_path'] for x in inv if x['absolute_path'].endswith('/config.yaml') and '/workspace/LRCP/opencood/logs/' in x['absolute_path']];require(len(configpaths)==1,'UNIQUE_ACTUAL_CONFIG_SOURCE');configpath=Path(configpaths[0]);tree=yaml.compose(configpath.read_text());post=next(v for k,v in tree.value if k.value=='postprocess');CONFIG=yaml.safe_load(yaml.serialize(post));require(CONFIG['target_args']['score_threshold']==.2 and CONFIG['nms_thresh']==.15 and CONFIG['order']=='hwl','FROZEN_POSTPROCESS')
 for x in inv:
  if x['kind']=='source_definition' or x['absolute_path']==str(configpath):require(sha(x['absolute_path'])==x['sha256'],'SOURCE_DEFINITION_HASH');SOURCES.append(x)
 for x in csvrows(A/'R5A_GT_IDENTITY_MAP.csv'):GT_MAP[int(x['sample_idx']),int(x['gt_idx'])]=x
 locators=csvrows(B/'R5B_RAW_SHARD_LOCATOR.csv');require(len(locators)==32,'LOCATOR32');baselines={(int(x['sample_idx']),x['condition']):x for x in csvrows(B/'R5B_SAMPLE_CONDITION_COUNTS.csv')}
 frozen=load(REPORTS/'RD3_MESSAGE_UTILITY_PILOT/FROZEN_MANIFEST.json');batch_cache={}
 fields=['decoded_before_dir','scores_all','score_mask','score_pass_centers_after_dir','score_pass_corners','large_mask','z_mask','quality_mask','nms_input_corners','nms_input_scores','native_nms_keep','range_mask','cls_preds','reg_preds','dir_preds','raw_corners','final_corners','final_scores','suppressor','suppressor_overlap','first_deletion_code','GT_canonical','IoU_GT_raw']+['ids_'+s for s in STAGES]+[p+str(t) for p in ('gt_tp','prediction_gt_idx','prediction_fp') for t in (50,70)]
 for loc in locators:
  sid=int(loc['sample_idx']);c=loc['condition'];key=sid,c;require(key not in LOCS and sha(loc['absolute_path'])==loc['sha256'],'NPZ_LOCATOR_SHA_OR_DUPLICATE');LOCS[key]=loc
  if sid not in batch_cache:
   r=next(x for x in frozen['samples'] if x['sample_idx']==sid);require(sha(r['input']['path'])==r['input']['sha256'],'FROZEN_INPUT_SHA');payload,_=load_storage_archive(r['input']['path']);batch_cache[sid]=payload['batch']
  with np.load(loc['absolute_path'],allow_pickle=False) as z:require(all(k in z for k in fields),'COMPLETE_CANDIDATE_CHAIN_ARRAYS');a={k:z[k].copy() for k in fields};require('iou_preds' not in z.files,'UNEXPECTED_QUALITY_BRANCH')
  stages_proof=native_stage_replay(a,CONFIG);headproof=head_decode_check(a,batch_cache[sid],CONFIG);actual_iou=matrix(a['GT_canonical'],a['raw_corners']);same(actual_iou,a['IoU_GT_raw'],'independent_raw_GT_IoU',1e-6);require(np.array_equal(actual_iou>=.5,a['IoU_GT_raw']>=.5) and np.array_equal(actual_iou>=.7,a['IoU_GT_raw']>=.7),'SAME_HIGH_IOU_MEMBERSHIP');GEOMETRY_IOU[key]=actual_iou
  counts={'sample_idx':sid,'condition':c,'scene_id':int(baselines[key]['scene_id']),'scene_frame':int(baselines[key]['scene_frame']),'GT_count':len(a['GT_canonical']),'final_prediction_count':len(a['final_scores'])}
  for gi in range(len(a['GT_canonical'])):require(digest(a['GT_canonical'][gi])==GT_MAP[sid,gi]['GT_hash'],'STABLE_PHYSICAL_GT')
  for t in (50,70):
   m=match_trace(a,t);MATCH[sid,c,t]=m;tp=sum(m['tp']);fp=sum(m['fp']);fn=len(m['flags'])-tp
   for n,v in [('TP',tp),('FP',fp),('FN',fn)]:counts[n+str(t)]=v;require(v==int(baselines[key][n+str(t)]),'R5A_R5B_COUNTS_CONFLICT')
  DATA[key]=a;COUNTS[key]=counts;REPLAY.append({**counts,**stages_proof,**headproof,'source_path':loc['absolute_path'],'source_sha256':loc['sha256'],'replayed_final_boxes_scores_NMS_keep_exact':True,'status':'PASS'});print('C0_NATIVE_STAGE_REPLAY_PASS',sid,c,counts['TP70'],counts['FP70'],counts['FN70'],flush=True)
 write('R5C_C0_REPLAY_VALIDATION.json',{'passed':True,'records':REPLAY,'all32_final_corners_scores_keep_exact':True,'native_large_box_quirk_preserved':True,'all_native_greedy_unique_GT_matches_exact':True,'native_BEV_IoU':True,'source_config_path':str(configpath),'source_config_sha256':sha(configpath),'frozen_postprocess_config':CONFIG,'quality_branch':'learned iou_preds not emitted by actual native model; no rescoring; large/z geometric filters active','upstream_feature_module_causality_not_identified':True,'new_forward':'NO','CUDA_initialized':'NO'})
 table('R5C_ALL_32_RESULT_REPLAY.csv',REPLAY)

def candidate_state(a,cid,t):
 ids=a['ids_final_range'];got=np.flatnonzero(ids==cid)
 if not len(got):return 'ABSENT_FINAL',None,None
 pi=int(got[0]);gi=int(a[f'prediction_gt_idx{t}'][pi]);return ('TP' if gi>=0 else 'FP'),pi,gi if gi>=0 else None

def first_missing(a,gi,t):
 high=GEOMETRY_IOU[next(k for k,v in DATA.items() if v is a)][gi]>=t/100
 counts=[int(np.count_nonzero(high[a['ids_'+s]])) for s in STAGES]
 first=next((i for i,n in enumerate(counts) if n==0),None)
 names=['RAW_GEOMETRY','SCORE_FILTER','LARGE_OR_Z_FILTER','TOP1000','NMS','FINAL_RANGE']
 return names[first] if first is not None else 'NONE_HIGH_IOU_SURVIVES_FINAL'
