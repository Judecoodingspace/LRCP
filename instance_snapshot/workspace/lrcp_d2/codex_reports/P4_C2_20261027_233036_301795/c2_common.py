"""CPU-only A/R separation; original native postprocess/NMS and matching."""
from c1_reused import *
C1=Path(INPUT['c1_directory'])
CONDITIONS=[f'G{g}A{a}R{r}' for g in [0,1] for a in [0,1] for r in [0,1]]
def c1case(sid,g,s):
 folder='baseline' if g==s else 'counterfactual';name=f'sample_{sid}_G{g}S{s}'
 with np.load(C1/folder/(name+'.npz'),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 return arr,json.loads((C1/folder/(name+'.json')).read_text())
def save_c2(sid,g,a,r,arr,stat):save_case('cases',f'sample_{sid}_G{g}A{a}R{r}',arr,stat)
def load_c2(sid,g,a,r):
 name=f'sample_{sid}_G{g}A{a}R{r}'
 with np.load(R/'cases'/(name+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 return d,json.loads((R/'cases'/(name+'.json')).read_text())
def output_csv(name,rows):
 assert rows
 with (R/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def factorial_postprocess(pp,inputs,geom,score_source,rank_source):
 guard();before=digest((geom['reg_preds'],geom['dir_preds'],inputs))
 heads={k:torch.from_numpy((score_source if k=='cls_preds' else geom)[k].copy()) for k in ['cls_preds','reg_preds','dir_preds']}
 assert all(t.dtype==torch.float32 and t.device.type=='cpu' for t in heads.values())
 c=PostTrace(pp);assert sys.gettrace() is None
 rank_all=array(torch.sigmoid(torch.from_numpy(rank_source['cls_preds'].copy()).permute(0,2,3,1)).reshape(-1))
 original_nms=box_utils.nms_rotated;native_calls=[]
 def ranked_nms(corners,scores,threshold):
  sm=c.arrays['score_mask'].astype(bool);q=np.flatnonzero(sm)[c.arrays['quality_mask'].astype(bool)]
  assert threshold==.15 and np.array_equal(array(scores),c.arrays['scores_all'][q])
  assert np.array_equal(array(corners),c.arrays['nms_input_corners'])
  native_calls.append(len(q))
  return original_nms(corners,torch.from_numpy(rank_all[q].copy()),threshold)
 box_utils.nms_rotated=ranked_nms
 rng_before=digest((random.getstate(),np.random.get_state(),torch.random.get_rng_state()))
 try:
  sys.settrace(c.globaltrace)
  with torch.no_grad():pred,score=pp.post_process(inputs,{'ego':heads})
 finally:
  sys.settrace(None);box_utils.nms_rotated=original_nms
 assert box_utils.nms_rotated is original_nms and len(native_calls)<=1
 a=c.arrays;a['rank_scores_all']=rank_all
 if pred is None:pred=torch.empty((0,8,3),dtype=torch.float32);score=torch.empty(0,dtype=torch.float32)
 if 'score_pass_corners' not in a:a['score_pass_corners']=np.empty((0,8,3),np.float32)
 if not len(a['score_pass_corners']):
  for k,v in {'large_mask':np.zeros(0,bool),'z_mask':np.zeros(0,bool),'quality_mask':np.zeros(0,bool),'nms_input_corners':np.empty((0,8,3),np.float32),'nms_input_scores':np.empty(0,np.float32),'native_nms_keep':np.empty(0,np.int32),'range_mask':np.empty(0,bool)}.items():a.setdefault(k,v)
 a['raw_corners']=full_geometry(pp,a['decoded_before_dir'],geom['dir_preds'],inputs['ego']['transformation_matrix'])
 sm=a['score_mask'].astype(bool);q=np.flatnonzero(sm).astype(np.int32)[a['quality_mask'].astype(bool)];nms=q[a['native_nms_keep']]
 a['ids_raw_decoded']=np.arange(12800,dtype=np.int32);a['ids_score_pass']=np.flatnonzero(sm).astype(np.int32);a['ids_quality_pass']=q;a['ids_nms_top1000']=q[rank_all[q].argsort()[::-1][:1000]];a['ids_nms_keep']=nms;a['ids_final_range']=nms[a['range_mask'].astype(bool)]
 a['final_corners']=array(pred);a['final_scores']=array(score);a['GT_canonical']=geom['GT_canonical'].copy()
 assert a['raw_corners'].shape==(12800,8,3)
 # Fixed geometry is represented by one all-anchor tensor, independent of S.
 if not np.array_equal(a['raw_corners'][sm],a['score_pass_corners']):raise RuntimeError('CPU_ALL_ANCHOR_NATIVE_CORNER_NOT_EXACT')
 assert np.array_equal(a['raw_corners'][a['ids_final_range']],a['final_corners'])
 shadow=replay_native_nms(a['nms_input_corners'],rank_all[q],.15,common_utils.convert_format,common_utils.compute_iou)
 assert np.array_equal(shadow['keep'],a['native_nms_keep'])
 assert set(a['ids_nms_keep']).issubset(set(a['ids_nms_top1000']))
 sup=np.full(12800,-1,np.int32);overlap=np.zeros(12800,np.float32)
 for rec in shadow['suppressed']:
  ai=int(q[rec['removed_nms_input_idx']]);sup[ai]=int(q[rec['suppressor_nms_input_idx']]);overlap[ai]=rec['bev_iou_float32']
 a['suppressor']=sup;a['suppressor_overlap']=overlap
 reason=np.full(12800,1,np.int8)
 for stage,code in zip(STAGES[1:],range(2,7)):reason[a['ids_'+stage]]=code
 a['first_deletion_code']=reason
 for stage in STAGES:a['mask_'+stage]=np.isin(np.arange(12800),a['ids_'+stage])
 stat70,matched,flags=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'])
 a['matched_gt_prediction']=matched;a['GT_TP70']=flags;gt_anchor=np.full(len(flags),-1,np.int32)
 for pi,gi in enumerate(matched):
  if gi>=0:gt_anchor[gi]=a['ids_final_range'][pi]
 a['GT_matched_anchor']=gt_anchor
 pred_status=np.full(12800,-1,np.int8);pred_status[a['ids_final_range']]=(matched>=0).astype(np.int8);a['anchor_final_TP70']=pred_status
 stat50,_,_=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],.5,False)
 assert digest((geom['reg_preds'],geom['dir_preds'],inputs))==before
 assert rng_before==digest((random.getstate(),np.random.get_state(),torch.random.get_rng_state()))
 guard()
 return a,{'50':stat50,'70':stat70,'native_shadow_NMS_exact':True,'NMS_tied_score_entries':shadow['tied_score_entries'],'native_NMS_calls':len(native_calls)}

