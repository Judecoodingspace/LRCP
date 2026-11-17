"""R5A independent NumPy/Shapely offline calculations; no LRCP/torch imports.
Matches source-defined native BEV IoU float32, score ordering, greedy unique GT,
VOC2010 PR integration and rotated NMS; never loads a model or repairs data.
"""
import numpy as np
from shapely.geometry import Polygon
STAGES=('raw_decoded','score_pass','quality_pass','nms_top1000','nms_keep','final_range')
CAUSES=('no_high_iou_raw_candidate','score_filter','size_or_z_filter','nms_top1000_truncation','nms_overlap_suppression','post_nms_range_filter')

def require(ok,reason):
 if not bool(ok):raise RuntimeError(reason)

def polys(boxes):
 result=[Polygon([(float(b[i,0]),float(b[i,1])) for i in range(4)]) for b in boxes]
 require(all(p.is_valid and p.area>0 for p in result),'INVALID_POLYGON')
 return result

def overlap(p,others):
 return np.asarray([p.intersection(q).area/p.union(q).area for q in others],dtype=np.float32)

def matrix(gt,boxes):
 require(np.isfinite(gt).all() and np.isfinite(boxes).all(),'NONFINITE_GEOMETRY')
 ps=polys(boxes);gs=polys(gt);low=boxes[:,:4,:2].min(1);high=boxes[:,:4,:2].max(1);result=np.zeros((len(gt),len(boxes)),np.float32)
 for gi,(g,p) in enumerate(zip(gt,gs)):
  lo=g[:4,:2].min(0);hi=g[:4,:2].max(0);ids=np.flatnonzero(np.all(high>lo,axis=1)&np.all(low<hi,axis=1))
  result[gi,ids]=overlap(p,[ps[int(i)] for i in ids])
 return result

def unique_match(pred,scores,gt,threshold):
 predps=polys(pred);gtps=polys(gt);remaining=list(range(len(gt)));order=np.argsort(-scores);matched=np.full(len(scores),-1,np.int32);tp=[];fp=[]
 for pi in order:
  ious=overlap(predps[int(pi)],[gtps[i] for i in remaining])
  if not remaining or float(np.max(ious))<threshold:tp.append(0);fp.append(1)
  else:matched[int(pi)]=remaining.pop(int(np.argmax(ious)));tp.append(1);fp.append(0)
 flags=np.zeros(len(gt),np.int8);flags[matched[matched>=0]]=1
 return {'score':scores[order].tolist(),'tp':tp,'fp':fp,'gt':len(gt),'matched':matched,'flags':flags}

def global_ap(stat):
 order=np.argsort(-np.asarray(stat['score']));tp=np.asarray(stat['tp'])[order].astype(np.float64);fp=np.asarray(stat['fp'])[order].astype(np.float64)
 tp=np.cumsum(tp);fp=np.cumsum(fp);rec=(tp/stat['gt']).tolist();prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)!=0).tolist()
 rec=[0.]+rec+[1.];prec=[0.]+prec+[0.]
 for i in range(len(prec)-2,-1,-1):prec[i]=max(prec[i],prec[i+1])
 ap=0.
 for i in range(1,len(rec)):
  if rec[i]!=rec[i-1]:ap+=(rec[i]-rec[i-1])*prec[i]
 return float(ap),rec,prec,order.tolist()

def nms_replay(boxes,scores,threshold=.15):
 ps=polys(boxes);pending=scores.argsort()[::-1][:1000].copy();picks=[];sup={};ious_by_id={}
 while len(pending):
  chosen=int(pending[0]);picks.append(chosen);iou=overlap(ps[chosen],[ps[int(i)] for i in pending[1:]])
  positions=np.where(iou>threshold)[0]+1
  for pos in positions:sup[int(pending[pos])]=chosen;ious_by_id[int(pending[pos])]=float(iou[pos-1])
  pending=np.delete(pending,positions);pending=np.delete(pending,0)
 return np.asarray(picks,np.int32),sup,ious_by_id

def stages(iou,ids,threshold):
 result=[]
 for gi in range(len(iou)):
  counts={k:int(np.count_nonzero(iou[gi,ids[k]]>=threshold)) for k in STAGES};best={k:float(iou[gi,ids[k]].max(initial=0.)) for k in STAGES}
  first=next((i for i,k in enumerate(STAGES) if counts[k]==0),None)
  result.append({'counts':counts,'best':best,'availability':'high_iou_survives_final' if first is None else CAUSES[first]})
 return result

def frozen_gt_corners(batch,order):
 allboxes=[];allids=[]
 template=np.asarray([[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,1],[1,1,1],[-1,1,1],[-1,-1,1]],np.float32)/2
 for e in batch.values():
  boxes=np.asarray(e['object_bbx_center'][e['object_bbx_mask']==1]).copy();ids=list(e['object_ids']);require(len(boxes)==len(ids),'GT_CENTER_ID_LENGTH')
  if order=='hwl':boxes=boxes[:,[0,1,2,5,4,3,6]]
  else:require(order=='lwh','GT_ORDER')
  local=boxes[:,None,3:6].astype(np.float32)*template;angle=boxes[:,6].astype(np.float32);c=np.cos(angle);s=np.sin(angle);zero=np.zeros_like(c);one=np.ones_like(c)
  rot=np.stack([c,s,zero,-s,c,zero,zero,zero,one],axis=1).reshape(-1,3,3)
  corners=np.matmul(local,rot)+boxes[:,None,:3].astype(np.float32)
  hom=np.concatenate([corners.transpose(0,2,1),np.ones((len(boxes),1,8),np.float32)],axis=1)
  projected=np.matmul(np.asarray(e['transformation_matrix_clean']),hom)[:,:3].transpose(0,2,1)
  allboxes.extend(projected);allids.extend(ids)
 return np.asarray(allboxes),allids

def canonical_gt_ids(canonical,batch,order):
 corners,ids=frozen_gt_corners(batch,order);error=np.max(np.abs(canonical[:,None]-corners[None]),axis=(2,3));hits=error<=1e-5
 require(np.all(hits.sum(1)==1),'GT_OBJECT_ID_GEOMETRY_BIJECTION')
 matched=np.argmax(hits,axis=1);physical=[ids[int(i)] for i in matched]
 require(len(set(physical))==len(canonical),'GT_OBJECT_ID_DUPLICATE')
 return physical,float(error[np.arange(len(canonical)),matched].max(initial=0.))
