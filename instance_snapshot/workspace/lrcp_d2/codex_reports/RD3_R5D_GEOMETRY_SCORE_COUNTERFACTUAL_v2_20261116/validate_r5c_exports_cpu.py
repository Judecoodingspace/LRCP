"""Pre-hybrid independent gate: actual full NPZ vs R5C CSV exports, CPU only.
Also executes after producer: originals must match all archived arrays exactly.
No producer import, model import, device allocation, or threshold adjustment.
"""
from cpu_common import *
CHECK=collections.Counter();FAILED=[]
def ck(ok,group,key=None):
 CHECK[group]+=1
 if not bool(ok):FAILED.append({'group':group,'key':str(key)})
def eq(a,b,group,key=None):
 a=np.asarray(a);b=np.asarray(b);ck(a.shape==b.shape and np.array_equal(a,b),group,key)
def main():
 src,_,_=load_sources(export=False)
 for r in rows(CROOT/'R5C_CANDIDATE_LINEAGE.csv'):
  key=int(r['sample_idx']),r['condition']
  if key not in src:continue
  a=src[key];cid=int(r['candidate_id']);tag=key+(cid,);scoreids=a['ids_score_pass'];sp=np.flatnonzero(scoreids==cid);kp=np.flatnonzero(a['ids_nms_keep']==cid);sup=int(a['suppressor'][cid]);qp=a['ids_quality_pass'];rank=np.flatnonzero(qp[a['scores_all'][qp].argsort()[::-1]]==cid)
  eq(float(r['score']),a['scores_all'][cid],'lineage_score',tag)
  eq(json.loads(r['raw_corners']),a['raw_corners'][cid],'lineage_full_3D_corners',tag)
  eq(json.loads(r['decoded_before_direction']),a['decoded_before_dir'].reshape(-1,7)[cid],'lineage_decoded',tag)
  eq(json.loads(r['reg_delta']),a['reg_preds'].transpose(0,2,3,1).reshape(-1,7)[cid],'lineage_reg',tag)
  eq(json.loads(r['dir_logits']),a['dir_preds'].transpose(0,2,3,1).reshape(-1,2)[cid],'lineage_dir',tag)
  eq(float(r['cls_logit']),a['cls_preds'].transpose(0,2,3,1).reshape(-1)[cid],'lineage_cls',tag)
  for f,b in [('score_pass',bool(len(sp))),('quality_pass',cid in set(qp)),('top1000_pass',cid in set(a['ids_nms_top1000'])),('native_NMS_keep',bool(len(kp))),('native_NMS_suppressed',sup>=0)]:ck((r[f]=='True')==b,'lineage_stage_'+f,tag)
  for f,m,pos in [('large_mask_if_evaluated','large_mask',sp),('z_mask_if_evaluated','z_mask',sp),('range_pass_if_evaluated','range_mask',kp)]:ck(r[f]==str(bool(a[m][pos[0]])) if len(pos) else r[f]=='','lineage_evaluated_'+f,tag)
  ck(r['native_score_sort_rank_if_quality_pass']==str(int(rank[0])) if len(rank) else r['native_score_sort_rank_if_quality_pass']=='','lineage_sort_order',tag)
  ck(r['native_suppressor_candidate_id']==str(sup) if sup>=0 else r['native_suppressor_candidate_id']=='','lineage_suppressor',tag)
  if sup>=0:eq(float(r['native_suppression_BEV_IoU']),a['suppressor_overlap'][cid],'lineage_NMS_overlap',tag)
  ck(int(r['first_deletion_code'])==int(a['first_deletion_code'][cid]),'lineage_first_deletion',tag)
  for t in (50,70):
   st,pi,gi=ci_state(a,cid,t);ck(r[f'final_status{t}']==st,'lineage_final_status',tag+(t,));ck(r[f'matched_gt_idx{t}']==(str(gi) if gi is not None else ''),'lineage_unique_GT',tag+(t,))
  ck(r['source_sha256']==LOC[key]['sha256'],'lineage_source_SHA',tag)
 edges=collections.defaultdict(dict)
 for r in rows(CROOT/'R5C_NMS_SUPPRESSOR_EDGES.csv'):
  key=int(r['sample_idx']),r['condition']
  if key not in src:continue
  a=src[key];cid=int(r['suppressed_id']);owner=int(r['suppressor_id']);edges[key][cid]=owner
  ck(int(a['suppressor'][cid])==owner,'edge_suppressor',key+(cid,));eq(float(r['pair_BEV_IoU']),a['suppressor_overlap'][cid],'edge_overlap',key+(cid,));eq(float(r['suppressed_score']),a['scores_all'][cid],'edge_score',key+(cid,));eq(float(r['suppressor_score']),a['scores_all'][owner],'edge_owner_score',key+(cid,))
 for key,a in src.items():ck(edges[key]=={int(i):int(a['suppressor'][i]) for i in np.flatnonzero(a['suppressor']>=0)},'all_NMS_edges_complete',key)
 matchrows=rows(CROOT/'R5C_GT_MATCH_ORDER.csv')
 for key,a in src.items():
  sid,c=key
  for t in (50,70):
   rr=[r for r in matchrows if int(r['sample_idx'])==sid and r['condition']==c and float(r['iou_threshold'])==t/100];order=np.argsort(-a['final_scores']);ck(len(rr)==len(order),'match_order_complete',key+(t,));remaining=list(range(len(a['GT_canonical'])))
   for rank,(r,pi) in enumerate(zip(rr,order)):
    cid=int(a['ids_final_range'][pi]);iv=a['IoU_GT_raw'][:,cid];eligible=np.flatnonzero(iv>=t/100).tolist();eligible_remaining=[i for i in remaining if iv[i]>=t/100];ck(int(r['candidate_id'])==cid and int(r['final_prediction_index'])==int(pi) and int(r['score_order_rank_zero_based'])==rank,'match_native_sort_ID',key+(t,cid,));eq(float(r['score']),a['scores_all'][cid],'match_score',key+(t,cid,))
    ck(json.loads(r['remaining_GT_before'])==remaining and json.loads(r['eligible_all_GT_indices'])==eligible and json.loads(r['eligible_remaining_GT_indices'])==eligible_remaining,'match_remaining_GT',key+(t,cid,));gi=int(a[f'prediction_gt_idx{t}'][pi]);ck(r['matched_gt_idx']==(str(gi) if gi>=0 else '') and r['result']==('TP' if gi>=0 else 'FP'),'match_unique_result',key+(t,cid,))
    if gi>=0:eq(float(r['matched_IoU']),iv[gi],'match_IoU',key+(t,cid,));remaining.remove(gi)
 originals=[]
 for p in PAIRS:
  for k in (0,1):
   rid=p['pair_id']+f'_G{k}_S{k}';path=run_path(rid)
   if path.exists():
    with np.load(path,allow_pickle=False) as z:
     a=src[p['sample_idx'],p['condition'+str(k)]]
     for n in SOURCE_FIELDS:
      if n in z.files:eq(z[n],a[n],'produced_original_arrays_exact',(rid,n))
    originals.append(rid)
 phase='post_producer' if originals else 'pre_hybrid'
 result={'state':'R5C_EXPORT_GATE_PASS' if not FAILED else 'R5C_EXPORT_GATE_FAIL','phase':phase,'passed_checks':sum(CHECK.values())-len(FAILED),'failed_checks':len(FAILED),'check_groups':dict(CHECK),'FAIL':FAILED,'original_runs_checked':originals,'source_records':6,'new_forward':0,'CUDA_initializations':0,'GPU_PPU_calls':0}
 write('R5D_R5C_EXPORT_'+phase.upper()+'_VALIDATION.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
 require(not FAILED,'R5C_EXPORT_GATE_FAIL')
if __name__=='__main__':main()
