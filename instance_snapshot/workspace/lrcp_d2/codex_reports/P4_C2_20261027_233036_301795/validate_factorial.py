"""Independent array-to-native-NMS replay, matching and full CSV readback."""
from c2_common import *
from collections import Counter
import traceback

def run():
 start=time.monotonic();metrics=json.loads((R/'factorial_metrics.json').read_text());assert json.loads((R/'baseline_validation.json').read_text())['passed']
 rows=list(csv.DictReader((R/'per_GT_factorial.csv').open()));lookup={(int(z['sample_idx']),int(z['gt_idx']),z['condition']):z for z in rows};assert len(rows)==len(lookup)==1296
 checks=[];aggregate={c:empty_stats() for c in CONDITIONS};expect_lineage={};stage_max=0;allanchor_rows=0
 af=(R/'all_anchor_factorial.csv').open();ar=iter(csv.DictReader(af))
 try:
  for sid in sorted(int(x['sample_idx']) for x in chosen()):
   for g in [0,1]:
    base=c1case(sid,g,g)[0];mat=base['IoU_GT_raw'];corners=base['raw_corners'];qall=array(box_utils.remove_large_pred_bbx(torch.from_numpy(corners.copy()))&box_utils.remove_bbx_abnormal_z(torch.from_numpy(corners.copy()))).astype(bool)
    S={s:c1case(sid,s,s)[0]['scores_all'] for s in [0,1]};cases={}
    for a in [0,1]:
     for r in [0,1]:
      d,stat=load_c2(sid,g,a,r);cases[(a,r)]=d;c=f'G{g}A{a}R{r}'
      assert np.array_equal(d['raw_corners'],corners) and np.array_equal(d['scores_all'],S[a]) and np.array_equal(d['rank_scores_all'],S[r])
      scoreids=np.flatnonzero(S[a]>.2);q=scoreids[qall[scoreids]]
      keep=array(box_utils.nms_rotated(torch.from_numpy(corners[q].copy()),torch.from_numpy(S[r][q].copy()),.15)).astype(int)
      shadow=replay_native_nms(corners[q],S[r][q],.15,common_utils.convert_format,common_utils.compute_iou)
      assert np.array_equal(keep,shadow['keep'])
      nmsids=q[keep];final,ran=box_utils.mask_boxes_outside_range_numpy(corners[nmsids].copy(),make_pp().params['gt_range'],order=None,return_mask=True);ids=nmsids[ran]
      stages=[np.arange(12800),scoreids,q,q[S[r][q].argsort()[::-1][:1000]],nmsids,ids]
      for st,ix in zip(STAGES,stages):assert np.array_equal(d['ids_'+st],ix) and np.array_equal(d['mask_'+st],np.isin(np.arange(12800),ix))
      assert np.array_equal(final,d['final_corners']) and np.array_equal(S[a][ids],d['final_scores'])
      sup=np.full(12800,-1,np.int32);ov=np.zeros(12800,np.float32)
      for z in shadow['suppressed']:ai=q[z['removed_nms_input_idx']];sup[ai]=q[z['suppressor_nms_input_idx']];ov[ai]=z['bev_iou_float32']
      assert np.array_equal(sup,d['suppressor']) and np.array_equal(ov,d['suppressor_overlap'])
      reason=np.ones(12800,np.int8)
      for ix,code in zip(stages[1:],range(2,7)):reason[ix]=code
      assert np.array_equal(reason,d['first_deletion_code'])
      st70,match,flags=eval_native(final,S[a][ids],base['GT_canonical']);st50,_,_=eval_native(final,S[a][ids],base['GT_canonical'],.5,False)
      assert st70==stat['70'] and st50==stat['50'] and np.array_equal(flags,d['GT_TP70']) and np.array_equal(match,d['matched_gt_prediction'])
      append_stats(aggregate[c],{'50':st50,'70':st70})
      if a==r:
       old,oldstat=c1case(sid,g,a)
       assert np.array_equal(final,old['final_corners']) and np.array_equal(S[a][ids],old['final_scores']) and st70==oldstat['70'] and st50==oldstat['50']
      for gi in range(len(mat)):
       z=lookup[(sid,gi,c)];assert int(z['TP70'])==flags[gi] and int(z['FN70'])==1-flags[gi]
       high=np.flatnonzero(mat[gi]>=.7)
       for st,ix in zip(STAGES,stages):hi=ix[mat[gi,ix]>=.7];assert int(z['n70_'+st])==len(hi) and json.loads(z['high_ids_'+st])==hi.tolist()
       assert json.loads(z['score_deleted_strict_ids'])==high[reason[high]==1].tolist() and json.loads(z['NMS_deleted_strict_ids'])==high[reason[high]==4].tolist()
       for ai in high:expect_lineage[(sid,gi,c,int(ai))]=(float(mat[gi,ai]),float(S[a][ai]),float(S[r][ai]),REASONS[int(reason[ai])],int(sup[ai]),float(mat[gi,sup[ai]]) if sup[ai]>=0 else None,float(ov[ai]))
      stage_max=max(stage_max,len(q));checks.append({'sample_idx':sid,'condition':c,'independent_native_NMS_exact':True,'independent_final_A_score_exact':True,'native_GT_matching_exact':True,'all_stage_lineage_exact':True})
    for ai in range(12800):
     z=next(ar);assert (int(z['sample_idx']),int(z['G']),int(z['anchor_id']))==(sid,g,ai);assert float(z['S0'])==float(S[0][ai]) and float(z['S1'])==float(S[1][ai]) and int(z['quality_geometry_pass'])==qall[ai]
     for (a,r),d in cases.items():
      k=f'A{a}R{r}';assert int(z[k+'_deletion_code'])==d['first_deletion_code'][ai] and int(z[k+'_suppressor'])==d['suppressor'][ai] and float(z[k+'_NMS_overlap'])==float(d['suppressor_overlap'][ai]) and int(z[k+'_final_TP70'])==d['anchor_final_TP70'][ai]
     allanchor_rows+=1
   guard();log('INDEPENDENT_SAMPLE_PASS',sample_idx=sid,elapsed=time.monotonic()-start)
  assert next(ar,None) is None and allanchor_rows==460800
 finally:af.close()
 count=0
 for z in csv.DictReader((R/'per_GT_candidate_lineage.csv').open()):
  key=(int(z['sample_idx']),int(z['gt_idx']),z['condition'],int(z['anchor_id']));iou,A,rank,reason,sup,supiou,ov=expect_lineage.pop(key)
  assert float(z['IoU_GT'])==iou and float(z['A_score'])==A and float(z['R_score'])==rank and z['first_deletion']==reason and int(z['suppressor'])==sup and float(z['NMS_IoU'])==ov
  if sup>=0:assert float(z['suppressor_IoU_GT'])==supiou and float(z['suppressor_R_score'])>rank
  count+=1
 assert not expect_lineage and count==metrics['rows']['strict_candidate_lineage']
 nmscount=0
 for sid in sorted(int(x['sample_idx']) for x in chosen()):
  sub=[z for z in csv.DictReader((R/'nms_priority_changes.csv').open()) if int(z['sample_idx'])==sid]
  cases={(g,a,r):load_c2(sid,g,a,r)[0] for g in [0,1] for a in [0,1] for r in [0,1]}
  for z in sub:
   g,a,ai=int(z['G']),int(z['A']),int(z['anchor_id'])
   for r in [0,1]:
    d=cases[(g,a,r)];assert int(z[f'keep_R{r}'])==d['mask_nms_keep'][ai] and int(z[f'suppressor_R{r}'])==d['suppressor'][ai] and float(z[f'overlap_R{r}'])==float(d['suppressor_overlap'][ai])
   nmscount+=1
 effects=list(csv.DictReader((R/'per_GT_effects.csv').open()));assert len(effects)==324
 for z in effects:
  sid,gi,g=int(z['sample_idx']),int(z['gt_idx']),int(z['G']);y={(a,r):int(lookup[(sid,gi,f'G{g}A{a}R{r}')]['TP70']) for a in [0,1] for r in [0,1]}
  assert int(z['interaction_TP'])==y[(1,1)]-y[(1,0)]-y[(0,1)]+y[(0,0)]
  assert int(z['delta_A_at_R0'])==y[(1,0)]-y[(0,0)] and int(z['delta_A_at_R1'])==y[(1,1)]-y[(0,1)] and int(z['delta_R_at_A0'])==y[(0,1)]-y[(0,0)] and int(z['delta_R_at_A1'])==y[(1,1)]-y[(1,0)]
  for st in STAGES:
   n={(a,r):int(lookup[(sid,gi,f'G{g}A{a}R{r}')]['n70_'+st]) for a,r in y};assert int(z['interaction_n70_'+st])==n[(1,1)]-n[(1,0)]-n[(0,1)]+n[(0,0)]
 # Independent grouping and AP integration; native default global sorting.
 for grp,conds in metrics['group_metrics'].items():
  for c,v in conds.items():
   selected=[z for z in rows if z['condition']==c and (grp=='all_GT' or (grp=='all_focal' and z['focal']=='1') or z['group'].startswith(grp+'_'))]
   assert v['GT_count']==len(selected) and v['TP70']==sum(int(z['TP70']) for z in selected)
 for c,stats in aggregate.items():
  for th in ['50','70']:
   ap,_,_=eval_utils.calculate_ap({int(th)/100:stats[th]},int(th)/100);v=metrics['metrics'][c][th];assert ap==v['AP_descriptive_selected18'] and sum(stats[th]['tp'])==v['TP'] and sum(stats[th]['fp'])==v['FP']
 alignment_rows=list(csv.DictReader((R/'score_geometry_alignment.csv').open()));assert len(alignment_rows)==2592
 for sid in sorted(int(x['sample_idx']) for x in chosen()):
  for g in [0,1]:
   d=load_c2(sid,g,0,0)[0];mat=d['IoU_GT_raw'];cor=torch.from_numpy(d['raw_corners'].copy());q=array(box_utils.remove_large_pred_bbx(cor)&box_utils.remove_bbx_abnormal_z(cor)).astype(bool)
   for s in [0,1]:
    score=load_c2(sid,g,s,s)[0]['scores_all']
    for z in [z for z in alignment_rows if int(z['sample_idx'])==sid and int(z['G'])==g and int(z['S'])==s]:
     v=mat[int(z['gt_idx'])];name=z['candidate_set'];mask=np.ones(12800,bool) if name=='raw_all_12800' else v>0
     if name.startswith('quality') or name.startswith('admitted'):mask &= q
     if name.startswith('admitted'):mask &= score>.2
     ids=np.flatnonzero(mask);assert len(ids)==int(z['candidate_count'])
     if len(ids):top=int(ids[np.argsort(score[ids])[::-1][0]]);maxv=float(v[ids].max());assert int(z['top_score_anchor'])==top and float(z['max_IoU'])==maxv and int(z['top_score_also_max_IoU'])==int(abs(float(v[top])-maxv)<=1e-6)
 for field in ['history_hashes','protected_hashes']:
  for path,h in INPUT[field].items():assert sha_file(path)==h,path
 assert json.loads((P/'WORKER_FINAL.json').read_text())['state']=='FAILED' and json.loads((P/'EXIT_CODE.json').read_text())['exit_code']==1
 assert not torch.cuda.is_initialized()
 save(R/'validation.json',{'passed':True,'case_count':len(checks),'homogeneous_cases':72,'mixed_cases':72,'GT_rows_verified':len(rows),'factorial_GT_rows_verified':len(effects),'all_anchor_rows_verified':allanchor_rows,'strict_candidate_lineage_verified':count,'nms_priority_rows_verified':nmscount,'alignment_rows_verified':len(alignment_rows),'max_quality_candidates':stage_max,'top1000_truncation_cases':0,'independent_native_NMS_and_GT_matching':True,'history_files_unchanged':len(INPUT['history_hashes']),'protected_files_unchanged':len(INPUT['protected_hashes']),'old_worker_FAILED_exit1_preserved':True,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-start,'case_checks':checks})
 log('INDEPENDENT_VALIDATION_COMPLETE',elapsed=time.monotonic()-start)
if __name__=='__main__':
 try:run()
 except BaseException:
  save(R/'validation.json',{'passed':False,'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0});raise
