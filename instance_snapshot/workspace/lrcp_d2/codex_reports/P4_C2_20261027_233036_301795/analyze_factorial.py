from c2_common import *
from collections import Counter

def alignment(sid,g,s,base,score,quality_all,gi):
 v=base['IoU_GT_raw'][gi];out=[]
 sets={'raw_positive_overlap':np.flatnonzero(v>0),'quality_positive_overlap':np.flatnonzero((v>0)&quality_all),'admitted_positive_overlap':np.flatnonzero((v>0)&quality_all&(score>.2)),'raw_all_12800':np.arange(12800)}
 for name,ids in sets.items():
  row={'sample_idx':sid,'gt_idx':gi,'G':g,'S':s,'candidate_set':name,'candidate_count':len(ids)}
  if len(ids):
   top=int(ids[score[ids].argsort()[::-1][0]]);best=int(ids[v[ids].argsort()[::-1][0]]);maxv=float(v[ids].max());maxs=float(score[ids].max());ties=ids[score[ids]==maxs]
   row.update(top_score_anchor=top,top_score=float(score[top]),top_score_anchor_IoU=float(v[top]),max_IoU_anchor=best,max_IoU=maxv,max_IoU_anchor_score=float(score[best]),same_anchor=int(top==best),top_score_also_max_IoU=int(abs(float(v[top])-maxv)<=1e-6),top_score_ties=len(ties),top_score_strict=int(v[top]>=.7),any_strict=int(maxv>=.7),IoU_regret=maxv-float(v[top]))
  else:row.update({k:'' for k in ['top_score_anchor','top_score','top_score_anchor_IoU','max_IoU_anchor','max_IoU','max_IoU_anchor_score','same_anchor','top_score_also_max_IoU','top_score_ties','top_score_strict','any_strict','IoU_regret']})
  out.append(row)
 return out

def run():
 assert json.loads((R/'mixed_execution.json').read_text())['passed']
 start=time.monotonic();selections=chosen();groups={(int(x['sample_idx']),int(x['gt_idx'])):x['group'] for x in selections}
 gtrows=[];lineage=[];factors=[];align=[];nmsrows=[];caserows=[];statsall={c:empty_stats() for c in CONDITIONS}
 af=(R/'all_anchor_factorial.csv').open('x',newline='');aw=None
 try:
  for sid in sorted(int(x['sample_idx']) for x in selections):
   for g in [0,1]:
    cases={(a,r):load_c2(sid,g,a,r) for a in [0,1] for r in [0,1]};base=cases[(g,g)][0];mat=base['IoU_GT_raw'];scores={s:cases[(s,s)][0]['scores_all'] for s in [0,1]}
    corners=torch.from_numpy(base['raw_corners'].copy());qa=array(box_utils.remove_large_pred_bbx(corners)&box_utils.remove_bbx_abnormal_z(corners)).astype(bool)
    for s in [0,1]:
     for gi in range(len(mat)):align.extend(alignment(sid,g,s,base,scores[s],qa,gi))
    for (adm,rank),(d,stat) in cases.items():
     c=f'G{g}A{adm}R{rank}';append_stats(statsall[c],stat);attr=attribute_gt(mat,{s:d['ids_'+s] for s in STAGES})
     caserows.append({'sample_idx':sid,'condition':c,'G':g,'A':adm,'R':rank,'GT_count':len(mat),'TP70':int(d['GT_TP70'].sum()),'FP70':len(d['final_scores'])-int(d['GT_TP70'].sum()),'predictions':len(d['final_scores']),**{s:len(d['ids_'+s]) for s in STAGES},'top1000_truncated':max(0,len(d['ids_quality_pass'])-1000)})
     for gi,rec in enumerate(attr):
      raw=np.flatnonzero(mat[gi]>=.7);ma=int(d['GT_matched_anchor'][gi]);orig=int(base['GT_TP70'][gi]);tp=int(d['GT_TP70'][gi]);strict_nms=raw[d['first_deletion_code'][raw]==4]
      low= [int(ai) for ai in strict_nms if mat[gi,d['suppressor'][ai]]<.7]
      low_high_A=[ai for ai in low if scores[adm][d['suppressor'][ai]]>scores[adm][ai]]
      low_high_R=[ai for ai in low if scores[rank][d['suppressor'][ai]]>scores[rank][ai]]
      row={'sample_idx':sid,'gt_idx':gi,'condition':c,'G':g,'A':adm,'R':rank,'focal':int((sid,gi) in groups),'group':groups.get((sid,gi),'other_GT'),'TP70':tp,'FN70':1-tp,'original_TP70':orig,'transition_from_original':('TP' if orig else 'FN')+'_to_'+('TP' if tp else 'FN'),'matched_anchor':ma,'matched_A_score':float(scores[adm][ma]) if ma>=0 else '', 'last_strict_disappearance':rec['availability_class'],'score_deleted_strict_ids':json.dumps(raw[d['first_deletion_code'][raw]==1].tolist()),'NMS_deleted_strict_ids':json.dumps(strict_nms.tolist()),'low_IoU_suppressor_strict_ids':json.dumps(low),'higher_A_low_IoU_suppressor_strict_ids':json.dumps(low_high_A),'higher_R_low_IoU_suppressor_strict_ids':json.dumps(low_high_R)}
      for stage in STAGES:
       ids=d['ids_'+stage];hi=ids[mat[gi,ids]>=.7];row['n70_'+stage]=len(hi);row['high_ids_'+stage]=json.dumps(hi.tolist());row['best_IoU_'+stage]=float(mat[gi,ids].max(initial=0))
      gtrows.append(row)
      for ai in raw:
       sup=int(d['suppressor'][ai])
       lineage.append({'sample_idx':sid,'gt_idx':gi,'condition':c,'G':g,'A':adm,'R':rank,'anchor_id':int(ai),'IoU_GT':float(mat[gi,ai]),'S0':float(scores[0][ai]),'S1':float(scores[1][ai]),'A_score':float(scores[adm][ai]),'R_score':float(scores[rank][ai]),'first_deletion':REASONS[int(d['first_deletion_code'][ai])],'suppressor':sup,'suppressor_IoU_GT':float(mat[gi,sup]) if sup>=0 else '', 'suppressor_A_score':float(scores[adm][sup]) if sup>=0 else '', 'suppressor_R_score':float(scores[rank][sup]) if sup>=0 else '', 'NMS_IoU':float(d['suppressor_overlap'][ai])})
    for gi in range(len(mat)):
     y={(a,r):int(cases[(a,r)][0]['GT_TP70'][gi]) for a in [0,1] for r in [0,1]}
     row={'sample_idx':sid,'gt_idx':gi,'G':g,'focal':int((sid,gi) in groups),'group':groups.get((sid,gi),'other_GT'),**{f'TP_A{a}R{r}':y[(a,r)] for a,r in y},'delta_A_at_R0':y[(1,0)]-y[(0,0)],'delta_A_at_R1':y[(1,1)]-y[(0,1)],'delta_R_at_A0':y[(0,1)]-y[(0,0)],'delta_R_at_A1':y[(1,1)]-y[(1,0)],'interaction_TP':y[(1,1)]-y[(1,0)]-y[(0,1)]+y[(0,0)]}
     for st in STAGES:
      n={(a,r):int(np.sum(mat[gi,cases[(a,r)][0]['ids_'+st]]>=.7)) for a,r in y};row['interaction_n70_'+st]=n[(1,1)]-n[(1,0)]-n[(0,1)]+n[(0,0)]
     factors.append(row)
    for adm in [0,1]:
     d0=cases[(adm,0)][0];d1=cases[(adm,1)][0]
     assert np.array_equal(d0['ids_quality_pass'],d1['ids_quality_pass'])
     for ai in d0['ids_quality_pass']:
      row={'sample_idx':sid,'G':g,'A':adm,'anchor_id':int(ai),'A_score':float(scores[adm][ai]),'R0_score':float(scores[0][ai]),'R1_score':float(scores[1][ai])}
      for rank,d in [(0,d0),(1,d1)]:
       row.update({f'top1000_R{rank}':int(d['mask_nms_top1000'][ai]),f'keep_R{rank}':int(d['mask_nms_keep'][ai]),f'suppressor_R{rank}':int(d['suppressor'][ai]),f'overlap_R{rank}':float(d['suppressor_overlap'][ai]),f'final_R{rank}':int(d['mask_final_range'][ai])})
      row['keep_changed']=int(row['keep_R0']!=row['keep_R1']);row['suppressor_changed']=int(row['suppressor_R0']!=row['suppressor_R1']);nmsrows.append(row)
    for ai in range(12800):
     row={'sample_idx':sid,'G':g,'anchor_id':ai,'S0':float(scores[0][ai]),'S1':float(scores[1][ai]),'quality_geometry_pass':int(qa[ai])}
     for (a,r),(d,_) in cases.items():
      k=f'A{a}R{r}';row[k+'_deletion_code']=int(d['first_deletion_code'][ai]);row[k+'_suppressor']=int(d['suppressor'][ai]);row[k+'_NMS_overlap']=float(d['suppressor_overlap'][ai]);row[k+'_final_TP70']=int(d['anchor_final_TP70'][ai])
     if aw is None:aw=csv.DictWriter(af,fieldnames=list(row));aw.writeheader()
     aw.writerow(row)
   guard();log('ANALYSIS_SAMPLE_COMPLETE',sample_idx=sid,elapsed=time.monotonic()-start)
 finally:af.close()
 for name,rows in [('per_GT_factorial.csv',gtrows),('per_GT_effects.csv',factors),('per_GT_candidate_lineage.csv',lineage),('nms_priority_changes.csv',nmsrows),('score_geometry_alignment.csv',align),('per_case_metrics.csv',caserows)]:output_csv(name,rows)
 output_csv('focal_GT_factorial.csv',[x for x in gtrows if x['focal']])
 output_csv('GT_with_nonzero_interaction.csv',[x for x in factors if x['interaction_TP']!=0] or [{k:'' for k in factors[0]}])
 groupmetrics={grp:{c:{'TP70':sum(x['TP70'] for x in gtrows if x['condition']==c and (grp=='all_GT' or (grp=='all_focal' and x['focal']) or x['group'].startswith(grp+'_'))),'GT_count':sum(x['condition']==c and (grp=='all_GT' or (grp=='all_focal' and x['focal']) or x['group'].startswith(grp+'_')) for x in gtrows)} for c in CONDITIONS} for grp in ['all_GT','all_focal','loss','gain','control']}
 metrics={}
 for c,stats in statsall.items():
  metrics[c]={}
  for key in ['50','70']:
   v=stats[key];ap,_,_=eval_utils.calculate_ap({int(key)/100:v},int(key)/100);metrics[c][key]={'AP_descriptive_selected18':ap,'TP':sum(v['tp']),'FP':sum(v['fp']),'FN':v['gt']-sum(v['tp']),'GT':v['gt'],'predictions':len(v['score'])}
 pairs=[]
 for g in [0,1]:
  for variable,fixed in [('A',0),('A',1),('R',0),('R',1)]:
   aa,rr=(0,fixed) if variable=='A' else (fixed,0);bb,ss=(1,fixed) if variable=='A' else (fixed,1)
   left={ (x['sample_idx'],x['gt_idx']):x for x in gtrows if x['condition']==f'G{g}A{aa}R{rr}'};right={ (x['sample_idx'],x['gt_idx']):x for x in gtrows if x['condition']==f'G{g}A{bb}R{ss}'}
   diff=[{'sample_idx':sid,'gt_idx':gi,'from_TP':left[(sid,gi)]['TP70'],'to_TP':right[(sid,gi)]['TP70'],'from_stage':left[(sid,gi)]['last_strict_disappearance'],'to_stage':right[(sid,gi)]['last_strict_disappearance']} for sid,gi in left if left[(sid,gi)]['TP70']!=right[(sid,gi)]['TP70']]
   pairs.append({'G':g,'variable':variable,'fixed_other':fixed,'from':f'G{g}A{aa}R{rr}','to':f'G{g}A{bb}R{ss}','gains':sum(x['to_TP']>x['from_TP'] for x in diff),'losses':sum(x['to_TP']<x['from_TP'] for x in diff),'targets':diff})
 alignsummary={}
 for g in [0,1]:
  for s in [0,1]:
   for name in ['raw_positive_overlap','quality_positive_overlap','admitted_positive_overlap','raw_all_12800']:
    rows=[x for x in align if x['G']==g and x['S']==s and x['candidate_set']==name and x['candidate_count']]
    alignsummary[f'G{g}S{s}_{name}']={'nonempty_GT':len(rows),'maxscore_is_maxIoU':sum(x['top_score_also_max_IoU'] for x in rows),'highest_score_below70_despite_strict_available':sum(x['any_strict'] and not x['top_score_strict'] for x in rows),'mean_IoU_regret':sum(x['IoU_regret'] for x in rows)/len(rows) if rows else None}
 suppression={}
 for c in CONDITIONS:
  rows=[x for x in gtrows if x['condition']==c]
  suppression[c]={'GTs_with_low_IoU_suppressor':sum(bool(json.loads(x['low_IoU_suppressor_strict_ids'])) for x in rows),'GTs_with_higher_A_low_IoU_suppressor':sum(bool(json.loads(x['higher_A_low_IoU_suppressor_strict_ids'])) for x in rows),'strict_candidate_events_low_IoU_suppressor':sum(len(json.loads(x['low_IoU_suppressor_strict_ids'])) for x in rows),'GTs_with_higher_R_low_IoU_suppressor':sum(bool(json.loads(x['higher_R_low_IoU_suppressor_strict_ids'])) for x in rows)}
 interactions={f'G{g}':{'nonzero_GT_count':sum(x['G']==g and x['interaction_TP']!=0 for x in factors),'positive_GT_count':sum(x['G']==g and x['interaction_TP']>0 for x in factors),'negative_GT_count':sum(x['G']==g and x['interaction_TP']<0 for x in factors),'sum_interaction_TP':sum(x['interaction_TP'] for x in factors if x['G']==g),'nonzero_stage_count':{st:sum(x['G']==g and x['interaction_n70_'+st]!=0 for x in factors) for st in STAGES}} for g in [0,1]}
 save(R/'aggregate_native_stats.json',statsall)
 save(R/'factorial_metrics.json',{'passed_execution':True,'metrics':metrics,'group_metrics':groupmetrics,'single_factor_transitions':pairs,'interactions':interactions,'alignment':alignsummary,'suppression':suppression,'top1000_truncated_cases':sum(x['top1000_truncated']>0 for x in caserows),'max_quality_candidates':max(x['quality_pass'] for x in caserows),'rows':{'GT':len(gtrows),'factorial_GT':len(factors),'strict_candidate_lineage':len(lineage),'nms_priority':len(nmsrows),'alignment':len(align),'all_anchor':18*2*12800},'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0,'elapsed_seconds':time.monotonic()-start})
 log('FACTORIAL_ANALYSIS_COMPLETE',elapsed=time.monotonic()-start)
if __name__=='__main__':run()
