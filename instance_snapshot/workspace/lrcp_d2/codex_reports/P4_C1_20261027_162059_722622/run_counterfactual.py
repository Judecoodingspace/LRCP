"""Full-anchor score crossover, then fresh native postprocess and matching."""
from c1_common import *
import traceback,collections
CONDITIONS=['G0S0','G0S1','G1S0','G1S1']

def csv_writer(name,fields):
 f=(R/name).open('x',newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();return f,w

def condition_GT_rows(sid,g,s,a,base,mat,groups):
 condition=f'G{g}S{s}';stage={k:a['ids_'+k] for k in STAGES};attr=attribute_gt(mat,stage);rows=[]
 for gi,rec in enumerate(attr):
  raw=np.flatnonzero(mat[gi]>=.7);current=int(a['GT_TP70'][gi]);prior=int(base['GT_TP70'][gi]);anchor=int(a['GT_matched_anchor'][gi])
  changed=raw[(a['first_deletion_code'][raw]!=base['first_deletion_code'][raw])|(a['suppressor'][raw]!=base['suppressor'][raw])]
  score_changed=raw[a['scores_all'][raw]!=base['scores_all'][raw]]
  r={'sample_idx':sid,'gt_idx':gi,'condition':condition,'geometry_alpha':g,'score_alpha':s,'focal':int((sid,gi) in groups),'group':groups.get((sid,gi),'other_GT_in_selected_sample'),'raw_high_IoU_exists':int(len(raw)>0),'availability_class':rec['availability_class'],'final_TP70':current,'final_FN70':1-current,'matched_anchor_id':anchor,'matched_anchor_score':float(a['scores_all'][anchor]) if anchor>=0 else '', 'TP70_under_original_same_geometry':prior,'transition_from_original_same_geometry':('TP' if prior else 'FN')+'_to_'+('TP' if current else 'FN'),'changed_high_IoU_anchor_ids':json.dumps(changed.tolist()),'score_numerically_changed_high_IoU_anchor_ids':json.dumps(score_changed.tolist())}
  newly_pass=raw[~base['mask_score_pass'][raw]&a['mask_score_pass'][raw]];lost_pass=raw[base['mask_score_pass'][raw]&~a['mask_score_pass'][raw]]
  r['new_score_pass_strict_ids']=json.dumps(newly_pass.tolist());r['lost_score_pass_strict_ids']=json.dumps(lost_pass.tolist())
  kept=raw[a['mask_final_range'][raw]]
  r['surviving_strict_candidate_ids']=json.dumps(kept.tolist())
  r['surviving_strict_original_deletion_reasons']=json.dumps({str(ai):REASONS[int(base['first_deletion_code'][ai])] for ai in kept})
  r['matching_competition_FN_despite_final_strict_candidate']=int(not current and len(kept)>0)
  for k in STAGES:
   ids=stage[k];high=ids[mat[gi,ids]>=.7];r['best_IoU_'+k]=rec['best_iou'][k];r['count_ge70_'+k]=len(high);r['high_IoU_ids_'+k]=json.dumps(high.tolist())
  rows.append(r)
 return rows

def run():
 assert json.loads((R/'baseline_validation.json').read_text())['passed']
 if (R/'counterfactual_metrics.json').exists():raise RuntimeError('REFUSE_COUNTERFACTUAL_RESTART')
 start=time.monotonic();pp=make_pp();selections=chosen();groups={(int(x['sample_idx']),int(x['gt_idx'])):x['group'] for x in selections}
 stat_all={k:empty_stats() for k in CONDITIONS};baseline=json.loads((R/'baseline_aggregate_stats.json').read_text());stat_all['G0S0']=baseline['G0S0'];stat_all['G1S1']=baseline['G1S1']
 gt_rows=[];case_rows=[];lineage=[];changed_counts=collections.Counter();checks=[]
 af,aw=csv_writer('per_anchor_changes.csv',['sample_idx','geometry_alpha','anchor_id','grid_y','grid_x','orientation_anchor','original_condition','swapped_condition','score_original','score_swapped','score_delta','score_changed','score_pass_original','score_pass_swapped','first_deletion_original','first_deletion_swapped','NMS_input_original','NMS_input_swapped','NMS_keep_original','NMS_keep_swapped','suppressor_original','suppressor_swapped','suppressor_overlap_original','suppressor_overlap_swapped','final_TP70_original','final_TP70_swapped','best_GT_idx_diagnostic','best_GT_IoU'])
 nf,nw=csv_writer('nms_changes.csv',['sample_idx','geometry_alpha','anchor_id','score_original','score_swapped','input_original','input_swapped','top1000_original','top1000_swapped','keep_original','keep_swapped','suppressor_original','suppressor_swapped','overlap_original','overlap_swapped','changed'])
 try:
  for sid in sorted(int(x['sample_idx']) for x in selections):
   inputs=anchor_input(sid,pp);pilots={g:load_pilot(sid,g) for g in [0,1]};bases={g:load_case('baseline',sid,g,g) for g in [0,1]}
   assert np.array_equal(pilots[0]['GT_canonical'],pilots[1]['GT_canonical'])
   for g in [0,1]:
    s=1-g;base,bstat=bases[g];donor=bases[s][0];a,stats=postprocess(pp,inputs,pilots[g],pilots[s])
    assert np.array_equal(a['decoded_before_dir'],base['decoded_before_dir']) and np.array_equal(a['raw_corners'],base['raw_corners'])
    assert np.array_equal(a['scores_all'],donor['scores_all'])
    assert np.array_equal(a['GT_canonical'],base['GT_canonical'])
    mat=base['IoU_GT_raw'];a['IoU_GT_raw']=mat.copy();a['anchor_definition']=array(inputs['ego']['anchor_box']);a['transform']=array(inputs['ego']['transformation_matrix'])
    assert np.array_equal(a['anchor_definition'],base['anchor_definition'])
    save_case('counterfactual',f'sample_{sid}_G{g}S{s}',a,stats);append_stats(stat_all[f'G{g}S{s}'],stats)
    for score_alpha,case in [(g,base),(s,a)]:
     gt_rows.extend(condition_GT_rows(sid,g,score_alpha,case,base,mat,groups))
     case_rows.append({'sample_idx':sid,'condition':f'G{g}S{score_alpha}','geometry_alpha':g,'score_alpha':score_alpha,'TP70':int(case['GT_TP70'].sum()),'FP70':int(len(case['final_scores'])-case['GT_TP70'].sum()),'predictions':len(case['final_scores']),'GT_count':len(mat),**{k:len(case['ids_'+k]) for k in STAGES},'NMS_shadow_exact':True})
    best=np.argmax(mat,axis=0);bestval=mat[best,np.arange(12800)];best[bestval<=0]=-1
    for ai in range(12800):
     changed=(base['scores_all'][ai]!=a['scores_all'][ai]);changed_counts['score_entries_changed']+=int(changed)
     aw.writerow({'sample_idx':sid,'geometry_alpha':g,'anchor_id':ai,'grid_y':ai//160,'grid_x':(ai//2)%80,'orientation_anchor':ai%2,'original_condition':f'G{g}S{g}','swapped_condition':f'G{g}S{s}','score_original':float(base['scores_all'][ai]),'score_swapped':float(a['scores_all'][ai]),'score_delta':float(a['scores_all'][ai])-float(base['scores_all'][ai]),'score_changed':int(changed),'score_pass_original':int(base['mask_score_pass'][ai]),'score_pass_swapped':int(a['mask_score_pass'][ai]),'first_deletion_original':REASONS[int(base['first_deletion_code'][ai])],'first_deletion_swapped':REASONS[int(a['first_deletion_code'][ai])],'NMS_input_original':int(base['mask_quality_pass'][ai]),'NMS_input_swapped':int(a['mask_quality_pass'][ai]),'NMS_keep_original':int(base['mask_nms_keep'][ai]),'NMS_keep_swapped':int(a['mask_nms_keep'][ai]),'suppressor_original':int(base['suppressor'][ai]),'suppressor_swapped':int(a['suppressor'][ai]),'suppressor_overlap_original':float(base['suppressor_overlap'][ai]),'suppressor_overlap_swapped':float(a['suppressor_overlap'][ai]),'final_TP70_original':int(base['anchor_final_TP70'][ai]),'final_TP70_swapped':int(a['anchor_final_TP70'][ai]),'best_GT_idx_diagnostic':int(best[ai]),'best_GT_IoU':float(bestval[ai])})
    union=np.union1d(base['ids_quality_pass'],a['ids_quality_pass'])
    for ai in union:
     nw.writerow({'sample_idx':sid,'geometry_alpha':g,'anchor_id':int(ai),'score_original':float(base['scores_all'][ai]),'score_swapped':float(a['scores_all'][ai]),'input_original':int(base['mask_quality_pass'][ai]),'input_swapped':int(a['mask_quality_pass'][ai]),'top1000_original':int(base['mask_nms_top1000'][ai]),'top1000_swapped':int(a['mask_nms_top1000'][ai]),'keep_original':int(base['mask_nms_keep'][ai]),'keep_swapped':int(a['mask_nms_keep'][ai]),'suppressor_original':int(base['suppressor'][ai]),'suppressor_swapped':int(a['suppressor'][ai]),'overlap_original':float(base['suppressor_overlap'][ai]),'overlap_swapped':float(a['suppressor_overlap'][ai]),'changed':int(any(base[k][ai]!=a[k][ai] for k in ['mask_quality_pass','mask_nms_top1000','mask_nms_keep','suppressor']))})
    for gi in range(len(mat)):
     for ai in np.flatnonzero(mat[gi]>=.7):
      bsu=int(base['suppressor'][ai]);nsu=int(a['suppressor'][ai])
      lineage.append({'sample_idx':sid,'gt_idx':gi,'geometry_alpha':g,'focal':int((sid,gi) in groups),'group':groups.get((sid,gi),'other_GT_in_selected_sample'),'anchor_id':int(ai),'IoU_GT_fixed_geometry':float(mat[gi,ai]),'score_original':float(base['scores_all'][ai]),'score_swapped':float(a['scores_all'][ai]),'first_deletion_original':REASONS[int(base['first_deletion_code'][ai])],'first_deletion_swapped':REASONS[int(a['first_deletion_code'][ai])],'suppressor_original':bsu,'suppressor_swapped':nsu,'suppressor_IoU_GT_original':float(mat[gi,bsu]) if bsu>=0 else '', 'suppressor_IoU_GT_swapped':float(mat[gi,nsu]) if nsu>=0 else '', 'suppressor_score_original':float(base['scores_all'][bsu]) if bsu>=0 else '', 'suppressor_score_swapped':float(a['scores_all'][nsu]) if nsu>=0 else '', 'original_NMS_overlap':float(base['suppressor_overlap'][ai]),'swapped_NMS_overlap':float(a['suppressor_overlap'][ai])})
    checks.append({'sample_idx':sid,'geometry_alpha':g,'score_alpha':s,'full_12800_score_source_exact':True,'reg_dir_anchor_geometry_unchanged':True,'full_geometry_bitwise_equal':True,'native_NMS_shadow_exact':True,'original_nms_keep_equal_to_swapped':bool(np.array_equal(base['ids_nms_keep'],a['ids_nms_keep'])),'prediction_label_changes_on_same_anchor':int(np.sum(base['anchor_final_TP70']!=a['anchor_final_TP70'])),'GT_TP_changes':int(np.sum(base['GT_TP70']!=a['GT_TP70']))})
    log('COUNTERFACTUAL_CASE_COMPLETE',sample_idx=sid,geometry_alpha=g,score_alpha=s,TP70=int(a['GT_TP70'].sum()),elapsed=time.monotonic()-start)
   guard()
 finally:af.close();nf.close()
 def output(name,rows):
  f,w=csv_writer(name,list(rows[0]));w.writerows(rows);f.close()
 assert len(gt_rows)==648 and len(case_rows)==72 and len(checks)==36
 output('per_GT_counterfactual.csv',gt_rows);output('per_case_metrics.csv',case_rows);output('per_GT_anchor_lineage.csv',lineage)
 output('focal_GT_counterfactual.csv',[x for x in gt_rows if x['focal']])
 groupmetrics={}
 for group in ['all_GT','all_focal']+sorted(set(groups.values())):
  sub=[x for x in gt_rows if group=='all_GT' or (group=='all_focal' and x['focal']) or x['group']==group]
  groupmetrics[group]={c:{'TP70':sum(x['final_TP70'] for x in sub if x['condition']==c),'FN70':sum(x['final_FN70'] for x in sub if x['condition']==c),'GT_count':sum(x['condition']==c for x in sub)} for c in CONDITIONS}
 metrics={}
 for c,stats in stat_all.items():
  metrics[c]={}
  for key in ['50','70']:
   v=stats[key];ap,mrec,mprec=eval_utils.calculate_ap({float(key)/100:v},float(key)/100)
   metrics[c][key]={'AP_descriptive_selected18':ap,'TP':sum(v['tp']),'FP':sum(v['fp']),'FN':v['gt']-sum(v['tp']),'GT':v['gt'],'predictions':len(v['score']),'precision_final':sum(v['tp'])/len(v['score']) if v['score'] else 0,'recall_final':sum(v['tp'])/v['gt']}
 transitions={}
 for g in [0,1]:
  cross=[x for x in gt_rows if x['condition']==f'G{g}S{1-g}']
  transitions[f'G{g}_S{g}_to_S{1-g}']={name:sum(x['transition_from_original_same_geometry']==name for x in cross) for name in ['TP_to_TP','FN_to_TP','TP_to_FN','FN_to_FN']}
 save(R/'aggregate_native_stats.json',stat_all)
 save(R/'counterfactual_metrics.json',{'passed':True,'metrics':metrics,'GT_group_metrics':groupmetrics,'fixed_geometry_GT_transitions':transitions,'checks':checks,'per_GT_rows':len(gt_rows),'per_anchor_rows':18*2*12800,'high_IoU_lineage_rows':len(lineage),'case_count':72,'CPU_only':True,'CUDA_initialized':False,'model_forwards':0,'elapsed_seconds':time.monotonic()-start})
 log('COUNTERFACTUAL_ALL_COMPLETE',groups=groupmetrics,elapsed=time.monotonic()-start)
if __name__=='__main__':
 try:run()
 except BaseException:
  save(R/'counterfactual_failure.json',{'passed':False,'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0});raise
