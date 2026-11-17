"""Gate all 36 native CPU head-to-final replays before any counterfactual."""
import traceback
from c1_common import *
def run():
 if (R/'baseline_validation.json').exists():raise RuntimeError('REFUSE_BASELINE_RESTART')
 start=time.monotonic();pp=make_pp();checks=[];stat_all={f'G{g}S{g}':empty_stats() for g in [0,1]}
 existing=list(csv.DictReader((P/'diagnostics/analysis/per_GT_stages.csv').open()));oldrow={(int(x['sample_idx']),int(x['gt_idx']),float(x['alpha']),x['stage']):x for x in existing}
 for sid in sorted(int(x['sample_idx']) for x in chosen()):
  inputs=anchor_input(sid,pp)
  for g in [0,1]:
   pilot=load_pilot(sid,g);a,stats=postprocess(pp,inputs,pilot,pilot)
   check={'sample_idx':sid,'alpha':g,'geometry_atol':GEOMETRY_ATOL,'score_atol':SCORE_ATOL,'max_geometry_error':{},'max_score_error':{}}
   for name in ['decoded_before_dir','raw_corners','score_pass_corners','score_pass_centers_after_dir','nms_input_corners','final_corners']:
    assert a[name].shape==pilot[name].shape,(sid,g,name,'shape')
    e=float(np.max(np.abs(a[name]-pilot[name]),initial=0));check['max_geometry_error'][name]=e
    if e>GEOMETRY_ATOL:raise RuntimeError(f'BASELINE_GEOMETRY_MISMATCH {sid} {g} {name} {e}')
   for name in ['scores_all','nms_input_scores','final_scores']:
    e=float(np.max(np.abs(a[name]-pilot[name]),initial=0));check['max_score_error'][name]=e
    if e>SCORE_ATOL:raise RuntimeError(f'BASELINE_SCORE_MISMATCH {sid} {g} {name} {e}')
   for name in ['score_mask','large_mask','z_mask','quality_mask','native_nms_keep','range_mask']+['ids_'+s for s in STAGES]:assert np.array_equal(a[name],pilot[name]),(sid,g,name)
   _,matched,flags=eval_native(pilot['final_corners'],pilot['final_scores'],pilot['GT_canonical'])
   assert np.array_equal(matched,a['matched_gt_prediction']) and np.array_equal(flags,a['GT_TP70']),(sid,g,'GT matching')
   mat=iou_matrix(a['GT_canonical'],a['raw_corners']);a['IoU_GT_raw']=mat
   oldmat=iou_matrix(pilot['GT_canonical'],pilot['raw_corners'])
   assert np.array_equal(mat>=.7,oldmat>=.7),(sid,g,'raw threshold classification')
   for gi in range(len(mat)):
    for stage in STAGES:
     ids=a['ids_'+stage];record=oldrow[(sid,gi,float(g),stage)]
     high=ids[mat[gi,ids]>=.7]
     assert high.tolist()==json.loads(record['high_iou_anchor_ids']),(sid,g,gi,stage,'strict candidate IDs')
     assert len(high)==int(record['high_iou_count']) and int(a['GT_TP70'][gi])==int(record['final_TP70'])
   check.update(passed=True,all_anchor_threshold_classes_exact=True,all_stage_ids_and_masks_exact=True,NMS_keep_order_exact=True,GT_matching_exact=True,TP70=int(a['GT_TP70'].sum()),FP70=int(len(a['final_scores'])-a['GT_TP70'].sum()),GT_count=len(mat))
   a['anchor_definition']=array(inputs['ego']['anchor_box']);a['transform']=array(inputs['ego']['transformation_matrix'])
   save_case('baseline',f'sample_{sid}_G{g}S{g}',a,stats);append_stats(stat_all[f'G{g}S{g}'],stats);checks.append(check);log('BASELINE_CASE_PASS',sample_idx=sid,alpha=g,elapsed=time.monotonic()-start)
 assert len(checks)==36 and sum(x['GT_count'] for x in checks)==324
 save(R/'baseline_aggregate_stats.json',stat_all)
 save(R/'baseline_validation.json',{'passed':True,'cases':checks,'case_count':36,'sample_count':18,'GT_count':162,'coordinates_atol':GEOMETRY_ATOL,'scores_atol':SCORE_ATOL,'strict_classification_and_matching_exact':True,'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0,'elapsed_seconds':time.monotonic()-start})
 log('BASELINE_GATE_PASSED',case_count=36,elapsed=time.monotonic()-start)
if __name__=='__main__':
 try:run()
 except BaseException:
  save(R/'baseline_validation.json',{'passed':False,'counterfactual_started':False,'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0});raise
