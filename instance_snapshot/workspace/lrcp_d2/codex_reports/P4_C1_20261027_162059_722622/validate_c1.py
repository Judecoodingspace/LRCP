"""Independent read-back of 72 cases, native matching/NMS, GT/anchor CSVs."""
from c1_common import *
import collections,traceback
CONDITIONS=['G0S0','G0S1','G1S0','G1S1']
def run():
 start=time.monotonic();metrics=json.loads((R/'counterfactual_metrics.json').read_text());assert metrics['passed']
 selected=sorted(int(x['sample_idx']) for x in chosen());cases={};records=[]
 for sid in selected:
  pilots={g:load_pilot(sid,g) for g in [0,1]}
  for g in [0,1]:
   base,_=load_case('baseline',sid,g,g)
   for s in [0,1]:
    folder='baseline' if g==s else 'counterfactual';a,stats=load_case(folder,sid,g,s);cases[(sid,g,s)]=a
    assert np.array_equal(a['decoded_before_dir'],base['decoded_before_dir']) and np.array_equal(a['raw_corners'],base['raw_corners'])
    logits=torch.from_numpy(pilots[s]['cls_preds']).permute(0,2,3,1)
    assert np.array_equal(torch.sigmoid(logits).reshape(-1).numpy(),a['scores_all'])
    assert np.array_equal(a['score_mask'],a['scores_all']>.2)
    raw=torch.from_numpy(a['raw_corners']);large=array(box_utils.remove_large_pred_bbx(raw));z=array(box_utils.remove_bbx_abnormal_z(raw))
    sm=a['score_mask'];assert np.array_equal(a['large_mask'],large[sm]) and np.array_equal(a['z_mask'],z[sm])
    q=np.flatnonzero(sm &large &z).astype(np.int32);assert np.array_equal(q,a['ids_quality_pass'])
    assert np.array_equal(q[a['scores_all'][q].argsort()[::-1][:1000]],a['ids_nms_top1000'])
    shadow=replay_native_nms(a['nms_input_corners'],a['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
    assert np.array_equal(shadow['keep'],a['native_nms_keep']) and np.array_equal(q[shadow['keep']],a['ids_nms_keep'])
    sup=np.full(12800,-1,np.int32);overlap=np.zeros(12800,np.float32)
    for row in shadow['suppressed']:
     ai=q[row['removed_nms_input_idx']];sup[ai]=q[row['suppressor_nms_input_idx']];overlap[ai]=row['bev_iou_float32']
    assert np.array_equal(sup,a['suppressor']) and np.array_equal(overlap,a['suppressor_overlap'])
    final,mask=box_utils.mask_boxes_outside_range_numpy(a['raw_corners'][a['ids_nms_keep']],make_pp().params['gt_range'],order=None,return_mask=True)
    assert np.array_equal(mask,a['range_mask']) and np.array_equal(final,a['final_corners'])
    result,matched,flags=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'])
    assert result==stats['70'] and np.array_equal(matched,a['matched_gt_prediction']) and np.array_equal(flags,a['GT_TP70'])
    assert len(a['ids_raw_decoded'])==12800
    # All anchors with common score admission must have identical quality decisions.
    assert np.array_equal(a['IoU_GT_raw'],base['IoU_GT_raw'])
    records.append({'sample_idx':sid,'geometry_alpha':g,'score_alpha':s,'native_matching_equal':True,'quality_and_range_recomputed':True,'native_NMS_shadow_exact':True,'all_12800_score_provenance_exact':True})
   guard()
  log('INDEPENDENT_SAMPLE_PASS',sample_idx=sid,elapsed=time.monotonic()-start)
 gtrows=list(csv.DictReader((R/'per_GT_counterfactual.csv').open()));assert len(gtrows)==648
 seen=set()
 for row in gtrows:
  sid=int(row['sample_idx']);g=int(row['geometry_alpha']);s=int(row['score_alpha']);gi=int(row['gt_idx']);a=cases[(sid,g,s)]
  identity=(sid,g,s,gi);assert identity not in seen;seen.add(identity)
  assert int(row['final_TP70'])==a['GT_TP70'][gi] and int(row['matched_anchor_id'])==a['GT_matched_anchor'][gi]
  for stage in STAGES:
   ids=a['ids_'+stage];strict=ids[a['IoU_GT_raw'][gi,ids]>=.7]
   assert json.loads(row['high_IoU_ids_'+stage])==strict.tolist() and int(row['count_ge70_'+stage])==len(strict)
   best=float(a['IoU_GT_raw'][gi,ids].max()) if len(ids) else 0.
   assert float(row['best_IoU_'+stage])==best
 anchor_rows=0;pairs=collections.Counter();previous=None
 with (R/'per_anchor_changes.csv').open() as f:
  for row in csv.DictReader(f):
   sid=int(row['sample_idx']);g=int(row['geometry_alpha']);ai=int(row['anchor_id']);base=cases[(sid,g,g)];a=cases[(sid,g,1-g)]
   expected_ai=pairs[(sid,g)];assert ai==expected_ai;pairs[(sid,g)]+=1;anchor_rows+=1
   assert float(row['score_original'])==float(base['scores_all'][ai]) and float(row['score_swapped'])==float(a['scores_all'][ai])
   assert float(row['score_delta'])==float(a['scores_all'][ai])-float(base['scores_all'][ai])
   assert row['first_deletion_original']==REASONS[int(base['first_deletion_code'][ai])] and row['first_deletion_swapped']==REASONS[int(a['first_deletion_code'][ai])]
   for field,name in [('score_pass','mask_score_pass'),('NMS_input','mask_quality_pass'),('NMS_keep','mask_nms_keep'),('suppressor','suppressor'),('final_TP70','anchor_final_TP70')]:
    assert int(row[field+'_original'])==base[name][ai] and int(row[field+'_swapped'])==a[name][ai]
 assert anchor_rows==460800 and len(pairs)==36 and all(x==12800 for x in pairs.values())
 nms_rows=0
 for row in csv.DictReader((R/'nms_changes.csv').open()):
  sid=int(row['sample_idx']);g=int(row['geometry_alpha']);ai=int(row['anchor_id']);base=cases[(sid,g,g)];a=cases[(sid,g,1-g)];nms_rows+=1
  for field,name in [('input','mask_quality_pass'),('top1000','mask_nms_top1000'),('keep','mask_nms_keep'),('suppressor','suppressor')]:assert int(row[field+'_original'])==base[name][ai] and int(row[field+'_swapped'])==a[name][ai]
 lineage=0;keys=set()
 for row in csv.DictReader((R/'per_GT_anchor_lineage.csv').open()):
  sid=int(row['sample_idx']);g=int(row['geometry_alpha']);gi=int(row['gt_idx']);ai=int(row['anchor_id']);key=(sid,g,gi,ai)
  assert key not in keys;keys.add(key);lineage+=1;base=cases[(sid,g,g)];a=cases[(sid,g,1-g)]
  assert base['IoU_GT_raw'][gi,ai]>=.7 and float(row['IoU_GT_fixed_geometry'])==float(base['IoU_GT_raw'][gi,ai])
  assert float(row['score_original'])==float(base['scores_all'][ai]) and float(row['score_swapped'])==float(a['scores_all'][ai])
  assert row['first_deletion_original']==REASONS[int(base['first_deletion_code'][ai])] and row['first_deletion_swapped']==REASONS[int(a['first_deletion_code'][ai])]
 assert lineage==sum(np.count_nonzero(cases[(sid,g,g)]['IoU_GT_raw']>=.7) for sid in selected for g in [0,1])
 stats=json.loads((R/'aggregate_native_stats.json').read_text())
 for c in CONDITIONS:
  for t in ['50','70']:
   v=stats[c][t];ap,*_=eval_utils.calculate_ap({float(t)/100:v},float(t)/100)
   assert ap==metrics['metrics'][c][t]['AP_descriptive_selected18']
   assert sum(v['tp'])==metrics['metrics'][c][t]['TP']
   if t=='70':assert sum(int(x['final_TP70']) for x in gtrows if x['condition']==c)==sum(v['tp'])
 assert json.loads((P/'WORKER_FINAL.json').read_text())['state']=='FAILED' and json.loads((P/'EXIT_CODE.json').read_text())['exit_code']==1
 locked=INPUT['history_hashes'];protected=INPUT['protected_hashes'];changed=[]
 for p,h in {**locked,**protected}.items():
  if sha_file(p)!=h:changed.append(p)
 assert not changed,changed
 guard()
 save(R/'validation.json',{'passed':True,'case_count':72,'postprocess_baseline_cases':36,'postprocess_counterfactual_cases':36,'native_evaluator_rechecked_cases':72,'GT_rows_verified':648,'anchor_rows_verified':anchor_rows,'NMS_rows_verified':nms_rows,'strict_candidate_lineage_rows_verified':lineage,'full_score_provenance_and_fixed_geometry_verified':True,'native_NMS_shadow_exact_cases':72,'history_and_source_unchanged':True,'history_files':len(locked),'protected_files':len(protected),'original_worker_state_preserved':'FAILED','original_worker_exit_code_preserved':1,'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-start,'case_checks':records})
 log('INDEPENDENT_VALIDATION_PASS',elapsed=time.monotonic()-start)
if __name__=='__main__':
 try:run()
 except BaseException:
  save(R/'validation.json',{'passed':False,'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0});raise
