"""Stage E CPU-only full anchor lineage / every GT attribution. No model calls."""
import csv,json,signal,time,collections,sys
from pathlib import Path
import numpy as np
from pilot_common import *
from contracts import attribute_gt,replay_native_nms

def analyze():
 state=json.load((R/'STATE.json').open())
 if (R/'FAILURE.json').exists() or state['phase']!='E_READY':raise RuntimeError('COLLECTION_NOT_ACCEPTED')
 signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('CPU_ANALYSIS_TIMEOUT')));signal.alarm(1800)
 start=time.monotonic();outdir=R/'diagnostics'/'analysis';outdir.mkdir(exist_ok=False)
 from opencood.utils import common_utils
 focal={(int(x['sample_idx']),int(x['gt_idx'])):x for x in selections()}
 stage_rows=[];gt_rows=[];high_rows=[];anchor_count=collections.Counter();case_counts=[];checks=[];feature_rows=[]
 with (outdir/'per_anchor_deletion.csv').open('x',newline='') as anchor_file:
  aw=csv.DictWriter(anchor_file,fieldnames=['sample_idx','alpha','anchor_id','grid_y','grid_x','orientation_anchor','score','first_deletion_stage','NMS_suppressor_anchor_id','NMS_suppressor_overlap']);aw.writeheader()
  for row in selections():
   sid=int(row['sample_idx'])
   for alpha in [0.,1.]:
    budget(start,1800)
    prefix=f'core_sample_{sid}_alpha_{alpha:.1f}'
    with np.load(R/'diagnostics'/(prefix+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    meta=json.load((R/'diagnostics'/(prefix+'.json')).open())
    stage={n:d['ids_'+n] for n in STAGES}
    mat=iou_matrix(d['GT_canonical'],d['raw_corners'])
    native=replay_native_nms(d['nms_input_corners'],d['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
    assert np.array_equal(native['keep'],d['native_nms_keep'])
    qids=stage['quality_pass'];sup={int(qids[x['removed_nms_input_idx']]):(int(qids[x['suppressor_nms_input_idx']]),x['bev_iou_float32']) for x in native['suppressed']}
    reason=np.full(12800,'score_filter',dtype=object)
    for i in range(1,len(STAGES)):
     reason[stage[STAGES[i]]]=['size_or_z_filter','nms_top1000_truncation','nms_overlap_suppression','post_nms_range_filter','final_survived'][i-1]
    assert set(sup)==set(stage['nms_top1000'])-set(stage['nms_keep'])
    assert set(native['top1000_excluded'])==set(range(len(qids)))-set(d['nms_input_scores'].argsort()[::-1][:1000])
    res=attribute_gt(mat,stage)
    pred_mat=mat[:,stage['final_range']]
    flags,matched=greedy(pred_mat,d['final_scores'])
    for ai in range(12800):
     suppressor,overlap=sup.get(ai,(-1,''))
     aw.writerow({'sample_idx':sid,'alpha':alpha,'anchor_id':ai,'grid_y':ai//160,'grid_x':(ai//2)%80,'orientation_anchor':ai%2,'score':float(d['scores_all'][ai]),'first_deletion_stage':reason[ai],'NMS_suppressor_anchor_id':suppressor,'NMS_suppressor_overlap':overlap})
     anchor_count[(str(alpha),reason[ai])]+=1
    for gi,rec in enumerate(res):
     focus=focal.get((sid,gi));group=focus['group'] if focus else 'other_GT_in_selected_sample'
     g={'sample_idx':sid,'gt_idx':gi,'alpha':alpha,'focal':int(focus is not None),'group':group,'availability_class':rec['availability_class'],'final_TP70':int(flags[gi])}
     for name in STAGES:
      cand=stage[name];bi=int(cand[mat[gi,cand].argmax()]) if len(cand) else -1
      high=cand[mat[gi,cand]>=.7]
      g['best_iou_'+name]=rec['best_iou'][name];g['count_ge70_'+name]=len(high)
      stage_rows.append({'sample_idx':sid,'gt_idx':gi,'alpha':alpha,'focal':g['focal'],'group':group,'stage':name,'best_iou':rec['best_iou'][name],'best_anchor_id':bi,'high_iou_count':len(high),'high_iou_anchor_ids':json.dumps(high.tolist()),'final_TP70':int(flags[gi])})
     gt_rows.append(g)
     for ai in np.flatnonzero(mat[gi]>=.7):
      ai=int(ai);supid,overlap=sup.get(ai,(-1,''))
      high_rows.append({'sample_idx':sid,'gt_idx':gi,'alpha':alpha,'focal':g['focal'],'group':group,'anchor_id':ai,'IoU_GT':float(mat[gi,ai]),'score':float(d['scores_all'][ai]),'first_deletion_stage':reason[ai],'NMS_suppressor_anchor_id':supid,'NMS_suppressor_overlap':overlap,'NMS_suppressor_IoU_GT':float(mat[gi,supid]) if supid>=0 else '', 'NMS_suppressor_score':float(d['scores_all'][supid]) if supid>=0 else ''})
    counts={n:len(stage[n]) for n in STAGES};case_counts.append({'sample_idx':sid,'alpha':alpha,**counts,'native_NMS_CPU_replay_equal':True})
    for name,info in meta['features'].items():
     feature_rows.append({'sample_idx':sid,'alpha':alpha,'feature':name,**{k:info[k] for k in ['shape','sha256','mean','std','min','max']}})
    checks.append({'sample_idx':sid,'alpha':alpha,'passed':True,'GT_count':len(res),'raw_anchor_count':12800,'NMS_replay_exact':True,'historical':meta['historical_validation']})
    event('CPU_ATTRIBUTION_COMPLETE',sample_idx=sid,alpha=alpha,elapsed=time.monotonic()-start,counts=counts)
 def csvout(name,rows):
  with (outdir/name).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 csvout('per_GT_stages.csv',stage_rows);csvout('per_GT_summary.csv',gt_rows);csvout('high_IoU_candidate_deletion.csv',high_rows);csvout('per_case_stage_counts.csv',case_counts);csvout('feature_summaries.csv',feature_rows)
 focuses=[x for x in gt_rows if x['focal']]
 csvout('focal_GT_summary.csv',focuses)
 assert len(gt_rows)==324 and len(stage_rows)==1944 and len(focuses)==36
 counts={}
 for alpha in [0.,1.]:
  counts[str(alpha)]=dict(collections.Counter(x['availability_class'] for x in gt_rows if x['alpha']==alpha))
 focal_counts={}
 for group in sorted({x['group'] for x in focuses}):
  focal_counts[group]={str(a):dict(collections.Counter(x['availability_class'] for x in focuses if x['alpha']==a and x['group']==group)) for a in [0.,1.]}
 total_diag=bytes_in(R)-bytes_in(R/'batches')
 assert total_diag<=2**30
 summary={'passed':True,'samples':18,'GT_count':162,'GT_condition_rows':len(gt_rows),'GT_stage_rows':len(stage_rows),'anchor_condition_rows':18*2*12800,'high_IoU_candidate_rows':len(high_rows),'all_GT_availability_counts':counts,'focal_availability_counts':focal_counts,'anchor_deletion_counts':{a:{reason:n for (alpha,reason),n in anchor_count.items() if alpha==a} for a in ['0.0','1.0']},'elapsed_seconds':time.monotonic()-start,'diagnostic_total_bytes_excluding_input_cache':total_diag}
 write_json(R/'ANALYSIS_SUMMARY.json',summary);write_json(R/'ACCEPTANCE.json',{'passed':True,'cases':checks,'samples':18,'GT_count':162,'NMS_replay_cases':36,'historical_all_cases_passed':True,'CPU_analysis_seconds':time.monotonic()-start,'full_model_forward_count':state['forward_completed']})
 state['phase']='E_COMPLETE';state['status']='COMPLETE';write_json(R/'STATE.json',state,exclusive=False)
 signal.alarm(0)
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
 try:analyze()
 except BaseException as exc:
  import traceback
  signal.alarm(0);write_json(R/'ANALYSIS_FAILURE.json',{'failure':str(exc),'traceback':traceback.format_exc()});raise
