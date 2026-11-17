"""CPU-only audit of both ALL-GT recoveries, including unchanged final anchor IDs."""
import os,sys,json,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
O=Path(__file__).parent;R=O.parent;sys.path.insert(0,str(R/'P5_B0_C2_20261031_cpu'))
from pilot_common import iou_matrix
m=json.loads((R/'P5_B0_C3_20261031_cpu/real_run/manifest.json').read_text());rs={r['sample_idx']:Path(r['path']) for r in m['records'] if r['condition']=='core'}
summary=json.loads((O/'summary.json').read_text());recoveries=[(x['sample_idx'],x['gt_idx']) for x in summary['changed_GT70'] if x['transition70']=='FN_to_TP'];out=[];details=[]
for sid,gid in recoveries:
 with np.load(R/f'P5_A_CPU_FIX_20261028_storage/capture/sample_{sid}/alpha_1.0.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 with np.load(rs[sid]/'payload.bin',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
 ai=iou_matrix(a['GT_canonical'][gid:gid+1],a['raw_corners'])[0];bi=b['IoU_GT_raw'][gid]
 strict=set(map(int,np.flatnonzero((ai>=.7)|(bi>=.7))));winners={int(v) for x in strict for v in (a['suppressor'][x],b['suppressor'][x]) if v>=0}
 for aid in sorted(strict|winners):
  row={'sample_idx':sid,'gt_idx':gid,'anchor_id':aid,'strict_any':aid in strict,'winner_for_strict':aid in winners}
  for label,z,iou in [('alpha1',a,ai),('B0',b,bi)]:
   row['score_'+label]=float(z['scores_all'][aid]);row['IoU_'+label]=float(iou[aid]);row['deletion_'+label]=int(z['first_deletion_code'][aid]);row['suppressor_'+label]=int(z['suppressor'][aid]);row['overlap_'+label]=float(z['suppressor_overlap'][aid]);row['final_'+label]=aid in z['ids_final_range']
  out.append(row)
 core_match=int(np.flatnonzero(b['prediction_gt_idx70']==gid)[0]);new_anchor=int(b['ids_final_range'][core_match]);old_final_iou=iou_matrix(a['GT_canonical'][gid:gid+1],a['final_corners'])[0];new_final_iou=iou_matrix(b['GT_canonical'][gid:gid+1],b['final_corners'])[0]
 best_idx=int(np.argmax(old_final_iou));old_best=int(a['ids_final_range'][best_idx]);old_suppressor=int(a['suppressor'][new_anchor]);old_same_indices=np.flatnonzero(a['ids_final_range']==new_anchor)
 details.append({'sample_idx':sid,'gt_idx':gid,'old_best_final_anchor':old_best,'old_best_final_IoU':float(old_final_iou[best_idx]),'old_best_final_score':float(a['final_scores'][best_idx]),'B0_matched_anchor':new_anchor,'B0_matched_IoU':float(new_final_iou[core_match]),'B0_threshold_margin':float(new_final_iou[core_match]-.7),'B0_matched_score':float(b['final_scores'][core_match]),'same_anchor_old_IoU':float(ai[new_anchor]),'same_anchor_old_score':float(a['scores_all'][new_anchor]),'same_anchor_already_final_alpha1':bool(len(old_same_indices)),'same_anchor_old_suppressor':old_suppressor,'old_suppressor_B0_IoU':float(bi[old_suppressor]) if old_suppressor>=0 else None,'old_suppressor_B0_score':float(b['scores_all'][old_suppressor]) if old_suppressor>=0 else None})
with (O/'recovered_GT_anchor_competition.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(O/'recovered_GT_mechanism.json').write_text(json.dumps(details,indent=2));print(json.dumps(details,indent=2))
