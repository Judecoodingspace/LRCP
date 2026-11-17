"""Check that both recovered predictions already survived native alpha1 NMS."""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
O=Path(__file__).parent;R=O.parent;sys.path.insert(0,str(R/'P5_B0_C2_20261031_cpu'))
from pilot_common import iou_matrix
m=json.loads((R/'P5_B0_C3_20261031_cpu/real_run/manifest.json').read_text());paths={x['sample_idx']:Path(x['path'])/'payload.bin' for x in m['records'] if x['condition']=='core'}
rows=json.loads((O/'recovered_GT_mechanism.json').read_text());checks=[]
for row in rows:
 sid=row['sample_idx'];gid=row['gt_idx'];anchor=row['B0_matched_anchor']
 with np.load(R/f'P5_A_CPU_FIX_20261028_storage/capture/sample_{sid}/alpha_1.0.npz',allow_pickle=False) as a,np.load(paths[sid],allow_pickle=False) as b:
  pa=np.flatnonzero(a['ids_final_range']==anchor);pb=np.flatnonzero(b['ids_final_range']==anchor);assert len(pa)==len(pb)==1
  old=float(iou_matrix(a['GT_canonical'][gid:gid+1],a['final_corners'][pa])[0,0]);new=float(iou_matrix(b['GT_canonical'][gid:gid+1],b['final_corners'][pb])[0,0])
  assert old<.7<=new and int(b['prediction_gt_idx70'][pb[0]])==gid
  assert abs(old-row['old_best_final_IoU'])<1e-8 and abs(new-row['B0_matched_IoU'])<1e-8
  assert int(a['suppressor'][anchor])==int(b['suppressor'][anchor])==-1
  assert float(b['final_scores'][pb[0]])<float(a['final_scores'][pa[0]])
  checks.append({'sample_idx':sid,'gt_idx':gid,'anchor':anchor,'old_final_IoU':old,'new_final_IoU':new,'same_final_anchor':True,'score_did_not_increase':True})
assert len(checks)==2 and not torch.cuda.is_initialized()
(O/'RECOVERY_VALIDATION.json').write_text(json.dumps({'passed':True,'checks':checks,'new_model_forward':0,'CUDA_initialized':False},indent=2));print(json.dumps(checks,indent=2))
