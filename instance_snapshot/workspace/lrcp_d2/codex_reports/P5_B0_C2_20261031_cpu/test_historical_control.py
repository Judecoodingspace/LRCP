import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
from INDEPENDENT_VALIDATOR import verify_historical_control,iou_matrix,ValidationError
R=Path(__file__).parent;P=R.parent/'P5_A_CPU_FIX_20261028_storage';cases=[]
for sid,kind in [(590,'off'),(590,'pass'),(918,'pass'),(918,'off')]:
 with np.load(P/f'capture/sample_{sid}/alpha_1.0.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 a['IoU_GT_raw']=iou_matrix(a['GT_canonical'],a['raw_corners']);assert verify_historical_control(a,sid)
 a['cls_preds'].flat[0]+=1
 try:verify_historical_control(a,sid)
 except ValidationError as e:assert 'CONTROL_cls_preds' in str(e)
 else:raise AssertionError('corrupt control accepted')
 cases.append({'sample_idx':sid,'condition':kind,'historical_self_test_passed':True,'corrupted_control_rejected':True})
assert not torch.cuda.is_initialized()
(R/'CONTROL_BASELINE_VALIDATION.json').write_text(json.dumps({'passed':True,'cases':cases,'real_ppu_control_executed':False,'scope':'P5-A historical self-test and deliberately corrupted historical copy only; not a new P5-B0 device control.','real_model_forward':0,'CUDA_initialized':False},indent=2))
