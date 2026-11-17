"""Real saved-head CPU postprocessor tests; never creates or forwards a model."""
from pathlib import Path
import sys,json,traceback,numpy as np,torch
R=Path(__file__).resolve().parent
C=R.parent/'P4_C1_20261027_162059_722622'
sys.path.insert(0,str(C))
from c1_common import make_pp,anchor_input,load_pilot,STAGES
from minimal_collector import collect_postprocess

def run():
 pp=make_pp();rows=[]
 for sid in [821,17,670]:
  inputs=anchor_input(sid,pp)
  for alpha in [0,1]:
   pilot=load_pilot(sid,alpha);heads={k:torch.from_numpy(pilot[k].copy()) for k in ['cls_preds','reg_preds','dir_preds']}
   a,meta=collect_postprocess(pp,inputs,heads)
   with np.load(C/f'baseline/sample_{sid}_G{alpha}S{alpha}.npz',allow_pickle=False) as z:
    for k in ['decoded_before_dir','scores_all','score_mask','quality_mask','native_nms_keep','range_mask','raw_corners','final_corners','final_scores','suppressor','suppressor_overlap','first_deletion_code']+['ids_'+s for s in STAGES]:
     if not np.array_equal(a[k],z[k]):raise RuntimeError(f'CPU_CAPTURE_MISMATCH {sid} {alpha} {k}')
   rows.append({'sample_idx':sid,'alpha':alpha,'passed':True,'native_NMS_replay_exact':meta['native_NMS_replay_exact']})
 assert len(rows)==6 and not torch.cuda.is_initialized()
 (R/'MINIMAL_CAPTURE_CPU_TEST.json').write_text(json.dumps({'passed':True,'cases':rows,'model_forwards':0,'CUDA_initialized':False,'PPU_capture_unverified':True},indent=2))
 print('MINIMAL_CAPTURE_SAVED_HEAD_TESTS_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  (R/'MINIMAL_CAPTURE_FAILURE.json').write_text(json.dumps({'passed':False,'traceback':traceback.format_exc()},indent=2));raise
