import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
from types import SimpleNamespace
from pilot_common import MODEL,iou_matrix,STAGES
from contracts import attribute_gt
from minimal_collector import collect_postprocess
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P4_C2_20261027_233036_301795')
from c1_reused import eval_native
R=Path(__file__).parent
h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));pp=build_postprocessor(h['postprocess'],False)
inputp='/workspace/lrcp_d2/codex_reports/P4_FULL_PREP_20261028_113714_782940/batches/sample_590.pt'
frozen=torch.load(inputp,map_location='cpu',weights_only=False,mmap=False)['batch']
p='/workspace/lrcp_d2/codex_reports/P5_A_CPU_FIX_20261028_storage/capture/sample_590/alpha_1.0.npz'
with np.load(p,allow_pickle=False) as old:
 heads={k:torch.from_numpy(old[k].copy()) for k in ('cls_preds','reg_preds','dir_preds')}
 with torch.no_grad():got,meta=collect_postprocess(pp,frozen,heads)
 for key in ('score_mask','quality_mask','native_nms_keep','range_mask','ids_final_range'):
  assert np.array_equal(got[key],old[key]),key
 for key,tol in [('final_corners',1e-5),('final_scores',1e-6),('raw_corners',1e-5)]:
  assert np.allclose(got[key],old[key],rtol=0,atol=tol),key
 iou=iou_matrix(old['GT_canonical'],got['raw_corners']);attr=attribute_gt(iou,{k:got['ids_'+k] for k in STAGES})
 assert len(attr)==len(old['GT_canonical']) and all(all(k in x['high_iou_count'] for k in STAGES) for x in attr)
 for t in (.5,.7):
  _,matched,flags=eval_native(got['final_corners'],got['final_scores'],old['GT_canonical'],t)
  assert len(flags)==len(old['GT_canonical'])
assert not torch.cuda.is_initialized()
(R/'REAL_POSTPROCESS_TEST.json').write_text(json.dumps({'passed':True,'sample_idx':590,'input':'P5-A historical saved heads','real_postprocessor':True,'real_nms_and_matcher':True,'real_full_anchor_gt_attribution':True,'real_model_forward':0,'CUDA_initialized':False},indent=2))
