import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
from pathlib import Path
from types import SimpleNamespace
from pilot_common import MODEL
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P4_C2_20261027_233036_301795')
from c1_reused import postprocess
r=Path(__file__).parent
pp=build_postprocessor(yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)))['postprocess'],False)
payload=torch.load('/workspace/lrcp_d2/codex_reports/P4_FULL_PREP_20261028_113714_782940/batches/sample_590.pt',map_location='cpu',weights_only=False,mmap=False);batch=payload['batch'];inp={'ego':{'anchor_box':batch['ego']['anchor_box'].clone(),'transformation_matrix':batch['ego']['transformation_matrix'].clone()}}
with np.load('/workspace/lrcp_d2/codex_reports/P5_A_CPU_FIX_20261028_storage/capture/sample_590/alpha_1.0.npz',allow_pickle=False) as old:
 geom={k:old[k].copy() for k in ('cls_preds','reg_preds','dir_preds','GT_canonical')};a,stats=postprocess(pp,inp,geom,geom)
 for k in ('score_mask','quality_mask','native_nms_keep','ids_final_range','suppressor'):assert np.array_equal(a[k],old[k]),k
 assert np.allclose(a['final_corners'],old['final_corners'],atol=1e-5,rtol=0) and np.allclose(a['final_scores'],old['final_scores'],atol=1e-6,rtol=0)
assert not torch.cuda.is_initialized()
(r/'INDEPENDENT_REPLAY_TEST.json').write_text(json.dumps({'passed':True,'sample':590,'real_historical_head_only':True,'model_forward':0,'CUDA_initialized':False},indent=2))
