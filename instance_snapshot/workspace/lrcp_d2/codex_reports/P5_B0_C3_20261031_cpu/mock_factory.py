"""Explicit MOCK_ONLY CPU model; reuses C2 native observer and full collectors."""
def build_mock():
 import sys
 from runtime_common import C2
 sys.path.insert(0,str(C2))
 import native_mock_support as m
 import real_backend as rb
 import numpy as np,torch
 from types import SimpleNamespace
 from opencood.hypes_yaml import yaml_utils
 from opencood.data_utils.post_processor import build_postprocessor
 model=m.Model().eval()
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(rb.MODEL)))
 pp=build_postprocessor(h['postprocess'],False)
 def makegt(batch):
  with np.load(rb.BASE/f'P3A_20261014_120929_boxes/sample_{pp.current_sid}_alpha_1.0.npz',allow_pickle=False) as z:return torch.from_numpy(z['gt_boxes'].copy())
 pp.generate_gt_bbx=makegt
 b=m.TestBackend(model,pp,'cpu',lambda x,d:x,m.eval_native,observer_cls=rb.Observer)
 b.synthetic_baselines={}
 return b
