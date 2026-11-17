import os,json,tempfile,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
from native_mock_support import Model,TestBackend,FakeObserver,rb,eval_native,yaml_utils,build_postprocessor,SimpleNamespace,run
R=Path(__file__).parent
class FailedControl(TestBackend):
 def baseline(self,sid):
  a=dict(self.synthetic_baselines[sid]);a['final_scores']=np.ones((len(a['final_scores'])+1,),np.float32);return a
model=Model().eval();h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(rb.MODEL)));pp=build_postprocessor(h['postprocess'],False)
def makegt(batch):
 with np.load(rb.BASE/f'P3A_20261014_120929_boxes/sample_{pp.current_sid}_alpha_1.0.npz',allow_pickle=False) as z:return torch.from_numpy(z['gt_boxes'].copy())
pp.generate_gt_bbx=makegt
b=FailedControl(model,pp,'cpu',lambda x,d:x,eval_native,observer_cls=rb.Observer);b.synthetic_baselines={}
tmp=Path(tempfile.mkdtemp(prefix='native_ctrl_fail_'))
try:
 try:run(tmp,R/'P5_SAMPLE_PROPOSAL_V2.csv',b)
 except RuntimeError as e:assert str(e).startswith('CONTROL_MISMATCH')
 else:raise AssertionError('failed control did not stop')
 attempts=json.loads((tmp/'budget.json').read_text())['attempts'];assert len(attempts)==1 and attempts[0]['kind']=='off' and not (tmp/'control_gate.json').exists()
 assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.cross) and not model.meta_flow._forward_hooks and not torch.cuda.is_initialized()
 (R/'CONTROL_FAILURE_TEST.json').write_text(json.dumps({'passed':True,'production_cpu_postprocess':True,'mock_model':True,'production_observer':True,'failed_controls':1,'core_forwards':0,'hooks_cleaned':True,'real_model_forward':0,'CUDA_initialized':False},indent=2))
finally:shutil.rmtree(tmp)
