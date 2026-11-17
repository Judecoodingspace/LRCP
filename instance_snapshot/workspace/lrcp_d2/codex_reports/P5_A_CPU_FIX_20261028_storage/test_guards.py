import os,sys,json,tempfile,pathlib,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
import p5_control as control
import p5_worker as worker
from attention_observer import Observer
assert not torch.cuda.is_initialized()
# The full model is never constructed; only synthetic arrays exercise disk guards.
with tempfile.TemporaryDirectory(prefix='p5_guards_') as tmp:
 root=pathlib.Path(tmp);(root/'capture').mkdir();old=worker.R;worker.R=root
 try:
  worker.state.update(attempted=40)
  try:worker.guard(__import__('time').monotonic())
  except RuntimeError as e:assert str(e)=='FORWARD_LIMIT'
  else:raise AssertionError('40 forward bound not enforced')
  worker.state.update(attempted=0)
  worker.atomic(root/'state.json',{'attempted':1});assert json.loads((root/'state.json').read_text())['attempted']==1
  rec=worker.save_case(1,0.,{'x':np.array([1,2],np.float32)},{'passed':True});assert pathlib.Path(rec['path']).exists()
  try:worker.save_case(1,0.,{'x':np.array([3])},{})
  except RuntimeError as e:assert str(e)=='REFUSE_OVERWRITE'
  else:raise AssertionError('overwrite not rejected')
 finally:worker.R=old
try:control.launch(dry=False,approve=False)
except RuntimeError as e:assert 'APPROVAL_REQUIRED' in str(e)
else:raise AssertionError('unauthorized launch not refused')
assert not (control.R/'PPU_LAUNCH.json').exists() and not torch.cuda.is_initialized()
(control.R/'GUARD_TEST.json').write_text(json.dumps({'passed':True,'max_40_enforced':True,'atomic_state_and_npz':True,'overwrite_rejected':True,'manual_approval_required':True,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
