import os,json,tempfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
import p5_worker as w
import p5_control as c
with tempfile.TemporaryDirectory(prefix='p5_empty_capture_') as td:
 root=Path(td); oldw=w.R; oldc=c.R; w.R=c.R=root
 try:
  sample=root/'input';sample.write_bytes(b'test')
  ready={'run_directory':str(root),'state':'PPU_READY_NOT_STARTED','budget':c.BUDGET,'passed':True,'locked_files':{},'samples':[{'sample_idx':i,'path':str(sample)} for i in range(18)]}
  try:c.verify_ready(False,ready)
  except RuntimeError as e:assert str(e)=='CAPTURE_DIRECTORY_MISSING_OR_NOT_EMPTY'
  else:raise AssertionError('missing capture directory accepted')
  # Exercise production writer directly from an initially empty filesystem.
  rec=w.save_case(590,0.,{'x':np.array([1,2],dtype=np.float32)},{'sample_idx':590})
  p=Path(rec['path']);assert p.is_file() and p.with_suffix('.json').is_file()
  assert rec['sha256']==w.sha_file(p) and rec['meta_sha256']==w.sha_file(p.with_suffix('.json'))
  try:w.save_case(590,0.,{'x':np.array([3])},{})
  except RuntimeError as e:assert str(e)=='REFUSE_OVERWRITE'
  else:raise AssertionError('overwrite accepted')
  try:c.verify_ready(False,ready)
  except RuntimeError as e:assert str(e)=='CAPTURE_DIRECTORY_MISSING_OR_NOT_EMPTY'
  else:raise AssertionError('nonempty capture directory accepted')
  (root/'capture'/'sample_590'/'alpha_0.0.npz').unlink()
  (root/'capture'/'sample_590'/'alpha_0.0.json').unlink()
  (root/'capture'/'sample_590').rmdir()
  assert c.verify_ready(False,ready)['passed']
 finally:w.R=oldw;c.R=oldc
assert not torch.cuda.is_initialized()
Path(__file__).with_name('EMPTY_CAPTURE_TEST.json').write_text(json.dumps({'passed':True,'initially_missing_capture':True,'missing_capture_rejected_before_launch':True,'production_writer_creates_parent':True,'nonempty_capture_rejected':True,'atomic_files_hash_verified':True,'overwrite_rejected':True,'CUDA_initialized':False,'model_forwards':0},indent=2))
