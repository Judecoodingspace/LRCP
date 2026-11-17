import json,sys,os,tempfile
from pathlib import Path
import p5_control as p
assert 'torch' not in sys.modules
original=p.verify_ready
p.verify_ready=lambda full:{'passed':True,'full_hash_check':full,'CUDA_initialized':False}
try:
 out=p.launch(dry=True,approve=False);assert out['passed'] and out['dry_run'] and not out['started'] and 'torch' not in sys.modules
finally:p.verify_ready=original
try:p.launch(dry=False,approve=False)
except RuntimeError as e:assert 'APPROVAL_REQUIRED' in str(e)
else:raise AssertionError('missing manual approval accepted')
assert not (p.R/'PPU_LAUNCH.json').exists() and 'torch' not in sys.modules
(p.R/'LAUNCHER_TEST.json').write_text(json.dumps({'passed':True,'dry_run_no_torch_import':True,'manual_approval_negative':True,'worker_not_started':True,'model_forwards':0},indent=2))
