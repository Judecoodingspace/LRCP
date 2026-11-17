import os,sys,ast,json,inspect
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import real_backend as b
import runner_engine as e
from attention_observer import Observer
from minimal_collector import collect_postprocess
from contracts import attribute_gt,replay_native_nms
sys.path.insert(0,str(b.ROOT.parent/'P4_C2_20261027_233036_301795'))
from c1_reused import eval_native
files=[b.P5A/'p5_worker.py',b.P5A/'attention_observer.py',b.P5A/'validate_cpu.py',b.FULL/'scientific_validator.py',b.ROOT.parent/'P4_C2_20261027_233036_301795'/'c1_reused.py',b.CHECKPOINT,b.MODEL/'config.yaml',b.P5A/'P5_READY.json',b.FULL/'INPUT_MANIFEST.json']
for p in files:
 if not p.is_file():raise RuntimeError('MISSING_DEPENDENCY '+str(p))
for p in b.ROOT.glob('*.py'):ast.parse(p.read_text())
assert tuple(inspect.signature(collect_postprocess).parameters)==('pp','inputs','heads')
assert tuple(inspect.signature(eval_native).parameters)[:4]==('pred','scores','gt','threshold')
assert tuple(inspect.signature(attribute_gt).parameters)==('iou_gt_candidate','stage_candidate_ids','threshold')
assert hasattr(Observer,'validate') and all(hasattr(b.LRCPBackend,n) for n in ('build_real','prepare','forward','collect','baseline','cleanup'))
assert not torch.cuda.is_initialized()
z={'passed':True,'dependency_paths':[str(p) for p in files],'signatures':{k:str(inspect.signature(v)) for k,v in {'collect_postprocess':collect_postprocess,'eval_native':eval_native,'attribute_gt':attribute_gt,'replay_native_nms':replay_native_nms,'runner_engine.run':e.run}.items()},'backend_module_imported':True,'CUDA_initialized':False,'real_model_forward':0,'limitations':'No real model constructed or device forward; static checks do not prove runtime scientific correctness.'}
(b.ROOT/'BACKEND_STATIC_VALIDATION.json').write_text(json.dumps(z,indent=2))
