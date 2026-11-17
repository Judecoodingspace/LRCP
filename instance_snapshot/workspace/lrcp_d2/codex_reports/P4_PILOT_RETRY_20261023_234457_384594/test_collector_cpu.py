"""Stage A tests: actual CPU postprocessor, exact-frame trace, NMS, identity, guards."""
import ast,sys,json,inspect,copy,dis
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from pilot_common import *
from collector import Collector,trace_markers
from contracts import attribute_gt,replay_native_nms
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils import box_utils,common_utils
results=[]
def check(name,condition):
 assert condition,name
 results.append({'name':name,'passed':True})
for p in R.glob('*.py'):
 tree=ast.parse(p.read_text());compile(tree,str(p),'exec');check('AST_compile_'+p.name,True)
check('selection_18_unique_821_first',len(selections())==18 and len({x['sample_idx'] for x in selections()})==18 and selections()[0]['sample_idx']=='821')
h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)))
pp=build_postprocessor(h['postprocess'],False)
markers=trace_markers()
lines={i for _,i in dis.findlinestarts(pp.post_process.__func__.__code__)}
check('all_post_markers_executable',all(v in lines for v in markers['post'].values()))
check('fixed_thresholds',pp.params['target_args']['score_threshold']==.2 and pp.params['nms_thresh']==.15 and pp.params['dir_args']['num_bins']==2)
class Dummy(torch.nn.Module):
 def __init__(self):super().__init__();self.meta_flow=torch.nn.Identity()
 def get_batch_pillar_features(self,*args):return []
model=Dummy()
anchors=torch.from_numpy(pp.generate_anchor_box())
check('anchor_shape',tuple(anchors.shape)==(80,80,2,7))
for empty in (False,True):
 cls=torch.full((1,2,80,80),-20.,dtype=torch.float32)
 if not empty:
  cls[0,:,40,40]=2.;cls[0,0,35,35]=2.
 heads={'cls_preds':cls,'reg_preds':torch.zeros(1,14,80,80),'dir_preds':torch.zeros(1,4,80,80)}
 ego={'anchor_box':anchors,'transformation_matrix':torch.eye(4)};batch={'ego':ego}
 c=Collector(model,pp,batch,1.,True)
 assert sys.gettrace() is None
 try:
  sys.settrace(c.globaltrace)
  pred,scores=pp.post_process(batch,{'ego':heads})
 finally:sys.settrace(None)
 if pred is None:pred=torch.empty(0,8,3);scores=torch.empty(0)
 c.meta['sampling']={str(k):{} for k in range(6)};c.meta['reference']={'CPU_mock':True}
 c.finish(heads,(pred,scores),torch.empty(0,8,3))
 check('actual_postprocessor_trace_complete_empty_'+str(empty),all('ids_'+n in c.arrays for n in STAGES))
 check('all_anchor_coverage_empty_'+str(empty),len(c.arrays['raw_corners'])==12800)
 check('score_strict_boundary_empty_'+str(empty),np.array_equal(c.arrays['score_mask'],c.arrays['scores_all']>.2))
 nr=replay_native_nms(c.arrays['nms_input_corners'],c.arrays['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
 check('actual_cpu_native_NMS_equal_empty_'+str(empty),np.array_equal(nr['keep'],c.arrays['native_nms_keep']))
 check('final_anchor_identity_empty_'+str(empty),np.array_equal(c.arrays['raw_corners'][c.arrays['ids_final_range']],c.arrays['final_corners']))
 if not empty:
  # Float32 tied logits give duplicate/perpendicular score-tied boxes; check suppressor.
  check('NMS_suppressor_and_ties',bool(nr['suppressed']) and nr['tied_score_entries']>=1)
  for a,b in zip(STAGES,STAGES[1:]):check('stage_nested_'+a,set(c.arrays['ids_'+b]).issubset(set(c.arrays['ids_'+a])))
mat=np.zeros((7,6),np.float32)
for i in range(1,7):mat[i,i-1]=.7
ids=dict(zip(STAGES,[np.arange(k,6) for k in range(6)]))
out=attribute_gt(mat,ids)
check('each_six_stage_deletion_classes_plus_survivor',len({x['availability_class'] for x in out})==7)
try:attribute_gt(np.full((1,6),np.nan),ids);raise AssertionError('accepted nonfinite')
except ValueError:check('invalid_geometry_rejected',True)
gt=np.arange(72,dtype=np.float32).reshape(3,8,3);m,mm=gt_mapping(gt,gt[[2,0,1]])
check('GT_mapping_unique_permutation',np.array_equal(m,[1,2,0]))
x={'a':torch.arange(9).reshape(3,3),'n':np.arange(5,dtype=np.float32),'none':None,'list':[2,'s']}
y=copy.deepcopy(x);check('deep_copy_identical_values',digest(x)==digest(y));y['a'][0,0]=99
check('deep_copy_no_mutable_alias',x['a'][0,0].item()==0)
check('CUDA_not_initialized',not torch.cuda.is_initialized())
write_json(R/'CPU_COLLECTOR_CHECKS.json',{'passed':True,'checks_count':len(results),'checks':results,'trace_markers':markers,'CUDA_initialized':torch.cuda.is_initialized(),'real_model_forwards':0})
print('CPU_COLLECTOR_CHECKS_PASSED',len(results))
