"""Ephemeral native postprocessing trace; no constructor registers model hooks."""
import sys
import numpy as np
import torch
from collector_reference import Collector,trace_markers
from pilot_common import array,STAGES
from contracts import replay_native_nms
from opencood.utils import box_utils,common_utils
from opencood.data_utils.post_processor.voxel_postprocessor import limit_period

class MinimalTrace(Collector):
 def __init__(self,pp):
  self.arrays={};self.trace_count={};self.meta={'markers':trace_markers()};self.ppcode=pp.post_process.__func__.__code__;self.attcode=None;self.enccode=None

def collect_postprocess(pp,inputs,heads):
 """Trace native postprocessing only. Caller owns any model forward and alpha hook."""
 c=MinimalTrace(pp)
 if sys.gettrace() is not None:raise RuntimeError('EXISTING_TRACE')
 try:
  sys.settrace(c.globaltrace)
  with torch.no_grad():pred,scores=pp.post_process(inputs,{'ego':heads})
 finally:sys.settrace(None)
 if pred is None:
  device=heads['cls_preds'].device;pred=torch.empty((0,8,3),dtype=torch.float32,device=device);scores=torch.empty(0,dtype=torch.float32,device=device)
 a=c.arrays
 if 'score_pass_corners' not in a:a['score_pass_corners']=np.empty((0,8,3),np.float32)
 if not len(a['score_pass_corners']):
  for k,v in {'large_mask':np.zeros(0,bool),'z_mask':np.zeros(0,bool),'quality_mask':np.zeros(0,bool),'nms_input_corners':np.empty((0,8,3),np.float32),'nms_input_scores':np.empty(0,np.float32),'native_nms_keep':np.empty(0,np.int32),'range_mask':np.empty(0,bool)}.items():a.setdefault(k,v)
 for k,shape in [('cls_preds',(1,2,80,80)),('reg_preds',(1,14,80,80)),('dir_preds',(1,4,80,80))]:
  if tuple(heads[k].shape)!=shape or heads[k].dtype!=torch.float32:raise RuntimeError('INVALID_HEAD '+k)
  a[k]=array(heads[k])
 decoded=torch.from_numpy(a['decoded_before_dir'].copy()).to(heads['reg_preds'].device)
 bins=pp.params['dir_args']['num_bins'];offset=pp.params['dir_args']['dir_offset'];period=2*np.pi/bins
 labels=heads['dir_preds'].permute(0,2,3,1).contiguous().reshape(-1,bins).argmax(-1)
 decoded[...,6]=limit_period(decoded[...,6]-offset,0,period)+offset+period*labels.to(decoded.dtype)
 decoded[...,6]=limit_period(decoded[...,6],.5,2*np.pi)
 a['raw_corners']=array(box_utils.project_box3d(box_utils.boxes_to_corners_3d(decoded,order=pp.params['order']),inputs['ego']['transformation_matrix']))
 sm=a['score_mask'].astype(bool);error=float(np.abs(a['raw_corners'][sm]-a['score_pass_corners']).max(initial=0.))
 if error>1e-5:raise RuntimeError('ALL_ANCHOR_CORNER_REPLAY_MISMATCH')
 a['raw_corners'][sm]=a['score_pass_corners']
 q=np.flatnonzero(sm).astype(np.int32)[a['quality_mask'].astype(bool)];keep=q[a['native_nms_keep']]
 a['ids_raw_decoded']=np.arange(12800,dtype=np.int32);a['ids_score_pass']=np.flatnonzero(sm).astype(np.int32);a['ids_quality_pass']=q
 a['ids_nms_top1000']=q[a['nms_input_scores'].argsort()[::-1][:1000]];a['ids_nms_keep']=keep;a['ids_final_range']=keep[a['range_mask'].astype(bool)]
 a['final_corners']=array(pred);a['final_scores']=array(scores)
 if not np.array_equal(a['raw_corners'][a['ids_final_range']],a['final_corners']) or not np.array_equal(a['scores_all'][a['ids_final_range']],a['final_scores']):raise RuntimeError('FINAL_LINEAGE_MISMATCH')
 shadow=replay_native_nms(a['nms_input_corners'],a['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
 if not np.array_equal(shadow['keep'],a['native_nms_keep']):raise RuntimeError('NATIVE_NMS_REPLAY_MISMATCH')
 sup=np.full(12800,-1,np.int32);overlap=np.zeros(12800,np.float32)
 for z in shadow['suppressed']:
  i=q[z['removed_nms_input_idx']];sup[i]=q[z['suppressor_nms_input_idx']];overlap[i]=z['bev_iou_float32']
 a['suppressor']=sup;a['suppressor_overlap']=overlap;reason=np.ones(12800,np.int8)
 for stage,code in zip(STAGES[1:],range(2,7)):reason[a['ids_'+stage]]=code
 a['first_deletion_code']=reason
 return a,{'all_anchor_corners_max_abs_error':error,'native_NMS_replay_exact':True,'NMS_tied_score_entries':shadow['tied_score_entries']}
