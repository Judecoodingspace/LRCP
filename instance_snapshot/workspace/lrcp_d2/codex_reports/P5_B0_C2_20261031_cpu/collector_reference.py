"""Temporary bounded hooks and exact-frame tracing. Never writes project source."""
import ast,sys,types,inspect
import numpy as np
import torch
from pilot_common import array,digest,STAGES


def node_line(path,qualname,predicate):
 tree=ast.parse(open(path).read())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==qualname[0])
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==qualname[1])
 matches=[n for n in ast.walk(fn) if predicate(n)]
 if len(matches)!=1:raise RuntimeError(f'AMBIGUOUS_TRACE_MARKER {qualname}: {len(matches)}')
 return matches[0].lineno

def assignment(n,target,call_part=None):
 return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets) and (call_part is None or call_part in ast.unparse(n.value))


def trace_markers():
 from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
 from opencood.models.sub_modules.defor_encoder_multi_scale import DeforAttn,DeforEncoderMultiScale
 p=inspect.getsourcefile(VoxelPostprocessor);q=('VoxelPostprocessor','post_process')
 raw=node_line(p,q,lambda n:assignment(n,'mask','torch.gt'))
 score=node_line(p,q,lambda n:isinstance(n,ast.If) and "'dir_preds'" in ast.unparse(n.test))
 direction=node_line(p,q,lambda n:isinstance(n,ast.If) and "'iou_preds'" in ast.unparse(n.test))
 projected=node_line(p,q,lambda n:isinstance(n,ast.If) and 'len(pred_box2d_list)' in ast.unparse(n.test))
 quality=node_line(p,q,lambda n:assignment(n,'keep_index','box_utils.nms_rotated'))
 # native NMS call completed at the first subsequent pred_box3d subscript assignment.
 tree=ast.parse(open(p).read());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='post_process')
 keep=min(n.lineno for n in ast.walk(f) if assignment(n,'pred_box3d_tensor','pred_box3d_tensor[keep_index]') and n.lineno>quality)
 ran=node_line(p,q,lambda n:isinstance(n,ast.Assert) and 'scores.shape[0]' in ast.unparse(n.test))
 a=inspect.getsourcefile(DeforAttn)
 attention=node_line(a,('DeforAttn','forward'),lambda n:assignment(n,'output','MSDeformAttnFunction.apply'))
 warp=node_line(a,('DeforEncoderMultiScale','forward'),lambda n:assignment(n,'feat','feat.flatten'))
 ref=node_line(a,('DeforEncoderMultiScale','forward'),lambda n:isinstance(n,ast.For) and 'enumerate(self.blocks)' in ast.unparse(n.iter))
 return {'post':{'raw':raw,'score':score,'direction':direction,'projected':projected,'quality':quality,'keep':keep,'range':ran},'attention':attention,'warp':warp,'reference':ref}


def summarize(t):
 a=array(t)
 if not np.isfinite(a).all():raise RuntimeError('NONFINITE_FEATURE')
 info={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest(a),'mean':float(a.mean(dtype=np.float64)),'std':float(a.std(dtype=np.float64)),'min':float(a.min()),'max':float(a.max())}
 maps={}
 if a.ndim==4:
  maps={'mean_map':a.mean(1),'std_map':a.std(1),'l2_map':np.sqrt((a.astype(np.float64)**2).sum(1)).astype(np.float32),'channel_mean':a.mean((0,2,3)),'channel_std':a.std((0,2,3))}
 return info,maps


class Collector:
 def __init__(self,model,postprocessor,batch,alpha,capture=True):
  self.model=model;self.pp=postprocessor;self.batch=batch;self.alpha=float(alpha);self.capture=capture
  self.arrays={};self.meta={'alpha':float(alpha),'features':{},'sampling':{},'offset_slots':{},'markers':trace_markers()};self.handles=[]
  self.slot=None;self.trace_count={};self._old_history=None
  from opencood.models.sub_modules.defor_encoder_multi_scale import DeforAttn,DeforEncoderMultiScale
  self.ppcode=self.pp.post_process.__func__.__code__;self.attcode=DeforAttn.forward.__code__;self.enccode=DeforEncoderMultiScale.forward.__code__
  self.names={id(mod):name for name,mod in model.named_modules()}
 def add(self,name,t):
  if name in self.arrays:raise RuntimeError('DUPLICATE_CAPTURE '+name)
  a=array(t)
  if a.dtype.kind=='f' and not np.isfinite(a).all():raise RuntimeError('NONFINITE_CAPTURE '+name)
  if sum(x.nbytes for x in self.arrays.values())+a.nbytes>32*2**20:raise RuntimeError('CAPTURE_RAM_LIMIT')
  self.arrays[name]=a
 def feature(self,name,t,roi=False):
  if name in self.meta['features']:raise RuntimeError('DUPLICATE_FEATURE '+name)
  info,maps=summarize(t);self.meta['features'][name]=info
  for k,v in maps.items():self.add('feature__'+name+'__'+k,v)
  if roi:
   g=array(self.batch['ego']['object_bbx_center'])[0]
   mask=array(self.batch['ego']['object_bbx_mask'])[0].astype(bool);g=g[mask]
   chosen=np.unique(np.linspace(0,len(g)-1,min(16,len(g)),dtype=int))
   a=array(t);patch=np.zeros((len(chosen),128,9,9),np.float32)
   for out,gi in enumerate(chosen):
    ix=int(np.floor((g[gi,0]+32)/.8));iy=int(np.floor((g[gi,1]+32)/.8))
    x0=max(ix-4,0);x1=min(ix+5,80);y0=max(iy-4,0);y1=min(iy+5,80)
    if x0<x1 and y0<y1:patch[out,:,y0-(iy-4):y1-(iy-4),x0-(ix-4):x1-(ix-4)]=a[0,:,y0:y1,x0:x1]
   self.add('fused_roi',patch);self.meta['roi_batch_object_indices']=chosen.tolist()
 def offset(self,module,args,out):
  if tuple(out.shape)!=(1,80,80,2):raise RuntimeError('METAFLOW_SHAPE')
  effective=out*self.alpha
  if self.capture:
   if self.slot is None:raise RuntimeError('UNKNOWN_REMOTE_SLOT')
   self.add(f'offset_raw_slot{self.slot}',out);self.add(f'offset_effective_slot{self.slot}',effective)
   md=self.batch['ego']['calibrate_data'][self.slot]
   self.meta['offset_slots'][str(self.slot)]={'physical_cav_ids':md['physical_cav_ids'],'valid_mask':array(md['valid_mask']).tolist(),'time_delay':array(md['time_delay']).tolist()}
  return effective
 def history(self,original,history,pairwise,agent,batch_indices=None):
  self.slot=int(agent)
  result=original(history,pairwise,agent,batch_indices)
  if self.capture:
   for level,t in enumerate(result):
    info,_=summarize(t);self.meta['features'][f'history_slot{agent}_level{level}']=info
  return result
 def fusion_pre(self,module,args):
  if tuple(args[4].shape)!=(1,6400,4,2):raise RuntimeError('PADDED_OFFSET_SHAPE')
  self.add('fusion_input_pred_offsets',args[4]);self.add('offset_GT_padded',args[3])
  for level,t in enumerate(args[0]):
   if t.shape[1]!=128 or tuple(t.shape[-2:])!=((80,80),(40,40),(20,20))[level]:raise RuntimeError('FUSION_INPUT_SHAPE')
   self.feature(f'fusion_input_level{level}',t)
 def sampling(self,frame):
  v=frame.f_locals;name=self.names.get(id(v['self']))
  if not name or not name.startswith('backbone.defor_encoder.'):raise RuntimeError('UNEXPECTED_ATTENTION_MODULE')
  loc=v['sampling_locations'].detach();weights=v['attention_weights'].detach();shapes=v['spatial_shapes']
  w=shapes[:,1].to(loc.dtype);h=shapes[:,0].to(loc.dtype)
  outside=((loc<0)|(loc>1)).any(-1)
  xx=loc[...,0]*w[None,None,None,:,None]-.5;yy=loc[...,1]*h[None,None,None,:,None]-.5
  supported=(xx>-1)&(xx<w[None,None,None,:,None])&(yy>-1)&(yy<h[None,None,None,:,None])
  dims=(0,1,2,4)
  summary={'shape':list(loc.shape),'spatial_shapes':array(shapes).tolist(),'outside_01_fraction_per_level':array(outside.float().mean(dims)).tolist(),'outside_kernel_fraction_per_level':array((~supported).float().mean(dims)).tolist(),'outside_kernel_weight_sum_per_level':array((weights*(~supported)).sum(dims)).tolist(),'weight_sum_per_level':array(weights.sum(dims)).tolist()}
  if name in self.meta['sampling']:return
  self.meta['sampling'][name]=summary
 def localtrace(self,frame,event,arg):
  if event!='line':return self.localtrace
  v=frame.f_locals;line=frame.f_lineno
  if frame.f_code==self.ppcode:
   markers=self.meta['markers']['post']
   for name,ln in markers.items():
    if line!=ln:continue
    if name in self.trace_count:continue
    self.trace_count[name]=1
    if name=='raw':
     self.add('decoded_before_dir',v['batch_box3d'][0]);self.add('scores_all',v['prob'][0])
    elif name=='score':self.add('score_mask',v['mask'][0])
    elif name=='direction':self.add('score_pass_centers_after_dir',v['boxes3d'])
    elif name=='projected':
     if v['pred_box3d_list']:self.add('score_pass_corners',v['pred_box3d_list'][0])
     else:self.add('score_pass_corners',np.empty((0,8,3),np.float32))
    elif name=='quality':
     self.add('large_mask',v['keep_index_1']);self.add('z_mask',v['keep_index_2']);self.add('quality_mask',v['keep_index']);self.add('nms_input_corners',v['pred_box3d_tensor']);self.add('nms_input_scores',v['scores'])
    elif name=='keep':self.add('native_nms_keep',v['keep_index'])
    elif name=='range':self.add('range_mask',v['mask'])
  elif frame.f_code==self.attcode and line==self.meta['markers']['attention']:self.sampling(frame)
  elif frame.f_code==self.enccode:
   if line==self.meta['markers']['warp']:self.feature(f'pose_warp_level{v["lvl"]}',v['feat'])
   elif line==self.meta['markers']['reference'] and 'reference' not in self.trace_count:
    ref=array(v['ref_2d']);self.meta['reference']={'shape':list(ref.shape),'min':ref.min((0,1)).tolist(),'max':ref.max((0,1)).tolist(),'sha256':digest(ref)}
    self.trace_count['reference']=1
  return self.localtrace
 def globaltrace(self,frame,event,arg):
  if event=='call' and frame.f_code in (self.ppcode,self.attcode,self.enccode):return self.localtrace
  return None
 def __enter__(self):
  if sys.gettrace() is not None:raise RuntimeError('EXISTING_TRACE_CONFLICT')
  self._old_history=self.model.get_batch_pillar_features
  original=self._old_history
  def history_wrapper(mod,history,pairwise,agent,batch_indices=None):return self.history(original,history,pairwise,agent,batch_indices)
  self.model.get_batch_pillar_features=types.MethodType(history_wrapper,self.model)
  self.handles.append(self.model.meta_flow.register_forward_hook(self.offset))
  if self.capture:
   self.handles.append(self.model.backbone.defor_encoder.register_forward_pre_hook(self.fusion_pre))
   self.handles.append(self.model.backbone.defor_encoder.register_forward_hook(lambda m,a,o:self.feature('fused',o,True)))
   self.handles.append(self.model.backbone.defor_encoder_single.register_forward_hook(lambda m,a,o:self.feature('single_control',o)))
   for i,block in enumerate(self.model.backbone.defor_encoder.blocks):
    def cap(m,a,o,i=i):
     t=o.transpose(1,2).reshape(1,128,80,80);self.feature(f'fusion_block{i}',t)
    self.handles.append(block.register_forward_hook(cap))
   sys.settrace(self.globaltrace)
  return self
 def __exit__(self,typ,value,tb):
  sys.settrace(None)
  for handle in self.handles:handle.remove()
  if self._old_history is not None:
   # Remove temporary instance override; original implementation is class method.
   del self.model.get_batch_pillar_features
  return False
 def finish(self,heads,final,gt):
  from opencood.utils import box_utils
  from opencood.data_utils.post_processor.voxel_postprocessor import limit_period
  for k,shape in [('cls_preds',(1,2,80,80)),('reg_preds',(1,14,80,80)),('dir_preds',(1,4,80,80))]:
   if tuple(heads[k].shape)!=shape or heads[k].dtype!=torch.float32:raise RuntimeError('HEAD_SHAPE_DTYPE')
   self.add(k,heads[k])
  self.add('final_corners',final[0]);self.add('final_scores',final[1]);self.add('GT_runtime',gt)
  if not self.capture:return
  if set(self.meta['markers']['post'])-set(self.trace_count):
   # Early score-empty return has no quality/NMS/range execution.
   if len(self.arrays.get('score_pass_corners',[])):
    raise RuntimeError('INCOMPLETE_POSTPROCESS_TRACE '+repr(self.trace_count))
   for k,v in {'large_mask':np.zeros(0,bool),'z_mask':np.zeros(0,bool),'quality_mask':np.zeros(0,bool),'nms_input_corners':np.empty((0,8,3),np.float32),'nms_input_scores':np.empty(0,np.float32),'native_nms_keep':np.empty(0,np.int32),'range_mask':np.empty(0,bool)}.items():self.add(k,v)
  if len(self.meta['sampling'])!=6 or 'reference' not in self.meta:raise RuntimeError('INCOMPLETE_SAMPLING_TRACE')
  if len(self.arrays['decoded_before_dir'])!=12800:raise RuntimeError('RAW_ANCHOR_COVERAGE')
  # Independent all-anchor completion on the original device, leaving native return untouched.
  d=torch.from_numpy(self.arrays['decoded_before_dir']).to(heads['reg_preds'].device)
  bins=self.pp.params['dir_args']['num_bins'];offset=self.pp.params['dir_args']['dir_offset']
  labels=heads['dir_preds'].permute(0,2,3,1).contiguous().reshape(-1,bins).argmax(-1)
  period=2*np.pi/bins
  d[...,6]=limit_period(d[...,6]-offset,0,period)+offset+period*labels.to(d.dtype)
  d[...,6]=limit_period(d[...,6],.5,2*np.pi)
  corners=box_utils.boxes_to_corners_3d(d,order=self.pp.params['order'])
  corners=box_utils.project_box3d(corners,self.batch['ego']['transformation_matrix'])
  self.add('raw_corners',corners)
  sm=self.arrays['score_mask'].astype(bool);actual=self.arrays['score_pass_corners']
  replay=self.arrays['raw_corners'][sm]
  if replay.shape!=actual.shape or not np.allclose(replay,actual,rtol=0,atol=1e-5):raise RuntimeError('ALL_ANCHOR_CORNER_REPLAY_MISMATCH')
  self.meta['all_anchor_corners_native_max_abs_error']=float(np.abs(replay-actual).max(initial=0.))
  # Use actual native retained-row corners for all stage IoU; no diagnostic rounding drift.
  self.arrays['raw_corners'][sm]=actual
  ids=np.arange(12800,dtype=np.int32);q=ids[sm][self.arrays['quality_mask'].astype(bool)]
  sorted_nms=self.arrays['nms_input_scores'].argsort()[::-1]
  keep=q[self.arrays['native_nms_keep']]
  stages=[ids,ids[sm],q,q[sorted_nms[:1000]],keep,keep[self.arrays['range_mask'].astype(bool)]]
  for name,ix in zip(STAGES,stages):self.add('ids_'+name,ix)
  if not np.array_equal(self.arrays['raw_corners'][stages[-1]],self.arrays['final_corners']):raise RuntimeError('FINAL_CORNER_LINEAGE')
  if not np.array_equal(self.arrays['scores_all'][stages[-1]],self.arrays['final_scores']):raise RuntimeError('FINAL_SCORE_LINEAGE')
