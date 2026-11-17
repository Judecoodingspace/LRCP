"""Unintegrated CPU-testable observer draft; no model loading, no device or launch entry.

Do not use as a full P5 runner: call identity/slot mapping, output equality, resource
watchdog and atomic capture integration are not implemented here.
"""
import sys
import torch
class RestrictedAttentionObserver:
 def __init__(self,code,target_line,module_names,query_indices,limit=256):
  self.code=code;self.target_line=target_line;self.module_names=dict(module_names)
  self.query_indices=tuple(int(i) for i in query_indices);self.limit=limit;self.events=[];self.old_trace=None
  if not self.query_indices or len(set(self.query_indices))!=len(self.query_indices) or len(self.query_indices)>limit:raise ValueError('query selection is empty, duplicated, or exceeds the bound')
 def local(self,frame,event,arg):
  if event=='line' and frame.f_lineno==self.target_line:
   v=frame.f_locals;name=self.module_names.get(id(v.get('self')))
   if name is None:return self.local
   vals={}
   for key in ('reference_points','sampling_offsets','sampling_locations','attention_weights'):
    t=v[key]
    if t.ndim<2 or min(self.query_indices)<0 or max(self.query_indices)>=t.shape[1]:raise ValueError('unexpected query shape')
    index=torch.as_tensor(self.query_indices,dtype=torch.long,device=t.device)
    vals[key]=t.detach().index_select(1,index).to(device='cpu').clone()
   vals['spatial_shapes']=v['spatial_shapes'].detach().to(device='cpu').clone()
   self.events.append((name,vals))
  return self.local
 def global_trace(self,frame,event,arg):return self.local if event=='call' and frame.f_code is self.code else None
 def __enter__(self):
  self.old_trace=sys.gettrace()
  if self.old_trace is not None:raise RuntimeError('existing trace must not be replaced')
  sys.settrace(self.global_trace);return self
 def __exit__(self,*args):sys.settrace(self.old_trace)
