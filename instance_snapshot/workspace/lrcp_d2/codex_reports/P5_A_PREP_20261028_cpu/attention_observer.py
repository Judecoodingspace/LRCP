"""Temporary observation only. Does not load a model or initialize a device."""
import sys,ast,inspect,textwrap
import numpy as np
import torch

def call_line(method):
 source=inspect.getsource(method);start=inspect.getsourcelines(method)[1];tree=ast.parse(textwrap.dedent(source))
 hits=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='apply' and isinstance(x.func.value,ast.Name) and x.func.value.id=='MSDeformAttnFunction']
 if len(hits)!=1:raise RuntimeError('ATTENTION_TRACE_TARGET_NOT_UNIQUE')
 return start+hits[0].lineno-1

def to_numpy(x):return x.detach().cpu().numpy().copy()
class Observer:
 def __init__(self,modules,query_indices,target_line=None):
  self.modules=list(modules);self.q=tuple(sorted(set(map(int,query_indices))));self.events=[];self.projections={};self.handles=[];self.prev=None
  if len(self.modules)!=4 or not self.q or len(self.q)>256 or min(self.q)<0 or max(self.q)>=6400:raise RuntimeError('INVALID_OBSERVER_TARGETS')
  if len({id(x) for x in self.modules})!=4:raise RuntimeError('DUPLICATE_ATTENTION_MODULE')
  self.names={id(m):i for i,m in enumerate(self.modules)};self.code=self.modules[0].forward.__func__.__code__
  if any(m.forward.__func__.__code__ is not self.code for m in self.modules):raise RuntimeError('ATTENTION_CODE_MISMATCH')
  self.target=target_line if target_line is not None else call_line(self.modules[0].forward.__func__)
 def make_hook(self,index,kind):
  def hook(module,args,out):
   q=args[0];idx=torch.as_tensor(self.q,device=q.device);self.projections[(index,kind)]={'input':to_numpy(q.index_select(1,idx)),'output':to_numpy(out.index_select(1,idx))}
  return hook
 def local(self,frame,event,arg):
  if event!='line' or frame.f_lineno!=self.target:return self.local
  d=frame.f_locals;block=self.names.get(id(d.get('self')))
  if block is None:return self.local
  if any(e['block']==block for e in self.events):raise RuntimeError('DUPLICATE_CROSS_CALL')
  ref=d['reference_points'];offs=d['sampling_offsets'];loc=d['sampling_locations'];weights=d['attention_weights'];shapes=d['spatial_shapes']
  if ref.shape[1]!=6400 or loc.shape[1]!=6400 or ref.shape[2]!=shapes.shape[0] or loc.shape[3]!=shapes.shape[0]:raise RuntimeError('ATTENTION_SHAPE_MISMATCH')
  if not all(torch.isfinite(t).all().item() for t in (ref,offs,loc,weights)):raise RuntimeError('NONFINITE_ATTENTION')
  normal=torch.stack([shapes[:,1],shapes[:,0]],-1).to(loc.dtype)
  index=torch.as_tensor(self.q,device=loc.device);l=loc.index_select(1,index);o=offs.index_select(1,index);re=ref.index_select(1,index);w=weights.index_select(1,index)
  expect=re[:,:,None,:,None,:]+o/normal[None,None,None,:,None,:]
  if not torch.equal(l,expect):raise RuntimeError('SAMPLING_LOCATION_FORMULA_MISMATCH')
  if not torch.allclose(w.sum((-1,-2)),torch.ones_like(w.sum((-1,-2))),rtol=1e-5,atol=1e-6):raise RuntimeError('ATTENTION_NORMALIZATION_MISMATCH')
  # One scalar per level, full query/head/point population; no GT guided masking.
  summaries=[]
  for level in range(shapes.shape[0]):
   v=loc[:,:,:,level,:,:];h=int(shapes[level,0]);ww=int(shapes[level,1]);outside=((v<0)|(v>1)).any(-1);xx=v[...,0]*ww-.5;yy=v[...,1]*h-.5;unsupported=(xx<=-1)|(xx>=ww)|(yy<=-1)|(yy>=h)
   summaries.append({'level':level,'H':h,'W':ww,'count':v.numel()//2,'normalized_oob':int(outside.sum().item()),'kernel_unsupported':int(unsupported.sum().item())})
  self.events.append({'block':block,'reference':to_numpy(re),'offset':to_numpy(o),'location':to_numpy(l),'weights':to_numpy(w),'spatial_shapes':to_numpy(shapes),'summary':summaries})
  return self.local
 def global_trace(self,frame,event,arg):return self.local if event=='call' and frame.f_code is self.code else None
 def __enter__(self):
  self.prev=sys.gettrace()
  if self.prev is not None:raise RuntimeError('EXISTING_TRACE')
  try:
   for i,m in enumerate(self.modules):
    for kind in ('sampling_offsets','attention_weights'):self.handles.append(getattr(m,kind).register_forward_hook(self.make_hook(i,kind)))
   sys.settrace(self.global_trace);return self
  except BaseException:
   for h in self.handles:h.remove()
   raise
 def __exit__(self,*exc):
  sys.settrace(self.prev)
  for h in self.handles:h.remove()
 def validate(self):
  if sorted(e['block'] for e in self.events)!=[0,1,2,3] or len(self.projections)!=8:raise RuntimeError('MISSING_ATTENTION_CAPTURE')
  return {'query_ids':list(self.q),'blocks':[{'block':e['block'],'summary':e['summary']} for e in self.events]}
