"""Scoped first-cross reference intervention; no model construction or device initialization."""
import torch

class ReferenceIntervention:
 def __init__(self,modules,valid_remote,expected_offsets,mode='intervene',height=80,width=80,levels=3):
  self.modules=list(modules);self.valid=tuple(valid_remote);self.offsets=expected_offsets
  self.mode=mode;self.height=height;self.width=width;self.levels=levels;self.handles=[];self.events=[];self.original=None
  if len(self.modules)!=4 or len({id(x) for x in self.modules})!=4 or mode not in ('intervene','passthrough'):raise ValueError('BAD_CONFIGURATION')
  if any(type(x) is not bool for x in self.valid):raise ValueError('INVALID_SLOT_MASK')
 def before(self,i):
  def hook(mod,args):
   if len(self.events)!=i:raise RuntimeError('CROSS_CALL_ORDER_OR_DUPLICATE')
   if len(args)<5:raise RuntimeError('CROSS_ARGUMENTS')
   ref=args[3];shapes=args[4];L=self.levels;N=1+len(self.valid)
   if ref.ndim!=4 or ref.shape!=(1,self.height*self.width,N*L,2) or tuple(shapes.shape)!=(N*L,2):raise RuntimeError('REFERENCE_SHAPE')
   if tuple(self.offsets.shape)!=(1,self.height*self.width,N-1,2):raise RuntimeError('OFFSET_SHAPE')
   if ref.dtype!=self.offsets.dtype or ref.device!=self.offsets.device or not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(self.offsets).all()):raise RuntimeError('REFERENCE_VALUE')
   base=ref[:,:,:1,:]
   if not torch.equal(ref[:,:,:L,:],base.expand_as(ref[:,:,:L,:])):raise RuntimeError('EGO_REFERENCE_INCONSISTENT')
   if i==0:
    normal=ref.new_tensor([self.width,self.height]);expected=base.unsqueeze(2)+self.offsets.unsqueeze(3)/normal
    for j,valid in enumerate(self.valid):
     if valid and not torch.allclose(ref[:,:,L*(j+1):L*(j+2),:],expected[:,:,j,:,:].expand(-1,-1,L,-1),rtol=0,atol=1e-6):raise RuntimeError('REMOTE_SHIFT_MISMATCH')
    self.original=ref.detach().clone()
    if self.mode=='intervene':
     new=ref.clone()
     for j,valid in enumerate(self.valid):
      if valid:new[:,:,L*(j+1):L*(j+2),:]=base.expand(-1,-1,L,-1)
    else:new=ref
    self.events.append(i)
    return (*args[:3],new,*args[4:])
   if not torch.equal(ref,self.original):raise RuntimeError('LATER_REFERENCE_CHANGED')
   self.events.append(i)
  return hook
 def __enter__(self):
  try:
   for i,m in enumerate(self.modules):self.handles.append(m.register_forward_pre_hook(self.before(i)))
   return self
  except BaseException:
   for h in self.handles:h.remove()
   raise
 def __exit__(self,typ,val,tb):
  for h in self.handles:h.remove()
  self.handles=[]
 def validate(self):
  if self.events!=[0,1,2,3]:raise RuntimeError('MISSING_CROSS_CALL')
  return True
