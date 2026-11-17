"""Fixed physical-slot logits mask, installed only on four fusion cross blocks.
No parameter/buffer writes; logits masked out of place before native softmax.
"""
from rd3_common import *
import torch
from attention_observer import Observer

class FixedSlotMask:
 def __init__(self,modules,keep):
  self.modules=list(modules);self.keep=tuple(keep);self.handles=[];self.calls=[0]*4;self.raw={}
  require(len(self.modules)==4 and len(set(map(id,self.modules)))==4,'CROSS_MODULE_COUNT')
  require(len(keep)==3 and all(type(x) is bool for x in keep) and keep[0],'EGO_MUST_STAY_VALID')
 def __enter__(self):
  try:
   for i,m in enumerate(self.modules):
    def pre(mod,args,i=i):
     require(args[2].shape[0]==3 and args[4].shape[0]==9,'THREE_PHYSICAL_SLOTS_REQUIRED')
     require(mod.feature_level==3 and mod.max_num_agent==5,'NATIVE_LEVEL_PROTOCOL')
     require(torch.equal(args[4],args[4].new_tensor([[80,80],[40,40],[20,20]]*3)),'AGENT_SCALE_ORDER')
    def project(mod,args,out,i=i,m=m):
     require(torch.isfinite(out).all().item(),'NONFINITE_LOGITS')
     self.calls[i]+=1;self.raw[i]=array(out[:,QUERIES,:])
     if all(self.keep):return out
     mask=torch.tensor(list(self.keep)+[True]*(m.max_num_agent-3),device=out.device,dtype=torch.bool)
     mask=mask.repeat_interleave(m.feature_level*m.num_points).view(1,1,1,-1)
     v=out.view(out.shape[0],out.shape[1],m.num_heads,-1)
     return v.masked_fill(~mask,float('-inf')).reshape_as(out)
    self.handles.append(m.register_forward_pre_hook(pre));self.handles.append(m.attention_weights.register_forward_hook(project))
   return self
  except BaseException:self.__exit__();raise
 def __exit__(self,*exc):
  for h in self.handles:h.remove()
  self.handles=[]
 def validate(self):require(self.calls==[1]*4,'MASK_NOT_APPLIED_ALL_BLOCKS')

class EvidenceObserver(Observer):
 def __init__(self,modules,keep):
  super().__init__(modules,QUERIES);self.keep=keep;self.full_weights={};self.full_checks=[]
 def local(self,frame,event,arg):
  before=len(self.events);result=super().local(frame,event,arg)
  if len(self.events)>before:
   d=frame.f_locals;w=d['attention_weights'];block=self.events[-1]['block']
   require(torch.isfinite(w).all().item() and bool((w>=0).all().item()),'INVALID_GLOBAL_WEIGHTS')
   require(torch.allclose(w.sum((-1,-2)),torch.ones_like(w.sum((-1,-2))),rtol=0,atol=1e-5),'GLOBAL_NORMALIZATION')
   for slot,keep in enumerate(self.keep):
    if not keep:require(torch.count_nonzero(w[:,:,:,slot*3:(slot+1)*3,:]).item()==0,'MASKED_AGENT_CONTRIBUTION')
   self.full_weights[block]=array(w)
   self.full_checks.append({'block':block,'valid_agent_slots':[i for i,v in enumerate(self.keep) if v],'mask':self.keep,'all_queries':int(w.shape[1]),'all_heads':int(w.shape[2]),'all_levels':9,'all_points':int(w.shape[-1]),'max_sum_error':float((w.sum((-1,-2))-1).abs().max().item()),'masked_nonzero':0})
  return result
