import os,sys,json,torch,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent
torch.manual_seed(17)
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.sampling_offsets=torch.nn.Linear(4,2);self.attention_weights=torch.nn.Linear(4,2)
 def forward(self,query,query_pos,reference_points,spatial_shapes):
  query=query+query_pos
  sampling_offsets=self.sampling_offsets(query)
  attention_weights=self.attention_weights(query).softmax(-1)
  sampling_locations=reference_points+sampling_offsets/spatial_shapes
  return sampling_locations*attention_weights
m=Toy().eval();q=torch.randn(1,4);p=torch.randn(1,4);ref=torch.randn(1,2);shape=torch.tensor([4.,8.]);rng=torch.random.get_rng_state().clone()
with torch.no_grad():expected=m(q,p,ref,shape)
records={};handles=[]
def hook(label):
 def run(module,args,out):records[label+'_input']=args[0].detach().clone();records[label+'_output']=out.detach().clone()
 return run
for name in ['sampling_offsets','attention_weights']:handles.append(getattr(m,name).register_forward_hook(hook(name)))
# Restricted frame observer only, no replacement of tensor or kernel.
code=Toy.forward.__code__
def local(frame,event,arg):
 if event=='line' and 'sampling_locations' in frame.f_locals:
  for k in ('reference_points','sampling_offsets','sampling_locations','attention_weights'):records['frame_'+k]=frame.f_locals[k].detach().clone()
 return local
def global_trace(frame,event,arg):return local if event=='call' and frame.f_code==code else None
assert sys.gettrace() is None
try:
 sys.settrace(global_trace)
 with torch.no_grad():actual=m(q,p,ref,shape)
finally:
 sys.settrace(None)
 for h in handles:h.remove()
assert torch.equal(actual,expected) and torch.equal(torch.random.get_rng_state(),rng)
assert torch.equal(records['sampling_offsets_input'],q+p) and torch.equal(records['attention_weights_input'],q+p)
assert torch.equal(records['frame_sampling_locations'],records['frame_reference_points']+records['frame_sampling_offsets']/shape)
assert torch.allclose(records['frame_attention_weights'].sum(-1),torch.ones(1))
assert not m.sampling_offsets._forward_hooks and not m.attention_weights._forward_hooks and sys.gettrace() is None and not torch.cuda.is_initialized()
(R/'SYNTHETIC_HOOK_TEST.json').write_text(json.dumps({'passed':True,'scope':'toy CPU module only; not LRCP model or actual PPU kernel','module_projection_hooks':True,'function_frame_observer':True,'outputs_equal':True,'rng_unchanged':True,'hooks_removed':True,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
