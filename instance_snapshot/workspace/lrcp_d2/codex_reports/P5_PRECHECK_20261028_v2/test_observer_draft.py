import os,sys,inspect,ast,json,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from attention_capture_draft import RestrictedAttentionObserver
R=Path(__file__).resolve().parent
class Toy(torch.nn.Module):
 def forward(self,reference_points,sampling_offsets,attention_weights,spatial_shapes):
  sampling_locations=reference_points+sampling_offsets
  return sampling_locations*attention_weights
src=inspect.getsource(Toy.forward);line=Toy.forward.__code__.co_firstlineno+2
m=Toy();ref=torch.tensor([[[1.,2.],[2.,3.]]]);off=torch.tensor([[[.5,.25],[.1,.2]]]);w=torch.tensor([[[.5,.5],[.7,.3]]]);shape=torch.tensor([[80,80]])
with torch.no_grad():base=m(ref,off,w,shape)
state=torch.random.get_rng_state().clone()
with RestrictedAttentionObserver(Toy.forward.__code__,line,{id(m):'test'},[1]) as ob:
 with torch.no_grad():out=m(ref,off,w,shape)
assert torch.equal(base,out) and len(ob.events)==1
assert torch.equal(ob.events[0][1]['sampling_locations'],(ref+off)[:,1:2])
assert torch.equal(torch.random.get_rng_state(),state) and sys.gettrace() is None
try:
 with RestrictedAttentionObserver(Toy.forward.__code__,line,{id(m):'test'},[9]):m(ref,off,w,shape)
except ValueError:pass
else:raise AssertionError('bounds did not fail')
assert sys.gettrace() is None and not torch.cuda.is_initialized()
(R/'OBSERVER_DRAFT_TEST.json').write_text(json.dumps({'passed':True,'scope':'synthetic CPU only','unchanged_output_rng':True,'bounded_queries':True,'trace_restored_on_exception':True,'actual_LRCP_unverified':True,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
