import os,json,sys,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from attention_observer import Observer
class MSDeformAttnFunction:
 @staticmethod
 def apply(value,shapes,start,locations,weights,step):return locations.sum()*0+weights.sum()*0
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.sampling_offsets=torch.nn.Linear(4,2);self.attention_weights=torch.nn.Linear(4,1)
 def forward(self,query,query_pos,value,reference_points,spatial_shapes):
  query=query+query_pos
  sampling_offsets=self.sampling_offsets(query).view(1,6400,1,1,1,2)
  attention_weights=self.attention_weights(query).view(1,6400,1,1,1).softmax(-1)
  sampling_locations=reference_points[:,:,None,:,None,:]+sampling_offsets/torch.tensor([80.,80.])
  level_start_index=torch.zeros(1,dtype=torch.long)
  output = MSDeformAttnFunction.apply(value,spatial_shapes,level_start_index,sampling_locations,attention_weights,64)
  return query+output
mods=[Toy().eval() for _ in range(4)];q=torch.randn(1,6400,4);p=torch.randn_like(q);ref=torch.full((1,6400,1,2),.5);sh=torch.tensor([[80,80]]);value=q
with torch.no_grad():base=[m(q,p,value,ref,sh) for m in mods]
rng=torch.random.get_rng_state().clone()
with Observer(mods,[0,80,6399]) as ob:
 with torch.no_grad():out=[m(q,p,value,ref,sh) for m in mods]
assert all(torch.equal(a,b) for a,b in zip(base,out)) and torch.equal(rng,torch.random.get_rng_state());assert len(ob.validate()['blocks'])==4
assert all(not m.sampling_offsets._forward_hooks and not m.attention_weights._forward_hooks for m in mods) and sys.gettrace() is None
try:
 with Observer(mods,[0]) as bad:
  raise ValueError('failure cleanup')
except ValueError:pass
assert sys.gettrace() is None and all(not m.sampling_offsets._forward_hooks for m in mods) and not torch.cuda.is_initialized()
Path(__file__).with_name('OBSERVER_TEST.json').write_text(json.dumps({'passed':True,'actual_model_forward':False,'CUDA_initialized':False,'four_blocks':True,'unchanged_output_rng':True,'hooks_cleaned_on_exception':True,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
