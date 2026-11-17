import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from pathlib import Path
from intervention import ReferenceIntervention
class SyntheticCross(torch.nn.Module):
 def forward(self,q,p,v,ref,shape):return ref.clone()
mods=[SyntheticCross() for _ in range(4)]
q=6400;levels=3;n=4
base=torch.linspace(0,1,q*2).view(1,q,1,2).expand(1,q,3,2).clone()
off=torch.arange(q*3*2,dtype=torch.float32).view(1,q,3,2).remainder(31)/10
ref=torch.cat([base]+[base+off[:,:,j:j+1,:]/torch.tensor([80.,80.]) for j in range(3)],dim=2)
old=ref.clone();sh=torch.tensor([[80,80],[40,40],[20,20]]*n);query=torch.zeros(1,q,1);v=query
with ReferenceIntervention(mods,[True,False,True],off) as hook:
 out=[m(query,query,v,ref,sh) for m in mods];assert hook.validate()
assert torch.equal(ref,old)
assert torch.equal(out[0][:,:,:3],ref[:,:,:3]) and torch.equal(out[0][:,:,6:9],ref[:,:,6:9])
assert torch.equal(out[0][:,:,3:6],base) and torch.equal(out[0][:,:,9:12],base)
assert all(torch.equal(x,ref) for x in out[1:])
with ReferenceIntervention(mods,[True,False,True],off,mode='passthrough') as hook:
 assert all(torch.equal(m(query,query,v,ref,sh),ref) for m in mods);assert hook.validate()
assert all(not m._forward_pre_hooks for m in mods) and not torch.cuda.is_initialized()
Path(__file__).with_name('FULL_QUERY_TEST.json').write_text(json.dumps({'passed':True,'query_count':6400,'synthetic':True,'valid_and_invalid_slots':True,'unchanged_shared_and_later_reference':True,'passthrough':True,'real_model_forwards':0,'CUDA_initialized':False},indent=2))
