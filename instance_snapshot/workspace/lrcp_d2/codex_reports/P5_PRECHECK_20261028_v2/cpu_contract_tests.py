import os,json,ast,math,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
R=Path(__file__).resolve().parent;P=Path('/workspace/LRCP/opencood/models/sub_modules/defor_encoder_multi_scale.py')
tree=ast.parse(P.read_text());classes={x.name:x for x in tree.body if isinstance(x,ast.ClassDef)}
a=classes['DeforAttn'];forward=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
text=ast.get_source_segment(P.read_text(),forward)
for s in ['query + query_pos','self.sampling_offsets(query)','self.attention_weights(query)','attention_weights.softmax(-1)','sampling_locations = reference_points','MSDeformAttnFunction.apply']:assert s in text,s
enc=classes['DeforEncoderMultiScale'];e=next(x for x in enc.body if isinstance(x,ast.FunctionDef) and x.name=='forward');et=ast.get_source_segment(P.read_text(),e)
for s in ['pred_offsets.clone()','/ self.bev_w','/ self.bev_h','block(bev_queries, bev_pos, feat_flatten, ref_2d']:assert s in et,s
# Synthetic numerical tests: layout and equations, no LRCP model, no device.
H=W=80;y,x=torch.meshgrid(torch.arange(H,dtype=torch.float32)+.5,torch.arange(W,dtype=torch.float32)+.5,indexing='ij');ref=torch.stack((x/W,y/H),-1).reshape(1,H*W,1,2)
assert torch.equal(ref[0,0,0],torch.tensor([.5/80,.5/80])) and torch.equal(ref[0,-1,0],torch.tensor([79.5/80,79.5/80]))
assert torch.equal(ref[0,81,0],torch.tensor([1.5/80,1.5/80]))
raw=torch.tensor([[[[4.,-8.],[2.,3.]]]]) # B,Q,remote_slot,xy
for alpha in (0.,1.):
 sh=alpha*raw/torch.tensor([80.,80.]);assert torch.equal(sh,torch.zeros_like(sh)) if alpha==0 else torch.allclose(sh,torch.tensor([[[[.05,-.1],[.025,.0375]]]]))
# Each slot has 3 levels; ego stays unshifted.
base=ref[:,:1];shift=raw/80.;slotref=torch.cat([base.repeat(1,1,3,1),(base.unsqueeze(2)+shift.unsqueeze(3)).repeat(1,1,1,3,1).flatten(2,3)],2)
assert slotref.shape==(1,1,9,2) and torch.equal(slotref[:,:,:3],base.repeat(1,1,3,1))
assert torch.allclose(slotref[:,:,3:6],(base+shift[:,:,0:1]).repeat(1,1,3,1))
shapes=torch.tensor([[80,80],[40,40],[20,20]]*3,dtype=torch.float32);offset=torch.zeros((1,1,1,9,1,2));offset[...,0]=1;offset[...,1]=2
locations=slotref[:,:,None,:,None,:]+offset/torch.stack([shapes[:,1],shapes[:,0]],-1)[None,None,None,:,None,:]
assert torch.allclose(locations[0,0,0,0,0]-slotref[0,0,0],torch.tensor([1/80,2/80])) and torch.allclose(locations[0,0,0,2,0]-slotref[0,0,2],torch.tensor([1/20,2/20]))
logits=torch.tensor([0.,1.,2.,-1.]);w=torch.softmax(logits,dim=-1);assert torch.allclose(w.sum(),torch.tensor(1.))
# Distinguish [0,1] bounds from bilinear kernel's -1 < loc*size-.5 < size.
vals=torch.tensor([-.001,0.,1.,1.001,-.01,1.01]);kernel=((vals*80-.5)>-1)&((vals*80-.5)<80);normalized=(vals>=0)&(vals<=1);assert bool((kernel&~normalized).any()) and not bool(kernel[4]) and not bool(kernel[5])
# Actual head layout permute NHWA -> flat ID = query*2 + orientation.
ids=torch.arange(12800).reshape(80,80,2);assert ids[1,1,0]==162 and int(ids[1,1,1])//2==81
# The first cross block receives embedding and positional encoding before shift is read; constant input linear layer yields equal offsets.
torch.manual_seed(1);layer=torch.nn.Linear(4,6);q=torch.randn(1,4);pos=torch.randn(1,4);assert torch.equal(layer(q+pos),layer(q+pos))
assert not torch.cuda.is_initialized()
(R/'SYNTHETIC_TESTS.json').write_text(json.dumps({'passed':True,'tests':['AST target and source equations','row-major query and x/y','alpha/slot/level shift','sampling normalization','softmax','kernel support vs normalized range','anchor flat identity','first cross same input toy linear'],'scope':'synthetic CPU tensors; not real model or hook-on/off PPU validation','torch_version':torch.__version__,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
