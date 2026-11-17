import os,json,sys,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from pathlib import Path
from intervention import ReferenceIntervention
class Block(torch.nn.Module):
 def forward(self,q,pos,value,ref,shapes):
  return ref.clone()
mods=[Block() for _ in range(4)];H,W,L=2,3,3;Q=H*W;valid=[True,False,True]
base=torch.arange(Q*2,dtype=torch.float32).view(1,Q,1,2)/20
base=base.expand(1,Q,L,2);offset=torch.arange(Q*3*2,dtype=torch.float32).view(1,Q,3,2)/15
ref=torch.cat([base]+[base+offset[:,:,j:j+1,:]/torch.tensor([W,H]) for j in range(3)],dim=2);snapshot=ref.clone();shape=torch.tensor([[H,W]]*12);q=torch.zeros(1,Q,1)
def call(m,r):return m(q,q,q,r,shape)
with ReferenceIntervention(mods,valid,offset,height=H,width=W) as c:
 result=[call(m,ref) for m in mods];assert c.validate()
assert torch.equal(ref,snapshot) and ref.data_ptr()!=result[0].data_ptr()
assert torch.equal(result[0][:,:,:3],ref[:,:,:3]) and torch.equal(result[0][:,:,6:9],ref[:,:,6:9])
assert torch.equal(result[0][:,:,3:6],base) and torch.equal(result[0][:,:,9:12],base)
assert all(torch.equal(x,ref) for x in result[1:])
with ReferenceIntervention(mods,valid,offset,mode='passthrough',height=H,width=W) as c:
 assert all(torch.equal(call(m,ref),ref) for m in mods);assert c.validate()
zero=torch.zeros_like(offset);zero_ref=torch.cat([base]*4,dim=2)
with ReferenceIntervention(mods,valid,zero,height=H,width=W) as c:
 assert all(torch.equal(call(m,zero_ref),zero_ref) for m in mods);assert c.validate()
try:
 with ReferenceIntervention(mods,valid,offset,height=H,width=W):call(mods[1],ref)
except RuntimeError as e:assert str(e)=='CROSS_CALL_ORDER_OR_DUPLICATE'
else:raise AssertionError('out of order accepted')
try:
 with ReferenceIntervention(mods,valid,offset,height=H,width=W):
  call(mods[0],ref);call(mods[0],ref)
except RuntimeError as e:assert str(e)=='CROSS_CALL_ORDER_OR_DUPLICATE'
else:raise AssertionError('duplicate accepted')
try:
 with ReferenceIntervention(mods,valid,offset+1,height=H,width=W):call(mods[0],ref)
except RuntimeError as e:assert str(e)=='REMOTE_SHIFT_MISMATCH'
else:raise AssertionError('bad shift accepted')
assert all(not m._forward_pre_hooks for m in mods) and not torch.cuda.is_initialized()
Path(__file__).with_name('P5_B0_SYNTHETIC_VALIDATION.json').write_text(json.dumps({'passed':True,'CPU_synthetic_only':True,'CUDA_initialized':False,'real_model_forward':0,'checks':['all_queries_valid_slots','ego_and_invalid_unchanged','xy_normalization','passthrough','reference_formula','no_inplace_shared_mutation','later_blocks_original_reference','zero_shift_noop','order_and_duplicate_rejected','shift_mismatch_rejected','exception_cleanup']},indent=2))
