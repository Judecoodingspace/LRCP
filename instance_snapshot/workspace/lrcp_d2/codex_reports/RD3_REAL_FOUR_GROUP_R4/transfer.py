"""Lossless tensor transfer preserving OrderedDict/type/order and shared aliases."""
from rd3_common import *
from input_integrity import snapshot
import torch,copy

def transfer_tree(tree,device):
 memo={}
 def move(v):
  if id(v) in memo:return memo[id(v)]
  if isinstance(v,torch.Tensor):result=v.detach().to(device).clone()
  elif isinstance(v,dict):
   result=type(v)();memo[id(v)]=result
   for k,x in v.items():result[k]=move(x)
   return result
  elif isinstance(v,list):result=[];memo[id(v)]=result;result.extend(move(x) for x in v);return result
  elif isinstance(v,tuple):result=tuple(move(x) for x in v)
  else:result=copy.deepcopy(v)
  memo[id(v)]=result;return result
 return move(tree)
def verify_transfer(before,after,device):
 a=snapshot(before);b=snapshot(after);require(a['nodes']==b['nodes'],'TRANSFER_CONTAINER_CHANGE')
 require(set(a['leaves'])==set(b['leaves']),'TRANSFER_FIELDS')
 for path,v in a['leaves'].items():
  expected=dict(v)
  if v['kind']=='tensor':expected['device']=str(device)
  require(b['leaves'][path]==expected,'TRANSFER_LEAF_CHANGE '+path)
