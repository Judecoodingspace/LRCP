"""Synthetic backend: no historic detection output and no real model."""
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
from pilot_common import digest

class Cross(torch.nn.Module):
 def forward(self,q,p,v,ref,shapes):return ref.clone()

class MockBackend:
 kind='mock'
 def __init__(self,fail_control=False):
  ready=json.loads(Path('/workspace/lrcp_d2/codex_reports/P5_A_CPU_FIX_20261028_storage/P5_READY.json').read_text())
  self.samples={int(x['sample_idx']):x for x in ready['samples']};self.fail=fail_control;self.forward_count=0
 def prepare(self,sid,kind):
  x=self.samples[sid];h=hashlib.sha256()
  with open(x['path'],'rb') as f:
   for data in iter(lambda:f.read(1048576),b''):h.update(data)
  if h.hexdigest()!=x['sha256']:raise RuntimeError('INPUT_HASH')
  payload=torch.load(x['path'],map_location='cpu',weights_only=False,mmap=False)
  if digest(payload['batch'])!=x['input_hash'] or digest(payload['rng'])!=x['RNG_hash']:raise RuntimeError('INPUT_SEMANTIC_HASH')
  del payload
  h,w=2,3;off=torch.ones(1,h*w,2,2)*0.2;base=torch.full((1,h*w,3,2),0.5)
  ref=torch.cat((base,base+off[:,:,0:1,:]/torch.tensor([w,h]),base+off[:,:,1:2,:]/torch.tensor([w,h])),2)
  return {'height':h,'width':w,'valid_remote':[True,False],'effective_offsets':off,'ref':ref,'modules':[Cross() for _ in range(4)]}
 def forward(self,p,sid,kind):
  self.forward_count+=1;q=torch.zeros(1,6,1);sh=torch.tensor([[2,3]]*9);before=p['ref'].clone();out=[m(q,q,q,p['ref'],sh) for m in p['modules']]
  assert torch.equal(before,p['ref']) and all(torch.equal(z,p['ref']) for z in out[1:])
  if kind=='core':assert torch.equal(out[0][:,:,3:6],p['ref'][:,:,:3]) and torch.equal(out[0][:,:,6:9],p['ref'][:,:,6:9])
  if self.fail and sid==918 and kind=='off':return {'v':1}
  return {'v':0}
 def collect(self,p,out,sid,kind,hooks):
  v=out['v'];arrays={k:np.array([v],dtype=np.float32) for k in ['cls_preds','reg_preds','dir_preds','final_corners','final_scores','native_nms_keep','gt_tp50','gt_tp70']}
  return arrays,{'mock':True,'full_schema':False}
 def baseline(self,sid):return {k:np.array([0],dtype=np.float32) for k in ['cls_preds','reg_preds','dir_preds','final_corners','final_scores','native_nms_keep','gt_tp50','gt_tp70']}
 def cleanup(self,p):
  assert all(not m._forward_pre_hooks for m in p['modules'])
