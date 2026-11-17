import os,json,tempfile,shutil,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
import real_backend as rb
from runner_engine import run
R=Path(__file__).parent
class Cross(torch.nn.Module):
 def __init__(self):super().__init__();self.sampling_offsets=torch.nn.Identity();self.attention_weights=torch.nn.Identity()
 def forward(self,q,pos,value,ref,shape):return q+ref[:,:,0,:1]*0
class Wrap:
 def __init__(self,m):self.cfgs=['cross_attn'];self.block_layers=[m]
class Meta(torch.nn.Module):
 def forward(self):return torch.ones(1,80,80,2,dtype=torch.float32)*.25
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.meta_flow=Meta();self.cross=torch.nn.ModuleList([Cross() for _ in range(4)])
  self.backbone=type('BB',(),{})();self.backbone.defor_encoder=type('Enc',(),{'blocks':[Wrap(m) for m in self.cross]})()
 def forward(self,b):
  entries=[]
  for v in b['calibrate_data'].values():entries.append(self.meta_flow() if bool(v['valid_mask'].any()) else torch.zeros(1,80,80,2))
  n=int(b['record_len'][0]);off=torch.stack([o.view(1,6400,2) for o in entries],2);base=torch.full((1,6400,3,2),.5)
  ref=torch.cat([base]+[base+off[:,:,i:i+1,:]/80 for i in range(n-1)],2);shape=torch.tensor([[80,80],[40,40],[20,20]]*n);q=torch.zeros(1,6400,128)
  for layer in self.cross:q=layer(q,q,q,ref,shape)
  return {'cls_preds':torch.zeros(1,2,80,80),'reg_preds':torch.zeros(1,14,80,80),'dir_preds':torch.zeros(1,4,80,80),'pred_offset':off}
class FakeObserver:
 def __init__(self,mods,qs):self.mods=mods;self.q=sorted(qs);self.events=[];self.projections={};self.handles=[]
 def __enter__(self):
  def pre(i):
   def go(m,args):
    ref=args[3][:,self.q,:,:].detach().numpy().copy();n=ref.shape[2];off=np.zeros((1,len(self.q),1,n,1,2),np.float32);weights=np.full((1,len(self.q),1,n,1),1/n,np.float32)
    self.events.append({'block':i,'reference':ref,'offset':off,'location':ref[:,:,None,:,None,:].copy(),'weights':weights,'spatial_shapes':args[4].numpy().copy(),'summary':[]})
    self.projections[(i,'sampling_offsets')]={'input':np.zeros((1,len(self.q),128),np.float32),'output':np.zeros((1,len(self.q),2),np.float32)}
    self.projections[(i,'attention_weights')]={'input':np.zeros((1,len(self.q),128),np.float32),'output':np.zeros((1,len(self.q),1),np.float32)}
   return go
  for i,m in enumerate(self.mods):self.handles.append(m.register_forward_pre_hook(pre(i)))
  return self
 def __exit__(self,*exc):
  for h in self.handles:h.remove()
 def validate(self):
  assert len(self.events)==4 and len(self.projections)==8
  return {'query_ids':self.q,'blocks':[{'block':i,'summary':[]} for i in range(4)]}
class PP:
 def generate_gt_bbx(self,b):
  sid=int(b['__fake_sid']);p=rb.BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_1.0.npz'
  with np.load(p,allow_pickle=False) as z:return torch.from_numpy(z['gt_boxes'].copy())
def fakepost(pp,b,heads):
 n=12800;score=np.zeros(n,np.float32);mask=np.zeros(n,bool);ids=np.empty(0,np.int32)
 z={'decoded_before_dir':np.zeros((n,7),np.float32),'scores_all':score,'score_mask':mask,'raw_corners':np.zeros((n,8,3),np.float32),'large_mask':np.zeros(0,bool),'z_mask':np.zeros(0,bool),'quality_mask':np.zeros(0,bool),'native_nms_keep':ids,'range_mask':np.zeros(0,bool),'nms_input_corners':np.zeros((0,8,3),np.float32),'nms_input_scores':np.zeros(0,np.float32),'final_corners':np.zeros((0,8,3),np.float32),'final_scores':np.zeros(0,np.float32),'suppressor':np.full(n,-1,np.int32),'suppressor_overlap':np.zeros(n,np.float32),'first_deletion_code':np.ones(n,np.int8),'ids_raw_decoded':np.arange(n,dtype=np.int32)}
 for k in ('score_pass','quality_pass','nms_top1000','nms_keep','final_range'):z['ids_'+k]=ids.copy()
 for k in ('cls_preds','reg_preds','dir_preds'):z[k]=heads[k].detach().numpy().copy()
 return z,{'mock':True}
def fakeeval(pred,scores,gt,t):return {},np.empty(0,np.int32),np.zeros(len(gt),np.int8)
class TestBackend(rb.LRCPBackend):
 kind='mock'
 def prepare(self,sid,kind):
  p=super().prepare(sid,kind);p['batch']['__fake_sid']=sid;return p
 def collect(self,p,heads,sid,kind,hooks):
  # fake marker only; remove before frozen model input checks
  result=super().collect(p,heads,sid,kind,hooks)
  if kind in ('off','pass'):self.synthetic_baselines[sid]=result[0]
  return result
 def baseline(self,sid):return self.synthetic_baselines[sid]
model=Model().eval();backend=TestBackend(model,PP(),'cpu',lambda x,d:x,fakeeval,observer_cls=FakeObserver,postprocess=fakepost,iou=lambda gt,boxes:np.zeros((len(gt),len(boxes)),np.float32));backend.synthetic_baselines={}
tmp=Path(tempfile.mkdtemp(prefix='b0_real_backend_substitute_'))
try:
 records=run(tmp,R/'P5_SAMPLE_PROPOSAL_V2.csv',backend);assert len(records)==22
 for rec in records:
  data=Path(rec['path'])/'payload.bin'
  with np.load(data,allow_pickle=False) as z:
   assert all(k in z.files for k in ['cls_preds','reg_preds','dir_preds','raw_corners','scores_all','score_mask','quality_mask','native_nms_keep','range_mask','suppressor','GT_canonical','gt_tp50','gt_tp70','block0_output_selected','reference_original_full','effective_offset_full','cross_0_location','cross_0_weights','cross_3_direct_reference'])
 assert json.loads((tmp/'state.json').read_text())['state']=='MOCK_COMPLETE'
 failroot=tmp/'fail_case'
 class FailedControl(TestBackend):
  def baseline(self,sid):
   a=dict(self.synthetic_baselines[sid]);a['final_scores']=np.ones((1,),np.float32);return a
 badmodel=Model().eval();bad=FailedControl(badmodel,PP(),'cpu',lambda x,d:x,fakeeval,observer_cls=FakeObserver,postprocess=fakepost,iou=lambda gt,boxes:np.zeros((len(gt),len(boxes)),np.float32));bad.synthetic_baselines={}
 try:run(failroot,R/'P5_SAMPLE_PROPOSAL_V2.csv',bad)
 except RuntimeError as e:assert str(e).startswith('CONTROL_MISMATCH')
 else:raise AssertionError('control failure did not stop')
 assert len(json.loads((failroot/'budget.json').read_text())['attempts'])==1
 assert not (failroot/'control_gate.json').exists()
 assert all(not m._forward_hooks and not m._forward_pre_hooks for m in badmodel.cross) and not badmodel.meta_flow._forward_hooks
 assert not torch.cuda.is_initialized() and all(not m._forward_pre_hooks and not m._forward_hooks for m in model.cross)
 (R/'BACKEND_INTEGRATION_TEST.json').write_text(json.dumps({'passed':True,'mode':'synthetic_backend_substitutions','engine_calls':22,'real_frozen_inputs_loaded':True,'reference_hook_integrated':True,'failed_control_stops_core':True,'hooks_cleaned':True,'production_postprocessor_executed':False,'production_observer_executed':False,'GT_matcher_substituted':True,'real_model_forward':0,'CUDA_initialized':False},indent=2))
finally:shutil.rmtree(tmp)
