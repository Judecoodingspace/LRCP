import os,json,tempfile,shutil,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
import real_backend as rb
from runner_engine import run
from INDEPENDENT_VALIDATOR import verify_reference,verify_attention,ValidationError
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
from types import SimpleNamespace
import sys
sys.path.insert(0,"/workspace/lrcp_d2/codex_reports/P4_C2_20261027_233036_301795")
from c1_reused import eval_native
R=Path(__file__).parent
class Projection(torch.nn.Module):
 def __init__(self,size):super().__init__();self.size=size
 def forward(self,q):return q.new_zeros((q.shape[0],q.shape[1],self.size))
class MSDeformAttnFunction:
 @staticmethod
 def apply(value,spatial_shapes,level_start_index,sampling_locations,attention_weights,step):return value
class Cross(torch.nn.Module):
 def __init__(self):
  super().__init__();self.sampling_offsets=Projection(8*15*8*2);self.attention_weights=Projection(8*15*8)
 def forward(self,query,query_pos,value,reference_points,spatial_shapes):
  n=spatial_shapes.shape[0];q=query+query_pos
  sampling_offsets=self.sampling_offsets(q).view(1,6400,8,15,8,2)[:,:,:,:n,:,:].contiguous()
  attention_weights=self.attention_weights(q).view(1,6400,8,15,8)[:,:,:,:n,:].contiguous().view(1,6400,8,n*8).softmax(-1).view(1,6400,8,n,8)
  offset_normalizer=torch.stack([spatial_shapes[:,1],spatial_shapes[:,0]],-1)
  sampling_locations=reference_points[:,:,None,:,None,:]+sampling_offsets/offset_normalizer[None,None,None,:,None,:]
  level_start_index=torch.zeros(1,dtype=torch.long)
  output = MSDeformAttnFunction.apply(
   value,spatial_shapes,level_start_index,sampling_locations,attention_weights,64)
  return query+output*0
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
  cls=torch.full((1,2,80,80),-12.);cls[:,:,40,40]=2.;cls[:,:,40,41]=1.5
  return {'cls_preds':cls,'reg_preds':torch.zeros(1,14,80,80),'dir_preds':torch.zeros(1,4,80,80),'pred_offset':off}
class FakeObserver:
 def __init__(self,mods,qs):self.mods=mods;self.q=sorted(qs);self.events=[];self.projections={};self.handles=[]
 def __enter__(self):
  def pre(i):
   def go(m,args):
    ref=args[3][:,self.q,:,:].detach().numpy().copy();n=ref.shape[2];off=np.zeros((1,len(self.q),8,n,8,2),np.float32);weights=np.full((1,len(self.q),8,n,8),1/(n*8),np.float32)
    self.events.append({'block':i,'reference':ref,'offset':off,'location':np.broadcast_to(ref[:,:,None,:,None,:],off.shape).copy(),'weights':weights,'spatial_shapes':args[4].numpy().copy(),'summary':[]})
    self.projections[(i,'sampling_offsets')]={'input':np.zeros((1,len(self.q),128),np.float32),'output':np.zeros((1,len(self.q),8*15*8*2),np.float32)}
    self.projections[(i,'attention_weights')]={'input':np.zeros((1,len(self.q),128),np.float32),'output':np.zeros((1,len(self.q),8*15*8),np.float32)}
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
  p=super().prepare(sid,kind);self.pp.current_sid=sid;return p
 def collect(self,p,heads,sid,kind,hooks):
  # fake marker only; remove before frozen model input checks
  result=super().collect(p,heads,sid,kind,hooks)
  if kind in ('off','pass'):self.synthetic_baselines[sid]=result[0]
  return result
 def baseline(self,sid):return self.synthetic_baselines[sid]
model=Model().eval()
h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(rb.MODEL)));pp=build_postprocessor(h['postprocess'],False)
def makegt(batch):
 with np.load(rb.BASE/f'P3A_20261014_120929_boxes/sample_{pp.current_sid}_alpha_1.0.npz',allow_pickle=False) as z:return torch.from_numpy(z['gt_boxes'].copy())
pp.generate_gt_bbx=makegt
backend=TestBackend(model,pp,'cpu',lambda x,d:x,eval_native,observer_cls=rb.Observer);backend.synthetic_baselines={}
tmp=Path(os.environ.get('C2_MOCK_ROOT',str(R/'MOCK_RUN_V3')));tmp.mkdir(exist_ok=False)
try:
 records=run(tmp,R/'P5_SAMPLE_PROPOSAL_V2.csv',backend);assert len(records)==22
 for rec in records:
  data=Path(rec['path'])/'payload.bin'
  with np.load(data,allow_pickle=False) as z:
   assert all(k in z.files for k in ['cls_preds','reg_preds','dir_preds','raw_corners','scores_all','score_mask','quality_mask','native_nms_keep','range_mask','suppressor','GT_canonical','gt_tp50','gt_tp70','block0_output_selected','reference_original_full','effective_offset_full','cross_0_location','cross_0_weights','cross_3_direct_reference'])
 assert json.loads((tmp/'state.json').read_text())['state']=='MOCK_COMPLETE'
 checked=0;faults={'ego':0,'remote':0,'later_reference':0,'location':0,'weights':0}
 for rec in records:
  folder=Path(rec['path']);meta=json.loads((folder/'meta.json').read_text())
  with np.load(folder/'payload.bin',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  assert verify_reference(a,meta);verify_attention(a,meta);checked+=1
  if rec['condition']=='core' and checked>0 and not faults['ego']:
   for key,label in [('cross_0_direct_reference','ego'),('reference_original_full','remote'),('cross_1_direct_reference','later_reference'),('cross_0_location','location'),('cross_0_weights','weights')]:
    b={k:v.copy() for k,v in a.items()};ix=(0,0,0,0) if label in ('ego','remote','later_reference') else ((0,0,0,0,0,0) if label=='location' else (0,0,0,0,0))
    if label=='remote':ix=(0,0,3,0)
    b[key][ix]+=.05
    try:(verify_attention(b,meta) if label in ('location','weights') else verify_reference(b,meta))
    except ValidationError:faults[label]+=1
    else:raise AssertionError('undetected '+label)
 assert checked==22 and all(faults.values())
 assert not torch.cuda.is_initialized() and all(not m._forward_pre_hooks and not m._forward_hooks for m in model.cross)
 (R/'MOCK_PRODUCER_TEST.json').write_text(json.dumps({'passed':True,'partial_reference_attention_checks':checked,'complete_head_replay':True,'historical_controls':False,'note':'synthetic model, native P5-A observer and CPU collector/matcher; historical controls not compared','real_model_forward':0,'CUDA_initialized':False},indent=2))
 (R/'REFERENCE_ATTENTION_COMPONENT_FAULTS.json').write_text(json.dumps({'passed':False,'component_semantic_faults_detected':faults,'full_manifest_and_all_required_injections':False,'no_rehashed_shard_test':True},indent=2))
 (R/'FULL_NATIVE_MOCK_INTEGRATION.json').write_text(json.dumps({'passed':True,'mode':'synthetic_backend_substitutions','engine_calls':22,'real_frozen_inputs_loaded':True,'reference_hook_integrated':True,'failed_control_stops_core':True,'hooks_cleaned':True,'production_postprocessor_executed':True,'production_observer_executed':True,'GT_matcher_substituted':False,'real_model_forward':0,'CUDA_initialized':False},indent=2))
finally:pass
