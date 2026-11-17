"""Explicit CPU composite; never instantiates/forwards PointPillarDeformable.
Real frozen voxels -> unchanged native history orchestration/PillarVFE/Scatter/
ResNet/warp; frozen component weights. MetaFlow RESPONSE and detection HEADS
are marked interface mocks. Fusion uses original 128-dim/8-head/8-point native
configuration and frozen encoder weights, with the repository CPU debug sampler.
"""
from rd3_common import *
import torch,copy,inspect,sys,collections
from opencood.models.point_pillar_deformable import PointPillarDeformable
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.defor_bev_backbone_resnet import DeforResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified,BasicBlock
from opencood.models.sub_modules import defor_encoder_multi_scale as native
from opencood.utils.ms_deform_attn_ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from opencood.hypes_yaml import yaml_utils
from types import SimpleNamespace

class CPUKernel:
 @staticmethod
 def apply(value,shapes,starts,locations,weights,step):return ms_deform_attn_core_pytorch(value,shapes,locations,weights)

def config():return yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)))
def load_component(module,state,prefix):
 subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
 require(subset,'CHECKPOINT_COMPONENT_MISSING '+prefix);module.load_state_dict(subset,strict=True)
 require(digest(module.state_dict())==digest(subset),'COMPONENT_LOAD_MISMATCH '+prefix)
 return {'prefix':prefix,'state_hash':digest(subset),'entries':len(subset)}

class NativeBackbone(torch.nn.Module):
 get_normalized_transformation=DeforResNetBEVBackbone.get_normalized_transformation
 get_projected_bev_features=DeforResNetBEVBackbone.get_projected_bev_features
 def __init__(self,cfg,fusion=True):
  super().__init__();self.num_levels=3;self.discrete_ratio=cfg['voxel_size'][0];self.downsample_rate=1
  self.resnet=ResNetModified(BasicBlock,cfg['layer_nums'],cfg['layer_strides'],cfg['num_filters'],inplanes=64)
  self.input_proj=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(a,b,1),torch.nn.BatchNorm2d(b)) for a,b in zip(cfg['num_filters'],cfg['num_upsample_filter'])])
  if fusion:self.defor_encoder=native.DeforEncoderMultiScale(cfg['defor_encoder'])

class MetaFlowInterfaceMock(torch.nn.Module):
 """Contract response from ACTUAL native history features, not a LRCP estimate."""
 def forward(self,features,time):
  require([list(f.shape) for f in features]==[[1,5,64,80,80],[1,5,128,40,40],[1,5,256,20,20]],'NATIVE_HISTORY_FEATURE_SHAPES')
  require(time.shape==(1,) and time.item()==300,'METAFLOW_TIME_INTERFACE')
  x=features[0].mean((1,2));y=features[0][:,:,0].mean(1)
  return torch.stack((x,y),-1)*.01

class HistoryOwner(torch.nn.Module):
 get_batch_pillar_features=PointPillarDeformable.get_batch_pillar_features
 def __init__(self,state,fusion=True):
  super().__init__();h=config()['model']['args'];self.pillar_vfe=PillarVFE(h['pillar_vfe'],4,h['voxel_size'],h['lidar_range']);self.scatter=PointPillarScatter(h['point_pillar_scatter']);self.backbone=NativeBackbone(h['base_bev_backbone'],fusion);self.meta_flow=MetaFlowInterfaceMock();self.component_identity=[]
  for mod,prefix in [(self.pillar_vfe,'pillar_vfe.'),(self.backbone.resnet,'backbone.resnet.'),(self.backbone.input_proj,'backbone.input_proj.')]+([(self.backbone.defor_encoder,'backbone.defor_encoder.')] if fusion else []):self.component_identity.append(load_component(mod,state,prefix))
  self.eval();self.requires_grad_(False);self.fault=None

class NativeHistoryMock(HistoryOwner):
 def forward(self,e):
  offsets=[];predicted=[]
  for slot,v in e['calibrate_data'].items():
   offsets.append(v['offset'].flatten(1,2).unsqueeze(2))
   if v['valid_mask'].any():
    histories=v['lidar_history']
    if self.fault=='swap_histories' and int(slot) in (1,2):histories=e['calibrate_data'][3-int(slot)]['lidar_history']
    features=self.get_batch_pillar_features(histories,e['pairwise_t_matrix'],slot,[0])
    predicted.append(self.meta_flow(features,v['time_delay']))
   else:predicted.append(torch.zeros_like(v['offset']))
  off=torch.cat(offsets,2);pred=torch.stack([p.reshape(1,6400,2) for p in predicted],2)
  if self.fault=='swap_offsets':pred=pred[:,:,torch.tensor([1,0,2,3]),:]
  temp=dict(e['processed_lidar']);self.pillar_vfe(temp);self.scatter(temp)
  x=self.backbone.resnet(temp['spatial_features']);mlvl=[proj(value) for proj,value in zip(self.backbone.input_proj,x)]
  pair=self.backbone.get_normalized_transformation(160,160,e['pairwise_t_matrix'])
  q=self.backbone.defor_encoder(mlvl,e['record_len'],pair,off,pred)
  cls=torch.full((1,2,80,80),-12.);cls[:,:,40,40]=2.+q[:,0,40,40]*.01;cls[:,:,40,41]=1.5
  e['label_dict']['offset']=off
  if self.fault=='unexpected_label':e['label_dict']['surprise']=torch.tensor(1.)
  return {'cls_preds':cls,'reg_preds':torch.zeros(1,14,80,80),'dir_preds':torch.zeros(1,4,80,80),'pred_offset':pred}

class NativeCallTrace:
 def __init__(self):
  self.counts=collections.Counter();self.codes={inspect.unwrap(PointPillarDeformable.get_batch_pillar_features).__code__:'native_get_batch_pillar_features',PillarVFE.forward.__code__:'native_PillarVFE',PointPillarScatter.forward.__code__:'native_Scatter',DeforResNetBEVBackbone.get_projected_bev_features.__code__:'native_history_projection',ResNetModified.forward.__code__:'native_ResNet',MetaFlowInterfaceMock.forward.__code__:'MetaFlow_interface_mock',native.DeforEncoderMultiScale.forward.__code__:'native_fusion_encoder'}
 def profile(self,frame,event,arg):
  if event=='call' and frame.f_code in self.codes:self.counts[self.codes[frame.f_code]]+=1
 def __enter__(self):self.previous=sys.getprofile();require(self.previous is None,'EXISTING_PROFILER');sys.setprofile(self.profile);return self
 def __exit__(self,*exc):sys.setprofile(self.previous)

def checkpoint():
 require(sha_file(CKPT)==CH,'CHECKPOINT_SHA');return torch.load(CKPT,map_location='cpu',weights_only=False)
def build():
 from backend import Backend
 require(not torch.cuda.is_initialized(),'CPU_ONLY');native.MSDeformAttnFunction=CPUKernel
 state=checkpoint();model=NativeHistoryMock(state);del state
 return Backend(model,'cpu','mock')
