"""CPU-only synthetic model using actual native multiscale encoder/DeforAttn.
Only the compiled device sampling op is replaced with its native CPU debug core.
No checkpoint model forward occurs. Real frozen batches/native postprocess used.
"""
from rd3_common import *
import torch
from backend import Backend
from opencood.models.sub_modules import defor_encoder_multi_scale as native
from opencood.utils.ms_deform_attn_ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
class CPUKernel:
 @staticmethod
 def apply(value,shapes,starts,locations,weights,step):return ms_deform_attn_core_pytorch(value,shapes,locations,weights)
class Meta(torch.nn.Module):
 def forward(self,hist,time_delay):
  return torch.full((1,80,80,2),float(hist),dtype=torch.float32)
class BB(torch.nn.Module):
 def __init__(self):
  super().__init__();cfg={'bev_h':80,'bev_w':80,'embed_dims':16,'max_num_agent':5,'feature_level':3,'calibrate':True,'train_stage':'stage2','block_cfgs':[[16,2,2,2,2,0.,5,3,['cross_attn','norm','ffn','norm']] for _ in range(4)]}
  self.defor_encoder=native.DeforEncoderMultiScale(cfg)
class MockModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.meta_flow=Meta();self.backbone=BB();self.calls=0
  self.features=[torch.randn(3,16,h,h)*.1+torch.arange(3)[:,None,None,None] for h in (80,40,20)]
 def forward(self,e):
  self.calls+=1;offset=[]
  for slot,v in e['calibrate_data'].items():offset.append(self.meta_flow(float(slot)*.25,v['time_delay']) if v['valid_mask'].any() else torch.zeros(1,80,80,2))
  off=torch.stack([x.reshape(1,6400,2) for x in offset],2)
  pair=torch.zeros((1,5,5,2,3));pair[:,:,:,0,0]=1;pair[:,:,:,1,1]=1
  q=self.backbone.defor_encoder(self.features,e['record_len'],pair,off,off)
  cls=torch.full((1,2,80,80),-12.);cls[:,:,40,40]=2.+q[:,0,40,40]*.01;cls[:,:,40,41]=1.5
  return {'cls_preds':cls,'reg_preds':torch.zeros(1,14,80,80),'dir_preds':torch.zeros(1,4,80,80),'pred_offset':off}
def build():
 require(not torch.cuda.is_initialized(),'MOCK_DEVICE');native.MSDeformAttnFunction=CPUKernel
 torch.manual_seed(901);m=MockModel().eval();return Backend(m,'cpu','mock')
