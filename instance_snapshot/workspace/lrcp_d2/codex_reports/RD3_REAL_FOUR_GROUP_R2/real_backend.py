"""New R2 REAL entry. No CPU-only runner or Mock implementation is imported."""
from rd3_common import *
from evidence_collector import EvidenceCollector,configuration
import torch,random,inspect
from opencood.models.point_pillar_deformable import PointPillarDeformable
from opencood.models.meta_flow import MetaFlow
from opencood.models.sub_modules import defor_encoder_multi_scale as fusion
from opencood.models.sub_modules import defor_encoder_multi_scale_single_agent as single
from opencood.utils.ms_deform_attn_ops.functions.ms_deform_attn_func import MSDeformAttnFunction
import MultiScaleDeformableAttention as native_extension
from opencood.tools import train_utils

def assert_native(model):
 require(type(model) is PointPillarDeformable and model.forward.__func__ is PointPillarDeformable.forward,'NATIVE_MODEL_FORWARD')
 require(type(model.meta_flow) is MetaFlow and model.meta_flow.forward.__func__ is MetaFlow.forward,'NATIVE_METAFLOW')
 require(all(type(getattr(model,k)) is torch.nn.Conv2d for k in ('cls_head','reg_head','dir_head')),'NATIVE_DETECTION_HEAD')
 require(fusion.MSDeformAttnFunction is MSDeformAttnFunction and single.MSDeformAttnFunction is MSDeformAttnFunction,'NATIVE_DEVICE_SAMPLER')
 require(inspect.getmodule(MSDeformAttnFunction).__name__=='opencood.utils.ms_deform_attn_ops.functions.ms_deform_attn_func' and str(native_extension.__file__).endswith('.so'),'NATIVE_EXTENSION_BINDING')
 require(not any('mock_backend' in k or k=='cpu_only' for k in sys.modules),'MOCK_MODULE_IN_REAL_ENTRY')
 require(not any(m.training for m in model.modules()) and all(p.dtype==torch.float32 for p in model.parameters()),'FP32_EVAL_PROTOCOL')
 ck=load(C2/'CHECKPOINT_IDENTITY.json');require(digest(dict(model.named_parameters()))==ck['parameter_digest'] and digest(dict(model.named_buffers()))==ck['buffer_digest'],'CHECKPOINT_PARAMETERS')
 return {'model_class':model.__class__.__module__+'.'+model.__class__.__name__,'MetaFlow_class':model.meta_flow.__class__.__module__+'.'+model.meta_flow.__class__.__name__,'head_classes':{k:type(getattr(model,k)).__name__ for k in ('cls_head','reg_head','dir_head')},'sampler':MSDeformAttnFunction.__module__+'.'+MSDeformAttnFunction.__name__,'extension_path':native_extension.__file__,'extension_sha256':sha_file(native_extension.__file__),'mock_modules_present':False,'parameters':ck['parameter_digest'],'buffers':ck['buffer_digest']}

def native_cpu_model():
 require(sha_file(CKPT)==CH,'CHECKPOINT_SHA')
 random.seed(303);np.random.seed(303);torch.manual_seed(303)
 model=train_utils.create_model(configuration());state=torch.load(CKPT,map_location='cpu',weights_only=False);model.load_state_dict(state,strict=True);del state;model.eval()
 return model,assert_native(model)
class RealBackend(EvidenceCollector):
 @classmethod
 def build(cls):
  model,identity=native_cpu_model();model=model.to('cuda:0');identity=assert_native(model)
  obj=cls(model,'cuda:0','REAL');obj.native_entry=identity;return obj
 def prepare(self,row):
  p=super().prepare(row);p['native_entry']=self.native_entry;return p
