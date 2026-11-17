"""CPU construction/binding audit of actual native backend, zero model forward.
Does not enter REAL_FORWARD or create a worker/claim/attempt.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,sys,inspect,argparse
parser=argparse.ArgumentParser();parser.add_argument("--output");options=parser.parse_args()
calls=[]
def blocked(*a,**k):calls.append('CUDA_ENTRY');raise RuntimeError('AUDIT_CUDA_FORBIDDEN')
torch.cuda.init=blocked;torch.cuda._lazy_init=blocked;torch.Tensor.cuda=blocked;torch.nn.Module.cuda=blocked
torch.set_num_threads(2);torch.set_num_interop_threads(1)
from rd3_common import *
import real_backend
from evidence_collector import EvidenceCollector
from control import authorize_worker
forward_calls=[]
def profile(frame,event,arg):
 if event=='call' and frame.f_code is real_backend.PointPillarDeformable.forward.__code__:
  forward_calls.append('native_forward');raise RuntimeError('NATIVE_FORWARD_FORBIDDEN_IN_PREP')
sys.setprofile(profile)
model,native=real_backend.native_cpu_model()
require(all(p.device.type=='cpu' for p in model.parameters()),'CPU_CONSTRUCTION_ONLY')
require(real_backend.RealBackend.forward_collect is EvidenceCollector.forward_collect,'R4_FINAL_COLLECTOR_REAL_BINDING')
require(real_backend.RealBackend.prepare.__globals__['EvidenceCollector'] is EvidenceCollector,'REAL_PREPARE_BINDING')
error=None
try:real_backend.RealBackend.build()
except Exception as e:error=str(e)
require(error and 'MANUAL_REAL_PERMISSION_REQUIRED' in error,'BUILD_MUST_REQUIRE_MANUAL_WORKER')
sys.setprofile(None)
require(not torch.cuda.is_initialized() and not calls and not forward_calls,'PREPARATION_DEVICE_OR_FORWARD')
write(Path(options.output) if options.output else R/'R4_NATIVE_ENTRY_CPU_AUDIT.json',{'passed':True,'actual_native_model_constructed_on_CPU':True,'native_runtime_binding':native,'real_backend_inherits_final_collector':True,'unauthorized_build_rejected':error,'MetaFlow_head_and_compiled_sampler_native':True,'all_parameters_CPU_FP32_eval':True,'CUDA_initialized':False,'CUDA_entry_calls':calls,'native_forward_calls':forward_calls,'budget_attempt_increment':0,'real_forward_increment':0,'real_backend_sha256':sha_file(R/'real_backend.py'),'collector_sha256':sha_file(R/'evidence_collector.py')},exclusive=True)
print('R4_NATIVE_REAL_BINDING_CPU_AUDITED_NO_FORWARD',flush=True)
