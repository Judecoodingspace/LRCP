"""CPU hard guard installed before component construction. No device queries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
CALLS=[]
_INSTALLED=False
def forbidden(*args,**kwargs):
 CALLS.append('CUDA_ENTRY_BLOCKED');raise RuntimeError('CPU_ONLY_CUDA_FORBIDDEN')
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 if torch.cuda.is_initialized():raise RuntimeError('CUDA_ALREADY_INITIALIZED')
 torch.cuda.init=forbidden;torch.cuda._lazy_init=forbidden
 torch.Tensor.cuda=forbidden;torch.nn.Module.cuda=forbidden
 torch.set_num_threads(2)
 torch.set_num_interop_threads(1)
def evidence():return {'CUDA_initialized':torch.cuda.is_initialized(),'blocked_cuda_entry_calls':list(CALLS),'true_model_forward_increment':0}
