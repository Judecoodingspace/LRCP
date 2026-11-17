"""Caller-only CPU ndarray -> tensor adapter. No protocol or source changes."""
import numpy as np
import torch

def cpu_tensor_inputs(*arrays):
 out=[]
 for a in arrays:
  if not isinstance(a,np.ndarray):raise TypeError('Expected saved NumPy array')
  t=torch.from_numpy(np.ascontiguousarray(a))
  if t.device.type!='cpu' or t.is_cuda or t.requires_grad:raise RuntimeError('Adapter must stay on CPU')
  if t.numpy().dtype!=a.dtype or t.shape!=a.shape or not np.array_equal(t.numpy(),a):raise RuntimeError('Adapter changed values/shape/dtype')
  out.append(t)
 return tuple(out)
