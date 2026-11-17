"""Restricted read-only torch-save ZIP storage reader implemented with NumPy.
Does NOT import torch or any model/pipeline. No arbitrary pickle globals.
Device locations are archival strings, never allocation requests.
"""
import io,pickle,zipfile,collections,codecs,hashlib
import numpy as np

class TensorArray(np.ndarray):
 def __array_finalize__(self,obj):
  self.archived_device=getattr(obj,'archived_device',None)
  self.torch_dtype=getattr(obj,'torch_dtype',None)

class StorageType:
 def __init__(self,dtype,torch_dtype):self.dtype=np.dtype(dtype);self.torch_dtype=torch_dtype

class Storage:
 def __init__(self,data,kind,location):self.data=data;self.kind=kind;self.location=location

TYPES={name:StorageType(dtype,td) for name,dtype,td in [('FloatStorage','<f4','torch.float32'),('DoubleStorage','<f8','torch.float64'),('LongStorage','<i8','torch.int64'),('IntStorage','<i4','torch.int32'),('ShortStorage','<i2','torch.int16'),('CharStorage','i1','torch.int8'),('ByteStorage','u1','torch.uint8'),('BoolStorage','?','torch.bool')]}

def rebuild(storage,offset,size,stride,*ignored):
 shape=tuple(map(int,size));strides=tuple(int(x)*storage.kind.dtype.itemsize for x in stride)
 v=np.ndarray(shape,dtype=storage.kind.dtype,buffer=storage.data,offset=int(offset)*storage.kind.dtype.itemsize,strides=strides).view(TensorArray)
 v.archived_device=storage.location;v.torch_dtype=storage.kind.torch_dtype;v.flags.writeable=False
 return v

class RestrictedUnpickler(pickle.Unpickler):
 def __init__(self,stream,z,prefix):super().__init__(stream);self.z=z;self.prefix=prefix;self.storages={};self.bytes_read=0
 def find_class(self,module,name):
  if module=='torch' and name in TYPES:return TYPES[name]
  if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):return rebuild
  allowed={('collections','OrderedDict'):collections.OrderedDict,('_codecs','encode'):codecs.encode,('numpy','dtype'):np.dtype,('numpy','ndarray'):np.ndarray,('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,('numpy._core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,('numpy.core.multiarray','scalar'):np.core.multiarray.scalar}
  if (module,name) not in allowed:raise pickle.UnpicklingError('UNAPPROVED_GLOBAL '+module+'.'+name)
  return allowed[module,name]
 def persistent_load(self,pid):
  if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage' or not isinstance(pid[1],StorageType):raise pickle.UnpicklingError('INVALID_STORAGE_PID')
  _,kind,key,location,count=pid
  if location not in ('cpu','cuda:0'):raise pickle.UnpicklingError('UNEXPECTED_ARCHIVED_DEVICE '+str(location))
  if key not in self.storages:
   member=self.prefix+'data/'+str(key);info=self.z.getinfo(member)
   if info.file_size!=int(count)*kind.dtype.itemsize:raise pickle.UnpicklingError('STORAGE_SIZE_MISMATCH')
   self.bytes_read+=info.file_size
   if self.bytes_read>512*2**20:raise pickle.UnpicklingError('OFFLINE_STORAGE_SIZE_BOUND')
   self.storages[key]=Storage(self.z.read(member),kind,location)
  storage=self.storages[key]
  if storage.location!=location or storage.kind is not kind:raise pickle.UnpicklingError('STORAGE_ALIAS_TYPE_OR_DEVICE_CHANGED')
  return storage

def load_storage_archive(path):
 with zipfile.ZipFile(path) as z:
  names=[x for x in z.namelist() if x.endswith('/data.pkl')]
  if len(names)!=1:raise ValueError('ONE_DATA_PICKLE_REQUIRED')
  name=names[0];prefix=name[:-len('data.pkl')]
  if prefix+'byteorder' in z.namelist() and z.read(prefix+'byteorder')!=b'little':raise ValueError('UNSUPPORTED_BYTEORDER')
  if z.getinfo(name).file_size>8*2**20:raise ValueError('PICKLE_SIZE_BOUND')
  reader=RestrictedUnpickler(io.BytesIO(z.read(name)),z,prefix);value=reader.load()
  return value,{'storage_count':len(reader.storages),'storage_bytes':reader.bytes_read,'archived_devices':sorted({v.location for v in reader.storages.values()}),'torch_imported':False}

def digest(value):
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(np.ascontiguousarray(v).tobytes())
  elif isinstance(v,dict):
   h.update(b'dict')
   for k,w in v.items():h.update(repr(k).encode());visit(w)
  elif isinstance(v,(list,tuple)):
   h.update(type(v).__name__.encode())
   for w in v:visit(w)
  else:h.update(repr(v).encode())
 visit(value);return h.hexdigest()

def snapshot(value):
 nodes={};leaves={}
 def visit(v,p):
  if isinstance(v,dict):
   nodes[p]={'kind':type(v).__name__,'keys':[[type(k).__name__,repr(k)] for k in v]}
   for k,w in v.items():visit(w,p+'/'+str(k))
  elif isinstance(v,(list,tuple)):
   nodes[p]={'kind':type(v).__name__,'length':len(v)}
   for i,w in enumerate(v):visit(w,p+'/'+str(i))
  elif isinstance(v,TensorArray):leaves[p]={'kind':'tensor','shape':list(v.shape),'dtype':v.torch_dtype,'device':v.archived_device,'layout':'torch.strided','value_hash':digest(v)}
  elif isinstance(v,np.ndarray):leaves[p]={'kind':'ndarray','shape':list(v.shape),'dtype':str(v.dtype),'value_hash':digest(v)}
  else:leaves[p]={'kind':type(v).__name__,'value_hash':digest(v),'repr':repr(v)}
 visit(value,'');body={'nodes':nodes,'leaves':leaves};return {**body,'hash':digest(body)}
