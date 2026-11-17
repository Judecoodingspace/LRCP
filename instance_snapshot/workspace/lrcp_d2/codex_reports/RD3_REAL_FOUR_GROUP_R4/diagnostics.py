"""Atomic recoverable UNACCEPTED evidence, always outside capture/acceptance."""
from rd3_common import *
from input_integrity import snapshot,differences,LABEL_PATH,validate_label_offset
import torch,uuid
MAX_ARRAY_BYTES=64*2**20

def save_unaccepted(root,attempt,model_returned,before,batch,error,arrays,extra=None):
 root=Path(root);target=root/'diagnostics'/('attempt_'+str(attempt['ordinal'])+'_'+uuid.uuid4().hex);tmp=target.with_name(target.name+'.tmp');tmp.mkdir(parents=True)
 try:delta=differences(before,batch,(LABEL_PATH,))
 except Exception as e:delta=[{'change':'contract_or_diff_error','error':str(e)}]
 arrays=dict(arrays)
 def collect_changed(v,p=''):
  if isinstance(v,dict):
   for k,w in v.items():collect_changed(w,p+'/'+str(k))
  elif isinstance(v,(list,tuple)):
   for k,w in enumerate(v):collect_changed(w,p+'/'+str(k))
  elif isinstance(v,torch.Tensor) and p in {d.get('path') for d in delta}:arrays['input_after:'+p]=v
 collect_changed(batch)
 selected={};skipped={};used=0
 for key,value in arrays.items():
  try:
   a=array(value)
   if a.dtype.hasobject or used+a.nbytes>MAX_ARRAY_BYTES:skipped[key]='unsafe dtype or bounded diagnostic size';continue
   selected[key]=a;used+=a.nbytes
  except Exception as e:skipped[key]=str(e)
 with (tmp/'arrays.npz').open('xb') as f:np.savez_compressed(f,**selected);f.flush();os.fsync(f.fileno())
 detail={'state':'UNACCEPTED','science_accepted':False,'attempt':attempt,'model_returned':bool(model_returned),'reason':str(error),'leaf_differences':getattr(error,'differences',delta),'global_input_differences':delta,'before':before,'after':snapshot(batch),'extra':extra or {},'array_inventory':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'value_hash':digest(v)} for k,v in selected.items()},'skipped_arrays':skipped}
 write(tmp/'diagnostic.json',detail);write(tmp/'manifest.json',{'state':'UNACCEPTED','files':{k:sha_file(tmp/k) for k in ('arrays.npz','diagnostic.json')}})
 os.rename(tmp,target);fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 return str(target)

def accepted_record(root,relative):
 path=(Path(root)/relative).resolve()
 require(path.parent==(Path(root)/'capture').resolve(),'DIAGNOSTIC_NOT_ACCEPTED_SCIENCE')
 return path
