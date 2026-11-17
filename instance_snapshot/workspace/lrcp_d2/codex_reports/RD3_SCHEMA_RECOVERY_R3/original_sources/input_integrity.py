"""Strict original-leaf contract and scoped disposable native histories.
Original tensors are intentionally shared into shallow dict copies. Any in-place
write, original-leaf replacement, metadata change, reordering or extra key fails.
"""
from rd3_common import *
import torch,copy,types,contextlib
ORIGINAL_HISTORY=('voxel_features','voxel_coords','voxel_num_points')
DERIVED_HISTORY=('pillar_features','spatial_features','pairwise_t_matrix')
LABEL_PATH='/ego/label_dict/offset'

class IntegrityError(RuntimeError):
 def __init__(self,differences):
  self.differences=differences
  super().__init__('OTHER_INPUT_MUTATION '+json.dumps(differences,ensure_ascii=False)[:3000])

def describe(x):
 if isinstance(x,torch.Tensor):
  d={'kind':'tensor','shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'layout':str(x.layout)}
  if x.device.type!='meta':d['value_hash']=digest(x)
  return d
 if isinstance(x,np.ndarray):return {'kind':'ndarray','shape':list(x.shape),'dtype':str(x.dtype),'value_hash':digest(x)}
 return {'kind':type(x).__name__,'value_hash':digest(x),'repr':repr(x)}

def snapshot(x,path=''):
 nodes={};leaves={}
 def visit(v,p):
  if isinstance(v,dict):
   keys=[[type(k).__name__,repr(k)] for k in v];nodes[p]={'kind':type(v).__name__,'keys':keys}
   for k,w in v.items():visit(w,p+'/'+str(k))
  elif isinstance(v,(list,tuple)):
   nodes[p]={'kind':type(v).__name__,'length':len(v)}
   for i,w in enumerate(v):visit(w,p+'/'+str(i))
  else:leaves[p]=describe(v)
 visit(x,path)
 body={'nodes':nodes,'leaves':leaves};return {**body,'hash':digest(body)}

def differences(before,x,allowed_additions=()):
 require(before['hash']==digest({k:before[k] for k in ('nodes','leaves')}),'INPUT_CONTRACT_HASH_CORRUPTED')
 after=snapshot(x);diff=[];allowed=set(allowed_additions)
 for p,b in before['leaves'].items():
  a=after['leaves'].get(p)
  if a!=b:diff.append({'path':p,'change':'missing_or_modified_leaf','before':b,'after':a})
 for p,a in after['leaves'].items():
  if p not in before['leaves'] and p not in allowed:diff.append({'path':p,'change':'unexpected_leaf','after':a})
 for p,b in before['nodes'].items():
  a=after['nodes'].get(p)
  if b==a:continue
  if a and b['kind']==a['kind'] and 'keys' in b and 'keys' in a:
   expected=b['keys'];actual=a['keys'];extras=[k for k in actual if k not in expected]
   okay=[k for k in actual if k in expected]==expected and all(p+'/'+str(_decode_key(k)) in allowed for k in extras)
   if okay:continue
  diff.append({'path':p,'change':'container_keys_order_or_type','before':b,'after':a})
 for p,a in after['nodes'].items():
  if p not in before['nodes']:diff.append({'path':p,'change':'unexpected_container','after':a})
 return diff

def _decode_key(k):
 import ast
 return ast.literal_eval(k[1])

def check(before,x,allowed_additions=()):
 diff=differences(before,x,allowed_additions)
 if diff:raise IntegrityError(diff)
 return snapshot(x)

def validate_label_offset(batch):
 e=batch['ego'];label=e['label_dict']
 require('offset' in label,'KNOWN_LABEL_OFFSET_MISSING')
 expected=torch.cat([v['offset'].flatten(1,2).unsqueeze(2) for v in e['calibrate_data'].values()],2)
 value=label['offset'];require(isinstance(value,torch.Tensor) and value.shape==expected.shape and value.dtype==expected.dtype and value.device==expected.device and torch.equal(value,expected),'LABEL_OFFSET_SEMANTICS')
 return describe(value)

class DisposableHistory:
 def __init__(self,model,batch,*,fault=None):
  self.model=model;self.batch=batch;self.fault=fault;self.receipts=[];self.outputs={};self.temps=[];self.expected_meta=[];self.meta_events=[];self.handles=[]
 def __enter__(self):
  m=self.model;self.had_instance='get_batch_pillar_features' in m.__dict__;self.instance=m.__dict__.get('get_batch_pillar_features');self.original=m.get_batch_pillar_features
  def isolated(this,histories,pairwise,agent_index,batch_indices=None):
   slot=int(agent_index);indices=list(range(len(histories))) if batch_indices is None else list(batch_indices)
   require(slot in (1,2),'HISTORY_SLOT_INVALID')
   e=self.batch['ego'];source=[e['calibrate_data'][slot]['lidar_history'][i] for i in indices]
   require(len(histories)==len(source) and all(a is b for a,b in zip(histories,source)),'HISTORY_AGENT_INPUT_MISMATCH')
   before=[snapshot(h) for h in source];pair_before=snapshot(pairwise);temp=[dict(h) for h in histories];self.temps.extend(temp)
   require(all(tuple(h)==ORIGINAL_HISTORY for h in source),'ORIGINAL_HISTORY_FIELDS')
   for h,t in zip(source,temp):
    require(h is not t and all(t[k] is h[k] and t[k].untyped_storage().data_ptr()==h[k].untyped_storage().data_ptr() for k in ORIGINAL_HISTORY),'SHALLOW_STORAGE_CONTRACT')
   try:out=self.original(temp,pairwise,agent_index,indices)
   except BaseException:
    for h,t,b in zip(source,temp,before):
     check(b,h);check(b,t,tuple('/'+k for k in DERIVED_HISTORY))
    check(pair_before,pairwise)
    raise
   if self.fault=='unknown_derived' and slot==1:temp[0]['unapproved_cache']=torch.tensor(1)
   for h,t,b in zip(source,temp,before):
    check(b,h);extra=[k for k in t if k not in ORIGINAL_HISTORY];require(extra==list(DERIVED_HISTORY),'UNEXPECTED_DERIVED_FIELDS')
    check(b,t,tuple('/'+k for k in DERIVED_HISTORY))
   check(pair_before,pairwise)
   output_hash=digest(out);self.outputs[slot]=out
   rec={'slot':slot,'physical_id':e['calibrate_data'][slot]['physical_cav_ids'][0],'batch_indices':indices,'original_before':before,'original_after':[snapshot(h) for h in source],'temporary_after':[snapshot(t) for t in temp],'pairwise_hash':digest(pairwise),'feature_hash':output_hash,'feature_shapes':[list(x.shape) for x in out],'shared_tensor_storage':True}
   self.receipts.append(rec);self.expected_meta.append((slot,output_hash,describe(e['calibrate_data'][slot]['time_delay'].index_select(0,e['calibrate_data'][slot]['time_delay'].new_tensor(indices,dtype=torch.long))),out))
   return out
  m.get_batch_pillar_features=types.MethodType(isolated,m)
  def meta_pre(mod,args):
   require(self.expected_meta,'METAFLOW_WITHOUT_HISTORY')
   slot,expected,time,origin=self.expected_meta.pop(0)
   require(args[0] is origin and digest(args[0])==expected and describe(args[1])==time,'METAFLOW_AGENT_INPUT_MAPPING')
   self.current_slot=slot
  def meta_post(mod,args,out):
   self.meta_events.append({'slot':self.current_slot,'physical_id':self.batch['ego']['calibrate_data'][self.current_slot]['physical_cav_ids'][0],'input_feature_hash':digest(args[0]),'time':describe(args[1]),'output_hash':digest(out),'output_shape':list(out.shape)})
  self.handles=[m.meta_flow.register_forward_pre_hook(meta_pre),m.meta_flow.register_forward_hook(meta_post)]
  return self
 def __exit__(self,*exc):
  for h in self.handles:h.remove()
  if self.had_instance:self.model.get_batch_pillar_features=self.instance
  else:del self.model.get_batch_pillar_features
 def validate(self):
  require([x['slot'] for x in self.receipts]==[1,2] and [x['slot'] for x in self.meta_events]==[1,2] and not self.expected_meta,'HISTORY_METAFLOW_CALL_ORDER')
  return {'receipts':self.receipts,'metaflow_events':self.meta_events}
