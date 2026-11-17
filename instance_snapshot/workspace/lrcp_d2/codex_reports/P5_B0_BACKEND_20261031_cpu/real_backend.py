"""LRCP P5-B0 backend for runner_engine. Import is CPU-safe; no model is built.

The class requires explicit runtime construction. PPU_READY is NOT implied.
"""
import os,sys,csv,json,copy,random,resource,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pilot_common import MODEL,BASE,array,digest,sha_file,gt_mapping,iou_matrix,STAGES
from contracts import attribute_gt
from attention_observer import Observer
from minimal_collector import collect_postprocess

ROOT=Path(__file__).resolve().parent
P5A=ROOT.parent/'P5_A_CPU_FIX_20261028_storage'
FULL=ROOT.parent/'P4_FULL_PREP_20261028_113714_782940'
CHECKPOINT=MODEL/'net_epoch_bestval_at19.pth'
CHECKPOINT_SHA='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'

class LRCPBackend:
 kind='real'
 def __init__(self,model,postprocessor,device,to_device,evaluate,*,observer_cls=Observer,postprocess=collect_postprocess,iou=iou_matrix,attribution=attribute_gt):
  self.model=model;self.pp=postprocessor;self.device=device;self.to_device=to_device;self.evaluate=evaluate
  self.observer_cls=observer_cls;self.postprocess=postprocess;self.iou=iou;self.attribution=attribution
  self.inputs={int(x['sample_idx']):x for x in json.loads((FULL/'INPUT_MANIFEST.json').read_text())['samples']}
  selection=list(csv.DictReader((ROOT/'P5_SAMPLE_PROPOSAL_V2.csv').open()));self.selection={int(x['sample_idx']):x for x in selection};self.queries={s:set() for s in self.selection}
  for name in ('P5_FOCAL_ANCHORS.csv','P5_FOCAL_ANCHORS_ALPHA0.csv'):
   for row in csv.DictReader((ROOT/name).open()):
    if row['anchor_query_index']:self.queries[int(row['sample_idx'])].add(int(row['anchor_query_index']))
    if row['suppressor_query_index']:self.queries[int(row['sample_idx'])].add(int(row['suppressor_query_index']))
  for q in self.queries.values():
   if not q:q.add(3240)
   if len(q)>256:raise RuntimeError('FOCAL_QUERY_LIMIT')
  man=json.loads((P5A/'CAPTURE_MANIFEST.json').read_text())
  self.baselines={(int(x['sample_idx']),int(x['alpha'])):x for x in man['records']}
  self.modules=[]
  for block in model.backbone.defor_encoder.blocks:
   for typ,layer in zip(block.cfgs,block.block_layers):
    if typ=='cross_attn':self.modules.append(layer)
  if len(self.modules)!=4:raise RuntimeError('CROSS_MODULE_COUNT')
  self.buffer_digest=digest({k:v for k,v in model.named_buffers()})
  self.parameter_digest=digest({k:v for k,v in model.named_parameters()})

 @classmethod
 def build_real(cls,device):
  """Future user-authorized worker entry; never called by CPU preparation."""
  if sha_file(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('CHECKPOINT_HASH')
  ready=json.loads((P5A/'P5_READY.json').read_text())
  for path,hash_ in ready['locked_files'].items():
   if sha_file(path)!=hash_:raise RuntimeError('LOCKED_SOURCE_CHANGED '+path)
  from opencood.hypes_yaml import yaml_utils
  from opencood.tools import train_utils
  from opencood.data_utils.datasets import build_dataset
  sys.path.insert(0,str(ROOT.parent/'P4_C2_20261027_233036_301795'))
  from c1_reused import eval_native
  random.seed(303);np.random.seed(303);torch.manual_seed(303)
  h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));h['train_params']['batch_size']=1;h['validate_dir']=h['test_dir']
  if h['time_delay']!=3 or h['history_frame']!=5:raise RuntimeError('TIME_PROTOCOL')
  model=train_utils.create_model(h);epoch,model=train_utils.load_saved_model(str(MODEL),model);model.eval()
  if epoch!=19 or not all(p.dtype==torch.float32 for p in model.parameters()):raise RuntimeError('MODEL_PROTOCOL')
  pp=build_dataset(h,visualize=True,train=False).post_processor
  if pp.params['target_args']['score_threshold']!=.2 or pp.params['nms_thresh']!=.15:raise RuntimeError('POSTPROCESS_PROTOCOL')
  model=model.to(device)
  return cls(model,pp,device,train_utils.to_device,eval_native)

 def prepare(self,sid,kind):
  if sid not in self.selection:raise RuntimeError('UNSELECTED_SAMPLE')
  if sha_file(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('CHECKPOINT_CHANGED')
  item=self.inputs[sid]
  if sha_file(item['path'])!=item['sha256']:raise RuntimeError('INPUT_FILE_CHANGED')
  payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False);frozen=payload['batch']
  if digest(frozen)!=item['input_hash'] or digest(payload['rng'])!=item['RNG_hash']:raise RuntimeError('INPUT_SEMANTICS_CHANGED')
  batch=self.to_device(copy.deepcopy(frozen),self.device)
  if digest(batch)!=item['input_hash']:raise RuntimeError('INPUT_TRANSFER_CHANGED')
  cal=batch['ego']['calibrate_data'];keys=[str(k) for k in cal.keys()];n=int(batch['ego']['record_len'][0].item())
  if n<1 or n-1>len(keys):raise RuntimeError('ACTIVE_SLOT_COUNT')
  if keys!=[str(k) for k in item['slots'].keys()]:raise RuntimeError('SLOT_ORDER_MISMATCH')
  active=keys[:n-1];valid_all=[str(k) for k,v in cal.items() if bool(v['valid_mask'].any().item())]
  for k,v in cal.items():
   expected=item['slots'].get(str(k))
   if expected is None or list(map(bool,array(v['valid_mask']).reshape(-1)))!=list(map(bool,expected['valid_mask'])):raise RuntimeError('SLOT_VALIDITY_MISMATCH')
   if any(expected['valid_mask']) and (not expected['physical_cav_ids'] or any(t!=300 for t in expected['time_delay'])):raise RuntimeError('SLOT_PROVENANCE_MISMATCH')
  mask=[k in valid_all for k in active];buf=torch.zeros((1,6400,len(active),2),dtype=torch.float32,device=self.device)
  for j,valid in enumerate(mask):
   if valid:buf[:,:,j,:]=float('nan')
  return {'sample_idx':sid,'item':item,'payload':payload,'batch':batch,'frozen':frozen,'valid_all':valid_all,'active':active,'valid_remote':mask,'effective_offsets':buf,'modules':self.modules,'height':80,'width':80,'handles':[],'references':[],'meta_offsets':[],'block0_output':[]}

 def forward(self,p,sid,kind):
  rng=p['payload']['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.random.set_rng_state(rng['torch_cpu'])
  def meta_hook(module,args,out):
   i=len(p['meta_offsets'])
   if i>=len(p['valid_all']) or tuple(out.shape)!=(1,80,80,2):raise RuntimeError('METAFLOW_CALL_OR_SHAPE')
   slot=p['valid_all'][i];p['meta_offsets'].append(array(out))
   if slot in p['active']:p['effective_offsets'][:,:,p['active'].index(slot),:].copy_(out.reshape(1,6400,2))
   return out
  def capture_ref(module,args):
   ref=args[3];idx=len(p['references']);h=p.get('intervention')
   shapes=args[4];expected_shapes=torch.tensor([[80,80],[40,40],[20,20]]*(len(p['active'])+1),device=shapes.device,dtype=shapes.dtype)
   if not torch.equal(shapes,expected_shapes):raise RuntimeError('LEVEL_ORDER_MISMATCH')
   if idx==0:
    original=h.original if h is not None else ref
    if tuple(original.shape)!=(1,6400,3*(1+len(p['active'])),2):raise RuntimeError('REFERENCE_SHAPE')
    base=original[:,:,:1,:]
    if not torch.equal(original[:,:,:3,:],base.expand(-1,-1,3,-1)):raise RuntimeError('EGO_REFERENCE_INCONSISTENT')
    normal=original.new_tensor([80,80]);expected=base.unsqueeze(2)+p['effective_offsets'].unsqueeze(3)/normal
    for j,valid in enumerate(p['valid_remote']):
     if valid and not torch.allclose(original[:,:,3*(j+1):3*(j+2),:],expected[:,:,j,:,:].expand(-1,-1,3,-1),rtol=0,atol=1e-6):raise RuntimeError('REMOTE_SHIFT_MISMATCH')
   p['references'].append(array(ref))
  def capture_output(module,args,out):
   idx=torch.as_tensor(sorted(self.queries[sid]),device=out.device);p['block0_output'].append(array(out.index_select(1,idx)))
  p['handles'].append(self.model.meta_flow.register_forward_hook(meta_hook))
  for m in self.modules:p['handles'].append(m.register_forward_pre_hook(capture_ref))
  p['handles'].append(self.modules[0].register_forward_hook(capture_output))
  observer=self.observer_cls(self.modules,self.queries[sid]);p['observer']=observer
  with torch.no_grad():
   with observer:heads=self.model(p['batch']['ego'])
  if self.kind=='real' and any(not heads[k].is_cuda or heads[k].dtype!=torch.float32 for k in ('cls_preds','reg_preds','dir_preds')):raise RuntimeError('DEVICE_OR_DTYPE_MISMATCH')
  if len(p['meta_offsets'])!=len(p['valid_all']) or len(p['references'])!=4 or len(p['block0_output'])!=1:raise RuntimeError('CAPTURE_CALL_COUNT')
  if not bool(torch.isfinite(p['effective_offsets']).all()):raise RuntimeError('MISSING_METAFLOW_SLOT')
  h=p.get('intervention');original=array(h.original) if h else p['references'][0]
  if any(not np.array_equal(x,original) for x in p['references'][1:]):raise RuntimeError('LATER_REFERENCE_CHANGED')
  actual=p['references'][0];expected=original.copy()
  if kind=='core':
   for j,valid in enumerate(p['valid_remote']):
    if valid:expected[:,:,3*(j+1):3*(j+2),:]=original[:,:,:1,:]
  if not np.array_equal(actual,expected):raise RuntimeError('FIRST_REFERENCE_INTERVENTION_MISMATCH')
  return heads

 def collect(self,p,heads,sid,kind,hooks):
  with torch.no_grad():
   a,meta=self.postprocess(self.pp,p['batch'],heads);gt=self.pp.generate_gt_bbx(p['batch'])
  path=BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_1.0.npz'
  with np.load(path,allow_pickle=False) as old:
   mapping,gtmeta=gt_mapping(old['gt_boxes'],array(gt));a['GT_canonical']=array(gt)[mapping]
  a['pred_offset']=array(heads['pred_offset'])
  a['IoU_GT_raw']=self.iou(a['GT_canonical'],a['raw_corners'])
  stages={stage:a['ids_'+stage] for stage in STAGES}
  meta['gt_candidate_stages']=self.attribution(a['IoU_GT_raw'],stages,threshold=.7)
  for t in (.5,.7):
   stat,matched,flags=self.evaluate(a['final_corners'],a['final_scores'],a['GT_canonical'],t)
   a[f'gt_tp{int(t*100)}']=flags;a[f'prediction_gt_idx{int(t*100)}']=matched
   a[f'prediction_fp{int(t*100)}']=(matched<0).astype(np.int8)
  obs=p['observer'];meta['attention']=obs.validate()
  for block in meta['attention']['blocks']:
   for entry in block['summary']:
    agent=int(entry['level'])//3
    if agent>len(p['active']):raise RuntimeError('ATTENTION_LEVEL_SLOT_MISMATCH')
    entry['agent_slot']='ego' if agent==0 else p['active'][agent-1]
    entry['valid_history']=True if agent==0 else entry['agent_slot'] in p['valid_all']
  meta.update(sample_idx=sid,condition=kind,scene_id=self.selection[sid]['scene_id'],input_hash=p['item']['input_hash'],checkpoint_sha256=CHECKPOINT_SHA,slots=p['item']['slots'],active_slots=p['active'],valid_slots=p['valid_all'],GT_mapping=gtmeta,query_ids=sorted(self.queries[sid]))
  for ev in obs.events:
   for key in ('reference','offset','location','weights','spatial_shapes'):a[f'cross_{ev["block"]}_{key}']=ev[key]
  for (block,key),v in obs.projections.items():
   for typ,tensor in v.items():a[f'cross_{block}_{key}_{typ}']=tensor
  a['block0_output_selected']=p['block0_output'][0]
  for i,ref in enumerate(p['references']):a[f'cross_{i}_direct_reference']=ref
  a['reference_original_full']=array(hooks.original) if hooks else p['references'][0].copy()
  a['effective_offset_full']=array(p['effective_offsets'])
  for i,offset in enumerate(p['meta_offsets']):a[f'metaflow_raw_{i}']=offset
  if digest(p['frozen'])!=p['item']['input_hash'] or digest({k:v for k,v in self.model.named_buffers()})!=self.buffer_digest or digest({k:v for k,v in self.model.named_parameters()})!=self.parameter_digest:raise RuntimeError('MODEL_OR_INPUT_MUTATED')
  return a,meta

 def baseline(self,sid):
  rec=self.baselines[sid,1];p=Path(rec['path'])
  if sha_file(p)!=rec['sha256']:raise RuntimeError('BASELINE_HASH')
  with np.load(p,allow_pickle=False) as data:a={k:data[k].copy() for k in data.files}
  a['IoU_GT_raw']=self.iou(a['GT_canonical'],a['raw_corners'])
  for t in (.5,.7):
   _,matched,a[f'gt_tp{int(t*100)}']=self.evaluate(a['final_corners'],a['final_scores'],a['GT_canonical'],t)
   a[f'prediction_gt_idx{int(t*100)}']=matched
  return a

 def cleanup(self,p):
  for h in p.get('handles',[]):h.remove()
  p['handles']=[]
