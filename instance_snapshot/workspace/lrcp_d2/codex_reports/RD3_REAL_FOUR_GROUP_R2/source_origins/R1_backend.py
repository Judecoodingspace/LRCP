"""Single RD3 inference route for F/G-B/G-C/E; native LRCP and P4 collectors."""
from rd3_common import *
import torch,copy,random,gc,contextlib
from input_integrity import snapshot,check,differences,DisposableHistory,validate_label_offset,LABEL_PATH
from diagnostics import save_unaccepted
from intervention import FixedSlotMask,EvidenceObserver
from minimal_collector import collect_postprocess
from contracts import attribute_gt
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.post_processor import build_postprocessor
from types import SimpleNamespace
_import_path=sys.path.copy()
sys.path.insert(2,str(REPORTS/'P4_C2_20261027_233036_301795'))
from c1_reused import eval_native
sys.path[:]=_import_path

def configuration():
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)))
 require(h['time_delay']==3 and h['history_frame']==5,'TIME_CONFIG')
 require(h['postprocess']['nms_thresh']==.15 and h['postprocess']['target_args']['score_threshold']==.2,'POSTPROCESS_CONFIG')
 return h

def modules(model):
 return [m for b in model.backbone.defor_encoder.blocks for t,m in zip(b.cfgs,b.block_layers) if t=='cross_attn']

class Backend:
 def __init__(self,model,device,kind):
  require(str(device)=='cpu' and kind=='mock','CPU_ONLY_BACKEND_REQUIRED')
  self.model=model;self.device=device;self.kind=kind;self.pp=build_postprocessor(configuration()['postprocess'],False)
  self.modules=modules(model);require(len(self.modules)==4,'FOUR_CROSS_BLOCKS')
  self.params=digest(dict(model.named_parameters()));self.buffers=digest(dict(model.named_buffers()))
  if kind=='real':
   ck=load(C2/'CHECKPOINT_IDENTITY.json');require(self.params==ck['parameter_digest'] and self.buffers==ck['buffer_digest'],'CHECKPOINT_MODEL_IDENTITY')
 @classmethod
 def real(cls):
  raise RuntimeError('CPU_ONLY_REAL_FORWARD_FORBIDDEN')
 def prepare(self,row):
  item=row['input'];require(sha_file(item['path'])==item['sha256'],'FROZEN_BATCH_SHA')
  p=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False)
  require(digest(p['batch'])==item['input_hash'] and digest(p['rng'])==item['RNG_hash'],'FROZEN_INPUT_OR_RNG')
  b=copy.deepcopy(p['batch']);e=b['ego']
  require(int(e['record_len'][0])==3 and list(map(str,e['cav_id_list']))==['1','2','3'],'PHYSICAL_SLOT_IDENTITY')
  require([str(k) for k,v in e['calibrate_data'].items() if v['valid_mask'].any().item()]==['1','2'],'METAFLOW_ACTIVE_SLOTS')
  return {'payload':p,'batch':b,'input_before':digest(b),'leaf_before':snapshot(b),'frozen_leaf_before':snapshot(p['batch']),'row':row}
 def forward_collect(self,p,condition,*,install_mask=True):
  keep=CONDITIONS[condition];m=self.model;row=p['row'];e=p['batch']['ego'];rng=p['payload']['rng']
  random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.random.set_rng_state(rng['torch_cpu'])
  meta_offsets=[];features=[];references=[];handles=[];heads=None;a={};model_returned=False
  history=DisposableHistory(m,p['batch'],fault=p.get('fault'));p['history_context']=history
  def meta_hook(mod,args,out):
   slot=len(meta_offsets)+1;require(slot<=2 and out.shape==(1,80,80,2),'METAFLOW_SLOT_CALL')
   meta_offsets.append(array(out))
  def fusion_hook(mod,args):
   require(torch.equal(args[1],args[1].new_tensor([3])),'FUSION_RECORD_LEN')
   features.extend(digest(x) for x in args[0])
   p['offset_GT']=array(args[3]);p['pred_offsets']=array(args[4])
  def cross_pre(mod,args):references.append(array(args[3]))
  try:
   check(p['leaf_before'],p['batch'])
   handles.append(m.meta_flow.register_forward_hook(meta_hook));handles.append(m.backbone.defor_encoder.register_forward_pre_hook(fusion_hook))
   for layer in self.modules:handles.append(layer.register_forward_pre_hook(cross_pre))
   with history:
    with (FixedSlotMask(self.modules,keep) if install_mask else contextlib.nullcontext(None)) as mask:
     with EvidenceObserver(self.modules,keep) as obs:
      with torch.no_grad():
       if 'on_model_event' in p:p['on_model_event']('MODEL_ENTERED')
       heads=m(e)
       model_returned=True
       p['returned_heads']={k:array(v) for k,v in heads.items()}
       if 'on_model_event' in p:p['on_model_event']('MODEL_RETURNED')
     if mask is not None:mask.validate()
   history.validate()
   for slot in (1,2):
    require(np.array_equal(array(heads['pred_offset'])[:,:,slot-1,:],meta_offsets[slot-1].reshape(1,6400,2)),'METAFLOW_OUTPUT_SLOT_IDENTITY')
   require(len(meta_offsets)==2 and len(features)==3 and len(references)==4,'CAPTURE_CALL_COUNT')
   require(all(np.array_equal(references[0],r) for r in references),'SHARED_REFERENCE_MUTATED')
   ref=references[0];require(ref.shape==(1,6400,9,2),'REFERENCE_SHAPE')
   for slot in (1,2):
    expected=ref[:,:,:1,:]+meta_offsets[slot-1].reshape(1,6400,1,2)/np.array([80,80],np.float32)
    require(np.allclose(ref[:,:,slot*3:(slot+1)*3,:],expected,rtol=0,atol=1e-6),'METAFLOW_SLOT_MISMATCH')
   if self.kind=='real':require(all(heads[k].is_cuda and heads[k].dtype==torch.float32 for k in ('cls_preds','reg_preds','dir_preds')),'REAL_HEAD_DEVICE')
   with torch.no_grad():a,meta=collect_postprocess(self.pp,p['batch'],heads);gt=array(self.pp.generate_gt_bbx(p['batch']))
   old=BASE/f"P3A_20261014_120929_boxes/sample_{row['sample_idx']}_alpha_1.0.npz"
   with np.load(old,allow_pickle=False) as z:mapping,gtmeta=gt_mapping(z['gt_boxes'],gt)
   a['GT_canonical']=gt[mapping];a['IoU_GT_raw']=iou_matrix(a['GT_canonical'],a['raw_corners'])
   meta['gt_candidate_stages']=attribute_gt(a['IoU_GT_raw'],{s:a['ids_'+s] for s in STAGES})
   for t in (.5,.7):
    stat,matched,flags=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],t);s=int(t*100)
    a[f'gt_tp{s}']=flags;a[f'prediction_gt_idx{s}']=matched;a[f'prediction_fp{s}']=(matched<0).astype(np.int8)
   a['pred_offset']=array(heads['pred_offset']);a['offset_GT']=p['offset_GT'];a['reference_full']=ref
   for i,offset in enumerate(meta_offsets):a[f'metaflow_slot_{i+1}']=offset
   for ev in obs.events:
    for k in ('reference','offset','location','weights','spatial_shapes'):a[f'cross_{ev["block"]}_{k}']=ev[k]
   for i,w in obs.full_weights.items():a[f'cross_{i}_full_weights']=w;a[f'cross_{i}_raw_logits_selected']=mask.raw[i] if mask is not None else obs.projections[(i,'attention_weights')]['output']
   for (i,k),v in obs.projections.items():
    for typ,arr in v.items():a[f'cross_{i}_{k}_{typ}']=arr
   par=digest(dict(m.named_parameters()));buf=digest(dict(m.named_buffers()))
   require(par==self.params and buf==self.buffers,'MODEL_OR_BUFFER_MUTATION')
   require(digest(p['payload']['batch'])==row['input']['input_hash'],'FROZEN_INPUT_MUTATION')
   label_evidence=validate_label_offset(p['batch'])
   leaf_after=check(p['leaf_before'],p['batch'],(LABEL_PATH,))
   check(p['frozen_leaf_before'],p['payload']['batch'])
   meta.update(schema='RD3_CAPTURE_V1',backend=self.kind,sample_idx=row['sample_idx'],scene_id=row['scene_id'],scene_frame=row['scene_frame'],condition=condition,keep=keep,slot_physical_ids=['1','2','3'],checkpoint_sha256=CH,frozen_sha256=sha_file(FROZEN),input_hash=row['input']['input_hash'],input_manifest=row,query_ids=QUERIES,attention=obs.validate(),full_attention_checks=obs.full_checks,feature_hashes=features,GT_mapping=gtmeta,parameter_digest=par,buffer_digest=buf,all_original_inputs_unchanged=True)
   meta.update(repair_schema='RD3_R1_CPU_V1',input_leaf_before=p['leaf_before'],input_leaf_after=leaf_after,history_integrity=history.validate(),label_offset=label_evidence,mask_installed=install_mask,component_identity=m.component_identity,mock_scope='native history components and fusion; MetaFlow response and heads are explicit interface mocks')
   return a,meta
  except BaseException as error:
   if p.get('run_root') is not None:
    arrays=dict(p.get('returned_heads',{}));arrays.update(a)
    p['diagnostic_path']=save_unaccepted(p['run_root'],p['attempt'],model_returned,p['leaf_before'],p['batch'],error,arrays,extra={'temporary_histories':[snapshot(t) for t in history.temps],'history_receipts':history.receipts,'metaflow_events':history.meta_events})
   raise
  finally:
   for h in handles:h.remove()
