"""Manual-launch P5-A capture worker; importing does not initialize CUDA."""
import os,sys,json,time,copy,random,resource,signal,shutil,traceback,csv,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from pilot_common import R,MODEL,BASE,array,digest,sha_file,gt_mapping,iou_matrix
from minimal_collector import collect_postprocess
from attention_observer import Observer
FULL=R.parent/'P4_FULL_PREP_20261028_113714_782940';PRE=R.parent/'P5_PRECHECK_20261028_v2';MAX_FORWARD=40;MAX_SECONDS=1800;MAX_OUTPUT=2*2**30
state={'phase':'INITIALIZING','attempted':0,'completed':0,'core':0,'guards':0,'completed_samples':0,'current_sample_idx':None,'alpha':None,'started_epoch':time.time()}
def atomic(p,obj):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def update(**kwargs):
 state.update(kwargs);state['updated_epoch']=time.time();atomic(R/'PPU_PROGRESS.json',state);print(json.dumps(state),flush=True)
def guard(start,for_next=True):
 if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('PPU_WALL_LIMIT')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_LIMIT')
 if shutil.disk_usage(R).free<10*2**30:raise RuntimeError('DISK_LIMIT')
 if sum(x.stat().st_size for x in (R/'capture').rglob('*') if x.is_file())>MAX_OUTPUT:raise RuntimeError('OUTPUT_LIMIT')
 if for_next and state['attempted']>=MAX_FORWARD:raise RuntimeError('FORWARD_LIMIT')
 if torch.cuda.is_initialized():
  free,_=torch.cuda.mem_get_info()
  if free<4*2**30 or torch.cuda.max_memory_allocated()>8*2**30 or torch.cuda.max_memory_reserved()>12*2**30:raise RuntimeError('DEVICE_MEMORY_LIMIT')
def rng_restore(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.random.set_rng_state(x['torch_cpu'])
def save_case(sid,alpha,arrays,meta):
 d=R/'capture'/f'sample_{sid}';d.mkdir(parents=True,exist_ok=True);p=d/f'alpha_{alpha:.1f}.npz';tmp=p.with_suffix('.npz.tmp')
 if p.exists() or tmp.exists():raise RuntimeError('REFUSE_OVERWRITE')
 with tmp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 if sum(x.stat().st_size for x in (R/'capture').rglob('*') if x.is_file())>MAX_OUTPUT:raise RuntimeError('OUTPUT_LIMIT')
 os.replace(tmp,p);atomic(p.with_suffix('.json'),meta);return {'path':str(p),'sha256':sha_file(p),'meta_sha256':sha_file(p.with_suffix('.json'))}
def equal(a,b,label):
 for k in ('cls_preds','reg_preds','dir_preds','pred_offset','final_corners','final_scores','GT_canonical','native_nms_keep'):
  if a[k].shape!=b[k].shape:raise RuntimeError(label+' SHAPE '+k)
  if k=='native_nms_keep':ok=np.array_equal(a[k],b[k])
  else:ok=np.allclose(a[k],b[k],rtol=0,atol=1e-6 if k in ('cls_preds','final_scores') else 1e-5)
  if not ok:raise RuntimeError(label+' VALUE '+k)
def run():
 from p5_control import verify_ready
 claim=json.loads((R/'PPU_LAUNCH.json').read_text());nonce=os.environ.get('P5_A_NONCE')
 if not nonce or nonce!=claim.get('nonce') or os.getppid()!=claim.get('worker_pid'):raise RuntimeError('MANUAL_HANDSHAKE_REQUIRED')
 verify_ready(True)
 rows=list(csv.DictReader((R/'P5_SAMPLE_PROPOSAL_V2.csv').open()));assert len(rows)==len({x['sample_idx'] for x in rows})==18
 ids=[int(x['sample_idx']) for x in rows];assert ids[0]==590
 manifest=json.loads((FULL/'INPUT_MANIFEST.json').read_text());inputs={int(x['sample_idx']):x for x in manifest['samples']};records={}
 for p in sorted((FULL/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
  for x in json.loads(p.read_text())['records']:records[(int(x['sample_idx']),int(x['alpha']))]=x
 queries={s:set() for s in ids}
 for fn in ('P5_FOCAL_ANCHORS.csv','P5_FOCAL_ANCHORS_ALPHA0.csv'):
  for x in csv.DictReader((R/fn).open()):
   s=int(x['sample_idx']);queries[s].add(int(x['anchor_query_index']))
   if x['suppressor_query_index']:queries[s].add(int(x['suppressor_query_index']))
 for s in ids:
  if not queries[s]:queries[s].add(3240)
  if len(queries[s])>256:raise RuntimeError('FOCAL_QUERY_LIMIT')
 random.seed(303);np.random.seed(303);torch.manual_seed(303)
 from opencood.hypes_yaml import yaml_utils
 from opencood.tools import train_utils
 from opencood.data_utils.datasets import build_dataset
 from opencood.utils import eval_utils
 sys.path.insert(0,str(R.parent/'P4_C2_20261027_233036_301795'))
 from c1_reused import eval_native
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));h['train_params']['batch_size']=1;h['validate_dir']=h['test_dir']
 if h['time_delay']!=3 or h['history_frame']!=5:raise RuntimeError('TIME_PROTOCOL')
 model=train_utils.create_model(h);epoch,model=train_utils.load_saved_model(str(MODEL),model);model.eval()
 if epoch!=19 or not all(p.dtype==torch.float32 for p in model.parameters()):raise RuntimeError('MODEL_PROTOCOL')
 dataset=build_dataset(h,visualize=True,train=False);pp=dataset.post_processor
 if pp.params['target_args']['score_threshold']!=.2 or pp.params['nms_thresh']!=.15:raise RuntimeError('POSTPROCESS_PROTOCOL')
 enc=model.backbone.defor_encoder;mods=[]
 for block in enc.blocks:
  for kind,layer in zip(block.cfgs,block.block_layers):
   if kind=='cross_attn':mods.append(layer)
 if len(mods)!=4:raise RuntimeError('CROSS_MODULE_COUNT')
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('PPU_WALL_LIMIT')));signal.alarm(MAX_SECONDS)
 if not torch.cuda.is_available():raise RuntimeError('DEVICE_UNAVAILABLE')
 device=torch.device('cuda:0');model=model.to(device);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();buffers=digest({k:v for k,v in model.named_buffers()})
 gtrows={}
 for x in csv.DictReader((R.parent/'P3B_20261020_103959_980233'/'per_gt.csv').open()):gtrows.setdefault(int(x['sample_idx']),[]).append(x)
 for v in gtrows.values():v.sort(key=lambda x:int(x['gt_idx']))
 def execute(sid,alpha,kind,observe):
  guard(start);item=inputs[sid];p=Path(item['path'])
  if sha_file(p)!=item['sha256']:raise RuntimeError('INPUT_HASH_CHANGED')
  payload=torch.load(p,map_location='cpu',weights_only=False,mmap=False);frozen=payload['batch']
  if digest(frozen)!=item['input_hash'] or digest(payload['rng'])!=item['RNG_hash']:raise RuntimeError('INPUT_OR_RNG_CHANGED')
  rng_restore(payload['rng']);batch=train_utils.to_device(copy.deepcopy(frozen),device)
  if digest(batch)!=item['input_hash'] or digest({k:v for k,v in model.named_buffers()})!=buffers:raise RuntimeError('DEVICE_INPUT_OR_BUFFER_MISMATCH')
  valid_slots=[str(k) for k,v in batch['ego']['calibrate_data'].items() if bool(v['valid_mask'].any().item())];slots=item['slots']
  if any(k not in slots or not any(slots[k]['valid_mask']) for k in valid_slots):raise RuntimeError('SLOT_IDENTITY_MISMATCH')
  offsets=[]
  def meta_hook(module,args,out):
   if tuple(out.shape)!=(1,80,80,2):raise RuntimeError('METAFLOW_SHAPE')
   raw=array(out);effective=out*alpha;offsets.append((raw,array(effective)));return effective
  handle=model.meta_flow.register_forward_hook(meta_hook);observer=Observer(mods,queries[sid]) if observe else None;fused=[]
  def fused_hook(module,args,out):
   t=out['fused_features'].detach();fused.append({'shape':list(t.shape),'mean':float(t.mean().item()),'std':float(t.std(unbiased=False).item()),'finite':bool(torch.isfinite(t).all().item())})
  fused_handle=model.backbone.register_forward_hook(fused_hook) if observe else None
  try:
   update(phase='FORWARD',current_sample_idx=sid,alpha=alpha,kind=kind,attempted=state['attempted']+1)
   with torch.no_grad():
    if observer:
     with observer:heads=model(batch['ego'])
    else:heads=model(batch['ego'])
    if not heads['cls_preds'].is_cuda:raise RuntimeError('NO_DEVICE_OUTPUT')
    update(completed=state['completed']+1)
    a,metadata=collect_postprocess(pp,batch,heads);gt=pp.generate_gt_bbx(batch)
   torch.cuda.synchronize();guard(start,for_next=False)
  finally:
   handle.remove()
   if fused_handle is not None:fused_handle.remove()
  if len(offsets)!=len(valid_slots):raise RuntimeError('METAFLOW_SLOT_COUNT')
  if digest(frozen)!=item['input_hash'] or digest({k:v for k,v in model.named_buffers()})!=buffers:raise RuntimeError('INPUT_OR_BUFFER_CHANGED')
  with np.load(BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz',allow_pickle=False) as old:
   gt_array=array(gt);mapping,gm=gt_mapping(old['gt_boxes'],gt_array);a['GT_canonical']=gt_array[mapping]
   for k,oldk,tol in [('final_corners','pred_boxes',1e-5),('final_scores','pred_scores',1e-6)]:
    if a[k].shape!=old[oldk].shape or not np.allclose(a[k],old[oldk],rtol=0,atol=tol):raise RuntimeError('HISTORY_MISMATCH '+k)
  with np.load(records[(sid,int(alpha))]['path'],allow_pickle=False) as old:
   for k in ('cls_preds','reg_preds','dir_preds','native_nms_keep'):
    if a[k].shape!=old[k].shape or not np.allclose(a[k],old[k],rtol=0,atol=0 if k=='native_nms_keep' else 1e-6):raise RuntimeError('HEAD_HISTORY_MISMATCH '+k)
  a['pred_offset']=array(heads['pred_offset']);meta={'sample_idx':sid,'alpha':alpha,'kind':kind,'input_hash':item['input_hash'],'slots':slots,'valid_slot_call_order':valid_slots,'GT_mapping':gm,'query_ids':sorted(queries[sid]),'attention':None}
  for n,(raw,eff) in enumerate(offsets):a[f'metaflow_raw_{n}']=raw;a[f'metaflow_effective_{n}']=eff
  if observer:
   meta['attention']=observer.validate()
   if len(fused)!=1 or not fused[0]['finite']:raise RuntimeError('FUSED_CAPTURE_MISMATCH')
   meta['fused_summary']=fused[0]
   n=int(batch['ego']['record_len'][0].item());active_slots=[str(k) for k in list(batch['ego']['calibrate_data'].keys())[:n-1]]
   meta['attention']['active_slots']=active_slots
   for block in meta['attention']['blocks']:
    for entry in block['summary']:
     agent=int(entry['level'])//3;entry['agent_slot']='ego' if agent==0 else active_slots[agent-1];entry['valid_history']=True if agent==0 else entry['agent_slot'] in valid_slots
   eff_by_slot={slot:offsets[j][1].reshape(6400,2) for j,slot in enumerate(valid_slots)}
   for ev in observer.events:
    ref=ev['reference'];
    if ref.shape[2]!=3*n:raise RuntimeError('REFERENCE_SLOT_COUNT_MISMATCH')
    base=ref[:,:,0,:]
    if not np.allclose(ref[:,:,:3,:],base[:,:,None,:],rtol=0,atol=1e-6):raise RuntimeError('EGO_REFERENCE_MISMATCH')
    for j,slot in enumerate(active_slots):
     eff=eff_by_slot.get(slot,np.zeros((6400,2),np.float32));expected=base+eff[sorted(queries[sid])][None,:,:]/80.
     if not np.allclose(ref[:,:,3*(j+1):3*(j+2),:],expected[:,:,None,:],rtol=0,atol=1e-6):raise RuntimeError('REMOTE_REFERENCE_MISMATCH '+slot)
   for ev in observer.events:
    i=ev['block']
    for k in ('reference','offset','location','weights','spatial_shapes'):a[f'cross_{i}_{k}']=ev[k]
   for (i,k),v in observer.projections.items():
    for typ,tensor in v.items():a[f'cross_{i}_{k}_{typ}']=tensor
  # Strict threshold and native matching use saved GT identities and original evaluator.
  for threshold in (.5,.7):
   stat,matched,flags=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],threshold)
   expected=np.array([int(x[f'tp{int(threshold*100)}_a{int(alpha)}']) for x in gtrows[sid]])
   if not np.array_equal(flags,expected):raise RuntimeError('HISTORICAL_GT_MATCH_MISMATCH')
   for pi,gi in enumerate(matched):
    if gi>=0 and int(gtrows[sid][gi][f'matched_pred_idx{int(threshold*100)}_a{int(alpha)}'])!=pi:raise RuntimeError('HISTORICAL_MATCH_ID_MISMATCH')
  metadata.update(meta);metadata['native_NMS_replay_exact']=True
  if kind=='core':state['core']+=1
  else:state['guards']+=1
  update(phase='COLLECTING',CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,PPU_allocated_bytes=torch.cuda.memory_allocated(),PPU_reserved_bytes=torch.cuda.memory_reserved())
  del payload,frozen,batch,heads,gt
  return a,metadata
 native={};core={};metas={}
 for alpha in (0.,1.):native[alpha],_=execute(590,alpha,'native',False)
 for alpha in (0.,1.):
  core[alpha],metas[alpha]=execute(590,alpha,'core',True);equal(native[alpha],core[alpha],'SMOKE')
  for k in core[alpha]:
   if k.startswith('metaflow_raw_') and not np.array_equal(native[alpha][k],core[alpha][k]):raise RuntimeError('SMOKE_META_RAW')
 if [digest(core[0.][k]) for k in sorted(core[0.]) if k.startswith('metaflow_raw_')]!=[digest(core[1.][k]) for k in sorted(core[1.]) if k.startswith('metaflow_raw_')]:raise RuntimeError('SMOKE_PAIR_RAW_OFFSET_MISMATCH')
 for alpha in (1.,0.):
  a,meta=execute(590,alpha,'reverse',True);equal(core[alpha],a,'REVERSE')
  if meta['valid_slot_call_order']!=metas[alpha]['valid_slot_call_order']:raise RuntimeError('REVERSE_SLOT_MISMATCH')
  for k in core[alpha]:
   if k.startswith(('cross_','metaflow_raw_','metaflow_effective_')) and (k not in a or a[k].shape!=core[alpha][k].shape or not np.allclose(a[k],core[alpha][k],rtol=0,atol=1e-6)):raise RuntimeError('REVERSE_CAPTURE_MISMATCH '+k)
 del native,a,meta
 update(smoke_passed=True,reverse_passed=True);manifest=[]
 for alpha in (0.,1.):manifest.append({'sample_idx':590,'alpha':alpha,**save_case(590,alpha,core[alpha],metas[alpha])})
 del core,metas
 update(completed_samples=1)
 for sid in ids[1:]:
  raw={}
  for alpha in (0.,1.):
   a,meta=execute(sid,alpha,'core',True);raw[alpha]=[digest(a[k]) for k in sorted(a) if k.startswith('metaflow_raw_')];manifest.append({'sample_idx':sid,'alpha':alpha,**save_case(sid,alpha,a,meta)});del a,meta
  if raw[0.]!=raw[1.]:raise RuntimeError('PAIR_RAW_OFFSET_MISMATCH')
  update(completed_samples=state['completed_samples']+1)
 if state['attempted']!=40 or state['completed']!=40 or state['core']!=36 or state['guards']!=4 or state['completed_samples']!=18:raise RuntimeError('FINAL_COUNT_MISMATCH')
 signal.alarm(0);atomic(R/'CAPTURE_MANIFEST.json',{'passed':True,'records':manifest});update(phase='PPU_CAPTURE_COMPLETE');atomic(R/'PPU_CAPTURE_COMPLETE.json',{'passed':True,'manifest_sha256':sha_file(R/'CAPTURE_MANIFEST.json'),**state});model.cpu();torch.cuda.empty_cache()
if __name__=='__main__':
 try:run()
 except BaseException as e:
  signal.alarm(0);update(phase='PPU_CAPTURE_FAILED',failure=str(e));atomic(R/'PPU_CAPTURE_FAILURE.json',{'passed':False,'error':str(e),'traceback':traceback.format_exc(),'state':state});raise
