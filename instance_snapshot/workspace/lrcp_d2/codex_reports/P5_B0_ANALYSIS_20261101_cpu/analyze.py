"""CPU-only descriptive analysis of accepted real P5-B0; no model construction."""
import os,sys,csv,json,hashlib,resource,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;R=ROOT.parent;C2=R/'P5_B0_C2_20261031_cpu';REAL=R/'P5_B0_C3_20261031_cpu/real_run';P5=R/'P5_A_CPU_FIX_20261028_storage'
sys.path.insert(0,str(C2));sys.path.insert(0,'/workspace/LRCP')
from pilot_common import sha_file,iou_matrix,STAGES
from contracts import attribute_gt
from INDEPENDENT_VALIDATOR import verify_reference,verify_attention
sys.path.insert(0,str(R/'P4_C2_20261027_233036_301795'))
from c1_reused import eval_native
from opencood.utils import eval_utils

def csvout(name,rows):
 if not rows:return
 with (ROOT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsout(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def diff(a,b):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));return {'equal':bool(np.array_equal(a,b)),'mean_abs':float(d.mean()) if d.size else 0.,'max_abs':float(d.max()) if d.size else 0.,'l2':float(np.linalg.norm(d.reshape(-1)))}
def trans(a,b):return ('TP' if a else 'FN')+'_to_'+('TP' if b else 'FN')
def guard():assert not torch.cuda.is_initialized();assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*2**30
sel=list(csv.DictReader((C2/'P5_SAMPLE_PROPOSAL_V2.csv').open()));selection={int(x['sample_idx']):x for x in sel}
accept=json.loads((REAL/'CPU_VALIDATION_COMPLETE.json').read_text());man=json.loads((REAL/'manifest.json').read_text());bud=json.loads((REAL/'budget.json').read_text())
assert accept['passed'] and not accept['mock_only'] and accept['result']['validated_records']==22 and man['kind']=='real'
assert sha_file(REAL/'manifest.json')==accept['manifest_sha256'] and len(bud['attempts'])==22
history={(int(x['sample_idx']),int(x['alpha'])):x for x in json.loads((P5/'CAPTURE_MANIFEST.json').read_text())['records']}
records={(x['sample_idx'],x['condition']):x for x in man['records']}
allgt=[];focal=[];stages_out=[];anchors=[];nms_changes=[];blockrows=[];boundaries=[];headrows=[];sample_rows=[];feature_rows=[];stats={};source_hashes={};stage_names=STAGES

def load_historical(sid,alpha):
 rec=history[sid,alpha];p=Path(rec['path']);assert sha_file(p)==rec['sha256'] and sha_file(p.with_suffix('.json'))==rec['meta_sha256'];source_hashes[str(p)]=rec['sha256'];source_hashes[str(p.with_suffix('.json'))]=rec['meta_sha256']
 with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 return a,json.loads(p.with_suffix('.json').read_text())
def load_real(sid,kind):
 rec=records[sid,kind];p=Path(rec['path']);assert sha_file(p/'manifest.json')==rec['manifest_sha256'];m=json.loads((p/'manifest.json').read_text())
 for n,h in m['files'].items():assert sha_file(p/n)==h;source_hashes[str(p/n)]=h
 source_hashes[str(p/'manifest.json')]=rec['manifest_sha256']
 with np.load(p/'payload.bin',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 return a,json.loads((p/'meta.json').read_text())
for n,x in enumerate(sel):
 sid=int(x['sample_idx']);gid=int(x['gt_idx']);scene=str(x['scene_id']);category=x['category'];print(json.dumps({'sample':sid,'index':n+1}),flush=True)
 a0,m0=load_historical(sid,0);a1,m1=load_historical(sid,1);b,mb=load_real(sid,'core');assert mb['query_ids']==m1['query_ids']==m0['query_ids'];assert np.array_equal(a1['GT_canonical'],b['GT_canonical']);assert np.array_equal(a0['GT_canonical'],b['GT_canonical'])
 verify_reference(b,mb);verify_attention(b,mb)
 cases={'alpha0':a0,'alpha1':a1,'B0':b};metas={'alpha0':m0,'alpha1':m1,'B0':mb};ng=len(b['GT_canonical']);condition={}
 for label,a in cases.items():
  iou=a['IoU_GT_raw'] if 'IoU_GT_raw' in a else iou_matrix(a['GT_canonical'],a['raw_corners']);a['IoU_GT_raw']=iou
  attrib=attribute_gt(iou,{k:a['ids_'+k] for k in STAGES});a['_attrib']=attrib
  predids=a['ids_final_range'];assert len(predids)==len(a['final_scores'])
  condition[label]={}
  for t in (.5,.7):
   stat,matched,flags=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],t);kk=int(t*100)
   if f'gt_tp{kk}' in a:assert np.array_equal(a[f'gt_tp{kk}'],flags) and np.array_equal(a[f'prediction_gt_idx{kk}'],matched)
   a[f'_tp{kk}']=flags;a[f'_match{kk}']=matched
   stats[sid,label,kk]=stat;condition[label][str(kk)]={'TP':int(flags.sum()),'FP':int((matched<0).sum()),'predictions':len(matched),'GT':ng}
  for g in range(ng):
   row={'sample_idx':sid,'scene_id':scene,'gt_idx':g,'is_focal':g==gid,'category':category if g==gid else 'nonfocal','condition':label,'TP50':int(a['_tp50'][g]),'TP70':int(a['_tp70'][g]),'stage':attrib[g]['availability_class']}
   for stage in STAGES:row['count_'+stage]=attrib[g]['high_iou_count'][stage];row['best_iou_'+stage]=attrib[g]['best_iou'][stage]
   stages_out.append(row)
  for block in range(4):
   loc=a[f'cross_{block}_location'];shape=a[f'cross_{block}_spatial_shapes'];q=len(mb['query_ids'])
   for level,(H,W) in enumerate(shape):
    agent=level//3;slot='ego' if agent==0 else mb['active_slots'][agent-1];valid=slot=='ego' or slot in mb['valid_slots'];v=loc[:,:,:,level,:,:]
    outside=((v<0)|(v>1)).any(-1);xx=v[...,0]*W-.5;yy=v[...,1]*H-.5;unsupported=(xx<=-1)|(xx>=W)|(yy<=-1)|(yy>=H)
    boundaries.append({'sample_idx':sid,'scene_id':scene,'condition':label,'block':block,'query_count':q,'level':level,'slot':slot,'group':'ego' if agent==0 else ('valid_remote' if valid else 'invalid_remote'),'count':int(outside.size),'oob':int(outside.sum()),'unsupported':int(unsupported.sum())})
 for g in range(ng):
  row={'sample_idx':sid,'scene_id':scene,'scene_frame':x['scene_frame'],'gt_idx':g,'is_focal':g==gid,'category':category if g==gid else 'nonfocal'}
  for label,a in cases.items():
   row[f'TP50_{label}']=int(a['_tp50'][g]);row[f'TP70_{label}']=int(a['_tp70'][g]);row[f'stage_{label}']=a['_attrib'][g]['availability_class'];row[f'max_raw_IoU_{label}']=float(a['IoU_GT_raw'][g].max())
   match=np.flatnonzero(a['_match70']==g);row[f'matched_anchor70_{label}']=int(a['ids_final_range'][match[0]]) if len(match) else -1
  row['transition70']=trans(row['TP70_alpha1'],row['TP70_B0']);row['transition50']=trans(row['TP50_alpha1'],row['TP50_B0']);allgt.append(row)
  if g==gid:focal.append({**row,'P4_rank_swap_label':x['mixed_transition'],'original_native_stage':x['native_stage']})
 for block in range(4):
  row={'sample_idx':sid,'scene_id':scene,'category':category,'block':block,'query_count':len(mb['query_ids'])}
  for key in ['reference','offset','location','weights','sampling_offsets_input','sampling_offsets_output','attention_weights_output']:
   for k,v in diff(a1[f'cross_{block}_{key}'],b[f'cross_{block}_{key}']).items():row[key+'_'+k]=v
  row['direct_reference_alpha1_preserved']=bool(np.array_equal(b[f'cross_{block}_direct_reference'],b['reference_original_full']))
  blockrows.append(row)
 if sid in (590,918):
  native,meta=load_real(sid,'off');assert np.array_equal(native['reference_original_full'],b['reference_original_full'])
  feature_rows.append({'sample_idx':sid,'query_count':len(mb['query_ids']),'scope':'selected queries only','baseline':'new verified hook-off control',**diff(native['block0_output_selected'],b['block0_output_selected'])})
 for head in ('cls_preds','reg_preds','dir_preds','scores_all','decoded_before_dir','raw_corners'):
  row={'sample_idx':sid,'scene_id':scene,'field':head}
  for pair,l,r in [('B0_vs_alpha1',a1,b),('alpha0_vs_alpha1',a1,a0),('B0_vs_alpha0',a0,b)]:
   for key,value in diff(l[head],r[head]).items():row[pair+'_'+key]=value
  headrows.append(row)
 raw_keys=[k for k in b if k.startswith('metaflow_raw_')];assert all(np.array_equal(a1[k],b[k]) for k in raw_keys)
 keep1=set(map(int,a1['ids_nms_keep']));keepb=set(map(int,b['ids_nms_keep']));allids=set(range(12800));changed=(a1['suppressor']!=b['suppressor'])|(a1['first_deletion_code']!=b['first_deletion_code'])
 for aid in np.flatnonzero(changed):
  nms_changes.append({'sample_idx':sid,'scene_id':scene,'anchor_id':int(aid),'score_alpha1':float(a1['scores_all'][aid]),'score_B0':float(b['scores_all'][aid]),'deletion_code_alpha1':int(a1['first_deletion_code'][aid]),'deletion_code_B0':int(b['first_deletion_code'][aid]),'suppressor_alpha1':int(a1['suppressor'][aid]),'suppressor_B0':int(b['suppressor'][aid]),'overlap_alpha1':float(a1['suppressor_overlap'][aid]),'overlap_B0':float(b['suppressor_overlap'][aid]),'kept_alpha1':int(aid in keep1),'kept_B0':int(aid in keepb),'focal_IoU_alpha1':float(a1['IoU_GT_raw'][gid,aid]),'focal_IoU_B0':float(b['IoU_GT_raw'][gid,aid])})
 strict=np.flatnonzero((a0['IoU_GT_raw'][gid]>=.7)|(a1['IoU_GT_raw'][gid]>=.7)|(b['IoU_GT_raw'][gid]>=.7));winners=set()
 for a in (a0,a1,b):
  for aid in strict:
   w=int(a['suppressor'][aid])
   if w>=0:winners.add(w)
 for aid in sorted(set(map(int,strict))|winners):
  row={'sample_idx':sid,'scene_id':scene,'gt_idx':gid,'category':category,'anchor_id':aid,'strict_in_any_condition':aid in set(strict),'suppressor_of_strict_candidate':aid in winners}
  for label,a in cases.items():
   row['score_'+label]=float(a['scores_all'][aid]);row['IoU_'+label]=float(a['IoU_GT_raw'][gid,aid]);row['deletion_code_'+label]=int(a['first_deletion_code'][aid]);row['suppressor_'+label]=int(a['suppressor'][aid]);row['overlap_'+label]=float(a['suppressor_overlap'][aid]);row['final_'+label]=int(aid in a['ids_final_range'])
  anchors.append(row)
 sr={'sample_idx':sid,'scene_id':scene,'category':category,'GT':ng,'focal_gt_idx':gid,'TP70_alpha0':condition['alpha0']['70']['TP'],'TP70_alpha1':condition['alpha1']['70']['TP'],'TP70_B0':condition['B0']['70']['TP'],'recoveries':int(((a1['_tp70']==0)&(b['_tp70']==1)).sum()),'degradations':int(((a1['_tp70']==1)&(b['_tp70']==0)).sum()),'nms_keep_order_changed':not np.array_equal(a1['ids_nms_keep'],b['ids_nms_keep']),'nms_keep_set_changed':keep1!=keepb,'nms_keep_added':json.dumps(sorted(keepb-keep1)),'nms_keep_removed':json.dumps(sorted(keep1-keepb)),'changed_stage_or_suppressor_anchors':int(changed.sum()),'raw_metaflow_unchanged':True,'block0_full_reference_max_abs_change':float(np.max(np.abs(b['cross_0_direct_reference']-b['reference_original_full'])))}
 for label,a in cases.items():sr['predictions_'+label]=len(a['final_scores'])
 sample_rows.append(sr);guard();del a0,a1,b,cases

def pooled(ids,label,th,curve=False):
 dest={'score':[],'tp':[],'fp':[],'gt':0}
 for sid in ids:
  d=stats[sid,label,th]
  for k in ('score','tp','fp'):dest[k].extend(d[k])
  dest['gt']+=d['gt']
 ap,_,_=eval_utils.calculate_ap({th/100:dest},th/100);tp=sum(dest['tp']);fp=sum(dest['fp'])
 result={'TP':tp,'FP':fp,'FN':dest['gt']-tp,'GT':dest['gt'],'predictions':len(dest['score']),'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/dest['gt'],'AP_selected18':float(ap)}
 if curve:
  order=np.argsort(-np.asarray(dest['score']));tpc=np.cumsum(np.asarray(dest['tp'])[order]);fpc=np.cumsum(np.asarray(dest['fp'])[order])
  rows=[{'condition':label,'threshold':th,'rank':i+1,'score':dest['score'][j],'TP_cumulative':int(tpc[i]),'FP_cumulative':int(fpc[i]),'recall':float(tpc[i]/dest['gt']),'precision':float(tpc[i]/(tpc[i]+fpc[i]))} for i,j in enumerate(order)]
  return result,rows
 return result
ids=[int(x['sample_idx']) for x in sel];metrics={};pr=[]
for label in ('alpha0','alpha1','B0'):
 metrics[label]={}
 for t in (50,70):metrics[label][str(t)],rows=pooled(ids,label,t,True);pr+=rows
robust=[]
for unit in ('sample','scene'):
 groups=[str(s) for s in ids] if unit=='sample' else sorted({str(x['scene_id']) for x in sel})
 for group in groups:
  remaining=[s for s in ids if (str(s) if unit=='sample' else str(selection[s]['scene_id']))!=group]
  old=pooled(remaining,'alpha1',70);new=pooled(remaining,'B0',70)
  robust.append({'omitted_unit':unit,'omitted_id':group,'samples_remaining':len(remaining),'GT_remaining':old['GT'],'net_TP70':new['TP']-old['TP'],'AP70_alpha1':old['AP_selected18'],'AP70_B0':new['AP_selected18'],'AP70_delta':new['AP_selected18']-old['AP_selected18']})
scenes=[]
for scene in sorted({x['scene_id'] for x in sample_rows},key=int):
 rr=[x for x in sample_rows if x['scene_id']==scene];scenes.append({'scene_id':scene,'samples':len(rr),'GT':sum(x['GT'] for x in rr),'TP70_alpha1':sum(x['TP70_alpha1'] for x in rr),'TP70_B0':sum(x['TP70_B0'] for x in rr),'recoveries':sum(x['recoveries'] for x in rr),'degradations':sum(x['degradations'] for x in rr)})
groups=[]
for group in dict.fromkeys(x['category'] for x in sel):
 rr=[x for x in focal if x['category']==group];groups.append({'category':group,'n':len(rr),'TP70_alpha0':sum(x['TP70_alpha0'] for x in rr),'TP70_alpha1':sum(x['TP70_alpha1'] for x in rr),'TP70_B0':sum(x['TP70_B0'] for x in rr),'recoveries':sum(x['transition70']=='FN_to_TP' for x in rr),'degradations':sum(x['transition70']=='TP_to_FN' for x in rr)})
summary={'samples':18,'scenes':len(scenes),'GT':len(allgt),'focal_GT':18,'metrics':metrics,'transitions70':{k:sum(x['transition70']==k for x in allgt) for k in ('TP_to_TP','FN_to_TP','TP_to_FN','FN_to_FN')},'changed_GT70':[x for x in allgt if x['transition70'] in ('FN_to_TP','TP_to_FN')],'focal_groups':groups,'nms_keep_order_changed_samples':sum(x['nms_keep_order_changed'] for x in sample_rows),'nms_keep_set_changed_samples':sum(x['nms_keep_set_changed'] for x in sample_rows),'changed_stage_or_suppressor_anchor_rows':len(nms_changes),'selected_queries_total':sum(x['query_count'] for x in blockrows if x['block']==0),'block_changes':{},'block0_output_comparison_samples':len(feature_rows),'CPU_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'CUDA_initialized':False,'new_model_forward':0,'scope':'mechanism-stratified descriptive selected18; not population-unbiased or full910 AP'}
for i in range(4):
 rr=[x for x in blockrows if x['block']==i];summary['block_changes'][str(i)]={k:sum(not x[k+'_equal'] for x in rr) for k in ['reference','offset','location','weights','sampling_offsets_input','sampling_offsets_output','attention_weights_output']}
for name,rows in [('per_GT.csv',allgt),('focal_results.csv',focal),('stage_migration.csv',stages_out),('focal_anchor_competition.csv',anchors),('nms_competition_changes.csv',nms_changes),('block_changes.csv',blockrows),('sampling_boundaries.csv',boundaries),('head_changes.csv',headrows),('per_sample.csv',sample_rows),('per_scene.csv',scenes),('focal_groups.csv',groups),('PR_curves.csv',pr),('leave_one_group_out.csv',robust),('block0_output_controls.csv',feature_rows)]:csvout(name,rows)
jsout('summary.json',summary);jsout('SOURCE_HASHES.json',{'accepted_manifest':accept['manifest_sha256'],'files':source_hashes});jsout('AP_METRICS.json',metrics)
assert {(x['sample_idx'],x['gt_idx'],x['TP70_alpha1'],x['TP70_B0']) for x in allgt}=={(x['sample_idx'],x['gt_idx'],x['native_tp70'],x['intervened_tp70']) for x in accept['result']['per_GT_transitions']}
guard();print(json.dumps(summary,indent=2,ensure_ascii=False))
