import os,sys,json,time,csv,resource,signal,traceback,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent; F=R.parent/'P4_FULL_PREP_20261028_113714_782940'; C=R.parent/'P4_C2_20261027_233036_301795'; B=R.parent/'P3B_20261020_103959_980233'
sys.path.insert(0,str(C));sys.path.insert(0,'/workspace/LRCP')
import numpy as np,torch
import c2_common as c
LIMIT=90*60
START=(R/'input_audit.log').stat().st_ctime
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('WALL_90_MIN')))
remaining=int(LIMIT-(time.time()-START));assert remaining>0;signal.alarm(remaining)
def guard():
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('RSS_8GiB')
 if torch.cuda.is_initialized():raise RuntimeError('CUDA_INITIALIZED')
 if shutil.disk_usage(R).free<10*2**30:raise RuntimeError('DISK_RESERVE')
 if time.time()-START>LIMIT:raise TimeoutError('WALL_90_MIN')
def load_indexes():
 manifest=json.loads((F/'INPUT_MANIFEST.json').read_text()); inputs={int(x['sample_idx']):x for x in manifest['samples']}
 records={}
 for p in sorted((F/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
  for z in json.loads(p.read_text())['records']:records[(int(z['sample_idx']),int(z['alpha']))]=z
 rows={}
 for x in csv.DictReader((B/'per_gt.csv').open()):rows.setdefault(int(x['sample_idx']),[]).append(x)
 for v in rows.values():v.sort(key=lambda x:int(x['gt_idx']))
 return inputs,records,rows
KEYS=['cls_preds','reg_preds','dir_preds','GT_canonical','raw_corners','scores_all','score_mask','quality_mask','native_nms_keep','range_mask','final_corners','final_scores']+['ids_'+s for s in c.STAGES]
def load_data(inp,records,sid):
 p=torch.load(inp[sid]['path'],map_location='cpu',weights_only=False,mmap=False);e=p['batch']['ego'];a={'ego':{'anchor_box':e['anchor_box'].clone(),'transformation_matrix':e['transformation_matrix'].clone()}};del p
 d={}
 for g in (0,1):
  with np.load(records[(sid,g)]['path'],allow_pickle=False) as z:d[g]={k:z[k].copy() for k in KEYS}
 assert np.array_equal(d[0]['GT_canonical'],d[1]['GT_canonical'])
 return a,d
def totals_init():return {g:{t:{'tp':[],'fp':[],'score':[],'gt':0} for t in ['50','70']} for g in (0,1)}
def add(tot,g,stat):
 for t in ('50','70'):
  for k in ('tp','fp','score'):tot[g][t][k].extend(stat[t][k])
  tot[g][t]['gt']+=stat[t]['gt']
def metrics(tot):
 out={}
 for g in (0,1):
  for t in ('50','70'):
   v=tot[g][t];tp=int(sum(v['tp']));fp=int(sum(v['fp']));ap,rec,prec=c.eval_utils.calculate_ap({float(t)/100:v},float(t)/100)
   out[f'G{g}_{t}']={'AP':ap,'TP':tp,'FP':fp,'GT':v['gt'],'precision_final':tp/(tp+fp) if tp+fp else 0,'recall_final':tp/v['gt']}
 return out

def baseline():
 assert json.loads((R/'INPUT_AUDIT.json').read_text())['passed'] and json.loads((R/'PILOT_GATE.json').read_text())['passed']
 assert not (R/'BASELINE_STARTED').exists();(R/'BASELINE_STARTED').write_text(str(time.time()));(R/'baseline_cases').mkdir()
 inp,rec,gt=load_indexes();pp=c.make_pp();tot=totals_init();cases=0
 for sid in inp:
  inputs,data=load_data(inp,rec,sid)
  for g in (0,1):
   a,stats=c.factorial_postprocess(pp,inputs,data[g],data[g],data[g]);old=data[g]
   for k in ['score_mask','quality_mask','native_nms_keep','range_mask']+['ids_'+s for s in c.STAGES]:
    if not np.array_equal(a[k],old[k]):raise RuntimeError(f'BASE_STAGE {sid} {g} {k}')
   for k,tol in [('raw_corners',1e-5),('final_corners',1e-5),('scores_all',1e-6),('final_scores',1e-6),('GT_canonical',1e-5)]:
    if a[k].shape!=old[k].shape or not np.allclose(a[k],old[k],rtol=0,atol=tol):raise RuntimeError(f'BASE_VALUE {sid} {g} {k}')
   st50,match50,flags50=c.eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],.5)
   if not np.array_equal(flags50,np.array([int(x[f'tp50_a{g}']) for x in gt[sid]],np.int8)):raise RuntimeError(f'BASE_GT_MATCH50 {sid} {g}')
   for pi,gi in enumerate(match50):
    if gi>=0 and int(gt[sid][gi][f'matched_pred_idx50_a{g}'])!=pi:raise RuntimeError(f'BASE_GT_ID50 {sid} {g} {pi}')
   flags=np.array([int(x[f'tp70_a{g}']) for x in gt[sid]],np.int8)
   if not np.array_equal(a['GT_TP70'],flags):raise RuntimeError(f'BASE_GT_MATCH {sid} {g}')
   for pi,gi in enumerate(a['matched_gt_prediction']):
    if gi>=0 and int(gt[sid][gi][f'matched_pred_idx70_a{g}'])!=pi:raise RuntimeError(f'BASE_GT_ID {sid} {g} {pi}')
   np.savez_compressed(R/'baseline_cases'/f'{sid}_{g}.npz',suppressor=a['suppressor'],keep=a['ids_nms_keep'],GT_TP70=a['GT_TP70'],matched_gt_prediction=a['matched_gt_prediction'])
   add(tot,g,stats);cases+=1;guard()
  if cases%64==0:print('baseline',cases,flush=True)
 result=metrics(tot);expected={(0,'50'):.8665290892796786,(0,'70'):.8071592298880969,(1,'50'):.8768414234452961,(1,'70'):.785423020395363}
 for (g,t),x in expected.items():
  if abs(result[f'G{g}_{t}']['AP']-x)>1e-12:raise RuntimeError(f'BASE_AP {g} {t}')
 (R/'baseline_validation.json').write_text(json.dumps({'passed':True,'cases':cases,'metrics':result,'all_nms_and_gt_match':True,'coordinate_tolerance':1e-5,'score_tolerance':1e-6,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
 print('baseline passed',flush=True)

def mixed():
 assert json.loads((R/'baseline_validation.json').read_text())['passed'];assert not (R/'MIXED_STARTED').exists();(R/'MIXED_STARTED').write_text(str(time.time()))
 inp,rec,gt=load_indexes();pp=c.make_pp();tot=totals_init();cases=0
 pergt={}
 for row in csv.DictReader((F/'per_GT_full.csv').open()):pergt[(int(row['sample_idx']),int(row['gt_idx']),int(row['alpha']))]=row
 assert len(pergt)==8164*2
 fields=['sample_idx','gt_idx','geometry','native_TP70','mixed_TP70','transition','historic_transition','native_stage','raw','score','quality','top1000','nms','range','raw_strict_anchor_ids','nms_strict_anchor_ids','range_strict_anchor_ids']
 with (R/'per_GT_mixed.csv').open('x',newline='') as gf,(R/'nms_priority_changes.csv').open('x',newline='') as nf,(R/'case_metrics.csv').open('x',newline='') as cf,(R/'score_geometry_alignment.csv').open('x',newline='') as af,(R/'GT_nms_events.csv').open('x',newline='') as ef:
  gw=csv.DictWriter(gf,fieldnames=fields);gw.writeheader()
  nfields=['sample_idx','geometry','anchor_id','native_kept','mixed_kept','native_suppressor','mixed_suppressor','native_suppressor_rank_score','mixed_suppressor_rank_score','admission_score','native_rank_score','mixed_rank_score','native_suppression_iou','mixed_suppression_iou'];nw=csv.DictWriter(nf,fieldnames=nfields);nw.writeheader()
  aw=csv.DictWriter(af,fieldnames=['sample_idx','gt_idx','geometry','positive_overlap_count','max_iou_anchor','max_iou','native_top_anchor','native_top_iou','native_top_score','mixed_top_anchor','mixed_top_iou','mixed_top_score','native_top_is_max_iou','mixed_top_is_max_iou']);aw.writeheader()
  ew=csv.DictWriter(ef,fieldnames=['sample_idx','gt_idx','geometry','condition','anchor_id','anchor_iou_GT','anchor_rank_score','suppressor','suppressor_iou_GT','suppressor_rank_score']);ew.writeheader()
  cw=csv.DictWriter(cf,fieldnames=['sample_idx','geometry','native_tp70','mixed_tp70','mixed_fp70','top1000_truncated']);cw.writeheader()
  for sid in inp:
   inputs,data=load_data(inp,rec,sid)
   for g in (0,1):
    a,stats=c.factorial_postprocess(pp,inputs,data[g],data[g],data[1-g]);old=data[g]
    with np.load(R/'baseline_cases'/f'{sid}_{g}.npz',allow_pickle=False) as b:base={k:b[k].copy() for k in b.files}
    if not np.array_equal(a['score_mask'],old['score_mask']) or not np.array_equal(a['quality_mask'],old['quality_mask']):raise RuntimeError('ADMISSION_CHANGED')
    if not np.allclose(a['raw_corners'],old['raw_corners'],rtol=0,atol=1e-5):raise RuntimeError('GEOMETRY_CHANGED')
    add(tot,g,stats);q=a['ids_quality_pass'];native=np.zeros(12800,bool);native[base['keep']]=1;mix=np.zeros(12800,bool);mix[a['ids_nms_keep']]=1
    native_sup=base['suppressor'];mixed_sup=a['suppressor'];nscores=old['scores_all'];rscores=data[1-g]['scores_all']
    shadow=c.replay_native_nms(old['raw_corners'][q],nscores[q],.15,c.common_utils.convert_format,c.common_utils.compute_iou)
    if not np.array_equal(q[shadow['keep']],base['keep']):raise RuntimeError('NATIVE_SHADOW_RECHECK_FAILED')
    mat=c.iou_matrix(a['GT_canonical'],old['raw_corners'][q]);qpos={int(x):j for j,x in enumerate(q)}
    native_overlap=np.zeros(12800,np.float32)
    for item in shadow['suppressed']:native_overlap[q[item['removed_nms_input_idx']]]=item['bev_iou_float32']
    for ai in q:
     ai=int(ai);s0=int(native_sup[ai]);s1=int(mixed_sup[ai])
     if native[ai]==mix[ai] and s0==s1:continue
     nw.writerow(dict(sample_idx=sid,geometry=g,anchor_id=ai,native_kept=int(native[ai]),mixed_kept=int(mix[ai]),native_suppressor=s0,mixed_suppressor=s1,native_suppressor_rank_score=float(nscores[s0]) if s0>=0 else '',mixed_suppressor_rank_score=float(rscores[s1]) if s1>=0 else '',admission_score=float(nscores[ai]),native_rank_score=float(nscores[ai]),mixed_rank_score=float(rscores[ai]),native_suppression_iou=float(native_overlap[ai]) if s0>=0 else '',mixed_suppression_iou=float(a['suppressor_overlap'][ai]) if s1>=0 else ''))
    for gi,x in enumerate(gt[sid]):
     hist=pergt[(sid,gi,g)];v=mat[gi];idx=np.flatnonzero(v>0)
     if len(idx):
      im=idx[v[idx].argsort()[::-1][0]];in0=idx[nscores[q[idx]].argsort()[::-1][0]];in1=idx[rscores[q[idx]].argsort()[::-1][0]]
      aw.writerow(dict(sample_idx=sid,gt_idx=gi,geometry=g,positive_overlap_count=len(idx),max_iou_anchor=int(q[im]),max_iou=float(v[im]),native_top_anchor=int(q[in0]),native_top_iou=float(v[in0]),native_top_score=float(nscores[q[in0]]),mixed_top_anchor=int(q[in1]),mixed_top_iou=float(v[in1]),mixed_top_score=float(rscores[q[in1]]),native_top_is_max_iou=int(abs(float(v[in0])-float(v[im]))<=1e-6),mixed_top_is_max_iou=int(abs(float(v[in1])-float(v[im]))<=1e-6)))
     for j in np.flatnonzero(v>=.7):
      ai=int(q[j])
      for label,sup,rank in [('native',int(native_sup[ai]),nscores),('mixed',int(mixed_sup[ai]),rscores)]:
       if sup>=0:ew.writerow(dict(sample_idx=sid,gt_idx=gi,geometry=g,condition=label,anchor_id=ai,anchor_iou_GT=float(v[j]),anchor_rank_score=float(rank[ai]),suppressor=sup,suppressor_iou_GT=float(v[qpos[sup]]),suppressor_rank_score=float(rank[sup])))
     rawids=np.array(json.loads(hist['ids_raw_decoded']),np.int32);stageids=[a['ids_'+s] for s in c.STAGES];lst=[rawids[np.isin(rawids,v)].tolist() for v in stageids]
     oldtp=int(base['GT_TP70'][gi]);newtp=int(a['GT_TP70'][gi]);tr=('TP' if oldtp else 'FN')+'_to_'+('TP' if newtp else 'FN')
     gw.writerow(dict(sample_idx=sid,gt_idx=gi,geometry=g,native_TP70=oldtp,mixed_TP70=newtp,transition=tr,historic_transition=x['match_transition70'],native_stage=hist['last_strict_disappearance'],raw=len(lst[0]),score=len(lst[1]),quality=len(lst[2]),top1000=len(lst[3]),nms=len(lst[4]),range=len(lst[5]),raw_strict_anchor_ids=json.dumps(lst[0]),nms_strict_anchor_ids=json.dumps(lst[4]),range_strict_anchor_ids=json.dumps(lst[5])))
    cw.writerow(dict(sample_idx=sid,geometry=g,native_tp70=int(base['GT_TP70'].sum()),mixed_tp70=int(a['GT_TP70'].sum()),mixed_fp70=sum(stats['70']['fp']),top1000_truncated=max(0,len(q)-1000)))
    cases+=1;guard()
   if cases%64==0:gf.flush();nf.flush();cf.flush();af.flush();ef.flush();print('mixed',cases,flush=True)
 with (R/'global_PR.csv').open('x',newline='') as prf:
  w=csv.writer(prf);w.writerow(['geometry','threshold','rank','score','TP','FP','recall','precision'])
  for g in (0,1):
   for th in ('50','70'):
    v=tot[g][th];ix=np.argsort(-np.asarray(v['score']));tp=np.cumsum(np.asarray(v['tp'])[ix]);fp=np.cumsum(np.asarray(v['fp'])[ix])
    for j,i in enumerate(ix):w.writerow([g,th,j+1,v['score'][i],v['tp'][i],v['fp'][i],tp[j]/v['gt'],tp[j]/(tp[j]+fp[j])])
 (R/'factorial_metrics.json').write_text(json.dumps({'passed':True,'cases':cases,'metrics':metrics(tot),'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
if __name__=='__main__':
 try:
  if sys.argv[1]=='baseline':baseline()
  elif sys.argv[1]=='mixed':mixed()
  else:raise RuntimeError('UNKNOWN_PHASE')
 except BaseException as e:
  (R/'FAILURE.json').write_text(json.dumps({'phase':sys.argv[1],'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
