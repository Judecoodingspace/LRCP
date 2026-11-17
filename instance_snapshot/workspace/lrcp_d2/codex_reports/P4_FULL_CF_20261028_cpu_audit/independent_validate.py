import os,sys,json,csv,time,signal,resource,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent;F=R.parent/'P4_FULL_PREP_20261028_113714_782940';C=R.parent/'P4_C2_20261027_233036_301795'
sys.path.insert(0,str(C));sys.path.insert(0,'/workspace/LRCP')
import numpy as np,torch
import c2_common as c
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('WALL_90_MIN')))
left=int(5400-(time.time()-(R/'input_audit.log').stat().st_ctime));assert left>0;signal.alarm(left)
try:
 assert json.loads((R/'factorial_metrics.json').read_text())['passed']
 results=json.loads((R/'factorial_metrics.json').read_text())['metrics']; pp=c.make_pp();rec={}
 for p in sorted((F/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
  for x in json.loads(p.read_text())['records']:rec[(int(x['sample_idx']),int(x['alpha']))]=x['path']
 ids=list(dict.fromkeys(s for s,a in rec));gr={}
 for x in csv.DictReader((R/'per_GT_mixed.csv').open()):gr.setdefault((int(x['sample_idx']),int(x['geometry'])),[]).append(x)
 tot={g:{t:{'tp':[],'fp':[],'score':[],'gt':0} for t in (.5,.7)} for g in (0,1)};cases=0
 for sid in ids:
  data={}
  for g in (0,1):
   with np.load(rec[(sid,g)],allow_pickle=False) as z:data[g]={k:z[k].copy() for k in ['scores_all','raw_corners','ids_quality_pass','GT_canonical']}
  for g in (0,1):
   geom=data[g];q=geom['ids_quality_pass'];rank=data[1-g]['scores_all'];shadow=c.replay_native_nms(geom['raw_corners'][q],rank[q],.15,c.common_utils.convert_format,c.common_utils.compute_iou);keep=q[shadow['keep']]
   corners,mask=c.box_utils.mask_boxes_outside_range_numpy(geom['raw_corners'][keep],pp.params['gt_range'],order=None,return_mask=True);final=keep[mask];scores=geom['scores_all'][final]
   st70,matched,flags=c.eval_native(corners,scores,geom['GT_canonical'],.7);st50,_,_=c.eval_native(corners,scores,geom['GT_canonical'],.5,False)
   rows=sorted(gr[(sid,g)],key=lambda x:int(x['gt_idx']));assert np.array_equal(flags,np.array([int(x['mixed_TP70']) for x in rows],np.int8))
   for th,stat in [(.5,st50),(.7,st70)]:
    for k in ('tp','fp','score'):tot[g][th][k].extend(stat[k])
    tot[g][th]['gt']+=stat['gt']
   cases+=1
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('RSS_LIMIT')
   assert not torch.cuda.is_initialized()
  if cases%64==0:print('independent',cases,flush=True)
 for g in (0,1):
  for th in (.5,.7):
   v=tot[g][th];ap,_,_=c.eval_utils.calculate_ap({th:v},th);want=results[f'G{g}_{int(th*100)}']
   assert abs(ap-want['AP'])<1e-12 and sum(v['tp'])==want['TP'] and sum(v['fp'])==want['FP'],(g,th,ap,want)
 assert cases==1820
 (R/'independent_validation.json').write_text(json.dumps({'passed':True,'cases':cases,'method':'independent direct polygon NMS replay and range filtering of validated captured full anchors; rematched final GT','exact_AP_tolerance':1e-12,'CPU_RSS_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'CUDA_initialized':False,'model_forwards':0},indent=2));print('independent validation passed',flush=True)
except BaseException as e:
 (R/'INDEPENDENT_FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
