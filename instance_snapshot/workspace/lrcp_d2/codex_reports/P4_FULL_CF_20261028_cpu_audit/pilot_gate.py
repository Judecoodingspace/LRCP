import os,sys,json,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent
C=R.parent/'P4_C2_20261027_233036_301795'; F=R.parent/'P4_FULL_PREP_20261028_113714_782940'
sys.path.insert(0,str(C));sys.path.insert(0,'/workspace/LRCP')
import numpy as np,torch
import c2_common as c
try:
 assert json.loads((R/'INPUT_AUDIT.json').read_text())['passed']
 assert json.loads((C/'validation.json').read_text())['passed']
 inp=json.loads((F/'INPUT_MANIFEST.json').read_text());by={int(x['sample_idx']):x for x in inp['samples']}
 records={}
 for p in sorted((F/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
  for z in json.loads(p.read_text())['records']:records[(int(z['sample_idx']),int(z['alpha']))]=z
 pp=c.make_pp();checks=[]; start=time.monotonic()
 for row in c.chosen():
  sid=int(row['sample_idx']); payload=torch.load(by[sid]['path'],map_location='cpu',weights_only=False,mmap=False)
  ego=payload['batch']['ego']; inputs={'ego':{'anchor_box':ego['anchor_box'].clone(),'transformation_matrix':ego['transformation_matrix'].clone()}};del payload
  pil={}
  for g in [0,1]:
   with np.load(records[(sid,g)]['path'],allow_pickle=False) as z:pil[g]={k:z[k].copy() for k in ['cls_preds','reg_preds','dir_preds','GT_canonical']}
  for g,a,r in [(0,0,0),(0,0,1),(1,1,1),(1,1,0)]:
   arr,stats=c.factorial_postprocess(pp,inputs,pil[g],pil[a],pil[r]);old,ost=c.load_c2(sid,g,a,r)
   keys=['raw_corners','scores_all','rank_scores_all','score_mask','quality_mask','native_nms_keep','range_mask','final_corners','final_scores','GT_canonical','GT_TP70','matched_gt_prediction','suppressor','suppressor_overlap','first_deletion_code']+['ids_'+x for x in c.STAGES]
   for k in keys:
    if not np.array_equal(arr[k],old[k]):raise RuntimeError(f'P4_C2_MISMATCH {sid} G{g}A{a}R{r} {k}')
   for t in ['50','70']:
    if stats[t]!=ost[t]:raise RuntimeError(f'P4_C2_STATS_MISMATCH {sid} {g}{a}{r} {t}')
   checks.append([sid,g,a,r])
  print('pilot',sid,'conditions',len(checks),flush=True)
 assert len(checks)==72 and not torch.cuda.is_initialized()
 (R/'PILOT_GATE.json').write_text(json.dumps({'passed':True,'cases':len(checks),'exact_array_and_stats':True,'elapsed_seconds':time.monotonic()-start,'CUDA_initialized':False,'model_forwards':0},indent=2))
except BaseException as e:
 (R/'PILOT_GATE_FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
