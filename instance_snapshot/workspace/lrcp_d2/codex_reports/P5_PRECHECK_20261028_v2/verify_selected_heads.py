import os,sys,csv,json,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent;F=R.parent/'P4_FULL_PREP_20261028_113714_782940';C=R.parent/'P4_C2_20261027_233036_301795';B=R.parent/'P3B_20261020_103959_980233'
sys.path.insert(0,str(C));sys.path.insert(0,'/workspace/LRCP')
import numpy as np,torch
import c2_common as c
records={}
for p in sorted((F/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
 for x in json.loads(p.read_text())['records']:records[(int(x['sample_idx']),int(x['alpha']))]=x['path']
rows={}
for x in csv.DictReader((B/'per_gt.csv').open()):rows.setdefault(int(x['sample_idx']),[]).append(x)
for v in rows.values():v.sort(key=lambda x:int(x['gt_idx']))
checks=[]
for s in csv.DictReader((R/'P5_SAMPLE_PROPOSAL_V2.csv').open()):
 sid=int(s['sample_idx']);gi=int(s['gt_idx'])
 for alpha in (0,1):
  with np.load(records[(sid,alpha)],allow_pickle=False) as z:
   gt=z['GT_canonical'];pred=z['final_corners'];score=z['final_scores']
   with np.load(f'/workspace/lrcp_d2/P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz',allow_pickle=False) as old:
    assert np.allclose(gt,old['gt_boxes'],atol=1e-5,rtol=0) and np.allclose(pred,old['pred_boxes'],atol=1e-5,rtol=0) and np.allclose(score,old['pred_scores'],atol=1e-6,rtol=0)
   for t in (.5,.7):
    stat,matched,flags=c.eval_native(pred,score,gt,t)
    assert np.array_equal(flags,np.array([int(x[f'tp{int(t*100)}_a{alpha}']) for x in rows[sid]]))
    for pi,idx in enumerate(matched):
     if idx>=0:assert int(rows[sid][idx][f'matched_pred_idx{int(t*100)}_a{alpha}'])==pi
   checks.append({'sample_idx':sid,'gt_idx':gi,'alpha':alpha,'GT_count':len(gt),'GT_TP70':int(flags[gi])})
assert len(checks)==36 and not torch.cuda.is_initialized()
(R/'HEAD_GT_BASELINE.json').write_text(json.dumps({'passed':True,'conditions':36,'checks':checks,'GT_mapping':'P3-A canonical','CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
