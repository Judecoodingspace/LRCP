import os,sys,json,csv,traceback,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
from types import SimpleNamespace
from pilot_common import MODEL,iou_matrix,STAGES,sha_file
from contracts import attribute_gt
from minimal_collector import collect_postprocess
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P4_C2_20261027_233036_301795')
from c1_reused import eval_native,postprocess
R=Path(__file__).parent;A=R.parent/'P5_A_CPU_FIX_20261028_storage';F=R.parent/'P4_FULL_PREP_20261028_113714_782940'
ready=json.loads((A/'P5_READY.json').read_text());selected={int(x['sample_idx']):x for x in ready['samples']};records=json.loads((A/'CAPTURE_MANIFEST.json').read_text())['records']
full={}
for p in (F/'capture').glob('shard_*/COLLECTION_MANIFEST.json'):
 for x in json.loads(p.read_text())['records']:full[int(x['sample_idx']),int(x['alpha'])]=x
ref={}
for x in csv.DictReader((R.parent/'P3B_20261020_103959_980233'/'per_gt.csv').open()):
 if int(x['sample_idx']) in selected:ref.setdefault(int(x['sample_idx']),[]).append(x)
for v in ref.values():v.sort(key=lambda x:int(x['gt_idx']))
h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));pp=build_postprocessor(h['postprocess'],False);out=[]
try:
 for x in records:
  sid=int(x['sample_idx']);alpha=int(x['alpha']);item=selected[sid];assert sha_file(item['path'])==item['sha256'] and sha_file(x['path'])==x['sha256']
  payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False);frozen=payload['batch'];p4=full[sid,alpha];assert sha_file(p4['path'])==p4['sha256']
  with np.load(x['path'],allow_pickle=False) as old,np.load(p4['path'],allow_pickle=False) as orig:
   heads={k:torch.from_numpy(old[k].copy()) for k in ('cls_preds','reg_preds','dir_preds')}
   for key in heads:assert np.allclose(old[key],orig[key],rtol=0,atol=1e-6),key
   with torch.no_grad():got,meta=collect_postprocess(pp,frozen,heads)
   for key in ('score_mask','quality_mask','native_nms_keep','range_mask','ids_final_range','suppressor'):
    assert np.array_equal(got[key],old[key]),key
   for key,tol in [('final_corners',1e-5),('final_scores',1e-6),('raw_corners',1e-5)]:assert np.allclose(got[key],old[key],rtol=0,atol=tol),key
   inp={'ego':{'anchor_box':frozen['ego']['anchor_box'].clone(),'transformation_matrix':frozen['ego']['transformation_matrix'].clone()}}
   geom={k:old[k].copy() for k in ('cls_preds','reg_preds','dir_preds','GT_canonical')}
   independent,_=postprocess(pp,inp,geom,geom)
   for key in ('score_mask','quality_mask','native_nms_keep','range_mask','ids_final_range','suppressor'):assert np.array_equal(independent[key],old[key]),'independent_'+key
   for key,tol in [('final_corners',1e-5),('final_scores',1e-6),('raw_corners',1e-5)]:assert np.allclose(independent[key],old[key],rtol=0,atol=tol),'independent_'+key
   mat=iou_matrix(old['GT_canonical'],got['raw_corners']);attr=attribute_gt(mat,{k:got['ids_'+k] for k in STAGES});assert len(attr)==len(ref[sid])
   for t in (.5,.7):
    _,matched,flags=eval_native(got['final_corners'],got['final_scores'],old['GT_canonical'],t)
    assert np.array_equal(flags,np.array([int(y[f'tp{int(t*100)}_a{alpha}']) for y in ref[sid]]))
    for pi,gi in enumerate(matched):
     if gi>=0:assert int(ref[sid][gi][f'matched_pred_idx{int(t*100)}_a{alpha}'])==pi
  out.append({'sample_idx':sid,'alpha':alpha,'passed':True});print(json.dumps(out[-1]),flush=True)
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*2**30
 assert len(out)==36 and not torch.cuda.is_initialized()
 result={'passed':True,'cases':out,'case_count':len(out),'samples':len(selected),'independent_c1_postprocess_cases':len(out),'real_model_forward':0,'CUDA_initialized':False,'limits':'Existing historical heads only; no P5-B0 interventions or full P5-B0 references.'}
except BaseException as e:
 result={'passed':False,'cases':out,'case_count':len(out),'error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'real_model_forward':0}
 (R/'HISTORICAL_HEAD_REGRESSION.json').write_text(json.dumps(result,indent=2));raise
(R/'HISTORICAL_HEAD_REGRESSION.json').write_text(json.dumps(result,indent=2))
