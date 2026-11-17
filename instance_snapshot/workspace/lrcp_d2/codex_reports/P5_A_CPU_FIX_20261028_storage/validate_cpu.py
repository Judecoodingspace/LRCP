"""Post-capture independent CPU acceptance. Does not run model or initialize CUDA."""
import os,sys,json,csv,traceback,hashlib,resource,signal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pilot_common import R,BASE,sha_file,iou_matrix
sys.path.insert(0,str(R.parent/'P4_C2_20261027_233036_301795'))
from c1_reused import eval_native
from contracts import replay_native_nms
from opencood.utils import common_utils

def save(p,x):
 tmp=p.with_suffix('.tmp')
 with tmp.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def run():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU_VALIDATION_TIMEOUT')));signal.alarm(3600)
 cap=json.loads((R/'PPU_CAPTURE_COMPLETE.json').read_text());ex=json.loads((R/'PPU_WORKER_EXIT.json').read_text())
 if not cap['passed'] or ex['exit_code']!=0 or cap['attempted']!=40 or cap['completed']!=40:raise RuntimeError('CAPTURE_GATE')
 m=R/'CAPTURE_MANIFEST.json';assert sha_file(m)==cap['manifest_sha256'];records=json.loads(m.read_text())['records'];assert len(records)==36
 gtrows={}
 source_records={}
 for folder in sorted((R.parent/'P4_FULL_PREP_20261028_113714_782940'/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
  for item in json.loads(folder.read_text())['records']:source_records[(int(item['sample_idx']),int(item['alpha']))]=item
 for x in csv.DictReader((R.parent/'P3B_20261020_103959_980233'/'per_gt.csv').open()):gtrows.setdefault(int(x['sample_idx']),[]).append(x)
 for v in gtrows.values():v.sort(key=lambda x:int(x['gt_idx']))
 evidence=[];pairs={}
 for rec in records:
  p=Path(rec['path']);assert p.is_file() and sha_file(p)==rec['sha256'] and sha_file(p.with_suffix('.json'))==rec['meta_sha256']
  sid=int(rec['sample_idx']);alpha=int(rec['alpha']);meta=json.loads(p.with_suffix('.json').read_text());assert meta['attention'] and len(meta['attention']['blocks'])==4
  with np.load(p,allow_pickle=False) as z:
   for b in range(4):
    ref=z[f'cross_{b}_reference'];off=z[f'cross_{b}_offset'];loc=z[f'cross_{b}_location'];weights=z[f'cross_{b}_weights'];shape=z[f'cross_{b}_spatial_shapes'];normal=shape[:,::-1]
    assert np.allclose(loc,ref[:,:,None,:,None,:]+off/normal[None,None,None,:,None,:],rtol=1e-6,atol=1e-7)
    assert np.allclose(weights.sum(axis=(-1,-2)),1,atol=1e-6,rtol=1e-5)
   q=np.flatnonzero(z['score_mask'])[z['quality_mask']];shadow=replay_native_nms(z['nms_input_corners'],z['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
   assert np.array_equal(shadow['keep'],z['native_nms_keep'])
   with np.load(BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz',allow_pickle=False) as old:
    assert np.allclose(z['final_corners'],old['pred_boxes'],atol=1e-5,rtol=0) and np.allclose(z['final_scores'],old['pred_scores'],atol=1e-6,rtol=0) and np.allclose(z['GT_canonical'],old['gt_boxes'],atol=1e-5,rtol=0)
    for t in (.5,.7):assert np.array_equal(iou_matrix(z['GT_canonical'],z['final_corners'])>=t,iou_matrix(old['gt_boxes'],old['pred_boxes'])>=t)
   source=source_records[(sid,alpha)]
   assert sha_file(source['path'])==source['sha256']
   with np.load(source['path'],allow_pickle=False) as original:
    now_raw=iou_matrix(z['GT_canonical'],z['raw_corners']);old_raw=iou_matrix(original['GT_canonical'],original['raw_corners'])
    for t in (.5,.7):assert np.array_equal(now_raw>=t,old_raw>=t)
   for t in (.5,.7):
    _,matched,flags=eval_native(z['final_corners'],z['final_scores'],z['GT_canonical'],t);expected=np.array([int(x[f'tp{int(t*100)}_a{alpha}']) for x in gtrows[sid]]);assert np.array_equal(flags,expected)
    for pi,gi in enumerate(matched):
     if gi>=0:assert int(gtrows[sid][gi][f'matched_pred_idx{int(t*100)}_a{alpha}'])==pi
   pairs.setdefault(sid,{})[alpha]={'first_input':hashlib.sha256(z['cross_0_sampling_offsets_input'].tobytes()).hexdigest(),'first_output':hashlib.sha256(z['cross_0_sampling_offsets_output'].tobytes()).hexdigest()}
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_VALIDATION_RSS_LIMIT')
  evidence.append({'sample_idx':sid,'alpha':alpha,'passed':True})
 assert len(pairs)==18 and all(set(p)=={0,1} for p in pairs.values()) and not torch.cuda.is_initialized()
 differences={str(s):{'first_input_same':p[0]['first_input']==p[1]['first_input'],'first_offset_output_same':p[0]['first_output']==p[1]['first_output']} for s,p in pairs.items()}
 signal.alarm(0)
 out={'passed':True,'cases':36,'sample_count':18,'model_forwards':0,'CUDA_initialized':False,'checks':evidence,'first_cross_alpha_comparison':differences}
 save(R/'CPU_VALIDATION_COMPLETE.json',out);save(R/'RESEARCH_ACCEPTED.json',{'passed':True,'CPU_validation_sha256':sha_file(R/'CPU_VALIDATION_COMPLETE.json'),'PPU_capture_sha256':sha_file(R/'PPU_CAPTURE_COMPLETE.json')})
if __name__=='__main__':
 try:
  if (R/'CPU_VALIDATION_STARTED.json').exists():raise RuntimeError('REFUSE_RETRY')
  save(R/'CPU_VALIDATION_STARTED.json',{'pid':os.getpid()});run()
 except BaseException as e:
  signal.alarm(0)
  save(R/'CPU_VALIDATION_FAILURE.json',{'passed':False,'error':str(e),'traceback':traceback.format_exc()});raise
