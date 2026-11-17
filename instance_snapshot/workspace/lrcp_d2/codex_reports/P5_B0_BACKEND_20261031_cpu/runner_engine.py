"""Shared orchestration for P5-B0 backends; no CUDA or LRCP model import.

Backend contract is explicit; it is not evidence that a real backend exists.
"""
import csv,io,json,os,fcntl,hashlib
from pathlib import Path
import numpy as np
from cpu_components import reserve,commit,atomic_json,sha
from intervention import ReferenceIntervention

CONTROL=((590,'off'),(590,'pass'),(918,'pass'),(918,'off'))

def read_ids(csv_path):
 rows=list(csv.DictReader(Path(csv_path).open()));ids=[int(x['sample_idx']) for x in rows]
 if len(ids)!=18 or len(set(ids))!=18 or not {590,918}<=set(ids):raise RuntimeError('SAMPLE_MANIFEST_INVALID')
 return ids

def close_enough(a,b):
 fields=[('cls_preds',1e-6),('reg_preds',1e-5),('dir_preds',1e-5),('pred_offset',1e-5),('final_corners',1e-5),('final_scores',1e-6),('native_nms_keep',0),('gt_tp50',0),('gt_tp70',0),('prediction_gt_idx50',0),('prediction_gt_idx70',0),('GT_canonical',1e-5),('score_mask',0),('quality_mask',0),('range_mask',0),('ids_score_pass',0),('ids_quality_pass',0),('ids_nms_top1000',0),('ids_nms_keep',0),('ids_final_range',0)]
 fields += [(f'cross_{i}_{key}',1e-6) for i in range(4) for key in ('reference','offset','location','weights')]
 for key,tol in fields:
  if key not in a or key not in b:raise RuntimeError('MISSING_CONTROL_FIELD '+key)
  x=np.asarray(a[key]);y=np.asarray(b[key])
  if x.shape!=y.shape or not (np.array_equal(x,y) if tol==0 else np.allclose(x,y,atol=tol,rtol=0,equal_nan=False)):raise RuntimeError('CONTROL_MISMATCH '+key)
 if 'IoU_GT_raw' not in a or 'IoU_GT_raw' not in b or a['IoU_GT_raw'].shape!=b['IoU_GT_raw'].shape:raise RuntimeError('CONTROL_IOU_MISSING')
 for t in (.5,.7):
  if not np.array_equal(a['IoU_GT_raw']>=t,b['IoU_GT_raw']>=t):raise RuntimeError('CONTROL_IOU_THRESHOLD')

def run(root,samples_csv,backend):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);ids=read_ids(samples_csv)
 # Exclusive persistent claim: crash never turns into an automatic retry.
 with (root/'run_claim.json').open('x') as f:
  json.dump({'pid':os.getpid(),'kind':backend.kind,'sample_count':18},f);f.flush();os.fsync(f.fileno())
 atomic_json(root/'state.json',{'state':'RUNNING','kind':backend.kind,'attempted':0,'committed':0})
 result={};records=[]
 def fault(stage):
  if hasattr(backend,'fault'):backend.fault(stage)
 def execute(sid,kind):
  prepared=backend.prepare(sid,kind) # validate hashes and build inputs before attempt
  n=reserve(root,sid,kind) # persists before every forward
  fault('after_reserve')
  atomic_json(root/'state.json',{'state':'RUNNING','kind':backend.kind,'attempted':n,'committed':len(records),'sample_idx':sid,'condition':kind})
  modules=prepared['modules'];hooks=None
  try:
   if kind in ('pass','core'):
    hooks=ReferenceIntervention(modules,prepared['valid_remote'],prepared['effective_offsets'],mode='passthrough' if kind=='pass' else 'intervene',height=prepared['height'],width=prepared['width'])
    prepared['intervention']=hooks
    with hooks:out=backend.forward(prepared,sid,kind)
    hooks.validate()
   else:out=backend.forward(prepared,sid,kind)
   fault('after_forward')
   arrays,meta=backend.collect(prepared,out,sid,kind,hooks)
  finally:
   backend.cleanup(prepared)
  if kind in ('off','pass'):close_enough(arrays,backend.baseline(sid))
  if any(np.asarray(v).dtype.hasobject for v in arrays.values()):raise RuntimeError('OBJECT_ARRAY_FORBIDDEN')
  buf=io.BytesIO();np.savez_compressed(buf,**arrays)
  # commit() has a numeric identifier; unique deterministic IDs encode condition.
  code={'off':0,'pass':1,'core':2}[kind]; shard_id=sid*3+code
  p=commit(root,shard_id,buf.getvalue(),{'sample_idx':sid,'condition':kind,'backend':backend.kind,**meta},fault=fault)
  fault('after_commit')
  records.append({'sample_idx':sid,'condition':kind,'path':str(p),'manifest_sha256':sha(p/'manifest.json')})
  return arrays
 try:
  for sid,kind in CONTROL:
   result[(sid,kind)]=execute(sid,kind)
  for sid in (590,918):close_enough(result[sid,'off'],result[sid,'pass'])
  atomic_json(root/'control_gate.json',{'passed':True,'kind':backend.kind})
  result.clear()
  for sid in ids:execute(sid,'core')
  if len(records)!=22:raise RuntimeError('INCOMPLETE')
  atomic_json(root/'manifest.json',{'kind':backend.kind,'records':records})
  atomic_json(root/'state.json',{'state':'MOCK_COMPLETE' if backend.kind=='mock' else 'CAPTURE_COMPLETE','kind':backend.kind,'attempted':22,'committed':22})
 except BaseException as e:
  atomic_json(root/'state.json',{'state':'FAILED','kind':backend.kind,'reason':str(e),'committed':len(records)})
  raise
 return records
