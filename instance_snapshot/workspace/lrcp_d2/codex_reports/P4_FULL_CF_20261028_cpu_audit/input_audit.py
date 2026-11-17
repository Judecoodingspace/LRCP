import os,json,hashlib,time,resource,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
R=Path(__file__).resolve().parent
F=R.parent/'P4_FULL_PREP_20261028_113714_782940'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
try:
 accept=json.loads((F/'RESEARCH_ACCEPTED.json').read_text());cap=json.loads((F/'PPU_CAPTURE_COMPLETE.json').read_text());val=json.loads((F/'CPU_VALIDATION_COMPLETE.json').read_text())
 assert accept['passed'] and cap['passed'] and val['passed']
 assert sha(F/'PPU_CAPTURE_COMPLETE.json')==accept['PPU_capture_sha256'] and sha(F/'CPU_VALIDATION_COMPLETE.json')==accept['CPU_validation_sha256']
 manifest=json.loads((F/'INPUT_MANIFEST.json').read_text());inputs={x['sample_idx']:x for x in manifest['samples']}; assert len(inputs)==910
 seen=set();start=time.monotonic();counts={'conditions':0,'inputs':0,'pilot':0}
 for sh in range(29):
  p=F/'capture'/f'shard_{sh:03d}'/'COLLECTION_MANIFEST.json';assert sha(p)==cap['shard_manifest_hashes'][str(sh)]
  m=json.loads(p.read_text());assert m['passed']
  for q in m['records']:
   key=(int(q['sample_idx']),int(q['alpha']));assert key not in seen;seen.add(key)
   assert sha(q['path'])==q['sha256'] and sha(q['meta_path'])==q['meta_sha256']
   with np.load(q['path'],allow_pickle=False) as z:
    for k in ('cls_preds','reg_preds','dir_preds','raw_corners','scores_all','GT_canonical','native_nms_keep','final_corners','final_scores'):
     assert k in z,(key,k)
    assert z['cls_preds'].size==12800 and z['reg_preds'].size==12800*7 and z['dir_preds'].size==12800*2
    assert z['raw_corners'].shape==(12800,8,3) and z['scores_all'].shape==(12800,)
    assert all(z[k].dtype==np.float32 for k in ['cls_preds','reg_preds','dir_preds','raw_corners','scores_all'])
   counts['conditions']+=1;counts['pilot']+=q['provenance']!='new'
  print('shard',sh,'conditions',counts['conditions'],flush=True)
 assert seen=={(s,a) for s in inputs for a in (0,1)}
 for s,row in inputs.items():
  assert sha(row['path'])==row['sha256'];counts['inputs']+=1
 result={'passed':True,**counts,'shards':29,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'note':'Anchor definitions and transforms are stored in frozen batches; their values were validated by P4-Full CPU acceptance.','model_forwards':0}
 (R/'INPUT_AUDIT.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
except BaseException as e:
 (R/'INPUT_AUDIT_FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
