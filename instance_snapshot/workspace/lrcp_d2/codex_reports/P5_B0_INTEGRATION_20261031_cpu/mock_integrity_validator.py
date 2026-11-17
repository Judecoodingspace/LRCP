"""Independent mock integrity validation only; NOT a scientific validator."""
import hashlib,json,io
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(root,ids):
 root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text());records=m['records'];budget=json.loads((root/'budget.json').read_text())
 expected={(i,'core') for i in ids}|{(590,'off'),(590,'pass'),(918,'pass'),(918,'off')}
 if m['kind']!='mock' or len(records)!=22 or len(budget['attempts'])!=22:raise RuntimeError('COUNT')
 keys=[(int(x['sample_idx']),x['condition']) for x in records]
 if len(set(keys))!=22 or set(keys)!=expected or {(int(x['sample']),x['kind']) for x in budget['attempts']}!=expected:raise RuntimeError('IDENTITY')
 for z in records:
  p=Path(z['path']).resolve()
  if p.parent!=root/'capture' or not p.is_dir() or digest(p/'manifest.json')!=z['manifest_sha256']:raise RuntimeError('SHARD_MANIFEST')
  info=json.loads((p/'manifest.json').read_text())
  if set(info['files'])!={'payload.bin','meta.json'}:raise RuntimeError('SHARD_FILES')
  for name,h in info['files'].items():
   if digest(p/name)!=h:raise RuntimeError('HASH')
  meta=json.loads((p/'meta.json').read_text())
  if (meta['sample_idx'],meta['condition'])!=(z['sample_idx'],z['condition']) or not meta['mock']:raise RuntimeError('META')
  with np.load(p/'payload.bin',allow_pickle=False) as a:
   if not {'cls_preds','reg_preds','dir_preds','final_corners','final_scores','native_nms_keep','gt_tp50','gt_tp70'}<=set(a.files):raise RuntimeError('MOCK_FIELDS')
 return {'passed':True,'mock_only':True,'scientific_acceptance':False}
