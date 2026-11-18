"""Restore only the approved minimum and nested text snapshot; no models/devices.
Without --apply, both restore phases are dry-runs and project files stay untouched.
"""
from pathlib import Path
import argparse,hashlib,tarfile,tempfile,subprocess,sys,shutil,json
EXPECTED='26b3d6a4e9ee6cf4feba2d7949e3632eed181db9c631f8daa143c312b53c78a4'
TEXT_EXPECTED='0f9634e0fac3ea862509bc4a07e2bb1872496f71dc00d1804eab4cfea33b9205'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--workspace',type=Path,default=Path('/workspace'));p.add_argument('--apply',action='store_true');a=p.parse_args()
 if sha(a.archive)!=EXPECTED:raise RuntimeError('Minimum archive SHA256 mismatch')
 helper=Path(__file__).resolve().parents[1]/'instance_20261117/restore_archives.py'
 def restore(src):
  args=[sys.executable,'-B',str(helper),str(src),'--target-root',str(a.workspace)]
  if a.apply:args.append('--apply')
  subprocess.run(args,check=True)
 restore(a.archive)
 if a.apply:
  nested=a.workspace/'_lrcp_handoff/PRIVATE_project_text_snapshot.tar.gz'
  if sha(nested)!=TEXT_EXPECTED:raise RuntimeError('Nested text SHA256 mismatch')
  restore(nested)
 else:
  with tempfile.TemporaryDirectory(prefix='lrcp-text-check-') as d:
   nested=Path(d)/'text.tar.gz'
   with tarfile.open(a.archive) as tf,tf.extractfile('handoff/PRIVATE_project_text_snapshot.tar.gz') as src,nested.open('wb') as out:shutil.copyfileobj(src,out)
   if sha(nested)!=TEXT_EXPECTED:raise RuntimeError('Nested text SHA256 mismatch')
   restore(nested)
 print(json.dumps({'state':'MINIMUM_AND_TEXT_RESTORED' if a.apply else 'MINIMUM_AND_TEXT_DRY_RUN_PASS','workspace':str(a.workspace),'dataset_included':False,'full_historical_binary_set_included':False,'new_forward':0,'CUDA_initializations':0}))
if __name__=='__main__':main()
