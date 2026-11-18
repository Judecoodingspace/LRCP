"""Verify all transport SHA first, then restore exact dataset layout.
No pickle loading, data regeneration, model imports or destructive overwrites.
"""
from pathlib import Path,PurePosixPath
import argparse,json,hashlib,tarfile,zipfile,shutil,stat
ROOT=Path(__file__).resolve().parent
def digest_stream(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sha(p):
 with p.open('rb') as f:return digest_stream(f)
def relative(n):
 p=PurePosixPath(n)
 if p.is_absolute() or '..' in p.parts:raise RuntimeError('Invalid archive path '+n)
 return p

def copy(src,dst,expected=None):
 if dst.is_symlink():raise RuntimeError('Refusing symlink destination '+str(dst))
 if dst.exists():
  h=digest_stream(src)
  if sha(dst)!=h or expected and h!=expected:raise RuntimeError('Existing different file '+str(dst))
  return
 dst.parent.mkdir(parents=True,exist_ok=True)
 with dst.open('xb') as out:
  h=hashlib.sha256()
  for b in iter(lambda:src.read(8*1024*1024),b''):out.write(b);h.update(b)
 if expected and h.hexdigest()!=expected:raise RuntimeError('Extracted member integrity failure '+str(dst))

def main():
 p=argparse.ArgumentParser();p.add_argument('--uploads',required=True,type=Path);p.add_argument('--metadata',required=True,type=Path);p.add_argument('--workspace',default=Path('/workspace'),type=Path);p.add_argument('--metadata-only',action='store_true');p.add_argument('--apply',action='store_true');a=p.parse_args();manifest=json.loads((ROOT/'ASSET_MANIFEST.json').read_text())
 required=[] if a.metadata_only else manifest['raw_dataset_archives']+[manifest['offset_map_archive']]
 for r in required:
  f=a.uploads/r['name']
  if not f.is_file() or f.stat().st_size!=r['bytes'] or sha(f)!=r['sha256']:raise RuntimeError('Missing/wrong transport asset '+str(f))
  print('TRANSPORT_SHA_PASS',r['name'],flush=True)
 if sha(a.metadata)!=manifest['dataset_metadata']['sha256']:raise RuntimeError('Dataset metadata sidecar SHA mismatch')
 # Validate all member paths before any extraction.
 for r in required:
  f=a.uploads/r['name']
  if f.suffix=='.zip':
   with zipfile.ZipFile(f) as z:
    for x in z.infolist():
     q=relative(x.filename)
     if q.parts and q.parts[0] not in ('sweeps','lidarseg','maps','v2.0'):raise RuntimeError('Unexpected ZIP top directory')
     if stat.S_ISLNK(x.external_attr>>16):raise RuntimeError('ZIP symlink member')
  else:
   with tarfile.open(f) as tf:
    for x in tf:
     q=relative(x.name)
     if q.parts[0]!='offset_maps' or not (x.isfile() or x.isdir()):raise RuntimeError('Unexpected offset archive member')
 with tarfile.open(a.metadata) as tf:
  entries=json.load(tf.extractfile('package_manifest.json'))
  for r in entries:
   q=relative(r['member'])
   if q.parts[:3]!=('workspace','datasets','v2xsim2_info_generated'):raise RuntimeError('Unexpected metadata path')
 if a.apply:
  data=a.workspace/'datasets/V2X-Sim-2.0'
  for r in required:
   f=a.uploads/r['name']
   if f.suffix=='.zip':
    with zipfile.ZipFile(f) as z:
     for x in z.infolist():
      if x.is_dir():continue
      dst=data.joinpath(*relative(x.filename).parts)
      if not dst.resolve().is_relative_to(data.resolve()):raise RuntimeError('Destination escapes dataset')
      with z.open(x) as src:copy(src,dst)
   else:
    with tarfile.open(f) as tf:
     for x in tf:
      if not x.isfile():continue
      dst=data.joinpath(*relative(x.name).parts)
      if not dst.resolve().is_relative_to(data.resolve()):raise RuntimeError('Destination escapes dataset')
      with tf.extractfile(x) as src:copy(src,dst)
   print('RESTORED',r['name'],flush=True)
  with tarfile.open(a.metadata) as tf:
   for r in entries:
    dst=a.workspace.joinpath(*relative(r['member']).parts[1:])
    if not dst.resolve().is_relative_to(a.workspace.resolve()):raise RuntimeError('Metadata destination escapes workspace')
    with tf.extractfile(r['member']) as src:copy(src,dst,r['sha256'])
 print(json.dumps({'state':'DATASET_RESTORED_PENDING_DATA_AUDIT' if a.apply else 'DATASET_TRANSPORT_AND_LAYOUT_DRY_RUN_PASS','metadata_only':a.metadata_only,'new_forward':0,'CUDA_initializations':0,'full_dataset_semantics_revalidated':False}))
if __name__=='__main__':main()
