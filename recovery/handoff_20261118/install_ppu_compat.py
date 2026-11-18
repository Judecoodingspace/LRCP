"""Restore custom PPU operators into a matching venv/project; default dry-run.
Never installs torch/SDK or queries/initializes an accelerator.
"""
from pathlib import Path,PurePosixPath
import argparse,json,tarfile,hashlib,sys,platform,importlib.metadata as md,sysconfig,tempfile,shutil,subprocess
EXPECTED='8ee2c18124f933d9b8c3d2e803c4be7ee32be969ac90eb4171660dd6dabb0059'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--workspace',default=Path('/workspace'),type=Path);p.add_argument('--apply',action='store_true');a=p.parse_args()
 if sys.prefix==sys.base_prefix:raise RuntimeError('Run from a dedicated venv, never modify provider system Python')
 if sha(a.archive)!=EXPECTED:raise RuntimeError('PPU sidecar SHA mismatch')
 with tarfile.open(a.archive) as tf:
  req=json.load(tf.extractfile('compatibility_requirements.json'));manifest=json.load(tf.extractfile('package_manifest.json'))
  if list(sys.version_info[:2])!=req['python'] or platform.machine()!=req['machine']:raise RuntimeError('Python/architecture mismatch; rebuild from source')
  for n in ['torch','numpy']:
   if md.version(n)!=req[n]:raise RuntimeError('Provider package version mismatch: '+n)
  sdk=Path('/usr/local/PPU_SDK/release.yaml')
  if not sdk.exists() or req['SDK'] not in sdk.read_text():raise RuntimeError('PPU SDK mismatch or unavailable')
  dist=md.distribution('torch')
  for n,h in req['torch_source_fingerprints'].items():
   if sha(Path(dist.locate_file(n)))!=h:raise RuntimeError('Provider torch fingerprint differs: '+n)
  site=Path(sysconfig.get_path('purelib'));wheel=None;planned=[]
  for r in manifest:
   n=r['member'];q=PurePosixPath(n)
   if q.is_absolute() or '..' in q.parts:raise RuntimeError('Invalid archive path')
   m=tf.getmember(n)
   if not m.isfile():raise RuntimeError('Non-regular archive member')
   h=hashlib.sha256(tf.extractfile(m).read()).hexdigest()
   if h!=r['sha256']:raise RuntimeError('Member SHA mismatch: '+n)
   if q.parts[0]=='workspace':dst=a.workspace.joinpath(*q.parts[1:])
   elif q.parts[0]=='site_packages':dst=site.joinpath(*q.parts[1:])
   elif q.parts[0]=='wheels':wheel=n;continue
   elif n=='compatibility_requirements.json':continue
   else:raise RuntimeError('Unexpected archive member '+n)
   if dst.exists() and sha(dst)!=h:raise RuntimeError('Existing different binary: '+str(dst))
   planned.append((n,dst))
  if not wheel:raise RuntimeError('spconv wheel missing')
  if a.apply:
   with tempfile.TemporaryDirectory(prefix='lrcp-spconv-') as d:
    target=Path(d)/PurePosixPath(wheel).name
    with tf.extractfile(wheel) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--ignore-installed',str(target)],check=True)
   for n,dst in planned:
    if dst.exists():continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(n) as src,dst.open('xb') as out:shutil.copyfileobj(src,out)
 print(json.dumps({'state':'PPU_CUSTOM_BINARIES_RESTORED_PENDING_IMPORT_VALIDATION' if a.apply else 'PPU_CUSTOM_BINARY_RESTORE_DRY_RUN_PASS','project_and_site_binaries':len(planned),'torch_imported':'torch' in sys.modules,'new_forward':0,'CUDA_initializations':0,'ABI_claim':'Same image/ABI still required; version and source checks alone are not operator execution validation.'}))
if __name__=='__main__':main()
