"""Restore a prepared LRCP archive without overwriting differing files.
Default is dry-run. Uses only Python standard library, never imports a model.
"""
from pathlib import Path,PurePosixPath
import argparse,tarfile,hashlib,shutil,os

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--target-root',type=Path,default=Path('/workspace'));p.add_argument('--apply',action='store_true');a=p.parse_args();root=a.target_root.resolve();members=[]
 with tarfile.open(a.archive,'r:*') as tf:
  for m in tf.getmembers():
   q=PurePosixPath(m.name)
   if not m.isfile() or q.is_absolute() or '..' in q.parts:raise RuntimeError('Unsafe member '+m.name)
   if q.parts[0]=='workspace':dst=root.joinpath(*q.parts[1:])
   elif q.parts[0]=='handoff':dst=root.joinpath('_lrcp_handoff',*q.parts[1:])
   else:raise RuntimeError('Unexpected archive prefix '+m.name)
   if not dst.resolve().is_relative_to(root):raise RuntimeError('Target escapes restore root')
   if dst.exists():
    if not dst.is_file():raise RuntimeError('Existing non-file '+str(dst))
    h=hashlib.sha256(tf.extractfile(m).read()).hexdigest();old=hashlib.sha256(dst.read_bytes()).hexdigest()
    if h!=old:raise RuntimeError('Refusing to overwrite differing file '+str(dst))
   members.append((m,dst))
  if a.apply:
   for m,dst in members:
    if dst.exists():continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as src,dst.open('xb') as out:shutil.copyfileobj(src,out)
    os.chmod(dst,m.mode&0o777)
 print(('RESTORED' if a.apply else 'DRY_RUN_OK'),len(members),'files; no differing file overwritten; no model executed')
if __name__=='__main__':main()
