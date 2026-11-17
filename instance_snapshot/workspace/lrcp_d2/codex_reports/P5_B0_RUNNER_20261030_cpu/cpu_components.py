"""CPU-only budget and isolated sample-commit primitives; no device imports."""
import os,json,fcntl,hashlib,shutil,uuid
from pathlib import Path
MAX_ATTEMPTS=22

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path,data):
 p=Path(path);tmp=p.parent/(p.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('x') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
  fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if tmp.exists():tmp.unlink()

def reserve(root,sample,kind):
 root=Path(root)
 if kind not in ('core','off','pass'):raise ValueError('BAD_KIND')
 with (root/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=root/'budget.json'
  old=json.loads(p.read_text()) if p.exists() else {'max':MAX_ATTEMPTS,'attempts':[]}
  if old.get('max')!=MAX_ATTEMPTS or not isinstance(old.get('attempts'),list):raise RuntimeError('BAD_BUDGET')
  attempts=old['attempts'];key={'sample':int(sample),'kind':kind}
  if len(attempts)>=MAX_ATTEMPTS:raise RuntimeError('FORWARD_LIMIT')
  if key in attempts:raise RuntimeError('DUPLICATE_FORWARD')
  attempts.append(key);atomic_json(p,old)
  return len(attempts)

def commit(root,sample,data,metadata):
 root=Path(root);capture=root/'capture';capture.mkdir(exist_ok=True)
 dest=capture/f'sample_{int(sample)}';tmp=capture/('.tmp_'+uuid.uuid4().hex)
 if dest.exists():raise RuntimeError('DUPLICATE_SAMPLE')
 tmp.mkdir()
 try:
  p=tmp/'payload.bin'
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  atomic_json(tmp/'meta.json',metadata)
  atomic_json(tmp/'manifest.json',{'sample':int(sample),'files':{'payload.bin':sha(p),'meta.json':sha(tmp/'meta.json')}})
  if dest.exists():raise RuntimeError('DUPLICATE_SAMPLE')
  os.rename(tmp,dest)
  fd=os.open(capture,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return dest
 finally:
  if tmp.exists():shutil.rmtree(tmp)

def verify_shard(path):
 p=Path(path);m=json.loads((p/'manifest.json').read_text())
 if p.name!=f"sample_{int(m['sample'])}" or set(m['files'])!={'payload.bin','meta.json'}:raise RuntimeError('BAD_MANIFEST')
 for n,h in m['files'].items():
  if not (p/n).is_file() or sha(p/n)!=h:raise RuntimeError('HASH_MISMATCH')
 return True
