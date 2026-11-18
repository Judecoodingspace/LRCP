"""Cross-platform binary split/join/hash with strict integrity checks."""
from pathlib import Path
import argparse,hashlib,json,os
BLOCK=8*1024*1024
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(BLOCK),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
 q=sp.add_parser('hash');q.add_argument('file',type=Path)
 q=sp.add_parser('split');q.add_argument('file',type=Path);q.add_argument('--out',type=Path,required=True);q.add_argument('--chunk-mib',type=int,default=512)
 q=sp.add_parser('join');q.add_argument('manifest',type=Path);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='hash':print(digest(a.file)+'  '+a.file.name);return
 if a.cmd=='split':
  if a.chunk_mib<1:raise ValueError('chunk-mib must be positive')
  a.out.mkdir(parents=True,exist_ok=True);whole=hashlib.sha256();parts=[]
  with a.file.open('rb') as src:
   i=0
   while True:
    first=src.read(min(BLOCK,a.chunk_mib*1024*1024))
    if not first:break
    name=a.file.name+f'.part-{i:05d}';target=a.out/name;tmp=a.out/(name+'.partial');h=hashlib.sha256();n=0
    with tmp.open('xb') as out:
     b=first
     while b:
      out.write(b);h.update(b);whole.update(b);n+=len(b)
      left=a.chunk_mib*1024*1024-n
      if not left:break
      b=src.read(min(BLOCK,left))
    if target.exists():
     if digest(target)!=h.hexdigest():raise RuntimeError('Existing part differs: '+str(target))
     tmp.unlink()
    else:tmp.rename(target)
    parts.append({'name':name,'bytes':n,'sha256':h.hexdigest()});i+=1
  data={'name':a.file.name,'bytes':a.file.stat().st_size,'sha256':whole.hexdigest(),'parts':parts};dest=a.out/(a.file.name+'.parts.json')
  if dest.exists() and json.loads(dest.read_text())!=data:raise RuntimeError('Existing manifest differs')
  dest.write_text(json.dumps(data,indent=2)+'\n');print('SPLIT_OK',len(parts),str(dest));return
 data=json.loads(a.manifest.read_text())
 if a.output.exists():
  if a.output.stat().st_size==data['bytes'] and digest(a.output)==data['sha256']:print('JOIN_ALREADY_VERIFIED',a.output);return
  raise RuntimeError('Refusing to replace an existing different output')
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_name(a.output.name+'.partial');whole=hashlib.sha256();size=0
 with tmp.open('xb') as out:
  for part in data['parts']:
   if Path(part['name']).name!=part['name']:raise RuntimeError('Invalid part path')
   f=a.manifest.parent/part['name'];h=hashlib.sha256();n=0
   with f.open('rb') as src:
    for b in iter(lambda:src.read(BLOCK),b''):h.update(b);whole.update(b);out.write(b);n+=len(b);size+=len(b)
   if n!=part['bytes'] or h.hexdigest()!=part['sha256']:raise RuntimeError('Corrupt/missing part: '+part['name'])
 if size!=data['bytes'] or whole.hexdigest()!=data['sha256']:raise RuntimeError('Whole-file integrity mismatch')
 tmp.rename(a.output);print('JOIN_SHA256_PASS',a.output)
if __name__=='__main__':main()
