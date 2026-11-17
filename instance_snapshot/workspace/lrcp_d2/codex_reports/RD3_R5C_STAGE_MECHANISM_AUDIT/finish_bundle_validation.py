"""Read-only ZIP verification plus receipts in R5C; no model/device operations."""
from pathlib import Path
import json,zipfile,hashlib,shutil
ROOT=Path(__file__).resolve().parent
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 archive=ROOT/'R5C_UPLOAD_BUNDLE.zip';smoke=json.loads((ROOT/'logs/bundle_cpu_smoke.log').read_text().strip().splitlines()[-1]);assert smoke['state']=='UPLOAD_EVIDENCE_REPLAY_PASS' and smoke['records']==32
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;lines=z.read('SHA256SUMS.txt').decode().splitlines()
  for line in lines:
   h,n=line.split('  ',1);assert hashlib.sha256(z.read(n)).hexdigest()==h and SHA(ROOT/n)==h,'PACKAGE_CHANGED '+n
  assert set(z.namelist())=={line.split('  ',1)[1] for line in lines}|{'SHA256SUMS.txt'}
 h=SHA(archive);v={'state':'R5C_UPLOAD_BUNDLE_VALIDATED','archive_path':str(archive),'archive_sha256':h,'archive_bytes':archive.stat().st_size,'CRC':'PASS','member_SHA256_checks':len(lines),'source_files_still_match_archive':True,'portable_replay_after_extraction_with_different_directory':smoke,'new_forward':0,'CUDA_initializations':0,'outside_archive_receipts':['R5C_UPLOAD_BUNDLE_SHA256.txt','R5C_UPLOAD_BUNDLE_VALIDATION.json','logs/bundle_cpu_smoke.log']}
 (ROOT/'R5C_UPLOAD_BUNDLE_SHA256.txt').write_text(h+'  R5C_UPLOAD_BUNDLE.zip\n');(ROOT/'R5C_UPLOAD_BUNDLE_VALIDATION.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 extracted=ROOT/'_bundle_smoke';assert extracted.parent==ROOT and extracted.name=='_bundle_smoke'
 if extracted.exists():shutil.rmtree(extracted)
 print(json.dumps(v,ensure_ascii=False))
if __name__=='__main__':main()
