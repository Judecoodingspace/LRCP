"""Check actual ZIP and independent-directory portable CPU replay; write receipts."""
from pathlib import Path
import json,hashlib,zipfile,shutil
ROOT=Path(__file__).resolve().parent
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 portable=json.loads((ROOT/'logs/bundle_smoke_cpu.log').read_text().strip().splitlines()[-1]);assert portable['state']=='R5D_UPLOAD_REPLAY_PASS' and portable['full12800_postprocessing_runs_replayed']==19
 archive=ROOT/'R5D_UPLOAD_BUNDLE.zip'
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;lines=z.read('SHA256SUMS.txt').decode().splitlines()
  for line in lines:
   h,n=line.split('  ',1);assert hashlib.sha256(z.read(n)).hexdigest()==h and SHA(ROOT/n)==h,'SEALED_MEMBER_CHANGED '+n
  assert set(z.namelist())=={x.split('  ',1)[1] for x in lines}|{'SHA256SUMS.txt'}
 h=SHA(archive);r={'state':'R5D_UPLOAD_BUNDLE_VALIDATED','archive_path':str(archive),'sha256':h,'bytes':archive.stat().st_size,'CRC':'PASS','member_SHA256_checks':len(lines),'server_files_match_ZIP':True,'different_directory_full_CPU_replay':portable,'new_model_forward':0,'CUDA_initializations':0,'outside_archive_receipts':['R5D_UPLOAD_BUNDLE_SHA256.txt','R5D_UPLOAD_BUNDLE_VALIDATION.json','logs/bundle_smoke_cpu.log']}
 (ROOT/'R5D_UPLOAD_BUNDLE_SHA256.txt').write_text(h+'  R5D_UPLOAD_BUNDLE.zip\n');(ROOT/'R5D_UPLOAD_BUNDLE_VALIDATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 temp=ROOT/'_bundle_smoke';assert temp.parent==ROOT and temp.name=='_bundle_smoke'
 if temp.exists():shutil.rmtree(temp)
 print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
