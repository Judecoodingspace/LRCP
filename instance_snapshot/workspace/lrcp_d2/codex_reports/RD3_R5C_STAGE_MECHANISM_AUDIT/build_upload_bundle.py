"""Package only the new R5C directory. Does not execute another process or model."""
from pathlib import Path
import ast,hashlib,json,zipfile,shutil
ROOT=Path(__file__).resolve().parent
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
REQUIRED=['R5C_CASE_MANIFEST.csv','R5C_GT_STAGE_TRACE.csv','R5C_CANDIDATE_LINEAGE.csv','R5C_GT_TRANSITION_MECHANISMS.csv','R5C_FP_PROVENANCE.csv','R5C_INTERACTION_MECHANISM_MATRIX.csv','R5C_INDEPENDENT_VALIDATION.json','RD3_R5C_SCIENTIFIC_ASSESSMENT.md']
def main():
 v=json.loads((ROOT/'R5C_INDEPENDENT_VALIDATION.json').read_text());assert v['state']=='R5C_PASS' and not v['FAIL'] and v['UNRESOLVED_required_count']==0
 for n in REQUIRED:assert (ROOT/n).is_file() and (ROOT/n).stat().st_size>0,n
 for n,h in v['source_sha256'].items():assert SHA(ROOT/n)==h,'VALIDATED_SOURCE_CHANGED '+n
 seal=json.loads((ROOT/'R5C_SOURCE_AND_EXECUTION_SEAL.json').read_text())
 assert seal['frozen_manifest_sha256']=='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c'
 assert seal['epoch19_checkpoint_sha256']=='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'
 for n,h in seal['current_scripts'].items():assert SHA(ROOT/n)==h,'FINAL_SCRIPT_CHANGED '+n
 for n in ['analyze_r5c.py','stage_common.py','validate_r5c_independent.py','validate_upload_cpu.py','write_report_and_scope.py','build_upload_bundle.py','finish_bundle_validation.py']:
  tree=ast.parse((ROOT/n).read_text(),filename=n)
  for item in ast.walk(tree):
   mods=([a.name for a in item.names] if isinstance(item,ast.Import) else [item.module or ''] if isinstance(item,ast.ImportFrom) else [])
   assert not any(x.split('.')[0] in ['torch','opencood','cupy','pycuda','pynvml'] for x in mods),'FORBIDDEN_EXECUTION_IMPORT '+n
 readiness={'state':'R5C_DELIVERY_READY','required_files':REQUIRED,'independent_checks_passed':v['passed_check_count'],'failed':v['failed_check_count'],'UNRESOLVED_required_candidate_chains':v['UNRESOLVED_required_count'],'final_validated_script_hashes_match':True,'static_model_device_imports':0,'new_forward':0,'CUDA_initializations':0,'historical_files_verified_unchanged':v['historical_files_verified_unchanged'],'compact_evidence_files':len(list((ROOT/'evidence').glob('*.npz'))),'zip_scope':'all files in new R5C directory except bundle, self checksum, outer delivery receipts and post-build smoke-test files'}
 (ROOT/'R5C_DELIVERY_READINESS.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2)+'\n')
 excluded={'R5C_UPLOAD_BUNDLE.zip','SHA256SUMS.txt','R5C_UPLOAD_BUNDLE_SHA256.txt','R5C_UPLOAD_BUNDLE_VALIDATION.json'}
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in excluded and not p.relative_to(ROOT).parts[0].startswith('_bundle_smoke') and p.relative_to(ROOT).as_posix() not in ['logs/bundle_cpu_smoke.log'] and '__pycache__' not in p.parts)
 (ROOT/'SHA256SUMS.txt').write_text(''.join(SHA(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files))
 archive=ROOT/'R5C_UPLOAD_BUNDLE.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in files:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==SHA(p)
  dest=ROOT/'_bundle_smoke';assert not dest.exists(),'EXTRACTION_DESTINATION_EXISTS';z.extractall(dest)
 print(json.dumps({'state':'BUNDLE_BUILT_CRC_AND_MEMBER_SHA_PASS','members':len(files)+1,'zip_bytes':archive.stat().st_size,'sha256':SHA(archive),'extracted_for_independent_smoke_test':str(dest)}))
if __name__=='__main__':main()
