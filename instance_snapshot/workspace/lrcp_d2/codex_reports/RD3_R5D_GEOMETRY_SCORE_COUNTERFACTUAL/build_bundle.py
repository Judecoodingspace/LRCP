"""R5D packaging/read-only history verification. No device or model operations."""
from pathlib import Path
import json,csv,hashlib,ast,zipfile
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(n):return json.loads((ROOT/n).read_text())
def rows(n):return list(csv.DictReader((ROOT/n).open()))
def main():
 required=['R5D_PAIR_MANIFEST.csv','R5D_FACTORIAL_RESULTS.csv','R5D_FOCAL_GT_OUTCOMES.csv','R5D_CANDIDATE_STAGE_CHANGES.csv','R5D_NMS_SCORE_SWAP.csv','R5D_GEOMETRY_SCORE_INTERACTION.csv','R5D_OFF_TARGET_EFFECTS.csv','R5D_INDEPENDENT_VALIDATION.json','RD3_R5D_SCIENTIFIC_ASSESSMENT.md']
 for n in required:assert (ROOT/n).is_file() and (ROOT/n).stat().st_size>0,n
 v=load('R5D_INDEPENDENT_VALIDATION.json');assert v['state']=='R5D_PASS' and not v['FAIL'] and not v['UNRESOLVED']
 for n,h in v['source_sha256'].items():assert sha(ROOT/n)==h,'VALIDATED_SOURCE_CHANGED '+n
 se=load('R5D_SOURCE_AND_EXECUTION_SEAL.json')
 for n,h in se['scripts'].items():assert sha(ROOT/n)==h,'FINAL_SCRIPT_CHANGED '+n
 for n,h in se['scientific_result_files'].items():assert sha(ROOT/n)==h,'FINAL_RESULT_CHANGED '+n
 assert load('R5D_FULL_GEOMETRY_PACKAGE_VALIDATION.json')['state']=='FULL_GEOMETRY_PACKAGE_PASS'
 # Verify reported focal patterns from actual independently accepted table rows.
 focal=rows('R5D_FOCAL_GT_OUTCOMES.csv');result=load('R5D_RESULT.json');combos=['G0_S0','G0_S1','G1_S0','G1_S1']
 def bits(pair,oid,t):return ''.join('1' if next(r for r in focal if r['run_id']==pair+'_'+c and r['physical_GT_id']==oid and float(r['iou_threshold'])==t)['TP_FN']=='TP' else '0' for c in combos)
 for pair in ['P1','P2','P3']:
  assert result['focal_GT14014_70_bits_00_01_10_11'][pair]==bits(pair,'14014',.7)
  assert result['focal_GT4071_50_bits'][pair]==bits(pair,'4071',.5)
 assert result['focal_GT29118_70_bits']==bits('P4','29118',.7)
 assert len(rows('R5D_FACTORIAL_RESULTS.csv'))==16 and len(rows('R5D_NMS_SCORE_SWAP.csv'))==3
 scripts=list(ROOT.glob('*.py'))+list((ROOT/'source_reference').glob('*.py'))
 for p in scripts:
  tree=ast.parse(p.read_text(),filename=str(p))
  for item in ast.walk(tree):
   modules=[a.name for a in item.names] if isinstance(item,ast.Import) else [item.module or ''] if isinstance(item,ast.ImportFrom) else []
   assert not any(m.split('.')[0] in ['torch','opencood','cupy','pycuda','pynvml'] for m in modules),'FORBIDDEN_EXECUTABLE_IMPORT '+str(p)
 history=load('R5D_HISTORY_READ_ONLY_SEAL.json')['files']
 for p,h in history.items():assert Path(p).is_file() and sha(p)==h,'HISTORY_CHANGED '+p
 receipt={'state':'R5D_FINAL_DELIVERY_READY','source_and_result_seals_match':True,'report_focal_patterns_checked_from_accepted_CSV':True,'all_required_files_present':required,'history_files_unchanged':len(history),'executable_scripts_parsed':len(scripts),'forbidden_imports':0,'full_source_candidate_NPZ_count':len(list((ROOT/'evidence/source').glob('*.npz'))),'run_evidence_NPZ_count':len(list((ROOT/'evidence/runs').glob('*.npz'))),'new_model_forward':0,'GPU_PPU_calls':0,'CUDA_initializations':0,'historical_modifications':0}
 (ROOT/'R5D_FINAL_READ_ONLY_VALIDATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 excluded={'R5D_UPLOAD_BUNDLE.zip','SHA256SUMS.txt','R5D_UPLOAD_BUNDLE_SHA256.txt','R5D_UPLOAD_BUNDLE_VALIDATION.json'}
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in excluded and not p.relative_to(ROOT).parts[0].startswith('_bundle_smoke') and p.relative_to(ROOT).as_posix()!='logs/bundle_smoke_cpu.log' and '__pycache__' not in p.parts)
 (ROOT/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files))
 zpath=ROOT/'R5D_UPLOAD_BUNDLE.zip'
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  for p in files:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
  dest=ROOT/'_bundle_smoke';assert not dest.exists();z.extractall(dest)
 print(json.dumps({'state':'R5D_BUNDLE_BUILT_CRC_AND_MEMBER_SHA_PASS','members':len(files)+1,'bytes':zpath.stat().st_size,'sha256':sha(zpath),'extraction_path':str(dest)}))
if __name__=='__main__':main()
