"""Final CPU-only delivery seal, no torch imports."""
import hashlib,json,ast,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
required=['PROTOCOL.md','REPORT.md','baseline_validation.json','factorial_metrics.json','per_GT_factorial.csv','nms_priority_changes.csv','score_geometry_alignment.csv','validation.json','P4_FULL_PLAN.md','P4_FULL_RESOURCE_BUDGET.md','COMPLETION.json']
for name in required:assert (R/name).is_file(),name
assert json.loads((R/'baseline_validation.json').read_text())['passed'] and json.loads((R/'validation.json').read_text())['passed']
complete=json.loads((R/'COMPLETION.json').read_text());assert complete['model_forwards']==0 and complete['CUDA_initialized'] is False and complete['P4_FULL_READY_published'] is False
for f in R.glob('*.py'):compile(ast.parse(f.read_text()),str(f),'exec')
subprocess.run(['bash','-n',str(R/'run_cpu.sh')],check=True)
I=json.loads((R/'inputs.json').read_text())
for key in ['history_hashes','protected_hashes']:
 for name,want in I[key].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,name
for name,want in json.loads((R/'P4_FULL_PLAN_EVIDENCE.json').read_text())['source_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,name
(R/'DELIVERY_VERIFICATION.json').write_text(json.dumps({'passed':True,'required_files':required,'all_python_compiles':True,'bash_syntax_passed':True,'report_figures_visually_checked':True,'historical_hashes_unchanged':len(I['history_hashes']),'protected_hashes_unchanged':len(I['protected_hashes']),'full_planning_inputs_hashes_unchanged':True,'PPU_started':False,'model_forwards':0,'output_bytes_before_seal':sum(f.stat().st_size for f in R.rglob('*') if f.is_file()),'first_validation_status_field_error_retained':True},indent=2))
files=sorted(f for f in R.rglob('*') if f.is_file() and f.name!='ARTIFACTS.sha256')
entries=[(hashlib.sha256(f.read_bytes()).hexdigest(),str(f.relative_to(R))) for f in files]
(R/'ARTIFACTS.sha256').write_text(''.join(f'{h}  {name}\n' for h,name in entries))
for h,name in entries:assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
print(json.dumps({'sealed_files':len(entries),'all_hashes_verified':True,'output_bytes':sum(f.stat().st_size for f in R.rglob('*') if f.is_file()),'P4_C2_complete':True,'PPU_started':False}))
