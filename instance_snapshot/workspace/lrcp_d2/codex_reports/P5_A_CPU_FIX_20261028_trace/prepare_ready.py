import os,sys,json,csv,ast,hashlib,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;F=R.parent/'P4_FULL_PREP_20261028_113714_782940';P=R.parent/'P5_PRECHECK_20261028_v2'
import p5_control as c
assert 'torch' not in sys.modules
if (R/'P5_READY.json').exists() or (R/'PPU_LAUNCH.json').exists():raise RuntimeError('REFUSE_OVERWRITE_READY')
for x in ['OBSERVER_TEST.json','GUARD_TEST.json','SCHEDULE_TEST.json','LAUNCHER_TEST.json','REGRESSION_TEST.json']:
 assert json.loads((R/x).read_text())['passed'],x
for x in ['SELECTION_EVIDENCE.json','SELECTED_INPUT_SEMANTIC_HASHES.json','HEAD_GT_BASELINE.json','SYNTHETIC_TESTS.json','SOURCE_STATIC.json']:
 assert json.loads((P/x).read_text())['passed'],x
files=['p5_worker.py','p5_control.py','attention_observer.py','validate_cpu.py','pilot_common.py','minimal_collector.py','collector_reference.py','contracts.py','test_observer.py','test_guards.py','test_schedule.py','test_launcher.py','test_multiline.py','test_regression.py','prepare_ready.py']
for n in files:
 raw=(R/n).read_bytes();ast.parse(raw);compile(raw,str(R/n),'exec',dont_inherit=True)
for n in ['launch_ppu.sh','ppu_worker.sh','status_ppu.sh','monitor_ppu.sh','validate_cpu.sh']:
 subprocess.run(['bash','-n',str(R/n)],check=True)
sel=list(csv.DictReader((R/'P5_SAMPLE_PROPOSAL_V2.csv').open()));assert len(sel)==len({x['sample_idx'] for x in sel})==18 and sel[0]['sample_idx']=='590'
full=json.loads((F/'INPUT_MANIFEST.json').read_text());samplemap={str(x['sample_idx']):x for x in full['samples']};samples=[]
for x in sel:
 s=samplemap[x['sample_idx']];assert s['sha256']==x['input_sha256'] and s['input_hash']==x['input_semantic_hash'];samples.append({'sample_idx':int(x['sample_idx']),'path':s['path'],'sha256':s['sha256'],'input_hash':s['input_hash'],'RNG_hash':s['RNG_hash']})
old=json.loads((F/'PPU_READY.json').read_text());protected=dict(old['protected_files']);protected.update(old['locked_files'])
for n in files+['launch_ppu.sh','ppu_worker.sh','status_ppu.sh','monitor_ppu.sh','validate_cpu.sh','P5_SAMPLE_PROPOSAL_V2.csv','P5_FOCAL_ANCHORS.csv','P5_FOCAL_ANCHORS_ALPHA0.csv','OBSERVER_TEST.json','GUARD_TEST.json','SCHEDULE_TEST.json','LAUNCHER_TEST.json','REGRESSION_TEST.json']:protected[str(R/n)]=c.sha(R/n)
for n in ['SELECTION_EVIDENCE.json','SELECTED_INPUT_SEMANTIC_HASHES.json','HEAD_GT_BASELINE.json','SYNTHETIC_TESTS.json','SOURCE_STATIC.json','P5_A_PROTOCOL.md','P5_SAMPLE_PROPOSAL_V2.csv']:protected[str(P/n)]=c.sha(P/n)
for n in ['INPUT_MANIFEST.json','PPU_CAPTURE_COMPLETE.json','CPU_VALIDATION_COMPLETE.json','RESEARCH_ACCEPTED.json']:protected[str(F/n)]=c.sha(F/n)
chosen={x['sample_idx'] for x in samples};cnt=0
for p in sorted((F/'capture').glob('shard_*/COLLECTION_MANIFEST.json')):
 protected[str(p)]=c.sha(p)
 for x in json.loads(p.read_text())['records']:
  if int(x['sample_idx']) in chosen:
   assert c.sha(x['path'])==x['sha256'] and c.sha(x['meta_path'])==x['meta_sha256'];protected[x['path']]=x['sha256'];protected[x['meta_path']]=x['meta_sha256'];cnt+=1
assert cnt==36
# Include exact historical output and the original per-GT labels used at runtime.
for sid in chosen:
 for alpha in (0.,1.):
  p=Path('/workspace/lrcp_d2')/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz';protected[str(p)]=c.sha(p)
protected[str(R.parent/'P3B_20261020_103959_980233'/'per_gt.csv')]=c.sha(R.parent/'P3B_20261020_103959_980233'/'per_gt.csv')
ready={'passed':True,'state':'PPU_READY_NOT_STARTED','run_directory':str(R),'budget':c.BUDGET,'samples':samples,'locked_files':protected,'created_epoch':time.time(),'manual_command':str(R/'launch_ppu.sh')+' --approve-40','monitor_command':str(R/'monitor_ppu.sh'),'prior_attempted_forwards':3,'combined_attempt_limit_if_restarted':43,'prior_failure':str(R.parent/'P5_A_PREP_20261028_cpu'/'PPU_CAPTURE_FAILURE.json'),'limitations':'Fresh 40-attempt budget requires new user approval; prior failed run attempted 3. No real model forward, PPU runtime, live hook output or PPU hook-on/off tests performed during preparation.'}
res=c.verify_ready(True,ready_override=ready);assert res['passed'] and 'torch' not in sys.modules
# Negative checks use in-memory mutated candidates, with no device import.
invalid=dict(ready,state='WRONG')
try:c.verify_ready(False,ready_override=invalid)
except RuntimeError:pass
else:raise AssertionError('bad state accepted')
assert not (R/'PPU_LAUNCH.json').exists() and not (R/'PPU_PROGRESS.json').exists()
(R/'CPU_STATIC_VALIDATION.json').write_text(json.dumps({'passed':True,'source_AST_and_bash':True,'synthetic_observer_and_guard_tests':True,'36_selected_history_conditions_verified':True,'actual_locked_hashes_valid':True,'candidate_dry_run':res,'negative_ready_test':True,'model_forwards':0,'CUDA_initialized':False},indent=2))
ready['locked_files'][str(R/'CPU_STATIC_VALIDATION.json')]=c.sha(R/'CPU_STATIC_VALIDATION.json')
# Final creation only after every static/dry-run gate passed.
tmp=R/'P5_READY.json.tmp'
with tmp.open('x') as f:json.dump(ready,f,indent=2);f.flush();os.fsync(f.fileno())
os.replace(tmp,R/'P5_READY.json')
print(json.dumps({'passed':True,'state':'PPU_READY_NOT_STARTED','samples':18,'locked_files':len(protected),'CUDA_initialized':False,'model_forwards':0}))
