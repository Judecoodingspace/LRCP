"""Pure stdlib launch/state unit tests: never import torch or spawn a worker."""
import ast,sys,json,tempfile,copy
from pathlib import Path
import launch_control as c
r=c.R;results=[]
def check(n,b):assert b,n;results.append(n)
for p in r.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec');check('AST '+p.name,True)
check('CPU_entry_zero_model_call_sites',sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model' for n in ast.walk(ast.parse((r/'prepare_cpu.py').read_text())))==0)
p=(r/'ppu_runner.py').read_text();check('PPU_one_model_call_site',sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model' for n in ast.walk(ast.parse(p)))==1)
check('counter_persisted_before_model',p.index("STATE['forward_attempted']+=1")<p.index("outputs=model(batch['ego'])"))
check('40_guard',"STATE['forward_attempted']>=40" in p)
check('manual_handshake','MANUAL_LAUNCH_HANDSHAKE_REQUIRED' in p)
check('cleanup_inside_wall_clock_cap','seconds-5' in (r/'launch_control.py').read_text() and 'deadline-time.monotonic()-.25' in (r/'launch_control.py').read_text())
check('no_automatic_launch_in_preparation','launch(' not in (r/'prepare_cpu.py').read_text())
original_ready=c.verify_ready;original_identity=c.identity;original_popen=c.subprocess.Popen
with tempfile.TemporaryDirectory(prefix='launch_guard_unit_',dir=r) as temp:
 c.R=Path(temp);c.verify_ready=lambda **kw:{'passed':True}
 c.subprocess.Popen=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('PROCESS_SPAWN_FORBIDDEN_IN_TEST'))
 check('dry_run_does_not_spawn',c.launch(dry=True)['started'] is False)
 check('READY_NOT_STARTED',c.status()['state']=='READY_NOT_STARTED')
 try:c.worker(None);raise AssertionError('direct worker accepted')
 except RuntimeError:check('direct_worker_rejected_without_state_mutation',not (c.R/'WORKER_FINAL.json').exists() and not (c.R/'WORKER_STARTED.json').exists())
 expected={'pid':123,'start_ticks':456,'boot_id':'example','state':'S','cmdline':str(c.R)+'/launch_control.py worker'}
 c.identity=lambda pid:dict(expected)
 check('identity_match',c.alive(expected))
 bad=dict(expected,start_ticks=455);check('PID_reuse_rejected',not c.alive(bad))
 c.identity=lambda pid:dict(expected,state='Z');check('zombie_not_running',not c.alive(expected));c.identity=lambda pid:dict(expected)
 c.put(c.R/'LAUNCH.json',{'worker':expected})
 check('RUNNING',c.status()['state']=='RUNNING')
 try:c.launch();raise AssertionError('duplicate accepted')
 except RuntimeError as e:check('duplicate_rejected_before_device_import','DUPLICATE' in str(e))
 c.put(c.R/'PPU_EXECUTION.json',{'passed':True,'full_forward_completed':40,'samples':list(range(18))})
 check('COMPLETED_PENDING_VALIDATION',c.status()['state']=='COMPLETED_PENDING_VALIDATION')
 for state in ['SMOKE_FAILED','FAILED']:
  c.put(c.R/'WORKER_FINAL.json',{'state':state});check(state,c.status()['state']==state)
 (c.R/'PPU_FINAL_ARTIFACTS.sha256').write_text('unit-test-only\n')
 c.put(c.R/'WORKER_FINAL.json',{'state':'VALIDATED','identity':expected,'final_artifact_manifest_sha256':c.sha(c.R/'PPU_FINAL_ARTIFACTS.sha256')})
 check('success_marker_alone_rejected',c.status()['state']!='VALIDATED')
 c.identity=lambda pid:None
 check('dead_worker_success_marker_without_evidence_rejected',c.status()['state']=='FAILED')
 c.identity=lambda pid:dict(expected)
 c.put(c.R/'ACCEPTANCE.json',{'passed':True});c.put(c.R/'INDEPENDENT_VALIDATION.json',{'passed':True,'protected_hashes_unchanged':True});c.put(c.R/'EXIT_CODE.json',{'exit_code':0,'state':'VALIDATED'})
 check('worker_must_really_exit_before_VALIDATED',c.status()['state']=='COMPLETED_PENDING_VALIDATION')
 c.identity=lambda pid:None
 check('VALIDATED_requires_acceptance_and_counters',c.status()['state']=='VALIDATED')
 c.put(c.R/'PPU_EXECUTION.json',{'passed':True,'full_forward_completed':39,'samples':list(range(18))})
 check('wrong_forward_count_rejected',c.status()['state']=='FAILED')
c.R=r;c.verify_ready=original_ready;c.identity=original_identity;c.subprocess.Popen=original_popen
check('torch_not_imported','torch' not in sys.modules)
out={'passed':True,'checks_count':len(results),'checks':results,'process_spawned':False,'torch_imported':False,'CUDA_initialized':False,'model_forwards':0}
c.put(r/'LAUNCH_GUARD_TESTS.json',out);print(json.dumps(out))
