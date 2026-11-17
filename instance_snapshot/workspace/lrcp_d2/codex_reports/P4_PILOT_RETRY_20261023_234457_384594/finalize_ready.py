"""CPU-only acceptance/publish transaction; stops at READY_NOT_STARTED.
Never invokes a worker. launch(dry=True) executes hash checks only.
"""
import ast,json,sys,os,time,subprocess,datetime,difflib
from pathlib import Path
import launch_control as c
R=c.R

def finalize():
 if (R/'PPU_READY.json').exists():raise RuntimeError('REFUSE_READY_REGENERATION')
 for name in ['CPU_VALIDATION.json','REAL_EQUIVALENCE.json','CPU_COLLECTOR_FINAL_CHECKS.json','LAUNCH_GUARD_TESTS.json','BASH_SYNTAX_CHECKS.json']:
  assert c.read(R/name)['passed'],name
 validation=c.read(R/'CPU_VALIDATION.json');prep=c.read(R/'INPUT_PREPARATION.json');state=c.read(R/'STATE.json')
 assert validation['sample_count']==18 and validation['GT_count']==162 and validation['model_forward_count']==0 and validation['CUDA_initialized'] is False
 assert not (R/'LAUNCH.json').exists() and not (R/'PPU_START.json').exists()
 sources=[]
 for p in R.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec');sources.append(str(p))
 for p in R.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
 protected=c.read(R/'PROTECTED_HASHES_BEFORE.json');changed=[p for p,h in protected.items() if c.sha(p)!=h]
 reused=c.read(R/'REUSED_ARTIFACTS_VERIFIED.json');changed_reused=[x['path'] for x in reused if c.sha(x['path'])!=x['sha256']]
 assert not changed and not changed_reused
 status=subprocess.run(['git','-C','/workspace/LRCP','status','--short'],capture_output=True,text=True,check=True).stdout
 assert status=='?? formal_run_logs/\n',repr(status)
 c.put(R/'DIFF_AUDIT.json',{'passed':True,'protected_file_count':len(protected),'reused_artifact_count':len(reused),'changed':changed,'changed_reused':changed_reused,'git_status':status,'scope':'new retry directory; authorized mutable READY entry pointer only'})
 old=R.parent/'P4_PILOT_20261022_193114_346841'
 diff=''.join(difflib.unified_diff((old/'bounded_offset_cache_proposal.py').read_text().splitlines(True),(R/'bounded_offset_cache_proposal.py').read_text().splitlines(True),fromfile='reviewed_proposal.py',tofile='retry_bounded_dataset.py'))
 (R/'MEMORY_FIX.diff').write_text(diff)
 c.put(R/'SCRIPT_STATIC_CHECKS.json',{'passed':True,'AST_compiled_python_files':sources,'Bash_syntax_scripts':[str(p) for p in R.glob('*.sh')],'launch_guard_checks':c.read(R/'LAUNCH_GUARD_TESTS.json')['checks_count'],'collector_CPU_checks':c.read(R/'CPU_COLLECTOR_FINAL_CHECKS.json')['checks_count'],'one_model_call_site_in_manual_worker':True,'CPU_preparation_has_no_model_call_site':True,'PPU_live_smoke_executed':False,'known_runtime_limitations':['PPU framework availability/device conflict check deliberately deferred to user manual launch','native/instrumented PPU equivalence, historical final detections, runtime/VRAM not measured']})
 c.put(R/'CPU_READY.json',{'passed':True,'sample_count':18,'GT_count':162,'CPU_validation_sha256':c.sha(R/'CPU_VALIDATION.json'),'input_preparation_sha256':c.sha(R/'INPUT_PREPARATION.json'),'CPU_retry_count':1,'model_forwards':0,'CUDA_initialized':False})
 resource=c.resources();assert resource['disk_free_bytes']>=25*2**30
 commands={'launch':f'bash {R}/launch_ppu.sh','monitor':f'bash {R}/monitor_ppu.sh','status':f'bash {R}/status_ppu.sh','log':f'tail -f {R}/ppu.log','validation':f'{c.PYTHON} -B {R}/check_results.py'}
 files=list(R.glob('*.py'))+list(R.glob('*.sh'))+[R/n for n in ['CPU_READY.json','CPU_VALIDATION.json','INPUT_PREPARATION.json','RAW_INPUT_HASHES.json','PROTECTED_HASHES_BEFORE.json','SCRIPT_STATIC_CHECKS.json','SAMPLE_SELECTION.csv','SELECTED_SAMPLES_ALL_GT.csv','REAL_EQUIVALENCE.json','MODEL_CPU_SETUP.json','CPU_COLLECTOR_FINAL_CHECKS.json','LAUNCH_GUARD_TESTS.json','BASH_SYNTAX_CHECKS.json']]
 ready={'state':'READY_NOT_STARTED','run_directory':str(R),'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_passed':True,'static_passed':True,'limits':c.LIMITS,'batches':[{k:x[k] for k in ['sample_idx','path','sha256','input_hash','archive_bytes']} for x in prep['samples']],'protected_input_hashes':protected,'preparation_file_hashes':{str(p):c.sha(p) for p in files},'resources':resource,'CPU_summary':{k:validation[k] for k in ['replay_elapsed_seconds','replay_peak_RSS_bytes','cache_archive_bytes','max_shm_used_bytes','delay_common_counts']},'commands':commands,'runtime_limitations':['No CUDA initialization, checkpoint transfer to PPU or model forward in CPU preparation/dry-run','PPU device usability, real smoke equivalence, historical post-NMS reproduction and PPU performance remain untested until the user launches','64 MiB shm remains unchanged; no shared-tensor transfers in CPU preparation','Only one manual launch permitted; failure requires separate authorization for a new retry']}
 c.put(R/'PPU_READY.json',ready)
 # Exercise the exact dry-run branch with a private pointer, before publishing the official entry.
 test_pointer=R/'READY_POINTER_TEST.txt'
 with test_pointer.open('x') as f:f.write(str(R)+'\n')
 official=c.POINTER;c.POINTER=test_pointer
 try:
  dry=c.launch(dry=True)
  assert dry['started'] is False and 'torch' not in sys.modules
 finally:c.POINTER=official
 c.put(R/'READY_PREPUBLICATION_VALIDATION.json',{'passed':True,'dry_run':dry,'torch_imported':False,'CUDA_initialized':False,'worker_started':False,'model_forwards':0})
 state.update(phase='READY_NOT_STARTED',status='READY_NOT_STARTED',updated_epoch=time.time());c.put(R/'STATE.json',state)
 previous=official.read_text() if official.exists() else None
 c.put(R/'READY_POINTER_PREVIOUS.json',{'path':str(official),'previous_value':previous})
 temporary=official.with_name('.P4_PILOT_READY_CURRENT.'+str(os.getpid())+'.tmp')
 with temporary.open('x') as f:f.write(str(R)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,official)
 fd=os.open(str(official.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
 try:
  verified=c.verify_ready(full=True);observed=c.status();assert observed['state']=='READY_NOT_STARTED'
  c.put(R/'READY_VERIFICATION.json',{'passed':True,'verification':verified,'status':observed,'CUDA_initialized':False,'PPU_started':False,'model_forwards':0})
 except BaseException:
  if previous is None:official.unlink()
  else:
   with temporary.open('x') as f:f.write(previous)
   os.replace(temporary,official)
  raise
 print(json.dumps({'passed':True,'state':'READY_NOT_STARTED','commands':commands,'CPU':ready['CPU_summary']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:finalize()
 except BaseException:
  import traceback
  c.put(R/'READY_FAILURE.json',{'passed':False,'traceback':traceback.format_exc(),'PPU_started':False});raise
