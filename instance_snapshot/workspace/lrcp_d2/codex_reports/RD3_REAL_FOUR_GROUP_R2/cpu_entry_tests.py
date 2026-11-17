"""Separate CPU process: explicitly injected R1 fixture is NEVER imported by
real worker. Exercises same collector/candidate/validator/state/budget/commit.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,importlib.util,traceback,fcntl,subprocess,gc
from evidence_collector import EvidenceCollector
from runner import run_sequence,reserve,event,used_bytes
from candidate_store import complete_candidate
from transfer import transfer_tree,verify_transfer

def rejected(fn,fragment):
 try:fn()
 except Exception as e:require(fragment in str(e),'WRONG_REJECTION '+str(e));return str(e)
 raise AssertionError('EXPECTED_REJECTION '+fragment)
def run():
 # Load fixture explicitly by file, no path tricks and no real backend factory.
 spec=importlib.util.spec_from_file_location('r1_test_fixture',REPAIR/'mock_backend.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 fixture.native.MSDeformAttnFunction=fixture.CPUKernel
 model=fixture.NativeHistoryMock(fixture.checkpoint());backend=EvidenceCollector(model,'cpu','CPU_TEST')
 root=R/'cpu_entry_workflow';root.mkdir();budget=root/'cpu_budget.json';write(budget,{'max_attempts':32,'attempts':[],'backend':'CPU_TEST'},exclusive=True)
 result=run_sequence(backend,root,budget,'CPU_TEST',load(FROZEN)['samples'][:1])
 require(result['committed']==4 and result['real_forwards']==0,'CPU_SEQUENCE')
 # No-mask control uses a separate CPU-only attempt and same native fixture.
 p=backend.prepare(load(FROZEN)['samples'][0]);control_root=R/'cpu_unhooked_control';control_root.mkdir();p.update(run_root=control_root,attempt={'ordinal':0,'kind':'CPU_TEST'},condition='F')
 a,meta=backend.forward_collect(p,'F',install_mask=False)
 with np.load(root/'capture/sample_324_F/arrays.npz',allow_pickle=False) as z:
  for k in a:require(np.array_equal(a[k],z[k]),'FULL_NO_MASK_DIFFERENCE '+k)
 del a,p;gc.collect()
 # Failed independent control cannot commit or call G-B, consumes its reservation.
 fail=R/'cpu_bad_control';fail.mkdir();b=fail/'cpu_budget.json';write(b,{'max_attempts':32,'attempts':[],'backend':'CPU_TEST'})
 rejected(lambda:run_sequence(backend,fail,b,'CPU_TEST',load(FROZEN)['samples'][:1],inject_fault='scientific_control'),'INDEPENDENT_CANDIDATE_REJECTED')
 state=load(fail/'manifest.json');require(state['reserved']==state['entered']==state['returned']==1 and state['committed']==0,'CONTROL_FAILURE_MUST_STOP')
 require(not list((fail/'capture').glob('*')),'FAILED_CONTROL_COMMITTED')
 # Reserve-before-call failure consumes attempt; no automatic retry.
 crash=R/'cpu_reserved_crash';crash.mkdir();b=crash/'cpu_budget.json';write(b,{'max_attempts':32,'attempts':[],'backend':'CPU_TEST'})
 rejected(lambda:run_sequence(backend,crash,b,'CPU_TEST',load(FROZEN)['samples'][:1],inject_fault='reserved_crash'),'TEST_RESERVED_CRASH')
 st=load(crash/'manifest.json');require(st['reserved']==1 and st['entered']==st['returned']==st['committed']==0,'RESERVATION_DURABILITY')
 rejected(lambda:reserve(b,324,'F','CPU_TEST'),'NO_AUTOMATIC_RETRY')
 races=R/'cpu_budget_race';races.mkdir();path=races/'budget.json';write(path,{'max_attempts':32,'attempts':[]})
 # Individual reservations are fsync+flock; max=32 and the 33rd rejects.
 for i in range(32):reserve(path,i,'F','CPU_TEST')
 rejected(lambda:reserve(path,32,'F','CPU_TEST'),'R2_FORWARD_BUDGET_EXHAUSTED')
 lock=root/'test.lock'
 with lock.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  p=subprocess.run(['/workspace/venvs/lrcp/bin/python','-B','-c',"import fcntl,sys; f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(lock)],capture_output=True,text=True)
  require(p.returncode!=0 and 'BlockingIOError' in p.stderr,'CROSS_PROCESS_LOCK')
 from device_guard import validate_telemetry,UUID
 good=f'0, {UUID}, PPU-ZW810E, 98304 MiB, 2 MiB, 0 %\n'
 for gpu,proc,disk,rss in [(good.replace(UUID,'wrong'),'',20*2**30,0),(good,f'{UUID}, 99999, other, 10 MiB\n',20*2**30,0),(good,'',0,0),(good,'',20*2**30,9*2**30)]:rejected(lambda:validate_telemetry(gpu,proc,disk,rss),'DEVICE_' if gpu!=good or proc else ('DISK_LOW' if disk==0 else 'RSS_HIGH'))
 require(not torch.cuda.is_initialized(),'CPU_CUDA_INITIALIZED')
 write(R/'CPU_ENTRY_WORKFLOW.json',{'passed':True,'same_formal_collector_and_orchestrator':True,'conditions_committed':4,'Full_unhooked_exact':True,'failed_independent_F_stops_all_later_calls':True,'failed_reserved_attempt_retained':True,'max_32_and_duplicate_refusal':True,'cross_process_lock':True,'device_parser_faults_rejected':True,'CUDA_initialized':False,'real_forwards':0,'explicit_test_fixture_never_used_by_real_worker':True},exclusive=True);print('CPU_ENTRY_WORKFLOW_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'CPU_ENTRY_TEST_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
