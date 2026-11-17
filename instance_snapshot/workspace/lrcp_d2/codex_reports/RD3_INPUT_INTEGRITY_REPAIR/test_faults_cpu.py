"""Negative cases use formal backend + unchanged native historical components.
All faults are isolated disposable test batches, never the frozen data or params.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_only
cpu_only.install()
from rd3_common import *
import torch,fcntl,subprocess,traceback,gc,copy,importlib.util
from mock_backend import build
from input_integrity import snapshot,check
from attention_validation import verify_attention
from diagnostics import accepted_record
PYTHON='/workspace/venvs/lrcp/bin/python'

def run():
 root=R/'fault_runs_r2';root.mkdir();backend=build();row=load(FROZEN)['samples'][0];tests=[]
 specifications=[('voxel_features','OTHER_INPUT_MUTATION'),('voxel_coords','OTHER_INPUT_MUTATION'),('voxel_num_points','OTHER_INPUT_MUTATION'),('swap_histories','HISTORY_AGENT_INPUT_MISMATCH'),('swap_offsets','METAFLOW_OUTPUT_SLOT_IDENTITY'),('pairwise','OTHER_INPUT_MUTATION'),('unknown_derived','UNEXPECTED_DERIVED_FIELDS'),('shape','OTHER_INPUT_MUTATION'),('dtype','OTHER_INPUT_MUTATION'),('device_meta','OTHER_INPUT_MUTATION'),('contract_hash','INPUT_CONTRACT_HASH_CORRUPTED'),('post_return_values','OTHER_INPUT_MUTATION'),('unexpected_label','OTHER_INPUT_MUTATION'),('invalid_label_offset','LABEL_OFFSET_SEMANTICS')]
 for ordinal,(name,expected) in enumerate(specifications,1):
  cpu_guard();p=backend.prepare(row);case=root/name;case.mkdir();p.update(run_root=case,attempt={'ordinal':ordinal,'sample_idx':324,'condition':'F','kind':'CPU_MOCK_FAULT'},fault=name)
  backend.model.fault=name;handles=[];once=[False]
  def modify(mod,args):
   if once[0]:return
   once[0]=True;h=p['batch']['ego']['calibrate_data'][1]['lidar_history'][0]
   if name in ('voxel_features','voxel_coords','voxel_num_points'):h[name].view(-1)[0]+=1
   elif name=='pairwise':p['batch']['ego']['pairwise_t_matrix'][0,0,1,0,3]+=.125
   elif name=='shape':h['voxel_features']=h['voxel_features'][:,:16,:]
   elif name=='dtype':h['voxel_features']=h['voxel_features'].to(torch.float64)
   elif name=='device_meta':h['voxel_features']=h['voxel_features'].to('meta')
  if name in ('voxel_features','voxel_coords','voxel_num_points','pairwise','shape','dtype','device_meta'):handles.append(backend.model.pillar_vfe.register_forward_pre_hook(modify))
  if name=='contract_hash':p['leaf_before']['hash']='f'*64
  if name in ('post_return_values','invalid_label_offset'):
   def post(mod,args,out):
    if name=='post_return_values':args[0]['calibrate_data'][1]['lidar_history'][0]['voxel_features'].view(-1)[0]+=1
    else:args[0]['label_dict']['offset'].view(-1)[0]+=1
   handles.append(backend.model.register_forward_hook(post))
  caught=None
  try:backend.forward_collect(p,'F')
  except Exception as e:caught=e
  finally:
   for handle in handles:handle.remove()
  require(caught is not None and expected in str(caught),'FAULT_NOT_REJECTED '+name+' '+str(caught))
  require(p.get('diagnostic_path') is not None,'MISSING_DIAGNOSTIC '+name)
  d=load(Path(p['diagnostic_path'])/'diagnostic.json');require(d['state']=='UNACCEPTED' and not d['science_accepted'],'DIAGNOSTIC_ACCEPTANCE')
  require(not (case/'capture').exists(),'DIAGNOSTIC_LEAKED_TO_CAPTURE')
  require(digest(dict(backend.model.named_parameters()))==backend.params and digest(dict(backend.model.named_buffers()))==backend.buffers,'FAULT_MUTATED_PARAMETERS')
  tests.append({'fault':name,'rejected':True,'reason':str(caught),'expected_reason':expected,'diagnostic_relative':str(Path(p['diagnostic_path']).relative_to(R)),'model_returned':d['model_returned']})
  print('FAULT_REJECTED',name,'returned',d['model_returned'],flush=True);del p,d;gc.collect()
 # Saved-evidence fault: keep sum=1 yet make a blocked contribution nonzero.
 main=R/'formal_cpu_r2';a_path=main/'capture/G-B/arrays.npz';meta=load(main/'capture/G-B/meta.json')
 with np.load(a_path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 original={k:v.copy() for k,v in a.items() if k=='cross_0_full_weights'}
 w=a['cross_0_full_weights'];available=np.unravel_index(np.argmax(w[0,0,0,:3]),w[0,0,0,:3].shape);amount=float(w[0,0,0,available[0],available[1]])*.5
 w[0,0,0,available[0],available[1]]-=amount;w[0,0,0,3,0]=amount
 bad=root/'masked_weight.npz'
 with bad.open('xb') as f:np.savez_compressed(f,cross_0_full_weights=w)
 try:verify_attention(a,meta)
 except Exception as e:require('BLOCKED_WEIGHT_NONZERO' in str(e),'WRONG_ATTENTION_FAILURE');tests.append({'fault':'nonzero_blocked_attention','rejected':True,'reason':str(e),'evidence':str(bad.relative_to(R))})
 else:raise AssertionError('BLOCKED_ATTENTION_ACCEPTED')
 # Independently rehashed on-disk input corruption is rejected by leaf matching.
 before=snapshot(row)
 try:check({**before,'hash':'broken'},row)
 except Exception as e:require('HASH_CORRUPTED' in str(e),'HASH_FAULT_REASON')
 else:raise AssertionError('HASH_CORRUPTION_ACCEPTED')
 # CPU lock conflict is tested in a DIFFERENT process. No real constructor.
 lockpath=root/'lock_race.lock'
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  code="import fcntl,sys; f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
  proc=subprocess.run([PYTHON,'-B','-c',code,str(lockpath)],capture_output=True,text=True)
  require(proc.returncode!=0 and 'BlockingIOError' in proc.stderr,'LOCK_CONFLICT_NOT_REJECTED')
 # Reentry uses actual formal CLI and must refuse without new model construction.
 before_ledger=sha_file(main/'ledger.json');cmd=[PYTHON,'-B',str(R/'runner.py'),'--root',str(main)]
 proc=subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=60)
 require(proc.returncode!=0 and 'PERSISTENT_CPU_CLAIM_NO_REENTRY' in proc.stderr and sha_file(main/'ledger.json')==before_ledger,'PERSISTENT_REENTRY')
 # UNACCEPTED diagnostic cannot enter normal accepted-record paths.
 try:accepted_record(root,tests[11]['diagnostic_relative'])
 except Exception as e:require('DIAGNOSTIC_NOT_ACCEPTED_SCIENCE' in str(e),'DIAGNOSTIC_PATH_REJECTION')
 else:raise AssertionError('DIAGNOSTIC_ACCEPTED_AS_SCIENCE')
 from backend import Backend
 try:Backend.real()
 except Exception as e:require(str(e)=='CPU_ONLY_REAL_FORWARD_FORBIDDEN','REAL_ENTRY')
 else:raise AssertionError('REAL_ENTRY_ALLOWED')
 result={'passed':True,'faults':tests,'runtime_tests':{'cross_process_lock_rejected':True,'formal_CLI_reentry_rejected_before_model':True,'diagnostic_rejected_from_science':True,'real_constructor_rejected':True},'real_forward_increment':0,**cpu_only.evidence()}
 write(R/'SCIENTIFIC_FAULT_TESTS.json',result,exclusive=True);print('ALL_FAULT_TESTS_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'FAULT_TEST_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),**cpu_only.evidence()});raise
