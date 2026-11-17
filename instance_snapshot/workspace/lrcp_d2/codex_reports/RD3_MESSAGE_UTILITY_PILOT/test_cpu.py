"""Formal runner Mock, scientific corruption and runtime fault gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,subprocess,copy,traceback,fcntl,multiprocessing
from runner import launch,reserve,gates,PYTHON
from mock_backend import build
from intervention import FixedSlotMask
from validate_cpu import validate,verify_attention,record

def rejected(fn,contains=None):
 try:fn()
 except Exception as e:
  if contains:require(contains in str(e),'WRONG_FAILURE '+str(e))
  return str(e)
 raise AssertionError('FAILURE_NOT_BLOCKED')

def reserve_one(i):
 try:reserve(R/'mock_runs/budget_race',True,i,'F',4);return True
 except RuntimeError:return False

def run():
 checks=[];row=load(R/'FROZEN_MANIFEST.json')['samples'][0];b=build();p=b.prepare(row)
 before=digest(dict(b.model.named_parameters()));original_input=digest(p['batch'])
 with torch.no_grad():native=b.model(p['batch']['ego'])
 masked,meta=b.forward_collect(p,'F')
 for k in ('cls_preds','reg_preds','dir_preds'):require(np.array_equal(array(native[k]),masked[k]),'FULL_NATIVE_NOT_EXACT')
 require(digest(dict(b.model.named_parameters()))==before and digest(p['batch'])==original_input,'MODEL_OR_INPUT_MUTATION')
 checks.append({'test':'F equals unhooked native CPU model','passed':True})
 for cond,slot in [('G-B',1),('G-C',2),('E',1)]:
  with torch.no_grad(),FixedSlotMask(b.modules,CONDITIONS[cond]):base=b.model(p['batch']['ego'])
  saved=[x.clone() for x in b.model.features]
  for x in b.model.features:x[slot]+=1000
  with torch.no_grad(),FixedSlotMask(b.modules,CONDITIONS[cond]):changed=b.model(p['batch']['ego'])
  for k in ('cls_preds','reg_preds','dir_preds'):require(torch.equal(base[k],changed[k]),'MASKED_AGENT_INDIRECT_INFLUENCE')
  b.model.features=saved;checks.append({'test':cond+' hidden features cannot affect output across four native blocks','passed':True})
 checks.append({'test':'Ego all-off mask rejected','error':rejected(lambda:FixedSlotMask(b.modules,[False,False,False]),'EGO_MUST_STAY_VALID'),'passed':True})
 require(not torch.cuda.is_initialized(),'CPU_INITIALIZED_CUDA')
 root=R/'mock_runs/formal_acceptance';launch(root,True,count=8)
 v=validate(root,True);require(v['validated_records']==32,'MOCK_FOUR_GROUP_COMPLETENESS');write(root/'independent_validation.json',v)
 checks.append({'test':'formal runner 8 real frozen batches x four CPU Mock conditions; independent native CPU schema acceptance','passed':True,'records':32,'mock_only':True})
 # All faults go through the same supervisor/worker route and cannot trigger a
 # real constructor. Failures after reservation keep consumed attempts.
 runtime=[]
 for fault in ('disk','rss','output','device','reserved_crash','forward','write','control','after_control'):
  r=R/'mock_runs'/('fault_'+fault);err=rejected(lambda:launch(r,True,fault=fault))
  budget=load(r/'budget.json') if (r/'budget.json').exists() else {'attempts':[]}
  count=len(budget['attempts']);require(count==(0 if fault in ('disk','rss','output','device') else 1),'FAULT_BUDGET')
  if fault not in ('disk','rss','output','device'):rejected(lambda:launch(r,True),'PERSISTENT_RUN_CLAIM')
  runtime.append({'fault':fault,'rejected':True,'reserved':count,'error':err})
 checks.append({'test':'runtime faults reject through formal launch; consumed budgets retained; no reentry','passed':True,'faults':runtime})
 # Rehashing data is insufficient to pass science validation.
 rec=load(root/'manifest.json')['records'][1];path=root/rec['relative_path']
 with np.load(path/'arrays.npz',allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
 meta=load(path/'meta.json');faults=[]
 for name in ('masked_weight','normalization','nonfinite','slot_reference','offset_swap','softmax','sampling_formula'):
  a={k:v.copy() for k,v in original.items()}
  if name=='masked_weight':a['cross_0_full_weights'][0,0,0,3,0]=.01
  if name=='normalization':a['cross_1_full_weights']*=.5
  if name=='nonfinite':a['cross_2_full_weights'][0,0,0,0,0]=np.nan
  if name=='slot_reference':a['reference_full'][:,:,3:6,:]+=.1
  if name=='offset_swap':a['metaflow_slot_1'],a['metaflow_slot_2']=a['metaflow_slot_2'],a['metaflow_slot_1']
  if name=='softmax':a['cross_0_raw_logits_selected'][:,:,0]+=10
  if name=='sampling_formula':a['cross_0_location']+=.01
  faults.append({'fault':name,'error':rejected(lambda:verify_attention(a,meta)),'passed':True})
 write(R/'SCIENTIFIC_FAULT_INJECTION.json',{'passed':True,'faults':faults})
 # Independent real entry checks CPU evidence before even importing backend.
 rejected(lambda:gates()) # CPU_READY intentionally absent until all tests finish.
 checks.append({'test':'real gate refuses before READY and before real backend creation','passed':True})
 race=R/'mock_runs/budget_race';race.mkdir();write(race/'budget.json',{'max_attempts':4,'attempts':[]})
 with multiprocessing.get_context('fork').Pool(8) as pool:results=pool.map(reserve_one,range(20))
 require(sum(results)==4 and len(load(race/'budget.json')['attempts'])==4,'BUDGET_RACE')
 checks.append({'test':'20 concurrent reservations; exactly four allowed with max=4','passed':True})
 write(R/'INTERVENTION_VALIDATION.json',{'passed':True,'checks':checks,'scientific_faults':faults,'real_model_forwards':0,'CUDA_initialized':torch.cuda.is_initialized(),'checkpoint_sha256':sha_file(CKPT),'mock_note':'Mock uses real native encoder/attention Python forward and CPU debug sampler, synthetic features and heads, original frozen GT/native postprocess; never scientific real-model evidence.'},exclusive=True)
 print('ALL_CPU_GATES_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'CPU_TEST_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
