"""R3 CPU-only formal collection over ALL frozen samples. No real launcher or
budget writer is present. Native history preparation precedes interface Mock.
"""
import cpu_only
cpu_only.install()
from rd3_common import *
from evidence_collector import EvidenceCollector
import torch,importlib.util,subprocess,gc,fcntl,argparse,traceback
from per_sample_schema import frozen_contract
PYTHON='/workspace/venvs/lrcp/bin/python'

def fixture_backend():
 spec=importlib.util.spec_from_file_location('r3_cpu_fixture',REPAIR/'mock_backend.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 fixture.native.MSDeformAttnFunction=fixture.CPUKernel
 model=fixture.NativeHistoryMock(fixture.checkpoint());return EvidenceCollector(model,'cpu','CPU_TEST'),fixture

def validate_separately(root,candidate,out):
 cmd=[PYTHON,'-B',str(R/'validate_cpu.py'),str(root),'--candidate',str(candidate),'--output',str(out),'--cpu-test']
 with Path(str(out)+'.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=180)
 require(proc.returncode==0,'INDEPENDENT_CPU_CANDIDATE_FAILED '+str(out));return load(out)

def run():
 root=R/'cpu_preflight';root.mkdir()
 with (root/'run.lock').open('x') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  backend,fixture=fixture_backend();rows=load(FROZEN)['samples'];inventory=[];records=[]
  for row in rows:
   cpu_guard();payload,contract=frozen_contract(row);inventory.append({'sample_idx':row['sample_idx'],'scene_id':row['scene_id'],'contract_hash':contract['hash'],'total_leaves':len(contract['leaves']),'Tensor_leaves':sum(v['kind']=='tensor' for v in contract['leaves'].values()),'GT_object_ids':payload['batch']['ego']['object_ids'],'GT_list_length':len(payload['batch']['ego']['object_ids']),'original_paths':list(contract['leaves']),'input_archive_sha256':row['input']['sha256']});del payload
   p=backend.prepare(row);ordinal=len(records)+1;p.update(run_root=root,attempt={'ordinal':ordinal,'kind':'CPU_TEST','sample_idx':row['sample_idx'],'condition':'F'},condition='F')
   with fixture.NativeCallTrace() as trace:a,meta=backend.forward_collect(p,'F')
   require(trace.counts['native_get_batch_pillar_features']==2 and trace.counts['native_PillarVFE']==3 and trace.counts['native_Scatter']==3 and trace.counts['native_history_projection']==2,'ACTUAL_NATIVE_HISTORY_REQUIRED')
   out=root/f'validation_{row["sample_idx"]}_F.json';validation=validate_separately(root,p['candidate_path'],out)
   require(validation['input_leaves']==len(contract['leaves']) and validation['input_contract_hash']==contract['hash'],'PRODUCER_VALIDATOR_SAMPLE_SCHEMA')
   record={'sample_idx':row['sample_idx'],'condition':'F','candidate_path':p['candidate_path'],'candidate_manifest_sha256':sha_file(Path(p['candidate_path'])/'manifest.json'),'validation_path':str(out),'validation_sha256':sha_file(out),'contract_hash':contract['hash'],'before':p['schema_before'],'after':p['schema_after'],'native_calls':dict(trace.counts),'producer_source_sha256':sha_file(R/'evidence_collector.py'),'validator_source_sha256':sha_file(R/'validate_cpu.py')};records.append(record)
   write(root/'progress.json',{'kind':'CPU_TEST_ONLY','records':records,'real_forward_increment':0})
   print('ALL_SAMPLE_CPU_PASS',row['sample_idx'],'leaves',validation['input_leaves'],'GT_ids',inventory[-1]['GT_list_length'],flush=True)
   del a,meta,p;gc.collect()
  require(len(records)==len(rows),'ALL_FROZEN_SAMPLES_REQUIRED')
  # Exercise unchanged mask under final collector/validator on three extra
  # conditions, separately from the all8 F schema preflight.
  masked=[]
  for condition in ('G-B','G-C','E'):
   row=rows[0];p=backend.prepare(row);ordinal=len(records)+len(masked)+1;p.update(run_root=root,attempt={'ordinal':ordinal,'kind':'CPU_TEST','sample_idx':row['sample_idx'],'condition':condition},condition=condition)
   a,meta=backend.forward_collect(p,condition);out=root/f'validation_{row["sample_idx"]}_{condition}.json';result=validate_separately(root,p['candidate_path'],out)
   masked.append({'sample_idx':row['sample_idx'],'condition':condition,'candidate_path':p['candidate_path'],'validation_path':str(out),'validation_sha256':sha_file(out),'passed':result['passed']});del a,meta,p;gc.collect()
  # Actual Full control without the mask hook vs installed Full; compare every
  # saved numeric array, including full attention and head/postprocess outputs.
  p=backend.prepare(rows[0]);p.update(run_root=root,attempt={'ordinal':12,'kind':'CPU_TEST','sample_idx':324,'condition':'F'},condition='F')
  a,meta=backend.forward_collect(p,'F',install_mask=False)
  out=root/'validation_324_F_unhooked.json';validate_separately(root,p['candidate_path'],out)
  with np.load(Path(records[0]['candidate_path'])/'arrays.npz',allow_pickle=False) as original:
   require(set(original.files)==set(a),'FULL_CONTROL_ARRAY_SET')
   for key in original.files:require(np.array_equal(original[key],a[key]),'FULL_CONTROL_DIFFERENCE '+key)
  write(R/'R4_FULL_UNHOOKED_CONTROL.json',{'passed':True,'all_numeric_arrays_exactly_equal':True,'numeric_array_count':len(a),'hooked_candidate':records[0]['candidate_path'],'unhooked_candidate':p['candidate_path'],'independent_validation_path':str(out),'CUDA_initialized':False,'new_real_forward':0},exclusive=True)
  del a,meta,p;gc.collect()
  write(R/'PER_SAMPLE_LEAF_CONTRACT_VALIDATION.json',{'passed':True,'derived_from_actual_frozen_archives_and_R1_contract':True,'no_hardcoded_leaf_count_table':True,'samples':inventory,'frozen_sha256':sha_file(FROZEN),'checkpoint_sha256':CH,'real_forward_increment':0,'CUDA_initialized':torch.cuda.is_initialized()},exclusive=True)
  write(R/'R4_ALL_8_CPU_PREFLIGHT.json',{'passed':True,'state':'ALL_8_CPU_PREFLIGHT_PASSED','scope':'actual frozen data + native history components; CPU response/head Mock, no real inference','samples':records,'additional_mask_conditions':masked,'all_final_source_samples_tested':True,'mode':'CPU_PREFLIGHT','source_hashes':{str(p):sha_file(p) for p in sorted(R.glob('*.py'))},'real_forward_increment':0,'CUDA_initialized':torch.cuda.is_initialized()},exclusive=True)
  write(R/'PRODUCER_VALIDATOR_REGRESSION.json',{'passed':True,'samples_validated_in_separate_processes':len(records),'all_original_leaf_counts_derived_per_sample':True,'GT_lengths_tested':sorted({x['GT_list_length'] for x in inventory}),'conditions':masked,'producer_sha256':sha_file(R/'evidence_collector.py'),'validator_sha256':sha_file(R/'validate_cpu.py'),'strict_OTHER_INPUT_MUTATION_retained':True,'real_forward_increment':0,'CUDA_initialized':False},exclusive=True)
  print('ALL_8_FINAL_COLLECTOR_AND_VALIDATOR_CPU_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'CPU_PREFLIGHT_FAILURE.json',{'passed':False,'state':'RD3_R3_CPU_REJECTED','error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
