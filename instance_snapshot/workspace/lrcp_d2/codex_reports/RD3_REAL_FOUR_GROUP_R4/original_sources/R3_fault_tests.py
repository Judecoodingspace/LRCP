"""Executed negative tests for FINAL collector and independent CLI validator.
All mutations affect copied CPU data only; original archives/results stay readonly.
"""
from rd3_common import *
from cpu_pipeline import fixture_backend
from input_integrity import snapshot
import torch,copy,subprocess,gc,traceback
PYTHON='/workspace/venvs/lrcp/bin/python'

def mutate(batch,name):
 e=batch['ego'];h=e['calibrate_data'][1]['lidar_history'][0]
 if name in ('GT_object_id_value','forged_equal_leaf_count_wrong_value'):e['object_ids'][0]=type(e['object_ids'][0])(e['object_ids'][0]+12345)
 elif name=='GT_object_id_delete':e['object_ids'].pop()
 elif name=='GT_object_id_extra':e['object_ids'].append(type(e['object_ids'][0])(99999999))
 elif name=='illegal_original_field':e['unauthorized_input']=torch.tensor(1.)
 elif name=='required_original_field_deleted':e['processed_lidar'].pop('voxel_num_points')
 elif name=='Tensor_value':h['voxel_features'].view(-1)[0]+=1
 elif name=='Tensor_shape':h['voxel_features']=h['voxel_features'][:,:16,:]
 elif name=='Tensor_dtype':h['voxel_features']=h['voxel_features'].double()
 elif name=='container_type':e['calibrate_data']=dict(e['calibrate_data'])
 elif name=='field_order':e['processed_lidar']=type(e['processed_lidar'])(reversed(list(e['processed_lidar'].items())))
 elif name=='GT_list_order':e['object_ids'][0],e['object_ids'][1]=e['object_ids'][1],e['object_ids'][0]
 elif name=='GT_list_type':e['object_ids']=tuple(e['object_ids'])
 elif name=='pairwise_t_matrix':e['pairwise_t_matrix'][0,0,1,0,3]+=.25
 elif name=='time_delay':e['calibrate_data'][1]['time_delay'][0]+=1
 elif name=='Tensor_device_meta':h['voxel_features']=h['voxel_features'].to('meta')
 else:raise AssertionError('UNKNOWN_MUTATION '+name)

def run():
 require(load(R/'ALL_8_CPU_PREFLIGHT.json')['passed'],'ALL_8_FIRST')
 backend,fixture=fixture_backend();row=load(FROZEN)['samples'][1];root=R/'faults';root.mkdir();records=[]
 source_hashes={str(R/name):sha_file(R/name) for name in ('evidence_collector.py','validate_cpu.py','per_sample_schema.py','input_integrity.py')}
 names=['GT_object_id_value','GT_object_id_delete','GT_object_id_extra','illegal_original_field','required_original_field_deleted','Tensor_value','Tensor_shape','Tensor_dtype','container_type','field_order','GT_list_order','GT_list_type','pairwise_t_matrix','time_delay','Tensor_device_meta','forged_equal_leaf_count_wrong_value']
 for i,name in enumerate(names,1):
  p=backend.prepare(row);original_count=len(p['leaf_before']['leaves']);mutate(p['batch'],name)
  # Deliberately forge the producer's self-snapshot as well. The new normative
  # per-sample path/value check must still reject against the real frozen file.
  p['leaf_before']=snapshot(p['batch']);case=root/name;case.mkdir();torch.save(p['batch'],case/'injected_inputs.pt')
  p.update(run_root=case,attempt={'ordinal':i,'kind':'CPU_TEST','sample_idx':row['sample_idx'],'condition':'F'},condition='F')
  error=None
  try:backend.forward_collect(p,'F')
  except Exception as e:error=str(e)
  require(error is not None and 'OTHER_INPUT_MUTATION' in error,'PRODUCER_MUST_REJECT '+name+' '+str(error))
  if name=='forged_equal_leaf_count_wrong_value':require(len(p['leaf_before']['leaves'])==original_count,'FORGED_COUNT_SETUP')
  records.append({'case_id':name,'executed_entry':'evidence_collector.EvidenceCollector.forward_collect','injection_location':'copied sample424 CPU batch before formal call; claimed before-snapshot forged','injected_input_path':str(case/'injected_inputs.pt'),'injected_input_sha256':sha_file(case/'injected_inputs.pt'),'expected_error':'OTHER_INPUT_MUTATION','actual_error':error,'correctly_rejected':True,'source_hashes':source_hashes,'diagnostic_path':p.get('diagnostic_path')});print('PRODUCER_FAULT_REJECTED',name,flush=True);del p;gc.collect()
 # Both identity/offset cases execute original history preparation with final
 # collector. Sample324 is used because its native B/C outputs differ.
 for name,flag,expected in [('B_C_history_exchange','swap_histories','HISTORY_AGENT_INPUT_MISMATCH'),('B_C_offset_exchange','swap_offsets','METAFLOW_OUTPUT_SLOT_IDENTITY')]:
  row=load(FROZEN)['samples'][0];p=backend.prepare(row);case=root/name;case.mkdir();backend.model.fault=flag;p.update(run_root=case,attempt={'ordinal':1,'kind':'CPU_TEST','sample_idx':row['sample_idx'],'condition':'F'},condition='F');error=None
  try:backend.forward_collect(p,'F')
  except Exception as e:error=str(e)
  require(error is not None and expected in error,'IDENTITY_FAULT_NOT_REJECTED '+name)
  records.append({'case_id':name,'executed_entry':'evidence_collector.EvidenceCollector.forward_collect','injection_location':'CPU model history argument/assembled predicted-offset slots','expected_error':expected,'actual_error':error,'correctly_rejected':True,'source_hashes':source_hashes,'diagnostic_path':p.get('diagnostic_path')});print('PRODUCER_FAULT_REJECTED',name,flush=True)
  backend.model.fault=None;del p;gc.collect()
 # On-disk, completely rehashed forged records test the ACTUAL independent CLI.
 base=next(x for x in load(R/'ALL_8_CPU_PREFLIGHT.json')['samples'] if x['sample_idx']==424);source=Path(base['candidate_path'])
 for name,mutation in [('file_hash_regenerated_manifest','GT_object_id_value'),('producer_passed_true_forgery','Tensor_value')]:
  case=root/name;candidate=case/'diagnostics/candidate_001';candidate.mkdir(parents=True)
  for file in ('raw_arrays.npz','arrays.npz'):os.link(source/file,candidate/file)
  after=torch.load(source/'inputs_after.pt',map_location='cpu',weights_only=False);mutate(after,mutation);before=copy.deepcopy(after);before['ego']['label_dict'].pop('offset')
  meta=load(source/'meta.json');raw=load(source/'raw_meta.json')
  for m in (meta,raw):m['input_leaf_before']=snapshot(before);m['input_leaf_after']=snapshot(after);m['passed']=True
  torch.save(after,candidate/'inputs_after.pt');write(candidate/'meta.json',meta);write(candidate/'raw_meta.json',raw)
  write(candidate/'manifest.json',{'state':'UNACCEPTED','stage':'COLLECTED','science_accepted':False,'files':{file:sha_file(candidate/file) for file in ('raw_arrays.npz','raw_meta.json','inputs_after.pt','arrays.npz','meta.json')}})
  write(case/'external_manifest.json',{'candidate_manifest_sha256':sha_file(candidate/'manifest.json'),'passed':True})
  out=case/'independent_result.json';cmd=[PYTHON,'-B',str(R/'validate_cpu.py'),str(case),'--candidate',str(candidate),'--output',str(out),'--cpu-test']
  with (case/'independent.log').open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=180)
  result=load(out);require(proc.returncode!=0 and not result['passed'] and ('ORIGINAL_SCALAR' in result['error'] or 'ORIGINAL_TENSOR_VALUES_METADATA' in result['error']),'FORGED_RECORD_NOT_INDEPENDENTLY_REJECTED')
  records.append({'case_id':name,'executed_entry':'validate_cpu.py CLI -> validate_candidate','injection_location':str(candidate),'expected_error':'independent original scalar/Tensor value mismatch despite new file hashes and passed=true','actual_error':result['error'],'correctly_rejected':True,'source_hashes':source_hashes,'candidate_path':str(candidate),'candidate_manifest_sha256':sha_file(candidate/'manifest.json'),'all_file_hashes_regenerated':True,'producer_passed_forged_true':True,'independent_result_path':str(out),'independent_result_sha256':sha_file(out)});print('VALIDATOR_FORGERY_REJECTED',name,flush=True)
 cpu_guard();write(R/'FAULT_INJECTION_RESULTS.json',{'passed':True,'executed_cases':records,'count':len(records),'same_final_collector_validator_sources':source_hashes,'new_real_forward':0,'CUDA_initialized':torch.cuda.is_initialized()},exclusive=True);print('R3_ALL_FAULT_CASES_EXECUTED_AND_REJECTED',len(records),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'FAULT_FAILURE.json',{'passed':False,'state':'RD3_R3_CPU_REJECTED','error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
