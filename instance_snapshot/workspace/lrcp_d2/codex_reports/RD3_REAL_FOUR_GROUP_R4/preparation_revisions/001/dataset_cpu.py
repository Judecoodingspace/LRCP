"""Independent CPU science registry. Always recomputes actual saved outputs.
No model construction or forward. Original source states are never rewritten.
"""
import cpu_only
cpu_only.install()
from rd3_common import *
from validate_cpu import validate_candidate
from dataset_cpu_shared import compare_condition_identity
import argparse,traceback

def revalidate_five():
 records=[];root=R2/'real_run'
 for condition in CONDITIONS:
  capture=root/'capture'/f'sample_324_{condition}';m=load(capture/'manifest.json')
  require(m['state']=='ACCEPTED' and m['backend']=='REAL','R2_ACCEPTANCE_IDENTITY')
  for file,h in m['files'].items():require(sha_file(capture/file)==h,'R2_SCIENTIFIC_FILE_HASH')
  require(sha_file(m['validation_path'])==m['validation_sha256'],'R2_ORIGINAL_VALIDATION_HASH')
  candidate=root/m['candidate'];require(sha_file(candidate/'manifest.json')==m['candidate_manifest_sha256'],'R2_CANDIDATE_IDENTITY')
  result=validate_candidate(root,candidate)
  for file,h in m['files'].items():require(sha_file(candidate/file)==h,'R2_ACCEPTED_CANDIDATE_BYTES_DIFFER')
  records.append({'sample_idx':324,'condition':condition,'source_stage':'R2_ACCEPTED','data_path':str(capture),'candidate_path':str(candidate),'source_manifest_path':str(capture/'manifest.json'),'source_manifest_sha256':sha_file(capture/'manifest.json'),'source_file_hashes':m['files'],'validation':result})
 recovered=R3/'recovered_candidates/sample_424_F';rm=load(recovered/'RECOVERY_MANIFEST.json');candidate=Path(rm['source_path'])
 require(candidate==root/'diagnostics/candidate_005' and rm['source_attempt_id']==5 and rm['source_R2_state_still']=='UNACCEPTED','R3_RECOVERY_SOURCE_IDENTITY')
 require(sha_file(rm['R3_validation_path'])==rm['R3_validation_sha256'],'R3_RECOVERY_VALIDATION_HASH')
 require(sha_file(R3/'validate_cpu.py')==rm['validator_source_sha256'],'R3_RECOVERY_VALIDATOR_SOURCE')
 require(sha_file(candidate/'manifest.json')==rm['original_manifest_sha256']==sha_file(recovered/'original_manifest.json'),'R3_SOURCE_MANIFEST_HASH')
 for file,h in rm['files_verified_copied'].items():
  require(sha_file(candidate/file)==sha_file(recovered/file)==h,'R3_RECOVERY_ACTUAL_BYTES')
  require(not (recovered/file).stat().st_mode&0o222,'R3_RECOVERY_COPY_NOT_READ_ONLY')
 original=load(candidate/'manifest.json');a=load(R2/'R2_RUN_BUDGET.json')['attempts'][4]
 require(original['state']=='UNACCEPTED' and a['status']=='MODEL_RETURNED' and a['entered_epoch'] is not None and a['returned_epoch'] is not None and a['committed_epoch'] is None,'R2_ATTEMPT5_PERMANENT_STATE')
 result=validate_candidate(root,candidate)
 records.append({'sample_idx':424,'condition':'F','source_stage':'R3_RECOVERY_REFERENCE_FROM_R2_UNACCEPTED','data_path':str(recovered),'candidate_path':str(candidate),'source_manifest_path':str(recovered/'RECOVERY_MANIFEST.json'),'source_manifest_sha256':sha_file(recovered/'RECOVERY_MANIFEST.json'),'source_file_hashes':rm['files_verified_copied'],'original_source_state':'UNACCEPTED','original_attempt_committed':0,'validation':result})
 for rec in records:
  if rec['sample_idx']==324 and rec['condition']!='F':compare_condition_identity(Path(records[0]['data_path']),Path(rec['data_path']))
 return {'passed':True,'records':records,'record_count':5,'independently_recomputed_actual_inputs_attention_postprocess_GT_and_baselines':True,'R2_committed_unchanged':4,'new_real_forward':0,'CUDA_initialized':torch.cuda.is_initialized()}

def prepare_index(result):
 items=[{k:v for k,v in rec.items() if k!='validation'} for rec in result['records']]
 path=R/'reused_records/index.json';write(path,{'schema':'RD3_R4_READ_ONLY_SCIENTIFIC_REFERENCES_V1','record_count':5,'records':items,'source_R2_states_unchanged':True},exclusive=True);path.chmod(0o444)

def validate_collection():
 five=revalidate_five();index=load(R/'reused_records/index.json')
 require(index['records']==[{k:v for k,v in rec.items() if k!='validation'} for rec in five['records']],'REUSED_INDEX_CHANGED')
 records=five['records'].copy();root=R/'real_run';run=load(root/'manifest.json') if (root/'manifest.json').exists() else {'records':[]}
 for rec in run['records']:
  capture=root/rec['relative_path'];require(sha_file(capture/'manifest.json')==rec['manifest_sha256'],'R4_ACCEPTED_MANIFEST_HASH')
  m=load(capture/'manifest.json');require(m['state']=='ACCEPTED' and m['backend']=='REAL','R4_REAL_ACCEPTED_ONLY')
  for name,h in m['files'].items():require(sha_file(capture/name)==h,'R4_CAPTURE_SHA')
  candidate=root/m['candidate'];require(sha_file(candidate/'manifest.json')==m['candidate_manifest_sha256'],'R4_CAPTURE_CANDIDATE_HASH')
  require(sha_file(m['validation_path'])==m['validation_sha256'],'R4_ORIGINAL_VALIDATION_HASH')
  result=validate_candidate(root,candidate)
  for name,h in m['files'].items():require(sha_file(candidate/name)==h,'R4_CAPTURE_CANDIDATE_BYTES')
  require(result['sample_idx']==rec['sample_idx'] and result['condition']==rec['condition'],'R4_RECORD_IDENTITY')
  records.append({'sample_idx':rec['sample_idx'],'condition':rec['condition'],'source_stage':'R4_NEW_REAL_ACCEPTED','data_path':str(capture),'candidate_path':str(candidate),'validation':result})
 expected={(x['sample_idx'],c) for x in load(FROZEN)['samples'] for c in CONDITIONS};actual=[(r['sample_idx'],r['condition']) for r in records]
 require(len(actual)==len(set(actual)) and set(actual)<=expected,'SCIENTIFIC_SET_DUPLICATE_OR_EXTRA')
 missing=[{'sample_idx':sid,'condition':c} for sid,c in sorted(expected-set(actual))]
 for rec in records:
  if rec['condition']!='F':
   f=next(x for x in records if x['sample_idx']==rec['sample_idx'] and x['condition']=='F');compare_condition_identity(Path(f['data_path']),Path(rec['data_path']))
 complete=not missing and len(records)==32
 if complete:
  b=load(R/'R4_NEW_RUN_BUDGET.json');require(len(b['attempts'])==27 and all(a['entered_epoch'] is not None and a['returned_epoch'] is not None and a['committed_epoch'] is not None for a in b['attempts']),'COMPLETE_27_ENTERED_RETURNED_COMMITTED')
 return {'passed':complete,'state':'R4_REAL_32_INDEPENDENTLY_ACCEPTED' if complete else 'NOT_ESTIMABLE','accepted_record_count':len(records),'missing':missing,'records':records,'new_model_forward':0,'CUDA_initialized':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--revalidate-five',action='store_true');p.add_argument('--prepare-index',action='store_true');p.add_argument('--output',required=True);args=p.parse_args()
 result=revalidate_five() if args.revalidate_five else validate_collection()
 if args.prepare_index:require(args.revalidate_five,'PREPARE_FIVE_ONLY');prepare_index(result)
 cpu_guard();write(args.output,result,exclusive=True);print(result['state'] if 'state' in result else 'FIVE_SAVED_REAL_RECORDS_REVALIDATED',flush=True)
 if not result['passed']:sys.exit(3)
if __name__=='__main__':main()
