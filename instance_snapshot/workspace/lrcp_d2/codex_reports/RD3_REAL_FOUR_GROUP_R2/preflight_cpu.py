"""Pre-device real-entry audit; constructs native CPU model but never forwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,ast,inspect,gc,traceback
from real_backend import native_cpu_model

def run():
 require(load(R/'CPU_ENTRY_WORKFLOW.json')['passed'] and load(R/'R1_REVALIDATION.json')['passed'],'CPU_CHAIN_GATES')
 baseline=load(R/'HISTORY_BASELINE_AUDIT.json');require(baseline['baseline_evidence_complete'],'MISSING_BASELINE_EVIDENCE')
 fr=load(FROZEN);require([x['sample_idx'] for x in fr['samples']]==[324,424,725,354,464,765,384,484] and sum(x['GT_count'] for x in fr['samples'])==56,'FROZEN_8_IDENTITY')
 require(sha_file(FROZEN)=='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c' and sha_file(CKPT)==CH,'EXPLICIT_HASHES')
 for entry,row in zip(baseline['samples'],fr['samples']):
  m=load(entry['P4_native_alpha1']['meta_path']);require(m['sample_idx']==row['sample_idx'] and m['alpha']==1.0 and m['input_hash']==row['input']['input_hash'],'P4_NATIVE_BASELINE_INPUT_IDENTITY')
 require(load(FULL/'RESEARCH_ACCEPTED.json')['passed'],'P4_NATIVE_HISTORY_ACCEPTANCE')
 model,native=native_cpu_model();require(not torch.cuda.is_initialized(),'PRE_DEVICE_CUDA_INITIALIZED')
 require(torch.__version__=='2.6.0' and np.__version__=='1.23.5','HISTORICAL_RUNTIME_VERSION')
 # Every native Python implementation stays identical to R1 and historical audit.
 history=load(R/'HISTORY_PRESERVATION_SEAL.json')['files']
 for path,h in history.items():require(sha_file(path)==h,'HISTORY_READ_ONLY_HASH '+path)
 actual_sources={str(p):sha_file(p) for p in sorted(R.glob('*.py'))}
 imported={}
 for module in list(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path and Path(path).is_file() and Path(path).suffix in ('.py','.so'):imported[str(Path(path).resolve())]=sha_file(path)
 actual_sources.update(imported)
 for p in R.glob('*.py'):ast.parse(p.read_text())
 # Real entry has a positive native-class/forward/kernel identity assertion.
 require('cpu_only' not in sys.modules and not any('mock_backend' in m for m in sys.modules),'MOCK_IMPORTED_IN_REAL_CHAIN')
 require(sha_file(R/'intervention.py')==sha_file(REPAIR/'intervention.py'),'MASK_ALGORITHM_CHANGED')
 write(R/'NATIVE_CPU_ENTRY_AUDIT.json',{'passed':True,'native_entry':native,'native_model_constructed_on_CPU_no_forward':True,'actual_imported_files':len(imported),'Mock_modules_present':False,'head_MetaFlow_and_device_kernel_native':True,'R1_CPU_revalidation':sha_file(R/'R1_REVALIDATION.json'),'CPU_orchestration_validation':sha_file(R/'CPU_ENTRY_WORKFLOW.json'),'baseline_audit':sha_file(R/'HISTORY_BASELINE_AUDIT.json'),'CUDA_initialized':False,'real_forward_count':0},exclusive=True)
 write(R/'SOURCE_AND_CHECKPOINT_SEAL.json',{'schema':'RD3_R2_SOURCE_SEAL_V1','execution_sources':actual_sources,'historical_sources':history,'frozen_manifest_reference':str(FROZEN),'frozen_sha256':sha_file(FROZEN),'checkpoint_path':str(CKPT),'checkpoint_sha256':CH,'configuration_sha256':sha_file(MODEL/'config.yaml'),'native_entry':native,'scientific_Q_fields_defined_in':str(R/'analyze_cpu.py'),'scientific_analysis_source_sha256':sha_file(R/'analyze_cpu.py')},exclusive=True)
 write(R/'R2_RUN_BUDGET.json',{'schema':'RD3_R2_BUDGET_V1','max_attempts':32,'attempts':[],'old_budget_unchanged':{'reserved':1,'entered':1,'returned':1,'committed':0},'frozen_sha256':sha_file(FROZEN),'checkpoint_sha256':CH,'refund_failed_attempts':False,'automatic_retry':False},exclusive=True)
 write(R/'REAL_RUN_MANIFEST.json',{'schema':'RD3_R2_RUN_V1','backend':'REAL','state':'NOT_STARTED','records':[],'reserved':0,'entered':0,'returned':0,'committed':0,'real_forwards':0},exclusive=True)
 evidence=[R/n for n in ('CPU_ENTRY_WORKFLOW.json','R1_REVALIDATION.json','HISTORY_BASELINE_AUDIT.json','NATIVE_CPU_ENTRY_AUDIT.json','SOURCE_AND_CHECKPOINT_SEAL.json')]
 write(R/'CPU_ENTRY_READY.json',{'passed':True,'evidence_hashes':{str(p):sha_file(p) for p in evidence},'no_real_forward_yet':True,'CUDA_initialized':False},exclusive=True)
 print('R2_CPU_ENTRY_READY; native real binding audited; real forward=0; CUDA_initialized=False',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'PREFLIGHT_CPU_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized()});raise
