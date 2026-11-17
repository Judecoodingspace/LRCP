"""Final dependency/checkpoint audit, no model forward, then immutable CPU READY."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,subprocess,ast,importlib.metadata
from backend import Backend,configuration
from opencood.tools import train_utils
sys.path.insert(2,str(C3))
from device_guard import validate_telemetry,UUID

def run():
 final=load(R/'mock_runs/final_source_entry/validation.json')
 require(final['passed'] and final['validated_records']==4,'FINAL_CLI_SCHEMA_GATE')
 attempts=load(R/'mock_runs/final_source_entry/budget.json')['attempts']
 require(len(attempts)==4 and all(x['status']=='MODEL_RETURNED' for x in attempts),'FINAL_FORWARD_ACCOUNTING_GATE')
 v=load(R/'INTERVENTION_VALIDATION.json');require(v['passed'] and v['CUDA_initialized']==False,'CPU_TEST_GATE')
 require(load(R/'CPU_DATA_AUDIT.json')['passed'],'DATA_AUDIT_GATE')
 require(load(R/'FROZEN_MANIFEST.json')['sufficient'],'ELIGIBILITY_GATE')
 historical_lock=load(REPORTS/'P5_A_CPU_FIX_20261028_storage/P5_READY.json')['locked_files']
 historical_sources={p:h for p,h in historical_lock.items() if p.startswith(str(SRC)+'/') and (p.endswith('.py') or p.endswith('config.yaml') or p==str(CKPT))}
 for p,hash_ in historical_sources.items():require(sha_file(p)==hash_,'HISTORICAL_NATIVE_SOURCE_CHANGED '+p)
 h=configuration();model=train_utils.create_model(h);state=torch.load(CKPT,map_location='cpu',weights_only=False);model.load_state_dict(state,strict=True);model.eval()
 par=digest(dict(model.named_parameters()));buf=digest(dict(model.named_buffers()));old=load(C2/'CHECKPOINT_IDENTITY.json')
 require(par==old['parameter_digest'] and buf==old['buffer_digest'] and sha_file(CKPT)==CH,'CHECKPOINT_MODEL_IDENTITY')
 # Check actual native configuration: no batch-statistics normalization in eval,
 # four cross blocks followed by two self blocks, single head is separate output.
 from backend import modules
 require(len(modules(model))==4 and not any(m.training for m in model.modules()),'EVAL_FUSION_PROTOCOL')
 require(all(m.track_running_stats for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)),'BATCH_COUPLING_PATH')
 require(h['model']['args']['base_bev_backbone']['resnet'] and h['model']['args']['use_seperate_head'],'FUSION_ARCHITECTURE')
 # Faults in physical UUID, other process, disk, RSS and low device memory.
 good=f'0, {UUID}, PPU-ZW810E, 98304 MiB, 2 MiB, 0 %\n';faults=[]
 for name,g,p,d,r in [('wrong_uuid',good.replace(UUID,'wrong'),'',20*2**30,0),('busy',good,f'{UUID}, 999999, other, 10 MiB\n',20*2**30,0),('disk',good,'',0,0),('rss',good,'',20*2**30,9*2**30),('device_memory',good.replace('2 MiB','98000 MiB'),'',20*2**30,0)]:
  try:validate_telemetry(g,p,d,r)
  except RuntimeError as e:faults.append({'fault':name,'rejected':str(e)})
  else:raise AssertionError('DEVICE_GUARD_FAULT_NOT_REJECTED')
 # Lock complete Python source trees of LRCP and all reused report components,
 # plus extension and runtime package files actually imported by this process.
 paths=set(SRC.rglob('*.py'))
 for dirname in ('P5_B0_C2_20261031_cpu','P5_B0_C3_20261031_cpu','P4_C2_20261027_233036_301795','P4_C1_20261027_162059_722622','P4_PILOT_RETRY_20261023_234457_384594','P4_CPU_VALIDATION_FIX_20261025_111314_916690'):
  paths.update((REPORTS/dirname).glob('*.py'))
 paths.update(R.glob('*.py'));paths.update([MODEL/'config.yaml',C2/'CHECKPOINT_IDENTITY.json',FULL/'INPUT_MANIFEST.json',R/'CPU_DATA_AUDIT.json',R/'INTERVENTION_VALIDATION.json',R/'SCIENTIFIC_FAULT_INJECTION.json',R/'FROZEN_MANIFEST.json'])
 imported={}
 for name,module in list(sys.modules.items()):
  file=getattr(module,'__file__',None)
  if file and Path(file).is_file() and Path(file).suffix in ('.so','.py'):
   imported[name]=str(Path(file).resolve());paths.add(Path(file).resolve())
 hashes={str(p):sha_file(p) for p in sorted(paths)}
 for p in R.glob('*.py'):ast.parse(p.read_text())
 cpu_guard()
 write(R/'DEPENDENCY_AUDIT.json',{'passed':True,'torch_version':torch.__version__,'numpy_version':np.__version__,'python':sys.version,'strict_checkpoint_load':True,'parameter_digest':par,'buffer_digest':buf,'imported_modules':imported,'historical_native_source_files_verified':len(historical_sources),'locked_file_count':len(hashes),'device_guard_faults':faults,'CUDA_initialized':False,'real_model_forwards':0},exclusive=True)
 write(R/'RUN_BUDGET.json',{'schema':'RD3_BUDGET_V1','max_attempts':32,'attempts':[],'checkpoint_sha256':CH,'frozen_sha256':sha_file(R/'FROZEN_MANIFEST.json'),'mock_only':False,'failure_attempts_count':True,'automatic_retry':False},exclusive=True)
 write(R/'REAL_RUN_MANIFEST.json',{'schema':'RD3_RUN_V1','backend':'real','state':'NOT_STARTED','records':[],'real_forwards':0},exclusive=True)
 hashes[str(R/'DEPENDENCY_AUDIT.json')]=sha_file(R/'DEPENDENCY_AUDIT.json')
 write(R/'CPU_READY.json',{'passed':True,'frozen_sha256':sha_file(R/'FROZEN_MANIFEST.json'),'locked_files':hashes,'checkpoint_sha256':CH,'real_model_forwards':0,'CUDA_initialized':False},exclusive=True)
 print('CPU_READY',len(hashes),'locked files; real forward=0',flush=True)
if __name__=='__main__':run()
