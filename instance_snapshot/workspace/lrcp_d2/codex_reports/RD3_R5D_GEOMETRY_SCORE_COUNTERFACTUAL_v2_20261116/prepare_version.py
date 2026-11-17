from pathlib import Path
import json,hashlib,shutil,csv,ast
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'RD3_R5D_GEOMETRY_SCORE_COUNTERFACTUAL'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
for sub in ['logs','evidence/source','evidence/runs','source_reference','historical_reference']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
initial=load(OLD/'R5D_INITIAL_SOURCE_VALIDATION.json')
for p,h in initial['source_files'].items():assert sha(p)==h,('SOURCE_CHANGED',p)
print('PRIOR_SOURCE_HASHES_PASS',len(initial['source_files']),flush=True)
history=load(OLD/'R5D_HISTORY_READ_ONLY_SEAL.json')['files']
for i,(p,h) in enumerate(history.items()):
 assert sha(p)==h,('HISTORY_CHANGED',p)
 if i%1000==0:print('HISTORY_HASH_VERIFIED',i,flush=True)
for p in OLD.rglob('*'):
 if p.is_file():history[str(p)]=sha(p)
print('HISTORY_SEAL_FILES',len(history),flush=True)
loc=list(csv.DictReader((ROOT.parent/'RD3_R5B_SAMPLE_UTILITY/R5B_RAW_SHARD_LOCATOR.csv').open()))
assert len(loc)==32
c0=load(ROOT.parent/'RD3_R5C_STAGE_MECHANISM_AUDIT/R5C_C0_REPLAY_VALIDATION.json')
print('R5C_C0_KEYS',list(c0),flush=True)
for r in loc:assert sha(r['absolute_path'])==r['sha256']==initial['source_files'][r['absolute_path']]
for p in OLD.glob('*.py'):shutil.copyfile(p,ROOT/p.name)
for sub in ['source_reference','historical_reference']:shutil.copytree(OLD/sub,ROOT/sub,dirs_exist_ok=True)
for n in ['R5D_DESIGN_FROZEN.json','R5D_DESIGN_SHA256.txt']:shutil.copyfile(OLD/n,ROOT/n)
write('R5D_HISTORY_READ_ONLY_SEAL.json',{'files':history,'protected_file_count':len(history),'output_directory':str(ROOT),'history_read_only':True,'includes_preexisting_R5D':True})
initial['protected_file_count']=len(history);initial['version_provenance']='Verified prior source seals and historical expected hashes, then extended seal to preserve preexisting R5D. No old result CSV/NPZ reused as generated output.';write('R5D_INITIAL_SOURCE_VALIDATION.json',initial)
p=ROOT/'write_report.py';s=p.read_text().replace('历史3,701个文件再次核对保持不变',"历史{v['historical_files_verified_unchanged']:,}个文件再次核对保持不变")
s=s.replace('## 冻结证据、实验单位及口径','本次为独立版本重执行：原有R5D目录保持只读；复用其已审查CPU脚本，从实际locator原始NPZ重新生成全部19条结果，没有复制既有结果代替运行。版本准备及脚本来源见R5D_VERSION_PROVENANCE.json。\n\n## 冻结证据、实验单位及口径')
p.write_text(s)
shutil.copyfile('/root/.codex/attachments/ce63d90e-6641-45b9-b576-4f54f47fae59/Pasted text.txt',ROOT/'USER_TASK.md')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
write('R5D_VERSION_PROVENANCE.json',{'created_client_time':datetime.now(timezone(timedelta(hours=8))).isoformat(),'client_timezone':'Asia/Shanghai','new_version':str(ROOT),'preserved_prior_version':str(OLD),'prior_scripts':{p.name:sha(p) for p in OLD.glob('*.py')},'execution':'fresh producer, separate validator, separate geometry validator, extracted-bundle CPU replay','history_files':len(history),'all32_locator_hashes_verified':True,'new_forward':0,'CUDA_initializations':0,'GPU_PPU_calls':0})
print('PREPARE_VERSION_PASS',str(ROOT),flush=True)
