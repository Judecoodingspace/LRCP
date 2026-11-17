"""Publish READY only after all CPU, reuse, static and CPU-only dry-run gates."""
import os,sys,json,time,ast,hashlib,subprocess,shutil,traceback,stat
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from pilot_common import R,sha_file,write_json
import full_launch_control as ctl

def shell(name,body):
 p=R/name
 if p.exists():raise FileExistsError(p)
 p.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+body+'\n');p.chmod(p.stat().st_mode|stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)

def run():
 if (R/'PPU_READY.json').exists() or (R/'PPU_LAUNCH.json').exists():raise RuntimeError('REFUSE_READY_REPUBLICATION')
 cpu=json.loads((R/'CPU_READY.json').read_text());data=json.loads((R/'CPU_INPUT_VALIDATION.json').read_text());manifest=json.loads((R/'INPUT_MANIFEST.json').read_text());equiv=json.loads((R/'INPUT_EQUIVALENCE.json').read_text());reuse=json.loads((R/'REUSE_18_VALIDATION.json').read_text())
 assert cpu['passed'] and data['passed'] and manifest['passed'] and equiv['passed'] and reuse['passed'] and len(manifest['samples'])==910
 if time.time()-json.loads((R/'CPU_STARTED.json').read_text())['time']>3600:raise RuntimeError('TOTAL_CPU_PREPARATION_TIMEOUT')
 for name in ['REUSE_PRECHECK.json','MINIMAL_CAPTURE_CPU_TEST.json','SCIENTIFIC_VALIDATOR_CPU_TEST.json']:
  if not json.loads((R/name).read_text())['passed']:raise RuntimeError('TEST_FAILED '+name)
 compiled=[]
 for p in sorted(R.glob('*.py')):compile(ast.parse(p.read_text()),str(p),'exec');compiled.append(p.name)
 # Exactly one full model call appears in the future worker; CPU scripts have none.
 code=ast.parse((R/'full_capture_worker.py').read_text());calls=[n for n in ast.walk(code) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
 assert len(calls)==1
 prep_ast=ast.parse((R/'prepare_full_cpu.py').read_text());assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model' for n in ast.walk(prep_ast))
 try:ctl.launch(dry=False,approve=False);raise AssertionError('UNAPPROVED_LAUNCH_NOT_BLOCKED')
 except RuntimeError as e:assert 'MANUAL_TERMINAL' in str(e)
 try:ctl.verify_ready(True,{});raise AssertionError('INVALID_READY_NOT_BLOCKED')
 except RuntimeError as e:assert 'READY_INVALID' in str(e)
 assert not torch.cuda.is_initialized()
 write_json(R/'PRE_STATIC_VALIDATION.json',{'passed':True,'compiled_files':compiled,'future_model_call_sites':1,'CPU_model_call_sites':0,'manual_approval_gate_negative_test_passed':True,'invalid_ready_negative_test_passed':True,'real_saved_head_capture_cases':6,'real_saved_head_scientific_cases':6,'reuse_cases':36,'model_forwards':0,'CUDA_initialized':False})
 # These are created only after CPU data and Python static tests passed.
 base='RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nPYTHON=/workspace/venvs/lrcp/bin/python\nexport PYTHONDONTWRITEBYTECODE=1\n'
 shell('launch_ppu.sh',base+'exec "$PYTHON" -B "$RUN_DIR/full_launch_control.py" launch "$@"')
 shell('ppu_worker.sh',base+'exec "$PYTHON" -B "$RUN_DIR/full_launch_control.py" worker --nonce "$1"')
 shell('status_ppu.sh',base+'exec "$PYTHON" -B "$RUN_DIR/full_launch_control.py" status')
 shell('monitor_ppu.sh',base+'exec "$PYTHON" -B "$RUN_DIR/full_launch_control.py" monitor')
 shell('validate_cpu.sh',base+"export CUDA_VISIBLE_DEVICES=''\nexec \"$PYTHON\" -B \"$RUN_DIR/scientific_validator.py\"")
 for name in ['launch_ppu.sh','ppu_worker.sh','status_ppu.sh','monitor_ppu.sh','validate_cpu.sh']:subprocess.run(['bash','-n',str(R/name)],check=True)
 before=ctl.status();assert before['state']=='CPU_READY'
 write_json(R/'STATIC_VALIDATION.json',{'passed':True,'python_AST_compiled':compiled,'shell_syntax_passed':True,'CPU_saved_head_tests':6,'CPU_scientific_validator_tests':6,'reuse_core_conditions_verified':36,'negative_manual_approval_gate':True,'negative_ready_guard':True,'prepublication_status':'CPU_READY','model_forwards':0,'CUDA_initialized':torch.cuda.is_initialized()})
 preppeak=max(manifest['peak_RSS_bytes'],data['CPU_RSS_peak_bytes']);totalbytes=sum(x['archive_bytes'] for x in manifest['samples'])
 report=f'''# P4-Full CPU准备报告\n\nCPU轨迹、910份冻结归档与独立回读通过；18份Pilot输入逐字段与RNG内容一致，36个Pilot核心条件独立重放通过。所有4个delay依原CPU轨迹重放，只有k=3归档；未执行模型forward，未初始化CUDA。\n\n- 归档数：910；GT：8164；归档字节：{totalbytes}（{totalbytes/2**30:.3f}GiB）；含元数据缓存：{manifest['cache_bytes']}字节。\n- CPU重放峰值RSS：{manifest['peak_RSS_bytes']/2**30:.3f}GiB；独立回读峰值：{data['CPU_RSS_peak_bytes']/2**30:.3f}GiB；两阶段最大：{preppeak/2**30:.3f}GiB。\n- 轨迹耗时：{manifest['elapsed_seconds']:.3f}秒；CPU回读完成距启动：{data['total_elapsed_from_CPU_start']:.3f}秒。\n- 18旧输入及RNG逐字段一致。36旧核心conditions已逐项检查head、α offset、NMS、P3-A输出、C1匹配及阈值。Pilot原worker FAILED/exit1仍保留。\n- 未来仅建议复用18对并采用1790次新增forward上限（1784核心+6检查）；未作PPU实际测试。源码和样本归档被READY哈希锁定。\n- 本轮未存历史原始batch的逐项对照（旧P3-A未保存892份batch）；这些样本只证明当前完整轨迹、批次内容哈希与GT对应，必须等未来PPU基线通过后才能声称最终结果复现。\n'''
 (R/'CPU_PREP_REPORT.md').write_text(report)
 c2=R.parent/'P4_C2_20261027_233036_301795';old=json.loads((c2/'inputs.json').read_text());protected={**old['history_hashes'],**old['protected_hashes']}
 for rec in json.loads((R/'REUSE_PRECHECK.json').read_text())['conditions']:
  protected[rec['pilot_npz']]=rec['pilot_npz_sha256'];protected[str(Path(rec['pilot_npz']).with_suffix('.json'))]=rec['meta_sha256']
 for p,h in protected.items():
  if sha_file(p)!=h:raise RuntimeError('PROTECTED_HASH_CHANGED '+p)
 filenames=[p for p in R.glob('*.py')]+[R/n for n in ['launch_ppu.sh','ppu_worker.sh','status_ppu.sh','monitor_ppu.sh','validate_cpu.sh','CPU_READY.json','INPUT_MANIFEST.json','INPUT_EQUIVALENCE.json','CPU_INPUT_VALIDATION.json','REUSE_PRECHECK.json','REUSE_18_VALIDATION.json','PRE_STATIC_VALIDATION.json','STATIC_VALIDATION.json','MINIMAL_CAPTURE_CPU_TEST.json','SCIENTIFIC_VALIDATOR_CPU_TEST.json','INPUT_LOCK.json','FULL_CAPTURE_PROTOCOL.md','FULL_CAPTURE_SCHEMA.md','RESOURCE_BUDGET.md','CPU_PREP_REPORT.md']]
 locked={str(p):sha_file(p) for p in filenames}
 candidate={'state':'PPU_READY_NOT_STARTED','run_directory':str(R),'budget':ctl.BUDGET,'CPU_passed':True,'static_passed':True,'frozen_batch_count':910,'frozen_batch_hashes':{str(x['sample_idx']):x['sha256'] for x in manifest['samples']},'input_manifest_sha256':sha_file(R/'INPUT_MANIFEST.json'),'locked_files':locked,'protected_files':protected,'checkpoint':str(ctl.R.parent.parent/'LRCP') if False else '/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth','checkpoint_sha256':'666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724','startup_command':str(R/'launch_ppu.sh')+' --approve-1790','monitor_command':str(R/'monitor_ppu.sh'),'status_command':str(R/'status_ppu.sh'),'validation_command':str(R/'validate_cpu.sh'),'PPU_runtime_unverified':True,'PPU_approval_required':True,'model_forwards_so_far':0,'CUDA_initialized':False,'created_epoch':time.time()}
 dry=ctl.verify_ready(full=True,ready_override=candidate);assert dry['passed'] and not torch.cuda.is_initialized()
 write_json(R/'DRY_RUN_RESULT.json',{**dry,'passed':True,'dry_run':True,'started':False,'CUDA_initialized':False,'model_forwards':0})
 candidate['locked_files'][str(R/'DRY_RUN_RESULT.json')]=sha_file(R/'DRY_RUN_RESULT.json');candidate['dry_run_passed']=True
 dry=ctl.verify_ready(full=True,ready_override=candidate);assert dry['passed']
 if time.time()-json.loads((R/'CPU_STARTED.json').read_text())['time']>3600:raise TimeoutError('TOTAL_CPU_PREPARATION_TIMEOUT')
 if torch.cuda.is_initialized():raise RuntimeError('CUDA_INITIALIZED')
 write_json(R/'PPU_READY.json',candidate)
 print('P4_FULL_PPU_READY_NOT_STARTED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  write_json(R/'READY_FAILURE.json',{'passed':False,'error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0});raise
