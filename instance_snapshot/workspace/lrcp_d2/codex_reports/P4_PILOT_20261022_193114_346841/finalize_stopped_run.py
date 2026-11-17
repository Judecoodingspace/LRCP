"""Seal partial P4-Pilot and failure evidence. Does not restart any stage."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,subprocess,csv,ast
R=Path(__file__).resolve().parent
state=json.loads((R/'STATE.json').read_text());failure=json.loads((R/'FAILURE.json').read_text())
static=json.loads((R/'STATIC_CHECKS_FINAL.json').read_text())
proposal=json.loads((R/'MEMORY_PROPOSAL_STATIC_CHECKS.json').read_text())
events=[json.loads(s) for s in (R/'events.jsonl').read_text().splitlines()]
progress=[e for e in events if e['event']=='CPU_REPLAY_PROGRESS']
completed=[e for e in events if e['event']=='CPU_DELAY_COMPLETE']
last=progress[-1];stop=next(e for e in events if e['event']=='STOPPED')
start_estimate=last['epoch']-last['elapsed']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
protected=json.loads((R/'PROTECTED_HASHES_BEFORE.json').read_text());changed=[p for p,d in protected.items() if not Path(p).is_file() or sha(p)!=d]
status=subprocess.run(['git','-C','/workspace/LRCP','status','--short'],capture_output=True,text=True,check=True).stdout
before=json.loads((R/'PREFLIGHT.json').read_text())
assert not Path('/proc/4025436').exists()
assert state['forward_attempted']==0 and state['forward_completed']==0 and not state['PPU_used']
assert not list((R/'batches').iterdir()) and not list((R/'diagnostics').iterdir())
for p in [R/'analyze_pilot.py',R/'runner_revision_candidate.py',R/'bounded_offset_cache_proposal.py']:
 compile(ast.parse(p.read_text()),str(p),'exec')
(R/'DIFF_AUDIT.json').write_text(json.dumps({'passed':not changed and before['git_status_before']==status,'protected_files':len(protected),'changed':changed,'git_status_before':before['git_status_before'],'git_status_after':status,'executed_runner_preserved':True,'new_code_and_proposal_only_in_this_run_directory':True},indent=2)+'\n')
summary={'status':'STOPPED_CPU_PREPARATION_MEMORY_LIMIT','completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'P4_Pilot_complete':False,'static_checks_passed':static['passed'],'static_checks_count':static['checks_count'],'CPU_delay0_common_batches_completed':completed[0]['common_count'],'CPU_delay0_elapsed_seconds':completed[0]['elapsed'],'CPU_delay1_last_logged_common_batches':last['valid_common_count'],'CPU_delay1_last_logged_sample_idx':last['sample_idx'],'CPU_last_logged_elapsed_seconds':last['elapsed'],'CPU_elapsed_approx_seconds':stop['epoch']-start_estimate,'exact_final_preprocessing_sample_idx':'not recorded; cannot infer from final progress log','CPU_max_RSS_bytes':failure['max_rss_bytes'],'CPU_RSS_limit_bytes':8*2**30,'CPU_RSS_over_limit_bytes':failure['max_rss_bytes']-8*2**30,'frozen_samples':0,'PPU_samples':0,'full_model_forward_attempted':0,'full_model_forward_completed':0,'PPU_model_inference_executed':False,'historical_input_and_final_prediction_equivalence':'not tested on selected sample821','mechanism_attribution':'not available; no real per-GT/anchor observation','full_910_collection_ready':False,'experiment_retries':0,'proposal_synthetic_CPU_checks':len(proposal['checks']),'proposed_real_data_replay_executed':False,'protected_files_unchanged':len(protected)-len(changed)}
(R/'RUN_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(R/'ACCEPTANCE.json').write_text(json.dumps({'passed':False,'stage_A_static_CPU':{'passed':True,'checks':34},'stage_B_input_preparation':{'passed':False,'reason':'CPU_RSS_LIMIT_EXCEEDED','max_RSS_bytes':failure['max_rss_bytes'],'frozen_samples':0},'stage_C_smoke':'not executed','stage_D_remaining_samples':'not executed','stage_E_real_GT_attribution':'not executed','halt_on_failure':{'passed':True,'automatic_retries':0,'forward_count':0},'source_and_history_protection':{'passed':not changed},'note':'Static CPU NMS/trace checks do not constitute validation on real PPU detections.'},ensure_ascii=False,indent=2)+'\n')
(R/'NEXT_ATTEMPT_PROPOSAL.md').write_text('''# 下一次重放的具体修正草案（未执行）

现有runner.py、STATE.json、FAILURE.json原样保留，不在失败目录重启。PROPOSED_MEMORY_FIX.diff及runner_revision_candidate.py是可审阅候选。

1. 在独立新目录采用BoundedOffsetDataset代理：每次原生__getitem__仍只调用一次，随后把scene_database中保留的offset视图复制为相同值/shape/dtype/device的小tensor。当前返回item不变，文件读取顺序和RNG不变，不修改数据集源码。目标是解除大offset NPY底层存储的持久引用。合成CPU测试检查值、storage、返回对象和RNG，8项通过；真实数据全过程等价性尚未验证。
2. 另有尚未触发的兼容性问题：环境为torch2.6.0，其torch.load默认weights_only=True，而自建batch归档包含NumPy RNG状态。候选明确对本实验自行创建、经SHA256校验的归档使用weights_only=False。此项不是本轮内存失败原因，也未在阶段C实际触发；不适用于不可信外部归档。
3. 保持8GiB RSS保护，不调高阈值。建议增加每个重放batch的当前/峰值RSS及缓存storage计数，精确记录正在处理的sample_idx；原采集失败没有最后样本的精确记录。
4. 仍需原有A/B/C门槛：完整CPU输入轨迹、18batch归档hash、sample821原生/采集等价及历史输出/GT/IoU0.7分类、逆序检查；不以合成测试替代实际验收。

需要明确批准：在全新P4_PILOT目录应用上述修正，重新进行一次真实CPU重放；成功后才进入原18样本/最多40次forward。每次仍遵守CPU30分钟、PPU15分钟、分析30分钟及原磁盘上限，不更改模型、配置、阈值、时间编码或样本。此前实验的失败即停止约束仍有效，本轮未自动重试。
''')
rss=failure['max_rss_bytes'];elapsed=summary['CPU_elapsed_approx_seconds']
text=f'''# P4-Pilot 部分进度与停止报告

**本轮未完成P4-Pilot。阶段B触发CPU内存上限后已停止，未重试；没有启动PPU模型推理。**

执行目录：{R}。原runner进程PID4025436已退出；源码、checkpoint、配置、历史产物及预检文件共{len(protected)}个锁定路径前后哈希一致。

## 实际执行与资源

| 项目 | 实际结果 |
|---|---|
| 阶段A | 正式采集代码完成；最终34项静态/CPU检查通过；无CUDA初始化或完整模型forward |
| 阶段B | delay0数据准备重放完成910公共批次，耗时{completed[0]['elapsed']:.3f}秒；delay1最后日志为sample553/500公共批次，之后触发内存保护 |
| CPU重放耗时 | 约{elapsed:.3f}秒，根据最后进度时钟与停止事件估计；未达到30分钟超时 |
| CPU峰值RSS | {rss:,}字节，约{rss/2**30:.6f}GiB；上限8GiB，超过{(rss-8*2**30)/2**20:.3f}MiB |
| 缓存18样本 | 0/18；batches目录0文件/0字节 |
| PPU样本/完整forward | 0样本；0次尝试、0次完成；未加载模型到cuda:0 |
| 内部诊断输出 | 0份真实样本数据；diagnostics目录0文件/0字节 |
| 重试次数 | 0；失败目录及已执行runner原样保留 |

CPU阶段前两delay的公共批次是预处理轨迹，不是新增模型推理样本。阶段B尚未到k3缓存所选18例的步骤。最后一个实际处理的sample_idx未持久化，不能把最后进度日志sample553当作精确停止位置；可确认至少完成910+500个公共批次的数据准备。

初始磁盘可用{before['disk_free_bytes']/2**40:.3f}TiB；cuda:0为PPU-ZW810E，初始无计算进程，设备空闲。只完成CPU模型创建和epoch19权重加载；环境torch2.6.0、NumPy1.23.5，参数FP32。设备可用性不等于真实用于推理；本轮没有PPU运行时间、峰值模型显存或设备检测结果可报告。

## 内存增长的具体证据与边界

V2XSIMBaseDataset.retrieve_base_data的211行把params直接引用scene_database里的字典；237/241行加载完整offset_map并将切片写回params['offset']。持久缓存因此可以保留整份多agent、多个delay的NPY底层存储。delay0分支存zeros，delay1开始存视图，与观察到的阶段差异相符。

仅做mmap元数据/别名核查：sample553的offset数组4,096,000字节，切片51,200字节，共享底层存储，比例80倍；sample821分别为6,400,000和51,200字节，比例125倍。详见MEMORY_RETENTION_EVIDENCE.json。这是明确的持久引用来源；没有退出前堆快照，不能给出其占峰值内存比例或认定唯一原因。内存失败不能作为MetaFlow检测机制的证据。

## 对研究验收问题的回答

1. 实际PPU处理0样本，完整模型forward尝试/完成均为0；34项CPU检查只验证语法、接口和合成后处理。
2. PPU没有真实用于模型推理。失败发生在model.to(cuda:0)之前，PPU_START/PPU_EXECUTION未生成。
3. 选定样本的冻结输入、GT映射、最终检测与历史P3-A的一致性尚未验收；不能宣称通过或失败。既有输入文件哈希一致与运行结果一致性是两种不同检查。
4. 六个损失焦点GT（821/2、941/8、17/5、486/2、670/3、918/6）均未采集，无法给出候选消失阶段或真实anchor ID；不是“没有发现损失”。
5. 恢复和双TP控制同样没有新推理观测，不能比较相反机制。
6. raw不足、score、quality、top1000、NMS、range六类在真实样本上的结果全部为未测量，不记为0次。CPU合成NMS重放通过不替代真实native keep验收。
7. 本轮不支持也不反驳P4检测机制假设H1–H8；只发现CPU输入准备的资源障碍。
8. 目前不具备启动910公共样本完整机制采集的条件：CPU重放、sample821 smoke、18样本完整采集和162GT追溯均未通过。

## 保留产物与下一步

runner.py、collector.py、pilot_common.py为本轮实现；test_cpu.py和STATIC_CHECKS_FINAL.json保存34项前置验收；execution.log/events.jsonl/STATE.json/FAILURE.json保留运行和停止证据。analyze_pilot.py已实现，但没有对真实采集数据执行阶段E。阶段A开发中的独占输出保护拦截也保留在stage_A_final.log，不与正式实验失败混淆。

已提供独立内存修复候选及PROPOSED_MEMORY_FIX.diff，未改动本轮执行代码。8项小型合成CPU检查通过，未再次进行历史重放或PPU运行。下一轮必须在新目录明确批准一次修正后的CPU重放，再按原门槛进入PPU；不得通过调高8GiB上限、跳过历史顺序或更换随机预处理来绕过失败。

需要审批的具体内容见NEXT_ATTEMPT_PROPOSAL.md。P4-Pilot状态保持未完成，所有保护文件无修改；没有生成伪造的逐GT/anchor机制结论。
'''
(R/'REPORT.md').write_text(text)
(R/'COMPLETION.json').write_text(json.dumps({'status':'STOPPED_INCOMPLETE','P4_Pilot_complete':False,'failure_report_complete':True,'PPU_model_inference_executed':False,'full_forward_completed':0,'samples_completed':0,'source_protection_passed':not changed,'new_real_data_retry_requires_explicit_approval':True,'report':str(R/'REPORT.md')},ensure_ascii=False,indent=2)+'\n')
files=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACTS.sha256')
with (R/'ARTIFACTS.sha256').open('x') as f:
 for p in files:f.write(sha(p)+'  '+str(p.relative_to(R))+'\n')
for line in (R/'ARTIFACTS.sha256').read_text().splitlines():
 d,n=line.split('  ',1);assert sha(R/n)==d
print(json.dumps({'protected_files':len(protected),'changed':changed,'artifacts_sealed':len(files),'summary':summary},ensure_ascii=False))
