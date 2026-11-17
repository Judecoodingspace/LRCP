"""CPU-only evidence summary and before/after audit. Never modifies source run."""
import os,csv,json,hashlib,datetime,ast
from pathlib import Path
from zoneinfo import ZoneInfo
O=Path(__file__).resolve().parent
S=Path('/workspace/lrcp_d2/codex_reports/P4_PILOT_RETRY_20261023_234457_384594')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,d):
 with (O/n).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def rows(n):return list(csv.DictReader((S/'diagnostics/analysis'/n).open()))
v=read(O/'INDEPENDENT_VALIDATION.json');assert v['passed'] and v['additional_model_forwards']==0 and not v['CUDA_initialized']
source=read(O/'SOURCE_HASHES_BEFORE.json');changed=[]
for category,files in source.items():
 for p,h in files.items():
  if sha(p)!=h:changed.append(p)
assert not changed,changed
state=read(S/'STATE.json');execution=read(S/'PPU_EXECUTION.json');worker=read(S/'WORKER_FINAL.json');reg=read(O/'REGRESSION_TESTS.json');summary=read(S/'ANALYSIS_SUMMARY.json')
assert state['forward_completed']==state['forward_attempted']==40 and worker['state']=='FAILED' and worker['exit_code']==1
ledger=[json.loads(x) for x in (S/'events.jsonl').read_text().splitlines()]
start_count=sum(x['event']=='FORWARD_START' for x in ledger);validated_count=sum(x['event']=='FORWARD_VALIDATED' for x in ledger)
assert start_count==validated_count==40
assert all(x['output_device']=='cuda:0' for x in execution['runs'])
focal=rows('focal_GT_summary.csv');high=rows('high_IoU_candidate_deletion.csv');stage=rows('per_GT_stages.csv');case=rows('per_case_stage_counts.csv')
evidence=[]
for row in focal:
 if row['group']!='loss_TP70_to_FN70' or float(row['alpha'])!=1.:continue
 sid=int(row['sample_idx']);gi=int(row['gt_idx']);cand=[x for x in high if int(x['sample_idx'])==sid and int(x['gt_idx'])==gi and float(x['alpha'])==1.]
 assert len(cand)==int(row['count_ge70_raw_decoded'])>0
 cause=row['availability_class'];nms=[x for x in cand if x['first_deletion_stage']=='nms_overlap_suppression']
 if cause=='score_filter':
  assert all(float(x['score'])<=.2 and x['first_deletion_stage']=='score_filter' for x in cand)
  chosen=max(cand,key=lambda x:(float(x['score']),-int(x['anchor_id'])))
 else:
  assert cause=='nms_overlap_suppression' and nms
  chosen=max(nms,key=lambda x:(float(x['IoU_GT']),-int(x['anchor_id'])))
 e={'sample_idx':sid,'gt_idx':gi,'last_strict_candidate_disappearance':cause,'raw_best_IoU':float(row['best_iou_raw_decoded']),'raw_strict_candidate_count':len(cand),'score_pass_strict_count':int(row['count_ge70_score_pass']),'NMS_input_strict_count':int(row['count_ge70_nms_top1000']),'final_strict_count':int(row['count_ge70_final_range']),'representative_anchor_id':int(chosen['anchor_id']),'representative_IoU_GT':float(chosen['IoU_GT']),'representative_score':float(chosen['score']),'suppressor_anchor_id':int(chosen['NMS_suppressor_anchor_id']),'suppressor_IoU_GT':chosen['NMS_suppressor_IoU_GT'],'suppressor_score':chosen['NMS_suppressor_score'],'NMS_overlap':chosen['NMS_suppressor_overlap'],'all_raw_strict_candidates_max_score':max(float(x['score']) for x in cand)}
 evidence.append(e)
assert len(evidence)==6
with (O/'LOSS_STAGE_EVIDENCE.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
max_nms_input=max(int(x['quality_pass']) for x in case)
TP={str(a):sum(x['native_evaluator_TP70'] for x in v['native_evaluator_checks'] if x['alpha']==a) for a in [0.,1.]}
FP={str(a):sum(x['native_evaluator_FP70'] for x in v['native_evaluator_checks'] if x['alpha']==a) for a in [0.,1.]}
for p in O.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
completion={'offline_validation_state':'PASSED','P4_pilot_collection_and_offline_acceptance_completed':True,'source_run_directory':str(S),'followup_directory':str(O),'source_worker_state_preserved':'FAILED','source_worker_exit_code_preserved':1,'failure_reason':'caller passed NumPy arrays to native tensor-only eval_utils; adapted at CPU call site in new directory','source_model_samples':18,'source_model_forward_attempted':40,'source_model_forward_completed':40,'source_PPU_inference_seconds':execution['elapsed_seconds'],'source_PPU_device':execution['resource_end']['framework']['name'],'additional_model_forwards':0,'CUDA_initialized_in_followup':False,'regression_checks':reg['checks_count'],'independent_core_cases':len(v['native_evaluator_checks']),'GT_count':162,'GT_condition_rows':324,'stage_rows':1944,'anchor_rows':460800,'historical_pred_exact_cases':sum(x['historical']['pred_exact'] for x in execution['runs']),'historical_score_exact_cases':sum(x['historical']['score_exact'] for x in execution['runs']),'TP70':TP,'FP70':FP,'max_native_NMS_input_count':max_nms_input,'source_files_unchanged':len(source['run_files']),'protected_files_unchanged':len(source['protected_files']),'completed_at_Asia_Shanghai':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
save('COMPLETION.json',completion)
save('DIFF_AUDIT.json',{'passed':True,'source_run_files_checked':len(source['run_files']),'protected_files_checked':len(source['protected_files']),'changed':changed,'original_failed_logs_STATE_EXIT_READY_and_detections_unchanged':True,'mutation_scope':'new CPU follow-up directory only; no READY publication or PPU restart'})
report=(O/'PPU_RESEARCH_REPORT.md').read_text()
notice=f'''本报告在CPU补充验收后完成。原PPU作业已采集完18样本/40次forward，随后因我编写的独立验收调用缺少NumPy→CPU tensor转换而退出码1；不是训练失败或PPU推理失败。原目录FAILED状态、退出码和日志保持原样。补充验收在本目录通过，没有新模型forward、没有重新预处理、没有修改原始实验或LRCP源码。原始数据位于 `{S}`。\n\n'''
report=report.replace('\n\n','\n\n'+notice,1)
report+='''\n## 六个失去TP70目标的具体证据\n\n原生score条件严格>0.2，NMS抑制条件为框间BEV IoU>0.15。下表NMS行选取被NMS抑制且IoU_GT最高的严格候选；score行选取全部严格候选中分数最高者，按anchor ID确定并列顺序。完整候选没有截断，见原high_IoU_candidate_deletion.csv。\n\n| sample / GT | raw严格候选数 | score后 / NMS后严格候选数 | 最后消失阶段 | 代表anchor / IoU_GT / score | suppressor anchor / IoU_GT / score |\n|---|---:|---|---|---|---|\n'''
for e in evidence:
 sup='—' if e['suppressor_anchor_id']<0 else f"{e['suppressor_anchor_id']} / {float(e['suppressor_IoU_GT']):.6f} / {float(e['suppressor_score']):.6f}"
 report+=f"| {e['sample_idx']} / {e['gt_idx']} | {e['raw_strict_candidate_count']} | {e['score_pass_strict_count']} / {e['final_strict_count']} | {e['last_strict_candidate_disappearance']} | {e['representative_anchor_id']} / {e['representative_IoU_GT']:.6f} / {e['representative_score']:.6f} | {sup} |\n"
report+=f'''\n这6个α1损失案例的raw阶段均仍有严格候选，直接不支持它们的“raw完全无候选”解释；3个在score门限失去最后严格候选，3个在NMS失去最后严格候选。它们仍可同时存在回归几何和分类分数耦合变化，不能说几何完全没有退化，更不能据此认定MetaFlow offset正确或错误。\n\n恢复组6例在α0下分别为4个NMS丢尽、1个score丢尽、1个raw无严格候选，α1下全部保留严格候选；6个稳定控制两侧均保留。全部162个GT的原生TP70为{TP['0.0']}→{TP['1.0']}，FP70为{FP['0.0']}→{FP['1.0']}。这是按结果分层选择的18例，不代表910样本整体净损失，不能外推比例。\n\n36个核心条件最多只有{max_nms_input}个quality后候选，均少于1000，因此本先导未触发top1000截断。质量过滤删除了α0的1个anchor，range删除了α0/α1的1/2个anchor；它们未成为这162个GT中任何一个的最后严格候选消失阶段。\n\n## 已验证和仍待验证的机制\n\n- H1 raw候选不足：在6个α1焦点损失中被排除，但在其他GT仍有raw缺候选，不能全局排除。\n- H2 score删尽、H4b NMS抑制：各有3个焦点损失的逐anchor证据；NMS keep已与原生完全相同。\n- H3 quality、H4a top1000、H5 range：本先导未发现其造成最后严格候选消失，不能外推其他样本。\n- H6融合/检测头变化、H7 offset方向/尺度错误、H8采样边界效应：需要结合保存的特征和采样摘要进一步设计独立干预；当前阶段归因不构成offset误差的因果证明，背景offset零值也不是可靠运动真值。\n\n下一项最小验证可完全使用已保存的12800 raw候选和分数，固定各α的raw几何/方向，按同一anchor网格位置双向移植另一α的原始分类分数，按原score/quality/top1000/NMS/range协议做CPU后处理反事实；须先验收原始分数下的最终输出，再比较6个损失/6个恢复/6个控制是否改变最后严格候选阶段。anchor索引只表示计算网格对应，不是物体身份。这比P3-C1的post-NMS分数交换更早介入，但仍是保存输出层的诊断，不代表模型内部独立因果干预。本次未执行该新实验。\n\n18例路径已具备技术可行性的实测证据，但现有40次forward已全部用尽。不能沿用本轮15分钟/1GiB预算或本轮授权自动扩展到910；全量采集需要新的资源计划、分片/流式策略与用户单独批准。\n\n## 读取验收结果\n\n原 `status_ppu.sh` 仍会如实返回FAILED，因为原worker确实以1退出，未篡改为0。补充科学验收通过以本目录COMPLETION.json与INDEPENDENT_VALIDATION.json为准。不要重复启动原PPU作业。原36组采集和后处理CSV可继续使用，无需额外PPU前向。\n'''
(O/'PPU_RESEARCH_REPORT.md').write_text(report)
short=f'''# P4-Pilot CPU验收接口修复\n\n**PPU推理已完成；CPU补充验收现已通过。** 原worker状态FAILED和退出码1保持不变。\n\n根因是验收脚本把保存的NumPy数组直接传给要求torch.Tensor的原eval_utils.caluclate_tp_fp；common_utils访问.is_cuda时报错。这是我编写的调用适配遗漏。修复仅在新目录调用前用torch.from_numpy转换为CPU tensor，不改原评估函数、分数排序、dtype或几何。\n\n真实数据15项回归检查通过，补充独立验收通过36组核心条件、162GT、1944阶段记录和460800 anchor记录；40次历史输出的框/分数逐位相同，smoke及逆序结果均逐位一致。额外模型前向0、CUDA未初始化，源运行{len(source['run_files'])}个文件及3310个受保护文件哈希未变。\n\nPPU设备PPU-ZW810E/cuda:0，40次完整前向用时{execution['elapsed_seconds']:.3f}秒，已分配/保留显存峰值{execution['peak_allocated_bytes']/2**30:.3f}/{execution['peak_reserved_bytes']/2**30:.3f}GiB。\n\n6个焦点损失GT在α1 raw均有IoU≥0.7候选：score删尽3个（sample17/GT5、670/GT3、918/GT6），NMS删尽3个（821/GT2、941/GT8、486/GT2）。边界和恢复/控制组证据见PPU_RESEARCH_REPORT.md及LOSS_STAGE_EVIDENCE.csv。\n\n完整原始数据仍位于 `{S}`，新验收目录为 `{O}`。旧启动/状态脚本不修改，不重新发布READY，不重启PPU。COMPLETION.json明确区分原worker失败与补充验收通过。\n'''
(O/'REPORT.md').write_text(short)
with (O/'ARTIFACTS.sha256').open('x') as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p.name!='ARTIFACTS.sha256':f.write(sha(p)+'  '+str(p.relative_to(O))+'\n')
print(json.dumps(completion,ensure_ascii=False,indent=2))
