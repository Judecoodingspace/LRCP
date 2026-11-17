"""Chinese evidence report from independently accepted pilot artifacts; no inference."""
import csv,json,time,sys
from pathlib import Path
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P4_PILOT_RETRY_20261023_234457_384594')
from pilot_common import *
OUT=Path(__file__).resolve().parent

def report():
 v=json.load((OUT/'INDEPENDENT_VALIDATION.json').open());assert v['passed']
 exe=json.load((R/'PPU_EXECUTION.json').open());summary=json.load((R/'ANALYSIS_SUMMARY.json').open())
 d=R/'diagnostics'/'analysis';focal=list(csv.DictReader((d/'focal_GT_summary.csv').open()));stages=list(csv.DictReader((d/'per_GT_stages.csv').open()))
 ix={(int(x['sample_idx']),int(x['gt_idx']),float(x['alpha'])):x for x in focal}
 si={(int(x['sample_idx']),int(x['gt_idx']),float(x['alpha']),x['stage']):x for x in stages}
 text=['# P4-Pilot：有限PPU机制采集报告','',f"实际完成18个固定样本、{exe['full_forward_completed']}次完整模型前向；实际设备为cuda:0，PPU阶段用时{exe['elapsed_seconds']:.3f}秒。本报告仅在全部历史复现、smoke、逆序、NMS与独立CPU验收通过后生成。",'',f"峰值PPU已分配显存{exe['peak_allocated_bytes']/2**30:.3f} GiB，峰值保留显存{exe['peak_reserved_bytes']/2**30:.3f} GiB。设备与真实逐次运行资源记录见PPU_START.json、PPU_EXECUTION.json与ppu.log。",'','全部162个GT、324个GT条件、1944个阶段记录及460800个anchor条件已验证。每个案例最终框/GT与历史P3-A通过原协议容差和TP70/IoU0.7关系检查；36个核心条件CPU NMS keep与原生一致。','', '| sample_idx | GT索引 | 选择组 | α0 六阶段高IoU数量 | α1 六阶段高IoU数量 | α1 最后消失阶段 | α1 raw最佳anchor / IoU |', '|---:|---:|---|---|---|---|---|']
 for sel in selections():
  sid=int(sel['sample_idx']);gi=int(sel['gt_idx']);a=ix[(sid,gi,0.)];b=ix[(sid,gi,1.)];raw=si[(sid,gi,1.,'raw_decoded')]
  counts=lambda x:'/'.join(x['count_ge70_'+s] for s in STAGES)
  text.append(f"| {sid} | {gi} | {sel['group']} | {counts(a)} | {counts(b)} | {b['availability_class']} | {raw['best_anchor_id']} / {float(raw['best_iou']):.9f} |")
 text+=['','六阶段依次为raw decoded→score→quality→top1000→NMS→range。上表列出既定18个焦点目标，全部162个GT均保存在per_GT_stages.csv和per_GT_summary.csv，没有剔除恢复组或控制组。','', '## 可检验机制的证据','', '以下为所选案例中的候选可用性分类，不能外推910样本的总体贡献：','', '```json',json.dumps(summary['focal_availability_counts'],ensure_ascii=False,indent=2),'```','', 'raw阶段不存在IoU≥0.7候选支持候选生成/回归不足的描述；raw存在而后续消失则定位到score、尺寸/z、top1000、NMS或range。high_IoU_candidate_deletion.csv保存每个高IoU候选的anchor ID、首次删因、NMS suppressor anchor/IoU/score，per_GT_stages.csv保存所有候选ID及最佳候选，不能把最佳框一个索引当成物体身份。','', '融合/MetaFlow摘要及检测头变化见逐条件NPZ/JSON及feature_summaries.csv。两α raw offset与上游特征相同、effective=α×raw由采集器检验；中间特征差异和采样越界摘要是关联证据，不独立证明offset方向或预测错误。尺寸/z过滤沿原实现，未修正原模型代码。','', '## 边界与后续','', '这是一项post-NMS损失定位的有限案例研究。候选生成、分数、几何与NMS相互耦合；本实验没有独立干预这些内部模块，不作因果贡献分解。offset运动真值缺乏可靠有效mask，不能用全图offset差异直接判定预测错误。','', '当前结果只验收18样本/40次forward的路径，不能视为910样本扩展授权。完整采集仍需另行批准，重新评估空间、时间、输出规模及抽样代表性；本作业不会自动扩展或重试。','', '原checkpoint/config/源码及历史保护文件前后哈希一致；详见INDEPENDENT_VALIDATION.json、RESULT_ARTIFACTS.sha256、ACCEPTANCE.json。']
 p=OUT/'PPU_RESEARCH_REPORT.md'
 with p.open('x') as f:f.write('\n'.join(text)+'\n')
 print('PPU_RESEARCH_REPORT_WRITTEN',flush=True)
if __name__=='__main__':report()
