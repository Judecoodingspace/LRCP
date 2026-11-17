"""Verify exported P3-C1 traces and write the Chinese interpretation/report. CPU only."""
import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import numpy as np

R = Path(__file__).resolve().parent
B = R.parent / 'P3B_20261020_103959_980233'
assert not (R/'REPORT.md').exists()
J = json.loads((R/'summary.json').read_text())
REF = json.loads((B/'summary.json').read_text())
assert J['validation']['passed'] and all(J['baseline_exact'].values())
checks = {}


def check(name, ok):
    checks[name] = bool(ok)
    if not ok:
        raise AssertionError(name)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def csv_write(name, rows):
    with (R/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(name,obj):
    with (R/name).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


src={(r['sample_idx'],r['alpha'],int(r['pred_idx'])):r for r in read_csv(B/'per_prediction.csv')}
pred=read_csv(R/'per_prediction_evaluation.csv')
gts=read_csv(R/'per_GT_evaluation.csv')
ranks=read_csv(R/'global_ranking_PR.csv')
pm={(r['scenario'],r['sample_idx'],int(r['pred_idx'])):r for r in pred}
check('unique_prediction_keys',len(pm)==len(pred)==66852)
check('GT_trace_row_count',len(gts)==130624)
for p in pred:
    original=src[p['sample_idx'],p['geometry_alpha'],int(p['pred_idx'])]
    donor=src[p['sample_idx'],p['source_alpha'],int(p['source_pred_idx'])]
    assert float(p['native_score'])==float(original['score'])
    assert float(p['new_score'])==float(donor['score'])
    if p['paired']=='0':
        assert p['source_alpha']==p['geometry_alpha'] and p['source_pred_idx']==p['pred_idx']
        assert float(p['native_score'])==float(p['new_score'])
check('all_scores_have_exact_source_identity',True)
for g in gts:
    key=str(int(float(g['eval_IoU'])*100))
    pi=int(g['matched_pred_idx'])
    if pi>=0:
        p=pm[g['scenario'],g['sample_idx'],pi]
        assert p['TP'+key]=='1' and int(p['matched_gt_idx'+key])==int(g['gt_idx'])
    assert int(g['TP'])==int(pi>=0)
check('GT_to_prediction_trace_consistent',True)

grouped=defaultdict(list)
for r in ranks:
    grouped[r['scenario'],r['eval_IoU']].append(r)
for (name,t),rows in grouped.items():
    m=J['conditions'][name]['eval'][str(int(float(t)*100))]
    tp=0;rec=[];prec=[];last_score=float('inf');seen=set()
    for rank,r in enumerate(rows,1):
        assert int(r['global_rank'])==rank
        score=float(r['score']);assert score<=last_score;last_score=score
        identity=(r['sample_idx'],int(r['pred_idx']));assert identity not in seen;seen.add(identity)
        p=pm[(name,*identity)];key=str(int(float(t)*100))
        assert p['TP'+key]==r['tp'] and float(p['new_score'])==score
        assert int(p['matched_gt_idx'+key])==int(r['matched_gt_idx'])
        tp+=int(r['tp'])
        assert tp==int(r['cumulative_TP']) and rank-tp==int(r['cumulative_FP'])
        rec.append(tp/8164);prec.append(tp/rank)
    envelope=np.maximum.accumulate(np.asarray(prec)[::-1])[::-1]
    ap=float(np.sum(np.diff(np.r_[0.,rec])*envelope))
    check('CSV_AP:'+name+':'+t,abs(ap-m['ap'])<=1e-12)
    check('CSV_counts:'+name+':'+t,tp==m['tp'] and len(rows)-tp==m['fp'])

distribution_rows=[]
transfer_rows=[]
coverage_rows=[]
for name,m in J['conditions'].items():
    rows=[p for p in pred if p['scenario']==name]
    for key in ['50','70']:
        for label in ['0','1']:
            rr=[p for p in rows if p['TP'+key]==label]
            distribution_rows.append({'scenario':name,'eval_IoU':int(key)/100,'target_TP':int(label),'count':len(rr),
                'native_score_mean':float(np.mean([float(p['native_score']) for p in rr])),
                'new_score_mean':float(np.mean([float(p['new_score']) for p in rr])),
                'native_score_ge_0p8':sum(float(p['native_score'])>=.8 for p in rr),
                'new_score_ge_0p8':sum(float(p['new_score'])>=.8 for p in rr)})
    if m['association_threshold'] is None:
        continue
    a=m['geometry_alpha'];tau=m['association_threshold']
    cross=REF['cross_condition_association'][str(tau)][a]
    coverage_rows.append({'scenario':name,'association_IoU':tau,'geometry_alpha':a,
        'paired':m['coverage']['paired'],'total_predictions':len(rows),'pred_pair_coverage':m['coverage']['pair_coverage'],
        'TP70_pair_coverage':cross['paired']['tp70']/REF['metrics'][a]['70']['tp'],
        'FP70_pair_coverage':(cross['paired']['n']-cross['paired']['tp70'])/REF['metrics'][a]['70']['fp'],
        'unpaired':m['coverage']['unpaired'],'unpaired_TP70':cross['unpaired']['tp70'],
        'unpaired_FP70':cross['unpaired']['n']-cross['unpaired']['tp70'],
        'actually_changed_scores':m['coverage']['scores_changed'],
        'local_order_changed_samples':m['coverage']['local_order_changed_samples']})
    for source_label in ['0','1']:
        for target_label in ['0','1']:
            rr=[p for p in rows if p['paired']=='1' and p['TP70']==target_label
                and src[p['sample_idx'],p['source_alpha'],int(p['source_pred_idx'])]['tp70']==source_label]
            transfer_rows.append({'scenario':name,'association_IoU':tau,'donor_TP70':int(source_label),
                'target_TP70':int(target_label),'count':len(rr),
                'native_score_mean':float(np.mean([float(p['native_score']) for p in rr])) if rr else None,
                'imported_score_mean':float(np.mean([float(p['new_score']) for p in rr])) if rr else None,
                'score_delta_mean':float(np.mean([float(p['new_score'])-float(p['native_score']) for p in rr])) if rr else None})
csv_write('score_distribution_changes.csv',distribution_rows)
csv_write('donor_target_label_groups.csv',transfer_rows)
csv_write('coverage.csv',coverage_rows)
all_changes_zero=all(v==0 for m in J['conditions'].values() for c in m['matching_changes'].values() for v in c.values())
check('observed_all_matching_relationships_unchanged',all_changes_zero)
all_tie_width=max(v['ties']['tp_first_AP']-v['ties']['fp_first_AP'] for m in J['conditions'].values() for v in m['eval'].values())
cf_tie_width=max(v['ties']['tp_first_AP']-v['ties']['fp_first_AP'] for m in J['conditions'].values() if m['association_threshold'] is not None for v in m['eval'].values())
g1_changes=[r['G1_AP70_change'] for r in J['sensitivity']]
g0_changes=[r['G0_AP70_change'] for r in J['sensitivity']]
check('G1_no_recovery_at_all_tau',all(v<0 for v in g1_changes))
check('G0_effect_changes_sign_with_tau',min(g0_changes)<0<max(g0_changes))

interpretation={
    'G1_matchedS0_AP70_restored':False,
    'G1_AP70_change_range':[min(g1_changes),max(g1_changes)],
    'G0_matchedS1_uniform_AP70_decrease':False,
    'G0_AP70_change_range':[min(g0_changes),max(g0_changes)],
    'all_pred_labels_and_GT_matching_indices_unchanged':all_changes_zero,
    'maximum_global_tie_AP_width_all_conditions':all_tie_width,
    'maximum_global_tie_AP_width_counterfactual_only':cf_tie_width,
    'scope':'Post-NMS saved-box score crossover. Not an independent causal intervention on internal model logits; no claim that pre-NMS behavior would remain unchanged.',
    'supported':['指定三个阈值下，配对S0移植未恢复G1 AP70；G0取S1的方向对关联阈值敏感。',
                 '本数据观测到TP/FP和匹配索引不变；反事实AP差异由固定已保存框的全局排序变化产生。',
                 '分数与其原几何条件下的定位质量相关；跨几何移植可能将原FP分数赋给目标TP，不能称为统一优劣排序器。'],
    'cannot_exclude':['未匹配框未干预，特别是相当比例FP70分数保留；混合分数分布也可能造成排序差异。',
                      '关联为IoU辅助关系，不是物理身份；改变阈值同时改变覆盖率和匹配集合。',
                      'G1缺失的高IoU候选不能由换分数恢复；但固定框集合包含数量和post-NMS选择，不是纯几何因果隔离。',
                      '缺少pre-NMS候选、被阈值删除的框、offset/中间特征，不能确认MetaFlow偏移错误。']}
save('interpretation.json',interpretation)

lines=[f'''# LRCP P3-C1：固定检测框集合下的置信度交叉实验

完成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}（Asia/Shanghai）。

**P3-C1已完成。α=1固定框使用α=0的配对分数后，三个关联阈值的AP70均未恢复，反而略降；α=0固定框使用α=1配对分数的效果随关联阈值改变方向。** 逐预测TP/FP、匹配GT及逐GT匹配框索引全部保持不变；这是重新执行贪心匹配后的实测结果，未预设标签固定。

研究对象为LRCP协同3D目标检测的BEV评估：公共910样本、8164个GT实例，历史帧差k=3，模型内部时间编码300，物理观测间隔约600 ms。仅CPU执行，不导入torch；未启动PPU或训练，不修改原始框、模型、源码、checkpoint或历史产物。

## 协议和复用

首先验证P3-B最终清单43个产物哈希，读取summary、per_gt、逐预测、关联CSV和完整报告。三个阈值的关联直接复用P3-B记录，未重新求解关联或重复P3-B几何统计；每份实际输入NPZ与P3-B记录的哈希一致。关联按同一样本预测-预测BEV IoU门限先最大化配对数、再最大化有效配对IoU总和，索引一对一、分数来源一致，算法和产物均锁定到P3-B。

G0/G1表示各自原始post-NMS框集合；“配对S1/S0”只替换成功配对框的分数，未配对框保留本条件分数。坐标、框数、框顺序和GT均固定，未重新过滤或执行NMS；保持float32源分数。完整协议见PROTOCOL.md。

## 基线验收

四个原始AP均逐位相等，既定1e-12容差内误差为0；TP/FP及逐GT/逐预测匹配与P3-B一致。基线门通过后才进行六种反事实条件。

| 固定框/原分数 | AP50 | TP50 / FP50 | AP70 | TP70 / FP70 |
|---|---:|---:|---:|---:|
| G0/S0 | 0.8665290892796786 | 7111 / 1064 | 0.8071592298880969 | 6754 / 1421 |
| G1/S1 | 0.8768414234452961 | 7209 / 1329 | 0.7854230203953630 | 6645 / 1893 |

## 反事实数值

| 关联IoU阈值τ | G0取配对S1 AP50 | G0取配对S1 AP70 | ΔAP70 对G0原分数 | G1取配对S0 AP50 | G1取配对S0 AP70 | ΔAP70 对G1原分数 |
|---|---:|---:|---:|---:|---:|---:|
''']
for r in J['sensitivity']:
    lines.append(f'| {r["association_IoU"]} | {r["G0_matchedS1_AP50"]:.10f} | {r["G0_matchedS1_AP70"]:.10f} | {r["G0_AP70_change"]:+.10f} | {r["G1_matchedS0_AP50"]:.10f} | {r["G1_matchedS0_AP70"]:.10f} | {r["G1_AP70_change"]:+.10f} |\n')
lines.append('''
数值为0–1尺度；AP70变化换算为百分点：G1取配对S0分别为−0.2583、−0.2363、−0.0723个百分点；G0取配对S1分别为−0.0996、+0.0239、+0.0528个百分点。所有G0反事实仍为TP50/FP50=7111/1064、TP70/FP70=6754/1421；所有G1反事实仍为7209/1329、6645/1893。

**回答1：G1使用配对S0不恢复AP70。** 三个阈值均下降，且仍低于G0原始AP70，差距分别为0.0243189、0.0240990、0.0224593。不能将其解读为“α0分数到哪里都更好”。

**回答2：G0使用配对S1并非稳定下降。** τ=0.5略降，τ=0.7/0.9略升，方向对关联策略敏感。这不支持“α1分数在所有相近框上普遍较差”的简单解释。

**回答3：敏感性体现在覆盖率及效应方向/幅度。** G1“不恢复”的方向在三个阈值一致，但幅度明显变化；G0连方向也变化。阈值提高后保留更接近的几何框，同时漏掉更多差异大或无对应框；三组不是同一干预强度，也不必是严格嵌套配对集合。

## 关联覆盖率与实际分数变化

| τ | 每侧配对框数 | G0覆盖率 | G1覆盖率 | G0未配对 | G1未配对 | 每侧实际数值改变的分数 |
|---|---:|---:|---:|---:|---:|---:|
''')
for r in J['sensitivity']:
    tau=r['association_IoU'];m0=next(m for m in J['conditions'].values() if m['association_threshold']==tau and m['geometry_alpha']=='0.0');m1=next(m for m in J['conditions'].values() if m['association_threshold']==tau and m['geometry_alpha']=='1.0')
    lines.append(f'| {tau} | {r["pairs_each"]} | {100*r["coverage_G0"]:.3f}% | {100*r["coverage_G1"]:.3f}% | {m0["coverage"]["unpaired"]} | {m1["coverage"]["unpaired"]} | {m0["coverage"]["scores_changed"]} |\n')
lines.append('''
部分配对框两侧分数本就逐位相等，因此配对数与实际数值改变数不同。六种交换在372–538个样本中改变了局部分数排序，并非未施加扰动。所有未配对分数逐位保留，两方向合并后的分数多重集守恒。

τ=0.5时，G1配对覆盖99.07%的原TP70，但只覆盖68.20%的FP70；仍有62个TP70、602个FP70保留S1。G0则保留32个TP70、269个FP70的S0。τ=0.9时G1未配对1935框，其中999个TP70、936个FP70。**整体覆盖率高，不等于FP或所有机制被充分干预。** 详见coverage.csv。

## 贪心匹配及同分排序检查

每个条件重新调用原LRCP caluclate_tp_fp和calculate_ap/voc_ap，另用P3-B同一instrumented matcher记录索引，两者完整score/TP/FP列表一致。

六个反事实在两个评估阈值下：逐预测TP/FP变化数=0、预测匹配GT变化数=0、逐GT TP/FN变化数=0、GT匹配预测索引变化数=0，未配对框也未出现标签变化。因此在本次已施加的干预中，AP改变确实来自既定框/标签上的全局排序，而不是贪心重分配。此观察不保证其他框集或其他分数干预也保持标签不变。

单样本内未发现完全相等的分数；全局存在不同样本间的同分。正式结果始终保留原默认np.argsort，未换成稳定排序。重复默认排序结果相同，但quicksort不保证并列元素稳定。另计算stable排序及同分组TP优先/FP优先上下界作为诊断，正式AP不变。

''')
lines.append(f'全部条件的最大同分组极端重排AP宽度为{all_tie_width:.12g}；六种反事实自身最大宽度为{cf_tie_width:.12g}。它们远小于本次主要反事实变化，不能解释效应方向和量级。所有同分信息与诊断值保存在summary.json各条件的ties字段及local_tie_diagnostics.csv。\n')
lines.append('''
## 分数移植与目标框质量的关系

本次数据不支持把S0/S1视作跨几何条件可无损互换的统一排序器。关联框的IoU相近，并不保证它们在eval IoU0.7下拥有相同TP标签。

以τ=0.5、G1取配对S0为例：

- G1 TP70分数均值0.880959→0.872382，score≥0.8的TP70由5410降至5247；FP70分数均值0.489417→0.485469，高分FP70由316略降至314。TP/FP均值都下降并不保证AP改善，必须检查相对排序。
- 107个“来源G0为FP70、目标G1为TP70”的配对框，导入分数平均下降0.084154；246个“来源G0为TP70、目标G1为FP70”的配对框，导入分数平均变化−0.008892。
- 这直接说明分数是在不同几何质量标签之间被移植；AP下降与“把另一条件较低分的原FP分数赋给当前TP”等现象相容，不能将所有变化解释为α1自身置信度变差。均值和标签分组仅为描述，未给出独立因果贡献。

其他阈值的完整分组见donor_target_label_groups.csv与score_distribution_changes.csv，未添加新的干预条件。

## 能够排除的解释及证据边界

在当前post-NMS框、公共910样本、指定三个关联规则及未配对保留策略下：

1. 可以排除“只需移植α0配对分数，就可恢复α1 AP70”的具体解释；三组均未恢复。
2. 可以排除本次反事实AP差异由框数、坐标、GT改变或贪心标签重分配造成；它们经过逐项检查保持不变。
3. 同分排序敏感性不足以解释本次主要AP变化。
4. 不支持“α1分数对任意固定几何均普遍有害”：G0的方向随阈值改变。

**仍不能排除全部置信度相关机制，也不能把剩余差距直接归为纯几何或offset错误。** 未配对分数未被干预；混用导入分数与原分数可能产生分布不一致；IoU关联不是物理身份；G0/G1集合本身在数量和已选候选上不同。G1没有保存的高IoU候选，无法靠重新排序创造，原TP70可达数差109仍保留；但该事实区分不了候选几何形成、置信度筛选和NMS选择。

**本实验是post-NMS检测框层面的诊断，不代表模型内部分类分数的独立因果干预，也不能推断pre-NMS过程保持不变。** 原分数可能已影响置信度过滤和NMS，真实模型层面替换分数可能改变被保留框；本次按协议固定的是已经保存的框，未重新模拟该过程。

仍需验证的机制假设包括：补偿是否改变框回归/融合后的定位，是否通过pre-NMS候选质量、分数过滤或NMS选择改变高IoU候选保留，以及置信度是否与各自几何条件的定位质量耦合。验证这些机制需要有针对性的pre-NMS候选、被删除框、offset与中间特征证据；本次没有启动此类采集，也没有训练或新PPU任务。结论限定当前checkpoint和k=3，不外推其他延迟或总体置信度性能。

## 交付、复现与追溯

- run_experiment.py：全部基线、六个反事实、原协议重评估及主图生成。
- finalize.py：CSV分数来源、GT/预测互引、PR积分复核及机制报告。
- PROTOCOL.md：预设协议和解释边界。
- baseline_validation.json、summary.json、validation.json、verification.json、interpretation.json：基线、指标、敏感性与核验证据。
- score_assignments.csv（66852行）：逐框原分数、新分数、供体sample_idx/pred_idx及关联标记。
- per_prediction_evaluation.csv（66852行）、per_GT_evaluation.csv（130624行）：全部条件匹配及原生对照索引。
- global_ranking_PR.csv（133704行）：每条件、每阈值的原协议全局排序及PR。
- metrics.csv、sensitivity.csv、coverage.csv、score_distribution_changes.csv、donor_target_label_groups.csv：可直接复用的结果表。
- fixed_geometry_manifest.csv、input_hashes.json：输入及固定坐标/GT校验。
- PR_curves.png/pdf、score_crossover_sensitivity.png/pdf：独立图表；PR图为precision 0.75–1.005的放大视图，完整曲线在CSV中。
- execution.log、COMPLETION.json、ARTIFACTS.sha256：执行、最终状态和交付文件哈希。

复现时将run_experiment.py、finalize.py、PROTOCOL.md复制到全新的codex_reports/P3C1_<new_run_id>/，依次使用/workspace/venvs/lrcp/bin/python -B运行两份代码。脚本拒绝覆盖核心产物；评估函数和关联来源文件固定为已核验的原文件。正式指标不使用GT指导的同分优先排序。
''')
with (R/'REPORT.md').open('x') as f:
    f.write(''.join(lines))
hashes=json.loads((R/'input_hashes.json').read_text())
check('all_original_inputs_unchanged',all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in hashes.items()))
check('no_torch_imported','torch' not in sys.modules)
save('verification.json',{'passed':all(checks.values()),'checks':checks,'source_validation':J['validation']})
save('COMPLETION.json',{'status':'P3-C1_COMPLETE','report':str(R/'REPORT.md'),'statistics_and_interpretation_complete':True,
                        'verification_passed':all(checks.values()),'PPU_used':False,
                        'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
with (R/'ARTIFACTS.sha256').open('x') as f:
    for p in sorted(R.iterdir()):
        if p.is_file() and p.name!='ARTIFACTS.sha256':
            f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(json.dumps({'status':'P3-C1_COMPLETE','report':str(R/'REPORT.md'),'verification_checks':len(checks),
                  'interpretation':interpretation},ensure_ascii=False,indent=2))
