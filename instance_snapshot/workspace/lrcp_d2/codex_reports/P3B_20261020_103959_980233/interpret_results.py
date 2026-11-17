"""Derive P3-B interpretation tables/figures/report from analyze.py outputs.

No model execution; no imports from analyze.py (which has execution side effects).
Run once after analyze.py in the same fresh P3B directory.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lrcp_p3b_matplotlib')
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
B = Path('/workspace/lrcp_d2')
assert not (R / 'REPORT.md').exists()
J = json.loads((R / 'summary.json').read_text())
GT = list(csv.DictReader((R / 'per_gt.csv').open()))
P = list(csv.DictReader((R / 'per_prediction.csv').open()))
PR = list(csv.DictReader((R / 'global_ranking_pr.csv').open()))
S = list(csv.DictReader((R / 'per_sample.csv').open()))
checks = {}


def check(name, ok):
    checks[name] = bool(ok)
    if not ok:
        raise AssertionError(name)


def write_csv(name, rows):
    with (R / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_json(name, value):
    with (R / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def dist(x):
    return {'n': len(x), 'mean': float(np.mean(x)), 'median': float(np.median(x))} if len(x) else {'n': 0}


pred_map = {(p['sample_idx'], p['alpha'], int(p['pred_idx'])): p for p in P}
gt_map = {(p['sample_idx'], int(p['gt_idx'])): p for p in GT}
check('unique_prediction_trace', len(pred_map) == len(P) == 16713)
check('unique_GT_trace', len(gt_map) == len(GT) == 8164)
check('sample_trace_910', len(S) == len({p['sample_idx'] for p in S}) == 910)
for a in ['0.0', '1.0']:
    suffix = str(int(float(a)))
    for key in ['50', '70']:
        gt_assigned = 0
        for g in GT:
            pi = int(g['matched_pred_idx' + key + '_a' + suffix])
            if pi >= 0:
                p = pred_map[g['sample_idx'], a, pi]
                assert p['tp' + key] == '1' and int(p['matched_gt_idx' + key]) == int(g['gt_idx'])
                gt_assigned += 1
        for p in P:
            if p['alpha'] != a or p['tp' + key] != '1':
                continue
            g = gt_map[p['sample_idx'], int(p['matched_gt_idx' + key])]
            assert int(g['matched_pred_idx' + key + '_a' + suffix]) == int(p['pred_idx'])
        check('bidirectional_GT_prediction_trace:' + a + ':' + key, gt_assigned == J['metrics'][a][key]['tp'])
        rows = [r for r in PR if r['alpha'] == a and r['iou_threshold'] == str(int(key)/100)]
        scores = [float(r['score']) for r in rows]
        check('descending_global_score:' + a + ':' + key, all(x >= y for x, y in zip(scores, scores[1:])))
        tp = 0
        rec, prec = [], []
        for rank, row in enumerate(rows, 1):
            p = pred_map[row['sample_idx'], a, int(row['pred_idx'])]
            assert p['tp' + key] == row['tp'] and float(p['score']) == float(row['score'])
            assert int(p['matched_gt_idx' + key]) == int(row['matched_gt_idx'])
            tp += int(row['tp'])
            assert int(row['cumulative_tp']) == tp and int(row['cumulative_fp']) == rank - tp
            rec.append(tp / 8164)
            prec.append(tp / rank)
        # Independent integration from the exported trace, without the repository evaluator.
        envelope = np.maximum.accumulate(np.asarray(prec)[::-1])[::-1]
        ap = float(np.sum(np.diff(np.r_[0., rec]) * envelope))
        check('CSV_AP:' + a + ':' + key, abs(ap - J['metrics'][a][key]['ap']) <= 1e-12)
        check('CSV_counts:' + a + ':' + key, tp == J['metrics'][a][key]['tp'] and len(rows)-tp == J['metrics'][a][key]['fp'])

common = [g for g in GT if float(g['best_iou_a0']) >= .5 and float(g['best_iou_a1']) >= .5]
details = {'common_7050_geometry': {a: dist([float(g['best_iou_a' + a]) for g in common]) for a in ['0', '1']},
           'mid_IoU_scores': {}, 'pairing_sensitivity': [], 'score_ties': {}}
for a in ['0.0', '1.0']:
    rr = [p for p in P if p['alpha'] == a and .5 <= float(p['best_gt_iou']) < .7]
    details['mid_IoU_scores'][a] = {**dist([float(p['score']) for p in rr]),
                                    'scores_ge_0.8': sum(float(p['score']) >= .8 for p in rr)}
    rr = [p for p in P if p['alpha'] == a]
    scores = np.asarray([float(p['score']) for p in rr])
    for key in ['50', '70']:
        mixed_groups, involved = 0, 0
        labels = np.asarray([int(p['tp' + key]) for p in rr])
        for sc, count in zip(*np.unique(scores, return_counts=True)):
            if count < 2:
                continue
            ls = labels[scores == sc]
            if ls.min() != ls.max():
                mixed_groups += 1; involved += count
        details['score_ties'][a + ':' + key] = {'mixed_TP_FP_tie_groups': mixed_groups, 'involved_predictions': int(involved)}
for threshold, values in J['cross_condition_association'].items():
    u0, u1 = values['0.0']['unpaired'], values['1.0']['unpaired']
    p0, p1 = values['0.0']['paired'], values['1.0']['paired']
    details['pairing_sensitivity'].append({
        'pair_IoU_threshold': float(threshold), 'paired_each': p0['n'],
        'unpaired_a0': u0['n'], 'unpaired_a1': u1['n'],
        'unpaired_tp50_a0': u0['tp50'], 'unpaired_tp50_a1': u1['tp50'],
        'unpaired_tp70_a0': u0['tp70'], 'unpaired_tp70_a1': u1['tp70'],
        'unpaired_fp70_a0': u0['n'] - u0['tp70'], 'unpaired_fp70_a1': u1['n'] - u1['tp70'],
        'paired_delta_tp70': p1['tp70'] - p0['tp70'],
        'unpaired_delta_tp70': u1['tp70'] - u0['tp70'],
        'unpaired_score_median_a0': u0['scores']['q50'], 'unpaired_score_median_a1': u1['scores']['q50'],
    })
write_csv('pairing_sensitivity_summary.csv', details['pairing_sensitivity'])

common_pr = []
for key in ['50', '70']:
    for a in ['0.0', '1.0']:
        rows = [r for r in PR if r['alpha'] == a and r['iou_threshold'] == str(int(key)/100)]
        rec = np.asarray([float(r['recall']) for r in rows])
        pp = np.asarray([float(r['interpolated_precision']) for r in rows])
        for x in [.2, .4, .6, .7, .75, .8]:
            good = rec >= x
            common_pr.append({'IoU_threshold': int(key)/100, 'alpha': a, 'target_recall': x,
                              'interpolated_precision': float(pp[good][0]) if good.any() else '',
                              'first_achieving_rank': int(np.argmax(good)) + 1 if good.any() else -1})
write_csv('pr_at_common_recall.csv', common_pr)
details['precision_at_common_recall'] = common_pr
metric_rows = []
for key in ['50', '70']:
    for a in ['0.0', '1.0']:
        m = J['metrics'][a][key]
        metric_rows.append({'IoU_threshold': int(key)/100, 'alpha': a,
                            **{k: m[k] for k in ['ap', 'tp', 'fp', 'fn', 'precision_final', 'recall_final', 'max_cardinality_tp', 'greedy_assignment_gap', 'global_ranking_gap']},
                            'equal_count_AP': m['equal_count_topK']['ap']})
write_csv('metric_summary.csv', metric_rows)

# Representatives chosen by a stated rule, not presented as frequency estimates.
loss = [g for g in GT if g['match_transition70'] == 'TP_to_FN' and .5 <= float(g['best_iou_a1']) < .7
        and float(g['best_pred_score_a1']) >= .8]
loss.sort(key=lambda g: float(g['delta_best_iou']))
gain = [g for g in GT if g['match_transition50'] == 'FN_to_TP']
gain.sort(key=lambda g: float(g['delta_best_iou']), reverse=True)
selected = [dict(loss[0], selection='largest_IoU_loss_with_retained_IoU50_and_score_ge_0.8'),
            dict(gain[0], selection='largest_IoU_gain_in_AP50_recovered_GT')]
write_csv('representative_examples.csv', selected)
details['representative_examples'] = selected
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, g in zip(axes, selected):
    sid, gi = g['sample_idx'], int(g['gt_idx'])
    for a, color in [('0.0', '#2166ac'), ('1.0', '#b2182b')]:
        s = str(int(float(a)))
        with np.load(B / 'P3A_20261014_120929_boxes' / f'sample_{sid}_alpha_{a}.npz', allow_pickle=False) as z:
            gt = z['gt_boxes'][gi]
            pi = int(g['best_pred_idx_a' + s]); pred = z['pred_boxes'][pi]
            polygon = np.r_[pred[:4, :2], pred[:1, :2]]
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, label=f'alpha={a}: IoU={float(g["best_iou_a"+s]):.3f}, score={float(g["best_pred_score_a"+s]):.3f}')
    polygon = np.r_[gt[:4, :2], gt[:1, :2]]
    ax.plot(polygon[:, 0], polygon[:, 1], '--', color='black', label='Same GT')
    center = gt[:4, :2].mean(0)
    radius = max(3., float(np.ptp(gt[:4, :2], axis=0).max()) / 2 + 1.)
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           xlabel='BEV x (m)', ylabel='BEV y (m)', title=f'sample_idx={sid}, GT index={gi}')
    ax.set_aspect('equal'); ax.legend(loc='lower left', fontsize=8); ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(R / ('representative_bev.' + ext), dpi=200, bbox_inches='tight')
plt.close(fig)

# Make the causal boundaries explicit before producing the narrative report.
details['supported_mechanisms'] = [
    '低阈值候选覆盖增加，同时部分原高IoU候选跌入0.5–0.7区间；平均IoU上升不能排除高阈值质量下降。',
    '当前保存框的最大基数TP等于原贪心TP；本数据没有可由改变GT匹配顺序恢复的额外TP。',
    'FP70数量增加且更多FP具有较高置信度，全局排序相对理想TP优先排序的AP差距增大。',
    '预测数量净差不是独立新增框集合；辅助关联显示两侧均有未配对框。',
    '等数量topK诊断未消除AP70下降，但该筛选依赖分数，不能给出纯数量因果贡献。',
]
details['unresolved_factors'] = [
    '缺少pre-NMS候选及被阈值删除的框，不能区分回归/偏移变化、NMS选择及检测阈值效应。',
    '缺少逐样本offset预测、offset GT及中间BEV特征，不能判定MetaFlow偏移本身错误或过补偿。',
    '排序损失是对当前TP/FP标签的描述性差距；标签会因几何变化而改变，并非纯分数变动贡献。',
    '跨条件预测关联依赖IoU阈值；GT实例在相邻帧之间相关，未估计跨场景置信区间。',
    '当前结论限定单checkpoint、k=3、公共910样本及保存的后处理框，不能外推所有延迟。',
]
details['next_minimal_validation'] = {
    'type': 'CPU-only, proposed; not executed in P3-B',
    'intervention': '固定每个条件的全部保存框及数量；只对明确辅助配对的框交换两条件置信度，未配对框分数保持，再按原LRCP协议重算AP。',
    'controls': '对称交换两个方向；辅助配对IoU阈值0.5/0.7/0.9敏感性；不交换分数的恒等对照。',
    'acceptance': '恒等对照精确恢复本报告AP；每个条件框数组、数量及GT哈希不变；记录每份分数替换的sample_idx/pred_idx；比较AP70排序差距是否缩小。',
    'limits': '只检验post-NMS固定框上的分数排序机制，不证明offset错误，也不排除pre-NMS选择因素；无需PPU。'}
write_json('interpretation.json', details)

def f(x, n=6):
    return f'{x:.{n}f}'

lines = [f'''# P3-B：BEV 检测误差离线分析

完成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}（Asia/Shanghai）。

**结论：P3-B已完成误差统计、机制解释及可追溯性验收。** 当前实验为LRCP协同3D目标检测的BEV评估，k=3，模型内部时间编码300，物理观测间隔约600 ms；α=0/1使用公共910样本。8164表示GT实例数，不是跨帧唯一车辆数。未执行训练或PPU推理，未修改原始数据、模型、checkpoint或既有结果。

## 主要数值发现及验收

从保存框重新执行仓库原函数，与P3-A JSON及既有审计结果对照，四个AP误差均为0，TP/FP/GT精确一致；instrumented matcher的完整score/TP/FP列表与原函数完全相同。CSV全局排名再次独立积分恢复AP，逐GT与逐预测的匹配引用已双向核验。

| α | 预测框数 | AP50 | TP50 / FP50 | AP70 | TP70 / FP70 |
|---|---:|---:|---:|---:|---:|
| 0 | 8175 | 0.8665290893 | 7111 / 1064 | 0.8071592299 | 6754 / 1421 |
| 1 | 8538 | 0.8768414234 | 7209 / 1329 | 0.7854230204 | 6645 / 1893 |
| 差值 | +363 | +0.0103123342 | +98 / +265 | −0.0217362095 | −109 / +472 |

AP在0–1尺度；AP50上升1.0312个百分点，AP70下降2.1736个百分点。最终阈值precision/recall分别为：AP50条件α0为0.86985/0.87102，α1为0.84434/0.88302；AP70条件α0为0.82618/0.82729，α1为0.77829/0.81394。最终precision不是AP。

## 同一GT的最佳IoU及匹配迁移

每GT取本条件保存预测框的最大BEV IoU，不要求两条件预测一一对应。GT按 `(sample_idx, gt_idx)` 固定配对，gt_idx为NPZ零起始行号。最佳候选可被多个GT共享，因此候选可用性通常不等于实际TP；本数据另行验证如下。

| 原α0最佳IoU → α1最佳IoU | <0.5 | 0.5–0.7 | ≥0.7 |
|---|---:|---:|---:|
| <0.5 | 894 | 95 | 64 |
| 0.5–0.7 | 24 | 228 | 105 |
| ≥0.7 | 37 | 241 | 6476 |

| 阈值 | TP→TP | FN→TP | TP→FN | FN→FN | 净TP变化 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 7050 | 159 | 61 | 894 | +98 |
| 0.7 | 6476 | 169 | 278 | 1241 | −109 |

两阈值下，GT候选跨阈值变化与实际贪心TP/FN迁移逐GT一致。两条件最大基数匹配的TP数均等于原贪心TP数，贪心竞争缺口为0；全部FP的最佳GT IoU均未达到当前阈值，没有“存在可达阈值GT却仅因贪心竞争未匹配”的FP。**因此当前保存框的TP70损失不能通过改变GT匹配优先级恢复。** FP属于预测框；不把GT的FN称为GT的FP。

278个失去TP70的GT中，241个（{241/278*100:.2f}%）仍有IoU≥0.5框，仅37个跌至0.5以下；169个恢复TP70的GT中，105个来自0.5–0.7、64个来自0.5以下。这支持“低阈值覆盖收益与高阈值定位损失同时存在”，不支持将所有变化概括为统一改善或统一恶化。

全部GT最佳IoU均值由0.759597升至0.763069（+0.003472），中位数由0.864356降至0.861466；以1e-6为变化容差，3180个提升、3658个下降、1326个近似不变。均值受少量大幅恢复和漏检尾部影响，不能用它否定AP70质量下降。

在两条件都达到IoU0.5的7050个共同GT上，最佳IoU均值为{f(details['common_7050_geometry']['0']['mean'])}→{f(details['common_7050_geometry']['1']['mean'])}；中心误差均值为0.17625→0.18816 m，中位数0.12026→0.13661 m。yaw绝对误差均值约1.2956°→1.3401°。这是最佳保存候选的几何描述，不是offset EPE，也不能单独定位误差来源。

![同GT最佳IoU与阈值迁移](gt_geometry_transitions.png)

## 数量变化、363净差及跨条件辅助关联

234个样本预测数增加、55个减少、621个相同；逐样本正差合计424、负差合计61，净差363。它不是363个具有确定身份的新框。预测质量区间数量如下：

| 预测框最佳GT IoU | α0 | α1 | 净差 |
|---|---:|---:|---:|
| <0.5 | 1064 | 1329 | +265 |
| 0.5–0.7 | 357 | 564 | +207 |
| ≥0.7 | 6754 | 6645 | −109 |

这解释了阈值统计恒等式：AP50下+98 TP和+265 FP；AP70下−109 TP和+472 FP，其中FP70净增472=低IoU框净增265+中间区间框净增207。这里是条件间总体数量变化，不能将207个框直接称为同一批框降级。

同样本内显式做预测-预测辅助一对一关联，先最大化达到IoU阈值的配对数，再最大化配对IoU。此规则并不赋予自然身份；“未配对”不等于真实新增/消失目标。

| 辅助关联IoU阈值 | 两侧配对数 | α0未配对 | α1未配对 | 配对组ΔTP70 | 未配对组ΔTP70 |
|---|---:|---:|---:|---:|---:|
''']
for r in details['pairing_sensitivity']:
    lines.append(f'| {r["pair_IoU_threshold"]} | {r["paired_each"]} | {r["unpaired_a0"]} | {r["unpaired_a1"]} | {r["paired_delta_tp70"]:+d} | {r["unpaired_delta_tp70"]:+d} |\n')
lines.append('''
以辅助IoU0.5为例，α1未配对664框中TP50=152、TP70=62，FP70=602；α0未配对301框中TP50=60、TP70=32，FP70=269。未配对组净贡献TP70 +30、FP70 +333；配对组TP70 −139、FP70 +139，两组相加恢复总变化−109/+472。α1未配对框分数中位数0.4899，α0为0.4233。未配对框主要是FP，但既包含有效新增候选也包含几何变化过大而失配的框，不能只称“新增噪声”。更严格关联阈值会改变分组，见完整敏感性CSV。

### 每样本等数量诊断

设每样本K=min(N0,N1)，各条件保留最高分K框并重匹配，合计各8114框：

| 条件 | α0 AP | α1 AP | 差值 |
|---|---:|---:|---:|
| 原始AP50 | 0.8665291 | 0.8768414 | +0.0103123 |
| 等数量AP50 | 0.8660039 | 0.8670278 | +0.0010239 |
| 原始AP70 | 0.8071592 | 0.7854230 | −0.0217362 |
| 等数量AP70 | 0.8069666 | 0.7791826 | −0.0277840 |

数量控制后AP70下降仍存在，AP50收益显著缩小。这个topK筛选依赖分数并改变候选集合，不能据差值声称已精确分离数量因果贡献。

## PR曲线与全局分数排序

![原LRCP PR曲线](pr_curves.png)

AP50曲线主要在可达recall末端拉开；AP70在相同recall的中高段也存在precision差距。global_ranking_pr.csv保留两阈值、两条件全部33426个排名点及样本/框/GT身份；pr_at_common_recall.csv给出相同recall的插值precision。并列分数沿用原np.argsort协议；本次并列分数组内没有混合TP/FP标签，因此并列排序不是当前差值的主要解释。

| 全局前K框，IoU0.7 | α0 TP/FP | α1 TP/FP |
|---|---:|---:|
| K=4000 | 3917 / 83 | 3892 / 108 |
| K=6000 | 5738 / 262 | 5621 / 379 |
| K=7500 | 6665 / 835 | 6505 / 995 |

固定score≥0.8时，α0/1的FP70为213/316（+103）；其中IoU0.5–0.7的框分别197/293，贡献+96。全部FP70分数均值0.4514→0.4894，中位数0.3660→0.4110。中间IoU区间的框在AP50中是TP，在AP70中转为FP，同时可能保留较高置信度，从而影响全局PR。**这不是所有预测分数普遍下降，也不是仅增加低分尾部框。**

![FP置信度分布](fp_confidence.png)

### 可核算的描述性AP恒等式

令Rmax为阈值图最大基数匹配TP/GT，Rgreedy为原贪心TP/GT；Lgreedy=Rmax−Rgreedy，Lrank=Rgreedy−AP：

`AP = Rmax - Lgreedy - Lrank`

当前Lgreedy=0。AP70可达recall由0.8272905降至0.8139392，变化−0.0133513；相对“当前TP全部排在FP前”的排序差距由0.0201313增至0.0285162，扩大0.0083849。因此：

`ΔAP70 = −0.0133513 − 0.0083849 = −0.0217362`

AP50对应：`ΔAP50 = +0.0120039 − 0.0016916 = +0.0103123`。

这是描述性恒等式，不是61%几何/39%分数的因果贡献分配：Rmax同时受框数量与几何/选择影响，而Lrank的TP/FP标签本身会随几何变化。它说明AP70损失既有当前候选集合可达recall降低，也有既定标签下的排序差距扩大。

## 可追溯实例

以下只作可核查例子，不代表发生频率；选择规则与完整行保存在representative_examples.csv。图中两条件的框分别为该同一GT的最佳候选，不假定跨条件框的自然身份。

| sample_idx | GT索引 | 最佳IoU α0→α1 | 最佳框分数 α0→α1 | 解释 |
|---|---:|---|---|---|
''')
for i, g in enumerate(selected):
    lines.append(f'| {g["sample_idx"]} | {g["gt_idx"]} | {float(g["best_iou_a0"]):.5f}→{float(g["best_iou_a1"]):.5f} | {float(g["best_pred_score_a0"]):.5f}→{float(g["best_pred_score_a1"]):.5f} | '+('保留AP50而失去AP70，仍有高分' if i == 0 else 'AP50候选覆盖恢复')+' |\n')
lines.append('''
![可追溯BEV实例](representative_bev.png)

## 支持的机制解释与未排除因素

现有数据支持：补偿条件增加部分原漏检GT的低阈值可用候选；同时一部分高IoU候选跨过0.7边界向下，且中间IoU框可能保留高分。当前保存框中没有GT匹配竞争造成的额外TP缺口；候选覆盖、几何阈值迁移及全局排序共同描述了AP50上升/AP70下降的现象。

尚不能判定MetaFlow偏移预测错误或过补偿。保存结果缺少pre-NMS候选、被score阈值删除的框、逐样本offset预测及offset GT、中间BEV特征；回归、特征融合、NMS和阈值选择仍可能共同作用。所有关联结论受辅助配对规则限制；未估计跨场景置信区间，时序GT实例不独立。结论只适用于当前checkpoint、k=3和公共910集合。

## 下一步最小验证实验（仅建议，未执行）

无需PPU的固定框分数交换诊断：固定每个条件全部保存框、GT和数量，仅对辅助一对一配对框交换两条件置信度，未配对分数保持，再按原LRCP协议重算AP。双向交换，并对辅助IoU0.5/0.7/0.9做敏感性检查；恒等对照必须精确恢复本报告AP，所有框/GT哈希和数量不变，每次交换记录sample_idx/pred_idx。

若仅换分数能缩小AP70差距，支持post-NMS固定框上的排序机制；若差距主要保留，则需重点检查候选几何/选择。该实验仍不能证明offset错误或排除pre-NMS NMS因素。当前P3-B无需启动PPU，也没有待启动的PPU命令。

## 产物、复现与验收标准

- analyze.py：完整CPU统计、原函数与instrumented matcher一致性检查、主图生成。
- interpret_results.py：从CSV独立复核AP/索引、生成解释表格、BEV实例和本报告。
- METHODS.md：GT身份、匹配协议、辅助配对、等数量诊断、AP恒等式与限制。
- summary.json、interpretation.json、validation.json、verification.json：汇总、机制解释及验收证据。
- per_gt.csv（8164行）、per_prediction.csv（16713行）、per_sample.csv（910行）：逐实例可追溯明细。
- global_ranking_pr.csv（33426行）、operating_points.csv、pr_at_common_recall.csv、confidence_bins.csv：PR/排序/分数统计。
- cross_condition_pairs.csv、cross_condition_groups.csv、pairing_sensitivity_summary.csv：不同关联阈值的配对/未配对记录。
- gt_transition_examples.csv、representative_examples.csv：GT迁移例子，索引可回到原NPZ。
- metric_summary.csv：便于复用的指标表；PNG与PDF均为独立导出图表。
- input_hashes.json：分析实际读取输入的SHA256；execution.log：CPU运行日志；ARTIFACTS.sha256：本目录交付文件哈希。

在全新的 /workspace/lrcp_d2/codex_reports/P3B_<new_run_id>/ 目录复制两份代码，使用 /workspace/venvs/lrcp/bin/python -B 依次执行 analyze.py、interpret_results.py 即可重现。脚本拒绝覆盖现有核心结果；原审计脚本只作为协议来源读取，不执行其P3-A目录审计代码。解释器、库版本、时间和固定输入哈希均已记录。无需联网、安装依赖或使用PPU。
''')
with (R / 'REPORT.md').open('x') as h:
    h.write(''.join(lines))

inputs = json.loads((R / 'input_hashes.json').read_text())
check('inputs_unchanged_after_analysis', all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in inputs.items()))
check('no_torch_imported', 'torch' not in sys.modules)
for p in R.glob('*.png'):
    check('PNG:' + p.name, p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
for p in R.glob('*.pdf'):
    check('PDF:' + p.name, p.read_bytes().startswith(b'%PDF'))
write_json('verification.json', {'passed': all(checks.values()), 'checks': checks,
                                'source_validation_passed': J['validation']['passed'],
                                'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
write_json('completion.json', {'status': 'P3-B_COMPLETE', 'error_statistics_complete': True,
                               'mechanism_interpretation_complete': True, 'verification_passed': all(checks.values()),
                               'report': str(R / 'REPORT.md'), 'PPU_inference_started': False,
                               'torch_imported': 'torch' in sys.modules})
with (R / 'ARTIFACTS.sha256').open('x') as h:
    for p in sorted(R.iterdir()):
        if p.is_file() and p.name != 'ARTIFACTS.sha256':
            h.write(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n')
print(json.dumps({'run': str(R), 'verification_passed': all(checks.values()), 'details': details}, ensure_ascii=False, indent=2))
