"""Finalize P3-B, retaining all initial outputs and documenting interpretation corrections."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lrcp_p3b_matplotlib')
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
assert not (R / 'REPORT_FINAL.md').exists()
summary = json.loads((R / 'summary.json').read_text())
interpret = json.loads((R / 'interpretation.json').read_text())
verification = json.loads((R / 'verification.json').read_text())
rows = list(csv.DictReader((R / 'global_ranking_pr.csv').open()))
bounds = {}


def ap_from_labels(labels):
    labels = np.asarray(labels, dtype=int)
    rec = np.cumsum(labels) / 8164
    prec = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    envelope = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0., rec]) * envelope))


for a in ['0.0', '1.0']:
    for key in ['50', '70']:
        rr = [r for r in rows if r['alpha'] == a and r['iou_threshold'] == str(int(key)/100)]
        scores = np.asarray([float(r['score']) for r in rr])
        labels = np.asarray([int(r['tp']) for r in rr])
        best, worst = [], []
        for sc in np.unique(scores)[::-1]:
            lab = labels[scores == sc]
            best.extend(sorted(lab, reverse=True))
            worst.extend(sorted(lab))
        lo, hi = ap_from_labels(worst), ap_from_labels(best)
        original = summary['metrics'][a][key]['ap']
        assert lo - 1e-12 <= original <= hi + 1e-12
        bounds[a + ':' + key] = {'AP_all_FP_first_within_equal_score': lo,
                                  'AP_all_TP_first_within_equal_score': hi,
                                  'original_AP': original, 'width': hi-lo,
                                  **interpret['score_ties'][a + ':' + key]}

examples = interpret['representative_examples']
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, g in zip(axes, examples):
    sid, gi = g['sample_idx'], int(g['gt_idx'])
    missing = []
    for a, color in [('0.0', '#2166ac'), ('1.0', '#b2182b')]:
        suffix = str(int(float(a)))
        with np.load(Path('/workspace/lrcp_d2/P3A_20261014_120929_boxes') / f'sample_{sid}_alpha_{a}.npz', allow_pickle=False) as z:
            gt = z['gt_boxes'][gi]
            if float(g['best_iou_a' + suffix]) <= 0:
                missing.append('alpha=' + a + ': no overlapping saved box')
                continue
            pi = int(g['best_pred_idx_a' + suffix])
            pred = z['pred_boxes'][pi]
            polygon = np.r_[pred[:4, :2], pred[:1, :2]]
            ax.plot(polygon[:, 0], polygon[:, 1], color=color,
                    label=f'alpha={a}: IoU={float(g["best_iou_a"+suffix]):.3f}, score={float(g["best_pred_score_a"+suffix]):.3f}')
    polygon = np.r_[gt[:4, :2], gt[:1, :2]]
    ax.plot(polygon[:, 0], polygon[:, 1], '--', color='black', label='Same GT')
    center = gt[:4, :2].mean(0)
    radius = max(3., float(np.ptp(gt[:4, :2], axis=0).max()) / 2 + 1.)
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           xlabel='BEV x (m)', ylabel='BEV y (m)', title=f'sample_idx={sid}, GT index={gi}')
    if missing:
        ax.text(.03, .95, '\n'.join(missing), transform=ax.transAxes, va='top', fontsize=8)
    ax.set_aspect('equal'); ax.legend(loc='lower left', fontsize=8); ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(R / ('representative_bev_final.' + ext), dpi=200, bbox_inches='tight')
plt.close(fig)

width = max(x['width'] for x in bounds.values())
text = (R / 'REPORT.md').read_text()
old = '本次并列分数组内没有混合TP/FP标签，因此并列排序不是当前差值的主要解释。'
assert old in text
text = text.replace(old, f'进一步核验发现，AP70两条件分别有1/3组TP/FP混合并列分数，涉及2/6个预测。保留原排序作为正式结果；在每个同分组内分别把TP或FP优先排列，四个AP的最大上下界宽度为{width:.12g}。因此并列分数确实存在，但该极端重排幅度远小于观察到的AP70差值；详见tie_order_bounds.json。')
g = examples[1]
old_row = f'| {g["sample_idx"]} | {g["gt_idx"]} | {float(g["best_iou_a0"]):.5f}→{float(g["best_iou_a1"]):.5f} | {float(g["best_pred_score_a0"]):.5f}→{float(g["best_pred_score_a1"]):.5f} | AP50候选覆盖恢复 |'
new_row = f'| {g["sample_idx"]} | {g["gt_idx"]} | {float(g["best_iou_a0"]):.5f}→{float(g["best_iou_a1"]):.5f} | 无重叠候选→{float(g["best_pred_score_a1"]):.5f} | AP50候选覆盖恢复 |'
assert old_row in text
text = text.replace(old_row, new_row).replace('](representative_bev.png)', '](representative_bev_final.png)')
text = text.replace('依次执行 analyze.py、interpret_results.py 即可重现。', '依次执行 analyze.py、interpret_results.py、finalize.py 即可重现最终报告。')
text = text.replace('在全新的 /workspace/lrcp_d2/codex_reports/P3B_<new_run_id>/ 目录复制两份代码', '在全新的 /workspace/lrcp_d2/codex_reports/P3B_<new_run_id>/ 目录复制三份代码')
text += '''
## 最终解释校正及交付入口

本文件REPORT_FINAL.md为最终报告；REPORT.md及初始代表图保留为分析过程记录。最终化程序修正了初版对并列分数的过强表述，并给出同分组极端重排的AP上下界；未改变正式AP、TP/FP、GT迁移或原始输入。completion.json为初次统计完成记录，FINAL_COMPLETION.json指向最终交付。

最佳IoU=0时，per_gt.csv中的best_pred_idx由argmax同值规则返回第一框，其分数、中心/yaw距离只是占位候选，不表示该GT被一个高分框检测。不能用这类占位候选推断定位质量；正文采用两条件IoU均≥0.5的7050个共同GT作中心/yaw对照。最终代表图对无重叠情况明确标注，不显示无关框的分数。原始明细保留以便重现argmax规则。

下一步分数交换诊断必须产生非单调的相对排序变化。对固定框统一做严格单调分数变换不会改变排序AP，不能把单纯温度缩放视作排序机制的有效验证。

finalize.py和tie_order_bounds.json记录以上校正；ARTIFACTS_FINAL.sha256覆盖最终交付文件。原始实验目录、PPU任务和既有P3-A结果均未改动。
'''
with (R / 'REPORT_FINAL.md').open('x') as f:
    f.write(text)
with (R / 'tie_order_bounds.json').open('x') as f:
    json.dump(bounds, f, ensure_ascii=False, indent=2)
assert summary['validation']['passed'] and verification['passed'] and 'torch' not in sys.modules
with (R / 'FINAL_COMPLETION.json').open('x') as f:
    json.dump({'status': 'P3-B_COMPLETE', 'statistics_verified': True, 'interpretation_completed': True,
               'final_report': str(R / 'REPORT_FINAL.md'),
               'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'original_inputs_unchanged': verification['checks']['inputs_unchanged_after_analysis'],
               'tie_sensitivity_computed': True, 'PPU_used': False}, f, ensure_ascii=False, indent=2)
with (R / 'ARTIFACTS_FINAL.sha256').open('x') as f:
    for p in sorted(R.iterdir()):
        if p.is_file() and p.name != 'ARTIFACTS_FINAL.sha256':
            f.write(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n')
print(json.dumps({'status': 'P3-B_COMPLETE', 'report': str(R / 'REPORT_FINAL.md'),
                  'tie_order_bounds': bounds, 'max_tie_AP_width': width}, ensure_ascii=False, indent=2))
