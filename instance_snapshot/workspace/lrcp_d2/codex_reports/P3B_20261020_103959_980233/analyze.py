"""P3-B: saved-box CPU analysis. No torch import or model execution.

Inputs are read-only. Execute once in a fresh P3B_<run_id> directory with:
/workspace/venvs/lrcp/bin/python -B /absolute/path/to/analyze.py
The presence of summary.json prevents accidental re-execution into this directory.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lrcp_p3b_matplotlib')
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import shapely
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent
BASE = Path('/workspace/lrcp_d2')
REPO = Path('/workspace/LRCP')
REFERENCE = BASE / 'codex_reports/audit_20261018_215046_447565/audit_results.json'
ALPHAS = ['0.0', '1.0']
THRESHOLDS = [0.5, 0.7]
assert RUN.name.startswith('P3B_') and RUN.parent == BASE / 'codex_reports'
assert not (RUN / 'summary.json').exists(), 'Use a new run directory'
LOG = (RUN / 'execution.log').open('x', buffering=1)
HASHES = {}
CHECKS = {}


def log(s):
    print(s, flush=True)
    LOG.write(str(s) + '\n')


def read(path):
    data = Path(path).read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def read_json(path):
    return json.loads(read(path))


def write_json(name, obj):
    with (RUN / name).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(name, rows):
    assert rows
    with (RUN / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check(name, ok):
    CHECKS[name] = bool(ok)
    if not ok:
        raise AssertionError(name)


def extract(path, names, env):
    tree = ast.parse(read(path).decode(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def distribution(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'mean': float(x.mean()), 'min': float(x.min()), 'max': float(x.max()),
            **{f'q{q:02d}': float(np.percentile(x, q)) for q in [10, 25, 50, 75, 90, 95, 99]}}


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return np.stack([GEO['compute_iou'](p, b) for p in a])


def greedy(matrix, scores, threshold):
    """Instrumented repository protocol, preserving float32 IoUs and argsort ties."""
    n, m = matrix.shape
    order = np.argsort(-scores)
    remaining = list(range(m))
    matched = np.full(n, -1, dtype=int)
    gt_pred = np.full(m, -1, dtype=int)
    for pi in order:
        if not remaining:
            continue
        overlaps = matrix[pi, remaining]
        if np.max(overlaps) < threshold:
            continue
        pos = int(np.argmax(overlaps))
        gi = remaining.pop(pos)
        matched[pi] = gi
        gt_pred[gi] = pi
    return {'order': order, 'pred_gt': matched, 'gt_pred': gt_pred,
            'tp': (matched >= 0).astype(int)}


def max_cardinality(matrix, threshold):
    if not matrix.size:
        return 0
    feasible = matrix.astype(np.float64) >= threshold
    ii, jj = linear_sum_assignment(-feasible.astype(float))
    return int(feasible[ii, jj].sum())


def new_stats():
    return {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in THRESHOLDS}


def append_stats(stats, match, score, gt_count, t):
    idx = match['order']
    stats[t]['tp'] += match['tp'][idx].tolist()
    stats[t]['fp'] += (1 - match['tp'][idx]).tolist()
    stats[t]['score'] += score[idx].tolist()
    stats[t]['gt'] += gt_count


def geometry_error(pred, gt):
    pc, gc = pred[:4, :2].mean(0), gt[:4, :2].mean(0)
    pe, ge = pred[1, :2] - pred[0, :2], gt[1, :2] - gt[0, :2]
    yaw = float(np.arctan2(pe[1], pe[0]) - np.arctan2(ge[1], ge[0]))
    yaw = abs((yaw + np.pi / 2) % np.pi - np.pi / 2)
    return float(np.linalg.norm(pc - gc)), float(np.degrees(yaw))


GEO = extract(REPO / 'opencood/utils/common_utils.py', ['convert_format', 'compute_iou'],
              {'np': np, 'Polygon': Polygon})
common = SimpleNamespace(convert_format=GEO['convert_format'], compute_iou=GEO['compute_iou'],
                         torch_tensor_to_numpy=lambda x: x)
EVAL = extract(REPO / 'opencood/utils/eval_utils.py', ['voc_ap', 'caluclate_tp_fp', 'calculate_ap'],
               {'np': np, 'common_utils': common})
audit_tree = ast.parse(read(BASE / 'codex_reports/audit_p3a_v2.py').decode())
audit_ref = read_json(REFERENCE)
p3rows = read_json(BASE / 'P3A_20261014_120929.json')
target = next(r for r in p3rows if r['delay_label_ms'] == 300)
manifest = read_json(BASE / 'P1_common910_indices.json')
ids = target['sample_ids']
check('cohort', len(ids) == len(set(ids)) == 910 and set(ids) == set(manifest['sample_ids']))
for p in [REPO / 'opencood/utils/common_utils.py', REPO / 'opencood/utils/eval_utils.py']:
    check('source_unchanged:' + p.name, HASHES[str(p)] == audit_ref['files'][str(p)]['sha256'])

native_stats = {a: new_stats() for a in ALPHAS}
instrumented_stats = {a: new_stats() for a in ALPHAS}
cap_stats = {a: new_stats() for a in ALPHAS}
PRED_ROWS, GT_ROWS, SAMPLE_ROWS, PAIR_ROWS, GROUP_ROWS = [], [], [], [], []
records = {a: {t: [] for t in THRESHOLDS} for a in ALPHAS}
upper_counts = {a: {t: 0 for t in THRESHOLDS} for a in ALPHAS}
cross_groups = {str(t): {a: {'paired': [], 'unpaired': []} for a in ALPHAS} for t in [0.5, 0.7, 0.9]}
log('P3-B CPU analysis started: k=3, model code=300, physical interval~600 ms, 910 paired samples')
for sn, sid in enumerate(ids):
    cases = {}
    for a in ALPHAS:
        path = BASE / 'P3A_20261014_120929_boxes' / f'sample_{sid}_alpha_{a}.npz'
        data = read(path)
        # Check only consumed inputs against the existing audit; no directory audit.
        check('input_hash:' + path.name, HASHES[str(path)] == audit_ref['files'][str(path)]['sha256'])
        with np.load(path, allow_pickle=False) as z:
            pred, score, gt = z['pred_boxes'].copy(), z['pred_scores'].copy(), z['gt_boxes'].copy()
            assert str(z['sample_idx'].item()) == sid and float(z['alpha']) == float(a)
        pp = list(GEO['convert_format'](pred))
        gp = list(GEO['convert_format'](gt))
        matrix = iou_matrix(pp, gp)
        n, m = matrix.shape
        best_p = np.argmax(matrix, axis=0) if n else np.full(m, -1)
        best_gt = np.argmax(matrix, axis=1) if m else np.full(n, -1)
        best_iou_gt = matrix.max(axis=0) if n else np.zeros(m)
        best_iou_pred = matrix.max(axis=1) if m else np.zeros(n)
        matches = {t: greedy(matrix, score, t) for t in THRESHOLDS}
        max_tp = {t: max_cardinality(matrix, t) for t in THRESHOLDS}
        for t in THRESHOLDS:
            EVAL['caluclate_tp_fp'](pred, score, gt, native_stats[a], t)
            append_stats(instrumented_stats[a], matches[t], score, m, t)
            upper_counts[a][t] += max_tp[t]
            assert max_tp[t] >= int(matches[t]['tp'].sum())
        cases[a] = {'pred': pred, 'score': score, 'gt': gt, 'pp': pp, 'gp': gp,
                    'matrix': matrix, 'best_p': best_p, 'best_gt': best_gt,
                    'best_iou_gt': best_iou_gt, 'best_iou_pred': best_iou_pred,
                    'matches': matches, 'max_tp': max_tp}
        for pi in range(n):
            gi = int(best_gt[pi]) if best_iou_pred[pi] > 0 else -1
            row = {'sample_idx': sid, 'alpha': a, 'pred_idx': pi, 'score': float(score[pi]),
                   'best_gt_idx': gi, 'best_gt_iou': float(best_iou_pred[pi]),
                   'iou_bin': '<0.5' if best_iou_pred[pi] < .5 else ('0.5-0.7' if best_iou_pred[pi] < .7 else '>=0.7')}
            for t in THRESHOLDS:
                suffix = str(int(t * 100))
                match = matches[t]
                mg = int(match['pred_gt'][pi])
                row['tp' + suffix] = int(mg >= 0)
                row['matched_gt_idx' + suffix] = mg
                row['matched_iou' + suffix] = float(matrix[pi, mg]) if mg >= 0 else ''
                reason = 'TP' if mg >= 0 else ('no_GT' if not m else ('duplicate_or_competition' if best_iou_pred[pi] >= t else 'below_IoU_threshold'))
                row['status' + suffix] = reason
            PRED_ROWS.append(row)
        for t in THRESHOLDS:
            for pi in matches[t]['order']:
                records[a][t].append({'sample_idx': sid, 'pred_idx': int(pi), 'score': float(score[pi]),
                                       'tp': int(matches[t]['tp'][pi]),
                                       'matched_gt_idx': int(matches[t]['pred_gt'][pi]),
                                       'best_gt_iou': float(best_iou_pred[pi])})
    c0, c1 = cases['0.0'], cases['1.0']
    assert np.array_equal(c0['gt'], c1['gt'])
    m = len(c0['gt'])
    for gi in range(m):
        row = {'sample_idx': sid, 'gt_idx': gi,
               'gt_center_x': float(c0['gt'][gi, :4, 0].mean()),
               'gt_center_y': float(c0['gt'][gi, :4, 1].mean())}
        for a, c in cases.items():
            suffix = str(int(float(a)))
            bp = int(c['best_p'][gi])
            row['best_iou_a' + suffix] = float(c['best_iou_gt'][gi])
            row['best_pred_idx_a' + suffix] = bp
            row['best_pred_score_a' + suffix] = float(c['score'][bp]) if bp >= 0 else ''
            ce, ye = geometry_error(c['pred'][bp], c['gt'][gi]) if bp >= 0 else ('', '')
            row['best_center_error_m_a' + suffix] = ce
            row['best_yaw_error_deg_a' + suffix] = ye
            for t in THRESHOLDS:
                key = str(int(t * 100)) + '_a' + suffix
                pi = int(c['matches'][t]['gt_pred'][gi])
                row['tp' + key] = int(pi >= 0)
                row['matched_pred_idx' + key] = pi
                row['matched_score' + key] = float(c['score'][pi]) if pi >= 0 else ''
                row['matched_iou' + key] = float(c['matrix'][pi, gi]) if pi >= 0 else ''
        row['delta_best_iou'] = row['best_iou_a1'] - row['best_iou_a0']
        for t in THRESHOLDS:
            s = str(int(t * 100))
            b0, b1 = row['best_iou_a0'] >= t, row['best_iou_a1'] >= t
            p0, p1 = row['tp' + s + '_a0'], row['tp' + s + '_a1']
            row['best_transition' + s] = ('above' if b0 else 'below') + '_to_' + ('above' if b1 else 'below')
            row['match_transition' + s] = ('TP' if p0 else 'FN') + '_to_' + ('TP' if p1 else 'FN')
            row['loss_class' + s] = ('geometric_unavailability' if not b1 else 'available_but_unmatched') if p0 and not p1 else ''
            row['gain_class' + s] = ('geometric_availability_gain' if not b0 else 'already_available_match_gain') if p1 and not p0 else ''
        GT_ROWS.append(row)
    sample = {'sample_idx': sid, 'gt_count': m, 'pred_count_a0': len(c0['pred']), 'pred_count_a1': len(c1['pred']),
              'delta_pred_count': len(c1['pred']) - len(c0['pred'])}
    budget = min(len(c0['pred']), len(c1['pred']))
    sample['equal_count_budget'] = budget
    for a, c in cases.items():
        suffix = str(int(float(a)))
        selected = np.argsort(-c['score'])[:budget]
        for t in THRESHOLDS:
            key = str(int(t * 100))
            sample['tp' + key + '_a' + suffix] = int(c['matches'][t]['tp'].sum())
            sample['max_cardinality_tp' + key + '_a' + suffix] = c['max_tp'][t]
            cm = greedy(c['matrix'][selected], c['score'][selected], t)
            append_stats(cap_stats[a], cm, c['score'][selected], m, t)
    for t in THRESHOLDS:
        s = str(int(t * 100))
        sample['delta_tp' + s] = sample['tp' + s + '_a1'] - sample['tp' + s + '_a0']
    SAMPLE_ROWS.append(sample)

    # Auxiliary cross-condition association, NOT an object/prediction identity map.
    # Maximize feasible pair count first, then total IoU among those pairings.
    cross = iou_matrix(c0['pp'], c1['pp'])
    for t in [0.5, 0.7, 0.9]:
        feasible = cross.astype(np.float64) >= t
        if cross.size:
            reward = np.where(feasible, (min(cross.shape) + 1) + cross.astype(np.float64), 0.)
            ii, jj = linear_sum_assignment(-reward)
            pairs = [(int(i), int(k)) for i, k in zip(ii, jj) if feasible[i, k]]
        else:
            pairs = []
        used0, used1 = {i for i, _ in pairs}, {k for _, k in pairs}
        for i, k in pairs:
            PAIR_ROWS.append({'sample_idx': sid, 'pair_iou_threshold': t, 'pred_idx_a0': i, 'pred_idx_a1': k,
                              'cross_iou': float(cross[i, k]), 'score_a0': float(c0['score'][i]),
                              'score_a1': float(c1['score'][k]), 'delta_score': float(c1['score'][k] - c0['score'][i]),
                              'tp50_a0': int(c0['matches'][.5]['tp'][i]), 'tp50_a1': int(c1['matches'][.5]['tp'][k]),
                              'tp70_a0': int(c0['matches'][.7]['tp'][i]), 'tp70_a1': int(c1['matches'][.7]['tp'][k])})
        for a, used in [('0.0', used0), ('1.0', used1)]:
            c = cases[a]
            for pi in range(len(c['pred'])):
                group = 'paired' if pi in used else 'unpaired'
                r = {'sample_idx': sid, 'pair_iou_threshold': t, 'alpha': a, 'pred_idx': pi, 'group': group,
                     'score': float(c['score'][pi]), 'best_gt_iou': float(c['best_iou_pred'][pi]),
                     'tp50': int(c['matches'][.5]['tp'][pi]), 'tp70': int(c['matches'][.7]['tp'][pi])}
                cross_groups[str(t)][a][group].append(r)
                GROUP_ROWS.append(r)
    if (sn + 1) % 100 == 0:
        log(f'Analyzed {sn + 1}/910 paired samples')

metrics, rank_rows, operating_rows, decomposition = {}, [], [], {}
pr_cache = {}
for a in ALPHAS:
    check('instrumentation_exact:' + a, native_stats[a] == instrumented_stats[a])
    metrics[a] = {}
    for t in THRESHOLDS:
        key = str(int(t * 100))
        st = native_stats[a][t]
        ap, _, _ = EVAL['calculate_ap'](native_stats[a], t)
        ref = audit_ref['recomputed'][a]['ap' + key]
        tp, fp, ngt = sum(st['tp']), sum(st['fp']), st['gt']
        check('AP_reference:' + a + ':' + key, abs(ap - ref['value']) <= 1e-12 and abs(ap - target['results'][a]['ap' + key]) <= 1e-12)
        check('counts_reference:' + a + ':' + key, tp == ref['tp'] and fp == ref['fp'] and ngt == ref['gt'])
        m = {'ap': float(ap), 'tp': tp, 'fp': fp, 'fn': ngt - tp, 'gt': ngt,
             'predictions': tp + fp, 'precision_final': tp / (tp + fp), 'recall_final': tp / ngt,
             'max_cardinality_tp': upper_counts[a][t],
             'geometry_graph_recall_ceiling': upper_counts[a][t] / ngt,
             'greedy_assignment_gap': (upper_counts[a][t] - tp) / ngt,
             'global_ranking_gap': tp / ngt - float(ap),
             'equal_count_topK': {'ap': float(EVAL['calculate_ap'](cap_stats[a], t)[0]),
                                 'tp': sum(cap_stats[a][t]['tp']), 'fp': sum(cap_stats[a][t]['fp'])}}
        metrics[a][key] = m
        sc = np.asarray(st['score'])
        order = np.argsort(-sc)
        labels = np.asarray(st['tp'])[order]
        ctp = np.cumsum(labels)
        cfp = np.arange(1, len(labels) + 1) - ctp
        precision = ctp / np.arange(1, len(labels) + 1)
        recall = ctp / ngt
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        check('AP_from_rank_trace:' + a + ':' + key,
              abs(float(np.sum(np.diff(np.r_[0., recall]) * envelope)) - ap) <= 1e-12)
        pr_cache[a, t] = (recall, precision, envelope, sc[order], labels)
        for rank, ri in enumerate(order):
            original = records[a][t][int(ri)]
            assert original['score'] == sc[ri] and original['tp'] == st['tp'][int(ri)]
            rank_rows.append({'alpha': a, 'iou_threshold': t, 'global_rank': rank + 1, **original,
                              'cumulative_tp': int(ctp[rank]), 'cumulative_fp': int(cfp[rank]),
                              'precision': float(precision[rank]), 'recall': float(recall[rank]),
                              'interpolated_precision': float(envelope[rank])})
        for cutoff in [.2, .3, .5, .7, .8, .9, .95]:
            select = sc >= cutoff
            n = int(select.sum()); ntp = int(np.asarray(st['tp'])[select].sum())
            operating_rows.append({'alpha': a, 'iou_threshold': t, 'selection': 'score_threshold', 'value': cutoff,
                                   'predictions': n, 'tp': ntp, 'fp': n - ntp,
                                   'precision': ntp / n if n else 0., 'recall': ntp / ngt})
        for n in [500, 1000, 2000, 4000, 6000, 7500, 8000, 8175]:
            if n > len(labels):
                continue
            ntp = int(ctp[n - 1])
            operating_rows.append({'alpha': a, 'iou_threshold': t, 'selection': 'global_topK', 'value': n,
                                   'predictions': n, 'tp': ntp, 'fp': n - ntp,
                                   'precision': ntp / n, 'recall': ntp / ngt})
        m['tied_score_entries'] = len(sc) - len(np.unique(sc))
for t in THRESHOLDS:
    key = str(int(t * 100))
    m0, m1 = metrics['0.0'][key], metrics['1.0'][key]
    d = {name: m1[name] - m0[name] for name in ['ap', 'tp', 'fp', 'recall_final', 'geometry_graph_recall_ceiling', 'greedy_assignment_gap', 'global_ranking_gap']}
    check('descriptive_decomposition:' + key, abs(d['ap'] - (d['geometry_graph_recall_ceiling'] - d['greedy_assignment_gap'] - d['global_ranking_gap'])) < 1e-12)
    decomposition[key] = d

gt_summary = {'total': len(GT_ROWS), 'best_iou': {}, 'thresholds': {}}
best0 = np.array([r['best_iou_a0'] for r in GT_ROWS])
best1 = np.array([r['best_iou_a1'] for r in GT_ROWS])
delta = best1 - best0
gt_summary['best_iou'] = {'alpha0': distribution(best0), 'alpha1': distribution(best1), 'delta': distribution(delta),
                           'improved_gt': int((delta > 1e-6).sum()), 'worsened_gt': int((delta < -1e-6).sum()),
                           'near_equal_gt': int((abs(delta) <= 1e-6).sum())}
bins = np.array([0., .5, .7, 1.00001])
b0, b1 = np.digitize(best0, bins) - 1, np.digitize(best1, bins) - 1
matrix = np.zeros((3, 3), dtype=int)
np.add.at(matrix, (b0, b1), 1)
gt_summary['best_iou_bin_transitions'] = {'labels': ['<0.5', '0.5-0.7', '>=0.7'], 'rows_alpha0_columns_alpha1': matrix.tolist()}
for t in THRESHOLDS:
    key = str(int(t * 100))
    gt_summary['thresholds'][key] = {
        'best_iou_transition': dict(Counter(r['best_transition' + key] for r in GT_ROWS)),
        'greedy_match_transition': dict(Counter(r['match_transition' + key] for r in GT_ROWS)),
        'lost_tp_classes': dict(Counter(r['loss_class' + key] for r in GT_ROWS if r['loss_class' + key])),
        'gained_tp_classes': dict(Counter(r['gain_class' + key] for r in GT_ROWS if r['gain_class' + key])),
        'best_candidate_count_a0': int((best0 >= t).sum()), 'best_candidate_count_a1': int((best1 >= t).sum()),
    }
    for a in ALPHAS:
        s = str(int(float(a)))
        check('GT_trace_counts:' + a + ':' + key, sum(r['tp' + key + '_a' + s] for r in GT_ROWS) == metrics[a][key]['tp'])

quantity = {'sample_delta_pred': distribution([r['delta_pred_count'] for r in SAMPLE_ROWS]),
            'samples_more': sum(r['delta_pred_count'] > 0 for r in SAMPLE_ROWS),
            'samples_fewer': sum(r['delta_pred_count'] < 0 for r in SAMPLE_ROWS),
            'samples_equal': sum(r['delta_pred_count'] == 0 for r in SAMPLE_ROWS),
            'gross_positive_sample_deltas': sum(max(0, r['delta_pred_count']) for r in SAMPLE_ROWS),
            'gross_negative_sample_deltas': -sum(min(0, r['delta_pred_count']) for r in SAMPLE_ROWS),
            'net_prediction_count_change': sum(r['delta_pred_count'] for r in SAMPLE_ROWS),
            'equal_count_topK_total_per_alpha': sum(r['equal_count_budget'] for r in SAMPLE_ROWS)}
check('net_363', quantity['net_prediction_count_change'] == 363)
score_groups = []
confidence = {}
for a in ALPHAS:
    rows = [r for r in PRED_ROWS if r['alpha'] == a]
    confidence[a] = {'scores_all': distribution([r['score'] for r in rows]),
                     'best_gt_iou_bins': dict(Counter(r['iou_bin'] for r in rows)), 'thresholds': {}}
    for t in THRESHOLDS:
        key = str(int(t * 100))
        confidence[a]['thresholds'][key] = {
            'TP_scores': distribution([r['score'] for r in rows if r['tp' + key]]),
            'FP_scores': distribution([r['score'] for r in rows if not r['tp' + key]]),
            'FP_classes': dict(Counter(r['status' + key] for r in rows if not r['tp' + key])),
            'FP_best_iou_bins': dict(Counter(r['iou_bin'] for r in rows if not r['tp' + key])),
            'high_conf_FP_ge_0.8': sum(r['score'] >= .8 and not r['tp' + key] for r in rows),
        }
        for lo, hi in zip([.2, .3, .5, .7, .8, .9, .95], [.3, .5, .7, .8, .9, .95, 1.000001]):
            rr = [r for r in rows if lo <= r['score'] < hi]
            n = len(rr); ntp = sum(r['tp' + key] for r in rr)
            score_groups.append({'alpha': a, 'iou_threshold': t, 'score_low_inclusive': lo,
                                 'score_high_exclusive': hi, 'predictions': n, 'tp': ntp, 'fp': n - ntp})
cross_summary = {}
for t, conditions in cross_groups.items():
    cross_summary[t] = {}
    for a, groups in conditions.items():
        cross_summary[t][a] = {}
        for name, rows in groups.items():
            cross_summary[t][a][name] = {'n': len(rows), 'tp50': sum(r['tp50'] for r in rows),
                                        'tp70': sum(r['tp70'] for r in rows),
                                        'scores': distribution([r['score'] for r in rows]),
                                        'best_iou': distribution([r['best_gt_iou'] for r in rows])}
    check('pairing_net363:' + t, cross_summary[t]['1.0']['unpaired']['n'] - cross_summary[t]['0.0']['unpaired']['n'] == 363)

matched_geometry = {}
for label, mask in [('all_gt', np.ones(len(best0), bool)), ('both_best_IoU_ge_0.5', (best0 >= .5) & (best1 >= .5))]:
    matched_geometry[label] = {}
    for s in ['0', '1']:
        rr = [r for r, keep in zip(GT_ROWS, mask) if keep]
        matched_geometry[label]['alpha' + s] = {
            'best_center_error_m': distribution([r['best_center_error_m_a' + s] for r in rr if r['best_center_error_m_a' + s] != '']),
            'best_yaw_error_deg': distribution([r['best_yaw_error_deg_a' + s] for r in rr if r['best_yaw_error_deg_a' + s] != ''])}

examples = []
for key in ['50', '70']:
    for direction in ['TP_to_FN', 'FN_to_TP']:
        selected = [r for r in GT_ROWS if r['match_transition' + key] == direction]
        selected.sort(key=lambda r: abs(r['delta_best_iou']), reverse=True)
        for r in selected[:15]:
            examples.append({'iou_threshold': int(key) / 100, 'transition': direction, **r})
check('no_torch_imported', 'torch' not in sys.modules)
write_csv('per_gt.csv', GT_ROWS)
write_csv('per_prediction.csv', PRED_ROWS)
write_csv('per_sample.csv', SAMPLE_ROWS)
write_csv('global_ranking_pr.csv', rank_rows)
write_csv('operating_points.csv', operating_rows)
write_csv('confidence_bins.csv', score_groups)
write_csv('cross_condition_pairs.csv', PAIR_ROWS)
write_csv('cross_condition_groups.csv', GROUP_ROWS)
write_csv('gt_transition_examples.csv', examples)

# Export standalone publication-compatible vector PDFs and high-resolution PNGs.
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 200})
colors = {'0.0': '#2166ac', '1.0': '#b2182b'}
figures = []


def figure_save(fig, stem):
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        name = stem + '.' + ext
        fig.savefig(RUN / name, bbox_inches='tight')
        figures.append(name)
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, t in zip(axes, THRESHOLDS):
    for a in ALPHAS:
        recall, precision, envelope, sc, labels = pr_cache[a, t]
        ax.plot(recall, envelope, color=colors[a], label=f'alpha={a}, AP={metrics[a][str(int(t*100))]["ap"]:.4f}')
        ax.plot(recall[::30], precision[::30], color=colors[a], alpha=.25, linewidth=.5)
    ax.set(xlabel='Recall', ylabel='Precision', title=f'BEV IoU >= {t}', xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=.2); ax.legend(loc='lower left')
figure_save(fig, 'pr_curves')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist2d(best0, best1, bins=50, range=[[0, 1], [0, 1]], cmap='Blues', cmin=1)
axes[0].plot([0, 1], [0, 1], '--', color='black', linewidth=.8)
for t in THRESHOLDS:
    axes[0].axhline(t, color='grey', linewidth=.7); axes[0].axvline(t, color='grey', linewidth=.7)
axes[0].set(xlabel='Best saved-box IoU, alpha=0', ylabel='Best saved-box IoU, alpha=1', title='Same-GT geometry (8,164 GTs)')
im = axes[1].imshow(matrix, cmap='Blues')
for i in range(3):
    for k in range(3):
        axes[1].text(k, i, str(matrix[i, k]), ha='center', va='center', color='white' if matrix[i, k] > matrix.max()/2 else 'black')
axes[1].set(xticks=range(3), yticks=range(3), xticklabels=['<0.5', '0.5-0.7', '>=0.7'],
            yticklabels=['<0.5', '0.5-0.7', '>=0.7'], xlabel='alpha=1', ylabel='alpha=0', title='Best-IoU transition counts')
fig.colorbar(im, ax=axes[1], shrink=.8)
figure_save(fig, 'gt_geometry_transitions')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, t in zip(axes, THRESHOLDS):
    for a in ALPHAS:
        rr = [r for r in PRED_ROWS if r['alpha'] == a and not r['tp'+str(int(t*100))]]
        ax.hist([r['score'] for r in rr], bins=np.linspace(.2, 1, 25), histtype='step', linewidth=1.5, color=colors[a], label=f'alpha={a}, FP={len(rr)}')
    ax.set(xlabel='Confidence score', ylabel='False-positive count', title=f'FP confidence, IoU >= {t}')
    ax.legend(); ax.grid(alpha=.2)
figure_save(fig, 'fp_confidence')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, t in zip(axes, THRESHOLDS):
    for a in ALPHAS:
        recall, precision, envelope, sc, labels = pr_cache[a, t]
        rank = np.arange(1, len(labels) + 1)
        ax.plot(rank, rank - np.cumsum(labels), color=colors[a], label=f'alpha={a}')
    ax.set(xlabel='Global confidence rank', ylabel='Cumulative false positives', title=f'Global ranking, IoU >= {t}')
    ax.legend(); ax.grid(alpha=.2)
figure_save(fig, 'global_ranking_fp')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
values = [r['delta_pred_count'] for r in SAMPLE_ROWS]
axes[0].hist(values, bins=np.arange(min(values)-.5, max(values)+1.5, 1), color='#4575b4')
axes[0].set(xlabel='Prediction count difference (alpha1-alpha0)', ylabel='Sample count', title='Per-sample count changes; net +363')
for t, axlabel in [(.5, 'AP50'), (.7, 'AP70')]:
    key = str(int(t*100))
    d = decomposition[key]
    axes[1].plot(['Graph ceiling', '- Greedy gap', '- Ranking gap', 'AP delta'],
                 [d['geometry_graph_recall_ceiling'], -d['greedy_assignment_gap'], -d['global_ranking_gap'], d['ap']],
                 marker='o', label=axlabel)
axes[1].axhline(0, color='black', linewidth=.7)
axes[1].set(ylabel='Change in metric (0-1 scale)', title='Descriptive identity, not causal attribution')
axes[1].tick_params(axis='x', rotation=15); axes[1].legend(); axes[1].grid(alpha=.2)
figure_save(fig, 'count_and_ap_decomposition')

summary = {'protocol': {'phase': 'P3-B', 'history_frame_gap_k': 3, 'model_time_code': 300,
                        'physical_observation_interval_ms': 600, 'sample_count': 910, 'gt_count': len(GT_ROWS),
                        'iou': 'BEV polygon IoU, first four corners, repository float32 geometry',
                        'gt_identity': '(sample_idx, gt_idx); exact GT array equality across alpha',
                        'fp_attribution': 'FP is prediction-level. best_gt_idx is diagnostic only, -1 for zero overlap.',
                        'cross_condition_pairing': 'auxiliary maximum-cardinality then max-IoU association at 0.5/0.7/0.9; not natural identities',
                        'equal_count_topK': 'within-sample min(N0,N1) highest scores per condition, rematched; confidence-dependent diagnostic',
                        'decomposition': 'AP = max-cardinality recall - greedy assignment gap - global ranking gap; descriptive identity only'},
           'metrics': metrics, 'gt_analysis': gt_summary, 'quantity': quantity, 'confidence': confidence,
           'cross_condition_association': cross_summary, 'best_candidate_geometry': matched_geometry,
           'descriptive_ap_decomposition': decomposition,
           'artifacts': {'per_gt_rows': len(GT_ROWS), 'per_prediction_rows': len(PRED_ROWS), 'per_sample_rows': len(SAMPLE_ROWS),
                         'pr_rank_rows': len(rank_rows), 'cross_pair_rows': len(PAIR_ROWS), 'figures': figures},
           'environment': {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'python': sys.executable,
                           'numpy': np.__version__, 'scipy': scipy.__version__, 'shapely': shapely.__version__, 'matplotlib': matplotlib.__version__,
                           'torch_imported': 'torch' in sys.modules},
           'validation': {'passed': all(CHECKS.values()), 'checks_count': len(CHECKS), 'ap_tolerance': 1e-12}}
read(Path(__file__))
write_json('input_hashes.json', HASHES)
write_json('validation.json', CHECKS)
write_json('summary.json', summary)
log('P3-B numeric analysis finished; original AP, TP/FP and GT identities verified.')
log(json.dumps({'metrics': metrics, 'gt_analysis': gt_summary, 'quantity': quantity,
                'decomposition': decomposition, 'run_directory': str(RUN)}, indent=2, ensure_ascii=False))
LOG.close()
