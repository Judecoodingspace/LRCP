"""P3-C1 CPU-only, immutable saved-box score crossover; run once in fresh P3C1 directory."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lrcp_p3c1_matplotlib')
import ast
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np
import scipy
import shapely
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent
BASE = Path('/workspace/lrcp_d2')
B = BASE / 'codex_reports/P3B_20261020_103959_980233'
REPO = Path('/workspace/LRCP')
assert RUN.name.startswith('P3C1_') and not (RUN / 'summary.json').exists()
LOG = (RUN / 'execution.log').open('x', buffering=1)
HASHES, CHECKS = {}, {}
T = [.5, .7]
TAUS = [.5, .7, .9]


def log(s):
    print(s, flush=True); LOG.write(str(s) + '\n')


def read(path):
    data = Path(path).read_bytes(); HASHES[str(path)] = hashlib.sha256(data).hexdigest(); return data


def js(path):
    return json.loads(read(path))


def csv_read(path):
    read(path)
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def check(name, ok):
    CHECKS[name] = bool(ok)
    if not ok:
        raise AssertionError(name)


def save(name, value):
    with (RUN / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def csv_write(name, rows, fields=None):
    with (RUN / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)


def extract(path, names, env):
    tree = ast.parse(read(path).decode(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def array_sha(x):
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


log('P3-C1 START: CPU only; validate consumed P3-B artifacts; reuse stored associations')
for line in read(B / 'ARTIFACTS_FINAL.sha256').decode().splitlines():
    expected, name = line.split('  ', 1)
    check('P3B_artifact:' + name, hashlib.sha256(read(B / name)).hexdigest() == expected)
reference = js(B / 'summary.json')
input_reference = js(B / 'input_hashes.json')
for name in ['REPORT_FINAL.md', 'METHODS.md']:
    read(B / name)
gt_ref_rows = csv_read(B / 'per_gt.csv')
pred_ref_rows = csv_read(B / 'per_prediction.csv')
gt_ref = {(r['sample_idx'], int(r['gt_idx'])): r for r in gt_ref_rows}
pred_ref = {(r['sample_idx'], r['alpha'], int(r['pred_idx'])): r for r in pred_ref_rows}
pair_rows = csv_read(B / 'cross_condition_pairs.csv')
pair_map = defaultdict(list)
for r in pair_rows:
    pair_map[float(r['pair_iou_threshold']), r['sample_idx']].append(r)
for tau in TAUS:
    check('association_count:' + str(tau), sum(len(v) for (t, s), v in pair_map.items() if t == tau) == reference['cross_condition_association'][str(tau)]['0.0']['paired']['n'])
p3 = js(BASE / 'P3A_20261014_120929.json')
k3 = next(r for r in p3 if r['delay_label_ms'] == 300)
ids = k3['sample_ids']
check('cohort', len(ids) == len(set(ids)) == 910 and len(gt_ref) == 8164)
GEO = extract(REPO / 'opencood/utils/common_utils.py', ['convert_format', 'compute_iou'], {'np': np, 'Polygon': Polygon})
common = SimpleNamespace(convert_format=GEO['convert_format'], compute_iou=GEO['compute_iou'], torch_tensor_to_numpy=lambda x: x)
EV = extract(REPO / 'opencood/utils/eval_utils.py', ['caluclate_tp_fp', 'calculate_ap', 'voc_ap'], {'np': np, 'common_utils': common})
helpers = extract(B / 'analyze.py', ['greedy', 'iou_matrix'], {'np': np, 'GEO': GEO})
greedy, iou_matrix = helpers['greedy'], helpers['iou_matrix']
for p in [REPO / 'opencood/utils/common_utils.py', REPO / 'opencood/utils/eval_utils.py']:
    check('eval_source:' + p.name, HASHES[str(p)] == input_reference[str(p)])

cases, fixed_rows = {}, []
for sn, sid in enumerate(ids):
    for a in ['0.0', '1.0']:
        path = BASE / 'P3A_20261014_120929_boxes' / f'sample_{sid}_alpha_{a}.npz'
        read(path); check('NPZ_hash:' + path.name, HASHES[str(path)] == input_reference[str(path)])
        with np.load(path, allow_pickle=False) as z:
            pred, scores, gt = z['pred_boxes'].copy(), z['pred_scores'].copy(), z['gt_boxes'].copy()
            assert str(z['sample_idx'].item()) == sid and float(z['alpha']) == float(a)
        for x in [pred, scores, gt]:
            x.flags.writeable = False
        matrix = iou_matrix(list(GEO['convert_format'](pred)), list(GEO['convert_format'](gt)))
        cases[sid, a] = {'pred': pred, 'score': scores, 'gt': gt, 'matrix': matrix,
                         'pred_sha': array_sha(pred), 'gt_sha': array_sha(gt), 'score_sha': array_sha(scores)}
        fixed_rows.append({'sample_idx': sid, 'geometry_alpha': a, 'pred_count': len(pred), 'gt_count': len(gt),
                           'pred_coordinates_sha256': array_sha(pred), 'gt_sha256': array_sha(gt), 'native_scores_sha256': array_sha(scores)})
    assert np.array_equal(cases[sid, '0.0']['gt'], cases[sid, '1.0']['gt'])
    for tau in TAUS:
        rows = pair_map[tau, sid]
        i0, i1 = [int(r['pred_idx_a0']) for r in rows], [int(r['pred_idx_a1']) for r in rows]
        assert len(i0) == len(set(i0)) == len(i1) == len(set(i1))
        for r, i, j in zip(rows, i0, i1):
            assert 0 <= i < len(cases[sid, '0.0']['pred']) and 0 <= j < len(cases[sid, '1.0']['pred'])
            assert float(r['cross_iou']) >= tau
            assert float(cases[sid, '0.0']['score'][i]) == float(r['score_a0'])
            assert float(cases[sid, '1.0']['score'][j]) == float(r['score_a1'])
check('all_associations_one_to_one_and_donor_scores_valid', True)
csv_write('fixed_geometry_manifest.csv', fixed_rows)

scenarios = [{'id': 'G0_native', 'a': '0.0', 'donor': '0.0', 'tau': None},
             {'id': 'G1_native', 'a': '1.0', 'donor': '1.0', 'tau': None}]
for tau in TAUS:
    for a, donor in [('0.0', '1.0'), ('1.0', '0.0')]:
        scenarios.append({'id': 'G' + a[0] + '_matchedS' + donor[0] + '_tau' + str(tau).replace('.', 'p'),
                          'a': a, 'donor': donor, 'tau': tau})
all_metrics = {}
baseline_matches, baseline_rank = {}, {}
pred_rows, gt_rows, rank_rows, score_rows, local_ties, label_changes = [], [], [], [], [], []
all_stats = {}
pr_cache = {}
baseline_exact = {}


def ap_labels(labels):
    labels = np.asarray(labels)
    rec = np.cumsum(labels) / 8164
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0., rec]) * envelope))


def stable_match(matrix, scores, t):
    # Diagnostic only: the formal result always comes from the original evaluator.
    order = np.argsort(-scores, kind='stable')
    view = greedy(matrix[order], np.arange(len(order), 0, -1, dtype=float), t)
    pred_gt = np.full(len(scores), -1, dtype=int)
    pred_gt[order] = view['pred_gt']
    gt_pred = np.full(matrix.shape[1], -1, dtype=int)
    for pi, gi in enumerate(pred_gt):
        if gi >= 0:
            gt_pred[gi] = pi
    return {'pred_gt': pred_gt, 'gt_pred': gt_pred, 'tp': (pred_gt >= 0).astype(int)}


for si, scenario in enumerate(scenarios):
    name, a, tau = scenario['id'], scenario['a'], scenario['tau']
    native = tau is None
    if si == 2:
        check('baseline_gate', len(baseline_exact) == 4 and all(baseline_exact.values()))
        save('baseline_validation.json', {'passed': True, 'AP_exact_equality': baseline_exact,
                                          'metrics': {s: all_metrics[s] for s in ['G0_native', 'G1_native']}})
        log('BASELINES PASSED: all four AP values exactly reproduced; TP/FP and GT matches verified. Begin score crossover.')
    stats = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in T}
    inst_stats = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in T}
    trace = {t: [] for t in T}
    coverage = {'paired': 0, 'unpaired': 0, 'scores_changed': 0, 'local_order_changed_samples': 0}
    changes = {str(t): {'pred_TP_FP_changed': 0, 'pred_matched_GT_changed': 0, 'GT_TP_FN_changed': 0,
                        'GT_matched_pred_changed': 0, 'unpaired_pred_TP_FP_changed': 0} for t in T}
    for sn, sid in enumerate(ids):
        c = cases[sid, a]
        scores = c['score'].copy()
        source_idx = np.arange(len(scores))
        paired = np.zeros(len(scores), dtype=bool)
        if not native:
            other = cases[sid, scenario['donor']]
            for pair in pair_map[tau, sid]:
                target_idx = int(pair['pred_idx_a' + a[0]])
                donor_idx = int(pair['pred_idx_a' + scenario['donor'][0]])
                assert not paired[target_idx]
                scores[target_idx] = other['score'][donor_idx]
                source_idx[target_idx] = donor_idx
                paired[target_idx] = True
        assert scores.dtype == c['score'].dtype and np.array_equal(scores[~paired], c['score'][~paired])
        assert array_sha(c['pred']) == c['pred_sha'] and array_sha(c['gt']) == c['gt_sha']
        coverage['paired'] += int(paired.sum())
        coverage['unpaired'] += int((~paired).sum())
        coverage['scores_changed'] += int((scores != c['score']).sum())
        order = np.argsort(-scores)
        assert np.array_equal(order, np.argsort(-scores))
        native_order = np.argsort(-c['score'])
        coverage['local_order_changed_samples'] += int(not np.array_equal(order, native_order))
        matches = {}
        tie_entry = {'scenario': name, 'sample_idx': sid, 'duplicate_score_entries': len(scores)-len(np.unique(scores)),
                     'quicksort_vs_stable_order_positions_changed': int((order != np.argsort(-scores, kind='stable')).sum())}
        for t in T:
            EV['caluclate_tp_fp'](c['pred'], scores, c['gt'], stats, t)
            match = greedy(c['matrix'], scores, t)
            matches[t] = match
            ins = inst_stats[t]
            ins['tp'] += match['tp'][order].tolist(); ins['fp'] += (1-match['tp'][order]).tolist()
            ins['score'] += scores[order].tolist(); ins['gt'] += len(c['gt'])
            if native:
                baseline_matches[sid, a, t] = match
                for pi, mg in enumerate(match['pred_gt']):
                    ref = pred_ref[sid, a, pi]
                    assert int(ref['tp' + str(int(t*100))]) == int(match['tp'][pi])
                    assert int(ref['matched_gt_idx' + str(int(t*100))]) == mg
                for gi, pi in enumerate(match['gt_pred']):
                    assert int(gt_ref[sid, gi]['matched_pred_idx' + str(int(t*100)) + '_a' + a[0]]) == pi
            base_match = baseline_matches[sid, a, t]
            cc = changes[str(t)]
            cc['pred_TP_FP_changed'] += int((match['tp'] != base_match['tp']).sum())
            cc['pred_matched_GT_changed'] += int((match['pred_gt'] != base_match['pred_gt']).sum())
            cc['GT_TP_FN_changed'] += int(((match['gt_pred'] >= 0) != (base_match['gt_pred'] >= 0)).sum())
            cc['GT_matched_pred_changed'] += int((match['gt_pred'] != base_match['gt_pred']).sum())
            cc['unpaired_pred_TP_FP_changed'] += int(((match['tp'] != base_match['tp']) & ~paired).sum())
            sm = stable_match(c['matrix'], scores, t)
            tie_entry['stable_diagnostic_TP_FP_changed' + str(int(t*100))] = int((sm['tp'] != match['tp']).sum())
            tie_entry['stable_diagnostic_matched_GT_changed' + str(int(t*100))] = int((sm['pred_gt'] != match['pred_gt']).sum())
            for pi in order:
                trace[t].append({'sample_idx': sid, 'pred_idx': int(pi), 'score': float(scores[pi]),
                                 'tp': int(match['tp'][pi]), 'matched_gt_idx': int(match['pred_gt'][pi])})
            for gi, pi in enumerate(match['gt_pred']):
                girow = {'scenario': name, 'geometry_alpha': a, 'association_threshold': '' if native else tau,
                         'eval_IoU': t, 'sample_idx': sid, 'gt_idx': gi, 'matched_pred_idx': int(pi),
                         'native_matched_pred_idx': int(base_match['gt_pred'][gi]), 'TP': int(pi >= 0),
                         'native_TP': int(base_match['gt_pred'][gi] >= 0),
                         'matched_score': float(scores[pi]) if pi >= 0 else '',
                         'matched_IoU': float(c['matrix'][pi, gi]) if pi >= 0 else ''}
                gt_rows.append(girow)
        if tie_entry['duplicate_score_entries']:
            local_ties.append(tie_entry)
        for pi in range(len(scores)):
            srow = {'scenario': name, 'geometry_alpha': a, 'association_threshold': '' if native else tau,
                    'sample_idx': sid, 'pred_idx': pi, 'paired': int(paired[pi]),
                    'source_alpha': scenario['donor'] if paired[pi] else a, 'source_pred_idx': int(source_idx[pi]),
                    'native_score': float(c['score'][pi]), 'new_score': float(scores[pi]),
                    'score_changed': int(scores[pi] != c['score'][pi])}
            score_rows.append(srow)
            pr = dict(srow)
            for t in T:
                key = str(int(t*100)); base_match = baseline_matches[sid, a, t]
                pr['TP' + key] = int(matches[t]['tp'][pi]); pr['native_TP' + key] = int(base_match['tp'][pi])
                pr['matched_gt_idx' + key] = int(matches[t]['pred_gt'][pi])
                pr['native_matched_gt_idx' + key] = int(base_match['pred_gt'][pi])
            pred_rows.append(pr)
        if (sn + 1) % 200 == 0:
            log(f'{name}: {sn+1}/910 samples re-matched')
    check('original_vs_instrumented:' + name, stats == inst_stats)
    all_stats[name] = stats
    metrics = {'geometry_alpha': a, 'donor_alpha_for_pairs': scenario['donor'], 'association_threshold': tau,
               'coverage': coverage, 'matching_changes': changes, 'eval': {}}
    n = len(stats[.5]['tp']); coverage['pair_coverage'] = coverage['paired']/n
    for t in T:
        key = str(int(t*100)); st = stats[t]
        ap = float(EV['calculate_ap'](stats, t)[0])
        ntp, nfp = sum(st['tp']), sum(st['fp'])
        sc = np.asarray(st['score']); labels = np.asarray(st['tp'])
        idx = np.argsort(-sc)
        check('repeat_default_global_sort:' + name + ':' + key, np.array_equal(idx, np.argsort(-sc)))
        sorted_sc, sorted_labels = sc[idx], labels[idx]
        cum_tp = np.cumsum(sorted_labels); ranks = np.arange(1, len(idx)+1)
        precision, recall = cum_tp/ranks, cum_tp/8164
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        check('trace_AP:' + name + ':' + key, abs(ap_labels(sorted_labels)-ap) <= 1e-12)
        best, worst = [], []
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_sc) != 0)+1, len(sorted_sc)]
        mixed, tied_entries = 0, 0
        for start, end in zip(starts[:-1], starts[1:]):
            group = sorted_labels[start:end]
            best.extend(sorted(group, reverse=True)); worst.extend(sorted(group))
            if end-start > 1:
                tied_entries += end-start-1
                mixed += int(group.min() != group.max())
        stable_idx = np.argsort(-sc, kind='stable')
        bounds = {'tp_first_AP': ap_labels(best), 'fp_first_AP': ap_labels(worst),
                  'stable_global_sort_AP_diagnostic': ap_labels(labels[stable_idx]),
                  'quicksort_vs_stable_global_positions_changed': int((idx != stable_idx).sum()),
                  'duplicate_score_entries': int(tied_entries), 'mixed_TP_FP_tie_groups': int(mixed)}
        check('AP_tie_bounds:' + name + ':' + key, bounds['fp_first_AP']-1e-12 <= ap <= bounds['tp_first_AP']+1e-12)
        metrics['eval'][key] = {'ap': ap, 'tp': ntp, 'fp': nfp, 'fn': 8164-ntp, 'gt': st['gt'], 'predictions': n,
                                'recall_final': ntp/8164, 'precision_final': ntp/n, 'ties': bounds}
        if native:
            ref = reference['metrics'][a][key]
            baseline_exact[name + ':' + key] = ap == ref['ap']
            check('baseline_exact:' + name + ':' + key, ap == ref['ap'] and ap == k3['results'][a]['ap'+key])
            check('baseline_counts:' + name + ':' + key, (ntp,nfp,st['gt']) == (ref['tp'],ref['fp'],ref['gt']))
        for rank, pi in enumerate(idx, 1):
            rr = trace[t][int(pi)]
            identity = (rr['sample_idx'], rr['pred_idx'])
            if native:
                baseline_rank[a,t,identity] = rank
            rank_rows.append({'scenario': name, 'geometry_alpha': a, 'association_threshold': '' if native else tau,
                              'eval_IoU': t, 'global_rank': rank, **rr,
                              'native_global_rank': baseline_rank[a,t,identity],
                              'cumulative_TP': int(cum_tp[rank-1]), 'cumulative_FP': rank-int(cum_tp[rank-1]),
                              'precision': float(precision[rank-1]), 'recall': float(recall[rank-1]),
                              'interpolated_precision': float(envelope[rank-1])})
        pr_cache[name,t] = (recall, envelope)
    all_metrics[name] = metrics
    log(name + ' COMPLETE ' + json.dumps({k:{x:v[x] for x in ['ap','tp','fp']} for k,v in metrics['eval'].items()}))

# Both directions together conserve the full two-condition score multiset.
for tau in TAUS:
    for sid in ids:
        original = np.concatenate([cases[sid,a]['score'] for a in ['0.0','1.0']])
        copied = {a:cases[sid,a]['score'].copy() for a in ['0.0','1.0']}
        for pair in pair_map[tau,sid]:
            i,j=int(pair['pred_idx_a0']),int(pair['pred_idx_a1'])
            copied['0.0'][i]=cases[sid,'1.0']['score'][j]
            copied['1.0'][j]=cases[sid,'0.0']['score'][i]
        assert np.array_equal(np.sort(original),np.sort(np.concatenate(list(copied.values()))))
check('bidirectional_score_multisets_conserved', True)
check('fixed_geometry_GT_and_unpaired_scores_unchanged', True)

sensitivity = []
for tau in TAUS:
    names = {a: next(s['id'] for s in scenarios if s['a']==a and s['tau']==tau) for a in ['0.0','1.0']}
    row = {'association_IoU':tau, 'pairs_each':all_metrics[names['0.0']]['coverage']['paired'],
           'coverage_G0':all_metrics[names['0.0']]['coverage']['pair_coverage'],
           'coverage_G1':all_metrics[names['1.0']]['coverage']['pair_coverage']}
    for key in ['50','70']:
        b0,b1=all_metrics['G0_native']['eval'][key],all_metrics['G1_native']['eval'][key]
        c0,c1=all_metrics[names['0.0']]['eval'][key],all_metrics[names['1.0']]['eval'][key]
        row.update({f'G0_matchedS1_AP{key}':c0['ap'], f'G1_matchedS0_AP{key}':c1['ap'],
                    f'G0_AP{key}_change':c0['ap']-b0['ap'], f'G1_AP{key}_change':c1['ap']-b1['ap'],
                    f'G1_matchedS0_minus_G0_native_AP{key}':c1['ap']-b0['ap'],
                    f'G1_native_minus_G0_matchedS1_AP{key}':b1['ap']-c0['ap'],
                    f'crossed_gap_AP{key}':c1['ap']-c0['ap']})
    sensitivity.append(row)
csv_write('score_assignments.csv',score_rows)
csv_write('per_prediction_evaluation.csv',pred_rows)
csv_write('per_GT_evaluation.csv',gt_rows)
csv_write('global_ranking_PR.csv',rank_rows)
csv_write('sensitivity.csv',sensitivity)
csv_write('local_tie_diagnostics.csv',local_ties,fields=['scenario','sample_idx','duplicate_score_entries','quicksort_vs_stable_order_positions_changed','stable_diagnostic_TP_FP_changed50','stable_diagnostic_matched_GT_changed50','stable_diagnostic_TP_FP_changed70','stable_diagnostic_matched_GT_changed70'])
metric_rows=[]
for name,m in all_metrics.items():
    for key,v in m['eval'].items():
        metric_rows.append({'scenario':name,'geometry_alpha':m['geometry_alpha'],'association_IoU':m['association_threshold'],
                            'eval_IoU':int(key)/100,**{k:v[k] for k in ['ap','tp','fp','fn','predictions','recall_final','precision_final']},
                            **m['matching_changes'][str(int(key)/100)]})
csv_write('metrics.csv',metric_rows)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(3,2,figsize=(12,11))
for ri,tau in enumerate(TAUS):
    for ci,t in enumerate(T):
        ax=axes[ri,ci]
        for a,color in [('0.0','#2166ac'),('1.0','#b2182b')]:
            for native,style in [(True,'-'),(False,'--')]:
                name='G'+a[0]+'_native' if native else next(s['id'] for s in scenarios if s['a']==a and s['tau']==tau)
                recall,precision=pr_cache[name,t]
                ax.plot(recall,precision,style,color=color,label=name+', AP='+f'{all_metrics[name]["eval"][str(int(t*100))]["ap"]:.4f}')
        ax.set(xlabel='Recall',ylabel='Precision',title=f'Association IoU={tau}; evaluation IoU={t}',xlim=(0,1),ylim=(.75,1.005))
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower left')
fig.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(RUN/('PR_curves.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key in zip(axes,['50','70']):
    for a,label,color in [('0.0','G0 with matched S1','#2166ac'),('1.0','G1 with matched S0','#b2182b')]:
        ax.plot(TAUS,[r[f'G{a[0]}_AP{key}_change'] for r in sensitivity],marker='o',color=color,label=label)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Association IoU threshold',ylabel='AP change vs native scores',title='AP'+key)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(RUN/('score_crossover_sensitivity.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)

check('no_torch_imported','torch' not in sys.modules)
check('inputs_unchanged_after_experiment',all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in HASHES.items()))
read(Path(__file__));read(RUN/'PROTOCOL.md')
save('input_hashes.json',HASHES)
save('validation.json',CHECKS)
save('summary.json',{'protocol':{'phase':'P3-C1','history_frame_gap_k':3,'model_time_code':300,'physical_observation_interval_ms':600,
                               'sample_count':910,'GT_count':8164,'geometry_conditions':['0.0','1.0'],'association_thresholds':TAUS,
                               'original_evaluator_preserved':True,'PPU_used':False,'association_reused_not_recomputed':True},
                     'baseline_exact':baseline_exact,'conditions':all_metrics,'sensitivity':sensitivity,
                     'local_tie_diagnostics':{'samples_with_ties':len(local_ties),'stable_label_changes50':sum(r['stable_diagnostic_TP_FP_changed50'] for r in local_ties),
                                              'stable_label_changes70':sum(r['stable_diagnostic_TP_FP_changed70'] for r in local_ties)},
                     'environment':{'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'python':sys.executable,'numpy':np.__version__,
                                    'scipy':scipy.__version__,'shapely':shapely.__version__,'torch_imported':'torch' in sys.modules},
                     'validation':{'passed':all(CHECKS.values()),'checks':len(CHECKS),'ap_tolerance':1e-12},
                     'artifacts':{'score_assignment_rows':len(score_rows),'prediction_rows':len(pred_rows),'GT_rows':len(gt_rows),'PR_rows':len(rank_rows)}})
log('P3-C1 NUMERIC PHASE COMPLETE: '+json.dumps(sensitivity,ensure_ascii=False))
LOG.close()
