"""CPU-only audit of existing LRCP P3-A artifacts; never imports torch.

Run with /workspace/venvs/lrcp/bin/python -B this_file.py.
Creates a unique audit subdirectory; never changes experimental artifacts.
"""
import ast
from collections import Counter
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import shapely
from shapely.geometry import Polygon

BASE = Path('/workspace/lrcp_d2')
REPO = Path('/workspace/LRCP')
MODEL = REPO / 'opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18'
RUN = BASE / 'codex_reports' / ('audit_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
RUN.mkdir(parents=True, exist_ok=False)
errors = []
evidence = {'audit_time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'python': sys.executable, 'numpy': np.__version__, 'shapely': shapely.__version__,
            'method': 'CPU NumPy/Shapely; AST-extracted existing eval functions; no torch import',
            'files': {}, 'checks': {}}


def save(name, value):
    with (RUN / name).open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))


def record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    evidence['files'][str(path)] = {'sha256': h.hexdigest(), 'size': path.stat().st_size,
                                  'mtime_epoch': path.stat().st_mtime}
    return h.hexdigest()


def read_json(path):
    record(path)
    return json.loads(Path(path).read_text())


def check(name, ok, detail=None):
    evidence['checks'][name] = {'passed': bool(ok), 'detail': detail}
    if not ok:
        errors.append(name)


def command(args):
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    return {'args': args, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}


process = command(['ps', '-eo', 'pid,ppid,lstart,etime,%cpu,%mem,args'])
save('process_snapshot.txt', process['stdout'])
evidence['git'] = {name: command(['git', '-C', str(REPO)] + args)
                   for name, args in {'head': ['rev-parse', 'HEAD'], 'status': ['status', '--short'],
                                      'diff': ['diff', '--stat']}.items()}
evidence['rules'] = {}
for path in [Path('/AGENTS.md'), Path('/workspace/AGENTS.md'), BASE / 'AGENTS.md', REPO / 'AGENTS.md']:
    evidence['rules'][str(path)] = path.read_text() if path.exists() else None
evidence['state_files'] = {}
for p in sorted(BASE.iterdir()):
    if p.is_file() and (p.suffix == '.pid' or p.name.endswith('_path')):
        record(p)
        evidence['state_files'][p.name] = p.read_text().strip()
pid = evidence['state_files']['p3a.pid']
evidence['p3a_recorded_pid'] = {'pid': pid, 'proc_exists': Path('/proc', pid).exists()}
evidence['p3a_process_matches'] = [line for line in process['stdout'].splitlines()
                                    if re.search(r'p3a_capture_k3\.py(?:\s|$)', line)]

p1path = BASE / 'P1C_20261013_114845.json'
manifestpath = BASE / 'P1_common910_indices.json'
expected = {
    MODEL / 'net_epoch_bestval_at19.pth': '666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724',
    p1path: '5ad4731c6173811731c5ebeeb5f18cb2852f66b72a3e467b56f380132110d191',
    manifestpath: '9ea235bdae98dfffa76f79bf67d88fa57d4077c26b08d24fb5e92671bb67630f',
}
for p, sha in expected.items():
    check('hash:' + p.name, record(p) == sha)
manifest = read_json(manifestpath)
p1 = read_json(p1path)
p3 = read_json(BASE / 'P3A_20261014_120929.json')
ids = manifest['sample_ids']
check('manifest_910_unique_strings', len(ids) == len(set(ids)) == 910 and all(isinstance(x, str) for x in ids))
check('p1_six_delays', [r['delay_label_ms'] for r in p1] == list(range(0, 600, 100)))
for row in p1:
    check('p1_samples:' + str(row['delay_label_ms']),
          row['valid_batches'] == len(row['sample_ids']) == len(set(row['sample_ids'])) == 910
          and set(row['sample_ids']) == set(ids))
check('p3_p1_exact_prefix_json', p3 == p1[:4])
check('unique_best_checkpoint', sorted(p.name for p in MODEL.glob('net_epoch_bestval_at*.pth')) == ['net_epoch_bestval_at19.pth'])
record(MODEL / 'config.yaml')
evidence['p1_delta_table'] = [
    {'history_frame_gap_k': r['delay_label_ms'] // 100, 'model_time_code': r['delay_label_ms'],
     'delta_ap50': r['results']['1.0']['ap50'] - r['results']['0.0']['ap50'],
     'delta_ap70': r['results']['1.0']['ap70'] - r['results']['0.0']['ap70']}
    for r in p1]

scripts = [BASE / 'p3a_capture_k3.py', *sorted(BASE.glob('paired_probe*.py'))]
for p in scripts:
    record(p)
    ast.parse(p.read_text(), filename=str(p))
    compile(p.read_text(), str(p), 'exec')
    check('syntax:' + p.name, True)
save('p1c_to_p3a.diff', ''.join(difflib.unified_diff(
    (BASE / 'paired_probe_common910.py').read_text().splitlines(True),
    (BASE / 'p3a_capture_k3.py').read_text().splitlines(True),
    fromfile=str(BASE / 'paired_probe_common910.py'), tofile=str(BASE / 'p3a_capture_k3.py'))))

# Extract only the reviewed pure geometry/AP functions, avoiding package imports
# that initialize deep-learning libraries. The only adapter replaces tensor->NumPy
# conversion with identity on already saved NumPy arrays. Matching, sort order,
# float32 IoU conversion, and AP integration remain the repository function bodies.
def extract(path, names, env):
    record(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError('Missing evaluation function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


geometry = extract(REPO / 'opencood/utils/common_utils.py', ['compute_iou', 'convert_format'],
                   {'np': np, 'Polygon': Polygon})
common = SimpleNamespace(compute_iou=geometry['compute_iou'], convert_format=geometry['convert_format'],
                         torch_tensor_to_numpy=lambda x: x)
evaluation = extract(REPO / 'opencood/utils/eval_utils.py', ['voc_ap', 'caluclate_tp_fp', 'calculate_ap'],
                     {'np': np, 'common_utils': common})
stats = {a: {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in [0.5, 0.7]} for a in ['0.0', '1.0']}
boxdir = BASE / 'P3A_20261014_120929_boxes'
expected_files = {f'sample_{sid}_alpha_{a}.npz' for sid in ids for a in stats}
actual_files = {p.name for p in boxdir.iterdir()}
check('1820_exact_filenames', actual_files == expected_files,
      {'actual': len(actual_files), 'expected': 1820, 'missing': sorted(expected_files - actual_files),
       'extra': sorted(actual_files - expected_files)})
counts = {a: {'records': 0, 'predictions': 0, 'gt': 0, 'empty_predictions': 0,
              'score_min': 1., 'score_max': 0.} for a in stats}
inventory = []
for index, sid in enumerate(p3[3]['sample_ids']):
    paired_gt = []
    for a in stats:
        path = boxdir / f'sample_{sid}_alpha_{a}.npz'
        sha = record(path)
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {'sample_idx', 'alpha', 'pred_boxes', 'pred_scores', 'gt_boxes'}:
                raise ValueError('Unexpected NPZ fields: ' + str(path))
            if z['sample_idx'].shape != () or str(z['sample_idx'].item()) != sid or z['alpha'].shape != () or float(z['alpha']) != float(a):
                raise ValueError('NPZ identity mismatch: ' + str(path))
            pred, score, gt = z['pred_boxes'], z['pred_scores'], z['gt_boxes']
            if pred.shape[1:] != (8, 3) or gt.shape[1:] != (8, 3) or score.shape != (len(pred),):
                raise ValueError('NPZ shape mismatch: ' + str(path))
            if not all(np.issubdtype(x.dtype, np.floating) and np.isfinite(x).all() for x in [pred, score, gt]):
                raise ValueError('NPZ invalid numbers: ' + str(path))
            if not ((score >= 0).all() and (score <= 1).all()):
                raise ValueError('Invalid confidence: ' + str(path))
            for array in [pred, gt]:
                for poly in geometry['convert_format'](array):
                    if not poly.is_valid or poly.area <= 0:
                        raise ValueError('Invalid BEV polygon: ' + str(path))
            for t in [0.5, 0.7]:
                evaluation['caluclate_tp_fp'](pred, score, gt, stats[a], t)
            paired_gt.append(gt.copy())
            c = counts[a]
            c['records'] += 1
            c['predictions'] += len(pred)
            c['gt'] += len(gt)
            c['empty_predictions'] += int(len(pred) == 0)
            if len(score):
                c['score_min'] = min(c['score_min'], float(score.min()))
                c['score_max'] = max(c['score_max'], float(score.max()))
            inventory.append({'path': str(path), 'sha256': sha, 'sample_idx': sid, 'alpha': float(a),
                              'predictions': len(pred), 'gt': len(gt)})
    if not np.array_equal(*paired_gt):
        raise ValueError('Paired GT mismatch: ' + sid)
    if (index + 1) % 100 == 0:
        print('CPU audit samples:', index + 1, flush=True)
check('all_npz_fields_shapes_finite_valid_and_paired_gt', True)
evidence['capture_counts'] = counts
evidence['recomputed'] = {}
for a in stats:
    evidence['recomputed'][a] = {}
    for t, key in [(0.5, 'ap50'), (0.7, 'ap70')]:
        ap = float(evaluation['calculate_ap'](stats[a], t)[0])
        ref = p1[3]['results'][a][key]
        evidence['recomputed'][a][key] = {'value': ap, 'p1c': ref, 'p3a': p3[3]['results'][a][key],
                                          'absolute_error': abs(ap - ref),
                                          'tp': sum(stats[a][t]['tp']), 'fp': sum(stats[a][t]['fp']),
                                          'gt': stats[a][t]['gt']}
        check('recomputed:' + a + ':' + key, abs(ap - ref) <= 1e-12)
save('capture_inventory.json', inventory)

# Re-establish P1 sample intersection and historical output completeness.
history = []
for p in sorted(BASE.glob('D2*.json')):
    rows = read_json(p)
    log = p.with_suffix('.log')
    record(log)
    text = log.read_text(errors='replace')
    history.append({'file': str(p), 'delays': [r['delay_label_ms'] for r in rows],
                    'valid_batches': [r['valid_batches'] for r in rows],
                    'sample_count_consistent': all(r['valid_batches'] == len(r['sample_ids']) == len(set(r['sample_ids'])) for r in rows),
                    'completion_marker': 'D2_PILOT_COMPLETE' in text,
                    'traceback': 'Traceback (most recent call last)' in text})
full_rows = sum([read_json(Path(p)) for p in manifest['source_files']], [])
check('P1_manifest_is_full_six_delay_intersection', set(ids) == set.intersection(*(set(r['sample_ids']) for r in full_rows)))
check('D2F_covers_six_delays', sorted(r['delay_label_ms'] for r in full_rows) == list(range(0, 600, 100)))
evidence['history_d2'] = history
evidence['logs'] = {}
for p in [BASE / 'P3A_20261014_120929.log', BASE / 'P1C_20261013_114845.log',
          Path('/workspace/lrcp_d1/D1_20261010_225833.log'),
          Path('/workspace/lrcp_d1/D1_20261010_224556.log'),
          Path('/workspace/lrcp_b0/Original_NoComp_20261010_200213/b0_inference.log'),
          Path('/workspace/lrcp_e1_logs/E1_20261008_100358.log'),
          Path('/workspace/lrcp_e1_logs/e1_stage2.log')]:
    record(p)
    lines = p.read_text(errors='replace').splitlines()
    relevant = [(i + 1, line) for i, line in enumerate(lines)
                if re.search(r'COMPLETE|Traceback|Error|WARNING|UserWarning|epoch|validation|saved|Training|Average Precision', line)]
    evidence['logs'][str(p)] = {'line_count': len(lines), 'relevant_tail': relevant[-35:],
                              'last_lines': [(i + 1, lines[i]) for i in range(max(0, len(lines)-6), len(lines))],
                              'traceback': any('Traceback (most recent call last)' in s for s in lines)}
check('P3_log_complete_no_traceback', not evidence['logs'][str(BASE / 'P3A_20261014_120929.log')]['traceback']
      and (BASE / 'P3A_20261014_120929.log').read_text().rstrip().endswith('D2_PILOT_COMPLETE'))

# Local timestamp is a frame counter: convert frame differences using the documented 5 Hz sampling rate.
pklpath = Path('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl')
samplepath = Path('/workspace/datasets/V2X-Sim-2.0/v2.0/sample.json')
record(pklpath)
with pklpath.open('rb') as f:
    data = pickle.load(f)
raw = read_json(samplepath)
rawmap = {x['token']: x for x in raw}
datamap = {x['token']: x for x in data}
freeze_path = REPO / 'summary_md/STAGE2_PRIMARY_RUN_FREEZE.md'
record(freeze_path)
record(REPO / 'summary_md/PAPER_CODE_RECONCILIATION_STAGE2.md')
check('documented_5Hz_mapping', 'Physical V2X-Sim reporting (5 Hz)' in freeze_path.read_text())
evidence['sampling_rate_basis'] = {'hz': 5, 'local_document': str(freeze_path), 'line': 18,
    'external_primary_source': 'https://arxiv.org/pdf/2202.08449',
    'interpretation': 'physical time inferred from verified frame chain and documented dataset sampling rate; raw timestamp is NOT seconds'}
time_results = []
for k in range(6):
    deltas = []
    for sid in ids:
        row = data[int(sid)]
        prev = row if k == 0 else datamap[row['prev_samples'][k]]
        r, q = rawmap[row['token']], rawmap[prev['token']]
        if r['scene_token'] != q['scene_token']:
            raise ValueError('Cross-scene predecessor')
        chain = r
        for _ in range(k):
            chain = rawmap[chain['prev']]
        if chain['token'] != q['token']:
            raise ValueError('prev_samples[k] is not k prev edges')
        if row['timestamp'] != r['timestamp'] or prev['timestamp'] != q['timestamp']:
            raise ValueError('Generated/raw timestamps disagree')
        deltas.append(float(r['timestamp']) - float(q['timestamp']))
    check('raw_timestamp_frame_delta:k' + str(k), np.allclose(deltas, k, rtol=0, atol=1e-9))
    time_results.append({'k': k, 'model_time_code': k * 100, 'samples': len(deltas),
                         'physical_interval_ms': k * 200, 'raw_timestamp_delta_min': min(deltas), 'raw_timestamp_delta_max': max(deltas)})
evidence['time_mapping'] = time_results
evidence['errors'] = errors
evidence['accepted'] = not errors
evidence['torch_imported'] = 'torch' in sys.modules
check('no_torch_imported', not evidence['torch_imported'])
evidence['errors'] = errors
evidence['accepted'] = not errors
record(Path(__file__))
save('audit_results.json', evidence)
print(json.dumps({'report_directory': str(RUN), 'accepted': evidence['accepted'], 'errors': errors,
                  'capture_counts': counts, 'recomputed': evidence['recomputed'],
                  'time_mapping': time_results}, indent=2, ensure_ascii=False), flush=True)
if errors:
    raise SystemExit(1)
