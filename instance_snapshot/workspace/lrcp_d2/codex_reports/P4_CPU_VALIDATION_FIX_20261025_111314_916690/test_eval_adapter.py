"""Regression test using real saved detections and the native LRCP evaluator."""
import sys,json,ast,random,resource
from pathlib import Path
import numpy as np
import torch
S=Path('/workspace/lrcp_d2/codex_reports/P4_PILOT_RETRY_20261023_234457_384594');O=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from pilot_common import digest,write_json,iou_matrix,greedy
from eval_adapter import cpu_tensor_inputs
from opencood.utils import eval_utils
checks=[]
def check(n,v):assert v,n;checks.append(n)
for p in O.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec');check('AST '+p.name,True)
with np.load(S/'diagnostics/core_sample_821_alpha_0.0.npz',allow_pickle=False) as z:pred=z['final_corners'].copy();scores=z['final_scores'].copy();gt=z['GT_canonical'].copy()
def stat():return {.7:{'score':[],'tp':[],'fp':[],'gt':0}}
before=digest((pred,scores,gt));rng_before=digest((random.getstate(),np.random.get_state(),torch.random.get_rng_state()))
try:eval_utils.caluclate_tp_fp(pred,scores,gt,stat(),.7);raise AssertionError('Expected old caller failure')
except AttributeError as e:check('old_numpy_caller_failure_reproduced','is_cuda' in str(e))
tensors=cpu_tensor_inputs(pred,scores,gt)
check('actual_CPU_tensors',all(torch.is_tensor(t) and not t.is_cuda for t in tensors))
check('all_values_shapes_dtypes_preserved',all(np.array_equal(t.numpy(),a) and t.numpy().dtype==a.dtype and tuple(t.shape)==a.shape for t,a in zip(tensors,[pred,scores,gt])))
check('RNG_unchanged',rng_before==digest((random.getstate(),np.random.get_state(),torch.random.get_rng_state())))
r=stat();eval_utils.caluclate_tp_fp(*tensors,r,.7)
flags,matched=greedy(iou_matrix(gt,pred),scores);ordered=(matched[np.argsort(-scores)]>=0).astype(int).tolist()
check('native_TP_sequence_equals_audited_greedy',r[.7]['tp']==ordered)
check('native_FP_sequence_equals_audited_greedy',r[.7]['fp']==[1-x for x in ordered])
check('native_score_order_unchanged',r[.7]['score']==scores[np.argsort(-scores)].tolist())
r0=stat();eval_utils.caluclate_tp_fp(*cpu_tensor_inputs(pred[:0],scores[:0],gt),r0,.7)
check('empty_predictions',r0[.7]['tp']==[] and r0[.7]['fp']==[] and r0[.7]['gt']==len(gt))
r1=stat();eval_utils.caluclate_tp_fp(*cpu_tensor_inputs(pred,scores,gt[:0]),r1,.7)
check('empty_GT_all_FP',sum(r1[.7]['tp'])==0 and sum(r1[.7]['fp'])==len(pred))
check('source_arrays_unmodified',before==digest((pred,scores,gt)))
check('CUDA_not_initialized',not torch.cuda.is_initialized())
check('peak_RSS_within_8GiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30)
write_json(O/'REGRESSION_TESTS.json',{'passed':True,'checks_count':len(checks),'checks':checks,'sample_idx':821,'alpha':0.,'native_TP70':sum(r[.7]['tp']),'native_FP70':sum(r[.7]['fp']),'CUDA_initialized':False,'model_forwards':0})
print('REGRESSION_TESTS_PASSED',len(checks),flush=True)
