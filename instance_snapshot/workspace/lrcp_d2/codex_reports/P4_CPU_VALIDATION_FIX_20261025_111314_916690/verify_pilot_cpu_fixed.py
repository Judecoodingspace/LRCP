"""Final independent read-back: native evaluator, stage CSVs, every anchor, hashes.
This CPU program is invoked only after the user-started finite PPU collection.
"""
import csv,json,time,collections,signal
import numpy as np
import sys
import torch
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P4_PILOT_RETRY_20261023_234457_384594')
from pilot_common import *
OUT=Path(__file__).resolve().parent
from eval_adapter import cpu_tensor_inputs

def verify():
 signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('CPU_REVALIDATION_TIMEOUT')));signal.alarm(1800)
 assert not torch.cuda.is_initialized()
 start=time.monotonic();state=json.load((R/'STATE.json').open());exe=json.load((R/'PPU_EXECUTION.json').open());acc=json.load((R/'ACCEPTANCE.json').open())
 assert exe['passed'] and exe['actual_inference_on_PPU'] and exe['full_forward_attempted']==exe['full_forward_completed']==40
 assert len(set(exe['samples']))==18 and set(exe['samples'])=={int(x['sample_idx']) for x in selections()}
 assert state['phase']=='E_COMPLETE' and acc['passed'] and acc['GT_count']==162
 assert json.load((R/'SMOKE_TEST.json').open())['passed']
 for a in [0.,1.]:
  assert json.load((R/f'SMOKE_EQUIVALENCE_alpha_{a:.1f}.json').open())['passed'];assert json.load((R/f'REVERSE_EQUIVALENCE_alpha_{a:.1f}.json').open())['passed']
 from opencood.utils import common_utils,eval_utils
 outdir=R/'diagnostics'/'analysis';stage_rows=list(csv.DictReader((outdir/'per_GT_stages.csv').open()));summary_rows=list(csv.DictReader((outdir/'per_GT_summary.csv').open()))
 assert len(stage_rows)==1944 and len(summary_rows)==324
 indexed={(int(x['sample_idx']),float(x['alpha']),int(x['gt_idx']),x['stage']):x for x in stage_rows};assert len(indexed)==1944
 checks=[];hashes={}
 for selection in selections():
  sid=int(selection['sample_idx'])
  for alpha in [0.,1.]:
   budget(start,1800);prefix=f'core_sample_{sid}_alpha_{alpha:.1f}'
   p=R/'diagnostics'/(prefix+'.npz')
   with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
   hashes[str(p)]=sha_file(p);gt=d['GT_canonical'];pred=d['final_corners'];score=d['final_scores']
   _,_,historical_validation=verify_historical(sid,alpha,pred,score,d['GT_runtime'])
   result={.7:{'score':[],'tp':[],'fp':[],'gt':0}}
   eval_utils.caluclate_tp_fp(*cpu_tensor_inputs(pred,score,gt),result,.7)
   persisted=[x for x in summary_rows if int(x['sample_idx'])==sid and float(x['alpha'])==alpha]
   assert sum(result[.7]['tp'])==sum(int(x['final_TP70']) for x in persisted)
   assert result[.7]['gt']==len(persisted)
   polygons=list(common_utils.convert_format(d['raw_corners']));gpolys=common_utils.convert_format(gt)
   lows=d['raw_corners'][:,:4,:2].min(1);highs=d['raw_corners'][:,:4,:2].max(1)
   for gi,poly in enumerate(gpolys):
    # Independent per-GT stage calculation from saved raw corners, without attribute_gt().
    lo=gt[gi,:4,:2].min(0);hi=gt[gi,:4,:2].max(0)
    relevant=np.flatnonzero(np.all(highs>lo,1)&np.all(lows<hi,1));values=np.zeros(12800,np.float32)
    values[relevant]=common_utils.compute_iou(poly,[polygons[i] for i in relevant])
    for stage in STAGES:
     row=indexed[(sid,alpha,gi,stage)];ids=d['ids_'+stage];expected=ids[values[ids]>=.7]
     assert json.loads(row['high_iou_anchor_ids'])==expected.tolist()
     assert int(row['high_iou_count'])==len(expected)
     best=float(values[ids].max()) if len(ids) else 0.
     assert float(row['best_iou'])==best
   assert len(d['ids_raw_decoded'])==12800
   assert np.array_equal(d['score_mask'],d['scores_all']>.2)
   assert np.array_equal(d['raw_corners'][d['ids_final_range']],pred)
   assert np.array_equal(d['scores_all'][d['ids_final_range']],score)
   checks.append({'sample_idx':sid,'alpha':alpha,'native_evaluator_TP70':sum(result[.7]['tp']),'native_evaluator_FP70':sum(result[.7]['fp']),'GT_count':len(gt),'all_stage_rows_recomputed':True,'historical':historical_validation})
 count=0;seen=set()
 with (outdir/'per_anchor_deletion.csv').open() as f:
  for row in csv.DictReader(f):
   key=(int(row['sample_idx']),row['alpha'],int(row['anchor_id']));assert key not in seen;seen.add(key);count+=1
 assert count==460800
 protected=json.load((R/'PROTECTED_HASHES_BEFORE.json').open())
 for p,h in protected.items():assert sha_file(p)==h,p
 for p,h in json.load((R/'RAW_INPUT_HASHES.json').open()).items():assert sha_file(p)==h,p
 assert bytes_in(R)-bytes_in(R/'batches')<=2**30
 verdict={'passed':True,'sample_count':18,'GT_count':162,'full_model_forward_count':40,'PPU_used':True,'protected_hashes_unchanged':True,'protected_file_count':len(protected),'native_evaluator_checks':checks,'anchor_rows':count,'stage_rows':1944,'core_output_hashes':hashes,'elapsed_seconds':time.monotonic()-start}
 assert not torch.cuda.is_initialized()
 verdict.update(CUDA_initialized=False,additional_model_forwards=0,source_worker_exit_code=1,validation_scope='offline follow-up; original worker FAILED state preserved',source_run_directory=str(R))
 write_json(OUT/'INDEPENDENT_VALIDATION.json',verdict)
 with (OUT/'RESULT_ARTIFACTS.sha256').open('x') as f:
  for p in sorted(R.rglob('*')):
   if p.is_file() and (p.is_relative_to(R/'diagnostics') or p.name in ['PPU_EXECUTION.json','ACCEPTANCE.json','INDEPENDENT_VALIDATION.json','ANALYSIS_SUMMARY.json']):f.write(sha_file(p)+'  '+str(p)+'\n')
 signal.alarm(0)
 print('INDEPENDENT_VALIDATION_PASSED',flush=True)
if __name__=='__main__':
 try:verify()
 except BaseException:
  import traceback
  write_json(OUT/'INDEPENDENT_VALIDATION_FAILURE.json',{'passed':False,'traceback':traceback.format_exc()});raise
