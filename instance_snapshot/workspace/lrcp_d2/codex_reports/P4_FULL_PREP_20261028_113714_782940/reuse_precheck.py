"""CPU independent inspection of prior 36 saved core conditions. No model."""
import sys,json,hashlib,time,traceback
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent
P=R.parent/'P4_PILOT_RETRY_20261023_234457_384594'
C=R.parent/'P4_C1_20261027_162059_722622'
F=R.parent/'P4_CPU_VALIDATION_FIX_20261025_111314_916690'
sys.path.insert(0,str(C))
from c1_common import make_pp,anchor_input,load_pilot,postprocess,eval_native,iou_matrix,STAGES,sha_file
from contracts import replay_native_nms
from opencood.utils import common_utils

def run():
 init=time.monotonic();prep=json.loads((P/'INPUT_PREPARATION.json').read_text());past=json.loads((P/'PPU_EXECUTION.json').read_text());follow=json.loads((F/'COMPLETION.json').read_text());worker=json.loads((P/'WORKER_FINAL.json').read_text())
 assert past['passed'] and past['actual_inference_on_PPU'] and past['full_forward_completed']==40 and follow['offline_validation_state']=='PASSED'
 assert worker['state']=='FAILED' and worker['exit_code']==1
 pp=make_pp();rows=[]
 for item in sorted(prep['samples'],key=lambda x:x['sample_idx']):
  sid=item['sample_idx'];inputs=anchor_input(sid,pp);pairs={}
  assert sha_file(item['path'])==item['sha256']
  for alpha in [0,1]:
   f=P/f'diagnostics/core_sample_{sid}_alpha_{alpha:.1f}.npz';meta=json.loads(f.with_suffix('.json').read_text())
   with np.load(f,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
   assert meta['alpha']==alpha and meta['sample_idx']==sid and meta['kind']=='core' and meta['input_hash']==item['input_hash']
   for key,shape in [('cls_preds',(1,2,80,80)),('reg_preds',(1,14,80,80)),('dir_preds',(1,4,80,80)),('raw_corners',(12800,8,3)),('scores_all',(12800,))]:assert a[key].shape==shape and a[key].dtype==np.float32 and np.isfinite(a[key]).all()
   for key in a:
    if key.startswith('offset_raw_slot'):
     eff='offset_effective_slot'+key.split('offset_raw_slot')[1];assert np.array_equal(a[eff],a[key]*alpha)
   old=P.parent.parent/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz'
   with np.load(old,allow_pickle=False) as z:
    assert np.array_equal(a['final_corners'],z['pred_boxes']) and np.array_equal(a['final_scores'],z['pred_scores'])
    assert np.allclose(a['GT_canonical'],z['gt_boxes'],rtol=0,atol=1e-5)
   shadow=replay_native_nms(a['nms_input_corners'],a['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
   assert np.array_equal(shadow['keep'],a['native_nms_keep'])
   replay,stat=postprocess(pp,inputs,load_pilot(sid,alpha),load_pilot(sid,alpha))
   for k in ['score_mask','large_mask','z_mask','quality_mask','native_nms_keep','range_mask']+['ids_'+s for s in STAGES]:assert np.array_equal(replay[k],a[k]),(sid,alpha,k)
   for k in ['decoded_before_dir','raw_corners','final_corners']:
    assert np.allclose(replay[k],a[k],rtol=0,atol=1e-5),(sid,alpha,k)
   assert np.allclose(replay['scores_all'],a['scores_all'],rtol=0,atol=1e-6)
   baseline=C/f'baseline/sample_{sid}_G{alpha}S{alpha}.npz'
   with np.load(baseline,allow_pickle=False) as z:
    assert np.array_equal(replay['final_corners'],z['final_corners']) and np.array_equal(replay['final_scores'],z['final_scores'])
    assert np.array_equal(replay['GT_TP70'],z['GT_TP70']) and np.array_equal(replay['matched_gt_prediction'],z['matched_gt_prediction'])
    m=iou_matrix(a['GT_canonical'],a['raw_corners']);assert np.array_equal(m>=.7,z['IoU_GT_raw']>=.7)
   pairs[alpha]=a
   rows.append({'sample_idx':sid,'alpha':alpha,'pilot_npz':str(f),'pilot_npz_sha256':sha_file(f),'meta_sha256':sha_file(f.with_suffix('.json')),'head_shape_dtype_finite':True,'input_hash_matches_PPU':True,'alpha_offset_scaling_exact':True,'CPU_postprocess_stage_lineage_equal':True,'native_NMS_replay_exact':True,'historical_P3A_boxes_scores_exact':True,'C1_GT_match_and_raw_thresholds_equal':True})
  for k in pairs[0]:
   if k.startswith('offset_raw_slot'):assert k in pairs[1] and np.array_equal(pairs[0][k],pairs[1][k])
  print(json.dumps({'sample_idx':sid,'conditions_verified':len(rows),'elapsed_seconds':time.monotonic()-init}),flush=True)
 assert len(rows)==36 and not torch.cuda.is_initialized()
 (R/'REUSE_PRECHECK.json').write_text(json.dumps({'passed':True,'conditions':rows,'condition_count':36,'pilot_worker_FAILED_exit1_preserved':True,'requires_new_18_batch_equivalence':True,'model_forwards':0,'CUDA_initialized':False,'elapsed_seconds':time.monotonic()-init},indent=2))
if __name__=='__main__':
 try:run()
 except BaseException:
  (R/'REUSE_PRECHECK_FAILURE.json').write_text(json.dumps({'passed':False,'traceback':traceback.format_exc()},indent=2));raise
