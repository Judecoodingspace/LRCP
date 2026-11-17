"""Portable full-candidate R5D replay, no original server NPZ required.
Requires NumPy + Shapely; stdout only. No torch/opencood or device operations.
"""
import os,sys,json,csv,hashlib,importlib.abc
from pathlib import Path
ROOT=Path(__file__).resolve().parent;os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ['torch','opencood','cupy','pycuda','pynvml']:raise ImportError('CPU_SAVED_OUTPUT_REPLAY_ONLY')
sys.meta_path.insert(0,NoModels())
import numpy as np
from source_reference.offline_science import matrix,nms_replay,unique_match,STAGES
CHECK=0

def ck(ok,n):
 global CHECK
 CHECK+=1
 if not bool(ok):raise RuntimeError(n)
def eq(a,b,n):ck(np.array_equal(a,b),n)
def rows(n):return list(csv.DictReader((ROOT/n).open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 if (ROOT/'SHA256SUMS.txt').is_file():
  for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
   h,n=line.split('  ',1);ck(sha(ROOT/n)==h,'BUNDLE_FILE_HASH '+n)
 conf=json.loads((ROOT/'historical_reference/R5C_C0_REPLAY_VALIDATION.json').read_text())['frozen_postprocess_config'];design=json.loads((ROOT/'R5D_DESIGN_FROZEN.json').read_text());sources={};iou={};cnt={};result={}
 for p in design['pairs']:
  for c in [p['condition0'],p['condition1']]:
   key=p['sample_idx'],c
   if key in sources:continue
   with np.load(ROOT/'evidence/source'/f'sample_{key[0]}_{c}.npz',allow_pickle=False) as z:sources[key]={n:z[n] for n in z.files}
   a=sources[key];eq(a['ids_raw_decoded'],np.arange(12800),'source full IDs');iou[key]=matrix(a['GT_canonical'],a['raw_corners']);ck(np.max(np.abs(iou[key]-a['IoU_GT_raw']),initial=0)<=1e-6,'source full BEV IoU')
 for r in rows('R5D_ALL_RUN_RESULTS.csv'):
  rid=r['run_id'];sid=int(r['sample_idx']);gc=r['geometry_condition'];sc=r['score_condition'];g=sources[sid,gc];score=sources[sid,sc]['scores_all'].copy();raw=g['raw_corners'];gt=g['GT_canonical']
  if rid.startswith('SWAP_'):ck(score[7028]>.2 and score[7030]>.2,'swap score threshold');score[[7028,7030]]=score[[7030,7028]]
  with np.load(ROOT/'evidence/runs'/(rid+'.npz'),allow_pickle=False) as z:a={n:z[n] for n in z.files}
  eq(a['raw_corners'],raw,'actual geometry source');eq(a['scores_all'],score,'actual score source');eq(a['GT_canonical'],gt,'actual GT source');eq(a['ids_raw_decoded'],np.arange(12800),'run no candidates lost')
  sm=score>conf['target_args']['score_threshold'];ids=np.flatnonzero(sm);bb=raw[ids];span=bb.max(1)-bb.min(1);lm=(span[:,0]<=6)&(span[:,1]<=6)&(span[:,1]!=0);zm=(bb[:,:,2].min(1)>=-3)&(bb[:,:,2].max(1)<=1);q=ids[lm&zm];top=q[score[q].argsort()[::-1][:1000]];pick,sup,ov=nms_replay(raw[q],score[q],conf['nms_thresh']);kept=q[pick];sr=np.full(12800,-1,np.int32);ovs=np.zeros(12800,np.float32)
  for child,owner in sup.items():sr[q[child]]=q[owner];ovs[q[child]]=ov[child]
  lim=np.asarray(conf['gt_range']);kb=raw[kept];rg=np.all((kb>=lim[:3])&(kb<=lim[3:]),axis=2).sum(1)>=8;final=kept[rg];stages=[np.arange(12800),ids,q,top,kept,final]
  for n,x in zip(STAGES,stages):eq(a['ids_'+n],x,'native stage '+n)
  for n,x in [('score_mask',sm),('large_mask',lm),('z_mask',zm),('quality_mask',lm&zm),('native_nms_keep',pick),('range_mask',rg),('suppressor',sr),('suppressor_overlap',ovs),('final_corners',raw[final]),('final_scores',score[final])]:eq(a[n],x,n)
  for t in (50,70):
   m=unique_match(raw[final],score[final],gt,t/100);eq(a[f'prediction_gt_idx{t}'],m['matched'],'native greedy GT match');eq(a[f'gt_tp{t}'],m['flags'],'native GT status');eq(a[f'prediction_fp{t}'],m['matched']<0,'native FP status');tp=int(m['flags'].sum());fp=len(final)-tp;fn=len(gt)-tp;ck((tp,fp,fn)==tuple(int(r[n+str(t)]) for n in ['TP','FP','FN']),'count result');cnt[rid,t]=(tp,fp,fn)
  if r['experiment_type']=='ORIGINAL_BASELINE':
   for n in ['ids_final_range','final_corners','final_scores','suppressor','suppressor_overlap','gt_tp50','gt_tp70']:eq(a[n],g[n],'natural original baseline exact')
  result[rid]=a
 for r in rows('R5D_FOCAL_GT_OUTCOMES.csv'):
  a=result[r['run_id']];gi=int(r['gt_idx']);t=int(round(float(r['iou_threshold'])*100));pp=np.flatnonzero(a[f'prediction_gt_idx{t}']==gi);iv=iou[int(r['sample_idx']),r['geometry_condition']][gi]
  for s in STAGES:
   h=a['ids_'+s][iv[a['ids_'+s]]>=t/100].astype(int).tolist();ck(h==json.loads(r['high_ids_'+s]) and len(h)==int(r['high_count_'+s]),'focal full candidate survival')
  if len(pp):
   pi=int(pp[0]);cid=int(a['ids_final_range'][pi]);ck(r['TP_FN']=='TP' and int(r['matched_candidate_id'])==cid and float(r['matched_IoU'])==float(iv[cid]) and float(r['matched_score'])==float(a['scores_all'][cid]),'focal actual matched values')
  else:ck(r['TP_FN']=='FN' and all(r[n]=='' for n in ['matched_IoU','matched_score','matched_candidate_id','matched_prediction_index']),'FN fields missing')
 for r in rows('R5D_GEOMETRY_SCORE_INTERACTION.csv'):
  p=r['pair_id'];t=int(round(float(r['iou_threshold'])*100))
  for i,n in enumerate(['TP','FP','FN']):
   v={c:cnt[p+'_'+c,t][i] for c in ['G0_S0','G0_S1','G1_S0','G1_S1']};ck(int(r['I_GS_'+n])==v['G1_S1']-v['G1_S0']-v['G0_S1']+v['G0_S0'],'factorial interaction')
 ck(len(result)==19 and len(sources)==6,'exact design counts')
 print(json.dumps({'state':'R5D_UPLOAD_REPLAY_PASS','checks':CHECK,'full_raw_source_records':6,'full12800_postprocessing_runs_replayed':19,'natural_baselines_exact':8,'factorial_runs':16,'score_swap_runs':3,'new_model_forward':0,'CUDA_initializations':0,'torch_imported':'torch' in sys.modules,'opencood_imported':'opencood' in sys.modules,'scope_limit':'All full raw candidate geometry and scores needed for the19 runs are packaged. Original NPZ bytes with unrelated attention/feature tensors, frozen input/checkpoint and historical provenance SHA checks require server originals.'},ensure_ascii=False))
if __name__=='__main__':main()
