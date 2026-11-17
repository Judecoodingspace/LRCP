"""Portable CPU replay of the compact R5C upload evidence (no server originals).
This validates packaged stage facts, not omitted raw geometry or model causality.
Run after extraction: python -B validate_upload_cpu.py
No torch/opencood/device imports. Writes only a local validation JSON.
"""
import os,sys,csv,json,hashlib,importlib.abc
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):raise ImportError('OFFLINE_ONLY '+fullname)
sys.meta_path.insert(0,NoModel())
import numpy as np
from source_reference.offline_science import polys,overlap,matrix,nms_replay,unique_match,STAGES
CHECKS=0

def require(ok,name):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise RuntimeError(name)
def eq(a,b,n):require(np.array_equal(a,b),n)
def rows(n):return list(csv.DictReader((HERE/n).open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
 if (HERE/'SHA256SUMS.txt').is_file():
  for line in (HERE/'SHA256SUMS.txt').read_text().splitlines():
   h,n=line.split('  ',1);require(sha(HERE/n)==h,'PACKAGE_SHA '+n)
 config=json.loads((HERE/'R5C_C0_REPLAY_VALIDATION.json').read_text())['frozen_postprocess_config'];traces=rows('R5C_GT_STAGE_TRACE.csv');locator=rows('historical_reference/R5B_RAW_SHARD_LOCATOR.csv');countrows=rows('R5C_ALL_32_RESULT_REPLAY.csv');counts={(int(r['sample_idx']),r['condition']):r for r in countrows};nms_edges=0
 for loc in locator:
  sid=int(loc['sample_idx']);c=loc['condition'];key=sid,c
  with np.load(HERE/'evidence'/f'sample_{sid}_{c}_selected_stage_evidence.npz',allow_pickle=False) as z:a={n:z[n] for n in z.files}
  ci=a['candidate_ids'];pos={int(cid):i for i,cid in enumerate(ci)};raw=a['raw_corners'];score=a['scores_all_full'];io=a['IoU_GT_full'];selectedscore=a['scores_all_selected'];eq(score[ci],selectedscore,'selected scores')
  recomputed=matrix(a['GT_canonical'],raw);require(np.max(np.abs(recomputed-a['IoU_GT_selected']),initial=0)<=1e-6,'selected BEV geometry');eq(recomputed>=.5,a['IoU_GT_selected']>=.5,'high50');eq(recomputed>=.7,a['IoU_GT_selected']>=.7,'high70')
  required=set(np.flatnonzero(score>.2))|set(np.flatnonzero(np.any(io>=.5,axis=0)));require(required<=set(ci),'all high or scorepass geometry present')
  sc=np.flatnonzero(score>.2);eq(sc,a['ids_score_pass'],'all12800 score filter');boxes=raw[[pos[int(cid)] for cid in sc]];span=boxes.max(1)-boxes.min(1);qm=(span[:,0]<=6)&(span[:,1]<=6)&(span[:,1]!=0)&(boxes[:,:,2].min(1)>=-3)&(boxes[:,:,2].max(1)<=1);q=sc[qm];eq(q,a['ids_quality_pass'],'size z native quirk');eq(q[score[q].argsort()[::-1][:1000]],a['ids_nms_top1000'],'top1000')
  qb=raw[[pos[int(cid)] for cid in q]];keep,sup,ov=nms_replay(qb,score[q],config['nms_thresh']);kid=q[keep];eq(kid,a['ids_nms_keep'],'NMS keep')
  for child,owner in sup.items():
   cid=int(q[child]);oid=int(q[owner]);require(a['suppressor_selected'][pos[cid]]==oid,'suppressor ID');require(float(a['suppressor_overlap_selected'][pos[cid]])==ov[child],'suppressor overlap');nms_edges+=1
  kb=raw[[pos[int(cid)] for cid in kid]];lim=np.asarray(config['gt_range']);mask=np.all((kb>=lim[:3])&(kb<=lim[3:]),axis=2).sum(1)>=8;final=kid[mask];eq(final,a['ids_final_range'],'final range IDs');eq(raw[[pos[int(cid)] for cid in final]],a['final_corners'],'exact final geometry');eq(score[final],a['final_scores'],'exact final score')
  for t in (50,70):
   m=unique_match(a['final_corners'],a['final_scores'],a['GT_canonical'],t/100);eq(m['matched'],a[f'prediction_gt_idx{t}'],'GT unique matches');eq(m['flags'],a[f'gt_tp{t}'],'GT flags');eq(m['matched']<0,a[f'prediction_fp{t}'],'FP flags');tp=int(sum(m['tp']));fp=len(final)-tp;fn=len(a['GT_canonical'])-tp
   require((tp,fp,fn)==tuple(int(counts[key][n+str(t)]) for n in ('TP','FP','FN')),'all32 counters')
   for r in traces:
    if (int(r['sample_idx']),r['condition'],float(r['iou_threshold']))!=(sid,c,t/100):continue
    gi=int(r['gt_idx']);high=io[gi]>=t/100
    for s in STAGES:
     expected=a['ids_'+s][high[a['ids_'+s]]].astype(int).tolist();require(json.loads(r['high_IoU_ids_'+s])==expected and int(r['high_IoU_count_'+s])==len(expected),'GT all12800 high counts')
    p=np.flatnonzero(m['matched']==gi)
    if len(p):
     pi=int(p[0]);cid=int(final[pi]);require(int(r['matched_candidate_id'])==cid and float(r['matched_score'])==float(score[cid]) and abs(float(r['matched_IoU'])-float(io[gi,cid]))<=1e-6,'actual matched fields')
    else:require(all(r[n]=='' for n in ['matched_candidate_id','matched_prediction_index','matched_IoU','matched_score']),'FN fields must be missing')
 result={'state':'UPLOAD_EVIDENCE_REPLAY_PASS','checks':CHECKS,'records':len(locator),'NMS_suppressor_edges':nms_edges,'all32_final_and_GT_replayed':True,'scope_limit':'Omitted low-score low-IoU raw geometry, full head tensors, original input/checkpoint/provenance and historical budget require server-side validate_r5c_independent.py and original SHA-bound files. Full12800 GT-IoU matrices are verified against original NPZ by server validator; this portable replay recomputes geometry only for exported candidates.','new_forward':0,'CUDA_initializations':0,'torch_imported':'torch' in sys.modules,'opencood_imported':'opencood' in sys.modules}
 print(json.dumps(result,ensure_ascii=False));return result
if __name__=='__main__':
 run()
