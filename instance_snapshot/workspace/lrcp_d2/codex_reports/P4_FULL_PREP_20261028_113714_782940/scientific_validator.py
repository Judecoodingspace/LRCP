"""Future CPU-only independent scientific acceptance for saved native captures."""
import os,sys,json,csv,time,resource,traceback,signal,shutil
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
from pilot_common import R,BASE,STAGES,sha_file,iou_matrix
from contracts import replay_native_nms,attribute_gt
from opencood.utils import common_utils,eval_utils
C=R.parent/'P4_C1_20261027_162059_722622';B=R.parent/'P3B_20261020_103959_980233'
sys.path.insert(0,str(C))
from c1_common import eval_native,make_pp,postprocess
REASONS=['unused','score_filter','size_or_z_filter','nms_top1000_truncation','nms_overlap_suppression','post_nms_range_filter','final_survived']
def check_case(sid,alpha,path,records,input_path=None):
 keys=['cls_preds','reg_preds','dir_preds','raw_corners','scores_all','score_mask','quality_mask','native_nms_keep','nms_input_corners','nms_input_scores','range_mask','GT_canonical','final_corners','final_scores']+['ids_'+s for s in STAGES]
 with np.load(path,allow_pickle=False) as z:
  a={k:z[k].copy() for k in keys}
  for k in ['suppressor','suppressor_overlap','first_deletion_code']:
   if k in z:a[k]=z[k].copy()
 if input_path is None:
  pilot=R.parent/'P4_PILOT_RETRY_20261023_234457_384594'
  prep=json.loads((pilot/'INPUT_PREPARATION.json').read_text())
  input_path=next(z['path'] for z in prep['samples'] if z['sample_idx']==sid)
 payload=torch.load(input_path,map_location='cpu',weights_only=False,mmap=False)
 ego=payload['batch']['ego'];inputs={'ego':{'anchor_box':ego['anchor_box'].clone(),'transformation_matrix':ego['transformation_matrix'].clone()}}
 pp=make_pp();replayed,_=postprocess(pp,inputs,a,a)
 for k in ['score_mask','quality_mask','native_nms_keep','range_mask']+['ids_'+s for s in STAGES]:
  if not np.array_equal(replayed[k],a[k]):raise RuntimeError('CPU_POSTPROCESS_STAGE_MISMATCH '+k)
 for k in ['raw_corners','final_corners']:
  if not np.allclose(replayed[k],a[k],rtol=0,atol=1e-5):raise RuntimeError('CPU_POSTPROCESS_GEOMETRY_MISMATCH '+k)
 if not np.allclose(replayed['scores_all'],a['scores_all'],rtol=0,atol=1e-6):raise RuntimeError('CPU_POSTPROCESS_SCORE_MISMATCH')
 del payload,inputs
 oldpath=BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz'
 with np.load(oldpath,allow_pickle=False) as z:
  if not np.allclose(a['GT_canonical'],z['gt_boxes'],rtol=0,atol=1e-5):raise RuntimeError('GT_MISMATCH')
  for key,oldkey,tol in [('final_corners','pred_boxes',1e-5),('final_scores','pred_scores',1e-6)]:
   if a[key].shape!=z[oldkey].shape or not np.allclose(a[key],z[oldkey],rtol=0,atol=tol):raise RuntimeError('P3A_FINAL_MISMATCH '+key)
 q=np.flatnonzero(a['score_mask'])[a['quality_mask']];shadow=replay_native_nms(a['nms_input_corners'],a['nms_input_scores'],.15,common_utils.convert_format,common_utils.compute_iou)
 if not np.array_equal(shadow['keep'],a['native_nms_keep']):raise RuntimeError('NMS_KEEP_MISMATCH')
 ids=[np.arange(12800),np.flatnonzero(a['score_mask']),q,q[a['nms_input_scores'].argsort()[::-1][:1000]],q[a['native_nms_keep']],q[a['native_nms_keep']][a['range_mask']]]
 for stage,v in zip(STAGES,ids):
  if not np.array_equal(v,a['ids_'+stage]):raise RuntimeError('STAGE_ID_MISMATCH '+stage)
 if not np.array_equal(a['raw_corners'][ids[-1]],a['final_corners']) or not np.array_equal(a['scores_all'][ids[-1]],a['final_scores']):raise RuntimeError('FINAL_LINEAGE_MISMATCH')
 sup=np.full(12800,-1,np.int32);overlap=np.zeros(12800,np.float32)
 for z in shadow['suppressed']:
  ai=q[z['removed_nms_input_idx']];sup[ai]=q[z['suppressor_nms_input_idx']];overlap[ai]=z['bev_iou_float32']
 reason=np.ones(12800,np.int8)
 for v,code in zip(ids[1:],range(2,7)):reason[v]=code
 for key,expected in [('suppressor',sup),('suppressor_overlap',overlap),('first_deletion_code',reason)]:
  if key in a and not np.array_equal(a[key],expected):raise RuntimeError('ANCHOR_DELETION_CHAIN_MISMATCH '+key)
 stats={};flags={};match={}
 for threshold in [.5,.7]:
  stats[threshold],match[threshold],flags[threshold]=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],threshold)
  expected=np.array([int(row['tp'+str(int(threshold*100))+'_a'+str(int(alpha))]) for row in records],dtype=np.int8)
  if not np.array_equal(flags[threshold],expected):raise RuntimeError('HISTORICAL_GT_MATCH_MISMATCH '+str(threshold))
  for pred,gi in enumerate(match[threshold]):
   if gi>=0 and int(records[gi]['matched_pred_idx'+str(int(threshold*100))+'_a'+str(int(alpha))])!=pred:raise RuntimeError('HISTORICAL_PREDICTION_MATCH_ID_MISMATCH')
 mat=iou_matrix(a['GT_canonical'],a['raw_corners']);cpu_mat=iou_matrix(a['GT_canonical'],replayed['raw_corners']);
 if not np.array_equal(mat>=.7,cpu_mat>=.7) or not np.array_equal(mat>=.5,cpu_mat>=.5):raise RuntimeError('CPU_GPU_RAW_IOU_THRESHOLD_MISMATCH')
 attr=attribute_gt(mat,{stage:v for stage,v in zip(STAGES,ids)})
 if len(attr)!=len(records):raise RuntimeError('GT_COUNT_MISMATCH')
 for gi,record in enumerate(records):
  best=float(mat[gi,ids[-1]].max(initial=0))
  expected=float(record['best_iou_a'+str(int(alpha))])
  for threshold in [.5,.7]:
   if (best>=threshold)!=(expected>=threshold):raise RuntimeError('HISTORICAL_BEST_IOU_THRESHOLD_MISMATCH')
 out=[];events=[]
 for gi,rec in enumerate(attr):
  raw=np.flatnonzero(mat[gi]>=.7);row={'sample_idx':sid,'gt_idx':gi,'alpha':int(alpha),'TP50':int(flags[.5][gi]),'TP70':int(flags[.7][gi]),'last_strict_disappearance':rec['availability_class']}
  for stage in STAGES:row['count_'+stage]=rec['high_iou_count'][stage];row['ids_'+stage]=json.dumps([int(v) for v in a['ids_'+stage] if mat[gi,v]>=.7])
  out.append(row)
  for ai in raw:
   s=int(sup[ai]);events.append({'sample_idx':sid,'gt_idx':gi,'alpha':int(alpha),'anchor_id':int(ai),'IoU_GT':float(mat[gi,ai]),'score':float(a['scores_all'][ai]),'first_deletion':REASONS[reason[ai]],'suppressor':s,'suppressor_score':float(a['scores_all'][s]) if s>=0 else '', 'suppressor_IoU_GT':float(mat[gi,s]) if s>=0 else '', 'suppression_overlap':float(overlap[ai])})
 return stats,out,events

def run():
 if not (R/'PPU_CAPTURE_COMPLETE.json').exists():raise RuntimeError('CAPTURE_NOT_COMPLETE')
 cap=json.loads((R/'PPU_CAPTURE_COMPLETE.json').read_text());assert cap['passed'] and cap['attempted']==1790 and cap['completed']==1790 and cap['core_new']==1784 and cap['guards']==6 and cap['completed_samples']==910
 if (R/'CPU_VALIDATION_STARTED.json').exists():raise RuntimeError('REFUSE_AUTOMATIC_VALIDATION_RETRY')
 (R/'CPU_VALIDATION_STARTED.json').write_text(json.dumps({'epoch':time.time(),'pid':os.getpid()}));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU_VALIDATION_TIMEOUT')));signal.alarm(7200)
 manifest=json.loads((R/'INPUT_MANIFEST.json').read_text());input_by_sid={z['sample_idx']:z['path'] for z in manifest['samples']}
 p3rows=list(csv.DictReader((B/'per_gt.csv').open()));by_sid={}
 for row in p3rows:by_sid.setdefault(int(row['sample_idx']),[]).append(row)
 for sid in by_sid:by_sid[sid].sort(key=lambda z:int(z['gt_idx']))
 totals={a:{t:{'tp':[],'fp':[],'score':[],'gt':0} for t in [.5,.7]} for a in [0,1]};gt_seen={};gtcount=0;eventcount=0;casecount=0;predcount={0:0,1:0};stage_counts={a:{} for a in [0,1]}
 gtfile=(R/'per_GT_full.csv.tmp').open('x',newline='');eventfile=(R/'strict_anchor_lineage_full.csv.tmp').open('x',newline='');gwriter=None;ewriter=None
 try:
  for shard in range(29):
   folder=R/'capture'/f'shard_{shard:03d}';manifest_file=folder/'COLLECTION_MANIFEST.json'
   if sha_file(manifest_file)!=cap['shard_manifest_hashes'][str(shard)]:raise RuntimeError('SHARD_MANIFEST_HASH_MISMATCH')
   man=json.loads(manifest_file.read_text());assert man['passed']
   for record in sorted(man['records'],key=lambda x:(x['sample_idx'],x['alpha'])):
    sid=int(record['sample_idx']);alpha=int(record['alpha']);p=Path(record['path'])
    if sha_file(p)!=record['sha256'] or sha_file(record['meta_path'])!=record['meta_sha256']:raise RuntimeError('CAPTURE_FILE_HASH_MISMATCH')
    stat,rows,events=check_case(sid,alpha,p,by_sid[sid],input_by_sid[sid]);casecount+=1;predcount[alpha]+=len(stat[.7]['score'])
    for t in [.5,.7]:
     for key in ['tp','fp','score']:totals[alpha][t][key].extend(stat[t][key])
     totals[alpha][t]['gt']+=stat[t]['gt']
    for row in rows:
     key=(sid,row['gt_idx']);gt_seen.setdefault(key,{})[alpha]=row['TP70'];gtcount+=1
     group=by_sid[sid][row['gt_idx']]['match_transition70'];stage_counts[alpha].setdefault(group,Counter())[row['last_strict_disappearance']]+=1
     if gwriter is None:gwriter=csv.DictWriter(gtfile,fieldnames=list(row));gwriter.writeheader()
     gwriter.writerow(row)
    for row in events:
     if ewriter is None:ewriter=csv.DictWriter(eventfile,fieldnames=list(row));ewriter.writeheader()
     ewriter.writerow(row);eventcount+=1
    if gtfile.tell()+eventfile.tell()>8*2**30:raise RuntimeError('CPU_DERIVED_OUTPUT_8GiB_LIMIT')
    if shutil.disk_usage(R).free<10*2**30:raise RuntimeError('CPU_VALIDATION_DISK_RESERVE')
   gtfile.flush();eventfile.flush()
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_LIMIT')
   (R/'CPU_VALIDATION_PROGRESS.json').write_text(json.dumps({'shards_validated':shard+1,'cases':casecount,'epoch':time.time()}))
 finally:gtfile.close();eventfile.close()
 assert casecount==1820 and gtcount==8164*2 and len(gt_seen)==8164 and predcount=={0:8175,1:8538}
 loss=sum(v=={0:1,1:0} for v in gt_seen.values());gain=sum(v=={0:0,1:1} for v in gt_seen.values());assert loss==278 and gain==169
 if sum(stage_counts[1]['TP_to_FN'].values())!=278 or sum(stage_counts[1]['FN_to_TP'].values())!=169:raise RuntimeError('TARGET_STAGE_COUNT_MISMATCH')
 ready=json.loads((R/'PPU_READY.json').read_text())
 for path,want in ready['protected_files'].items():
  if sha_file(path)!=want:raise RuntimeError('PROTECTED_SOURCE_CHANGED')
 expected={(0,.5):.8665290892796786,(0,.7):.8071592298880969,(1,.5):.8768414234452961,(1,.7):.7854230203953630};metrics={}
 for alpha in [0,1]:
  for t in [.5,.7]:
   ap,_,_=eval_utils.calculate_ap({t:totals[alpha][t]},t)
   if abs(ap-expected[(alpha,t)])>1e-12:raise RuntimeError('HISTORICAL_AP_MISMATCH')
   metrics[f'{alpha}_{t}']={'AP':ap,'TP':sum(totals[alpha][t]['tp']),'FP':sum(totals[alpha][t]['fp'])}
 assert metrics['0_0.7']['TP']==6754 and metrics['0_0.7']['FP']==1421 and metrics['1_0.7']['TP']==6645 and metrics['1_0.7']['FP']==1893
 os.replace(R/'per_GT_full.csv.tmp',R/'per_GT_full.csv');os.replace(R/'strict_anchor_lineage_full.csv.tmp',R/'strict_anchor_lineage_full.csv');signal.alarm(0)
 (R/'CPU_VALIDATION_COMPLETE.json').write_text(json.dumps({'passed':True,'cases':casecount,'GT_conditions':gtcount,'strict_anchor_rows':eventcount,'all_native_NMS_replay_exact':True,'metrics':metrics,'loss':loss,'gain':gain,'GT_stage_distribution':stage_counts,'model_forwards':0,'CUDA_initialized':torch.cuda.is_initialized()},indent=2))
 (R/'RESEARCH_ACCEPTED.json').write_text(json.dumps({'passed':True,'CPU_validation_sha256':sha_file(R/'CPU_VALIDATION_COMPLETE.json'),'PPU_capture_sha256':sha_file(R/'PPU_CAPTURE_COMPLETE.json'),'all_910_and_278_loss_traced':True,'loss_stage_distribution':stage_counts[1]['TP_to_FN']},indent=2))
if __name__=='__main__':
 try:run()
 except BaseException as e:
  signal.alarm(0);(R/'CPU_VALIDATION_FAILURE.json').write_text(json.dumps({'passed':False,'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
