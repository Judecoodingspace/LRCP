"""Independent descriptive science, only after complete real 8x4 acceptance.
Six unordered pairs, unique GT matching, native all-sample AP. No scalar weights.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,argparse,itertools,collections
from contracts import attribute_gt,CAUSES
from INDEPENDENT_VALIDATOR import same
_import_path=sys.path.copy();sys.path.insert(1,str(REPORTS/'P4_C2_20261027_233036_301795'))
from c1_reused import eval_native
sys.path[:]=_import_path
from opencood.utils.eval_utils import calculate_ap
LOSS_NAMES=list(CAUSES)+['survives_final_but_unmatched']
Q_FIELDS=['TP50','FP50','FN50','TP70','FP70','FN70','best_raw_IoU_sum','best_final_IoU_sum','matched_IoU50_sum','matched_IoU70_sum','matched_score50_sum','matched_score70_sum']+[f'loss{s}_{k}' for s in (50,70) for k in LOSS_NAMES]

def gt_match(a,gi,s):
 ids=np.flatnonzero(a[f'prediction_gt_idx{s}']==gi)
 if not len(ids):return (-1,0.,0.)
 pi=int(ids[0]);anchor=int(a['ids_final_range'][pi]);return anchor,float(a['IoU_GT_raw'][gi,anchor]),float(a['final_scores'][pi])

def derived(a):
 v={};stages={};statistics={}
 for s in (50,70):
  st,matched,flags=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],s/100);statistics[s/100]=st
  same(a[f'prediction_gt_idx{s}'],matched,'ANALYSIS_MATCH_RECOMPUTE');same(a[f'gt_tp{s}'],flags,'ANALYSIS_TP_RECOMPUTE')
  tp=sum(st['tp']);fp=sum(st['fp']);v.update({f'TP{s}':tp,f'FP{s}':fp,f'FN{s}':len(flags)-tp})
  stages[s]=attribute_gt(a['IoU_GT_raw'],{k:a['ids_'+k] for k in STAGES},s/100)
  losses=collections.Counter(row['availability_class'] if row['availability_class']!='high_iou_survives_final' else 'survives_final_but_unmatched' for gi,row in enumerate(stages[s]) if not flags[gi])
  for name in LOSS_NAMES:v[f'loss{s}_{name}']=losses[name]
  matches=[gt_match(a,gi,s) for gi in range(len(flags))];v[f'matched_IoU{s}_sum']=sum(x[1] for x in matches);v[f'matched_score{s}_sum']=sum(x[2] for x in matches)
 v['best_raw_IoU_sum']=sum(float(row.max(initial=0.)) for row in a['IoU_GT_raw'])
 v['best_final_IoU_sum']=sum(float(row[a['ids_final_range']].max(initial=0.)) for row in a['IoU_GT_raw'])
 return v,stages,statistics

def write_table(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r));csvwrite(path,rows,fields)

def run(root):
 root=Path(root);accept=load(R/'INDEPENDENT_REAL_VALIDATION.json');require(accept['passed'] and accept['state']=='RD3_R2_REAL_FOUR_GROUP_ACCEPTED','COMPLETE_REAL_SCIENCE_GATE')
 run=load(root/'manifest.json');require(run['backend']=='REAL' and len(run['records'])==32,'REAL_32_ONLY');frozen=load(FROZEN)
 snapshots={};data={};stageinfo={};vectors={};stats={c:{s/100:{'tp':[],'fp':[],'score':[],'gt':0} for s in (50,70)} for c in CONDITIONS}
 nmsrows=[];samplemetrics=[]
 for rec in run['records']:
  p=root/rec['relative_path'];m=load(p/'manifest.json');require(m['state']=='ACCEPTED' and m['backend']=='REAL','ACCEPTED_REAL_ONLY')
  for name,h in m['files'].items():require(sha_file(p/name)==h,'READ_ONLY_SCIENCE_HASH');snapshots[str(p/name)]=h
  with np.load(p/'arrays.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files if k in ('GT_canonical','final_corners','final_scores','IoU_GT_raw','scores_all','suppressor','suppressor_overlap','first_deletion_code') or k.startswith(('ids_','gt_tp','prediction_gt','prediction_fp'))}
  sid=rec['sample_idx'];c=rec['condition'];row=next(x for x in frozen['samples'] if x['sample_idx']==sid);v,st,stat=derived(a);data[sid,c]=a;stageinfo[sid,c]=st;vectors[sid,c]=v
  samplemetrics.append({'sample_idx':sid,'scene_id':row['scene_id'],'scene_frame':row['scene_frame'],'condition':c,'final_predictions':len(a['final_scores']),'nms_overlap_removed':int((a['suppressor']>=0).sum()),**v})
  for threshold,source in stat.items():
   dest=stats[c][threshold]
   for k in ('tp','fp','score'):dest[k].extend(source[k])
   dest['gt']+=source['gt']
  nmsrows.append({'record_type':'condition_summary','sample_idx':sid,'scene_id':row['scene_id'],'condition':c,'raw_decoded_count':len(a['ids_raw_decoded']),'score_pass_count':len(a['ids_score_pass']),'quality_pass_count':len(a['ids_quality_pass']),'nms_top1000_count':len(a['ids_nms_top1000']),'native_nms_keep_count':len(a['ids_nms_keep']),'NMS_overlap_suppressed_count':int((a['suppressor']>=0).sum()),'top1000_truncated_count':len(a['ids_quality_pass'])-len(a['ids_nms_top1000']),'final_prediction_count':len(a['final_scores']),'nms_keep_anchor_ids':json.dumps(a['ids_nms_keep'].tolist()),'final_anchor_ids':json.dumps(a['ids_final_range'].tolist()),'suppressor_links':json.dumps({str(i):int(a['suppressor'][i]) for i in np.flatnonzero(a['suppressor']>=0)})})
  for s in (50,70):
   for gi,info in enumerate(st[s]):
    anchor,iou,score=gt_match(a,gi,s)
    nmsrows.append({'record_type':'GT_stage','sample_idx':sid,'scene_id':row['scene_id'],'condition':c,'gt_idx':gi,'iou_threshold':s/100,'native_unique_TP':int(anchor>=0),'native_match_anchor':anchor,'native_match_IoU':iou,'native_match_score':score,'candidate_availability':info['availability_class'],**{'best_IoU_'+k:info['best_iou'][k] for k in STAGES},**{'high_IoU_count_'+k:info['high_iou_count'][k] for k in STAGES}})
 metrics=[]
 for c in CONDITIONS:
  row={'condition':c,'sample_count':8,'scene_count':3,'GT_count':56,'final_predictions':sum(len(data[r['sample_idx'],c]['final_scores']) for r in frozen['samples']),'NMS_overlap_removed':sum(int((data[r['sample_idx'],c]['suppressor']>=0).sum()) for r in frozen['samples'])}
  for s in (50,70):
   st=stats[c][s/100];require(st['gt']==56,'FROZEN_GT_DENOMINATOR');tp=sum(st['tp']);fp=sum(st['fp']);row.update({f'TP{s}':tp,f'FP{s}':fp,f'FN{s}':56-tp,f'Precision{s}':tp/(tp+fp) if tp+fp else 0.,f'Recall{s}':tp/56,f'local_AP{s}':calculate_ap(stats[c],s/100)[0]})
  metrics.append(row)
 write_table(R/'FOUR_GROUP_REAL_METRICS.csv',metrics);write_table(R/'PER_SAMPLE_METRICS.csv',samplemetrics)
 utilities=[];interactions=[];transitions=[]
 for row in frozen['samples']:
  sid=row['sample_idx'];scene=row['scene_id']
  for name,pos,neg in [('B|C','F','G-B'),('C|B','F','G-C'),('B|A','G-C','E'),('C|A','G-B','E')]:utilities.append({'scope':'sample','sample_idx':sid,'scene_id':scene,'utility':name,**{k:vectors[sid,pos][k]-vectors[sid,neg][k] for k in Q_FIELDS}})
  interactions.append({'scope':'sample','sample_idx':sid,'scene_id':scene,**{k:vectors[sid,'F'][k]-vectors[sid,'G-C'][k]-vectors[sid,'G-B'][k]+vectors[sid,'E'][k] for k in Q_FIELDS}})
  for source,target in itertools.combinations(CONDITIONS,2):
   x=data[sid,source];y=data[sid,target]
   for s in (50,70):
    for gi in range(len(x['GT_canonical'])):
     xa,xi,xs=gt_match(x,gi,s);ya,yi,ys=gt_match(y,gi,s);sup=int(x['suppressor'][ya]) if ya>=0 else -1;rawx=int(np.argmax(x['IoU_GT_raw'][gi]));rawy=int(np.argmax(y['IoU_GT_raw'][gi]));sx=stageinfo[sid,source][s][gi];sy=stageinfo[sid,target][s][gi]
     transitions.append({'sample_idx':sid,'scene_id':scene,'scene_frame':row['scene_frame'],'condition_from':source,'condition_to':target,'gt_idx':gi,'GT_hash':digest(x['GT_canonical'][gi]),'iou_threshold':s/100,'transition':('TP' if xa>=0 else 'FN')+'→'+('TP' if ya>=0 else 'FN'),'from_matched_anchor':xa,'to_matched_anchor':ya,'matched_anchor_changed':xa>=0 and ya>=0 and xa!=ya,'from_matched_IoU':xi,'to_matched_IoU':yi,'from_matched_score':xs,'to_matched_score':ys,'from_best_raw_anchor':rawx,'to_best_raw_anchor':rawy,'from_best_raw_IoU':float(x['IoU_GT_raw'][gi,rawx]),'to_best_raw_IoU':float(y['IoU_GT_raw'][gi,rawy]),'from_raw_best_score':float(x['scores_all'][rawx]),'to_raw_best_score':float(y['scores_all'][rawy]),'from_best_final_IoU':sx['best_iou']['final_range'],'to_best_final_IoU':sy['best_iou']['final_range'],'from_candidate_availability':sx['availability_class'],'to_candidate_availability':sy['availability_class'],'to_TP_anchor_from_deletion_code':int(x['first_deletion_code'][ya]) if ya>=0 else -1,'to_TP_anchor_from_score':float(x['scores_all'][ya]) if ya>=0 else 0.,'to_TP_anchor_from_IoU_same_GT':float(x['IoU_GT_raw'][gi,ya]) if ya>=0 else 0.,'to_TP_anchor_from_NMS_suppressor':sup,'from_suppressor_IoU_same_GT':float(x['IoU_GT_raw'][gi,sup]) if sup>=0 else 0.,'suppressor_is_from_unique_match':sup>=0 and sup==xa,**{'from_best_IoU_'+k:sx['best_iou'][k] for k in STAGES},**{'to_best_IoU_'+k:sy['best_iou'][k] for k in STAGES}})
 for scene in sorted({r['scene_id'] for r in frozen['samples']}):
  for name in ('B|C','C|B','B|A','C|A'):
   src=[r for r in utilities if r['scope']=='sample' and r['scene_id']==scene and r['utility']==name];utilities.append({'scope':'scene','sample_idx':'','scene_id':scene,'utility':name,**{k:sum(r[k] for r in src) for k in Q_FIELDS}})
  src=[r for r in interactions if r['scope']=='sample' and r['scene_id']==scene];interactions.append({'scope':'scene','sample_idx':'','scene_id':scene,**{k:sum(r[k] for r in src) for k in Q_FIELDS}})
 write_table(R/'AGENT_MARGINAL_UTILITY.csv',utilities);write_table(R/'AGENT_INTERACTION.csv',interactions);write_table(R/'PER_GT_TRANSITIONS.csv',transitions);write_table(R/'NMS_STAGE_ANALYSIS.csv',nmsrows)
 for path,h in snapshots.items():require(sha_file(path)==h,'ANALYSIS_MUTATED_SCIENCE')
 write(R/'GLOBAL_PR_DATA.json',{c:{str(t):st for t,st in v.items()} for c,v in stats.items()})
 cpu_guard();summary={'passed':True,'state':'RD3_R2_REAL_FOUR_GROUP_ACCEPTED','metrics':metrics,'Q_fields':Q_FIELDS,'utility_signs':'TP and IoU/score positive denote larger components; FP/FN/loss positive denote larger errors. No combined weighted utility.','pairs':list(itertools.combinations(CONDITIONS,2)),'GT_transition_rows':len(transitions),'transitions_all_6_pairs_50_and_70':dict(collections.Counter(r['transition'] for r in transitions)),'read_only_science':True,'local_AP_protocol':'native global PR integral across all eight frames, never mean per-frame AP','inferential_units':'8 frozen frames clustered in 3 scenes; 56 GT are not 56 independent experiments','CUDA_initialized':False,'model_forwards':0}
 write(R/'ANALYSIS_SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
