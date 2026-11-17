"""Read-only CPU scientific analysis after standalone independent validation.
AP uses native global PR integration over all frozen predictions, never frame AP.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import torch,collections,argparse
from contracts import attribute_gt,CAUSES
from backend import eval_native
from opencood.utils.eval_utils import calculate_ap
LOSS_NAMES=list(CAUSES)+['survives_final_but_unmatched']
COMPONENTS=['TP50','FP50','FN50','TP70','FP70','FN70']+[f'loss{s}_{k}' for s in (50,70) for k in LOSS_NAMES]

def vector(a):
 result={}
 for s in (50,70):
  flags=a[f'gt_tp{s}'];tp=int(flags.sum());result.update({f'TP{s}':tp,f'FP{s}':int(a[f'prediction_fp{s}'].sum()),f'FN{s}':len(flags)-tp})
  stages=attribute_gt(a['IoU_GT_raw'],{k:a['ids_'+k] for k in STAGES},threshold=s/100)
  loss=collections.Counter(row['availability_class'] if row['availability_class']!='high_iou_survives_final' else 'survives_final_but_unmatched' for gi,row in enumerate(stages) if not flags[gi])
  for name in LOSS_NAMES:result[f'loss{s}_{name}']=loss[name]
 return result

def matched(a,gi,s):
 ids=np.flatnonzero(a[f'prediction_gt_idx{s}']==gi)
 if not len(ids):return -1,0.
 p=int(ids[0]);anchor=int(a['ids_final_range'][p]);return anchor,float(a['IoU_GT_raw'][gi,anchor])

def run(root):
 root=Path(root);accept=load(R/'INDEPENDENT_VALIDATION.json');require(accept['passed'] and accept['kind']=='REAL_CPU_SCIENTIFIC_ACCEPTED','INDEPENDENT_SCIENCE_GATE')
 manifest=load(root/'manifest.json');require(manifest['backend']=='real','NO_MOCK_SCIENCE')
 frozen=load(R/'FROZEN_MANIFEST.json');before={r['relative_path']:r['manifest_sha256'] for r in manifest['records']}
 arrays={};meta={};vectors={};stats={c:{s/100:{'tp':[],'fp':[],'score':[],'gt':0} for s in (50,70)} for c in CONDITIONS}
 for rec in manifest['records']:
  p=root/rec['relative_path'];sid=rec['sample_idx'];c=rec['condition'];meta[sid,c]=load(p/'meta.json')
  with np.load(p/'arrays.npz',allow_pickle=False) as z:
   a={k:z[k].copy() for k in z.files if k in ('GT_canonical','final_corners','final_scores','IoU_GT_raw','scores_all','suppressor','first_deletion_code') or k.startswith(('gt_tp','prediction_gt','prediction_fp','ids_'))}
  arrays[sid,c]=a;vectors[sid,c]=vector(a)
  for s in (50,70):
   st,_,_=eval_native(a['final_corners'],a['final_scores'],a['GT_canonical'],s/100);dest=stats[c][s/100]
   for k in ('score','tp','fp'):dest[k].extend(st[k])
   dest['gt']+=st['gt']
 metrics=[]
 for c in CONDITIONS:
  row={'condition':c,'sample_count':len(frozen['samples']),'scene_count':len({x['scene_id'] for x in frozen['samples']})}
  for s in (50,70):
   st=stats[c][s/100];tp=sum(st['tp']);fp=sum(st['fp']);fn=st['gt']-tp
   row.update({f'TP{s}':tp,f'FP{s}':fp,f'FN{s}':fn,f'Precision{s}':tp/(tp+fp) if tp+fp else 0.,f'Recall{s}':tp/st['gt'] if st['gt'] else 0.,f'local_AP{s}':calculate_ap(stats[c],s/100)[0]})
  metrics.append(row)
 csvwrite(R/'FOUR_CONDITION_METRICS.csv',metrics);write(R/'GLOBAL_PR_CURVES.json',{'stats':{c:{str(k):v for k,v in st.items()} for c,st in stats.items()},'scope':'descriptive local pilot AP, native global sorted PR integral'})
 utilities=[];interactions=[];transitions=[];samplemetrics=[]
 for row in frozen['samples']:
  sid=row['sample_idx'];scene=row['scene_id']
  for c in CONDITIONS:samplemetrics.append({'sample_idx':sid,'scene_id':scene,'condition':c,**vectors[sid,c]})
  for name,positive,negative in [('B|C','F','G-B'),('C|B','F','G-C'),('B|A','G-C','E'),('C|A','G-B','E')]:
   utilities.append({'scope':'sample','sample_idx':sid,'scene_id':scene,'utility':name,**{k:vectors[sid,positive][k]-vectors[sid,negative][k] for k in COMPONENTS}})
  interactions.append({'scope':'sample','sample_idx':sid,'scene_id':scene,**{k:vectors[sid,'F'][k]-vectors[sid,'G-C'][k]-vectors[sid,'G-B'][k]+vectors[sid,'E'][k] for k in COMPONENTS}})
  full=arrays[sid,'F']
  for c in ('G-B','G-C'):
   other=arrays[sid,c]
   for s in (50,70):
    fstage=attribute_gt(full['IoU_GT_raw'],{k:full['ids_'+k] for k in STAGES},s/100);ostage=attribute_gt(other['IoU_GT_raw'],{k:other['ids_'+k] for k in STAGES},s/100)
    for gi in range(len(full['GT_canonical'])):
     fmatch,fiou=matched(full,gi,s);omatch,oiou=matched(other,gi,s);sup=int(full['suppressor'][omatch]) if omatch>=0 else -1
     fclass=fstage[gi]['availability_class'];oclass=ostage[gi]['availability_class'];transition=('TP' if fmatch>=0 else 'FN')+'→'+('TP' if omatch>=0 else 'FN')
     diagnosis='no_unique_match_change'
     if transition in ('FN→TP','TP→FN'):
      lose=fclass if transition=='FN→TP' else oclass
      diagnosis={'no_high_iou_raw_candidate':'raw_geometry','score_filter':'score','nms_top1000_truncation':'NMS_top1000','nms_overlap_suppression':'NMS_suppression','size_or_z_filter':'quality','post_nms_range_filter':'range','high_iou_survives_final':'unique_GT_competition'}[lose]
     result={'sample_idx':sid,'scene_id':scene,'scene_frame':row['scene_frame'],'condition':c,'iou_threshold':s/100,'gt_idx':gi,'GT_identity_hash':digest(full['GT_canonical'][gi]),'transition':transition,'diagnostic_losing_condition_stage':diagnosis,'full_matched_anchor':fmatch,'removed_matched_anchor':omatch,'matched_anchor_changed':fmatch>=0 and omatch>=0 and fmatch!=omatch,'full_matched_IoU':fiou,'removed_matched_IoU':oiou,'full_best_final_IoU':fstage[gi]['best_iou']['final_range'],'removed_best_final_IoU':ostage[gi]['best_iou']['final_range'],'full_candidate_availability':fclass,'removed_candidate_availability':oclass,'removed_TP_anchor_full_deletion_code':int(full['first_deletion_code'][omatch]) if omatch>=0 else -1,'removed_TP_anchor_full_score':float(full['scores_all'][omatch]) if omatch>=0 else 0.,'removed_TP_anchor_full_IoU':float(full['IoU_GT_raw'][gi,omatch]) if omatch>=0 else 0.,'removed_TP_anchor_full_suppressor':sup,'full_suppressor_IoU_same_GT':float(full['IoU_GT_raw'][gi,sup]) if sup>=0 else 0.,'same_GT_suppressor_is_full_TP':bool(sup==fmatch and sup>=0)}
     for st in STAGES:
      result['full_best_IoU_'+st]=fstage[gi]['best_iou'][st];result['removed_best_IoU_'+st]=ostage[gi]['best_iou'][st]
      result['full_candidates_'+st]=fstage[gi]['high_iou_count'][st];result['removed_candidates_'+st]=ostage[gi]['high_iou_count'][st]
     transitions.append(result)
 for scene in sorted({x['scene_id'] for x in frozen['samples']}):
  members=[x['sample_idx'] for x in frozen['samples'] if x['scene_id']==scene]
  for name in ('B|C','C|B','B|A','C|A'):
   src=[x for x in utilities if x['scope']=='sample' and x['scene_id']==scene and x['utility']==name]
   utilities.append({'scope':'scene','sample_idx':'','scene_id':scene,'utility':name,**{k:sum(x[k] for x in src) for k in COMPONENTS}})
  src=[x for x in interactions if x['scope']=='sample' and x['scene_id']==scene];interactions.append({'scope':'scene','sample_idx':'','scene_id':scene,**{k:sum(x[k] for x in src) for k in COMPONENTS}})
 csvwrite(R/'AGENT_UTILITY.csv',utilities);csvwrite(R/'AGENT_INTERACTION.csv',interactions);csvwrite(R/'PER_GT_TRANSITIONS.csv',transitions);csvwrite(R/'PER_SAMPLE_METRICS.csv',samplemetrics)
 for rec in manifest['records']:require(sha_file(root/rec['relative_path']/'manifest.json')==before[rec['relative_path']],'ANALYSIS_MUTATED_SHARD')
 require(not torch.cuda.is_initialized(),'CPU_DEVICE')
 summary={'passed':True,'metrics':metrics,'gt_transitions':dict(collections.Counter(x['transition'] for x in transitions)),'FN_to_TP_losing_stage':dict(collections.Counter(x['diagnostic_losing_condition_stage'] for x in transitions if x['transition']=='FN→TP')),'TP_to_FN_losing_stage':dict(collections.Counter(x['diagnostic_losing_condition_stage'] for x in transitions if x['transition']=='TP→FN')),'forward_count':manifest['real_forwards'],'reserved_attempts':len(load(R/'RUN_BUDGET.json')['attempts']),'sample_count':len(frozen['samples']),'scene_count':len({x['scene_id'] for x in frozen['samples']}),'scope':'8-sample descriptive local pilot, not 910 validation AP; scenes are clustered and no population claim','Q_components':COMPONENTS,'utility_sign':'TP positive better; FP/FN/loss positive worse; no scalar utility weights','interaction_interpretation':'componentwise additive count interaction; TP sign denotes super/subadditivity; FP/FN signs reverse desirability','original_shards_unchanged':True,'CUDA_initialized':False}
 write(R/'ANALYSIS_SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();run(args.root)
