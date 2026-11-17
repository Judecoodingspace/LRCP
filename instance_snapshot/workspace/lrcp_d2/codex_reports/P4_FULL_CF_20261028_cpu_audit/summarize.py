import csv,json,pickle,re,hashlib,collections,resource,os
from pathlib import Path
R=Path(__file__).resolve().parent;F=R.parent/'P4_FULL_PREP_20261028_113714_782940';B=R.parent/'P3B_20261020_103959_980233'
assert json.loads((R/'baseline_validation.json').read_text())['passed'] and json.loads((R/'factorial_metrics.json').read_text())['passed']
rows=list(csv.DictReader((R/'per_GT_mixed.csv').open()));assert len(rows)==8164*2
info=pickle.load(open('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl','rb'))
def scene(sid):
 i=info[sid];m=re.search(r'/scene_(\d+)_(\d+)\.pcd\.bin$',i['lidar_path_1']);assert m and int(m[2])==i['timestamp'];return int(m[1]),int(m[2])
scene_groups={};sample_groups={};cross=collections.Counter()
for x in rows:
 sid=int(x['sample_idx']);g=int(x['geometry']);sc,fr=scene(sid);cross[(g,x['historic_transition'],x['native_stage'],x['transition'])]+=1
 for dic,key in [(scene_groups,(sc,g)),(sample_groups,(sid,g))]:
  y=dic.setdefault(key,collections.Counter());y['GT']+=1;y[x['transition']]+=1;y['net']+=int(x['mixed_TP70'])-int(x['native_TP70'])
for label,dic in [('scene_summary.csv',scene_groups),('sample_summary.csv',sample_groups)]:
 with (R/label).open('x',newline='') as f:
  w=csv.writer(f);w.writerow([label.split('_')[0]+'_id','geometry','GT','TP_to_TP','TP_to_FN','FN_to_TP','FN_to_FN','net_TP70'])
  for (identifier,g),v in sorted(dic.items()):w.writerow([identifier,g,v['GT'],v['TP_to_TP'],v['TP_to_FN'],v['FN_to_TP'],v['FN_to_FN'],v['net']])
with (R/'stage_transition.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['geometry','historical_transition','native_last_strict_stage','mixed_TP70_transition','GT_count'])
 for k,n in sorted(cross.items()):w.writerow([*k,n])
agg={}
for g in (0,1):
 sub=[x for x in rows if int(x['geometry'])==g];trans=collections.Counter(x['transition'] for x in sub)
 filt=lambda hist,stage=None:collections.Counter(x['transition'] for x in sub if x['historic_transition']==hist and (stage is None or x['native_stage']==stage))
 agg[str(g)]={'all':dict(trans),'historical_loss_nms':dict(filt('TP_to_FN','nms_overlap_suppression')),'historical_loss_score':dict(filt('TP_to_FN','score_filter')),'historical_loss_raw':dict(filt('TP_to_FN','no_high_iou_raw_candidate')),'historical_gain':dict(filt('FN_to_TP')),'scenes':len({scene(int(x['sample_idx']))[0] for x in sub})}
assert sum(agg['1']['historical_loss_nms'].values())==205 and sum(agg['1']['historical_loss_score'].values())==64 and sum(agg['1']['historical_loss_raw'].values())==9 and sum(agg['1']['historical_gain'].values())==169
align=list(csv.DictReader((R/'score_geometry_alignment.csv').open()));events=list(csv.DictReader((R/'GT_nms_events.csv').open()))
summary={'passed':True,'GT_rows':len(rows),'groups':agg,'alignment':{},'events':{},'independent_checks':{'rows':True,'historical_counts':True},'CUDA_initialized':False,'model_forwards':0}
for g in (0,1):
 a=[x for x in align if int(x['geometry'])==g]
 summary['alignment'][str(g)]={'candidate_set':'score-admitted, quality-passing candidates with positive IoU to GT; GT can share candidate; top1000 not applied','GT_nonempty':len(a),'native_score_top_is_max_iou':sum(int(x['native_top_is_max_iou']) for x in a),'mixed_score_top_is_max_iou':sum(int(x['mixed_top_is_max_iou']) for x in a)}
 for condition in ('native','mixed'):
  ev=[x for x in events if int(x['geometry'])==g and x['condition']==condition];low=[x for x in ev if float(x['suppressor_iou_GT'])<.7];summary['events'][f'{g}_{condition}']={'strict_candidate_low_iou_suppressor_events':len(low),'distinct_GT':len({(x['sample_idx'],x['gt_idx']) for x in low})}
# Selection is deterministic and prioritizes distinct original scenes and samples; this is a proposal, not causal evidence.
categories={'nms_loss':lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='nms_overlap_suppression','score_loss':lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='score_filter','raw_absent':lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='no_high_iou_raw_candidate','recovered':lambda x:x['historic_transition']=='FN_to_TP','stable_tp':lambda x:x['historic_transition']=='TP_to_TP'}
used_scenes=set();used_samples=set();selected=[]
for name,fn in categories.items():
 pool=[x for x in rows if int(x['geometry'])==1 and fn(x)]
 for _ in range(3):
  choices=[x for x in pool if int(x['sample_idx']) not in used_samples]
  if not choices:break
  x=min(choices,key=lambda x:(scene(int(x['sample_idx']))[0] in used_scenes,hashlib.sha256(f"P5-v1:{name}:{x['sample_idx']}:{x['gt_idx']}".encode()).hexdigest()))
  sid=int(x['sample_idx']);sc,fr=scene(sid);used_samples.add(sid);used_scenes.add(sc);selected.append({'category':name,'sample_idx':sid,'gt_idx':int(x['gt_idx']),'scene_id':sc,'scene_frame':fr,'native_stage':x['native_stage'],'mixed_transition':x['transition']})
with (R/'P5_SAMPLE_PROPOSAL.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=selected[0].keys());w.writeheader();w.writerows(selected)
summary['P5_selection_count']=len(selected);summary['P5_scene_count']=len(used_scenes)
(R/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2))
