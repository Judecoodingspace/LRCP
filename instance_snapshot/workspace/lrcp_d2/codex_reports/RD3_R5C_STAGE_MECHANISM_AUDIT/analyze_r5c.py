"""R5C producer: deterministic raw-candidate lineage and stage observations.
No model forward or module-level causal intervention. All historical files read-only.
"""
from stage_common import *
TRACES=[];TRACE_INDEX={};LINEAGE=[];MECHANISMS=[];FP_ROWS=[];FP_SUMMARY=[];INTERACTION_ROWS=[];ANCHOR_INTER=[];NMS_EDGES=[];MATCH_ORDER=[];GEOM_ASSOC=[];CANDIDATE_IDS={}


def gi_for(sid,oid):
 hits=[gi for (s,gi),g in GT_MAP.items() if s==sid and g['physical_object_id']==str(oid)];require(len(hits)==1,'UNIQUE_PHYSICAL_GT');return hits[0]
def ids_at(a,stage):return np.asarray(a['ids_'+stage],np.int32)
def match_info(sid,c,gi,t):
 a=DATA[sid,c];m=MATCH[sid,c,t];pi=np.flatnonzero(m['matched']==gi)
 if not len(pi):return {'TP':False,'prediction_index':None,'candidate_id':None,'IoU':None,'score':None}
 p=int(pi[0]);cid=int(a['ids_final_range'][p]);return {'TP':True,'prediction_index':p,'candidate_id':cid,'IoU':float(GEOMETRY_IOU[sid,c][gi,cid]),'score':float(a['final_scores'][p])}

def full_matching_trace():
 for (sid,c),a in DATA.items():
  predpolys=polys(a['final_corners']);gtpolys=polys(a['GT_canonical'])
  for t in (50,70):
   remaining=list(range(len(gtpolys)));ordered=np.argsort(-a['final_scores']);claimed={}
   for rank,pi0 in enumerate(ordered):
    pi=int(pi0);cid=int(a['ids_final_range'][pi]);all_iou=overlap(predpolys[pi],gtpolys);available=overlap(predpolys[pi],[gtpolys[j] for j in remaining]);selected=None
    before=remaining.copy()
    if len(available) and float(available.max())>=t/100:selected=remaining.pop(int(np.argmax(available)));claimed[selected]=cid
    require((selected if selected is not None else -1)==int(a[f'prediction_gt_idx{t}'][pi]),'MATCH_ORDER_REPLAY')
    eligible=np.flatnonzero(all_iou>=t/100).astype(int).tolist();eligible_remaining=[gi for gi in before if float(all_iou[gi])>=t/100]
    reason='MATCHED_UNIQUE_GT' if selected is not None else 'GT_ALREADY_CLAIMED' if eligible else 'BELOW_ALL_GT_IOU_THRESHOLD'
    row={'sample_idx':sid,'condition':c,'iou_threshold':t/100,'score_order_rank_zero_based':rank,'final_prediction_index':pi,'candidate_id':cid,'score':float(a['final_scores'][pi]),'max_all_GT_IoU':float(all_iou.max(initial=0.)),'eligible_all_GT_indices':json.dumps(eligible),'eligible_remaining_GT_indices':json.dumps(eligible_remaining),'remaining_GT_before':json.dumps(before),'matched_gt_idx':selected,'matched_physical_GT_id':GT_MAP[sid,selected]['physical_object_id'] if selected is not None else None,'matched_IoU':float(all_iou[selected]) if selected is not None else None,'result':'TP' if selected is not None else 'FP','reason':reason,'already_claimed_eligible_GT':json.dumps({str(gi):claimed[gi] for gi in eligible if gi not in before}),'source_path':LOCS[sid,c]['absolute_path']}
    MATCH_ORDER.append(row)
 table('R5C_GT_MATCH_ORDER.csv',MATCH_ORDER)


def gt_traces():
 for (sid,c),a in DATA.items():
  ious=GEOMETRY_IOU[sid,c];global_score_id=int(np.argmax(a['scores_all']))
  for gi in range(len(a['GT_canonical'])):
   g=GT_MAP[sid,gi];oid=g['physical_object_id'];rawbest=int(np.argmax(ious[gi]));finalids=a['ids_final_range'];finalbest=int(finalids[np.argmax(ious[gi,finalids])]) if len(finalids) else None
   for t in (50,70):
    high=ious[gi]>=t/100;counts={s:int(np.count_nonzero(high[ids_at(a,s)])) for s in STAGES};highsets={s:ids_at(a,s)[high[ids_at(a,s)]].astype(int).tolist() for s in STAGES};mi=match_info(sid,c,gi,t);first=next((s for s in STAGES if counts[s]==0),None)
    dropped={s:sorted(set(highsets[prev])-set(highsets[s])) for prev,s in zip(STAGES,STAGES[1:])};suppressed=[i for i in highsets['nms_top1000'] if int(a['suppressor'][i])>=0];competition=[]
    if not mi['TP'] and counts['final_range']>0:
     for cid in highsets['final_range']:
      pi=int(np.flatnonzero(finalids==cid)[0]);other=int(a[f'prediction_gt_idx{t}'][pi]);competition.append({'candidate_id':cid,'matched_other_gt_idx':other if other>=0 else None,'matched_other_physical_gt_id':GT_MAP[sid,other]['physical_object_id'] if other>=0 else None,'target_GT_IoU':float(ious[gi,cid]),'score':float(a['scores_all'][cid])})
    rawhigh=np.flatnonzero(high);raw_highscore=int(rawhigh[np.argmax(a['scores_all'][rawhigh])]) if len(rawhigh) else None
    stage='DETECTED' if mi['TP'] else 'GT_MATCH_COMPETITION' if counts['final_range']>0 else {'raw_decoded':'RAW_GEOMETRY','score_pass':'SCORE_FILTER','quality_pass':'LARGE_OR_Z_FILTER','nms_top1000':'TOP1000','nms_keep':'NMS','final_range':'FINAL_RANGE'}[first]
    row={'sample_idx':sid,'scene_id':int(g['scene_id']),'scene_frame':int(g['scene_frame']),'condition':c,'gt_idx':gi,'physical_GT_id':oid,'GT_hash':g['GT_hash'],'canonical_GT_corners':json.dumps(a['GT_canonical'][gi].tolist()),'iou_threshold':t/100,'is_core_GT':(sid,oid) in CORE,'interaction_focus_sample':sid in INTERACTION_SAMPLES,'raw_max_IoU_candidate_id':rawbest,'raw_max_IoU':float(ious[gi,rawbest]),'raw_max_IoU_candidate_score':float(a['scores_all'][rawbest]),'raw_highest_score_candidate_id':global_score_id,'raw_highest_score':float(a['scores_all'][global_score_id]),'raw_highest_score_candidate_IoU':float(ious[gi,global_score_id]),'raw_high_IoU_highest_score_candidate_id':raw_highscore,'raw_high_IoU_highest_score':float(a['scores_all'][raw_highscore]) if raw_highscore is not None else None,**{'high_IoU_count_'+s:n for s,n in counts.items()},**{'high_IoU_ids_'+s:json.dumps(ids) for s,ids in highsets.items()},'high_IoU_dropped_by_stage':json.dumps(dropped),'high_IoU_suppressed_ids':json.dumps(suppressed),'some_high_IoU_candidate_NMS_suppressed':bool(suppressed),'all_preNMS_high_IoU_candidates_eliminated_by_NMS':counts['nms_top1000']>0 and counts['nms_keep']==0,'first_stage_all_high_IoU_disappear':first,'observed_final_bottleneck':stage,'final_max_IoU_candidate_id':finalbest,'final_max_IoU':float(ious[gi,finalbest]) if finalbest is not None else None,'final_max_IoU_candidate_score':float(a['scores_all'][finalbest]) if finalbest is not None else None,'matched_prediction_index':mi['prediction_index'],'matched_candidate_id':mi['candidate_id'],'matched_IoU':mi['IoU'],'matched_score':mi['score'],'TP_FN':'TP' if mi['TP'] else 'FN','GT_match_competition_evidence':json.dumps(competition),'direct_evidence_level':'DIRECT_SAVED_CANDIDATES_AND_EXACT_NATIVE_EQUIVALENT_CPU_REPLAY','source_path':LOCS[sid,c]['absolute_path'],'source_sha256':LOCS[sid,c]['sha256'],'source_record_key':json.dumps({'sample_idx':sid,'condition':c,'gt_idx':gi,'physical_GT_id':oid,'iou_threshold':t/100})}
    TRACES.append(row);TRACE_INDEX[sid,c,gi,t]=row
 require(len(TRACES)==448,'GT_TRACE56x4x2');table('R5C_GT_STAGE_TRACE.csv',TRACES)


def transition_mechanisms():
 transitions=csvrows(B/'R5B_GT_TRANSITION_AUDIT.csv')
 for old in transitions:
  sid=int(old['sample_idx']);oid=old['physical_object_id'];gi=int(old['gt_idx']);t=int(round(float(old['iou_threshold'])*100));before=old['condition_before'];after=old['condition_after'];key=sid,oid
  if old['transition_state'] not in ('TP_TO_FN','FN_TO_TP') and key not in CORE:continue
  b=TRACE_INDEX[sid,before,gi,t];a=TRACE_INDEX[sid,after,gi,t];bb=DATA[sid,before];aa=DATA[sid,after];bi=GEOMETRY_IOU[sid,before][gi];ai=GEOMETRY_IOU[sid,after][gi];highunion=np.flatnonzero((bi>=t/100)|(ai>=t/100));labels=[];proof={}
  if bool(b['high_IoU_count_raw_decoded'])!=bool(a['high_IoU_count_raw_decoded']):labels.append('RAW_GEOMETRY_DIFFERENCE');proof['RAW_GEOMETRY_DIFFERENCE']={'before_count':b['high_IoU_count_raw_decoded'],'after_count':a['high_IoU_count_raw_decoded']}
  fntraces=[('before',b,bb,bi),('after',a,aa,ai)];bottleneck_evidence=[]
  for side,tr,data,iv in fntraces:
   if tr['TP_FN']=='TP':continue
   bottleneck=tr['observed_final_bottleneck'];mapping={'SCORE_FILTER':'SCORE_OR_QUALITY_FILTERING','LARGE_OR_Z_FILTER':'SCORE_OR_QUALITY_FILTERING','TOP1000':'TOPK_SELECTION','NMS':'NMS_SUPPRESSION','FINAL_RANGE':'FINAL_RANGE_OR_SIZE_FILTER','GT_MATCH_COMPETITION':'GT_MATCH_COMPETITION','RAW_GEOMETRY':'RAW_GEOMETRY_ABSENCE'}
   if bottleneck=='RAW_GEOMETRY':
    bottleneck_evidence.append({'side':side,'observation':'NO_RAW_HIGH_IOU_CANDIDATE','count':0});continue
   label=mapping.get(bottleneck,'UNRESOLVED')
   if label not in labels:labels.append(label)
   if label=='NMS_SUPPRESSION':
    ids=json.loads(tr['high_IoU_ids_nms_top1000']);edges=[]
    for cid in ids:
     sup=int(data['suppressor'][cid]);require(sup>=0,'NMS_ALL_HIGH_CANDIDATES_MUST_HAVE_SUPPRESSOR');edges.append({'suppressed_id':cid,'suppressor_id':sup,'suppressed_score':float(data['scores_all'][cid]),'suppressor_score':float(data['scores_all'][sup]),'BEV_IoU':float(data['suppressor_overlap'][cid]),'suppressed_GT_IoU':float(iv[cid]),'suppressor_GT_IoU':float(iv[sup]),'all_high_IoU_lost_at_NMS':True})
    proof.setdefault(label,[]).append({'side':side,'edges':edges})
   elif label=='SCORE_OR_QUALITY_FILTERING':
    proof.setdefault(label,[]).append({'side':side,'first_empty_stage':tr['first_stage_all_high_IoU_disappear'],'raw_high_ids':json.loads(tr['high_IoU_ids_raw_decoded']),'score_high_ids':json.loads(tr['high_IoU_ids_score_pass']),'quality_high_ids':json.loads(tr['high_IoU_ids_quality_pass']),'score_threshold':.2})
   elif label=='GT_MATCH_COMPETITION':proof.setdefault(label,[]).append({'side':side,'records':json.loads(tr['GT_match_competition_evidence'])})
   else:proof.setdefault(label,[]).append({'side':side,'first_empty_stage':tr['first_stage_all_high_IoU_disappear'],'dropped':json.loads(tr['high_IoU_dropped_by_stage'])})
  score_change=[int(i) for i in highunion if bool(bb['score_mask'][i])!=bool(aa['score_mask'][i])]
  nms_change=[int(i) for i in highunion if (i in set(bb['ids_nms_top1000'])) and (i in set(aa['ids_nms_top1000'])) and ((i in set(bb['ids_nms_keep']))!=(i in set(aa['ids_nms_keep'])))]
  matched_cross=[]
  for tr in (b,a):
   cid=tr['matched_candidate_id']
   if cid is not None and ((float(bi[cid])>=t/100)!=(float(ai[cid])>=t/100)):matched_cross.append({'candidate_id':cid,'before_GT_IoU':float(bi[cid]),'after_GT_IoU':float(ai[cid]),'threshold':t/100})
  material={'matched_anchor_geometry_threshold_crossing':matched_cross,'score_pass_changed_on_high_IoU_union':score_change,'native_NMS_keep_changed_on_high_IoU_union':nms_change,'raw_high_IoU_existence_changed':bool(b['high_IoU_count_raw_decoded'])!=bool(a['high_IoU_count_raw_decoded'])}
  material_count=sum(bool(x) for x in material.values())
  if material_count>=2:labels.append('MULTIPLE_STAGE_CHANGES');proof['MULTIPLE_STAGE_CHANGES']=material
  if old['transition_state'] in ('TP_TO_FN','FN_TO_TP') and not labels:labels.append('UNRESOLVED');proof['UNRESOLVED']='No supported stage predicate explains the status change.'
  if not labels:labels=['NO_GT_DETECTION_STATE_CHANGE'];proof['NO_GT_DETECTION_STATE_CHANGE']='Core case retained to show stable status at the other threshold.'
  MECHANISMS.append({'sample_idx':sid,'scene_id':old['scene_id'],'scene_frame':old['scene_frame'],'physical_GT_id':oid,'GT_hash':old['GT_hash'],'gt_idx':gi,'contrast':old['contrast'],'condition_before':before,'condition_after':after,'iou_threshold':t/100,'transition_state':old['transition_state'],'before_bottleneck':b['observed_final_bottleneck'],'after_bottleneck':a['observed_final_bottleneck'],'stage_labels':json.dumps(list(dict.fromkeys(labels))),'is_core_GT':key in CORE,'raw_candidate_geometry_changed':not np.array_equal(bb['raw_corners'],aa['raw_corners']),'raw_max_IoU_before':b['raw_max_IoU'],'raw_max_IoU_after':a['raw_max_IoU'],'raw_max_abs_corner_change':float(np.max(np.abs(bb['raw_corners']-aa['raw_corners']))),'matched_anchor_threshold_crossing':json.dumps(matched_cross),'supported_stage_predicates':json.dumps(proof),'raw_geometry_and_downstream_changes_not_isolated_interventions':True,'evidence_level':'DIRECT_STAGE_OBSERVATION; not independent-module causality','uncertainty':'Upstream feature/MetaFlow/attention contribution is not identifiable from these saved outputs.','UNRESOLVED':any(x=='UNRESOLVED' for x in labels),'source_before':LOCS[sid,before]['absolute_path'],'source_after':LOCS[sid,after]['absolute_path'],'source_sha_before':LOCS[sid,before]['sha256'],'source_sha_after':LOCS[sid,after]['sha256']})
 table('R5C_GT_TRANSITION_MECHANISMS.csv',MECHANISMS)
 write('R5C_LABEL_RULES.json',{'RAW_GEOMETRY_DIFFERENCE':'raw high-IoU existence changes, not just any coordinate change','SCORE_OR_QUALITY_FILTERING':'FN side first high-IoU disappearance at score_pass or quality_pass, exact raw IDs retained','TOPK_SELECTION':'FN side first disappearance at nms_top1000','NMS_SUPPRESSION':'FN side has preNMS high-IoU IDs but zero NMS survivors; every such ID has a native suppressor edge','FINAL_RANGE_OR_SIZE_FILTER':'FN side first complete disappearance at final_range; large/z filtering occurs earlier and is reported separately','GT_MATCH_COMPETITION':'FN side still has final high-IoU IDs, with actual greedy-order matching targets saved','MULTIPLE_STAGE_CHANGES':'at least two material observed changes among same successful-anchor threshold crossing, score-pass membership on high-IoU union, NMS keep on high-IoU union, raw existence','raw_geometry_changed_recorded_separately':True,'stage_labels_are_not_upstream_module_causal_estimates':True,'missing_matched_fields_on_FN':'blank CSV / null JSON, not zero or maximum candidate IoU'})

def geometric_proxy(sid,before,after):
 b=DATA[sid,before];a=DATA[sid,after];bp=polys(b['final_corners']);ap=polys(a['final_corners']);m=np.zeros((len(bp),len(ap)),np.float32)
 for i,p in enumerate(bp):m[i]=overlap(p,ap)
 hits=m>=.5;result={}
 for side,src,dst,adj in [('before',b,a,hits),('after',a,b,hits.T)]:
  for pi,cid0 in enumerate(src['ids_final_range']):
   cid=int(cid0);partners=np.flatnonzero(adj[pi]);unique=len(partners)==1 and int(adj[:,int(partners[0])].sum())==1;partner=int(dst['ids_final_range'][int(partners[0])]) if unique else None
   status='UNIQUE_GEOMETRIC_PROXY' if unique else 'AMBIGUOUS_GEOMETRIC_PROXY' if len(partners) else 'NO_GEOMETRIC_PROXY_ABOVE_THRESHOLD'
   item={'status':status,'partner_candidate_id':partner,'eligible_partner_ids':[int(dst['ids_final_range'][j]) for j in partners],'threshold':.5,'physical_object_identity_verified':False};result[side,cid]=item
   GEOM_ASSOC.append({'sample_idx':sid,'condition_before':before,'condition_after':after,'side':side,'final_candidate_id':cid,'status':status,'threshold_BEV_IoU':.5,'eligible_partner_ids':json.dumps(item['eligible_partner_ids']),'unique_partner_candidate_id':partner,'max_cross_condition_BEV_IoU':float(adj[pi].max(initial=0.)) if False else float((m[pi] if side=='before' else m[:,pi]).max(initial=0.)),'physical_object_identity_verified':False})
 return result


def fp_path(a,cid,before_status,after_status,other):
 code=int(a['first_deletion_code'][cid]);reason={1:'SCORE_FILTER',2:'LARGE_OR_Z_FILTER',3:'TOP1000',4:'NMS_SUPPRESSION',5:'FINAL_RANGE',6:'FINAL_SURVIVOR'}[code]
 return reason


def fp_provenance():
 harmful=[x for x in csvrows(B/'R5B_SAMPLE_AGENT_UTILITY.csv') if x['class70']=='CLEAR_HARMFUL'];expected={(324,'U_B_given_A'),(324,'U_C_given_A'),(354,'U_C_given_B'),(354,'U_C_given_A'),(384,'U_B_given_C'),(384,'U_B_given_A'),(725,'U_B_given_C'),(725,'U_B_given_A'),(725,'U_C_given_A')};require({(int(x['sample_idx']),x['contrast']) for x in harmful}==expected,'EXACT_NINE_HARMFUL_PREDECLARED')
 comparisons=[(int(x['sample_idx']),x['contrast'],'MANDATED_CLEAR_HARMFUL') for x in harmful]+[(424,name,'POSITIVE_FP_INTERACTION_COMPARISON') for name in CONTRASTS]
 for sid,contrast,role in comparisons:
  before,after=CONTRASTS[contrast];b=DATA[sid,before];a=DATA[sid,after];proxy=geometric_proxy(sid,before,after);fb={int(cid) for cid,gi in zip(b['ids_final_range'],b['prediction_gt_idx70']) if gi<0};fa={int(cid) for cid,gi in zip(a['ids_final_range'],a['prediction_gt_idx70']) if gi<0};counter=collections.Counter();eventrows=[]
  for cid in sorted(fb|fa):
   bs,bpi,bgi=candidate_state(b,cid,70);aa,api,agi=candidate_state(a,cid,70);types={('ABSENT_FINAL','FP'):'ADDED_FINAL_FP_AT_ANCHOR',('TP','FP'):'TP_TO_FP_AT_ANCHOR',('FP','FP'):'PERSISTENT_FP_AT_ANCHOR',('FP','TP'):'FP_TO_TP_AT_ANCHOR',('FP','ABSENT_FINAL'):'REMOVED_FINAL_FP_AT_ANCHOR'};event=types[bs,aa];counter[event]+=1
   bv=GEOMETRY_IOU[sid,before][:,cid];av=GEOMETRY_IOU[sid,after][:,cid];pairiou=float(overlap(polys(b['raw_corners'][cid:cid+1])[0],polys(a['raw_corners'][cid:cid+1]))[0]);bcode=int(b['first_deletion_code'][cid]);acode=int(a['first_deletion_code'][cid]);tags=[];facts={}
   if event=='ADDED_FINAL_FP_AT_ANCHOR':
    cause={1:'ENTERS_FINAL_FROM_SCORE_FILTER',2:'ENTERS_FINAL_FROM_LARGE_Z_FILTER',3:'ENTERS_FINAL_FROM_TOP1000',4:'ENTERS_FINAL_FROM_NMS_SUPPRESSION',5:'ENTERS_FINAL_FROM_RANGE'}[bcode];tags.append(cause);facts['before_first_deletion_code']=bcode
   elif event=='REMOVED_FINAL_FP_AT_ANCHOR':tags.append({1:'REMOVED_BY_SCORE_FILTER',2:'REMOVED_BY_LARGE_Z_FILTER',3:'REMOVED_BY_TOP1000',4:'REMOVED_BY_NMS_SUPPRESSION',5:'REMOVED_BY_RANGE'}[acode]);facts['after_first_deletion_code']=acode
   if event=='TP_TO_FP_AT_ANCHOR':
    require(bgi is not None,'BEFORE_TP_GT_ID');samegt=float(av[bgi]);facts['before_matched_physical_GT_id']=GT_MAP[sid,bgi]['physical_object_id'];facts['after_IoU_to_before_matched_GT']=samegt
    if samegt<.7:tags.append('SAME_ANCHOR_GEOMETRY_BELOW_PREVIOUS_GT_THRESHOLD')
   if event=='FP_TO_TP_AT_ANCHOR':
    require(agi is not None,'AFTER_TP_GT_ID');facts['after_matched_physical_GT_id']=GT_MAP[sid,agi]['physical_object_id'];facts['before_IoU_to_after_matched_GT']=float(bv[agi])
    if float(bv[agi])<.7:tags.append('SAME_ANCHOR_GEOMETRY_CROSSES_GT_THRESHOLD')
   if aa=='FP':
    if float(av.max(initial=0.))>=.7:tags.append('AFTER_FP_UNIQUE_GT_MATCH_COMPETITION');facts['after_eligible_GT_ids']=[GT_MAP[sid,int(i)]['physical_object_id'] for i in np.flatnonzero(av>=.7)]
    else:tags.append('AFTER_FP_BELOW_ALL_GT_IOU70')
   if bs=='FP' and float(bv.max(initial=0.))>=.7:tags.append('BEFORE_FP_UNIQUE_GT_MATCH_COMPETITION')
   if event=='PERSISTENT_FP_AT_ANCHOR':tags.append('PERSISTENT_GRID_CANDIDATE_FP_NOT_PHYSICAL_OBJECT_ID')
   b_sup=int(b['suppressor'][cid]);a_sup=int(a['suppressor'][cid]);bpr=proxy.get(('before',cid));apr=proxy.get(('after',cid));scoreb=float(b['scores_all'][cid]);scorea=float(a['scores_all'][cid]);centerb=b['raw_corners'][cid].mean(0);centera=a['raw_corners'][cid].mean(0)
   row={'sample_idx':sid,'scene_id':COUNTS[sid,before]['scene_id'],'scene_frame':COUNTS[sid,before]['scene_frame'],'contrast':contrast,'analysis_role':role,'condition_before':before,'condition_after':after,'iou_threshold':.7,'anchor_id':cid,'head_bev_query_index':cid//2,'candidate_correspondence':'same immutable head grid/anchor row, not a physical object','event_type':event,'FP_before':int(cid in fb),'FP_after':int(cid in fa),'delta_FP_contribution':int(cid in fa)-int(cid in fb),'final_status_before':bs,'final_status_after':aa,'final_prediction_index_before':bpi,'final_prediction_index_after':api,'final_matched_GT_idx_before':bgi,'final_matched_GT_idx_after':agi,'final_matched_physical_GT_before':GT_MAP[sid,bgi]['physical_object_id'] if bgi is not None else None,'final_matched_physical_GT_after':GT_MAP[sid,agi]['physical_object_id'] if agi is not None else None,'score_before':scoreb,'score_after':scorea,'score_delta':scorea-scoreb,'max_GT_IoU_before':float(bv.max(initial=0.)),'max_GT_IoU_after':float(av.max(initial=0.)),'raw_corners_before':json.dumps(b['raw_corners'][cid].tolist()),'raw_corners_after':json.dumps(a['raw_corners'][cid].tolist()),'same_anchor_BEV_IoU':pairiou,'center_displacement_m':float(np.linalg.norm(centera-centerb)),'substantial_geometry_shift_BEV_IoU_below_0p5':pairiou<.5,'raw_geometry_changed':not np.array_equal(b['raw_corners'][cid],a['raw_corners'][cid]),'first_deletion_code_before':bcode,'first_deletion_code_after':acode,'native_suppressor_before':b_sup if b_sup>=0 else None,'native_suppressor_after':a_sup if a_sup>=0 else None,'suppressor_score_before':float(b['scores_all'][b_sup]) if b_sup>=0 else None,'suppressor_score_after':float(a['scores_all'][a_sup]) if a_sup>=0 else None,'suppressor_IoU_before':float(b['suppressor_overlap'][cid]) if b_sup>=0 else None,'suppressor_IoU_after':float(a['suppressor_overlap'][cid]) if a_sup>=0 else None,'observable_path_labels':json.dumps(tags),'direct_path_facts':json.dumps(facts),'geometric_proxy_before':json.dumps(bpr) if bpr else None,'geometric_proxy_after':json.dumps(apr) if apr else None,'cross_condition_physical_FP_identity':'NOT_ESTABLISHED_FROM_ANCHOR_OR_GEOMETRY_PROXY','source_before':LOCS[sid,before]['absolute_path'],'source_after':LOCS[sid,after]['absolute_path'],'source_sha_before':LOCS[sid,before]['sha256'],'source_sha_after':LOCS[sid,after]['sha256'],'lineage_key_before':json.dumps([sid,before,cid]),'lineage_key_after':json.dumps([sid,after,cid])}
   FP_ROWS.append(row);eventrows.append(row)
  delta=len(fa)-len(fb);require(sum(r['delta_FP_contribution'] for r in eventrows)==delta,'ANCHOR_FP_COUNTING_IDENTITY');require(delta==COUNTS[sid,after]['FP70']-COUNTS[sid,before]['FP70'],'FP_DIFF_BASELINE')
  nadd=counter['ADDED_FINAL_FP_AT_ANCHOR'];tp2fp=counter['TP_TO_FP_AT_ANCHOR'];nremove=counter['REMOVED_FINAL_FP_AT_ANCHOR'];fp2tp=counter['FP_TO_TP_AT_ANCHOR'];require(delta==nadd+tp2fp-nremove-fp2tp,'FP_TRANSITION_PARTITION')
  FP_SUMMARY.append({'sample_idx':sid,'scene_id':COUNTS[sid,before]['scene_id'],'contrast':contrast,'analysis_role':role,'condition_before':before,'condition_after':after,'FP_before':len(fb),'FP_after':len(fa),'delta_FP':delta,**{k:counter[k] for k in ('ADDED_FINAL_FP_AT_ANCHOR','TP_TO_FP_AT_ANCHOR','PERSISTENT_FP_AT_ANCHOR','REMOVED_FINAL_FP_AT_ANCHOR','FP_TO_TP_AT_ANCHOR')},'observable_positive_events':json.dumps([{'anchor_id':r['anchor_id'],'event_type':r['event_type'],'labels':json.loads(r['observable_path_labels'])} for r in eventrows if r['delta_FP_contribution']>0]),'physical_object_association_not_inferred':True,'identity_uncertain_geometric_proxy_rows':sum(1 for r in eventrows if any(json.loads(r[k])['status']!='UNIQUE_GEOMETRIC_PROXY' for k in ('geometric_proxy_before','geometric_proxy_after') if r[k])),'status':'PASS'})
 require(len(FP_SUMMARY)==13,'ALL_NINE_PLUS_FOUR_424_COMPARISONS');table('R5C_FP_PROVENANCE.csv',FP_ROWS);table('R5C_FP_MARGINAL_ACCOUNTING.csv',FP_SUMMARY);table('R5C_GEOMETRIC_ASSOCIATION.csv',GEOM_ASSOC)


def interaction_matrix():
 bpatterns=csvrows(B/'R5B_GT_FOUR_CONDITION_PATTERNS.csv')
 for sid in INTERACTION_SAMPLES:
  allfinal=sorted(set().union(*(set(DATA[sid,c]['ids_final_range'].astype(int).tolist()) for c in CONDITIONS)))
  for t in (50,70):
   fp_bits={};nonzero=[]
   for cid in allfinal:
    bits=[int(candidate_state(DATA[sid,c],cid,t)[0]=='FP') for c in STATE_ORDER];iv=bits[3]-bits[2]-bits[1]+bits[0]
    item={'sample_idx':sid,'iou_threshold':t/100,'anchor_id':cid,'FP_E':bits[0],'FP_GB':bits[1],'FP_GC':bits[2],'FP_F':bits[3],'I_FP_anchor':iv,'state_order':'E,G-B,G-C,F','stage_codes':json.dumps({c:int(DATA[sid,c]['first_deletion_code'][cid]) for c in CONDITIONS}),'physical_FP_identity_not_inferred':True,'source_records':json.dumps({c:LOCS[sid,c]['absolute_path'] for c in CONDITIONS})};ANCHOR_INTER.append(item)
    if iv:nonzero.append(item)
   metrics={c:{m:COUNTS[sid,c][m+str(t)] for m in ('TP','FP','FN')} for c in CONDITIONS};I={m:metrics['F'][m]-metrics['G-C'][m]-metrics['G-B'][m]+metrics['E'][m] for m in ('TP','FP','FN')};require(sum(x['I_FP_anchor'] for x in ANCHOR_INTER if x['sample_idx']==sid and x['iou_threshold']==t/100)==I['FP'],'FP_ANCHOR_SECOND_ORDER_SUM')
   gtpatterns=[x for x in bpatterns if int(x['sample_idx'])==sid and int(round(float(x['iou_threshold'])*100))==t];gt_nonzero=[{'physical_GT_id':x['physical_object_id'],'gt_idx':int(x['gt_idx']),'pattern_E_GB_GC_F':x['pattern_bits'],'I_GT':int(x['I_GT']),'first_FN_bottlenecks':{c:TRACE_INDEX[sid,c,int(x['gt_idx']),t]['observed_final_bottleneck'] for c in CONDITIONS}} for x in gtpatterns if int(x['I_GT'])!=0];require(sum(int(x['I_GT']) for x in gtpatterns)==I['TP'],'TP_GT_INTERACTION_SUM')
   stagecounts={c:{s:len(DATA[sid,c]['ids_'+s]) for s in STAGES} for c in CONDITIONS}
   INTERACTION_ROWS.append({'sample_idx':sid,'scene_id':COUNTS[sid,'F']['scene_id'],'scene_frame':COUNTS[sid,'F']['scene_frame'],'iou_threshold':t/100,'four_condition_counts':json.dumps(metrics),'I_TP':I['TP'],'I_FP':I['FP'],'I_FN':I['FN'],'GT_state_patterns':json.dumps([{'physical_GT_id':x['physical_object_id'],'pattern':x['pattern_bits'],'I_GT':int(x['I_GT'])} for x in gtpatterns]),'nonzero_GT_interaction_stage_paths':json.dumps(gt_nonzero),'all_stage_counts':json.dumps(stagecounts),'nonzero_FP_anchor_interactions':json.dumps([{'anchor_id':x['anchor_id'],'FP_E':x['FP_E'],'FP_GB':x['FP_GB'],'FP_GC':x['FP_GC'],'FP_F':x['FP_F'],'I_FP_anchor':x['I_FP_anchor'],'stage_codes':json.loads(x['stage_codes'])} for x in nonzero]),'raw_geometry_and_score_values_change_across_conditions':any(not np.array_equal(DATA[sid,c]['raw_corners'],DATA[sid,'E']['raw_corners']) or not np.array_equal(DATA[sid,c]['scores_all'],DATA[sid,'E']['scores_all']) for c in CONDITIONS),'observable_level':'candidate/score/geometric-filter/NMS/greedy-match saved-output paths','unidentified_upstream_mechanisms':'Attention versus MetaFlow versus other feature computations cannot be isolated from these outputs; no controlled intervention performed.'})
 table('R5C_INTERACTION_MECHANISM_MATRIX.csv',INTERACTION_ROWS);table('R5C_FP_ANCHOR_INTERACTION.csv',ANCHOR_INTER)


def export_lineage():
 # Deterministic exhaustive selection, never discarding candidates in replay:
 # every score-pass candidate, every raw IoU>=0.5 against any GT, every per-GT
 # raw maximum, plus all final IDs; close these IDs over all four conditions and
 # all NMS suppressors. This preserves ALL candidate evidence at both thresholds.
 for sid in sorted({k[0] for k in DATA}):
  selected=set()
  for c in CONDITIONS:
   a=DATA[sid,c];mat=GEOMETRY_IOU[sid,c];selected.update(np.flatnonzero(np.any(mat>=.5,axis=0)).astype(int));selected.update(a['ids_score_pass'].astype(int));selected.update(np.argmax(mat,axis=1).astype(int));selected.add(int(np.argmax(a['scores_all'])));selected.update(a['ids_final_range'].astype(int))
  while True:
   extra={int(DATA[sid,c]['suppressor'][i]) for c in CONDITIONS for i in selected if int(DATA[sid,c]['suppressor'][i])>=0}-selected
   if not extra:break
   selected|=extra
  CANDIDATE_IDS[sid]=sorted(map(int,selected))
  for c in CONDITIONS:
   a=DATA[sid,c];loc=LOCS[sid,c];scorepos={int(i):p for p,i in enumerate(a['ids_score_pass'])};qpos={int(i):p for p,i in enumerate(a['ids_quality_pass'])};keepos={int(i):p for p,i in enumerate(a['ids_nms_keep'])};finalpos={int(i):p for p,i in enumerate(a['ids_final_range'])};rank={int(i):p for p,i in enumerate(a['ids_quality_pass'][a['nms_input_scores'].argsort()[::-1]])};cl=a['cls_preds'].transpose(0,2,3,1).reshape(-1);reg=a['reg_preds'].transpose(0,2,3,1).reshape(-1,7);dr=a['dir_preds'].transpose(0,2,3,1).reshape(-1,2)
   for cid in CANDIDATE_IDS[sid]:
    sp=scorepos.get(cid);kp=keepos.get(cid);pi=finalpos.get(cid);sup=int(a['suppressor'][cid]);matched={t:int(a[f'prediction_gt_idx{t}'][pi]) if pi is not None else None for t in (50,70)};bestgi=int(np.argmax(GEOMETRY_IOU[sid,c][:,cid]))
    row={'sample_idx':sid,'condition':c,'candidate_id':cid,'head_bev_query_index':cid//2,'grid_y':(cid//2)//80,'grid_x':(cid//2)%80,'anchor_rotation_index':cid%2,'cls_logit':float(cl[cid]),'score':float(a['scores_all'][cid]),'reg_delta':json.dumps(reg[cid].tolist()),'dir_logits':json.dumps(dr[cid].tolist()),'dir_bin':int(np.argmax(dr[cid])),'decoded_before_direction':json.dumps(a['decoded_before_dir'].reshape(-1,7)[cid].tolist()),'raw_corners':json.dumps(a['raw_corners'][cid].tolist()),'score_pass':sp is not None,'large_mask_if_evaluated':bool(a['large_mask'][sp]) if sp is not None else None,'z_mask_if_evaluated':bool(a['z_mask'][sp]) if sp is not None else None,'quality_pass':cid in qpos,'quality_rescoring_enabled':False,'quality_branch_prediction':None,'native_score_sort_rank_if_quality_pass':rank.get(cid),'top1000_pass':cid in set(a['ids_nms_top1000']),'native_NMS_keep':kp is not None,'native_NMS_suppressed':sup>=0,'native_suppressor_candidate_id':sup if sup>=0 else None,'native_suppressor_score':float(a['scores_all'][sup]) if sup>=0 else None,'native_suppression_BEV_IoU':float(a['suppressor_overlap'][cid]) if sup>=0 else None,'range_pass_if_evaluated':bool(a['range_mask'][kp]) if kp is not None else None,'final_prediction_index':pi,'final_status50':candidate_state(a,cid,50)[0],'final_status70':candidate_state(a,cid,70)[0],'matched_gt_idx50':matched[50] if matched[50] is not None and matched[50]>=0 else None,'matched_gt_idx70':matched[70] if matched[70] is not None and matched[70]>=0 else None,'matched_physical_GT50':GT_MAP[sid,matched[50]]['physical_object_id'] if matched[50] is not None and matched[50]>=0 else None,'matched_physical_GT70':GT_MAP[sid,matched[70]]['physical_object_id'] if matched[70] is not None and matched[70]>=0 else None,'max_GT_IoU':float(GEOMETRY_IOU[sid,c][bestgi,cid]),'argmax_GT_idx':bestgi,'argmax_physical_GT_id':GT_MAP[sid,bestgi]['physical_object_id'],'first_deletion_code':int(a['first_deletion_code'][cid]),'source_path':loc['absolute_path'],'source_sha256':loc['sha256'],'lineage_key':json.dumps([sid,c,cid])}
    LINEAGE.append(row)
    if sup>=0:NMS_EDGES.append({'sample_idx':sid,'condition':c,'suppressed_id':cid,'suppressor_id':sup,'suppressed_score':float(a['scores_all'][cid]),'suppressor_score':float(a['scores_all'][sup]),'pair_BEV_IoU':float(a['suppressor_overlap'][cid]),'threshold':.15,'suppressed_native_keep':False,'suppressor_native_keep':sup in keepos,'source_path':loc['absolute_path'],'source_sha256':loc['sha256']})
   selected_ids=np.asarray(CANDIDATE_IDS[sid],np.int32);export={'scores_all_full':a['scores_all'],'IoU_GT_full':GEOMETRY_IOU[sid,c],'candidate_ids':selected_ids,'raw_corners':a['raw_corners'][selected_ids],'scores_all_selected':a['scores_all'][selected_ids],'cls_logits_selected':cl[selected_ids],'reg_delta_selected':reg[selected_ids],'dir_logits_selected':dr[selected_ids],'decoded_before_dir_selected':a['decoded_before_dir'].reshape(-1,7)[selected_ids],'GT_canonical':a['GT_canonical'],'IoU_GT_selected':GEOMETRY_IOU[sid,c][:,selected_ids],'suppressor_selected':a['suppressor'][selected_ids],'suppressor_overlap_selected':a['suppressor_overlap'][selected_ids],'first_deletion_code_selected':a['first_deletion_code'][selected_ids],'final_corners':a['final_corners'],'final_scores':a['final_scores'],**{'ids_'+s:a['ids_'+s] for s in STAGES},**{k+str(t):a[k+str(t)] for k in ('prediction_gt_idx','gt_tp','prediction_fp') for t in (50,70)}}
   with (ROOT/'evidence'/f'sample_{sid}_{c}_selected_stage_evidence.npz').open('wb') as f:np.savez_compressed(f,**export)
 table('R5C_CANDIDATE_LINEAGE.csv',LINEAGE);table('R5C_NMS_SUPPRESSOR_EDGES.csv',NMS_EDGES)
 write('R5C_CANDIDATE_SELECTION_AND_PACKAGE_SCOPE.json',{'rule':'all score-pass + all raw IoU>=0.5 against any frozen GT + per-GT raw argmax + raw global score argmax + all final IDs; union same anchor IDs over all four conditions; close under NMS suppressors','selection_is_for_export_only_full12800_replayed_for_all32':True,'every_high_IoU_candidate_at_0p5_or_0p7_is_included':True,'anchors_per_sample':{str(s):len(v) for s,v in CANDIDATE_IDS.items()},'full_scores_and_GT_IoU_matrices_exported_for_all12800':True,'full_raw_geometry_not_exported_for_unselected_low_score_low_IoU_anchors':True,'unselected_raw_candidates_remain_in_verified_server_NPZ_paths':True,'FP_correspondence':'fixed grid identity; optional geometry proxy threshold .5 requiring mutual degree1; no physical FP identity claim'})


def case_manifest():
 rows=[]
 for sid,oid in CORE:
  gi=gi_for(sid,oid);g=GT_MAP[sid,gi]
  for c in CONDITIONS:
   for t in (50,70):rows.append({'case_id':f'CORE_{sid}_{oid}_{c}_{t}','case_type':'PREDECLARED_CORE_GT','sample_idx':sid,'scene_id':g['scene_id'],'scene_frame':g['scene_frame'],'physical_GT_id':oid,'gt_idx':gi,'condition':c,'condition_before':None,'condition_after':None,'iou_threshold':t/100,'source_path':LOCS[sid,c]['absolute_path'],'source_sha256':LOCS[sid,c]['sha256']})
 for x in FP_SUMMARY:
  before=x['condition_before'];after=x['condition_after'];sid=x['sample_idx'];rows.append({'case_id':f"FP_{sid}_{x['contrast']}_70",'case_type':x['analysis_role'],'sample_idx':sid,'scene_id':x['scene_id'],'physical_GT_id':None,'condition':None,'condition_before':before,'condition_after':after,'iou_threshold':.7,'source_path':json.dumps([LOCS[sid,before]['absolute_path'],LOCS[sid,after]['absolute_path']]),'source_sha256':json.dumps([LOCS[sid,before]['sha256'],LOCS[sid,after]['sha256']])})
 for sid in INTERACTION_SAMPLES:
  for t in (50,70):rows.append({'case_id':f'INTERACTION_{sid}_{t}','case_type':'PREDECLARED_INTERACTION_SAMPLE','sample_idx':sid,'scene_id':COUNTS[sid,'F']['scene_id'],'physical_GT_id':None,'condition':None,'condition_before':None,'condition_after':None,'iou_threshold':t/100,'source_path':json.dumps([LOCS[sid,c]['absolute_path'] for c in CONDITIONS]),'source_sha256':json.dumps([LOCS[sid,c]['sha256'] for c in CONDITIONS])})
 require(len(rows)==45 and len({x['case_id'] for x in rows})==45,'PREDECLARED_CASE_MANIFEST');table('R5C_CASE_MANIFEST.csv',rows)


def main():
 read_sources_and_replay();full_matching_trace();gt_traces();transition_mechanisms();fp_provenance();interaction_matrix();export_lineage();case_manifest()
 unresolved=[x for x in MECHANISMS if x['UNRESOLVED']];protected=load(ROOT/'R5C_HISTORY_READ_ONLY_SEAL.json')['files'];changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];require(not changed,'HISTORICAL_FILE_CHANGED '+str(changed));require(not DENIED and 'torch' not in sys.modules and 'opencood' not in sys.modules,'CPU_ONLY_GUARD')
 write('R5C_PRODUCER_RESULT.json',{'state':'PRODUCER_COMPLETE_PENDING_INDEPENDENT_VALIDATION','all32_stage_replays_pass':True,'trace_rows':len(TRACES),'candidate_lineage_rows':len(LINEAGE),'GT_transition_mechanism_rows':len(MECHANISMS),'FP_event_rows':len(FP_ROWS),'FP_comparison_count':len(FP_SUMMARY),'interaction_rows':len(INTERACTION_ROWS),'NMS_edges':len(NMS_EDGES),'required_chain_UNRESOLVED':unresolved,'physical_FP_identity_is_not_measured':True,'upstream_module_causality_NOT_IDENTIFIED':True,'historical_files_unchanged':len(protected),'new_forward':'NO','CUDA_initialized':'NO','history_modified':'NO','torch_imported':False,'opencood_imported':False,'R5D_executed':False})
 print('R5C_PRODUCER_COMPLETE',len(TRACES),len(LINEAGE),len(FP_ROWS),'UNRESOLVED',len(unresolved),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write('R5C_PRODUCER_FAILURE.json',{'state':'R5C_FAIL','error':str(e),'traceback':traceback.format_exc(),'new_forward':'NO','CUDA_initialized':'NO','history_modified':'NO'});raise
