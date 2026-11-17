"""Produce exactly 16 frozen G/S combinations +3 specified two-Score swaps.
Replays complete native postprocessing from all 12800 recorded candidates.
No backbone, head, forward, model/device import or upstream intervention.
"""
from cpu_common import *
SRC={};RUN={};SPEC={};SUMMARY=[];FOCAL_ROWS=[];ALL_GT=[];CAND_ROWS=[];OFF=[];SWAPS=[];EFFECT=[];PER_GT=[];IOU={};BASE_PROOFS=[]

def pipeline(g,scores,gt,io):
 n=len(scores);raw=g;require(n==12800 and raw.shape==(n,8,3) and scores.dtype==np.float32 and raw.dtype==np.float32,'FULL_REPLAY_INPUTS');scoremask=scores>CONFIG['target_args']['score_threshold'];sc=np.flatnonzero(scoremask).astype(np.int32);bb=raw[sc];span=bb.max(1)-bb.min(1);lm=(span[:,0]<=6)&(span[:,1]<=6)&(span[:,1]!=0);zm=(bb[:,:,2].min(1)>=-3)&(bb[:,:,2].max(1)<=1);q=sc[lm&zm];top=q[scores[q].argsort()[::-1][:1000]];keep,sup,ovs=nms_replay(raw[q],scores[q],CONFIG['nms_thresh']);kept=q[keep];suppressor=np.full(n,-1,np.int32);overlaps=np.zeros(n,np.float32)
 for child,owner in sup.items():suppressor[q[child]]=q[owner];overlaps[q[child]]=ovs[child]
 lim=np.asarray(CONFIG['gt_range']);nr=raw[kept];rg=((nr>=lim[:3])&(nr<=lim[3:])).all(2).sum(1)>=8;final=kept[rg];stages=[np.arange(n,dtype=np.int32),sc,q,top,kept,final];code=np.ones(n,np.int8)
 for k,ids in enumerate(stages[1:],2):code[ids]=k
 out={'raw_corners':raw,'scores_all':scores,'GT_canonical':gt,'IoU_GT_raw':io,'score_mask':scoremask,'large_mask':lm,'z_mask':zm,'quality_mask':lm&zm,'native_nms_keep':keep,'range_mask':rg,'final_corners':raw[final],'final_scores':scores[final],'suppressor':suppressor,'suppressor_overlap':overlaps,'first_deletion_code':code,**{'ids_'+k:ids for k,ids in zip(STAGES,stages)}}
 for t in (50,70):
  m=unique_match(out['final_corners'],out['final_scores'],gt,t/100);out[f'prediction_gt_idx{t}']=m['matched'];out[f'gt_tp{t}']=m['flags'];out[f'prediction_fp{t}']=m['matched']<0
  require(int(m['flags'].sum())+int((m['matched']<0).sum())==len(final),'TP_FP_FINAL');require(int(m['flags'].sum())+int((m['flags']==0).sum())==len(gt),'TP_FN_GT')
 return out

def execute(spec):
 sid=spec['sample_idx'];gc=spec['geometry_condition'];sc=spec['score_condition'];g=SRC[sid,gc];s=SRC[sid,sc]['scores_all'];apply=s.copy() if spec['experiment_type']=='SCORE_SWAP_7028_7030' else s
 if spec['experiment_type']=='SCORE_SWAP_7028_7030':
  require(float(s[7028])>.2 and float(s[7030])>.2 and s[7028]!=s[7030],'SWAP_TWO_DISTINCT_SCOREPASS');apply[[7028,7030]]=s[[7030,7028]];require(np.flatnonzero(apply!=s).tolist()==[7028,7030],'ONLY_TWO_SCORES_CHANGED')
 a=pipeline(g['raw_corners'],apply,g['GT_canonical'],IOU[sid,gc]);rid=spec['run_id'];RUN[rid]=a;SPEC[rid]=spec
 out={**a,'geometry_source_condition':np.asarray(gc),'score_source_condition':np.asarray(sc),'geometry_sha256':np.asarray(digest(g['raw_corners'])),'geometry_decoded_before_dir_sha256':np.asarray(digest(g['decoded_before_dir'])),'geometry_dir_logits_sha256':np.asarray(digest(g['dir_preds'])),'geometry_reg_preds_sha256':np.asarray(digest(g['reg_preds'])),'applied_score_sha256':np.asarray(digest(apply)),'run_id':np.asarray(rid),'sample_idx':np.asarray(sid),'experiment_type':np.asarray(spec['experiment_type'])}
 with run_path(rid).open('wb') as f:np.savez_compressed(f,**out)
 cnt=counts(a);focal={str(t):{oid:GT_info(a,sid,next(gi for (ss,gi),r in GM.items() if ss==sid and r['physical_object_id']==oid),t)['TP_FN'] for oid in FOCAL[sid]} for t in (50,70)}
 SUMMARY.append({**spec,**cnt,'raw_candidate_count':len(a['ids_raw_decoded']),'score_pass_count':len(a['ids_score_pass']),'quality_pass_count':len(a['ids_quality_pass']),'top1000_count':len(a['ids_nms_top1000']),'NMS_keep_count':len(a['ids_nms_keep']),'NMS_suppressed_count':int((a['suppressor']>=0).sum()),'final_range_removed':len(a['ids_nms_keep'])-len(a['ids_final_range']),'focal_GT_states':j(focal),'geometry_sha256':digest(g['raw_corners']),'score_sha256':digest(apply),'full_pipeline_evidence':str(run_path(rid)),'full_pipeline_evidence_sha256':sha(run_path(rid)),'geometry_source_path':LOC[sid,gc]['absolute_path'],'geometry_source_sha256':LOC[sid,gc]['sha256'],'score_source_path':LOC[sid,sc]['absolute_path'],'score_source_sha256':LOC[sid,sc]['sha256'],'new_model_forward':0})
 print('CPU_POSTPROCESS',rid,'TP/FP/FN50',cnt['TP50'],cnt['FP50'],cnt['FN50'],'70',cnt['TP70'],cnt['FP70'],cnt['FN70'],flush=True)
 return a

def validate_original(a,source,sid,c,rid):
 fields=['raw_corners','scores_all','score_mask','large_mask','z_mask','quality_mask','native_nms_keep','range_mask','final_corners','final_scores','suppressor','suppressor_overlap','first_deletion_code']+['ids_'+s for s in STAGES]+[x+str(t) for x in ['gt_tp','prediction_gt_idx','prediction_fp'] for t in (50,70)]
 for k in fields:same(a[k],source[k],'BASELINE_NATIVE_EXACT '+rid+' '+k)
 same(a['IoU_GT_raw'],source['IoU_GT_raw'],'BASELINE_BEV_IoU_EXISTING_TOL',1e-6)
 crows=rows(CROOT/'R5C_GT_STAGE_TRACE.csv')
 for r in crows:
  if int(r['sample_idx'])!=sid or r['condition']!=c:continue
  gi=int(r['gt_idx']);t=int(round(float(r['iou_threshold'])*100));x=GT_info(a,sid,gi,t);require(x['TP_FN']==r['TP_FN'],'BASELINE_R5C_GT_STATUS')
  for s in STAGES:require(x['high_count_'+s]==int(r['high_IoU_count_'+s]) and json.loads(x['high_ids_'+s])==json.loads(r['high_IoU_ids_'+s]),'BASELINE_R5C_HIGH_IOU_STAGE')
 BASE_PROOFS.append({'run_id':rid,'sample_idx':sid,'condition':c,'all_candidate_stage_final_and_GT_arrays_exact':True,'R5C_high_IoU_GT_sets_exact':True,'source_path':LOC[sid,c]['absolute_path'],'source_sha256':LOC[sid,c]['sha256']})

def all_GT_and_candidates():
 selection={sid:{6288,6292,6452,7024,7026,7028,7030,7184,7186,7188} if sid==725 else {6512,6514,6516,6518} for sid in (725,324)}
 for (sid,c),s in SRC.items():
  selection[sid]|=set(map(int,s['ids_score_pass']));selection[sid]|=set(np.flatnonzero(np.any(IOU[sid,c]>=.5,axis=0)).astype(int).tolist())
 for rid,a in RUN.items():
  sid=SPEC[rid]['sample_idx'];selection[sid]|=set(map(int,a['ids_final_range']))
 for rid,a in RUN.items():
  sp=SPEC[rid];sid=sp['sample_idx']
  for gi in range(len(a['GT_canonical'])):
   g=GM[sid,gi];oid=g['physical_object_id']
   for t in (50,70):
    info=GT_info(a,sid,gi,t);row={**sp,'scene_id':int(g['scene_id']),'scene_frame':int(g['scene_frame']),'gt_idx':gi,'physical_GT_id':oid,'GT_hash':g['GT_hash'],'iou_threshold':t/100,'is_focal_GT':oid in FOCAL[sid],**info,'evidence_path':str(run_path(rid))};ALL_GT.append(row)
    if oid in FOCAL[sid]:FOCAL_ROWS.append(row)
  for cid in sorted(selection[sid]):
   spos=np.flatnonzero(a['ids_score_pass']==cid);kpos=np.flatnonzero(a['ids_nms_keep']==cid);sup=int(a['suppressor'][cid]);q=a['ids_quality_pass'];ordered=q[a['scores_all'][q].argsort()[::-1]];rpos=np.flatnonzero(ordered==cid);st={t:ci_state(a,cid,t) for t in (50,70)};iv=a['IoU_GT_raw'][:,cid]
   CAND_ROWS.append({**sp,'candidate_id':cid,'query_index':cid//2,'grid_y':cid//2//80,'grid_x':cid//2%80,'rotation_index':cid%2,'score':float(a['scores_all'][cid]),'raw_corner_hash':digest(a['raw_corners'][cid]),'geometry_IoU_by_physical_GT':j({GM[sid,gi]['physical_object_id']:float(x) for gi,x in enumerate(iv)}),'score_pass':bool(len(spos)),'large_pass_if_evaluated':bool(a['large_mask'][spos[0]]) if len(spos) else None,'z_pass_if_evaluated':bool(a['z_mask'][spos[0]]) if len(spos) else None,'quality_pass':cid in set(q),'score_sort_rank_zero_based':int(rpos[0]) if len(rpos) else None,'top1000_pass':cid in set(a['ids_nms_top1000']),'NMS_keep':bool(len(kpos)),'suppressor_candidate_id':sup if sup>=0 else None,'suppressed_score':float(a['scores_all'][cid]) if sup>=0 else None,'suppressor_score':float(a['scores_all'][sup]) if sup>=0 else None,'suppression_pair_BEV_IoU':float(a['suppressor_overlap'][cid]) if sup>=0 else None,'range_pass_if_evaluated':bool(a['range_mask'][kpos[0]]) if len(kpos) else None,'final_status50':st[50][0],'final_status70':st[70][0],'matched_gt_idx50':st[50][2],'matched_gt_idx70':st[70][2],'final_prediction_index':st[50][1],'first_deletion_code':int(a['first_deletion_code'][cid]),'full_raw_geometry_evidence':str(run_path(rid))})
 table('R5D_FOCAL_GT_OUTCOMES.csv',FOCAL_ROWS);table('R5D_ALL_GT_OUTCOMES.csv',ALL_GT);table('R5D_CANDIDATE_STAGE_CHANGES.csv',CAND_ROWS)
 write('R5D_CANDIDATE_EXPORT_RULE.json',{'selection':'union all six source Score-pass candidates, all >=0.5 GT-IoU raw candidates, named focal anchors and every final candidate across19; per-sample same ID selection','all19_pipeline_runs_start_at_full12800':True,'all_post_Score_candidates_and_NMS_competitors_in_CSV':True,'all_full_raw_geometry_and_scores_in_each_run_NPZ':True,'counts_per_sample':{str(s):len(v) for s,v in selection.items()}})


def compare_runs(rid,refid,role):
 a=RUN[rid];b=RUN[refid];sp=SPEC[rid];sid=sp['sample_idx'];cf=counts(a);cb=counts(b);ka=set(map(int,a['ids_nms_keep']));kb=set(map(int,b['ids_nms_keep']));fa=set(map(int,a['ids_final_range']));fb=set(map(int,b['ids_final_range']));edges=np.flatnonzero(a['suppressor']!=b['suppressor']).astype(int).tolist();out=[]
 for t in (50,70):
  changes=[];focal=[];other=[];matched_changes=[]
  for gi in range(len(a['GT_canonical'])):
   ga=GT_info(a,sid,gi,t);gb=GT_info(b,sid,gi,t);oid=GM[sid,gi]['physical_object_id']
   if ga['TP_FN']!=gb['TP_FN']:
    x={'physical_GT_id':oid,'gt_idx':gi,'before':gb['TP_FN'],'after':ga['TP_FN'],'before_match':gb['matched_candidate_id'],'after_match':ga['matched_candidate_id']};changes.append(x);(focal if oid in FOCAL[sid] else other).append(x)
   elif ga['matched_candidate_id']!=gb['matched_candidate_id']:matched_changes.append({'physical_GT_id':oid,'before_match':gb['matched_candidate_id'],'after_match':ga['matched_candidate_id']})
  fpb={int(cid) for cid,gi in zip(b['ids_final_range'],b[f'prediction_gt_idx{t}']) if gi<0};fpa={int(cid) for cid,gi in zip(a['ids_final_range'],a[f'prediction_gt_idx{t}']) if gi<0};tp2fp=[cid for cid in sorted(fpa-fpb) if ci_state(b,cid,t)[0]=='TP'];fp2tp=[cid for cid in sorted(fpb-fpa) if ci_state(a,cid,t)[0]=='TP'];newpredfp=[cid for cid in sorted(fpa-fpb) if cid not in fb];gonepredfp=[cid for cid in sorted(fpb-fpa) if cid not in fa]
  require(len(fpa)-len(fpb)==len(newpredfp)+len(tp2fp)-len(gonepredfp)-len(fp2tp),'OFFTARGET_FP_PARTITION')
  row={**sp,'reference_run_id':refid,'comparison_role':role,'iou_threshold':t/100,**{'delta_'+m:cf[m+str(t)]-cb[m+str(t)] for m in ['TP','FP','FN']},'delta_final_prediction_count':len(fa)-len(fb),'all_GT_status_changes':j(changes),'focal_GT_status_changes':j(focal),'other_GT_status_changes':j(other),'GT_match_anchor_changes_without_status_change':j(matched_changes),'NMS_keep_added':j(sorted(ka-kb)),'NMS_keep_removed':j(sorted(kb-ka)),'NMS_suppressor_changed_candidate_ids':j(edges),'NMS_suppressor_changes':j([{'candidate_id':cid,'before_suppressor':int(b['suppressor'][cid]) if b['suppressor'][cid]>=0 else None,'after_suppressor':int(a['suppressor'][cid]) if a['suppressor'][cid]>=0 else None} for cid in edges]),'final_candidate_added':j(sorted(fa-fb)),'final_candidate_removed':j(sorted(fb-fa)),'FP_anchor_added':j(sorted(fpa-fpb)),'FP_anchor_removed':j(sorted(fpb-fpa)),'added_final_FP_ids':j(newpredfp),'removed_final_FP_ids':j(gonepredfp),'TP_to_FP_ids':j(tp2fp),'FP_to_TP_ids':j(fp2tp),'FP_physical_identity_not_established':True};out.append(row);OFF.append(row)
 return out


def differences_and_swaps():
 for p in PAIRS:
  ids={c:p['pair_id']+'_'+c for c in COMBOS};sid=p['sample_idx']
  for c,rid in ids.items():
   for base in ['G0_S0','G1_S1']:
    if rid!=ids[base]:compare_runs(rid,ids[base],'FACTORIAL_VS_ORIGINAL_'+base)
  for t in (50,70):
   measures={c:counts(RUN[rid]) for c,rid in ids.items()};row={'pair_id':p['pair_id'],'sample_idx':sid,'iou_threshold':t/100}
   for name in ['TP','FP','FN','final_prediction_count']:
    n=name+str(t) if name!='final_prediction_count' else name;v={c:measures[c][n] for c in COMBOS};d={'Delta_G_given_S0':v['G1_S0']-v['G0_S0'],'Delta_G_given_S1':v['G1_S1']-v['G0_S1'],'Delta_S_given_G0':v['G0_S1']-v['G0_S0'],'Delta_S_given_G1':v['G1_S1']-v['G1_S0'],'I_GS':v['G1_S1']-v['G1_S0']-v['G0_S1']+v['G0_S0']};require(d['I_GS']==d['Delta_G_given_S1']-d['Delta_G_given_S0']==d['Delta_S_given_G1']-d['Delta_S_given_G0'],'FACTORIAL_IDENTITY')
    row.update({k+'_'+name:x for k,x in d.items()});row['four_counts_'+name]=j(v)
   EFFECT.append(row)
   for gi in range(len(RUN[ids['G0_S0']]['GT_canonical'])):
    bits={c:int(RUN[rid][f'gt_tp{t}'][gi]) for c,rid in ids.items()};PER_GT.append({'pair_id':p['pair_id'],'sample_idx':sid,'physical_GT_id':GM[sid,gi]['physical_object_id'],'gt_idx':gi,'iou_threshold':t/100,'states_order':'G0_S0,G0_S1,G1_S0,G1_S1','state_bits':''.join(str(bits[c]) for c in COMBOS),'I_GT':bits['G1_S1']-bits['G1_S0']-bits['G0_S1']+bits['G0_S0']})
 for p in PAIRS[:3]:
  sid=725;c=p['condition1'];rid='SWAP_'+c;ref=p['pair_id']+'_G1_S1';a=RUN[rid];b=RUN[ref];off=compare_runs(rid,ref,'TWO_SCORE_SWAP_VS_SAME_ORIGINAL');gi=next(gi for (s,gi),g in GM.items() if s==725 and g['physical_object_id']=='4071');row={'run_id':rid,'sample_idx':sid,'condition':c,'reference_run_id':ref,'changed_score_ids':j(np.flatnonzero(a['scores_all']!=b['scores_all']).astype(int).tolist()),'changed_geometry_count':int(np.any(a['raw_corners']!=b['raw_corners'],axis=(1,2)).sum())}
  for cid in [7028,7030]:
   row.update({f'candidate_{cid}_score_before':float(b['scores_all'][cid]),f'candidate_{cid}_score_after':float(a['scores_all'][cid]),f'candidate_{cid}_NMS_keep_before':cid in set(b['ids_nms_keep']),f'candidate_{cid}_NMS_keep_after':cid in set(a['ids_nms_keep']),f'candidate_{cid}_suppressor_before':int(b['suppressor'][cid]) if b['suppressor'][cid]>=0 else None,f'candidate_{cid}_suppressor_after':int(a['suppressor'][cid]) if a['suppressor'][cid]>=0 else None,f'candidate_{cid}_GT4071_IoU':float(a['IoU_GT_raw'][gi,cid])})
  for t in (50,70):
   bi=GT_info(b,sid,gi,t);ai=GT_info(a,sid,gi,t);row.update({f'GT4071_before{t}':bi['TP_FN'],f'GT4071_after{t}':ai['TP_FN'],f'GT4071_match_after{t}':ai['matched_candidate_id'],f'GT4071_match_IoU_after{t}':ai['matched_IoU'],f'GT4071_match_score_after{t}':ai['matched_score'],f'all_GT_status_changes{t}':off[(t-50)//20]['all_GT_status_changes'],f'other_GT_status_changes{t}':off[(t-50)//20]['other_GT_status_changes'],**{f'delta_{n}{t}':counts(a)[n+str(t)]-counts(b)[n+str(t)] for n in ['TP','FP','FN']}})
  row['NMS_suppressor_changes']=off[0]['NMS_suppressor_changes'];row['NMS_keep_added']=off[0]['NMS_keep_added'];row['NMS_keep_removed']=off[0]['NMS_keep_removed'];row['full_pipeline_evidence']=str(run_path(rid));SWAPS.append(row)
 table('R5D_OFF_TARGET_EFFECTS.csv',OFF);table('R5D_NMS_SCORE_SWAP.csv',SWAPS);table('R5D_GEOMETRY_SCORE_INTERACTION.csv',EFFECT);table('R5D_PER_GT_FACTORIAL.csv',PER_GT)

def main():
 global SRC
 require(load(ROOT/'R5D_R5C_EXPORT_PRE_HYBRID_VALIDATION.json')['state']=='R5C_EXPORT_GATE_PASS','R5C_EXPORT_GATE_REQUIRED_BEFORE_PRODUCTION')
 SRC,meta,proof=load_sources(export=True);initial={key:{n:digest(a[n]) for n in ['raw_corners','scores_all','decoded_before_dir','reg_preds','dir_preds','GT_canonical','ids_raw_decoded']} for key,a in SRC.items()}
 for key,a in SRC.items():IOU[key]=matrix(a['GT_canonical'],a['raw_corners']);same(IOU[key],a['IoU_GT_raw'],'SOURCE_RAW_BEV_IOU',1e-6)
 manifest=[]
 for p in PAIRS:
  sid=p['sample_idx'];manifest.append({**p,'focal_GT':j(p['focal_GT']),'candidate_count':12800,'candidate_ID_sha256':digest(SRC[sid,p['condition0']]['ids_raw_decoded']),'geometry0_sha256':digest(SRC[sid,p['condition0']]['raw_corners']),'geometry1_sha256':digest(SRC[sid,p['condition1']]['raw_corners']),'score0_sha256':digest(SRC[sid,p['condition0']]['scores_all']),'score1_sha256':digest(SRC[sid,p['condition1']]['scores_all']),'source0_path':LOC[sid,p['condition0']]['absolute_path'],'source0_sha256':LOC[sid,p['condition0']]['sha256'],'source1_path':LOC[sid,p['condition1']]['absolute_path'],'source1_sha256':LOC[sid,p['condition1']]['sha256'],'frozen_alignment_proof':str(ROOT/'R5D_INPUT_ALIGNMENT_VALIDATION.json')})
 table('R5D_PAIR_MANIFEST.csv',manifest)
 # All eight ORIGINAL records are accepted before any hybrid calculation.
 for p in PAIRS:
  for g,s in [(0,0),(1,1)]:
   combo=f'G{g}_S{s}';spec={'run_id':p['pair_id']+'_'+combo,'pair_id':p['pair_id'],'sample_idx':p['sample_idx'],'combination':combo,'geometry_condition':p['condition'+str(g)],'score_condition':p['condition'+str(s)],'experiment_type':'ORIGINAL_BASELINE'};a=execute(spec);validate_original(a,SRC[p['sample_idx'],p['condition'+str(g)]],p['sample_idx'],p['condition'+str(g)],spec['run_id'])
 write('R5D_ORIGINAL_BASELINE_VALIDATION.json',{'passed':True,'baseline_records':BASE_PROOFS,'all_originals_passed_before_any_hybrid':True})
 for p in PAIRS:
  for g,s in [(0,1),(1,0)]:execute({'run_id':p['pair_id']+f'_G{g}_S{s}','pair_id':p['pair_id'],'sample_idx':p['sample_idx'],'combination':f'G{g}_S{s}','geometry_condition':p['condition'+str(g)],'score_condition':p['condition'+str(s)],'experiment_type':'OFFLINE_GEOMETRY_SCORE_HYBRID'})
 for p in PAIRS[:3]:execute({'run_id':'SWAP_'+p['condition1'],'pair_id':p['pair_id'],'sample_idx':725,'combination':'TWO_SCORE_SWAP','geometry_condition':p['condition1'],'score_condition':p['condition1'],'experiment_type':'SCORE_SWAP_7028_7030'})
 require(len(RUN)==19 and len(SUMMARY)==19,'EXACT_STUDY_RUN_COUNT');SUMMARY.sort(key=lambda r:([p['pair_id'] for p in PAIRS].index(r['pair_id']),COMBOS.index(r['combination']) if r['combination'] in COMBOS else 4));table('R5D_FACTORIAL_RESULTS.csv',[r for r in SUMMARY if r['combination'] in COMBOS]);table('R5D_ALL_RUN_RESULTS.csv',SUMMARY)
 all_GT_and_candidates();differences_and_swaps()
 for key,a in SRC.items():require(initial[key]=={n:digest(a[n]) for n in initial[key]},'SOURCE_ARRAY_MUTATED')
 hist=sealed_history_unchanged();require(not DENIED and not any(x in sys.modules for x in ['torch','opencood','cupy','pycuda','pynvml']),'CPU_ONLY_GUARD')
 write('R5D_PRODUCER_RESULT.json',{'state':'PRODUCER_COMPLETE_PENDING_INDEPENDENT_VALIDATION','factorial_runs':16,'score_swap_runs':3,'total_study_CPU_postprocess_runs':19,'all_raw_candidates_per_run':12800,'original_baselines':8,'unique_source_records':6,'focal_GT_rows':len(FOCAL_ROWS),'all_GT_rows':len(ALL_GT),'candidate_rows':len(CAND_ROWS),'off_target_comparison_rows':len(OFF),'history_files_verified_unchanged':hist,'source_original_leaf_hashes_unchanged':True,'new_forward':0,'CUDA_initializations':0,'GPU_PPU_calls':0,'historical_modifications':0,'unsupported_upstream_causal_identification':False,'no_AP_or_weighted_utility_computed':True})
 print('R5D_PRODUCER_COMPLETE',len(RUN),'NO_NEW_FORWARD',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write('R5D_PRODUCER_FAILURE.json',{'state':'R5D_FAIL','exception':str(e),'traceback':traceback.format_exc(),'new_forward':0,'CUDA_initializations':0});raise
