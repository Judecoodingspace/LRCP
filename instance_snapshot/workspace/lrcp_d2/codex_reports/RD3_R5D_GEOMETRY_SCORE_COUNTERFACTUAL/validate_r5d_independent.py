"""Separate R5D CPU validator: reads original full NPZs, no producer import.
Independent active-mask NMS + greedy matching; shared SHA-bound float32 BEV
polygon primitives only. Cross-checks run inputs, all arrays, tables and algebra.
"""
from cpu_common import *
CHECK=collections.Counter();FAIL=[];SRC={};R={};SPEC={};CNT={};IOU={}
def ck(ok,group,detail=''):
 CHECK[group]+=1
 if not bool(ok):
  FAIL.append({'group':group,'detail':str(detail)})
  if len(FAIL)<15:print('CHECK_FAIL',group,detail,flush=True)
def eq(a,b,group,detail=''):
 a=np.asarray(a);b=np.asarray(b);ck(a.shape==b.shape and np.array_equal(a,b),group,detail)
def close(a,b,group,detail='',tol=1e-6):
 a=np.asarray(a);b=np.asarray(b);ck(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.abs(a.astype(float)-b.astype(float))<=tol),group,detail)
def num(s):return None if s=='' else float(s)
def bol(s):return s=='True'
def arrlist(x):return json.loads(x)
def actual_specs():
 ck([(p['pair_id'],p['sample_idx'],p['condition0'],p['condition1']) for p in PAIRS]==[('P1',725,'E','G-B'),('P2',725,'E','G-C'),('P3',725,'E','F'),('P4',324,'G-B','F')],'exact_authorized_pair_set')
 out={}
 for p in PAIRS:
  for g,s in [(0,0),(0,1),(1,0),(1,1)]:
   combo=f'G{g}_S{s}';out[p['pair_id']+'_'+combo]={'sample_idx':p['sample_idx'],'geometry_condition':p['condition'+str(g)],'score_condition':p['condition'+str(s)],'combination':combo,'pair_id':p['pair_id'],'experiment_type':'ORIGINAL_BASELINE' if g==s else 'OFFLINE_GEOMETRY_SCORE_HYBRID'}
 for p in PAIRS[:3]:out['SWAP_'+p['condition1']]={'sample_idx':725,'geometry_condition':p['condition1'],'score_condition':p['condition1'],'combination':'TWO_SCORE_SWAP','pair_id':p['pair_id'],'experiment_type':'SCORE_SWAP_7028_7030'}
 return out

def independently_replay(raw,scores,gt,io):
 n=len(scores);mask=scores>CONFIG['target_args']['score_threshold'];sc=np.where(mask)[0].astype(np.int32);bb=raw[sc];spans=bb.max(axis=1)-bb.min(axis=1);lm=np.logical_and(np.logical_and(spans[:,0]<=6,spans[:,1]<=6),spans[:,1]!=0);zm=np.logical_and(bb[:,:,2].min(axis=1)>=-3,bb[:,:,2].max(axis=1)<=1);q=sc[lm&zm];priority=scores[q].argsort()[::-1][:1000];top=q[priority];pp=polys(raw[q]);active=np.zeros(len(q),bool);active[priority]=True;keep=[];sup=np.full(n,-1,np.int32);ov=np.zeros(n,np.float32)
 for index0 in priority:
  index=int(index0)
  if not active[index]:continue
  cid=int(q[index]);keep.append(index);active[index]=False;others=np.flatnonzero(active);values=overlap(pp[index],[pp[i] for i in others]);removed=others[values>CONFIG['nms_thresh']]
  for index2 in removed:
   child=int(q[index2]);sup[child]=cid;ov[child]=values[int(np.flatnonzero(others==index2)[0])]
  active[removed]=False
 keep=np.asarray(keep,np.int32);kept=q[keep];lim=np.asarray(CONFIG['gt_range']);nr=raw[kept];rg=np.sum(np.all((nr>=lim[:3])&(nr<=lim[3:]),axis=2),axis=1)>=8;final=kept[rg];stage=[np.arange(n,dtype=np.int32),sc,q,top,kept,final];code=np.ones(n,np.int8)
 for s,ids in enumerate(stage[1:],2):code[ids]=s
 out={'raw_corners':raw,'scores_all':scores,'GT_canonical':gt,'IoU_GT_raw':io,'score_mask':mask,'large_mask':lm,'z_mask':zm,'quality_mask':lm&zm,'native_nms_keep':keep,'range_mask':rg,'final_corners':raw[final],'final_scores':scores[final],'suppressor':sup,'suppressor_overlap':ov,'first_deletion_code':code,**{'ids_'+s:a for s,a in zip(STAGES,stage)}}
 for t in (50,70):
  available=list(range(len(gt)));p2g=np.full(len(final),-1,np.int32);flags=np.zeros(len(gt),np.int8)
  for pi0 in np.argsort(-scores[final]):
   pi=int(pi0);cid=int(final[pi]);v=io[available,cid]
   if len(available) and float(v.max())>=t/100:
    selected=available.pop(int(np.argmax(v)));p2g[pi]=selected;flags[selected]=1
  out[f'prediction_gt_idx{t}']=p2g;out[f'prediction_fp{t}']=p2g<0;out[f'gt_tp{t}']=flags
 return out

def load_and_replay():
 global SRC,SPEC
 SRC,meta,proof=load_sources(export=False);SPEC=actual_specs();rr=rows(ROOT/'R5D_ALL_RUN_RESULTS.csv');ck(len(rr)==19 and {r['run_id'] for r in rr}==set(SPEC),'exact_19_run_manifest');byid={r['run_id']:r for r in rr}
 for key,a in SRC.items():
  with np.load(source_path(*key),allow_pickle=False) as z:
   for n in SOURCE_FIELDS:eq(z[n],a[n],'packaged_source_exact_original',(key,n))
  IOU[key]=matrix(a['GT_canonical'],a['raw_corners']);close(IOU[key],a['IoU_GT_raw'],'original_BEV_IoU',key)
 for rid,sp in SPEC.items():
  sid=sp['sample_idx'];gc=sp['geometry_condition'];sc=sp['score_condition'];g=SRC[sid,gc];s=SRC[sid,sc]['scores_all'].copy()
  if rid.startswith('SWAP_'):
   ck(s[7028]>.2 and s[7030]>.2 and s[7028]!=s[7030],'swap_original_scores_above_threshold',rid);s[[7028,7030]]=s[[7030,7028]]
  expected=independently_replay(g['raw_corners'],s,g['GT_canonical'],IOU[sid,gc]);p=run_path(rid);ck(sha(p)==byid[rid]['full_pipeline_evidence_sha256'],'run_NPZ_SHA',rid)
  with np.load(p,allow_pickle=False) as z:
   a={n:z[n].copy() for n in z.files}
  for n,e in expected.items():eq(a[n],e,'independent_full_pipeline_arrays',(rid,n))
  for n in ['geometry_source_condition','score_source_condition','experiment_type']:
   field={'geometry_source_condition':'geometry_condition','score_source_condition':'score_condition','experiment_type':'experiment_type'}[n];ck(str(a[n])==sp[field],'run_source_spec',(rid,n))
  for n,raw in [('geometry_sha256',g['raw_corners']),('geometry_decoded_before_dir_sha256',g['decoded_before_dir']),('geometry_dir_logits_sha256',g['dir_preds']),('geometry_reg_preds_sha256',g['reg_preds']),('applied_score_sha256',s)]:ck(str(a[n])==digest(raw),'geometry_orientation_score_package_source_hash',(rid,n))
  eq(a['ids_raw_decoded'],np.arange(12800),'candidate_full_set_order',rid);ck(a['raw_corners'].dtype==np.float32 and a['scores_all'].dtype==np.float32,'float32_not_relaxed',rid);eq(a['GT_canonical'],g['GT_canonical'],'same_GT_coordinate_geometry',rid)
  if rid.startswith('SWAP_'):
   changed=np.flatnonzero(a['scores_all']!=SRC[sid,sc]['scores_all']).astype(int).tolist();ck(changed==[7028,7030],'only_two_scores_changed',rid);eq(a['raw_corners'],g['raw_corners'],'swap_geometry_unchanged',rid)
  if sp['experiment_type']=='ORIGINAL_BASELINE':
   for n in expected:
    if n=='IoU_GT_raw':close(a[n],g[n],'original_baseline_native_IoU_tol',rid)
    else:eq(a[n],g[n],'all_original_baseline_arrays_exact',(rid,n))
  R[rid]=a;CNT[rid]=counts(expected)
  for n,v in CNT[rid].items():ck(int(byid[rid][n])==v,'summary_raw_counts',(rid,n))
  for n,field in [('raw_candidate_count','ids_raw_decoded'),('score_pass_count','ids_score_pass'),('quality_pass_count','ids_quality_pass'),('top1000_count','ids_nms_top1000'),('NMS_keep_count','ids_nms_keep')]:ck(int(byid[rid][n])==len(expected[field]),'summary_stage_counts',(rid,n))
  for t in (50,70):
   ck(CNT[rid]['TP'+str(t)]+CNT[rid]['FN'+str(t)]==len(g['GT_canonical']),'TP_FN_GT_invariant',(rid,t));ck(CNT[rid]['TP'+str(t)]+CNT[rid]['FP'+str(t)]==len(a['final_scores']),'TP_FP_final_count_invariant',(rid,t))
  print('INDEPENDENT_CPU_REPLAY',rid,CNT[rid]['TP70'],CNT[rid]['FP70'],flush=True)
 for r in rows(CROOT/'R5C_GT_STAGE_TRACE.csv'):
  sid=int(r['sample_idx']);c=r['condition'];gi=int(r['gt_idx']);t=int(round(float(r['iou_threshold'])*100));rr=[rid for rid,sp in SPEC.items() if sp['sample_idx']==sid and sp['geometry_condition']==c and sp['experiment_type']=='ORIGINAL_BASELINE']
  for rid in rr:
   a=R[rid];high=IOU[sid,c][gi]>=t/100
   for stage in STAGES:
    actual=a['ids_'+stage][high[a['ids_'+stage]]].astype(int).tolist();ck(actual==arrlist(r['high_IoU_ids_'+stage]) and len(actual)==int(r['high_IoU_count_'+stage]),'original_baseline_vs_R5C_all_GT_stage_sets',(rid,gi,t,stage))
   ck(r['TP_FN']==('TP' if a[f'gt_tp{t}'][gi] else 'FN'),'original_baseline_vs_R5C_GT_status',(rid,gi,t))
 for r in rows(ROOT/'R5D_FACTORIAL_RESULTS.csv'):ck(r==byid[r['run_id']],'factorial_table_exact_subset',r['run_id'])
 ck(len(rows(ROOT/'R5D_FACTORIAL_RESULTS.csv'))==16,'factorial_count16')
 # E original is intentionally repeated in P1/P2/P3; arrays must agree.
 for suffix in ['P2_G0_S0','P3_G0_S0']:
  for n in ['final_corners','final_scores','ids_nms_keep','suppressor','gt_tp50','gt_tp70']:eq(R['P1_G0_S0'][n],R[suffix][n],'repeated_original_consistency',(suffix,n))


def validate_GT_and_candidates():
 gtrows=rows(ROOT/'R5D_ALL_GT_OUTCOMES.csv');seen=set();lookup={}
 for r in gtrows:
  rid=r['run_id'];sid=SPEC[rid]['sample_idx'];gi=int(r['gt_idx']);t=int(round(float(r['iou_threshold'])*100));k=rid,gi,t;ck(k not in seen,'GT_result_primary_key',k);seen.add(k);lookup[k]=r;a=R[rid];iv=a['IoU_GT_raw'][gi];g=GM[sid,gi];tp=bool(a[f'gt_tp{t}'][gi]);ck(r['physical_GT_id']==g['physical_object_id'] and r['GT_hash']==g['GT_hash'] and digest(a['GT_canonical'][gi])==g['GT_hash'],'physical_GT_identity_stable',k);ck(r['TP_FN']==('TP' if tp else 'FN'),'GT_actual_state',k)
  pp=np.flatnonzero(a[f'prediction_gt_idx{t}']==gi)
  if tp:
   pi=int(pp[0]);cid=int(a['ids_final_range'][pi]);ck(int(r['matched_candidate_id'])==cid and int(r['matched_prediction_index'])==pi,'GT_actual_match_ID',k);close(float(r['matched_IoU']),iv[cid],'GT_actual_match_IoU',k);close(float(r['matched_score']),a['scores_all'][cid],'GT_actual_match_score',k)
  else:ck(all(r[n]=='' for n in ['matched_prediction_index','matched_candidate_id','matched_IoU','matched_score']),'FN_match_fields_null_not_zero',k)
  empty=None
  for stage in STAGES:
   expected=a['ids_'+stage][iv[a['ids_'+stage]]>=t/100].astype(int).tolist();ck(arrlist(r['high_ids_'+stage])==expected and int(r['high_count_'+stage])==len(expected),'all_GT_high_IoU_stage_membership',(k,stage))
   if not expected and empty is None:empty=stage
  ck(r['first_empty_stage']==(empty or ''),'first_complete_high_candidate_disappearance',k)
  expected='DETECTED' if tp else 'GT_MATCH_COMPETITION' if arrlist(r['high_ids_final_range']) else {'raw_decoded':'RAW_GEOMETRY','score_pass':'SCORE_FILTER','quality_pass':'LARGE_Z_FILTER','nms_top1000':'TOP1000','nms_keep':'NMS','final_range':'FINAL_RANGE'}[empty];ck(r['bottleneck']==expected,'GT_bottleneck_predicate',k)
 ck(len(gtrows)==sum(len(a['GT_canonical'])*2 for a in R.values()),'all_GT_outcomes_complete')
 focal=rows(ROOT/'R5D_FOCAL_GT_OUTCOMES.csv');ck(len(focal)==68,'focal_GT_rows68')
 for r in focal:
  rid=r['run_id'];k=rid,int(r['gt_idx']),int(round(float(r['iou_threshold'])*100));ck(r==lookup[k] and r['physical_GT_id'] in FOCAL[SPEC[rid]['sample_idx']],'focal_subset_exact',k)
 cr=rows(ROOT/'R5D_CANDIDATE_STAGE_CHANGES.csv');seen=set();selection=collections.defaultdict(set)
 for r in cr:
  rid=r['run_id'];sid=SPEC[rid]['sample_idx'];cid=int(r['candidate_id']);key=rid,cid;ck(key not in seen,'candidate_result_primary_key',key);seen.add(key);selection[rid].add(cid);a=R[rid];sp=np.flatnonzero(a['ids_score_pass']==cid);kp=np.flatnonzero(a['ids_nms_keep']==cid);sup=int(a['suppressor'][cid]);rank=np.flatnonzero(a['ids_quality_pass'][a['scores_all'][a['ids_quality_pass']].argsort()[::-1]]==cid)
  close(float(r['score']),a['scores_all'][cid],'candidate_applied_score',key);ck(r['raw_corner_hash']==digest(a['raw_corners'][cid]),'candidate_applied_geometry_hash',key);actual={GM[sid,gi]['physical_object_id']:float(x) for gi,x in enumerate(a['IoU_GT_raw'][:,cid])};ck(arrlist(r['geometry_IoU_by_physical_GT'])==actual,'candidate_actual_IoU_all_GT',key)
  ck(int(r['query_index'])==cid//2 and int(r['grid_y'])==cid//2//80 and int(r['grid_x'])==cid//2%80 and int(r['rotation_index'])==cid%2,'candidate_anchor_grid_one_to_one',key)
  ck(bol(r['score_pass'])==bool(len(sp)) and bol(r['quality_pass'])==(cid in set(a['ids_quality_pass'])) and bol(r['NMS_keep'])==bool(len(kp)) and bol(r['top1000_pass'])==(cid in set(a['ids_nms_top1000'])),'candidate_native_stage_flags',key);ck(num(r['score_sort_rank_zero_based'])==(int(rank[0]) if len(rank) else None),'candidate_native_sort_rank',key)
  for n,stage,pos in [('large_pass_if_evaluated','large_mask',sp),('z_pass_if_evaluated','z_mask',sp),('range_pass_if_evaluated','range_mask',kp)]:
   if len(pos):ck(bol(r[n])==bool(a[stage][pos[0]]),'candidate_evaluated_geometry_masks',(key,n))
   else:ck(r[n]=='','unevaluated_mask_missing_not_false',(key,n))
  ck(num(r['suppressor_candidate_id'])==(sup if sup>=0 else None),'candidate_suppressor_ID',key)
  if sup>=0:
   ck(sup in set(a['ids_nms_keep']) and cid not in set(a['ids_nms_keep']) and a['scores_all'][sup]>=a['scores_all'][cid] and a['suppressor_overlap'][cid]>.15,'NMS_suppressor_native_relationship',key);close(float(r['suppressor_score']),a['scores_all'][sup],'suppressor_applied_score',key);close(float(r['suppressed_score']),a['scores_all'][cid],'suppressed_applied_score',key);close(float(r['suppression_pair_BEV_IoU']),a['suppressor_overlap'][cid],'suppression_pair_IoU',key)
  else:ck(all(r[n]=='' for n in ['suppressor_score','suppressed_score','suppression_pair_BEV_IoU']),'missing_suppressor_evidence_not_zero',key)
  for t in (50,70):
   state,pi,gi=ci_state(a,cid,t);ck(r[f'final_status{t}']==state and num(r['final_prediction_index'])==pi and num(r[f'matched_gt_idx{t}'])==gi,'candidate_final_matching',(key,t))
 for rid,a in R.items():
  required=set(a['ids_score_pass'])|set(np.flatnonzero(np.any(a['IoU_GT_raw']>=.5,axis=0)))|set(a['ids_final_range']);ck(required<=selection[rid],'candidate_CSV_covers_all_postScore_and_highIoU',rid)
 return len(gtrows),len(focal),len(cr)


def validate_effects():
 rr=rows(ROOT/'R5D_GEOMETRY_SCORE_INTERACTION.csv');ck(len(rr)==8,'factorial_interaction_rows8')
 for r in rr:
  p=r['pair_id'];t=int(round(float(r['iou_threshold'])*100));v={c:CNT[p+'_'+c] for c in COMBOS}
  for m in ['TP','FP','FN','final_prediction_count']:
   n=m+str(t) if m!='final_prediction_count' else m;x={c:v[c][n] for c in COMBOS};expected={'Delta_G_given_S0':x['G1_S0']-x['G0_S0'],'Delta_G_given_S1':x['G1_S1']-x['G0_S1'],'Delta_S_given_G0':x['G0_S1']-x['G0_S0'],'Delta_S_given_G1':x['G1_S1']-x['G1_S0'],'I_GS':x['G1_S1']-x['G1_S0']-x['G0_S1']+x['G0_S0']}
   ck(arrlist(r['four_counts_'+m])==x,'factorial_four_counts',(p,t,m))
   for k,z in expected.items():ck(int(r[k+'_'+m])==z,'factorial_effects_algebra',(p,t,k,m))
   ck(expected['I_GS']==expected['Delta_G_given_S1']-expected['Delta_G_given_S0']==expected['Delta_S_given_G1']-expected['Delta_S_given_G0'],'both_interaction_identities',(p,t,m))
 pg=rows(ROOT/'R5D_PER_GT_FACTORIAL.csv');ck(len(pg)==42,'per_GT_factorial42')
 for r in pg:
  p=r['pair_id'];gi=int(r['gt_idx']);t=int(round(float(r['iou_threshold'])*100));bits=[int(R[p+'_'+c][f'gt_tp{t}'][gi]) for c in COMBOS];ck(r['state_bits']==''.join(map(str,bits)) and int(r['I_GT'])==bits[3]-bits[2]-bits[1]+bits[0],'per_GT_factorial_bits_and_interaction',(p,t,gi))
 for r in rr:ck(sum(int(g['I_GT']) for g in pg if g['pair_id']==r['pair_id'] and g['iou_threshold']==r['iou_threshold'])==int(r['I_GS_TP']),'per_GT_interactions_sum_TP',(r['pair_id'],r['iou_threshold']))


def validate_offtarget_and_swaps():
 off=rows(ROOT/'R5D_OFF_TARGET_EFFECTS.csv');ck(len(off)==54,'offtarget_comparisons54');lookup={}
 for r in off:
  rid=r['run_id'];ref=r['reference_run_id'];t=int(round(float(r['iou_threshold'])*100));key=rid,ref,t;ck(key not in lookup,'offtarget_key_unique',key);lookup[key]=r;a=R[rid];b=R[ref];sid=SPEC[rid]['sample_idx'];ck(sid==SPEC[ref]['sample_idx'],'offtarget_no_frame_mixing',key)
  for m in ['TP','FP','FN']:ck(int(r['delta_'+m])==CNT[rid][m+str(t)]-CNT[ref][m+str(t)],'offtarget_count_difference',(key,m))
  ck(int(r['delta_final_prediction_count'])==len(a['final_scores'])-len(b['final_scores']),'offtarget_final_count_difference',key)
  changes=[];focal=[];other=[];matchchanges=[]
  for gi in range(len(a['GT_canonical'])):
   oid=GM[sid,gi]['physical_object_id'];pa=np.flatnonzero(a[f'prediction_gt_idx{t}']==gi);pb=np.flatnonzero(b[f'prediction_gt_idx{t}']==gi);ca=int(a['ids_final_range'][pa[0]]) if len(pa) else None;cb=int(b['ids_final_range'][pb[0]]) if len(pb) else None
   if bool(len(pa))!=bool(len(pb)):
    x={'physical_GT_id':oid,'gt_idx':gi,'before':'TP' if len(pb) else 'FN','after':'TP' if len(pa) else 'FN','before_match':cb,'after_match':ca};changes.append(x);(focal if oid in FOCAL[sid] else other).append(x)
   elif ca!=cb:matchchanges.append({'physical_GT_id':oid,'before_match':cb,'after_match':ca})
  ck(arrlist(r['all_GT_status_changes'])==changes and arrlist(r['focal_GT_status_changes'])==focal and arrlist(r['other_GT_status_changes'])==other,'offtarget_every_GT_status_change',key);ck(arrlist(r['GT_match_anchor_changes_without_status_change'])==matchchanges,'offtarget_match_anchor_changes',key)
  ka=set(map(int,a['ids_nms_keep']));kb=set(map(int,b['ids_nms_keep']));fa=set(map(int,a['ids_final_range']));fb=set(map(int,b['ids_final_range']));changed=np.flatnonzero(a['suppressor']!=b['suppressor']).astype(int).tolist();edge=[{'candidate_id':cid,'before_suppressor':int(b['suppressor'][cid]) if b['suppressor'][cid]>=0 else None,'after_suppressor':int(a['suppressor'][cid]) if a['suppressor'][cid]>=0 else None} for cid in changed]
  for n,v in [('NMS_keep_added',sorted(ka-kb)),('NMS_keep_removed',sorted(kb-ka)),('final_candidate_added',sorted(fa-fb)),('final_candidate_removed',sorted(fb-fa)),('NMS_suppressor_changed_candidate_ids',changed),('NMS_suppressor_changes',edge)]:ck(arrlist(r[n])==v,'offtarget_all_NMS_final_changes',(key,n))
  fpa={int(cid) for cid,gi in zip(a['ids_final_range'],a[f'prediction_gt_idx{t}']) if gi<0};fpb={int(cid) for cid,gi in zip(b['ids_final_range'],b[f'prediction_gt_idx{t}']) if gi<0};new=[cid for cid in sorted(fpa-fpb) if cid not in fb];gone=[cid for cid in sorted(fpb-fpa) if cid not in fa];tf=[cid for cid in sorted(fpa-fpb) if ci_state(b,cid,t)[0]=='TP'];ft=[cid for cid in sorted(fpb-fpa) if ci_state(a,cid,t)[0]=='TP']
  for n,v in [('FP_anchor_added',sorted(fpa-fpb)),('FP_anchor_removed',sorted(fpb-fpa)),('added_final_FP_ids',new),('removed_final_FP_ids',gone),('TP_to_FP_ids',tf),('FP_to_TP_ids',ft)]:ck(arrlist(r[n])==v,'offtarget_all_FP_events',(key,n))
  ck(len(fpa)-len(fpb)==len(new)+len(tf)-len(gone)-len(ft),'offtarget_FP_partition_identity',key)
 for r in rows(ROOT/'R5D_NMS_SCORE_SWAP.csv'):
  rid=r['run_id'];c=r['condition'];ref=r['reference_run_id'];a=R[rid];b=R[ref];ck(rid=='SWAP_'+c and arrlist(r['changed_score_ids'])==[7028,7030] and int(r['changed_geometry_count'])==0,'swap_table_exact_intervention',rid)
  for cid in [7028,7030]:
   for side,z in [('before',b),('after',a)]:
    close(float(r[f'candidate_{cid}_score_{side}']),z['scores_all'][cid],'swap_table_score',(rid,cid,side));ck(bol(r[f'candidate_{cid}_NMS_keep_{side}'])==(cid in set(z['ids_nms_keep'])),'swap_table_keep',(rid,cid,side));owner=int(z['suppressor'][cid]);ck(num(r[f'candidate_{cid}_suppressor_{side}'])==(owner if owner>=0 else None),'swap_table_suppressor',(rid,cid,side))
  gi=next(gi for (s,gi),g in GM.items() if s==725 and g['physical_object_id']=='4071')
  for t in (50,70):
   ck(r[f'GT4071_before{t}']==('TP' if b[f'gt_tp{t}'][gi] else 'FN') and r[f'GT4071_after{t}']==('TP' if a[f'gt_tp{t}'][gi] else 'FN'),'swap_GT4071_actual_states',(rid,t));pp=np.flatnonzero(a[f'prediction_gt_idx{t}']==gi)
   if len(pp):cid=int(a['ids_final_range'][pp[0]]);ck(int(r[f'GT4071_match_after{t}'])==cid,'swap_match_ID',(rid,t));close(float(r[f'GT4071_match_IoU_after{t}']),a['IoU_GT_raw'][gi,cid],'swap_match_IoU',(rid,t));close(float(r[f'GT4071_match_score_after{t}']),a['scores_all'][cid],'swap_match_Score',(rid,t))
   else:ck(all(r[n+str(t)]=='' for n in ['GT4071_match_after','GT4071_match_IoU_after','GT4071_match_score_after']),'swap_FN_fields_missing',(rid,t))
   ck(r[f'all_GT_status_changes{t}']==lookup[rid,ref,t]['all_GT_status_changes'] and r[f'other_GT_status_changes{t}']==lookup[rid,ref,t]['other_GT_status_changes'],'swap_complete_offtarget_GT_record',(rid,t))
  ck(r['NMS_suppressor_changes']==lookup[rid,ref,50]['NMS_suppressor_changes'],'swap_all_NMS_changes_saved',rid)
 return len(off)


def main():
 load_and_replay();ng,nf,nc=validate_GT_and_candidates();validate_effects();no=validate_offtarget_and_swaps();manifest=rows(ROOT/'R5D_PAIR_MANIFEST.csv');ck(len(manifest)==4,'pair_manifest4')
 for p,r in zip(PAIRS,manifest):
  sid=p['sample_idx'];ck(r['pair_id']==p['pair_id'] and int(r['sample_idx'])==sid and r['condition0']==p['condition0'] and r['condition1']==p['condition1'],'pair_manifest_fixed_design')
  for i in [0,1]:ck(r[f'source{i}_path']==LOC[sid,p[f'condition{i}']]['absolute_path'] and r[f'source{i}_sha256']==LOC[sid,p[f'condition{i}']]['sha256'],'pair_manifest_source_identity')
 hist=load(ROOT/'R5D_HISTORY_READ_ONLY_SEAL.json')['files']
 for p,h in hist.items():ck(Path(p).is_file() and sha(p)==h,'historical_file_unchanged',p)
 ck(not DENIED and not any(n in sys.modules for n in ['torch','opencood','cupy','pycuda','pynvml']),'CPU_only_no_device_import')
 state='R5D_FAIL' if FAIL else 'R5D_PASS';v={'state':state,'validator':'separate process; no producer import; original full NPZ input; independent active-mask NMS and unique-GT loop; shared only SHA-verified BEV polygon primitives and identity loaders','passed_check_count':sum(CHECK.values())-len(FAIL),'failed_check_count':len(FAIL),'check_count':sum(CHECK.values()),'check_groups':dict(CHECK),'FAIL':FAIL,'UNRESOLVED':[],'unverifiable_required_items':0,'run_count':19,'factorial_runs':16,'specified_score_swap_runs':3,'original_baselines_exact':8,'unique_original_full_NPZ':6,'all32_source_NPZ_SHA_checked':True,'all_GT_outcome_rows':ng,'focal_GT_outcome_rows':nf,'candidate_CSV_rows':nc,'offtarget_comparison_rows':no,'historical_files_verified_unchanged':len(hist),'source_sha256':{n:sha(ROOT/n) for n in ['cpu_common.py','analyze_r5d.py','validate_r5d_independent.py','source_reference/offline_science.py','source_reference/offline_storage_reader.py']},'new_model_forward':0,'GPU_PPU_calls':0,'CUDA_initializations':0,'historical_modifications':0,'model_modules_imported':False,'inference_budget_attempt_increment':0,'causal_scope':'offline replacement sensitivity in saved candidate space only; possible off-natural joint distributions; no attention/MetaFlow/AoI/network causal claim','statistical_significance_test':False}
 write('R5D_INDEPENDENT_VALIDATION.json',v);print(state,'PASS_CHECKS',v['passed_check_count'],'FAIL',len(FAIL),'UNVERIFIABLE',0,flush=True)
 if FAIL:sys.exit(1)
if __name__=='__main__':
 try:main()
 except Exception as e:write('R5D_INDEPENDENT_EXCEPTION.json',{'state':'R5D_FAIL','exception':str(e),'traceback':traceback.format_exc(),'FAIL':FAIL,'new_model_forward':0,'CUDA_initializations':0});raise
