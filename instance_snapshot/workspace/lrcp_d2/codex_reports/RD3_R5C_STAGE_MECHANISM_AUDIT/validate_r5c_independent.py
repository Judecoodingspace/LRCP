"""Independent R5C CPU validator. Does not import analyze_r5c or trust its passed flags.
Reopens immutable original NPZs; independently implements stage set construction,
active-mask NMS and score-ordered unique GT matching. Shared float32 BEV polygon
arithmetic is the SHA-verified R5A primitive, not the producer's stage procedure.
"""
from stage_common import ROOT,A,B,REPORTS,sha,load,csvrows,write,table,np,json,sys,Path,DENIED,polys,overlap,matrix,digest,STAGES,CORE,CONTRASTS,STATE_ORDER,CONDITIONS,INTERACTION_SAMPLES
import traceback,collections
CHECKS=collections.Counter();FAILED=[];R={};M={};I={};C={};TR={};RAW_HASH={};GT={};EVENTS=[]
def check(ok,group,detail=''):
 CHECKS[group]+=1
 if not bool(ok):
  FAILED.append({'group':group,'detail':str(detail)})
  if len(FAILED)<20: print('INDEPENDENT_CHECK_FAILED',group,detail,flush=True)
def eq(x,y,group,detail=''):
 a=np.asarray(x);b=np.asarray(y);check(a.shape==b.shape and np.array_equal(a,b),group,detail)
def close(x,y,group,detail='',tol=1e-6):
 a=np.asarray(x);b=np.asarray(y);check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.abs(a.astype(float)-b.astype(float))<=tol),group,detail)
def boolean(v):return str(v)=='True'
def number(v):return None if v=='' else float(v)
def ids(v):return json.loads(v)
def native_independent(a,config,key):
 raw=a['raw_corners'];scores=a['scores_all'];check(raw.shape==(12800,8,3),'all_raw_candidates',key)
 sc=np.flatnonzero(scores>config['target_args']['score_threshold']);boxes=raw[sc];span=boxes.max(axis=1)-boxes.min(axis=1)
 # Preserve native historical y-span-as-z boolean quirk, without repairing it.
 lm=(span[:,0]<=6)&(span[:,1]<=6)&(span[:,1]!=0);zm=(boxes[:,:,2].min(axis=1)>=-3)&(boxes[:,:,2].max(axis=1)<=1);q=sc[lm&zm];top=q[scores[q].argsort()[::-1][:1000]]
 eq(a['score_mask'],scores>config['target_args']['score_threshold'],'native_score_mask',key)
 for n,v in [('large_mask',lm),('z_mask',zm),('quality_mask',lm&zm)]:eq(a[n],v,'native_size_z_quality',key)
 # An independent active-mask algorithm, distinct from producer np.delete loop.
 pp=polys(raw[top]);active=np.ones(len(top),bool);keep=[];sup=np.full(12800,-1,np.int32);ov=np.zeros(12800,np.float32)
 for rank,cid0 in enumerate(top):
  if not active[rank]:continue
  cid=int(cid0);keep.append(cid);active[rank]=False;other=np.flatnonzero(active);v=overlap(pp[rank],[pp[j] for j in other]);removed=other[v>config['nms_thresh']]
  active[removed]=False
  for j in removed:
   sup[int(top[j])]=cid;ov[int(top[j])]=v[np.flatnonzero(other==j)[0]]
 keep=np.asarray(keep,np.int32);lim=np.asarray(config['gt_range']);rr=raw[keep];rg=np.sum(np.all((rr>=lim[:3])&(rr<=lim[3:]),axis=2),axis=1)>=8;final=keep[rg]
 stages=[np.arange(12800),sc,q,top,keep,final]
 for s,v in zip(STAGES,stages):eq(a['ids_'+s],v,'independent_stage_IDs',(key,s))
 eq(a['native_nms_keep'],np.asarray([int(np.flatnonzero(q==cid)[0]) for cid in keep]),'independent_native_keep_indices',key)
 eq(a['suppressor'],sup,'independent_all_native_suppressors',key);eq(a['suppressor_overlap'],ov,'independent_all_suppression_IoUs',key);eq(a['range_mask'],rg,'independent_final_range',key)
 eq(a['final_corners'],raw[final],'exact_final_corners',key);eq(a['final_scores'],scores[final],'exact_final_scores',key)
 eq(a['nms_input_corners'],raw[q],'exact_NMS_input',key);eq(a['nms_input_scores'],scores[q],'exact_NMS_scores',key)
 code=np.ones(12800,np.int8)
 for n,v in enumerate(stages[1:],2):code[v]=n
 eq(code,a['first_deletion_code'],'independent_first_deletion',key)
 for cid in np.flatnonzero(sup>=0):
  owner=int(sup[cid]);check(owner in set(keep) and cid not in set(keep) and scores[owner]>=scores[cid] and ov[cid]>config['nms_thresh'],'NMS_edge_native_predicates',(key,int(cid),owner))
 return stages

def independent_match(a,io,threshold,key):
 gtcount=len(a['GT_canonical']);remaining=set(range(gtcount));p2g=np.full(len(a['final_scores']),-1,np.int32);flags=np.zeros(gtcount,np.int8);order=np.argsort(-a['final_scores']);rows=[];claimed={}
 for rank,pi0 in enumerate(order):
  pi=int(pi0);cid=int(a['ids_final_range'][pi]);available=sorted(remaining);v=io[:,cid];hit=None
  if available:
   j=int(np.argmax(v[available]));candidate=available[j]
   if float(v[candidate])>=threshold:hit=candidate;remaining.remove(hit);p2g[pi]=hit;flags[hit]=1;claimed[hit]=cid
  eligible=np.flatnonzero(v>=threshold).astype(int).tolist()
  rows.append({'rank':rank,'pi':pi,'cid':cid,'eligible':eligible,'available':available,'hit':hit,'reason':'MATCHED_UNIQUE_GT' if hit is not None else 'GT_ALREADY_CLAIMED' if eligible else 'BELOW_ALL_GT_IOU_THRESHOLD'})
 t=int(round(threshold*100));eq(p2g,a[f'prediction_gt_idx{t}'],'independent_prediction_GT_match',(key,t));eq(flags,a[f'gt_tp{t}'],'independent_GT_flags',(key,t));eq(p2g<0,a[f'prediction_fp{t}'],'independent_FP_flags',(key,t))
 return {'p2g':p2g,'flags':flags,'order_rows':rows}
def cand(key,cid,t):
 a=R[key];p=np.flatnonzero(a['ids_final_range']==cid)
 if not len(p):return ('ABSENT_FINAL',None,None)
 pi=int(p[0]);gi=int(M[key,t]['p2g'][pi]);return ('TP' if gi>=0 else 'FP',pi,gi if gi>=0 else None)
def initialize():
 for p,h in load(ROOT/'R5C_INITIAL_SOURCE_VALIDATION.json')['source_files'].items():check(sha(p)==h,'initial_source_SHA',p)
 for folder,names in [(A,['R5A_GT_IDENTITY_MAP.csv','R5A_SOURCE_INVENTORY.csv']), (B,['R5B_SAMPLE_CONDITION_COUNTS.csv','R5B_GT_TRANSITION_AUDIT.csv','R5B_GT_FOUR_CONDITION_PATTERNS.csv','R5B_RAW_SHARD_LOCATOR.csv','R5B_SAMPLE_AGENT_UTILITY.csv'])]:
  sums={x.split('  ',1)[1]:x.split('  ',1)[0] for x in (folder/'SHA256SUMS.txt').read_text().splitlines()}
  for name in names:check(sha(folder/name)==sums[name],'prior_sealed_CSV_SHA',name)
 for x in csvrows(A/'R5A_GT_IDENTITY_MAP.csv'):GT[int(x['sample_idx']),int(x['gt_idx'])]=x
 config=load(ROOT/'R5C_C0_REPLAY_VALIDATION.json')['frozen_postprocess_config']
 # Config itself is separately hash bound to R5A inventory and C0 seal.
 import yaml
 inv=csvrows(A/'R5A_SOURCE_INVENTORY.csv');cp=Path(load(ROOT/'R5C_C0_REPLAY_VALIDATION.json')['source_config_path']);check(any(x['absolute_path']==str(cp) and x['sha256']==sha(cp) for x in inv),'actual_frozen_config_SHA')
 tree=yaml.compose(cp.read_text());post=next(v for k,v in tree.value if k.value=='postprocess');actual=yaml.safe_load(yaml.serialize(post));check(actual==config,'actual_config_vs_C0_not_trust_flag')
 baselines={(int(x['sample_idx']),x['condition']):x for x in csvrows(B/'R5B_SAMPLE_CONDITION_COUNTS.csv')};baselinesA={(int(x['sample_idx']),x['condition']):x for x in csvrows(A/'R5A_SAMPLE_CONDITION_COUNTS.csv')}
 keepfields=['raw_corners','scores_all','GT_canonical','IoU_GT_raw','final_corners','final_scores','suppressor','suppressor_overlap','score_mask','large_mask','z_mask','quality_mask','native_nms_keep','nms_input_corners','nms_input_scores','range_mask','first_deletion_code','cls_preds','reg_preds','dir_preds','decoded_before_dir']+['ids_'+s for s in STAGES]+[p+str(t) for p in ('gt_tp','prediction_gt_idx','prediction_fp') for t in (50,70)]
 for loc in csvrows(B/'R5B_RAW_SHARD_LOCATOR.csv'):
  key=int(loc['sample_idx']),loc['condition'];check(key not in R,'original_record_key_uniqueness',key);check(sha(loc['absolute_path'])==loc['sha256'],'original_NPZ_SHA',key);RAW_HASH[key]=loc
  with np.load(loc['absolute_path'],allow_pickle=False) as z:a={k:z[k].copy() for k in keepfields};check('iou_preds' not in z.files,'actual_no_learned_quality_branch',key)
  native_independent(a,config,key);R[key]=a;io=matrix(a['GT_canonical'],a['raw_corners']);close(io,a['IoU_GT_raw'],'independent_float32_BEV_GT_geometry',key);I[key]=io
  sig=(1/(1+np.exp(-a['cls_preds'].transpose(0,2,3,1).reshape(-1)))).astype(np.float32);close(sig,a['scores_all'],'actual_head_sigmoid_score',key)
  for gi,g in enumerate(a['GT_canonical']):check(digest(g)==GT[key[0],gi]['GT_hash'],'stable_physical_GT_geometry',(key,gi))
  counts={'final_prediction_count':len(a['final_scores']),'GT_count':len(a['GT_canonical'])}
  for t in (50,70):
   m=independent_match(a,io,t/100,key);M[key,t]=m;tp=int(m['flags'].sum());counts.update({f'TP{t}':tp,f'FP{t}':len(m['p2g'])-tp,f'FN{t}':len(m['flags'])-tp})
  for n,v in counts.items():
   check(v==int(baselines[key][n]),'all32_counts_vs_R5B',(key,n));check(v==int(baselinesA[key][{'final_prediction_count':'final_predictions'}.get(n,n)]),'all32_counts_vs_R5A',(key,n))
  C[key]=counts;print('INDEPENDENT_RAW_REPLAY',*key,counts['TP70'],counts['FP70'],flush=True)
 check(len(R)==32,'record_count32')

def validate_GT():
 traces=csvrows(ROOT/'R5C_GT_STAGE_TRACE.csv');check(len(traces)==448,'GT_trace448');seen=set()
 for row in traces:
  sid=int(row['sample_idx']);c=row['condition'];gi=int(row['gt_idx']);t=int(round(float(row['iou_threshold'])*100));k=sid,c,gi,t;check(k not in seen,'GT_trace_unique_primary_key',k);seen.add(k);TR[k]=row;a=R[sid,c];iv=I[sid,c][gi];high=iv>=t/100
  g=GT[sid,gi];check(row['physical_GT_id']==g['physical_object_id'] and row['GT_hash']==g['GT_hash'] and row['scene_id']==g['scene_id'] and row['scene_frame']==g['scene_frame'],'GT_identity_fields',k)
  eq(json.loads(row['canonical_GT_corners']),a['GT_canonical'][gi],'GT_saved_corners',k)
  counts=[];highsets={}
  for s in STAGES:
   members=a['ids_'+s][high[a['ids_'+s]]].astype(int).tolist();highsets[s]=members;counts.append(len(members));check(int(row['high_IoU_count_'+s])==len(members) and ids(row['high_IoU_ids_'+s])==members,'GT_exact_high_IoU_membership',(k,s))
  first=next((s for s,n in zip(STAGES,counts) if n==0),'');check(row['first_stage_all_high_IoU_disappear']==first,'GT_first_complete_disappearance',k)
  p=np.flatnonzero(M[(sid,c),t]['p2g']==gi);tp=len(p)>0;check(row['TP_FN']==('TP' if tp else 'FN'),'GT_final_status',k)
  if tp:
   pi=int(p[0]);cid=int(a['ids_final_range'][pi]);check(int(row['matched_prediction_index'])==pi and int(row['matched_candidate_id'])==cid,'GT_exact_matched_candidate',k);close(float(row['matched_IoU']),iv[cid],'GT_actual_matched_IoU',k);close(float(row['matched_score']),a['scores_all'][cid],'GT_actual_matched_score',k)
  else:check(all(row[n]=='' for n in ['matched_prediction_index','matched_candidate_id','matched_IoU','matched_score']),'FN_match_fields_missing_never_zero',k)
  rawmax=int(np.argmax(iv));scoremax=int(np.argmax(a['scores_all']));check(int(row['raw_max_IoU_candidate_id'])==rawmax and int(row['raw_highest_score_candidate_id'])==scoremax,'GT_raw_extremal_IDs',k)
  for n,v in [('raw_max_IoU',iv[rawmax]),('raw_max_IoU_candidate_score',a['scores_all'][rawmax]),('raw_highest_score',a['scores_all'][scoremax]),('raw_highest_score_candidate_IoU',iv[scoremax])]:close(float(row[n]),v,'GT_raw_extremal_values',(k,n))
  highids=np.flatnonzero(high);highest=int(highids[np.argmax(a['scores_all'][highids])]) if len(highids) else None
  check(number(row['raw_high_IoU_highest_score_candidate_id'])==highest,'GT_high_IoU_score_ID_nullable',k)
  if highest is None:check(row['raw_high_IoU_highest_score']=='','GT_missing_high_candidate_score')
  else:close(float(row['raw_high_IoU_highest_score']),a['scores_all'][highest],'GT_high_IoU_score',k)
  fin=a['ids_final_range'];best=int(fin[np.argmax(iv[fin])]) if len(fin) else None;check(number(row['final_max_IoU_candidate_id'])==best,'GT_final_best_separate_from_match',k)
  if best is not None:
   close(float(row['final_max_IoU']),iv[best],'GT_final_max_IoU',k);close(float(row['final_max_IoU_candidate_score']),a['scores_all'][best],'GT_final_max_score',k)
  suppressed=[i for i in highsets['nms_top1000'] if a['suppressor'][i]>=0];check(ids(row['high_IoU_suppressed_ids'])==suppressed and boolean(row['some_high_IoU_candidate_NMS_suppressed'])==bool(suppressed),'GT_some_high_NMS_suppressed',k)
  all_lost=counts[3]>0 and counts[4]==0;check(boolean(row['all_preNMS_high_IoU_candidates_eliminated_by_NMS'])==all_lost,'GT_some_vs_all_NMS',k)
  expected='DETECTED' if tp else 'GT_MATCH_COMPETITION' if counts[-1]>0 else dict(zip(STAGES,['RAW_GEOMETRY','SCORE_FILTER','LARGE_OR_Z_FILTER','TOP1000','NMS','FINAL_RANGE']))[first]
  check(row['observed_final_bottleneck']==expected,'GT_observed_final_bottleneck',k)
  check(row['source_path']==RAW_HASH[sid,c]['absolute_path'] and row['source_sha256']==RAW_HASH[sid,c]['sha256'],'GT_raw_source_pointer',k)
 for b in csvrows(B/'R5B_GT_TRANSITION_AUDIT.csv'):
  sid=int(b['sample_idx']);t=int(round(float(b['iou_threshold'])*100));gi=int(b['gt_idx']);be=b['condition_before'];af=b['condition_after'];db=int(M[(sid,be),t]['flags'][gi]);da=int(M[(sid,af),t]['flags'][gi]);check(db==int(b['D_before']) and da==int(b['D_after']) and da-db==int(b['delta_GT_TP']),'all448_GT_transitions_vs_B',(sid,t,gi))
 for sid,oid in CORE:
  gi=next(gi for (s,gi),g in GT.items() if s==sid and g['physical_object_id']==oid)
  patterns={t:''.join(str(int(M[(sid,c),t]['flags'][gi])) for c in STATE_ORDER) for t in (50,70)};expected={(725,'14014'):{50:'1111',70:'1000'},(725,'4071'):{50:'1000',70:'0000'},(324,'29118'):{50:'0101',70:'0001'}}[sid,oid];check(patterns==expected,'all_core_patterns_independent',(sid,oid))


def validate_order_rows():
 rows=csvrows(ROOT/'R5C_GT_MATCH_ORDER.csv');lookup={};expectedcount=0
 for key,a in R.items():
  for t in (50,70):
   for x in M[key,t]['order_rows']:lookup[key[0],key[1],t,x['pi']]=x;expectedcount+=1
 check(len(rows)==expectedcount,'all_final_prediction_threshold_matching_events')
 seen=set()
 for r in rows:
  sid=int(r['sample_idx']);c=r['condition'];t=int(round(float(r['iou_threshold'])*100));pi=int(r['final_prediction_index']);k=sid,c,t,pi;check(k not in seen,'match_order_key_unique',k);seen.add(k);x=lookup[k];a=R[sid,c];v=I[sid,c][:,x['cid']]
  check(int(r['candidate_id'])==x['cid'] and int(r['score_order_rank_zero_based'])==x['rank'] and ids(r['remaining_GT_before'])==x['available'] and ids(r['eligible_all_GT_indices'])==x['eligible'],'exact_native_match_order_and_eligible_GT',k)
  check(ids(r['eligible_remaining_GT_indices'])==[g for g in x['available'] if float(v[g])>=t/100] and r['reason']==x['reason'],'match_competition_reason',k)
  check(number(r['matched_gt_idx'])==x['hit'] and r['result']==('TP' if x['hit'] is not None else 'FP'),'match_order_result',k)
  if x['hit'] is None:check(r['matched_IoU']=='' and r['matched_physical_GT_id']=='','match_order_missing_fields_not_zero',k)
  else:close(float(r['matched_IoU']),v[x['hit']],'match_order_actual_IoU',k);check(r['matched_physical_GT_id']==GT[sid,x['hit']]['physical_object_id'],'match_order_physical_GT',k)
  claimed={}
  for earlier in M[(sid,c),t]['order_rows']:
   if earlier['rank']>=x['rank']:break
   if earlier['hit'] is not None and earlier['hit'] in x['eligible']:claimed[str(earlier['hit'])]=earlier['cid']
  check(ids(r['already_claimed_eligible_GT'])==claimed,'match_order_exact_prior_claims',k)


def validate_labels():
 rows=csvrows(ROOT/'R5C_GT_TRANSITION_MECHANISMS.csv');seen=set()
 for r in rows:
  sid=int(r['sample_idx']);gi=int(r['gt_idx']);t=int(round(float(r['iou_threshold'])*100));be=r['condition_before'];af=r['condition_after'];k=sid,r['contrast'],gi,t;check(k not in seen,'mechanism_key_uniqueness',k);seen.add(k);bb=R[sid,be];aa=R[sid,af];bi=I[sid,be][gi];ai=I[sid,af][gi];b=TR[sid,be,gi,t];a=TR[sid,af,gi,t];expected=[]
  if bool(np.any(bi>=t/100))!=bool(np.any(ai>=t/100)):expected.append('RAW_GEOMETRY_DIFFERENCE')
  mapping={'SCORE_FILTER':'SCORE_OR_QUALITY_FILTERING','LARGE_OR_Z_FILTER':'SCORE_OR_QUALITY_FILTERING','TOP1000':'TOPK_SELECTION','NMS':'NMS_SUPPRESSION','FINAL_RANGE':'FINAL_RANGE_OR_SIZE_FILTER','GT_MATCH_COMPETITION':'GT_MATCH_COMPETITION'}
  for tr,raw,iv in [(b,bb,bi),(a,aa,ai)]:
   if tr['TP_FN']=='TP':continue
   label=mapping.get(tr['observed_final_bottleneck'])
   if label and label not in expected:expected.append(label)
   if label=='NMS_SUPPRESSION':
    pres=raw['ids_nms_top1000'][iv[raw['ids_nms_top1000']]>=t/100];post=raw['ids_nms_keep'][iv[raw['ids_nms_keep']]>=t/100];check(len(pres)>0 and len(post)==0 and np.all(raw['suppressor'][pres]>=0),'NMS_label_all_available_high_lost',k)
  highunion=np.flatnonzero((bi>=t/100)|(ai>=t/100));scorechange=[int(i) for i in highunion if bool(bb['score_mask'][i])!=bool(aa['score_mask'][i])];nb=set(bb['ids_nms_top1000']);na=set(aa['ids_nms_top1000']);kb=set(bb['ids_nms_keep']);ka=set(aa['ids_nms_keep']);nmschange=[int(i) for i in highunion if i in nb and i in na and (i in kb)!=(i in ka)];cross=[]
  for tr in (b,a):
   if tr['matched_candidate_id']!='':
    cid=int(tr['matched_candidate_id'])
    if (bi[cid]>=t/100)!=(ai[cid]>=t/100):cross.append({'candidate_id':cid,'before_GT_IoU':float(bi[cid]),'after_GT_IoU':float(ai[cid]),'threshold':t/100})
  material=[bool(cross),bool(scorechange),bool(nmschange),bool(np.any(bi>=t/100))!=bool(np.any(ai>=t/100))]
  if sum(material)>=2:expected.append('MULTIPLE_STAGE_CHANGES')
  changed=(b['TP_FN']!=a['TP_FN'])
  if changed and not expected:expected=['UNRESOLVED']
  if not expected:expected=['NO_GT_DETECTION_STATE_CHANGE']
  check(ids(r['stage_labels'])==expected,'every_label_has_exact_stage_predicate',k);check(boolean(r['UNRESOLVED'])==('UNRESOLVED' in expected),'UNRESOLVED_truthful',k)
  proof=ids(r['supported_stage_predicates']);check(set(proof)==set(expected),'label_support_keys',k)
  for item in proof.get('SCORE_OR_QUALITY_FILTERING',[]):
   key=sid,be if item['side']=='before' else af;raw=R[key];iv=I[key][gi]
   for stage,field in [('raw_decoded','raw_high_ids'),('score_pass','score_high_ids'),('quality_pass','quality_high_ids')]:
    actual=raw['ids_'+stage][iv[raw['ids_'+stage]]>=t/100].astype(int).tolist();check(item[field]==actual,'label_score_raw_ID_evidence',(k,field))
   check(item['score_threshold']==.2,'label_actual_score_threshold',k)
  if 'RAW_GEOMETRY_DIFFERENCE' in proof:
   item=proof['RAW_GEOMETRY_DIFFERENCE'];check(item['before_count']==int(np.count_nonzero(bi>=t/100)) and item['after_count']==int(np.count_nonzero(ai>=t/100)),'label_raw_existence_counts',k)
  if 'MULTIPLE_STAGE_CHANGES' in proof:
   item=proof['MULTIPLE_STAGE_CHANGES'];check(item['matched_anchor_geometry_threshold_crossing']==cross and item['score_pass_changed_on_high_IoU_union']==scorechange and item['native_NMS_keep_changed_on_high_IoU_union']==nmschange and item['raw_high_IoU_existence_changed']==material[-1],'label_multiple_actual_changes',k)
  for item in proof.get('NMS_SUPPRESSION',[]):
   key=sid,be if item['side']=='before' else af;raw=R[key];iv=I[key][gi]
   for e in item['edges']:
    cid=e['suppressed_id'];owner=e['suppressor_id'];check(int(raw['suppressor'][cid])==owner and raw['suppressor_overlap'][cid]>.15,'label_suppressor_edge_actual',(k,cid));close(e['BEV_IoU'],raw['suppressor_overlap'][cid],'label_suppressor_edge_IoU');close(e['suppressed_score'],raw['scores_all'][cid],'label_suppressed_score');close(e['suppressor_score'],raw['scores_all'][owner],'label_suppressor_score');check(float(iv[cid])>=t/100,'label_suppressed_GT_high')
  check(boolean(r['raw_candidate_geometry_changed'])==(not np.array_equal(bb['raw_corners'],aa['raw_corners'])),'raw_geometry_not_hidden',k);check(r['source_sha_before']==RAW_HASH[sid,be]['sha256'] and r['source_sha_after']==RAW_HASH[sid,af]['sha256'],'mechanism_source_SHA',k)
 check(all(any(int(r['sample_idx'])==int(b['sample_idx']) and r['contrast']==b['contrast'] and r['gt_idx']==b['gt_idx'] and float(r['iou_threshold'])==float(b['iou_threshold']) for r in rows) for b in csvrows(B/'R5B_GT_TRANSITION_AUDIT.csv') if b['transition_state'] in ('TP_TO_FN','FN_TO_TP')),'all_changed_GT_events_covered')
 return [r for r in rows if boolean(r['UNRESOLVED'])]


def validate_lineage_exports():
 rows=csvrows(ROOT/'R5C_CANDIDATE_LINEAGE.csv');seen=set();exportkeys=set();selection={};lookup={}
 for r in rows:
  sid=int(r['sample_idx']);c=r['condition'];cid=int(r['candidate_id']);k=sid,c,cid;check(k not in seen,'candidate_unique_primary_key',k);seen.add(k);lookup[k]=r;selection.setdefault((sid,c),set()).add(cid);a=R[sid,c];sp=np.flatnonzero(a['ids_score_pass']==cid);kp=np.flatnonzero(a['ids_nms_keep']==cid);pi=np.flatnonzero(a['ids_final_range']==cid);sup=int(a['suppressor'][cid]);score=a['scores_all'][cid]
  check(json.loads(r['lineage_key'])==[sid,c,cid] and int(r['head_bev_query_index'])==cid//2 and int(r['grid_y'])==(cid//2)//80 and int(r['grid_x'])==(cid//2)%80 and int(r['anchor_rotation_index'])==cid%2,'lineage_anchor_query_grid',k)
  close(float(r['score']),score,'lineage_raw_score',k);eq(ids(r['raw_corners']),a['raw_corners'][cid],'lineage_raw_corners',k)
  for n,w in [('cls_logit',a['cls_preds'].transpose(0,2,3,1).reshape(-1)[cid])]:close(float(r[n]),w,'lineage_cls_head',k)
  for n,w in [('reg_delta',a['reg_preds'].transpose(0,2,3,1).reshape(-1,7)[cid]),('dir_logits',a['dir_preds'].transpose(0,2,3,1).reshape(-1,2)[cid]),('decoded_before_direction',a['decoded_before_dir'].reshape(-1,7)[cid])]:eq(ids(r[n]),w,'lineage_reg_dir_head',(k,n))
  check(boolean(r['score_pass'])==bool(len(sp)) and boolean(r['quality_pass'])==(cid in set(a['ids_quality_pass'])) and boolean(r['top1000_pass'])==(cid in set(a['ids_nms_top1000'])) and boolean(r['native_NMS_keep'])==bool(len(kp)) and boolean(r['native_NMS_suppressed'])==(sup>=0),'lineage_all_stage_flags',k)
  if len(sp):check(boolean(r['large_mask_if_evaluated'])==bool(a['large_mask'][sp[0]]) and boolean(r['z_mask_if_evaluated'])==bool(a['z_mask'][sp[0]]),'lineage_size_z_when_evaluated',k)
  else:check(r['large_mask_if_evaluated']=='' and r['z_mask_if_evaluated']=='','lineage_unevaluated_masks_not_zero',k)
  if len(kp):check(boolean(r['range_pass_if_evaluated'])==bool(a['range_mask'][kp[0]]),'lineage_range_when_evaluated',k)
  else:check(r['range_pass_if_evaluated']=='','lineage_unevaluated_range_missing',k)
  if sup>=0:
   check(int(r['native_suppressor_candidate_id'])==sup,'lineage_suppressor_ID',k);close(float(r['native_suppressor_score']),a['scores_all'][sup],'lineage_suppressor_score');close(float(r['native_suppression_BEV_IoU']),a['suppressor_overlap'][cid],'lineage_suppression_IoU')
  else:check(all(r[n]=='' for n in ['native_suppressor_candidate_id','native_suppressor_score','native_suppression_BEV_IoU']),'lineage_missing_suppressor_not_zero',k)
  for t in (50,70):
   status,ppi,gi=cand((sid,c),cid,t);check(r[f'final_status{t}']==status and number(r['final_prediction_index'])==ppi,'lineage_final_state',(k,t));check(number(r[f'matched_gt_idx{t}'])==gi,'lineage_nullable_GT_id',(k,t));check(r[f'matched_physical_GT{t}']==(GT[sid,gi]['physical_object_id'] if gi is not None else ''),'lineage_physical_GT',(k,t))
  check(int(r['first_deletion_code'])==int(a['first_deletion_code'][cid]),'lineage_first_deletion',k);check(r['source_sha256']==RAW_HASH[sid,c]['sha256'] and r['source_path']==RAW_HASH[sid,c]['absolute_path'],'lineage_raw_source_SHA',k)
 for key,a in R.items():
  sid,c=key;selected=selection[key];required=set(a['ids_score_pass'])|set(np.flatnonzero(np.any(I[key]>=.5,axis=0)))|set(np.argmax(I[key],axis=1))|{int(np.argmax(a['scores_all']))}|set(a['ids_final_range']);check(required<=selected,'export_covers_all_scorepass_all_highIoU_and_extrema',key)
  check(all(int(a['suppressor'][cid]) in selected for cid in selected if a['suppressor'][cid]>=0),'export_suppressor_closure',key)
  with np.load(ROOT/'evidence'/f'sample_{sid}_{c}_selected_stage_evidence.npz',allow_pickle=False) as e:
   ci=e['candidate_ids'];check(set(ci)==selected,'compact_evidence_selection',key)
   pairs=[('raw_corners',a['raw_corners'][ci]),('scores_all_selected',a['scores_all'][ci]),('scores_all_full',a['scores_all']),('IoU_GT_full',I[key]),('IoU_GT_selected',I[key][:,ci]),('GT_canonical',a['GT_canonical']),('suppressor_selected',a['suppressor'][ci]),('suppressor_overlap_selected',a['suppressor_overlap'][ci]),('first_deletion_code_selected',a['first_deletion_code'][ci]),('final_corners',a['final_corners']),('final_scores',a['final_scores']),('cls_logits_selected',a['cls_preds'].transpose(0,2,3,1).reshape(-1)[ci]),('reg_delta_selected',a['reg_preds'].transpose(0,2,3,1).reshape(-1,7)[ci]),('dir_logits_selected',a['dir_preds'].transpose(0,2,3,1).reshape(-1,2)[ci]),('decoded_before_dir_selected',a['decoded_before_dir'].reshape(-1,7)[ci])]+[('ids_'+s,a['ids_'+s]) for s in STAGES]
   for n,v in pairs:eq(e[n],v,'compact_evidence_original_array_exact',(key,n))
  exportkeys.add(key)
 check(exportkeys==set(R),'compact_evidence_all32')
 edges=csvrows(ROOT/'R5C_NMS_SUPPRESSOR_EDGES.csv');edgekeys=set()
 for e in edges:
  key=int(e['sample_idx']),e['condition'];cid=int(e['suppressed_id']);owner=int(e['suppressor_id']);a=R[key];ek=key,cid;check(ek not in edgekeys,'NMS_edge_primary_key',ek);edgekeys.add(ek);check(owner==int(a['suppressor'][cid]) and owner in set(a['ids_nms_keep']) and cid not in set(a['ids_nms_keep']),'NMS_export_keep_suppressor',ek)
  close(float(e['pair_BEV_IoU']),a['suppressor_overlap'][cid],'NMS_export_exact_IoU',ek);close(float(e['suppressed_score']),a['scores_all'][cid],'NMS_export_suppressed_score',ek);close(float(e['suppressor_score']),a['scores_all'][owner],'NMS_export_suppressor_score',ek)
 check(len(edges)==sum(int((a['suppressor']>=0).sum()) for a in R.values()),'all_NMS_edges_exported')
 return len(rows),len(edges)

def validate_FP():
 summaries=csvrows(ROOT/'R5C_FP_MARGINAL_ACCOUNTING.csv');rows=csvrows(ROOT/'R5C_FP_PROVENANCE.csv');groups=collections.defaultdict(list);seen=set();expected_harm={(324,'U_B_given_A'),(324,'U_C_given_A'),(354,'U_C_given_B'),(354,'U_C_given_A'),(384,'U_B_given_C'),(384,'U_B_given_A'),(725,'U_B_given_C'),(725,'U_B_given_A'),(725,'U_C_given_A')}
 check(len(summaries)==13 and {(int(r['sample_idx']),r['contrast']) for r in summaries if r['analysis_role']=='MANDATED_CLEAR_HARMFUL'}==expected_harm,'all9_harmful_plus424_four_contrasts')
 eventtypes={('ABSENT_FINAL','FP'):'ADDED_FINAL_FP_AT_ANCHOR',('TP','FP'):'TP_TO_FP_AT_ANCHOR',('FP','FP'):'PERSISTENT_FP_AT_ANCHOR',('FP','TP'):'FP_TO_TP_AT_ANCHOR',('FP','ABSENT_FINAL'):'REMOVED_FINAL_FP_AT_ANCHOR'}
 for r in rows:
  sid=int(r['sample_idx']);name=r['contrast'];be,af=CONTRASTS[name];cid=int(r['anchor_id']);k=sid,name,cid;check(k not in seen,'FP_provenance_unique_key',k);seen.add(k);groups[sid,name].append(r);bs,bpi,bgi=cand((sid,be),cid,70);ast,api,agi=cand((sid,af),cid,70);bb=R[sid,be];aa=R[sid,af];bi=I[sid,be][:,cid];ai=I[sid,af][:,cid];bcode=int(bb['first_deletion_code'][cid]);acode=int(aa['first_deletion_code'][cid]);ev=eventtypes[bs,ast]
  check(r['condition_before']==be and r['condition_after']==af and float(r['iou_threshold'])==.7 and r['final_status_before']==bs and r['final_status_after']==ast and r['event_type']==ev,'FP_event_partition_from_raw',k);check(int(r['delta_FP_contribution'])==int(ast=='FP')-int(bs=='FP'),'FP_event_signed_accounting',k)
  for side,a,iv,pi,gi in [('before',bb,bi,bpi,bgi),('after',aa,ai,api,agi)]:
   eq(ids(r['raw_corners_'+side]),a['raw_corners'][cid],'FP_actual_same_anchor_geometry',(k,side));close(float(r['score_'+side]),a['scores_all'][cid],'FP_actual_same_anchor_score',(k,side));close(float(r['max_GT_IoU_'+side]),iv.max(),'FP_actual_max_GT_IoU',(k,side));check(number(r['final_prediction_index_'+side])==pi and number(r['final_matched_GT_idx_'+side])==gi,'FP_nullable_match_fields',(k,side));check(int(r['first_deletion_code_'+side])==int(a['first_deletion_code'][cid]),'FP_actual_deletion_code',(k,side))
   sup=int(a['suppressor'][cid]);check(number(r['native_suppressor_'+side])==(sup if sup>=0 else None),'FP_actual_native_suppressor',(k,side))
   if sup>=0:close(float(r['suppressor_score_'+side]),a['scores_all'][sup],'FP_suppressor_score');close(float(r['suppressor_IoU_'+side]),a['suppressor_overlap'][cid],'FP_suppressor_IoU')
   else:check(r['suppressor_score_'+side]=='' and r['suppressor_IoU_'+side]=='','FP_missing_suppressor_fields_not_zero')
  close(float(r['score_delta']),float(aa['scores_all'][cid])-float(bb['scores_all'][cid]),'FP_score_delta',k)
  pair=float(overlap(polys(bb['raw_corners'][cid:cid+1])[0],polys(aa['raw_corners'][cid:cid+1]))[0]);close(float(r['same_anchor_BEV_IoU']),pair,'FP_same_anchor_overlap',k);check(boolean(r['substantial_geometry_shift_BEV_IoU_below_0p5'])==(pair<.5),'FP_substantial_geometry_shift_predicate',k)
  labels=[]
  if ev=='ADDED_FINAL_FP_AT_ANCHOR':labels.append({1:'ENTERS_FINAL_FROM_SCORE_FILTER',2:'ENTERS_FINAL_FROM_LARGE_Z_FILTER',3:'ENTERS_FINAL_FROM_TOP1000',4:'ENTERS_FINAL_FROM_NMS_SUPPRESSION',5:'ENTERS_FINAL_FROM_RANGE'}[bcode])
  elif ev=='REMOVED_FINAL_FP_AT_ANCHOR':labels.append({1:'REMOVED_BY_SCORE_FILTER',2:'REMOVED_BY_LARGE_Z_FILTER',3:'REMOVED_BY_TOP1000',4:'REMOVED_BY_NMS_SUPPRESSION',5:'REMOVED_BY_RANGE'}[acode])
  if ev=='TP_TO_FP_AT_ANCHOR' and float(ai[bgi])<.7:labels.append('SAME_ANCHOR_GEOMETRY_BELOW_PREVIOUS_GT_THRESHOLD')
  if ev=='FP_TO_TP_AT_ANCHOR' and float(bi[agi])<.7:labels.append('SAME_ANCHOR_GEOMETRY_CROSSES_GT_THRESHOLD')
  if ast=='FP':labels.append('AFTER_FP_UNIQUE_GT_MATCH_COMPETITION' if float(ai.max())>=.7 else 'AFTER_FP_BELOW_ALL_GT_IOU70')
  if bs=='FP' and float(bi.max())>=.7:labels.append('BEFORE_FP_UNIQUE_GT_MATCH_COMPETITION')
  if ev=='PERSISTENT_FP_AT_ANCHOR':labels.append('PERSISTENT_GRID_CANDIDATE_FP_NOT_PHYSICAL_OBJECT_ID')
  check(ids(r['observable_path_labels'])==labels,'all_FP_stage_labels_have_raw_predicate',k);check(r['cross_condition_physical_FP_identity']=='NOT_ESTABLISHED_FROM_ANCHOR_OR_GEOMETRY_PROXY','no_unproven_physical_FP_identity',k)
  check(r['source_sha_before']==RAW_HASH[sid,be]['sha256'] and r['source_sha_after']==RAW_HASH[sid,af]['sha256'],'FP_source_SHA',k)
 for s in summaries:
  sid=int(s['sample_idx']);name=s['contrast'];be,af=CONTRASTS[name];group=groups[sid,name];counter=collections.Counter(r['event_type'] for r in group);fb={int(cid) for cid,gi in zip(R[sid,be]['ids_final_range'],M[(sid,be),70]['p2g']) if gi<0};fa={int(cid) for cid,gi in zip(R[sid,af]['ids_final_range'],M[(sid,af),70]['p2g']) if gi<0};check({int(r['anchor_id']) for r in group}==fb|fa,'FP_all_actual_FPs_covered',(sid,name));check(int(s['FP_before'])==len(fb) and int(s['FP_after'])==len(fa) and int(s['delta_FP'])==len(fa)-len(fb),'FP_comparison_raw_count_difference',(sid,name))
  for ev in eventtypes.values():check(int(s[ev])==counter[ev],'FP_event_count_partition',(sid,name,ev))
  check(sum(int(r['delta_FP_contribution']) for r in group)==int(s['delta_FP']),'FP_event_sum_exact',(sid,name))
 for b in csvrows(B/'R5B_SAMPLE_AGENT_UTILITY.csv'):
  sid=int(b['sample_idx']);be,af=CONTRASTS[b['contrast']]
  for t in (50,70):
   for n in ('TP','FP','FN'):check(C[sid,af][n+str(t)]-C[sid,be][n+str(t)]==int(b['delta_'+n+str(t)]),'all32_marginals_both_thresholds_vs_B',(sid,b['contrast'],t,n))
 # Geometric association is only a mutually unique >=.5 BEV proxy, never identity.
 cached={};assoc=csvrows(ROOT/'R5C_GEOMETRIC_ASSOCIATION.csv')
 for r in assoc:
  sid=int(r['sample_idx']);be=r['condition_before'];af=r['condition_after'];key=sid,be,af
  if key not in cached:
   bp=polys(R[sid,be]['final_corners']);ap=polys(R[sid,af]['final_corners']);mat=np.asarray([overlap(p,ap) for p in bp],np.float32);cached[key]=mat
  mat=cached[key];side=r['side'];src=R[sid,be if side=='before' else af];dst=R[sid,af if side=='before' else be];cid=int(r['final_candidate_id']);pi=int(np.flatnonzero(src['ids_final_range']==cid)[0]);m=mat if side=='before' else mat.T;adj=m>=.5;partners=np.flatnonzero(adj[pi]);unique=len(partners)==1 and adj[:,partners[0]].sum()==1;pid=int(dst['ids_final_range'][partners[0]]) if unique else None
  expected='UNIQUE_GEOMETRIC_PROXY' if unique else 'AMBIGUOUS_GEOMETRIC_PROXY' if len(partners) else 'NO_GEOMETRIC_PROXY_ABOVE_THRESHOLD';check(r['status']==expected and number(r['unique_partner_candidate_id'])==pid and ids(r['eligible_partner_ids'])==[int(dst['ids_final_range'][i]) for i in partners],'geometric_proxy_uniqueness_and_ambiguity',(key,side,cid));check(r['physical_object_identity_verified']=='False' and float(r['threshold_BEV_IoU'])==.5,'geometry_proxy_not_physical_identity')
 return len(rows),len(summaries),collections.Counter(r['status'] for r in assoc)


def validate_interactions():
 sample_i={};totals={}
 for sid in sorted({s for s,c in R}):
  for t in (50,70):
   values={n:C[sid,'F'][n+str(t)]-C[sid,'G-C'][n+str(t)]-C[sid,'G-B'][n+str(t)]+C[sid,'E'][n+str(t)] for n in ('TP','FP','FN')};sample_i[sid,t]=values
 for t in (50,70):totals[t]={n:sum(sample_i[sid,t][n] for sid in {s for s,c in R}) for n in ('TP','FP','FN')}
 check(totals[70]=={'TP':2,'FP':-3,'FN':-2} and totals[50]=={'TP':1,'FP':-2,'FN':-1},'global_interactions_both_thresholds')
 for c in CONDITIONS:
  check(tuple(sum(C[s,c][n+'70'] for s in {s for s,cc in R}) for n in ('TP','FP','FN'))=={'F':(45,18,11),'G-B':(40,18,16),'G-C':(42,16,14),'E':(39,13,17)}[c],'global_four_condition_counts70',c)
 for b in csvrows(B/'R5B_GT_FOUR_CONDITION_PATTERNS.csv'):
  sid=int(b['sample_idx']);t=int(round(float(b['iou_threshold'])*100));gi=int(b['gt_idx']);bits=[int(M[(sid,c),t]['flags'][gi]) for c in STATE_ORDER];check(''.join(map(str,bits))==b['pattern_bits'] and bits[3]-bits[2]-bits[1]+bits[0]==int(b['I_GT']),'all112_GT_patterns_and_interactions_vs_B',(sid,t,gi))
 anchor_rows=csvrows(ROOT/'R5C_FP_ANCHOR_INTERACTION.csv');sums=collections.Counter();seen=set()
 for r in anchor_rows:
  sid=int(r['sample_idx']);t=int(round(float(r['iou_threshold'])*100));cid=int(r['anchor_id']);k=sid,t,cid;check(k not in seen,'FP_anchor_interaction_key',k);seen.add(k);bits=[int(cand((sid,c),cid,t)[0]=='FP') for c in STATE_ORDER];check([int(r[n]) for n in ['FP_E','FP_GB','FP_GC','FP_F']]==bits and int(r['I_FP_anchor'])==bits[3]-bits[2]-bits[1]+bits[0],'FP_anchor_interaction_raw_bits',k);sums[sid,t]+=int(r['I_FP_anchor']);check(boolean(r['physical_FP_identity_not_inferred']),'FP_interaction_no_physical_identity',k)
 for r in csvrows(ROOT/'R5C_INTERACTION_MECHANISM_MATRIX.csv'):
  sid=int(r['sample_idx']);t=int(round(float(r['iou_threshold'])*100));check({n:int(r['I_'+n]) for n in ('TP','FP','FN')}==sample_i[sid,t] and sums[sid,t]==sample_i[sid,t]['FP'],'all_focus_interactions_exact',(sid,t));counts=ids(r['four_condition_counts']);check(counts=={c:{n:C[sid,c][n+str(t)] for n in ('TP','FP','FN')} for c in CONDITIONS},'interaction_four_condition_counts',(sid,t))
  patterns=ids(r['GT_state_patterns'])
  for g in patterns:
   gi=next(gi for (s,gi),gm in GT.items() if s==sid and gm['physical_object_id']==g['physical_GT_id']);bits=[int(M[(sid,c),t]['flags'][gi]) for c in STATE_ORDER];check(g['pattern']==''.join(map(str,bits)) and g['I_GT']==bits[3]-bits[2]-bits[1]+bits[0],'interaction_physical_GT_pattern',(sid,t,gi))
  check(sum(x['I_GT'] for x in patterns)==sample_i[sid,t]['TP'],'interaction_GT_sum_matches_TP',(sid,t))
 cases=csvrows(ROOT/'R5C_CASE_MANIFEST.csv');check(len(cases)==45 and len({r['case_id'] for r in cases})==45,'45_case_manifest_unique');check(sum(r['case_type']=='PREDECLARED_CORE_GT' for r in cases)==24 and sum(r['case_type']=='MANDATED_CLEAR_HARMFUL' for r in cases)==9,'all_requested_cases_in_manifest')
 for r in cases:
  if r['case_type']=='PREDECLARED_CORE_GT':
   sid=int(r['sample_idx']);c=r['condition'];gi=int(r['gt_idx']);check(GT[sid,gi]['physical_object_id']==r['physical_GT_id'] and RAW_HASH[sid,c]['sha256']==r['source_sha256'] and RAW_HASH[sid,c]['absolute_path']==r['source_path'],'case_manifest_actual_GT_source')
 return totals,sample_i

def finish(unresolved,counts,totals,proxy):
 protected=load(ROOT/'R5C_HISTORY_READ_ONLY_SEAL.json')['files']
 for p,h in protected.items():check(Path(p).is_file() and sha(p)==h,'historical_file_unchanged',p)
 check(not DENIED and not any(k in sys.modules for k in ('torch','opencood','cupy','pycuda','pynvml')),'no_forbidden_import_or_mutation')
 source={n:sha(ROOT/n) for n in ['validate_r5c_independent.py','analyze_r5c.py','stage_common.py','source_reference/offline_science.py','source_reference/offline_storage_reader.py']}
 state='R5C_FAIL' if FAILED else 'R5C_PARTIAL' if unresolved else 'R5C_PASS'
 result={'state':state,'validator_role':'separate CPU process; never imports producer; original NPZ replay; independent active-mask NMS and unique GT loop; shared SHA-verified float32 polygon primitive only','check_count':sum(CHECKS.values()),'passed_check_count':sum(CHECKS.values())-len(FAILED),'failed_check_count':len(FAILED),'check_groups':dict(CHECKS),'FAIL':FAILED,'UNRESOLVED_required_candidate_chains':unresolved,'UNRESOLVED_required_count':len(unresolved),'counts':counts,'global_interactions':totals,'physical_FP_proxy_counts':dict(proxy),'unidentifiable_not_validation_failures':{'upstream_attention_MetaFlow_causal_separation':'NOT_IDENTIFIED: no controlled upstream intervention; all 32 saved raw head geometries/scores may differ','physical_FP_object_identity':'NOT_MEASURED: same grid/anchor and mutual unique >=.5 BEV proxy do not prove a physical object'},'source_sha256':source,'historical_files_verified_unchanged':len(protected),'source_npz_count':len(R),'new_forward':0,'GPU_PPU_model_calls':0,'CUDA_initializations':0,'torch_imported':False,'opencood_imported':False,'historical_evidence_modifications':0,'R5D_or_new_experiments':False,'package_scope':'all score-pass and every >=.5 raw GT candidate geometry exported, all12800 score/GT-IoU matrices exported; omitted unselected raw geometry must be rechecked at original SHA-bound server NPZ'}
 write('R5C_INDEPENDENT_VALIDATION.json',result);print(state,'CHECKS',sum(CHECKS.values()),'FAIL',len(FAILED),'UNRESOLVED',len(unresolved),flush=True)
 return state

def main():
 initialize();validate_GT();validate_order_rows();unresolved=validate_labels();nline,nedge=validate_lineage_exports();nfp,ncomp,proxy=validate_FP();totals,sample_i=validate_interactions();return finish(unresolved,{'raw_real_records':32,'GT_trace_rows':len(TR),'candidate_lineage_rows':nline,'native_suppressor_edges':nedge,'FP_provenance_rows':nfp,'FP_comparisons':ncomp,'harmful_comparisons':9,'core_GT_conditions_thresholds':24,'focus_interaction_rows':8},totals,proxy)
if __name__=='__main__':
 try:status=main()
 except Exception as e:
  write('R5C_INDEPENDENT_VALIDATION.json',{'state':'R5C_FAIL','validator_exception':str(e),'traceback':traceback.format_exc(),'checks_completed':sum(CHECKS.values()),'FAIL':FAILED,'new_forward':0,'CUDA_initializations':0});raise
 if status=='R5C_FAIL':sys.exit(1)
