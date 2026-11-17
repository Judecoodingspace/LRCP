"""R5B CPU result-level analysis. No score/NMS/attention mechanism attribution.
B1/B2 are explicitly specified; generic GT state/interaction products stay
provisional until the continuation of the truncated user request is received.
"""
import os,sys,json,csv,hashlib,collections,itertools,importlib.abc,datetime,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent;REPORTS=ROOT.parent;R5A=REPORTS/'RD3_R5A_EVIDENCE_AUDIT';R4=REPORTS/'RD3_REAL_FOUR_GROUP_R4'
DENIED=[]
class OfflineOnly(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):
   DENIED.append('import:'+fullname);raise ImportError('R5B_FORBIDDEN_MODEL_DEVICE_IMPORT '+fullname)
sys.meta_path.insert(0,OfflineOnly());os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True

def allowed(p):
 if isinstance(p,int):return True
 try:return Path(os.fsdecode(p)).resolve().is_relative_to(ROOT)
 except (TypeError,ValueError):return False

def guard(event,args):
 if event=='subprocess.Popen':DENIED.append(event);raise RuntimeError('NO_R5B_SUBPROCESS')
 if event=='open':
  p,mode,flags=args;writing=(isinstance(mode,str) and any(x in mode for x in 'wa+x')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not allowed(p):DENIED.append('write:'+str(p));raise PermissionError('R5B_WRITE_OUTSIDE_NEW_DIRECTORY')
 if event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime') and not allowed(args[0]):DENIED.append(event);raise PermissionError('R5B_MUTATION_OUTSIDE_NEW_DIRECTORY')
 if event in ('os.rename','os.replace') and (not allowed(args[0]) or not allowed(args[1])):DENIED.append(event);raise PermissionError('R5B_RENAME_OUTSIDE_NEW_DIRECTORY')
 if event=='ctypes.dlopen' and args[0] and any(x in str(args[0]).lower() for x in ('libcuda','libcudart','libtorch','libalinpu','ppu_sdk')):DENIED.append(event);raise RuntimeError('R5B_DEVICE_LIBRARY_FORBIDDEN')
sys.addaudithook(guard)
import numpy as np
from source_reference.offline_science import unique_match,require
from source_reference.offline_storage_reader import digest
CONDS=('F','G-B','G-C','E');STATE_ORDER=('E','G-B','G-C','F')
CONTRASTS={'U_B_given_C':('G-B','F','B','B|C'),'U_C_given_B':('G-C','F','C','C|B'),'U_B_given_A':('E','G-C','B','B|A'),'U_C_given_A':('E','G-B','C','C|A')}
CATEGORIES=('CLEAR_BENEFICIAL','CLEAR_HARMFUL','TRADEOFF','COUNT_NEUTRAL')
REFERENCE={324:(28,30,6),424:(29,30,7),725:(93,31,5),354:(28,60,6),464:(29,70,10),765:(93,71,6),384:(28,90,7),484:(29,90,9)}
EXPECTED70={'F':(45,18,11),'G-B':(40,18,16),'G-C':(42,16,14),'E':(39,13,17)}
COUNTS={};RAW={};STATES={};MARGINALS=[];SOURCES=[]

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read_csv(p):
 with Path(p).open(newline='') as f:r=csv.DictReader(f);cols=r.fieldnames;rows=list(r)
 return cols,rows
def table(n,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with (ROOT/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def write(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def equal(a,b,reason):require(np.array_equal(np.asarray(a),np.asarray(b)),reason)

def classification(dt,df):
 conditions={'CLEAR_BENEFICIAL':dt>=0 and df<=0 and (dt>0 or df<0),'CLEAR_HARMFUL':dt<=0 and df>=0 and (dt<0 or df>0),'TRADEOFF':(dt>0 and df>0) or (dt<0 and df<0),'COUNT_NEUTRAL':dt==0 and df==0}
 hits=[k for k,v in conditions.items() if v];require(len(hits)==1,'CLASSIFICATION_MUST_BE_EXHAUSTIVE_AND_EXCLUSIVE');return hits[0]

def verify_sources():
 sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (R5A/'SHA256SUMS.txt').read_text().splitlines()}
 for n in ['R5A_GT_IDENTITY_MAP.csv','R5A_SOURCE_INVENTORY.csv','R5A_SAMPLE_CONDITION_COUNTS.csv','R5A_CSV_SCHEMA_AUDIT.json','RD3_R5A_FINAL_REPORT.md','R5A_SCIENTIFIC_RECORD_AUDIT.json']:
  require(sha(R5A/n)==sums[n],'R5A_VERIFIED_OUTPUT_HASH '+n);SOURCES.append({'path':str(R5A/n),'sha256':sums[n],'role':'R5A_verified_basis'})
 require(load(R5A/'R5A_RESULT.json')['state']=='R5A_PASS','R5A_NOT_ACCEPTED')
 schema=load(R5A/'R5A_CSV_SCHEMA_AUDIT.json');selected=[x for x in schema['versions'] if x['selected_for_audit']]
 for s in selected:
  p=Path(s['absolute_path']);require(sha(p)==s['sha256'],'ORIGINAL_CSV_HASH_CHANGED '+str(p));cols,rows=read_csv(p);require(cols==s['columns'] and len(rows)==s['data_row_count'],'ACTUAL_CSV_SCHEMA_CHANGED')
  SOURCES.append({'path':str(p),'sha256':s['sha256'],'role':'original_real_CSV','rows':len(rows),'columns':len(cols)})
 require(len(selected)==4,'FOUR_ORIGINAL_CSVS_REQUIRED')
 return {x['absolute_path']:x for x in read_csv(R5A/'R5A_SOURCE_INVENTORY.csv')[1]}

def b1(inventory):
 source_rows=read_csv(R5A/'R5A_SAMPLE_CONDITION_COUNTS.csv')[1];require(len(source_rows)==32,'R5A_COUNTS_EXACT32');frozen=load(REPORTS/'RD3_MESSAGE_UTILITY_PILOT/FROZEN_MANIFEST.json');require([(r['sample_idx'],r['scene_id'],r['scene_frame'],r['GT_count']) for r in frozen['samples']]==[(sid,*info) for sid,info in REFERENCE.items()],'FROZEN_IDENTITIES')
 gm=read_csv(R5A/'R5A_GT_IDENTITY_MAP.csv')[1];gmap={(int(x['sample_idx']),int(x['gt_idx'])):x for x in gm};require(len(gmap)==56,'GT_MAP_UNIQUE56')
 rows=[]
 for src in source_rows:
  sid=int(src['sample_idx']);c=src['condition'];key=(sid,c);require(key not in COUNTS and c in CONDS and sid in REFERENCE,'DUPLICATE_OR_UNKNOWN_CONDITION')
  scene,frame,gt=REFERENCE[sid];require((int(src['scene_id']),int(src['scene_frame']),int(src['GT_count']))==(scene,frame,gt),'SAMPLE_SCENE_FRAME_GT')
  path=Path(src['source_path']);require(str(path) in inventory and sha(path)==inventory[str(path)]['sha256'],'RAW_SCIENTIFIC_ARRAY_HASH')
  SOURCES.append({'path':str(path),'sha256':inventory[str(path)]['sha256'],'role':'necessary_existing_real_result_shard','sample_idx':sid,'condition':c})
  with np.load(path,allow_pickle=False) as z:
   names=['GT_canonical','final_corners','final_scores','ids_final_range']+[f'{prefix}{t}' for prefix in ['gt_tp','prediction_gt_idx','prediction_fp'] for t in (50,70)]
   require(all(n in z for n in names),'RESULT_LEVEL_ARRAY_MISSING');a={n:z[n].copy() for n in names}
  require(a['GT_canonical'].shape==(gt,8,3) and len(a['final_corners'])==len(a['final_scores'])==len(a['ids_final_range']),'RESULT_COUNTS_SHAPES')
  row={'sample_idx':sid,'scene_id':scene,'scene_frame':frame,'GT_count':gt,'condition':c,'final_prediction_count':len(a['final_scores']),'source_file':str(path),'source_record_key':json.dumps({'sample_idx':sid,'condition':c}),'source_stage':src['source_stage'],'R5A_count_file':str(R5A/'R5A_SAMPLE_CONDITION_COUNTS.csv'),'evidence_level':'existing real final boxes/scores/GT rematched on CPU, crosschecked against R5A independent counts'}
  RAW[key]=a
  for gi in range(gt):require(digest(a['GT_canonical'][gi])==gmap[sid,gi]['GT_hash'],'ACTUAL_CANONICAL_GT_PHYSICAL_MAPPING')
  for t in (50,70):
   match=unique_match(a['final_corners'],a['final_scores'],a['GT_canonical'],t/100)
   equal(match['matched'],a[f'prediction_gt_idx{t}'],'EXACT_UNIQUE_MATCH_ID');equal(match['flags'],a[f'gt_tp{t}'],'EXACT_GT_STATUS');equal((match['matched']<0).astype(np.int8),a[f'prediction_fp{t}'],'EXACT_FP_FLAGS')
   tp=sum(match['tp']);fp=sum(match['fp']);fn=gt-tp
   for k,v in [(f'TP{t}',tp),(f'FP{t}',fp),(f'FN{t}',fn)]:require(v==int(src[k]),'R5A_INDEPENDENT_COUNT_CONFLICT '+str(key)+k);row[k]=v
   require(tp+fn==gt and tp+fp==row['final_prediction_count'],'B1_CONSERVATION')
   STATES[sid,c,t]=match
  COUNTS[key]=row;rows.append(row);print('B1_REAL_RESULT_REMATCHED',sid,c,row['TP70'],row['FP70'],row['FN70'],flush=True)
 require(set(COUNTS)=={(sid,c) for sid in REFERENCE for c in CONDS},'EXACT_FROZEN_8x4_COMPLETE')
 for sid in REFERENCE:
  for c in CONDS:equal(RAW[sid,c]['GT_canonical'],RAW[sid,'F']['GT_canonical'],'FOUR_CONDITION_GT_EQUAL')
 # NMS CSV is used only for saved final prediction and unique-GT state checks.
 # No candidate-stage/score/suppressor/attention fields are analyzed.
 nmsrows=read_csv(R4/'analysis/NMS_STAGE_ANALYSIS.csv')[1]
 for x in nmsrows:
  sid=int(x['sample_idx']);c=x['condition']
  if x['record_type']=='condition_summary':require(int(x['final_prediction_count'])==COUNTS[sid,c]['final_prediction_count'],'NMS_TABLE_FINAL_COUNT')
  elif x['record_type']=='GT_stage':
   gi=int(x['gt_idx']);t=int(round(float(x['iou_threshold'])*100));require(int(x['native_unique_TP'])==int(STATES[sid,c,t]['flags'][gi]),'SAVED_FINAL_GT_STATUS')
  else:raise RuntimeError('UNKNOWN_CSV_RECORD_VARIANT')
 aggregate=[]
 for c in CONDS:
  a={'condition':c,'sample_count':8,'scene_count':3,'GT_count':sum(COUNTS[sid,c]['GT_count'] for sid in REFERENCE),'final_prediction_count':sum(COUNTS[sid,c]['final_prediction_count'] for sid in REFERENCE)}
  for t in (50,70):
   for metric in ('TP','FP','FN'):a[metric+str(t)]=sum(COUNTS[sid,c][metric+str(t)] for sid in REFERENCE)
  require(tuple(a[k] for k in ['TP70','FP70','FN70'])==EXPECTED70[c],'AGGREGATE_IOU70_BASELINE_CONFLICT');require(a['GT_count']==56,'AGGREGATE_GT56');aggregate.append(a)
 table('R5B_SAMPLE_CONDITION_COUNTS.csv',rows);table('R5B_FOUR_GROUP_COUNTS.csv',aggregate)
 return gmap,aggregate

def b2():
 # Test all 9 sign combinations and a wider grid, without adding synthetic science rows.
 grid=collections.Counter(classification(tp,fp) for tp in range(-10,11) for fp in range(-10,11));require(sum(grid.values())==441 and classification(1,1)=='TRADEOFF' and classification(-1,-1)=='TRADEOFF','CLASSIFIER_SANITY')
 legacy=read_csv(R4/'analysis/AGENT_MARGINAL_UTILITY.csv')[1]
 for sid,(scene,frame,gt) in REFERENCE.items():
  for contrast,(before,after,remote,oldname) in CONTRASTS.items():
   b=COUNTS[sid,before];a=COUNTS[sid,after];row={'sample_idx':sid,'scene_id':scene,'scene_frame':frame,'GT_count':gt,'contrast':contrast,'remote_added':remote,'remote_physical_slot':1 if remote=='B' else 2,'remote_physical_id':'2' if remote=='B' else '3','condition_before':before,'condition_after':after,'delta_sign':'after_remote_added_minus_before','source_before':b['source_file'],'source_after':a['source_file'],'source_record_key_before':b['source_record_key'],'source_record_key_after':a['source_record_key']}
   old=[x for x in legacy if x['scope']=='sample' and int(x['sample_idx'])==sid and x['utility']==oldname];require(len(old)==1,'ORIGINAL_MARGINAL_ROW_KEY')
   for t in (50,70):
    for m in ('TP','FP','FN'):
     field=m+str(t);delta=a[field]-b[field];row['before_'+field]=b[field];row['after_'+field]=a[field];row['delta_'+field]=delta;require(delta==int(old[0][field]),'ORIGINAL_MARGINAL_DELTA_CONFLICT')
    require(row[f'delta_FN{t}']==-row[f'delta_TP{t}'],'MARGINAL_FN_TP_SIGN');row['class'+str(t)]=classification(row[f'delta_TP{t}'],row[f'delta_FP{t}'])
   row['classification_changed_at_IoU50_vs70']=row['class50']!=row['class70'];MARGINALS.append(row)
 require(len(MARGINALS)==len({(x['sample_idx'],x['contrast']) for x in MARGINALS})==32,'B2_EXACT32_CONTRASTS')
 table('R5B_SAMPLE_AGENT_UTILITY.csv',MARGINALS)
 summaries=[]
 for t in (70,50):
  for contrast in ('ALL_FOUR_CONTRASTS',*CONTRASTS):
   subset=MARGINALS if contrast=='ALL_FOUR_CONTRASTS' else [x for x in MARGINALS if x['contrast']==contrast]
   for cat in CATEGORIES:
    got=[x for x in subset if x['class'+str(t)]==cat]
    summaries.append({'iou_threshold':t/100,'contrast':contrast,'classification':cat,'contrast_count':len(got),'denominator_contrasts':len(subset),'unique_sample_count':len({x['sample_idx'] for x in got}),'sample_ids':json.dumps(sorted({x['sample_idx'] for x in got})),'unique_scene_count':len({x['scene_id'] for x in got}),'scene_ids':json.dumps(sorted({x['scene_id'] for x in got})),'unit':'sample×conditional-marginal contrast; 32 contrasts are repeated within 8 frames/3 scenes, not 32 independent samples'})
   require(sum(x['contrast_count'] for x in summaries if x['iou_threshold']==t/100 and x['contrast']==contrast)==len(subset),'CLASSIFICATION_PARTITION')
 table('R5B_MARGINAL_CLASS_SUMMARY.csv',summaries)
 cross=[]
 for c70,c50 in itertools.product(CATEGORIES,repeat=2):
  got=[x for x in MARGINALS if (x['class70'],x['class50'])==(c70,c50)];cross.append({'class70':c70,'class50':c50,'contrast_count':len(got),'contrast_keys':json.dumps([[x['sample_idx'],x['contrast']] for x in got])})
 table('R5B_IOU50_70_CLASSIFICATION_CROSSTAB.csv',cross)
 harmful=[x for x in MARGINALS if x['class70']=='CLEAR_HARMFUL'];changed=[x for x in MARGINALS if x['classification_changed_at_IoU50_vs70']]
 summary={'B1_B2_state':'COMPLETED','IoU70_category_counts':dict(collections.Counter(x['class70'] for x in MARGINALS)),'IoU50_category_counts':dict(collections.Counter(x['class50'] for x in MARGINALS)),'clear_harmful_contrast_count70':len(harmful),'clear_harmful_unique_samples70':sorted({x['sample_idx'] for x in harmful}),'clear_harmful_unique_sample_count70':len({x['sample_idx'] for x in harmful}),'clear_harmful_unique_scenes70':sorted({x['scene_id'] for x in harmful}),'clear_harmful_unique_scene_count70':len({x['scene_id'] for x in harmful}),'IoU50_vs70_changed_contrast_count':len(changed),'IoU50_vs70_changed_keys':[[x['sample_idx'],x['contrast']] for x in changed],'all_four_categories_exclusive_exhaustive':True,'classifier_grid_checks':441,'classifiers_do_not_use_weighted_utility':True,'independent_experimental_units':'8 frozen frames clustered in 3 scenes; within-frame contrasts and GT are dependent; no significance test or population prevalence inferred'}
 write('R5B_B1_B2_VALIDATION.json',summary);print('B1_B2_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
 return summary

TRANSITIONS=[];TRANSITION_SUMMARIES=[];PATTERNS=[];INTERACTIONS=[]

def physical_key(row):return str(row['scene_id'])+'/'+str(row['physical_object_id'])
def event_totals(rows):
 return {'event_count':len(rows),'unique_frame_GT_count':len({(x['sample_idx'],str(x['physical_object_id'])) for x in rows}),'distinct_scene_object_count':len({(x['scene_id'],str(x['physical_object_id'])) for x in rows}),'unique_sample_count':len({x['sample_idx'] for x in rows}),'sample_ids':sorted({x['sample_idx'] for x in rows}),'unique_scene_count':len({x['scene_id'] for x in rows}),'scene_ids':sorted({x['scene_id'] for x in rows})}

def b3(gmap):
 fields,original=read_csv(R4/'analysis/PER_GT_TRANSITIONS.csv');index={}
 for row in original:
  key=(int(row['sample_idx']),row['condition_from'],row['condition_to'],int(row['gt_idx']),int(round(float(row['iou_threshold'])*100)));require(key not in index,'ORIGINAL_TRANSITION_KEY_DUPLICATE');index[key]=row
  sid,p,q,gi,t=key;g=gmap[sid,gi];require(row['GT_hash']==g['GT_hash'],'CSV_PHYSICAL_GT_HASH');before=int(STATES[sid,p,t]['flags'][gi]);after=int(STATES[sid,q,t]['flags'][gi]);require(row['transition']==('TP' if before else 'FN')+'→'+('TP' if after else 'FN'),'ORIGINAL_TRANSITION_STATUS')
 require(len(index)==672,'ALL_ORIGINAL_GT_TRANSITIONS')
 marginal_index={(x['sample_idx'],x['contrast']):x for x in MARGINALS}
 for sid,(scene,frame,gt) in REFERENCE.items():
  for contrast,(before,after,remote,_) in CONTRASTS.items():
   m=marginal_index[sid,contrast]
   for t in (50,70):
    local=[];counter=collections.Counter()
    for gi in range(gt):
     g=gmap[sid,gi];db=int(STATES[sid,before,t]['flags'][gi]);da=int(STATES[sid,after,t]['flags'][gi]);state=('TP' if db else 'FN')+'_TO_'+('TP' if da else 'FN');counter[state]+=1
     key=(sid,before,after,gi,t);reverse=False
     if key not in index:key=(sid,after,before,gi,t);reverse=True
     source=index[key];require(source['GT_hash']==g['GT_hash'],'MARGINAL_GT_MAP')
     bpi=np.flatnonzero(STATES[sid,before,t]['matched']==gi);api=np.flatnonzero(STATES[sid,after,t]['matched']==gi)
     ba=int(RAW[sid,before]['ids_final_range'][int(bpi[0])]) if len(bpi) else -1;aa=int(RAW[sid,after]['ids_final_range'][int(api[0])]) if len(api) else -1
     row={'sample_idx':sid,'scene_id':scene,'scene_frame':frame,'GT_count':gt,'contrast':contrast,'remote_added':remote,'condition_before':before,'condition_after':after,'iou_threshold':t/100,'gt_idx':gi,'physical_object_id':g['physical_object_id'],'scene_object_key':str(scene)+'/'+g['physical_object_id'],'frame_GT_key':str(sid)+'/'+g['physical_object_id'],'GT_hash':g['GT_hash'],'D_before':db,'D_after':da,'transition_state':state,'delta_GT_TP':da-db,'before_matched_anchor':ba,'after_matched_anchor':aa,'source_file_before':COUNTS[sid,before]['source_file'],'source_file_after':COUNTS[sid,after]['source_file'],'source_record_key':json.dumps({'sample_idx':sid,'contrast':contrast,'physical_object_id':g['physical_object_id'],'iou_threshold':t/100}),'source_transition_csv':str(R4/'analysis/PER_GT_TRANSITIONS.csv'),'source_transition_row_key':json.dumps({'sample_idx':sid,'condition_from':key[1],'condition_to':key[2],'gt_idx':gi,'iou_threshold':t/100}),'source_csv_direction_reversed':reverse,'physical_GT_mapping_source':str(R5A/'R5A_GT_IDENTITY_MAP.csv')}
     local.append(row)
    gains=counter['FN_TO_TP'];losses=counter['TP_TO_FN'];dt=m['delta_TP'+str(t)];dfn=m['delta_FN'+str(t)]
    require(sum(counter.values())==gt and gains-losses==dt and losses-gains==dfn,'GT_TRANSITION_CONSERVATION')
    hidden=dt>=0 and losses>0;turnover=dt==0 and gains>0 and losses>0
    summary={'sample_idx':sid,'scene_id':scene,'scene_frame':frame,'GT_count':gt,'contrast':contrast,'remote_added':remote,'condition_before':before,'condition_after':after,'iou_threshold':t/100,**{k:counter[k] for k in ('FN_TO_TP','TP_TO_FN','TP_TO_TP','FN_TO_FN')},'delta_TP':dt,'delta_FP':m['delta_FP'+str(t)],'delta_FN':dfn,'sample_classification':m['class'+str(t)],'HIDDEN_GT_REGRESSION':hidden,'strict_net_TP_increase_with_GT_loss':dt>0 and losses>0,'GT_IDENTITY_TURNOVER':turnover,'lost_physical_GT_keys':json.dumps([r['scene_object_key'] for r in local if r['transition_state']=='TP_TO_FN']),'lost_frame_GT_keys':json.dumps([r['frame_GT_key'] for r in local if r['transition_state']=='TP_TO_FN']),'gained_physical_GT_keys':json.dumps([r['scene_object_key'] for r in local if r['transition_state']=='FN_TO_TP']),'gained_frame_GT_keys':json.dumps([r['frame_GT_key'] for r in local if r['transition_state']=='FN_TO_TP']),'source_file':str(R4/'analysis/PER_GT_TRANSITIONS.csv'),'status':'PASS'}
    TRANSITION_SUMMARIES.append(summary)
    for row in local:
     row['case_HIDDEN_GT_REGRESSION']=hidden;row['case_strict_net_TP_increase_with_GT_loss']=dt>0 and losses>0;row['case_GT_IDENTITY_TURNOVER']=turnover;row['is_hidden_loss_event']=hidden and row['transition_state']=='TP_TO_FN';row['is_turnover_changed_GT_event']=turnover and row['transition_state'] in ('FN_TO_TP','TP_TO_FN')
    TRANSITIONS.extend(local)
 require(len(TRANSITIONS)==448 and len(TRANSITION_SUMMARIES)==64,'EXACT_GT_MARGINAL_EVENT_COVERAGE')
 for contrast in CONTRASTS:
  for t in (50,70):require(len([x for x in TRANSITIONS if x['contrast']==contrast and x['iou_threshold']==t/100])==56,'EVERY_CONTRAST_HAS_56_GT_PAIRS')
 table('R5B_GT_TRANSITION_AUDIT.csv',TRANSITIONS);table('R5B_GT_TRANSITION_SUMMARY.csv',TRANSITION_SUMMARIES)
 grouped=[]
 for t in (50,70):
  for contrast in ('ALL_FOUR_CONTRASTS',*CONTRASTS):
   subset=[r for r in TRANSITIONS if r['iou_threshold']==t/100 and (contrast=='ALL_FOUR_CONTRASTS' or r['contrast']==contrast)]
   for state in ('FN_TO_TP','TP_TO_FN','TP_TO_TP','FN_TO_FN'):
    got=[r for r in subset if r['transition_state']==state];grouped.append({'iou_threshold':t/100,'contrast':contrast,'transition_state':state,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in event_totals(got).items()}})
 table('R5B_GT_TRANSITION_EVENT_COUNTS.csv',grouped)
 cases=[]
 for s in TRANSITION_SUMMARIES:
  labels=[]
  if s['sample_classification']=='CLEAR_HARMFUL':labels.append('SAMPLE_CLEAR_HARMFUL')
  if s['sample_classification']=='TRADEOFF':labels.append('SAMPLE_TRADEOFF')
  if s['HIDDEN_GT_REGRESSION']:labels.append('HIDDEN_GT_REGRESSION')
  if not labels:continue
  cases.append({**s,'case_dimensions':json.dumps(labels),'labels_are_distinct_dimensions_not_single_utility':True,'main_analysis_IoU70':s['iou_threshold']==.7})
 table('R5B_NEGATIVE_AND_TRADEOFF_CASES.csv',cases,fields=list(TRANSITION_SUMMARIES[0])+['case_dimensions','labels_are_distinct_dimensions_not_single_utility','main_analysis_IoU70'])
 hidden_events=[x for x in TRANSITIONS if x['is_hidden_loss_event']];table('R5B_HIDDEN_GT_REGRESSION_EVENTS.csv',hidden_events,fields=list(TRANSITIONS[0]))
 hidden_summary={}
 for t in (50,70):
  threshold=t/100;ss=[x for x in TRANSITION_SUMMARIES if x['iou_threshold']==threshold];events=[x for x in TRANSITIONS if x['iou_threshold']==threshold]
  hidden=[x for x in ss if x['HIDDEN_GT_REGRESSION']];turn=[x for x in ss if x['GT_IDENTITY_TURNOVER']]
  hkeys={(x['sample_idx'],x['contrast']) for x in hidden};turnkeys={(x['sample_idx'],x['contrast']) for x in turn}
  he=[x for x in events if (x['sample_idx'],x['contrast']) in hkeys and x['transition_state']=='TP_TO_FN'];strict=[x for x in he if x['case_strict_net_TP_increase_with_GT_loss']];te=[x for x in events if (x['sample_idx'],x['contrast']) in turnkeys and x['transition_state'] in ('FN_TO_TP','TP_TO_FN')]
  hidden_summary[str(t)]={'FN_TO_TP':event_totals([x for x in events if x['transition_state']=='FN_TO_TP']),'TP_TO_FN':event_totals([x for x in events if x['transition_state']=='TP_TO_FN']),'hidden_case_count':len(hidden),'hidden_case_keys':[[x['sample_idx'],x['contrast']] for x in hidden],'hidden_loss_events':event_totals(he),'strict_net_TP_increase_hidden_loss_events':event_totals(strict),'turnover_case_count':len(turn),'turnover_case_keys':[[x['sample_idx'],x['contrast']] for x in turn],'turnover_changed_GT_events':event_totals(te)}
 write('R5B_GT_EVENT_DEDUPLICATION.json',{'physical_key':'(scene_id, original physical_object_id), reported separately from frame GT key (sample_idx, physical_object_id)','frozen_GT_frame_observations':56,'distinct_scene_object_ids':len({(x['scene_id'],x['physical_object_id']) for x in gmap.values()}),'by_threshold':hidden_summary,'transition_events_are_not_independent_GT_or_samples':True})
 return hidden_summary


def b4():
 marginal_index={(x['sample_idx'],x['contrast']):x for x in MARGINALS};rows=[];signed=[]
 for t in (50,70):
  sample=[]
  for sid,(scene,frame,gt) in REFERENCE.items():
   row={'aggregation_level':'sample','sample_idx':sid,'scene_id':scene,'scene_frame':frame,'sample_count':1,'GT_count':gt,'iou_threshold':t/100}
   for metric in ('TP','FP','FN'):
    key=metric+str(t);iv=COUNTS[sid,'F'][key]-COUNTS[sid,'G-C'][key]-COUNTS[sid,'G-B'][key]+COUNTS[sid,'E'][key]
    byB=marginal_index[sid,'U_B_given_C']['delta_'+key]-marginal_index[sid,'U_B_given_A']['delta_'+key];byC=marginal_index[sid,'U_C_given_B']['delta_'+key]-marginal_index[sid,'U_C_given_A']['delta_'+key]
    require(iv==byB==byC,'TWO_EQUIVALENT_CONTEXT_INTERACTIONS');row['I_BC_'+metric]=iv;row['I_from_B_context_'+metric]=byB;row['I_from_C_context_'+metric]=byC
   row['I_prediction_count']=COUNTS[sid,'F']['final_prediction_count']-COUNTS[sid,'G-C']['final_prediction_count']-COUNTS[sid,'G-B']['final_prediction_count']+COUNTS[sid,'E']['final_prediction_count'];require(row['I_BC_FP']==row['I_prediction_count']-row['I_BC_TP'] and row['I_BC_FN']==-row['I_BC_TP'],'INTERACTION_COMPONENT_IDENTITIES');sample.append(row)
  for metric in ('TP','FP','FN'):
   absolute=sum(abs(r['I_BC_'+metric]) for r in sample)
   for row in sample:row['abs_I_'+metric]=abs(row['I_BC_'+metric]);row['abs_I_'+metric+'_share']=abs(row['I_BC_'+metric])/absolute if absolute else 'NOT_APPLICABLE';row['abs_I_'+metric+'_denominator']=absolute
  rows.extend(sample)
  for scene in sorted({x[0] for x in REFERENCE.values()}):
   src=[r for r in sample if r['scene_id']==scene];row={'aggregation_level':'scene','sample_idx':'','scene_id':scene,'scene_frame':'','sample_count':len(src),'GT_count':sum(r['GT_count'] for r in src),'iou_threshold':t/100}
   for metric in ('TP','FP','FN'):row['I_BC_'+metric]=sum(r['I_BC_'+metric] for r in src)
   row['I_prediction_count']=sum(r['I_prediction_count'] for r in src);rows.append(row)
  total={'aggregation_level':'all_8_frames','sample_idx':'','scene_id':'ALL','scene_frame':'','sample_count':8,'GT_count':56,'iou_threshold':t/100,**{'I_BC_'+m:sum(r['I_BC_'+m] for r in sample) for m in ('TP','FP','FN')},'I_prediction_count':sum(r['I_prediction_count'] for r in sample)};rows.append(total)
  if t==70:require((total['I_BC_TP'],total['I_BC_FP'],total['I_BC_FN'])==(2,-3,-2),'REQUIRED_INTERACTION_TOTAL70')
  old=read_csv(R4/'analysis/AGENT_INTERACTION.csv')[1]
  for row in [x for x in rows if x['iou_threshold']==t/100 and x['aggregation_level']!='all_8_frames']:
   found=[x for x in old if x['scope']==row['aggregation_level'] and x['sample_idx']==str(row['sample_idx']) and x['scene_id']==str(row['scene_id'])];require(len(found)==1,'ORIGINAL_INTERACTION_SOURCE_KEY')
   for metric in ('TP','FP','FN'):require(row['I_BC_'+metric]==int(found[0][metric+str(t)]),'ORIGINAL_SAMPLE_SCENE_INTERACTION')
  for metric in ('TP','FP','FN'):
   positive=sum(max(r['I_BC_'+metric],0) for r in sample);negative=sum(min(r['I_BC_'+metric],0) for r in sample);denom=positive-negative;nonzero=[r for r in sample if r['I_BC_'+metric]!=0];scenevals={s:sum(r['I_BC_'+metric] for r in sample if r['scene_id']==s) for s in sorted({x[0] for x in REFERENCE.values()})}
   sortedshares=sorted([(r['sample_idx'],abs(r['I_BC_'+metric])/denom) for r in sample],key=lambda x:(-x[1],x[0])) if denom else []
   signed.append({'iou_threshold':t/100,'metric':metric,'positive_sum':positive,'negative_sum':negative,'net':positive+negative,'absolute_sample_interaction_sum':denom,'nonzero_sample_count':len(nonzero),'nonzero_sample_ids':json.dumps([r['sample_idx'] for r in nonzero]),'nonzero_scene_sum_count':sum(v!=0 for v in scenevals.values()),'nonzero_scene_ids':json.dumps([s for s,v in scenevals.items() if v!=0]),'scenes_containing_any_nonzero_sample_count':len({r['scene_id'] for r in nonzero}),'positive_and_negative_cancel':positive>0 and negative<0,'cancelled_magnitude_per_sign':min(positive,-negative),'top1_absolute_share':sortedshares[0][1] if sortedshares else 'NOT_APPLICABLE','top2_absolute_share':sum(x[1] for x in sortedshares[:2]) if sortedshares else 'NOT_APPLICABLE','sample_absolute_shares':json.dumps(sortedshares)})
 INTERACTIONS.extend(rows);table('R5B_SAMPLE_SCENE_INTERACTION.csv',rows);table('R5B_INTERACTION_CONCENTRATION.csv',signed);return signed


def b5(gmap):
 distribution=[]
 for t in (50,70):
  local=[]
  for sid,(scene,frame,gt) in REFERENCE.items():
   for gi in range(gt):
    g=gmap[sid,gi];de=int(STATES[sid,'E',t]['flags'][gi]);dgb=int(STATES[sid,'G-B',t]['flags'][gi]);dgc=int(STATES[sid,'G-C',t]['flags'][gi]);df=int(STATES[sid,'F',t]['flags'][gi]);ig=df-dgc-dgb+de
    db_c=df-dgb;db_a=dgc-de;dc_b=df-dgc;dc_a=dgb-de;require(ig==db_c-db_a==dc_b-dc_a,'GT_CONTEXT_INTERACTION')
    losses=[]
    for name,delta in [('U_B_given_C',db_c),('U_C_given_B',dc_b),('U_B_given_A',db_a),('U_C_given_A',dc_a)]:
     if delta<0:losses.append(name)
    row={'sample_idx':sid,'scene_id':scene,'scene_frame':frame,'gt_idx':gi,'physical_object_id':g['physical_object_id'],'scene_object_key':str(scene)+'/'+g['physical_object_id'],'frame_GT_key':str(sid)+'/'+g['physical_object_id'],'GT_hash':g['GT_hash'],'iou_threshold':t/100,'D_E':de,'D_GB':dgb,'D_GC':dgc,'D_F':df,'D_GB_definition':'G-B = A+C','D_GC_definition':'G-C = A+B','pattern_order':'D_E,D_GB,D_GC,D_F','pattern_bits':''.join(map(str,[de,dgb,dgc,df])),'I_GT':ig,'delta_GT_B_given_C':db_c,'delta_GT_B_given_A':db_a,'delta_GT_C_given_B':dc_b,'delta_GT_C_given_A':dc_a,'FULL_ONLY_TP':df==1 and de==dgb==dgc==0,'REMOTE_ADDITION_LOSS':bool(losses),'loss_contrasts':json.dumps(losses),'CONDITIONAL_DETECTION_CHANGE':db_c!=db_a or dc_b!=dc_a,'source_R5A_GT_map':str(R5A/'R5A_GT_IDENTITY_MAP.csv'),'source_files':json.dumps({c:COUNTS[sid,c]['source_file'] for c in CONDS}),'source_record_key':json.dumps({'sample_idx':sid,'physical_object_id':g['physical_object_id'],'iou_threshold':t/100})}
    require(row['CONDITIONAL_DETECTION_CHANGE']==(ig!=0),'CONDITIONAL_CHANGE_IS_NONZERO_GT_INTERACTION');local.append(row)
  PATTERNS.extend(local)
  for row in [x for x in INTERACTIONS if x['iou_threshold']==t/100]:
   selected=[x for x in local if (row['aggregation_level']=='all_8_frames' or (row['aggregation_level']=='scene' and x['scene_id']==row['scene_id']) or (row['aggregation_level']=='sample' and x['sample_idx']==row['sample_idx']))];require(sum(x['I_GT'] for x in selected)==row['I_BC_TP'],'GT_INTERACTION_SUM_EQUALS_SAMPLE_SCENE_AND_ALL')
  for bits in itertools.product((0,1),repeat=4):
   pattern=''.join(map(str,bits));selected=[x for x in local if x['pattern_bits']==pattern];ig=bits[3]-bits[2]-bits[1]+bits[0]
   distribution.append({'iou_threshold':t/100,'pattern_order':'D_E,D_GB,D_GC,D_F','pattern_bits':pattern,'D_E':bits[0],'D_GB':bits[1],'D_GC':bits[2],'D_F':bits[3],'GT_frame_observation_count':len(selected),'distinct_scene_object_count':len({(x['scene_id'],x['physical_object_id']) for x in selected}),'unique_sample_count':len({x['sample_idx'] for x in selected}),'sample_ids':json.dumps(sorted({x['sample_idx'] for x in selected})),'unique_scene_count':len({x['scene_id'] for x in selected}),'scene_ids':json.dumps(sorted({x['scene_id'] for x in selected})),'physical_GT_keys':json.dumps(sorted({x['scene_object_key'] for x in selected})),'I_GT_per_observation':ig,'TP_interaction_contribution':sum(x['I_GT'] for x in selected),'FULL_ONLY_TP':bool(bits[3] and not any(bits[:3])),'REMOTE_ADDITION_LOSS':any(d<0 for d in (bits[3]-bits[1],bits[3]-bits[2],bits[2]-bits[0],bits[1]-bits[0])),'CONDITIONAL_DETECTION_CHANGE':ig!=0})
  require(sum(x['GT_frame_observation_count'] for x in distribution if x['iou_threshold']==t/100)==56,'ALL_16_PATTERNS_EXHAUSTIVE')
 require(len(PATTERNS)==112,'56_GT_TIMES_TWO_THRESHOLDS');table('R5B_GT_FOUR_CONDITION_PATTERNS.csv',PATTERNS);table('R5B_PATTERN_DISTRIBUTION.csv',distribution)
 return distribution


def b6():
 marginal_index={(x['sample_idx'],x['contrast']):x for x in MARGINALS};ts={(x['sample_idx'],x['contrast'],int(round(x['iou_threshold']*100))):x for x in TRANSITION_SUMMARIES};sensitivity=[]
 for key,m in marginal_index.items():
  x=ts[(*key,50)];y=ts[(*key,70)];sensitivity.append({'sample_idx':key[0],'scene_id':m['scene_id'],'contrast':key[1],'class50':m['class50'],'class70':m['class70'],'classification_changed':m['class50']!=m['class70'],'CLEAR_HARMFUL50':m['class50']=='CLEAR_HARMFUL','CLEAR_HARMFUL70':m['class70']=='CLEAR_HARMFUL','CLEAR_HARMFUL_changed':(m['class50']=='CLEAR_HARMFUL')!=(m['class70']=='CLEAR_HARMFUL'),'HIDDEN_GT_REGRESSION50':x['HIDDEN_GT_REGRESSION'],'HIDDEN_GT_REGRESSION70':y['HIDDEN_GT_REGRESSION'],'HIDDEN_GT_REGRESSION_changed':x['HIDDEN_GT_REGRESSION']!=y['HIDDEN_GT_REGRESSION'],'GT_IDENTITY_TURNOVER50':x['GT_IDENTITY_TURNOVER'],'GT_IDENTITY_TURNOVER70':y['GT_IDENTITY_TURNOVER'],'GT_IDENTITY_TURNOVER_changed':x['GT_IDENTITY_TURNOVER']!=y['GT_IDENTITY_TURNOVER']})
 table('R5B_THRESHOLD_SENSITIVITY.csv',sensitivity)
 pi={(x['sample_idx'],x['physical_object_id'],int(round(x['iou_threshold']*100))):x for x in PATTERNS};changes=[]
 for sid,(_,_,gt) in REFERENCE.items():
  for row in [x for x in PATTERNS if x['sample_idx']==sid and x['iou_threshold']==.7]:
   other=pi[sid,row['physical_object_id'],50];changes.append({'sample_idx':sid,'scene_id':row['scene_id'],'physical_object_id':row['physical_object_id'],'frame_GT_key':row['frame_GT_key'],'GT_hash':row['GT_hash'],'pattern50':other['pattern_bits'],'pattern70':row['pattern_bits'],'pattern_changed':other['pattern_bits']!=row['pattern_bits'],'I_GT50':other['I_GT'],'I_GT70':row['I_GT'],'interaction_changed':other['I_GT']!=row['I_GT'],'REMOTE_ADDITION_LOSS50':other['REMOTE_ADDITION_LOSS'],'REMOTE_ADDITION_LOSS70':row['REMOTE_ADDITION_LOSS'],'CONDITIONAL_DETECTION_CHANGE50':other['CONDITIONAL_DETECTION_CHANGE'],'CONDITIONAL_DETECTION_CHANGE70':row['CONDITIONAL_DETECTION_CHANGE']})
 table('R5B_GT_PATTERN_THRESHOLD_COMPARISON.csv',changes)
 nonzero={str(t):[x['sample_idx'] for x in INTERACTIONS if x['aggregation_level']=='sample' and x['iou_threshold']==t/100 and x['I_BC_TP']!=0] for t in (50,70)}
 result={'classification_changed_cases':[[x['sample_idx'],x['contrast']] for x in sensitivity if x['classification_changed']],'CLEAR_HARMFUL_changed_cases':[[x['sample_idx'],x['contrast']] for x in sensitivity if x['CLEAR_HARMFUL_changed']],'HIDDEN_GT_REGRESSION_changed_cases':[[x['sample_idx'],x['contrast']] for x in sensitivity if x['HIDDEN_GT_REGRESSION_changed']],'GT_IDENTITY_TURNOVER_changed_cases':[[x['sample_idx'],x['contrast']] for x in sensitivity if x['GT_IDENTITY_TURNOVER_changed']],'GT_pattern_changed_observations':[x['frame_GT_key'] for x in changes if x['pattern_changed']],'nonzero_TP_interaction_sample_ids':nonzero,'nonzero_TP_interaction_sample_set_changed':set(nonzero['50'])!=set(nonzero['70']),'no_extra_IoU_thresholds_scanned':True,'no_per_frame_AP_or_weighted_utility':True,'no_population_or_causal_latency_or_communication_load_claim':True}
 write('R5B_THRESHOLD_SENSITIVITY.json',result);return result


def main():
 inventory=verify_sources();gmap,aggregate=b1(inventory);summary=b2();hidden=b3(gmap);concentration=b4();distribution=b5(gmap);sensitivity=b6()
 table('R5B_SOURCE_INVENTORY.csv',SOURCES)
 protected=load(ROOT/'R5B_HISTORY_READ_ONLY_SEAL.json')['files'];changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];require(not changed,'HISTORY_MODIFIED '+str(changed));require(not DENIED,'GUARD_EVENT '+str(DENIED));require('torch' not in sys.modules and 'opencood' not in sys.modules,'FORBIDDEN_MODEL_IMPORT')
 write('R5B_HISTORY_PRESERVATION_VALIDATION.json',{'passed':True,'verified_unchanged_file_count':len(protected),'changed_files':changed,'new_forward':'NO','CUDA_initialized':'NO','historical_data_modified':'NO'})
 write('R5B_PRODUCER_RESULT.json',{'state':'PRODUCER_COMPLETE_PENDING_INDEPENDENT_VALIDATOR','tasks_completed':['B1','B2','B3','B4','B5','B6'],'counts':aggregate,'B1_B2':summary,'GT_events':hidden,'interaction_concentration':concentration,'threshold_sensitivity':sensitivity,'source_CSV_hashes':load(ROOT/'R5B_INITIAL_SOURCE_HASH_CHECK.json')['original_CSVs_verified_before_analysis'],'full_request_received':True,'GT_frame_observations':56,'distinct_scene_object_ids':len({(x['scene_id'],x['physical_object_id']) for x in gmap.values()}),'new_forward':'NO','CUDA_initialized':'NO','historical_data_modified':'NO','torch_imported':False,'opencood_imported':False,'R5C_executed':False,'no_mechanism_attribution':True,'completed_local_time':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
 print('R5B_B1_TO_B6_PRODUCER_COMPLETE_NO_TORCH_NO_FORWARD',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write('R5B_PRODUCER_FAILURE.json',{'state':'R5B_FAIL','error':str(e),'traceback':traceback.format_exc(),'historical_data_modified':False,'new_forward':False,'CUDA_initialized':False});raise
