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
CONDS=('F','G-B','G-C','E');STATE_ORDER=('E','G-C','G-B','F')
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
 table('R5B_SAMPLE_REMOTE_MARGINALS.csv',MARGINALS)
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
