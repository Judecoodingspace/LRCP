"""Independent R5B validator. Does not import the producer. Starts from R5A
raw-count basis and immutable REAL npz GT matches; recomputes every invariant.
No torch/opencood, model loading, AP recomputation or R5C mechanism analysis.
"""
import os,sys,csv,json,hashlib,itertools,collections,traceback,importlib.abc
from pathlib import Path
R=Path(__file__).resolve().parent;A=R.parent/'RD3_R5A_EVIDENCE_AUDIT';REAL=R.parent/'RD3_REAL_FOUR_GROUP_R4/analysis'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
BLOCKED=[]
class BlockModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):
   BLOCKED.append(fullname);raise ImportError('INDEPENDENT_CPU_NO_MODEL_IMPORT')
sys.meta_path.insert(0,BlockModels())
def permitted(p):
 if isinstance(p,int):return True
 try:return Path(os.fsdecode(p)).resolve().is_relative_to(R)
 except (ValueError,TypeError):return False

def guard(event,args):
 if event=='subprocess.Popen':BLOCKED.append(event);raise RuntimeError('CPU_VALIDATOR_NO_SUBPROCESS')
 if event=='open':
  p,mode,flags=args
  if ((isinstance(mode,str) and any(x in mode for x in 'wa+x')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) and not permitted(p):BLOCKED.append(str(p));raise PermissionError('VALIDATOR_WRITES_ONLY_R5B')
 if event=='ctypes.dlopen' and args[0] and any(x in str(args[0]).lower() for x in ('libcuda','libcudart','libtorch','libalinpu','ppu_sdk')):BLOCKED.append(event);raise RuntimeError('VALIDATOR_DEVICE_LIBRARY_BLOCKED')
sys.addaudithook(guard)
import numpy as np
CHECKS=[];SOURCE_HASHES=[];PK=[];ERRORS=[]
CONDS=['F','G-B','G-C','E'];PAIRS={'U_B_given_C':('G-B','F','B|C'),'U_C_given_B':('G-C','F','C|B'),'U_B_given_A':('E','G-C','B|A'),'U_C_given_A':('E','G-B','C|A')};CATS=['CLEAR_BENEFICIAL','CLEAR_HARMFUL','TRADEOFF','COUNT_NEUTRAL']
SAMPLES={324:(28,30,6),424:(29,30,7),725:(93,31,5),354:(28,60,6),464:(29,70,10),765:(93,71,6),384:(28,90,7),484:(29,90,9)}
def check(ok,name,details=None):
 item={'name':name,'passed':bool(ok)}
 if details is not None:item['details']=details
 CHECKS.append(item)
 if not ok:ERRORS.append(item);raise RuntimeError('R5B_INVARIANT_FAILED '+name+' '+str(details))
def j(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def csvrows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def threshold(r):return int(round(float(r['iou_threshold'])*100))
def truth(v):check(v in ('True','False'),'valid_boolean');return v=='True'
def classify(tp,fp):
 if tp==0 and fp==0:return 'COUNT_NEUTRAL'
 if tp>=0 and fp<=0:return 'CLEAR_BENEFICIAL'
 if tp<=0 and fp>=0:return 'CLEAR_HARMFUL'
 return 'TRADEOFF'
def unique(name,keys,expected):
 check(len(keys)==expected and len(set(keys))==expected,'primary_key_unique_'+name);PK.append({'file':name,'row_count':len(keys),'expected_row_count':expected,'unique':True})
def array_hash(x):return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+np.ascontiguousarray(x).tobytes()).hexdigest()

def run():
 # R5A archive checksums anchor basis documents and mappings independently.
 sums={x.split('  ',1)[1]:x.split('  ',1)[0] for x in (A/'SHA256SUMS.txt').read_text().splitlines()}
 for n in ['R5A_SAMPLE_CONDITION_COUNTS.csv','R5A_GT_IDENTITY_MAP.csv','R5A_CSV_SCHEMA_AUDIT.json','R5A_SOURCE_INVENTORY.csv']:check(sha(A/n)==sums[n],'R5A_basis_hash_'+n)
 selected=[x for x in j(A/'R5A_CSV_SCHEMA_AUDIT.json')['versions'] if x['selected_for_audit']]
 for x in selected:
  h=sha(x['absolute_path']);check(h==x['sha256'],'original_CSV_hash_'+Path(x['absolute_path']).name);SOURCE_HASHES.append({'path':x['absolute_path'],'expected_R5A_sha256':x['sha256'],'actual_sha256':h,'passed':True})
 check(len(selected)==4,'four_original_CSVs')
 inv={x['absolute_path']:x['sha256'] for x in csvrows(A/'R5A_SOURCE_INVENTORY.csv')};basis=csvrows(A/'R5A_SAMPLE_CONDITION_COUNTS.csv');gm={(int(x['sample_idx']),int(x['gt_idx'])):x for x in csvrows(A/'R5A_GT_IDENTITY_MAP.csv')};check(len(gm)==56,'GT_physical_map_56_frame_observations')
 originals={};states={};physical={};paths={}
 for x in basis:
  sid=int(x['sample_idx']);c=x['condition'];scene,frame,gt=SAMPLES[sid];p=Path(x['source_path']);check(sha(p)==inv[str(p)],'REAL_npz_source_hash_'+str((sid,c)));paths[sid,c]=str(p)
  with np.load(p,allow_pickle=False) as z:
   canonical=z['GT_canonical'];preds=len(z['final_scores']);check(canonical.shape==(gt,8,3),'raw_GT_shape_'+str((sid,c)));record={'GT_count':gt,'final_prediction_count':preds}
   for gi in range(gt):check(array_hash(canonical[gi])==gm[sid,gi]['GT_hash'],'raw_GT_physical_identity_'+str((sid,c,gi)));physical[sid,gi]=gm[sid,gi]['physical_object_id']
   for t in [50,70]:
    m=z[f'prediction_gt_idx{t}'];flags=z[f'gt_tp{t}'];good=m[m>=0];check(len(np.unique(good))==len(good),'raw_match_is_unique_'+str((sid,c,t)));expected_flags=np.zeros(gt,np.int8);expected_flags[good]=1;check(np.array_equal(flags,expected_flags),'raw_match_GT_flags_'+str((sid,c,t)));check(np.array_equal(z[f'prediction_fp{t}'],m<0),'raw_FP_flags_'+str((sid,c,t)))
    tp=int(len(good));fp=preds-tp;fn=gt-tp
    for k,v in [(f'TP{t}',tp),(f'FP{t}',fp),(f'FN{t}',fn)]:check(v==int(x[k]),'R5A_raw_counts_'+str((sid,c,k)));record[k]=v
    states[sid,c,t]=flags.copy()
   originals[sid,c]=record
 check(set(originals)=={(sid,c) for sid in SAMPLES for c in CONDS},'exact_real_sample_condition_coverage')
 countrows=csvrows(R/'R5B_SAMPLE_CONDITION_COUNTS.csv');unique('R5B_SAMPLE_CONDITION_COUNTS.csv',[(int(x['sample_idx']),x['condition']) for x in countrows],32)
 for r in countrows:
  sid=int(r['sample_idx']);c=r['condition'];scene,frame,gt=SAMPLES[sid];check((int(r['scene_id']),int(r['scene_frame']),int(r['GT_count']))==(scene,frame,gt),'B1_identity_'+str((sid,c)));check(r['source_file']==paths[sid,c],'B1_real_source_path_'+str((sid,c)));key=json.loads(r['source_record_key']);check(key=={'sample_idx':sid,'condition':c},'B1_source_record_key')
  for k,v in originals[sid,c].items():check(int(r[k])==v,'B1_count_'+str((sid,c,k)))
  for t in [50,70]:check(int(r[f'TP{t}'])+int(r[f'FN{t}'])==gt and int(r[f'TP{t}'])+int(r[f'FP{t}'])==int(r['final_prediction_count']),'B1_count_conservation_'+str((sid,c,t)))
 for c,wanted in {'F':(45,18,11),'G-B':(40,18,16),'G-C':(42,16,14),'E':(39,13,17)}.items():check(tuple(sum(originals[sid,c][k] for sid in SAMPLES) for k in ['TP70','FP70','FN70'])==wanted,'four_group_IoU70_expected_'+c)
 oldm={(int(x['sample_idx']),x['utility']):x for x in csvrows(REAL/'AGENT_MARGINAL_UTILITY.csv') if x['scope']=='sample'}
 utility=csvrows(R/'R5B_SAMPLE_AGENT_UTILITY.csv');unique('R5B_SAMPLE_AGENT_UTILITY.csv',[(int(x['sample_idx']),x['contrast']) for x in utility],32);um={(int(x['sample_idx']),x['contrast']):x for x in utility};deltas={};classes={}
 for sid in SAMPLES:
  for contrast,(before,after,oldlabel) in PAIRS.items():
   r=um[sid,contrast];check((r['condition_before'],r['condition_after'])==(before,after),'B2_direction_'+str((sid,contrast)))
   for t in [50,70]:
    ds={}
    for metric in ['TP','FP','FN']:
     k=metric+str(t);v=originals[sid,after][k]-originals[sid,before][k];ds[metric]=v;check(int(r['delta_'+k])==v==int(oldm[sid,oldlabel][k]),'B2_delta_and_original_R4_'+str((sid,contrast,k)));check(int(r['before_'+k])==originals[sid,before][k] and int(r['after_'+k])==originals[sid,after][k],'B2_raw_before_after_'+str((sid,contrast,k)))
    deltas[sid,contrast,t]=ds;category=classify(ds['TP'],ds['FP']);classes[sid,contrast,t]=category;check(r['class'+str(t)]==category,'B2_exact_class_'+str((sid,contrast,t)));check(ds['FN']==-ds['TP'],'B2_delta_FN_identity')
   check(truth(r['classification_changed_at_IoU50_vs70'])==(classes[sid,contrast,50]!=classes[sid,contrast,70]),'B2_threshold_class_change')
 for dt,df in itertools.product(range(-10,11),repeat=2):check(classify(dt,df) in CATS,'independent_classification_grid')
 check(classify(1,1)=='TRADEOFF' and classify(-1,-1)=='TRADEOFF','tradeoff_must_not_be_clear_harmful')
 events=csvrows(R/'R5B_GT_TRANSITION_AUDIT.csv');unique('R5B_GT_TRANSITION_AUDIT.csv',[(int(x['sample_idx']),x['contrast'],x['physical_object_id'],threshold(x)) for x in events],448);evgroups=collections.defaultdict(list);rawtrans={(int(x['sample_idx']),x['condition_from'],x['condition_to'],int(x['gt_idx']),int(round(float(x['iou_threshold'])*100))):x for x in csvrows(REAL/'PER_GT_TRANSITIONS.csv')}
 for r in events:
  sid=int(r['sample_idx']);gi=int(r['gt_idx']);name=r['contrast'];t=threshold(r);before,after,_=PAIRS[name];db=int(states[sid,before,t][gi]);da=int(states[sid,after,t][gi]);typ={(0,0):'FN_TO_FN',(0,1):'FN_TO_TP',(1,0):'TP_TO_FN',(1,1):'TP_TO_TP'}[db,da]
  check(r['physical_object_id']==physical[sid,gi] and r['GT_hash']==gm[sid,gi]['GT_hash'],'B3_physical_GT_not_anchor_'+str((sid,gi)));check((int(r['D_before']),int(r['D_after']),r['transition_state'],int(r['delta_GT_TP']))==(db,da,typ,da-db),'B3_raw_GT_transition_'+str((sid,name,gi,t)))
  sourcekey=json.loads(r['source_transition_row_key']);key=(sid,sourcekey['condition_from'],sourcekey['condition_to'],gi,t);check(key in rawtrans and rawtrans[key]['GT_hash']==gm[sid,gi]['GT_hash'],'B3_original_transition_CSV_key');check(r['source_file_before']==paths[sid,before] and r['source_file_after']==paths[sid,after],'B3_raw_source_paths');evgroups[sid,name,t].append(r)
 summaries=csvrows(R/'R5B_GT_TRANSITION_SUMMARY.csv');unique('R5B_GT_TRANSITION_SUMMARY.csv',[(int(x['sample_idx']),x['contrast'],threshold(x)) for x in summaries],64);sm={};expected_cases=set()
 for r in summaries:
  sid=int(r['sample_idx']);name=r['contrast'];t=threshold(r);rows=evgroups[sid,name,t];counts=collections.Counter(x['transition_state'] for x in rows);gt=SAMPLES[sid][2];check(len(rows)==gt,'B3_complete_GT_count_'+str((sid,name,t)))
  for k in ['FN_TO_TP','TP_TO_FN','TP_TO_TP','FN_TO_FN']:check(int(r[k])==counts[k],'B3_transition_count_'+str((sid,name,t,k)))
  dt=counts['FN_TO_TP']-counts['TP_TO_FN'];check(dt==deltas[sid,name,t]['TP'] and -dt==deltas[sid,name,t]['FN'],'B3_gains_minus_losses_equals_delta_TP');check(sum(counts.values())==gt,'B3_four_states_partition_GT')
  hidden=dt>=0 and counts['TP_TO_FN']>0;turn=dt==0 and counts['FN_TO_TP']>0 and counts['TP_TO_FN']>0;strict=dt>0 and counts['TP_TO_FN']>0
  for k,want in [('HIDDEN_GT_REGRESSION',hidden),('GT_IDENTITY_TURNOVER',turn),('strict_net_TP_increase_with_GT_loss',strict)]:check(truth(r[k])==want,'B3_hidden_turnover_definition_'+k)
  check(r['sample_classification']==classes[sid,name,t],'B3_sample_classification')
  losskeys={x['scene_object_key'] for x in rows if x['transition_state']=='TP_TO_FN'};gainkeys={x['scene_object_key'] for x in rows if x['transition_state']=='FN_TO_TP'};check(set(json.loads(r['lost_physical_GT_keys']))==losskeys and set(json.loads(r['gained_physical_GT_keys']))==gainkeys,'B3_physical_loss_gain_ID_lists')
  for e in rows:check(truth(e['is_hidden_loss_event'])==(hidden and e['transition_state']=='TP_TO_FN') and truth(e['is_turnover_changed_GT_event'])==(turn and e['transition_state'] in ['FN_TO_TP','TP_TO_FN']),'B3_hidden_events_are_distinct_from_case_class')
  sm[sid,name,t]={'counts':counts,'hidden':hidden,'turnover':turn,'strict':strict}
  if classes[sid,name,t] in ['CLEAR_HARMFUL','TRADEOFF'] or hidden:expected_cases.add((sid,name,t))
 for t in [50,70]:
  for name in PAIRS:check(sum(len(evgroups[sid,name,t]) for sid in SAMPLES)==56,'B3_every_contrast_threshold_56_GT')
 cases=csvrows(R/'R5B_NEGATIVE_AND_TRADEOFF_CASES.csv');casekeys=[(int(x['sample_idx']),x['contrast'],threshold(x)) for x in cases];check(set(casekeys)==expected_cases and len(casekeys)==len(expected_cases),'negative_tradeoff_hidden_case_set_exact');unique('R5B_NEGATIVE_AND_TRADEOFF_CASES.csv',casekeys,len(expected_cases))
 for r in cases:
  sid=int(r['sample_idx']);name=r['contrast'];t=threshold(r);labels=[]
  if classes[sid,name,t]=='CLEAR_HARMFUL':labels.append('SAMPLE_CLEAR_HARMFUL')
  if classes[sid,name,t]=='TRADEOFF':labels.append('SAMPLE_TRADEOFF')
  if sm[sid,name,t]['hidden']:labels.append('HIDDEN_GT_REGRESSION')
  check(json.loads(r['case_dimensions'])==labels,'different_negative_dimensions_not_collapsed')
 interaction=csvrows(R/'R5B_SAMPLE_SCENE_INTERACTION.csv');unique('R5B_SAMPLE_SCENE_INTERACTION.csv',[(x['aggregation_level'],x['sample_idx'],x['scene_id'],threshold(x)) for x in interaction],24);ivals={};oldi={(x['scope'],x['sample_idx'],x['scene_id']):x for x in csvrows(REAL/'AGENT_INTERACTION.csv')}
 for t in [50,70]:
  for sid in SAMPLES:
   for metric in ['TP','FP','FN']:
    k=metric+str(t);v=originals[sid,'F'][k]-originals[sid,'G-C'][k]-originals[sid,'G-B'][k]+originals[sid,'E'][k];ivals[sid,t,metric]=v
    check(v==deltas[sid,'U_B_given_C',t][metric]-deltas[sid,'U_B_given_A',t][metric]==deltas[sid,'U_C_given_B',t][metric]-deltas[sid,'U_C_given_A',t][metric],'B4_both_context_difference_identities')
 for r in interaction:
  t=threshold(r);level=r['aggregation_level'];sids=[int(r['sample_idx'])] if level=='sample' else [sid for sid in SAMPLES if str(SAMPLES[sid][0])==r['scene_id']] if level=='scene' else list(SAMPLES);check(bool(sids),'B4_aggregation_keys')
  for metric in ['TP','FP','FN']:
   expected=sum(ivals[sid,t,metric] for sid in sids);check(int(r['I_BC_'+metric])==expected,'B4_aggregate_'+str((level,r['sample_idx'],r['scene_id'],t,metric)))
   if level!='all_8_frames':check(expected==int(oldi[level,r['sample_idx'],r['scene_id']][metric+str(t)]),'B4_existing_R4_interaction')
   if level=='sample':
    den=sum(abs(ivals[sid,t,metric]) for sid in SAMPLES);check(int(r['abs_I_'+metric])==abs(expected) and int(r['abs_I_'+metric+'_denominator'])==den,'B4_absolute_interaction_magnitude')
    if den:check(abs(float(r['abs_I_'+metric+'_share'])-abs(expected)/den)<1e-12,'B4_absolute_share')
    else:check(r['abs_I_'+metric+'_share']=='NOT_APPLICABLE','B4_zero_denominator_share_NA')
  ip=sum(originals[sid,'F']['final_prediction_count']-originals[sid,'G-C']['final_prediction_count']-originals[sid,'G-B']['final_prediction_count']+originals[sid,'E']['final_prediction_count'] for sid in sids);check(int(r['I_prediction_count'])==ip and int(r['I_BC_FP'])==ip-int(r['I_BC_TP']) and int(r['I_BC_FN'])==-int(r['I_BC_TP']),'B4_FP_prediction_count_identity_not_GT_FP_tracking')
 check(tuple(sum(ivals[sid,70,m] for sid in SAMPLES) for m in ['TP','FP','FN'])==(2,-3,-2),'B4_required_global_interaction_plus2_minus3_minus2')
 concentration=csvrows(R/'R5B_INTERACTION_CONCENTRATION.csv')
 for r in concentration:
  t=threshold(r);metric=r['metric'];values={sid:ivals[sid,t,metric] for sid in SAMPLES};pos=sum(v for v in values.values() if v>0);neg=sum(v for v in values.values() if v<0);check((int(r['positive_sum']),int(r['negative_sum']),int(r['net']),int(r['absolute_sample_interaction_sum']))==(pos,neg,pos+neg,pos-neg),'B4_gross_positive_negative_cancellation')
  scenes={s:sum(v for sid,v in values.items() if SAMPLES[sid][0]==s) for s in [28,29,93]};check(int(r['nonzero_sample_count'])==sum(v!=0 for v in values.values()) and int(r['nonzero_scene_sum_count'])==sum(v!=0 for v in scenes.values()),'B4_nonzero_sample_scene_counts')
 patterns=csvrows(R/'R5B_GT_FOUR_CONDITION_PATTERNS.csv');unique('R5B_GT_FOUR_CONDITION_PATTERNS.csv',[(int(x['sample_idx']),x['physical_object_id'],threshold(x)) for x in patterns],112);pmap={}
 for r in patterns:
  sid=int(r['sample_idx']);gi=int(r['gt_idx']);t=threshold(r);de=int(states[sid,'E',t][gi]);gb=int(states[sid,'G-B',t][gi]);gc=int(states[sid,'G-C',t][gi]);f=int(states[sid,'F',t][gi]);ig=f-gc-gb+de
  check(r['physical_object_id']==physical[sid,gi] and r['GT_hash']==gm[sid,gi]['GT_hash'],'B5_GT_physical_identity');check(tuple(int(r[k]) for k in ['D_E','D_GB','D_GC','D_F'])==(de,gb,gc,f) and r['D_GB_definition']=='G-B = A+C' and r['D_GC_definition']=='G-C = A+B','B5_no_B_C_name_swap');check(r['pattern_bits']==''.join(str(v) for v in [de,gb,gc,f]) and int(r['I_GT'])==ig,'B5_pattern_and_interaction')
  lossc=[name for name,(b,a,_) in PAIRS.items() if int(states[sid,a,t][gi])-int(states[sid,b,t][gi])<0];check(truth(r['REMOTE_ADDITION_LOSS'])==bool(lossc) and json.loads(r['loss_contrasts'])==lossc,'B5_remote_addition_loss_definition');check(truth(r['FULL_ONLY_TP'])==(f==1 and de==gb==gc==0),'B5_full_only_TP');conditional=(f-gb)!=(gc-de) or (f-gc)!=(gb-de);check(truth(r['CONDITIONAL_DETECTION_CHANGE'])==conditional==(ig!=0),'B5_conditional_final_detection_not_mechanism');pmap[sid,r['physical_object_id'],t]=r
 for r in interaction:
  t=threshold(r);selected=[p for p in patterns if threshold(p)==t and (r['aggregation_level']=='all_8_frames' or r['aggregation_level']=='sample' and p['sample_idx']==r['sample_idx'] or r['aggregation_level']=='scene' and p['scene_id']==r['scene_id'])];check(sum(int(p['I_GT']) for p in selected)==int(r['I_BC_TP']),'B5_GT_sum_equals_sample_scene_global_TP_interaction')
 distribution=csvrows(R/'R5B_PATTERN_DISTRIBUTION.csv');unique('R5B_PATTERN_DISTRIBUTION.csv',[(x['pattern_bits'],threshold(x)) for x in distribution],32)
 for r in distribution:
  t=threshold(r);selected=[p for p in patterns if threshold(p)==t and p['pattern_bits']==r['pattern_bits']];check(int(r['GT_frame_observation_count'])==len(selected) and int(r['TP_interaction_contribution'])==sum(int(p['I_GT']) for p in selected),'B5_pattern_counts_and_contributions');check(int(r['distinct_scene_object_count'])==len({(p['scene_id'],p['physical_object_id']) for p in selected}),'B5_physical_ID_dedup_pattern_count')
 for t in [50,70]:check(sum(int(r['GT_frame_observation_count']) for r in distribution if threshold(r)==t)==56 and sum(int(r['TP_interaction_contribution']) for r in distribution if threshold(r)==t)==sum(ivals[sid,t,'TP'] for sid in SAMPLES),'B5_all16_patterns_partition56_and_interaction')
 sensitivity=csvrows(R/'R5B_THRESHOLD_SENSITIVITY.csv');unique('R5B_THRESHOLD_SENSITIVITY.csv',[(int(x['sample_idx']),x['contrast']) for x in sensitivity],32)
 for r in sensitivity:
  sid=int(r['sample_idx']);name=r['contrast'];check(truth(r['classification_changed'])==(classes[sid,name,50]!=classes[sid,name,70]),'B6_class_sensitivity');check(truth(r['CLEAR_HARMFUL_changed'])==((classes[sid,name,50]=='CLEAR_HARMFUL')!=(classes[sid,name,70]=='CLEAR_HARMFUL')),'B6_CLEAR_HARMFUL_sensitivity');check(truth(r['HIDDEN_GT_REGRESSION_changed'])==(sm[sid,name,50]['hidden']!=sm[sid,name,70]['hidden']),'B6_hidden_sensitivity');check(truth(r['GT_IDENTITY_TURNOVER_changed'])==(sm[sid,name,50]['turnover']!=sm[sid,name,70]['turnover']),'B6_turnover_sensitivity')
 patternchanges=csvrows(R/'R5B_GT_PATTERN_THRESHOLD_COMPARISON.csv');unique('R5B_GT_PATTERN_THRESHOLD_COMPARISON.csv',[(int(x['sample_idx']),x['physical_object_id']) for x in patternchanges],56)
 for r in patternchanges:
  sid=int(r['sample_idx']);oid=r['physical_object_id'];p50=pmap[sid,oid,50];p70=pmap[sid,oid,70];check(r['pattern50']==p50['pattern_bits'] and r['pattern70']==p70['pattern_bits'] and truth(r['pattern_changed'])==(p50['pattern_bits']!=p70['pattern_bits']),'B6_GT_pattern_sensitivity')
 allold=j(R/'R5B_HISTORY_READ_ONLY_SEAL.json')['files'];changed=[p for p,h in allold.items() if not Path(p).is_file() or sha(p)!=h];check(not changed,'all_historical_sources_unchanged',changed)
 check(not BLOCKED and 'torch' not in sys.modules and 'opencood' not in sys.modules,'no_model_device_import_or_execution');check('analyze_r5b' not in sys.modules,'independent_validator_did_not_import_producer')
 result={'state':'R5B_PASS','passed':True,'check_count':len(CHECKS),'passed_check_count':sum(x['passed'] for x in CHECKS),'failed_check_count':0,'errors':[],'missing':[],'not_verifiable':[],'CSV_source_SHA256':SOURCE_HASHES,'primary_key_checks':PK,'sample_condition_coverage':{'samples':8,'conditions':4,'records':32,'complete':True},'GT_identity_mapping_validation':{'frame_GT_observations':56,'distinct_scene_object_ids':len({(r['scene_id'],r['physical_object_id']) for r in gm.values()})},'marginal_arithmetic_consistent':True,'GT_transition_consistent':True,'GT_interaction_equals_sample_interaction':True,'scene_and_all8_sums_consistent':True,'correct_IoU50_70_flags_used':True,'original_sources_verified_unchanged':len(allold),'new_forward':'NO','CUDA_initialized':'NO','history_modified':'NO','torch_imported':False,'opencood_imported':False,'R5C_executed':False,'independent_of_producer_module':True,'source_code_sha256':sha(__file__),'producer_code_sha256':sha(R/'analyze_r5b.py'),'checks':CHECKS}
 (R/'R5B_INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('R5B_PASS_INDEPENDENT_VALIDATION',len(CHECKS),'checks; failed=0; no torch/no forward',flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:
  result={'state':'R5B_FAIL','passed':False,'passed_check_count':sum(x['passed'] for x in CHECKS),'failed_check_count':max(1,len(ERRORS)),'errors':ERRORS or [{'error':str(e)}],'traceback':traceback.format_exc(),'checks':CHECKS,'new_forward':'NO','CUDA_initialized':'NO','history_modified':'NO'};(R/'R5B_INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');raise
