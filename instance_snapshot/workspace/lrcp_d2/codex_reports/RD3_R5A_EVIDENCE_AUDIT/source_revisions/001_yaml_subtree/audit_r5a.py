"""RD3-R5A offline evidence audit. Imports no torch, LRCP or historic pipeline.
Every output is confined to this newly created directory. Source bytes stay read-only.
"""
import os,sys,json,csv,hashlib,copy,collections,itertools,traceback,shutil,zipfile,importlib.abc,datetime,ast
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent;REPORTS=ROOT.parent
R2=REPORTS/'RD3_REAL_FOUR_GROUP_R2';R3=REPORTS/'RD3_SCHEMA_RECOVERY_R3';R4=REPORTS/'RD3_REAL_FOUR_GROUP_R4';OLD=REPORTS/'RD3_MESSAGE_UTILITY_PILOT'
FROZEN=OLD/'FROZEN_MANIFEST.json';FH='66b1411eeb8fad939c448881a03f919484e03fc8d57be83066c38c35463e2c6c';CH='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'
CKPT=Path('/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth')
CORE=['AGENT_MARGINAL_UTILITY.csv','AGENT_INTERACTION.csv','PER_GT_TRANSITIONS.csv','NMS_STAGE_ANALYSIS.csv']
CONDS={'F':[True,True,True],'G-B':[True,False,True],'G-C':[True,True,False],'E':[True,False,False]}
MARGINAL={'B|C':('F','G-B'),'C|B':('F','G-C'),'B|A':('G-C','E'),'C|A':('G-B','E')}
REFERENCE={324:(28,30,6),424:(29,30,7),725:(93,31,5),354:(28,60,6),464:(29,70,10),765:(93,71,6),384:(28,90,7),484:(29,90,9)}
EXPECTED={c:dict(zip(['final_predictions','NMS_overlap_removed','TP50','FP50','FN50','TP70','FP70','FN70','local_AP50','local_AP70'],v)) for c,v in {
 'F':[63,254,47,16,9,45,18,11,.8343240758,.7901104667],
 'G-B':[58,221,43,15,13,40,18,16,.7655817414,.7031124363],
 'G-C':[58,229,43,15,13,42,16,14,.7655068212,.7397427458],
 'E':[52,180,40,12,16,39,13,17,.7138501742,.6955574913]}.items()}
DENIED=[]
class NoDeviceImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):
   DENIED.append('import:'+fullname);raise ImportError('R5A_MODEL_DEVICE_IMPORT_FORBIDDEN '+fullname)
sys.meta_path.insert(0,NoDeviceImports())
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True

def path_allowed(p):
 if isinstance(p,int):return True
 try:return Path(os.fsdecode(p)).resolve().is_relative_to(ROOT)
 except (TypeError,ValueError):return False

def audit_guard(event,args):
 if event=='subprocess.Popen':DENIED.append(event);raise RuntimeError('R5A_SUBPROCESS_FORBIDDEN')
 if event=='open':
  p,mode,flags=args;writing=(isinstance(mode,str) and any(c in mode for c in ('w','a','+','x'))) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not path_allowed(p):DENIED.append('write:'+str(p));raise PermissionError('R5A_WRITE_OUTSIDE_REPORT_DIRECTORY')
 if event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime') and not path_allowed(args[0]):DENIED.append(event);raise PermissionError('R5A_MUTATION_OUTSIDE_REPORT_DIRECTORY')
 if event in ('os.rename','os.replace') and (not path_allowed(args[0]) or not path_allowed(args[1])):DENIED.append(event);raise PermissionError('R5A_RENAME_OUTSIDE_REPORT_DIRECTORY')
 if event=='ctypes.dlopen' and args[0] and any(s in str(args[0]).lower() for s in ('libcuda','libcudart','libtorch','libalinpu','ppu_sdk')):DENIED.append(event);raise RuntimeError('R5A_DEVICE_LIBRARY_FORBIDDEN')
sys.addaudithook(audit_guard)
import numpy as np
import yaml
from offline_storage_reader import load_storage_archive,snapshot,digest,TensorArray
from offline_science import require,polys,overlap,matrix,unique_match,global_ap,nms_replay,stages,canonical_gt_ids,STAGES,CAUSES
INVENTORY={};SCHEMAS={};ISSUES=[];PR={};DATA={};METADATA={};COUNTS=[];GT_MAP=[];RECORD_AUDIT=[];INPUT_AUDIT=[];CSV_CHECKS=[]

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write(n,d):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r));p=ROOT/n
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def register(p,kind,stage='',selected=True,reason='',record=''):
 p=Path(p).resolve();require(p.is_file(),'MISSING_SOURCE '+str(p));key=str(p)
 if key not in INVENTORY:INVENTORY[key]={'absolute_path':key,'file_size_bytes':p.stat().st_size,'sha256':sha(p),'kind':kind,'source_stage':stage,'selected':selected,'selection_basis':reason,'scientific_record':record,'rows':'','columns':''}
 elif record and record not in INVENTORY[key]['scientific_record']:INVENTORY[key]['scientific_record']+=';'+record
 return INVENTORY[key]
def same(a,b,reason,tol=0):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape,'SHAPE '+reason)
 require(np.isfinite(a).all() and np.isfinite(b).all(),'NONFINITE '+reason)
 require(np.all(np.abs(a.astype(float)-b.astype(float))<=tol),'VALUE '+reason+' maxdiff='+str(float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))))
def scalar(observed,expected,reason,tol=0):require(abs(float(observed)-float(expected))<=tol,reason+' observed='+str(observed)+' expected='+str(expected))
def csv_read(path):
 with Path(path).open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
 return fields,rows

def csv_schema(path,selected):
 fields,rows=csv_read(path);name=Path(path).name
 if name.endswith(CORE[0]):key=['scope','scene_id','sample_idx','utility'];grain='sample×four marginal contrasts plus scene sums (32 + 12)'
 elif name.endswith(CORE[1]):key=['scope','scene_id','sample_idx'];grain='sample interaction plus scene sums (8 + 3)'
 elif name.endswith(CORE[2]):key=['sample_idx','condition_from','condition_to','gt_idx','iou_threshold'];grain='sample × six ordered display pairs × physical canonical GT × two IoU thresholds (672)'
 else:key=['record_type','sample_idx','condition','gt_idx','iou_threshold'];grain='tagged union: 32 condition_summary rows + 448 GT_stage rows'
 effective=[x for x in key if x in fields];keys=[tuple(row.get(k,'') for k in effective) for row in rows];duplicates=len(keys)-len(set(keys));issues=[]
 definitions={}
 for k in fields:
  try:definitions[k]=field_definition(k,name)
  except RuntimeError:
   if selected:raise
   definitions[k]='LEGACY_VERSION_NOT_INTERPRETED: excluded placeholder/incomplete schema; not used to infer science'
   issues.append({'field':k,'status':'AMBIGUOUS_IN_EXCLUDED_LEGACY_SCHEMA'})
 if selected:
  require(duplicates==0,'CSV_COMPOSITE_KEY_DUPLICATE '+str(path))
  require(len(effective)==len(key),'CSV_REQUIRED_KEY_FIELDS '+str(path))
  for row in rows:
   require(all(row.get(k) is not None for k in fields),'CSV_TRUNCATED_ROW '+str(path))
  if 'scope' in fields:require({r['scope'] for r in rows}=={'sample','scene'},'CSV_SCOPES')
 rowtypes=dict(collections.Counter(r.get('record_type',r.get('scope','event')) for r in rows))
 return {'absolute_path':str(path),'selected_for_audit':selected,'file_size_bytes':Path(path).stat().st_size,'sha256':sha(path),'columns':fields,'column_count':len(fields),'data_row_count':len(rows),'primary_key':effective,'duplicate_key_count':duplicates,'primary_key_unique':duplicates==0,'data_granularity':grain,'variant_row_counts':rowtypes,'field_definitions':definitions,'field_definition_source':str(R4/'analyze_cpu.py'),'scope_specific_blanks_are_structural':True,'all_rows_included':True,'missing_or_ambiguous_fields':issues}

def field_definition(k,name):
 exact={'scope':'sample rows or sums over frozen samples in one scene','sample_idx':'frozen sample identity; intentionally blank on scene aggregate rows','scene_id':'frozen scene identity, not an independent GT count','scene_frame':'original frame index from immutable frozen manifest','condition':'F=ABC, G-B=AC, G-C=AB, E=A; fixed physical slots A/B/C=0/1/2','condition_from':'first condition in combinations([F,G-B,G-C,E],2); direction is explicit, not a causal claim','condition_to':'second displayed condition in the unordered-pair enumeration','utility':'B|C=F−G-B; C|B=F−G-C; B|A=G-C−E; C|A=G-B−E','gt_idx':'index in sample-specific GT_canonical, never an unconstrained global object ID','GT_hash':'pilot digest(dtype, shape, bytes) of canonical 8×3 GT corners; sample-scoped geometry identity, validated against physical object_id','iou_threshold':'0.5 or 0.7 native float32 BEV polygon IoU, not volumetric 3D IoU','transition':'source and target unique-GT matching status, TP→TP/TP→FN/FN→TP/FN→FN','record_type':'condition_summary or GT_stage tagged union; other-variant columns are intentionally empty','matched_anchor_changed':'both conditions match this GT but use different raw anchor IDs','suppressor_is_from_unique_match':'source-condition suppressor of target matched anchor equals source-condition uniquely matched anchor; association only','candidate_availability':'first stage lacking any high-IoU candidate; surviving availability does not imply unique TP or causal attribution','native_unique_TP':'whether this canonical GT obtained a unique final prediction at this threshold','native_match_anchor':'raw anchor ID of final matched prediction; −1 for unmatched','native_match_IoU':'BEV IoU of native unique matched prediction; 0 when unmatched','native_match_score':'score of unique matched prediction; 0 when unmatched','nms_keep_anchor_ids':'JSON list mapping native NMS keep order back to raw anchor IDs','final_anchor_ids':'JSON list mapping final-range prediction order to raw anchor IDs','suppressor_links':'JSON mapping removed raw anchor ID to its first native-NMS suppressor raw anchor ID','NMS_overlap_suppressed_count':'count(suppressor>=0); excludes top1000 truncation, score/quality/range filtering','top1000_truncated_count':'quality-pass count minus nms_top1000 count','raw_decoded_count':'all decoded anchor rows before score filtering (12800)','score_pass_count':'score>0.2 retained raw anchor IDs','quality_pass_count':'score pass then original large_mask AND z_mask','nms_top1000_count':'quality-pass scores.argsort()[::-1][:1000] length','native_nms_keep_count':'native rotated BEV NMS IoU>0.15 survivors before final range','final_prediction_count':'predictions surviving score/quality/NMS/final range'}
 if k in exact:return exact[k]
 if k.startswith(('TP50','FP50','FN50','TP70','FP70','FN70')):return ('unweighted component delta according to utility column' if 'MARGINAL' in name else 'F−G-C−G-B+E component interaction')+'; FN=GT−TP, FP larger means more false positives'
 if k.startswith('change_class'):return 'existing R4 sign summary over (ΔTP,−ΔFP,−ΔFN); definition audited only; no new R5B classification'
 if k.startswith('loss'):return 'delta/interaction in counts of unmatched GT grouped by first-stage availability; survives_final_but_unmatched covers unique-matching loss; not a causal count'
 if k.startswith(('best_raw_IoU_sum','best_final_IoU_sum')):return 'delta/interaction of per-GT maximum BEV IoU sums over all raw or final anchor IDs'
 if k.startswith('matched_'):return 'delta/interaction of IoU or score summed over unique GT matches at named threshold'
 if k.startswith(('best_IoU_','high_IoU_count_')):return 'maximum BEV IoU / number of IoU>=threshold candidates for this GT at named nested stage'
 if k.startswith(('from_','to_')):
  if 'candidate_availability' in k:return 'named-condition per-GT first missing-stage availability, not unique matching and not causal attribution'
  if 'best_IoU_' in k:return 'named-condition maximum BEV IoU at indicated captured stage'
  if 'best_raw_anchor' in k:return 'named-condition argmax over all raw anchors; first index wins ties'
  if 'best_raw_IoU' in k or 'best_final_IoU' in k:return 'named-condition per-GT maximum raw/final BEV IoU'
  if 'matched_anchor' in k:return 'named-condition uniquely matched final raw anchor ID or −1'
  if 'matched_IoU' in k:return 'named-condition unique matched BEV IoU, or 0'
  if 'matched_score' in k:return 'named-condition unique matched score, or 0'
  if 'raw_best_score' in k:return 'score at the named-condition best-raw-IoU anchor'
  if k.startswith('to_TP_anchor_from_'):return 'look up TARGET matched raw anchor ID in SOURCE-condition deletion/score/IoU/suppressor arrays; the same anchor index is not a fixed physical predicted object or causal intervention'
  if 'suppressor_IoU_same_GT' in k:return 'source suppressor BEV IoU against this same canonical GT, or 0'
 raise RuntimeError('UNDEFINED_CSV_FIELD '+name+':'+k)

def discover_sources():
 reportpaths=[R2/'RD3_R2_FINAL_REPORT.md',R3/'RD3_R3_FINAL_REPORT.md',R4/'R4_PREPARATION_REPORT.md',R4/'analysis/FINAL_SCIENTIFIC_REPORT.md',R4/'analysis/INDEPENDENT_REAL_VALIDATION.json',R4/'analysis/ANALYSIS_SUMMARY.json',R4/'analyze_cpu.py',R4/'dataset_cpu.py',R4/'validate_cpu.py',R4/'finalize_cpu.py',R4/'reused_records/index.json',R3/'recovered_candidates/sample_424_F/RECOVERY_MANIFEST.json',R3/'CANDIDATE_005_RECOVERY_VALIDATION.json',R3/'ACCEPTED_324_REUSE_VALIDATION.json',R4/'R4_NEW_RUN_BUDGET.json',R2/'R2_RUN_BUDGET.json',R4/'R4_EXECUTION_PLAN_27.json',R4/'real_run/manifest.json',FROZEN,CKPT,CKPT.parent/'config.yaml']
 for p in reportpaths:register(p,'identity_report_script_or_ledger','historical',reason='actual source/provenance referenced by R4 final collection')
 native=[Path('/workspace/LRCP/opencood/utils')/n for n in ['common_utils.py','box_utils.py','eval_utils.py']]+[Path('/workspace/LRCP/opencood/data_utils/post_processor')/n for n in ['base_postprocessor.py','voxel_postprocessor.py']]+[REPORTS/'P5_B0_C2_20261031_cpu'/n for n in ['contracts.py','minimal_collector.py','pilot_common.py']]
 for p in native:register(p,'source_definition','native_or_archival',reason='read as text only; field/math definitions independently reimplemented')
 for p in sorted(REPORTS.rglob('*.csv')):
  if p.is_relative_to(ROOT) or not any(p.name.endswith(n) for n in CORE):continue
  selected=p.parent==R4/'analysis' and p.name in CORE
  sch=csv_schema(p,selected);SCHEMAS[str(p)]=sch
  reason='selected: exact paths written by sealed R4 analyze_cpu.py, tied to final 32-real-record validation and report' if selected else 'excluded historical version: failed/incomplete RD3 or R2 run; NOT_ESTIMABLE placeholders/empty event table, not accepted 32-real dataset'
  inv=register(p,'core_csv_version',p.parent.name,selected,reason);inv.update(rows=sch['data_row_count'],columns=sch['column_count'])
 for n in CORE:
  p=R4/'analysis'/n;require(str(p) in SCHEMAS,'MISSING_SOURCE '+str(p));dest=ROOT/'original_csv'/n;shutil.copy2(p,dest);require(sha(dest)==sha(p),'ORIGINAL_CSV_COPY_NOT_EXACT')
 for n in ['FOUR_GROUP_REAL_METRICS.csv','PER_SAMPLE_METRICS.csv','GLOBAL_PR_DATA.json']:register(R4/'analysis'/n,'reported_statistics','R4_final',reason='secondary saved-statistics/report layer, never used to create independent raw counts')
 # Verify sealed code as actual files, without importing it.
 for root,name in [(R2,'SOURCE_AND_CHECKPOINT_SEAL.json'),(R3,'R3_SOURCE_SEAL.json'),(R4,'R4_SOURCE_AND_CHECKPOINT_SEAL.json')]:
  p=root/name;register(p,'source_seal','historical',reason='version anchors for native execution and final CSV generator');d=load(p)
  entries=d.get('execution_sources',d.get('final_source_hashes',{}))
  for path,h in entries.items():require(sha(path)==h,'SEALED_SOURCE_CHANGED '+path)
 return load(R4/'analysis/INDEPENDENT_REAL_VALIDATION.json')

def runtime_expected(contract):
 d=copy.deepcopy(contract)
 for v in d['leaves'].values():
  if v['kind']=='tensor':v['device']='cuda:0'
 d['hash']=digest({'nodes':d['nodes'],'leaves':d['leaves']});return d

def check_actual_inputs(row,meta,candidate,before,contract):
 archived,reader=load_storage_archive(candidate/'inputs_after.pt');actual=snapshot(archived);expected=runtime_expected(contract)
 require(meta['input_leaf_before']==expected,'ACTUAL_BEFORE_NORMATIVE_CONTRACT')
 require(meta['input_leaf_after']==actual,'ACTUAL_AFTER_SERIALIZED_VALUES_METADATA')
 for path,v in expected['leaves'].items():require(actual['leaves'].get(path)==v,'ORIGINAL_LEAF_MUTATED '+path)
 require(set(actual['leaves'])-set(expected['leaves'])=={'/ego/label_dict/offset'},'ONLY_LABEL_OFFSET_ADDITION')
 for path,v in expected['nodes'].items():
  wanted=copy.deepcopy(v)
  if path=='/ego/label_dict':wanted['keys'].append(['str',repr('offset')])
  require(actual['nodes'].get(path)==wanted,'ORIGINAL_CONTAINER_ORDER_TYPE '+path)
 require(set(actual['nodes'])==set(expected['nodes']),'UNEXPECTED_CONTAINER')
 ego=archived['ego'];want=np.concatenate([np.asarray(v['offset']).reshape(1,6400,1,2) for v in ego['calibrate_data'].values()],axis=2)
 same(ego['label_dict']['offset'],want,'NATIVE_LABEL_OFFSET',0)
 for slot in (1,2):require(list(ego['calibrate_data'][slot]['lidar_history'][0])==['voxel_features','voxel_coords','voxel_num_points'],'HISTORY_CONTAINER_DERIVED_LEAK')
 INPUT_AUDIT.append({'sample_idx':row['sample_idx'],'condition':meta['condition'],'source_path':str(candidate/'inputs_after.pt'),'actual_after_sha256':sha(candidate/'inputs_after.pt'),'all_original_leaf_values_shapes_dtypes_devices_and_container_order_independently_read':True,'original_leaf_count':len(expected['leaves']),'Tensor_leaf_count':sum(v['kind']=='tensor' for v in expected['leaves'].values()),'allowed_new_fields':['/ego/label_dict/offset'],'original_history_three_fields_unchanged':True,'reader':reader,'no_torch_import':True})
 return archived

def prove_provenance(rec,row):
 sid=rec['sample_idx'];cond=rec['condition'];pair=f'{sid}/{cond}';data=Path(rec['data_path']);candidate=Path(rec['candidate_path']);stage=rec['source_stage']
 require(data.is_dir() and candidate.is_dir(),'MISSING_CORE_RECORD '+pair)
 cm=load(candidate/'manifest.json');register(candidate/'manifest.json','raw_candidate_manifest',stage,record=pair)
 require(cm['state']=='UNACCEPTED' and cm['stage']=='COLLECTED' and set(cm['files'])=={'raw_arrays.npz','raw_meta.json','inputs_after.pt','arrays.npz','meta.json'},'COMPLETE_SCIENTIFIC_CANDIDATE '+pair)
 for name,h in cm['files'].items():
  register(candidate/name,'raw_scientific_candidate_file',stage,record=pair);require(sha(candidate/name)==h,'RAW_CANDIDATE_HASH '+pair+'/'+name)
 for name in ('arrays.npz','meta.json','inputs_after.pt'):
  register(data/name,'final_scientific_record_file',stage,record=pair);require(sha(data/name)==cm['files'][name],'SCIENTIFIC_REFERENCE_BYTES '+pair+'/'+name)
 meta=load(data/'meta.json');raw=load(candidate/'raw_meta.json');require(meta['backend']=='REAL' and meta['sample_idx']==sid and meta['condition']==cond,'REAL_DATA_NOT_MOCK_OR_RELABELLED '+pair)
 require((meta['scene_id'],meta['scene_frame'])==(row['scene_id'],row['scene_frame']) and meta['input_manifest']==row,'SCENE_FRAME_INPUT_IDENTITY '+pair)
 require(meta['checkpoint_sha256']==CH and meta['frozen_sha256']==FH and meta['input_hash']==row['input']['input_hash'],'CHECKPOINT_FROZEN_INPUT_IDENTITY '+pair)
 require(meta['keep']==CONDS[cond] and meta['slot_physical_ids']==['1','2','3'],'PHYSICAL_SLOT_CONDITION '+pair)
 require(raw['model_returned'] is True and raw['sample_idx']==sid and raw['input_leaf_before']==meta['input_leaf_before'] and raw['input_leaf_after']==meta['input_leaf_after'],'DURABLE_RAW_RETURN '+pair)
 root=R4 if stage=='R4_NEW_REAL_ACCEPTED' else R2
 sealpath=root/('R4_SOURCE_AND_CHECKPOINT_SEAL.json' if root==R4 else 'SOURCE_AND_CHECKPOINT_SEAL.json');require(meta['source_seal_sha256']==raw['source_seal_sha256']==sha(sealpath),'EXECUTION_SOURCE_SEAL '+pair)
 budget=load(root/('R4_NEW_RUN_BUDGET.json' if root==R4 else 'R2_RUN_BUDGET.json'));ordinal=raw['attempt']['ordinal'];attempt=budget['attempts'][ordinal-1]
 require((attempt['sample_idx'],attempt['condition'])==(sid,cond) and attempt['kind']=='REAL' and attempt['entered_epoch'] is not None and attempt['returned_epoch']>=attempt['entered_epoch'],'ENTERED_RETURNED_IDENTITY '+pair)
 if stage=='R3_RECOVERY_REFERENCE_FROM_R2_UNACCEPTED':
  require((sid,cond)==(424,'F') and ordinal==5 and root==R2 and attempt['status']=='MODEL_RETURNED' and attempt['committed_epoch'] is None,'RECOVERY_NOT_PROMOTED_IN_R2')
  rm=load(data/'RECOVERY_MANIFEST.json');register(data/'RECOVERY_MANIFEST.json','R3_recovery_reference',stage,record=pair)
  require(rm['source_path']==str(candidate) and rm['source_R2_state_still']=='UNACCEPTED' and rm['original_manifest_sha256']==sha(candidate/'manifest.json')==sha(data/'original_manifest.json'),'RECOVERY_MANIFEST_CHAIN')
  require(sha(rm['R3_validation_path'])==rm['R3_validation_sha256'] and sha(R3/'validate_cpu.py')==rm['validator_source_sha256'],'R3_INDEPENDENT_RECOVERY_HASH')
  for name,h in rm['files_verified_copied'].items():require(sha(data/name)==h==sha(candidate/name),'RECOVERY_COPY_HASH')
  original_accept='R3 independently verified recovery reference; source R2 remains UNACCEPTED/committed=0'
 else:
  m=load(data/'manifest.json');register(data/'manifest.json','accepted_capture_manifest',stage,record=pair)
  require(m['state']=='ACCEPTED' and m['backend']=='REAL' and sha(candidate/'manifest.json')==m['candidate_manifest_sha256'],'ACCEPTED_CAPTURE_CHAIN '+pair)
  require((root/'real_run'/m['candidate']).resolve()==candidate.resolve(),'ACCEPTED_CANDIDATE_PATH')
  for name,h in m['files'].items():require(sha(data/name)==h,'ACCEPTED_FILE_HASH '+pair)
  require(sha(m['validation_path'])==m['validation_sha256'],'INDIVIDUAL_VALIDATION_HASH '+pair)
  val=load(m['validation_path']);require(val['passed'] and val['backend']=='REAL' and (val['sample_idx'],val['condition'])==(sid,cond) and val['candidate_manifest_sha256']==sha(candidate/'manifest.json'),'INDIVIDUAL_VALIDATION_IDENTITY '+pair)
  register(m['validation_path'],'independent_acceptance_certificate',stage,record=pair);require(attempt['committed_epoch'] is not None,'PERSISTENT_COMMITTED '+pair);original_accept='ACCEPTED with matching immutable candidate and independent certificate'
 # Final R4 validator certificate is authenticated by its source hash; raw science is recalculated separately.
 cert=rec['validation'];require(cert['passed'] and cert['backend']=='REAL' and cert['candidate_manifest_sha256']==sha(candidate/'manifest.json') and cert['validator_source_sha256']==sha(R4/'validate_cpu.py'),'FINAL_CERTIFICATE_IDENTITY '+pair)
 runtime=load(root/'real_run/NATIVE_ENTRY_RUNTIME.json');require(meta['native_entry']==runtime,'NATIVE_RUNTIME_IDENTITY '+pair)
 require(runtime['model_class']=='opencood.models.point_pillar_deformable.PointPillarDeformable' and runtime['MetaFlow_class']=='opencood.models.meta_flow.MetaFlow' and runtime['mock_modules_present'] is False,'NATIVE_MODEL_CLASS_NOT_MOCK')
 require(sha(runtime['extension_path'])==runtime['extension_sha256'],'NATIVE_KERNEL_HASH')
 resources=load(root/'real_run/resources.json');worker=load(root/'real_run/worker_claim.json');require(resources['worker_pid']==worker['pid'] and resources['device']['uuid']=='GPU-019e1215-c270-091c-0000-0000c0610973' and resources['device']['name']=='PPU-ZW810E','ACTUAL_DEVICE_WORKER_PROVENANCE')
 for name in ('NATIVE_ENTRY_RUNTIME.json','resources.json','worker_claim.json'):register(root/'real_run'/name,'runtime_identity',stage,record=pair)
 return meta,{'sample_idx':sid,'condition':cond,'source_stage':stage,'source_path':str(data),'candidate_path':str(candidate),'model_attempt_ordinal':ordinal,'original_acceptance':original_accept,'source_manifest_sha256':sha(candidate/'manifest.json'),'arrays_sha256':sha(data/'arrays.npz'),'input_after_sha256':sha(candidate/'inputs_after.pt'),'persistent_entered_returned_verified':True,'R4_final_certificate_verified':True,'actual_model_forward_in_R5A':0}


def check_attention(z,meta):
 keep=CONDS[meta['condition']];q=[0,1620,3240,4860,6399];ref=z['reference_full'];require(ref.shape==(1,6400,9,2),'REFERENCE_SHAPE')
 for slot in (1,2):
  off=z[f'metaflow_slot_{slot}'];same(z['pred_offset'][:,:,slot-1,:],off.reshape(1,6400,2),'METAFLOW_PHYSICAL_OFFSET',0)
  wanted=np.broadcast_to(ref[:,:,:1,:]+off.reshape(1,6400,1,2)/np.asarray([80,80],np.float32),(1,6400,3,2));same(ref[:,:,slot*3:(slot+1)*3,:],wanted,'METAFLOW_REFERENCE',1e-6)
 for i in range(4):
  w=z[f'cross_{i}_full_weights'];require(w.shape==(1,6400,8,9,8) and np.isfinite(w).all() and (w>=0).all(),'ALL_BLOCK_HEAD_LEVEL_POINT_ATTENTION')
  same(w.sum((-1,-2)),np.ones((1,6400,8),np.float32),'ATTENTION_NORMALIZATION',1e-5)
  for slot,active in enumerate(keep):
   block=w[:,:,:,slot*3:(slot+1)*3,:]
   require(bool(np.any(block>0)) if active else np.count_nonzero(block)==0,'ACTUAL_AGENT_MASK_IDENTITY')
  raw=z[f'cross_{i}_raw_logits_selected'];require(np.isfinite(raw).all(),'FINITE_RAW_ATTENTION')
  v=raw.reshape(1,len(q),8,5,3,8)[:,:,:,:3,:,:].copy()
  for slot,active in enumerate(keep):
   if not active:v[:,:,:,slot,:,:]=-np.inf
  flat=v.reshape(1,len(q),8,-1);ex=np.exp(flat-flat.max(-1,keepdims=True));expected=(ex/ex.sum(-1,keepdims=True)).reshape(1,len(q),8,9,8)
  same(w[:,q],expected,'ATTENTION_SOFTMAX',1e-6);same(z[f'cross_{i}_weights'],w[:,q],'SELECTED_FULL_ATTENTION',0)
 return True


def check_native_stages(a,config):
 raw=a['raw_corners'];scores=a['scores_all'];ids={k:a['ids_'+k] for k in STAGES};require(raw.shape==(12800,8,3) and scores.shape==(12800,),'COMPLETE_RAW_CANDIDATES')
 same(ids['raw_decoded'],np.arange(12800),'RAW_ANCHOR_IDS')
 for k in STAGES:
  require(len(ids[k])==len(np.unique(ids[k])) and np.all((ids[k]>=0)&(ids[k]<12800)),'UNIQUE_STAGE_IDS '+k)
 for x,y in zip(STAGES,STAGES[1:]):require(set(ids[y])<=set(ids[x]),'NESTED_STAGES')
 sm=scores>np.float32(config['target_args']['score_threshold']);same(a['score_mask'],sm,'NATIVE_SCORE_MASK');same(ids['score_pass'],np.flatnonzero(sm),'SCORE_PASS_IDS')
 boxes=raw[ids['score_pass']];span=boxes.max(1)-boxes.min(1)
 # Mirror the existing source exactly: remove_large_pred_bbx reuses y span as z_len,
 # then logical_and(index,z_len). This audit documents it and never repairs it.
 large=(span[:,0]<=6)&(span[:,1]<=6)&(span[:,1]!=0);zm=(boxes[:,:,2].min(1)>=-3)&(boxes[:,:,2].max(1)<=1)
 same(a['large_mask'],large,'NATIVE_EXISTING_LARGE_MASK');same(a['z_mask'],zm,'NATIVE_Z_MASK');same(a['quality_mask'],large&zm,'QUALITY_MASK')
 quality=ids['score_pass'][large&zm];same(ids['quality_pass'],quality,'QUALITY_IDS');same(a['nms_input_scores'],scores[quality],'NMS_SCORES');same(a['nms_input_corners'],raw[quality],'NMS_GEOMETRY')
 same(ids['nms_top1000'],quality[scores[quality].argsort()[::-1][:1000]],'NMS_SORT_AND_TOP1000')
 keep,sup,ovs=nms_replay(raw[quality],scores[quality],config['nms_thresh']);same(a['native_nms_keep'],keep,'NATIVE_NMS_KEEP');same(ids['nms_keep'],quality[keep],'NMS_KEEP_IDS')
 suppressed=np.full(12800,-1,np.int32);overlaps=np.zeros(12800,np.float32)
 for removed,owner in sup.items():suppressed[quality[removed]]=quality[owner];overlaps[quality[removed]]=ovs[removed]
 same(a['suppressor'],suppressed,'NMS_SUPPRESSOR_IDENTITY');same(a['suppressor_overlap'],overlaps,'NMS_SUPPRESSOR_IOU')
 rangeboxes=raw[ids['nms_keep']];limits=np.asarray(config['gt_range']);rm=((rangeboxes>=limits[:3])&(rangeboxes<=limits[3:])).all(2).sum(1)>=8
 same(a['range_mask'],rm,'NATIVE_POST_NMS_RANGE_MASK');same(ids['final_range'],ids['nms_keep'][rm],'FINAL_IDS');same(a['final_corners'],raw[ids['final_range']],'FINAL_CORNERS_LINEAGE');same(a['final_scores'],scores[ids['final_range']],'FINAL_SCORE_LINEAGE')
 first=np.ones(12800,np.int8)
 for stage,code in zip(STAGES[1:],range(2,7)):first[ids[stage]]=code
 same(a['first_deletion_code'],first,'FIRST_DELETION_CODE')
 return ids

def match_anchor(a,match,gi):
 candidates=np.flatnonzero(match['matched']==gi)
 if not len(candidates):return -1,0.,0.
 pi=int(candidates[0]);anchor=int(a['ids_final_range'][pi]);return anchor,float(a['IoU_GT_raw'][gi,anchor]),float(a['final_scores'][pi])

def vector(a,matches,stageinfo):
 v={}
 for t in (50,70):
  m=matches[t];tp=sum(m['tp']);fp=sum(m['fp']);v.update({f'TP{t}':tp,f'FP{t}':fp,f'FN{t}':len(a['GT_canonical'])-tp})
  losses=collections.Counter(x['availability'] if x['availability']!='high_iou_survives_final' else 'survives_final_but_unmatched' for gi,x in enumerate(stageinfo[t]) if not m['flags'][gi])
  for name in list(CAUSES)+['survives_final_but_unmatched']:v[f'loss{t}_{name}']=losses[name]
  got=[match_anchor(a,m,gi) for gi in range(len(a['GT_canonical']))];v[f'matched_IoU{t}_sum']=sum(x[1] for x in got);v[f'matched_score{t}_sum']=sum(x[2] for x in got)
 v['best_raw_IoU_sum']=sum(float(r.max(initial=0.)) for r in a['IoU_GT_raw']);v['best_final_IoU_sum']=sum(float(r[a['ids_final_range']].max(initial=0.)) for r in a['IoU_GT_raw'])
 return v


def audit_records(final,frozen,config):
 expected={(r['sample_idx'],c) for r in frozen['samples'] for c in CONDS};records=final['records'];pairs=[(r['sample_idx'],r['condition']) for r in records]
 require(len(pairs)==32 and len(set(pairs))==32 and set(pairs)==expected,'EXACT_REAL_8_BY_4_NO_DUPLICATE_OR_MISSING')
 require(final['passed'] and final['state']=='R4_REAL_32_INDEPENDENTLY_ACCEPTED' and not final['missing'],'FINAL_VALIDATION_RECORDED_COMPLETE')
 require(dict(collections.Counter(r['source_stage'] for r in records))=={'R2_ACCEPTED':4,'R3_RECOVERY_REFERENCE_FROM_R2_UNACCEPTED':1,'R4_NEW_REAL_ACCEPTED':27},'FOUR_PLUS_ONE_PLUS_27_SCIENTIFIC_ORIGIN')
 contracts={x['sample_idx']:x['all_original_fields_and_leaves'] for x in load(REPORTS/'RD3_INPUT_INTEGRITY_REPAIR/INPUT_LEAF_CONTRACT.json')['samples']};frozen_payloads={}
 for row in frozen['samples']:
  sid=row['sample_idx'];item=row['input'];register(item['path'],'frozen_actual_input','frozen',record=str(sid));require(sha(item['path'])==item['sha256'],'FROZEN_ARCHIVE_HASH')
  payload,reader=load_storage_archive(item['path']);require(digest(payload['batch'])==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'ACTUAL_FROZEN_INPUT_RNG_VALUES')
  require(snapshot(payload['batch'])==contracts[sid],'ACTUAL_FROZEN_FULL_R1_CONTRACT');frozen_payloads[sid]=payload
 fields=['raw_corners','scores_all','score_mask','large_mask','z_mask','quality_mask','nms_input_corners','nms_input_scores','native_nms_keep','range_mask','final_corners','final_scores','GT_canonical','IoU_GT_raw','suppressor','suppressor_overlap','first_deletion_code','pred_offset','offset_GT','reference_full','metaflow_slot_1','metaflow_slot_2','cls_preds','reg_preds','dir_preds']+['ids_'+s for s in STAGES]+[f'{prefix}{t}' for prefix in ('gt_tp','prediction_gt_idx','prediction_fp') for t in (50,70)]
 for rec in records:
  sid,c=rec['sample_idx'],rec['condition'];row=next(r for r in frozen['samples'] if r['sample_idx']==sid);meta,proof=prove_provenance(rec,row);candidate=Path(rec['candidate_path']);batch=frozen_payloads[sid]['batch'];after=check_actual_inputs(row,meta,candidate,batch,contracts[sid])
  with np.load(Path(rec['data_path'])/'arrays.npz',allow_pickle=False) as z:
   require(all(k in z for k in fields),'MISSING_SCIENTIFIC_ARRAY');a={k:z[k].copy() for k in fields};check_attention(z,meta)
  require(all(np.isfinite(v).all() for v in a.values()),'NONFINITE_RAW_ARRAYS')
  same(a['GT_canonical'].shape,[row['GT_count'],8,3],'GT_COUNT_GEOMETRY')
  oldpath=Path('/workspace/lrcp_d2/P3A_20261014_120929_boxes')/f'sample_{sid}_alpha_1.0.npz';register(oldpath,'original_GT_Full_reference','P3-A',record=str(sid))
  with np.load(oldpath,allow_pickle=False) as old:same(a['GT_canonical'],old['gt_boxes'],'P3_CANONICAL_GT',1e-5)
  physical,error=canonical_gt_ids(a['GT_canonical'],batch,config['order']);require(len(physical)==row['GT_count'],'ACTUAL_GT_PHYSICAL_IDENTITIES')
  if c=='F':
   for gi,oid in enumerate(physical):GT_MAP.append({'sample_idx':sid,'scene_id':row['scene_id'],'scene_frame':row['scene_frame'],'gt_idx':gi,'physical_object_id':oid,'GT_hash':digest(a['GT_canonical'][gi]),'GT_corner_digest':digest(a['GT_canonical'][gi]),'all_four_conditions_geometry_and_identity_agree':True,'frozen_input_source':row['input']['path'],'canonical_to_frozen_geometry_max_abs_error':error})
  ids=check_native_stages(a,config);rawiou=matrix(a['GT_canonical'],a['raw_corners']);same(rawiou,a['IoU_GT_raw'],'ALL_RAW_GT_IOU_RECOMPUTE',1e-6)
  require(np.array_equal((rawiou>=.5),(a['IoU_GT_raw']>=.5)) and np.array_equal((rawiou>=.7),(a['IoU_GT_raw']>=.7)),'EXACT_RAW_IOU_THRESHOLD_DECISIONS')
  proof['raw_IoU_max_abs_error']=float(np.max(np.abs(rawiou-a['IoU_GT_raw']),initial=0.));a['IoU_GT_raw']=rawiou
  matches={};stageinfo={}
  for t in (50,70):
   m=unique_match(a['final_corners'],a['final_scores'],a['GT_canonical'],t/100);same(m['matched'],a[f'prediction_gt_idx{t}'],'SAVED_UNIQUE_MATCH_'+str(t));same(m['flags'],a[f'gt_tp{t}'],'SAVED_GT_TP_FLAGS_'+str(t));same((m['matched']<0).astype(np.int8),a[f'prediction_fp{t}'],'SAVED_FP_FLAGS_'+str(t));matches[t]=m;stageinfo[t]=stages(rawiou,ids,t/100)
   destination=PR.setdefault(c,{}).setdefault(t,{'tp':[],'fp':[],'score':[],'gt':0})
   for k in ('tp','fp','score'):destination[k].extend(m[k])
   destination['gt']+=m['gt']
  v=vector(a,matches,stageinfo);actual={'sample_idx':sid,'scene_id':row['scene_id'],'scene_frame':row['scene_frame'],'condition':c,'GT_count':len(physical),'final_predictions':len(a['final_scores']),'NMS_overlap_removed':int((a['suppressor']>=0).sum()),**{k:v[k] for k in ('TP50','FP50','FN50','TP70','FP70','FN70')},'source_stage':rec['source_stage'],'source_path':str(Path(rec['data_path'])/'arrays.npz'),'evidence_level':'RAW_FINAL_BOXES_SCORES_GT_RECOMPUTED','original_input_leaf_values_verified':True,'GT_physical_identity_verified':True,'status':'PASS'}
  for t in (50,70):require(actual[f'TP{t}']+actual[f'FN{t}']==actual['GT_count'] and actual[f'TP{t}']+actual[f'FP{t}']==actual['final_predictions'],'SAMPLE_COUNT_IDENTITIES')
  if c=='F':
   baseline=row['historical_full'];require(sha(baseline['path'])==baseline['sha256'],'ORIGINAL_P4_FULL_HASH');register(baseline['path'],'original_head_NMS_Full_reference','P4-Full',record=str(sid))
   with np.load(baseline['path'],allow_pickle=False) as b:
    for key,tol in [('cls_preds',1e-6),('reg_preds',1e-5),('dir_preds',1e-5),('final_corners',1e-5),('final_scores',1e-6),('native_nms_keep',0)]:same(a[key],b[key],'ORIGINAL_P4_FULL_'+key,tol)
   with np.load(oldpath,allow_pickle=False) as old:same(a['final_corners'],old['pred_boxes'],'ORIGINAL_P3_FULL_BOX',1e-5);same(a['final_scores'],old['pred_scores'],'ORIGINAL_P3_FULL_SCORE',1e-6)
  COUNTS.append(actual);proof.update(input_leaf_actual_read_verified=True,GT_count=len(physical),canonical_to_physical_object_ids=physical,GT_geometry_mapping_error=error,native_NMS_and_suppressors_independently_replayed=True,unique_GT_matching_independently_recomputed=True,attention_all_four_blocks_actual_weights_verified=True,status='PASS');RECORD_AUDIT.append(proof)
  DATA[sid,c]={'a':a,'matches':matches,'stageinfo':stageinfo,'vector':v,'physical_ids':physical,'source':str(Path(rec['data_path'])/'arrays.npz')};METADATA[sid,c]=meta
  print('R5A_RAW_RECORD_PASS',sid,c,'GT',actual['GT_count'],'TP70',actual['TP70'],'FP70',actual['FP70'],flush=True);del after
 for row in frozen['samples']:
  sid=row['sample_idx'];f=DATA[sid,'F'];fm=METADATA[sid,'F']
  for c in CONDS:
   d=DATA[sid,c];m=METADATA[sid,c];require(d['physical_ids']==f['physical_ids'],'CROSS_CONDITION_GT_OBJECT_IDS');same(d['a']['GT_canonical'],f['a']['GT_canonical'],'CROSS_CONDITION_GT_CORNERS')
   for key in ('pred_offset','offset_GT','reference_full','metaflow_slot_1','metaflow_slot_2'):same(d['a'][key],f['a'][key],'UNCHANGED_AGENT_METAFLOW_'+key)
   for key in ('feature_hashes','parameter_digest','buffer_digest','history_integrity','input_leaf_before','input_leaf_after','GT_mapping'):require(m[key]==fm[key],'UNCHANGED_INPUT_AGENT_IDENTITY_'+key)
 require(len(GT_MAP)==56,'CANONICAL_PHYSICAL_GT_MAP_56')
 # Ledger/plan identity uses actual pairs, rather than the number of files.
 plan=load(R4/'R4_EXECUTION_PLAN_27.json');r4pairs={(x['sample_idx'],x['condition']) for x in plan['tasks']};actualnew={(r['sample_idx'],r['condition']) for r in records if r['source_stage']=='R4_NEW_REAL_ACCEPTED'}
 require(actualnew==r4pairs and len(r4pairs)==27,'ACTUAL_R4_PLAN_27_IDENTITY');require(sha(R4/'R4_EXECUTION_PLAN_27.json')==(R4/'R4_EXECUTION_PLAN_27.sha256').read_text().split()[0],'IMMUTABLE_PLAN_HASH')


def audit_core_csvs(frozen):
 for name in CORE:
  path=R4/'analysis'/name;sch=SCHEMAS[str(path)];fields,rows=csv_read(path);checked=0
  if name==CORE[0]:
   for row in rows:
    utility=row['utility'];require(utility in MARGINAL,'MARGINAL_DEFINITION');pos,neg=MARGINAL[utility]
    sids=[int(row['sample_idx'])] if row['scope']=='sample' else [r['sample_idx'] for r in frozen['samples'] if r['scene_id']==int(row['scene_id'])]
    require(bool(sids),'MARGINAL_SCENE_COVERAGE')
    if row['scope']=='sample':require(REFERENCE[sids[0]][0]==int(row['scene_id']),'MARGINAL_SAMPLE_SCENE')
    for key in DATA[sids[0],pos]['vector']:
     expected=sum(DATA[sid,pos]['vector'][key]-DATA[sid,neg]['vector'][key] for sid in sids);scalar(row[key],expected,'CSV_MARGINAL '+str(sids)+utility+key,1e-6 if 'IoU' in key or 'score_sum' in key else 0);checked+=1
   require(len(rows)==44 and sum(r['scope']=='sample' for r in rows)==32 and sum(r['scope']=='scene' for r in rows)==12,'MARGINAL_ROWS_FULL')
   sch['condition_coverage']={'samples':8,'marginal_contrasts_per_sample':list(MARGINAL),'scene_sums':3,'all_required_contrasts_present':True}
  elif name==CORE[1]:
   for row in rows:
    sids=[int(row['sample_idx'])] if row['scope']=='sample' else [r['sample_idx'] for r in frozen['samples'] if r['scene_id']==int(row['scene_id'])]
    for key in DATA[sids[0],'F']['vector']:
     expected=sum(DATA[sid,'F']['vector'][key]-DATA[sid,'G-C']['vector'][key]-DATA[sid,'G-B']['vector'][key]+DATA[sid,'E']['vector'][key] for sid in sids);scalar(row[key],expected,'CSV_INTERACTION '+str(sids)+key,1e-6 if 'IoU' in key or 'score_sum' in key else 0);checked+=1
   require(len(rows)==11 and sum(r['scope']=='sample' for r in rows)==8,'INTERACTION_ROWS_FULL');sch['condition_coverage']={'samples':8,'conditions_in_contrast':list(CONDS),'scene_sums':3,'complete':True}
  elif name==CORE[2]:
   expectedkeys={(r['sample_idx'],p,q,gi,str(t/100)) for r in frozen['samples'] for p,q in itertools.combinations(CONDS,2) for gi in range(r['GT_count']) for t in (50,70)};actualkeys=set()
   for row in rows:
    sid=int(row['sample_idx']);gi=int(row['gt_idx']);t=int(round(float(row['iou_threshold'])*100));p=row['condition_from'];q=row['condition_to'];actualkeys.add((sid,p,q,gi,str(t/100)))
    x=DATA[sid,p];y=DATA[sid,q];xa,xi,xs=match_anchor(x['a'],x['matches'][t],gi);ya,yi,ys=match_anchor(y['a'],y['matches'][t],gi);sup=int(x['a']['suppressor'][ya]) if ya>=0 else -1
    require(int(row['scene_id'])==REFERENCE[sid][0] and int(row['scene_frame'])==REFERENCE[sid][1],'TRANSITION_SAMPLE_SCENE_FRAME')
    require(row['GT_hash']==digest(x['a']['GT_canonical'][gi])==digest(y['a']['GT_canonical'][gi]),'TRANSITION_STABLE_GT_HASH')
    transition=('TP' if xa>=0 else 'FN')+'→'+('TP' if ya>=0 else 'FN');require(row['transition']==transition,'RAW_GEOMETRY_GT_TRANSITION')
    values={'from_matched_anchor':xa,'to_matched_anchor':ya,'from_matched_IoU':xi,'to_matched_IoU':yi,'from_matched_score':xs,'to_matched_score':ys,'from_best_raw_anchor':int(np.argmax(x['a']['IoU_GT_raw'][gi])),'to_best_raw_anchor':int(np.argmax(y['a']['IoU_GT_raw'][gi])),'from_best_raw_IoU':float(x['a']['IoU_GT_raw'][gi].max()),'to_best_raw_IoU':float(y['a']['IoU_GT_raw'][gi].max()),'from_raw_best_score':float(x['a']['scores_all'][int(np.argmax(x['a']['IoU_GT_raw'][gi]))]),'to_raw_best_score':float(y['a']['scores_all'][int(np.argmax(y['a']['IoU_GT_raw'][gi]))]),'to_TP_anchor_from_deletion_code':int(x['a']['first_deletion_code'][ya]) if ya>=0 else -1,'to_TP_anchor_from_score':float(x['a']['scores_all'][ya]) if ya>=0 else 0.,'to_TP_anchor_from_IoU_same_GT':float(x['a']['IoU_GT_raw'][gi,ya]) if ya>=0 else 0.,'to_TP_anchor_from_NMS_suppressor':sup,'from_suppressor_IoU_same_GT':float(x['a']['IoU_GT_raw'][gi,sup]) if sup>=0 else 0.}
    for side,d in [('from',x),('to',y)]:
     info=d['stageinfo'][t][gi];require(row[side+'_candidate_availability']==info['availability'],'TRANSITION_STAGE_AVAILABILITY');values[side+'_best_final_IoU']=info['best']['final_range']
     for stage in STAGES:values[side+'_best_IoU_'+stage]=info['best'][stage]
    for k,v in values.items():scalar(row[k],v,'TRANSITION_FIELD '+str((sid,p,q,gi,t))+k,1e-6 if ('IoU' in k or 'score' in k) else 0);checked+=1
    require(row['matched_anchor_changed']==str(xa>=0 and ya>=0 and xa!=ya) and row['suppressor_is_from_unique_match']==str(sup>=0 and sup==xa),'TRANSITION_BOOLEAN_DEFINITION')
   require(actualkeys==expectedkeys and len(rows)==672,'EXACT_672_GT_TRANSITION_COVERAGE');sch['condition_coverage']={'pairs':[list(x) for x in itertools.combinations(CONDS,2)],'thresholds':[.5,.7],'GT':56,'complete':True};sch['GT_identity_mapping']='sample_idx + gt_idx + GT_hash checked against equal four-condition corners and raw frozen physical object_id; explicit mapping in R5A_GT_IDENTITY_MAP.csv'
  else:
   summary=set();gtkeys=set()
   for row in rows:
    sid=int(row['sample_idx']);c=row['condition'];d=DATA[sid,c];a=d['a'];require(int(row['scene_id'])==REFERENCE[sid][0],'NMS_SCENE')
    if row['record_type']=='condition_summary':
     summary.add((sid,c));wanted={k:len(a['ids_'+stage]) for k,stage in [('raw_decoded_count','raw_decoded'),('score_pass_count','score_pass'),('quality_pass_count','quality_pass'),('nms_top1000_count','nms_top1000'),('native_nms_keep_count','nms_keep'),('final_prediction_count','final_range')]};wanted.update(NMS_overlap_suppressed_count=int((a['suppressor']>=0).sum()),top1000_truncated_count=len(a['ids_quality_pass'])-len(a['ids_nms_top1000']))
     for k,v in wanted.items():scalar(row[k],v,'NMS_SUMMARY '+str((sid,c))+k);checked+=1
     require(json.loads(row['nms_keep_anchor_ids'])==a['ids_nms_keep'].tolist() and json.loads(row['final_anchor_ids'])==a['ids_final_range'].tolist(),'NMS_FINAL_ORDERED_IDS')
     expected={str(i):int(a['suppressor'][i]) for i in np.flatnonzero(a['suppressor']>=0)};require(json.loads(row['suppressor_links'])==expected,'NMS_SUPPRESSOR_LINKS_CSV')
    else:
     require(row['record_type']=='GT_stage','NMS_RECORD_VARIANT');gi=int(row['gt_idx']);t=int(round(float(row['iou_threshold'])*100));gtkeys.add((sid,c,gi,t));anchor,iou,score=match_anchor(a,d['matches'][t],gi);info=d['stageinfo'][t][gi]
     values={'native_unique_TP':int(anchor>=0),'native_match_anchor':anchor,'native_match_IoU':iou,'native_match_score':score}
     for stage in STAGES:values['best_IoU_'+stage]=info['best'][stage];values['high_IoU_count_'+stage]=info['counts'][stage]
     for k,v in values.items():scalar(row[k],v,'NMS_GT_STAGE '+str((sid,c,gi,t))+k,1e-6 if ('IoU' in k and not k.startswith('high_')) or k=='native_match_score' else 0);checked+=1
     require(row['candidate_availability']==info['availability'],'NMS_GT_AVAILABILITY')
   expectedSummary={(r['sample_idx'],c) for r in frozen['samples'] for c in CONDS};expectedGT={(r['sample_idx'],c,gi,t) for r in frozen['samples'] for c in CONDS for gi in range(r['GT_count']) for t in (50,70)}
   require(summary==expectedSummary and gtkeys==expectedGT and len(rows)==480,'NMS_EXACT_32_PLUS_448_COVERAGE');sch['condition_coverage']={'condition_summaries':32,'per_GT_condition_threshold_rows':448,'all_four_conditions_complete':True};sch['GT_identity_mapping']='gt_idx is canonical and derives stable GT_hash/physical object_id via sample and R5A_GT_IDENTITY_MAP.csv'
  sch['actual_source_fields_reconciled']=checked;sch['all_rows_values_reconciled_with_raw_science']=True;sch['status']='PASS';CSV_CHECKS.append({'file':str(path),'rows':len(rows),'fields_checked':checked,'status':'PASS'})
  print('R5A_ORIGINAL_CSV_ALL_ROWS_RECONCILED',name,len(rows),checked,flush=True)

def reconcile_groups():
 reported={r['condition']:r for r in csv_read(R4/'analysis/FOUR_GROUP_REAL_METRICS.csv')[1]};sample_reported={(int(r['sample_idx']),r['condition']):r for r in csv_read(R4/'analysis/PER_SAMPLE_METRICS.csv')[1]};reconciliation=[];aggregates={};pr_output={}
 for count in COUNTS:
  saved=sample_reported[count['sample_idx'],count['condition']]
  for k in ('TP50','FP50','FN50','TP70','FP70','FN70','final_predictions'):scalar(saved[k],count[k],'SAVED_PER_SAMPLE_COUNT_SUM_CHECK '+k)
  scalar(saved['nms_overlap_removed'],count['NMS_overlap_removed'],'SAVED_PER_SAMPLE_NMS_SUM_CHECK')
 for c in CONDS:
  selected=[r for r in COUNTS if r['condition']==c];v={k:sum(r[k] for r in selected) for k in ('TP50','FP50','FN50','TP70','FP70','FN70','final_predictions','NMS_overlap_removed','GT_count')};v.update(sample_count=len(selected),scene_count=len({r['scene_id'] for r in selected}));pr_output[c]={}
  for t in (50,70):
   require(v['GT_count']==56 and v[f'TP{t}']+v[f'FN{t}']==56,'GT_56_DENOMINATOR');v[f'Precision{t}']=v[f'TP{t}']/(v[f'TP{t}']+v[f'FP{t}']) if v['final_predictions'] else 0.;v[f'Recall{t}']=v[f'TP{t}']/56
   ap,mrec,mprec,order=global_ap(PR[c][t]);v[f'local_AP{t}']=ap;pr_output[c][str(t)]={'statistic_from_independent_final_geometry_matches':PR[c][t],'globally_sorted_indices':order,'recall_with_endpoints':mrec,'precision_envelope':mprec,'AP':ap,'not_mean_of_frame_AP':True}
  aggregates[c]=v
  for key,observed in v.items():
   full_expected=float(reported[c][key]);rounded_expected=EXPECTED[c].get(key,full_expected);tol=1e-9 if key.startswith('local_AP') else 1e-12 if key.startswith(('Precision','Recall')) else 0
   scalar(observed,rounded_expected,'USER_EXPECTED_BASELINE '+c+key,tol);scalar(observed,full_expected,'R4_FULL_PRECISION_METRIC '+c+key,1e-12 if tol else 0)
   evidence='RAW_FINAL_BOXES_SCORES_GT_GREEDY_MATCH_AND_GLOBAL_PR_RECOMPUTED'
   if key=='NMS_overlap_removed':evidence='RAW_QUALITY_PASS_CANDIDATES_NATIVE_EQUIVALENT_NMS_SHADOW_REPLAY'
   if key in ('GT_count','sample_count','scene_count'):evidence='FROZEN_IDENTITY_AND_ACTUAL_CANONICAL_GT_RECOMPUTED'
   reconciliation.append({'condition':c,'metric':key,'expected_value':rounded_expected,'observed_value':observed,'difference':observed-rounded_expected,'reported_full_precision_value':full_expected,'difference_vs_R4_full_precision':observed-full_expected,'comparison_tolerance':tol,'source_path':json.dumps([r['source_path'] for r in selected]),'expected_source_path':str(R4/'analysis/FOUR_GROUP_REAL_METRICS.csv'),'evidence_level':evidence,'saved_per_sample_statistics_also_reconciled':True,'status':'PASS'})
 table('R5A_FOUR_GROUP_RECONCILIATION.csv',reconciliation);write('R5A_INDEPENDENT_GLOBAL_PR.json',pr_output)
 return aggregates,reconciliation


def reconcile_interactions(frozen):
 result=[];by_sample={};source=R4/'analysis/AGENT_INTERACTION.csv';old_inter=csv_read(source)[1];old_util=csv_read(R4/'analysis/AGENT_MARGINAL_UTILITY.csv')[1]
 labels={'B|C':'Delta_B_given_C','C|B':'Delta_C_given_B','B|A':'Delta_B_given_A','C|A':'Delta_C_given_A'}
 for row in frozen['samples']:
  sid=row['sample_idx'];one={'scope':'sample','sample_idx':sid,'scene_id':row['scene_id'],'GT_count':row['GT_count']}
  for label,(pos,neg) in MARGINAL.items():
   for metric in ('TP70','FP70','FN70'):one[labels[label]+'_'+metric]=DATA[sid,pos]['vector'][metric]-DATA[sid,neg]['vector'][metric]
   require(one[labels[label]+'_FN70']==-one[labels[label]+'_TP70'],'DELTA_FN_EXACT_NEGATIVE_TP')
  for metric in ('TP70','FP70','FN70'):one['I_BC_'+metric]=DATA[sid,'F']['vector'][metric]-DATA[sid,'G-C']['vector'][metric]-DATA[sid,'G-B']['vector'][metric]+DATA[sid,'E']['vector'][metric]
  one.update(evidence_level='RAW_FINAL_BOXES_SCORES_GT_RECOMPUTED_THEN_ARITHMETIC',source_path=json.dumps([DATA[sid,c]['source'] for c in CONDS]),status='PASS');result.append(one);by_sample[sid]=one
 numerical=[k for k in result[0] if k.startswith(('Delta_','I_BC_'))]
 for scene in sorted({r['scene_id'] for r in frozen['samples']}):
  ids=[r['sample_idx'] for r in frozen['samples'] if r['scene_id']==scene];one={'scope':'scene','sample_idx':'','scene_id':scene,'GT_count':sum(REFERENCE[sid][2] for sid in ids),**{k:sum(by_sample[sid][k] for sid in ids) for k in numerical},'evidence_level':'SUM_OF_INDEPENDENT_SAMPLE_ARITHMETIC','source_path':str(ROOT/'R5A_SAMPLE_CONDITION_COUNTS.csv'),'status':'PASS'};result.append(one)
 total={'scope':'all_8_frames','sample_idx':'','scene_id':'ALL','GT_count':56,**{k:sum(r[k] for r in by_sample.values()) for k in numerical},'evidence_level':'SUM_OF_8_INDEPENDENT_SAMPLE_ARITHMETIC','source_path':str(ROOT/'R5A_SAMPLE_CONDITION_COUNTS.csv'),'status':'PASS'};result.append(total)
 wanted={'Delta_B_given_A_TP70':3,'Delta_C_given_A_TP70':1,'Delta_B_given_C_TP70':5,'Delta_C_given_B_TP70':3,'Delta_B_given_A_FP70':3,'Delta_C_given_A_FP70':5,'Delta_B_given_C_FP70':0,'Delta_C_given_B_FP70':2,'I_BC_TP70':2,'I_BC_FP70':-3}
 for k,v in wanted.items():scalar(total[k],v,'EXPECTED_ALL8_ARITHMETIC '+k)
 for row in result:
  require(row['I_BC_FN70']==-row['I_BC_TP70'],'INTERACTION_FN_TP_IDENTITY')
  for prefix in labels.values():require(row[prefix+'_FN70']==-row[prefix+'_TP70'],'AGGREGATE_DELTA_FN_TP')
  if row['scope']=='all_8_frames':continue
  matches=[x for x in old_inter if x['scope']==row['scope'] and x['sample_idx']==str(row['sample_idx']) and x['scene_id']==str(row['scene_id'])];require(len(matches)==1,'EXISTING_INTERACTION_ROW_IDENTITY')
  for metric in ('TP70','FP70','FN70'):scalar(matches[0][metric],row['I_BC_'+metric],'EXISTING_INTERACTION_SCENE_SAMPLE')
  for name,prefix in labels.items():
   matches=[x for x in old_util if x['scope']==row['scope'] and x['sample_idx']==str(row['sample_idx']) and x['scene_id']==str(row['scene_id']) and x['utility']==name];require(len(matches)==1,'EXISTING_MARGINAL_ROW_IDENTITY')
   for metric in ('TP70','FP70','FN70'):scalar(matches[0][metric],row[prefix+'_'+metric],'EXISTING_MARGINAL_SCENE_SAMPLE')
 table('R5A_INTERACTION_RECONCILIATION.csv',result)
 return result,total


def readiness_report():
 items=[
  ('stable_physical_GT','DERIVABLE','AVAILABLE','CSV has sample + gt_idx + GT_hash (transition), stage CSV has sample + gt_idx. Frozen actual object_ids/centers + four-condition equal canonical corners yield independently checked physical mapping.','R5A_GT_IDENTITY_MAP.csv; frozen input archives; GT_canonical'),
  ('TP_FN_status_and_condition_pairs','AVAILABLE','AVAILABLE','672 transition rows include explicit condition_from/to and both IoU thresholds, all recomputed from final boxes.','PER_GT_TRANSITIONS.csv; prediction_gt_idx50/70'),
  ('matched_final_prediction_identity','AVAILABLE','AVAILABLE','Matched raw anchor IDs join to ids_final_range; raw anchor IDs identify grid rows, not physical objects.','from/to_matched_anchor; ids_final_range'),
  ('matched_final_corners','DERIVABLE','AVAILABLE','No corner arrays in CSV; verified raw_corners[anchor] equals final_corners[prediction index] for every record.','arrays.npz: raw_corners, final_corners, ids_final_range'),
  ('matched_final_IoU_and_score','AVAILABLE','AVAILABLE','CSV carries matched IoU/score and native unique mapping; independently recomputed using float32 BEV polygon IoU.','PER_GT_TRANSITIONS.csv; NMS_STAGE_ANALYSIS.csv'),
  ('raw_candidate_stable_identifier','DERIVABLE','AVAILABLE','All 12800 raw anchor rows have complete stage ID maps. Summary CSV includes keep/final IDs and suppressor links, not one row per raw candidate.','arrays.npz: ids_raw_decoded, ids_*'),
  ('raw_corners_and_score','DERIVABLE','AVAILABLE','CSV lacks complete raw coordinates and all scores; all 32 scientific arrays contain raw_corners[12800,8,3] and scores_all[12800].','arrays.npz: raw_corners, scores_all'),
  ('raw_GT_IoU','DERIVABLE','AVAILABLE','CSV stores per-GT maxima/counts only. Complete GT×12800 matrix exists and was independently reconstructed from raw geometry.','arrays.npz: IoU_GT_raw, GT_canonical'),
  ('score_and_quality_filter_state','DERIVABLE','AVAILABLE','CSV gives aggregate counts and selected cross-condition deletion lookups; full masks/IDs/code are in original arrays.','score_mask, large_mask, z_mask, quality_mask, first_deletion_code'),
  ('NMS_keep_and_suppressor_identity','AVAILABLE','AVAILABLE','Summary rows contain ordered keep IDs and every suppressor link; independent NMS replay agrees exactly.','nms_keep_anchor_ids, suppressor_links; suppressor, native_nms_keep'),
  ('suppressor_score_and_rank','DERIVABLE','AVAILABLE','Suppressor score/rank is not a dedicated CSV column. Join anchor ID to scores_all and quality IDs; native descending argsort/reverse/top1000 and strict IoU>0.15 are known.','nms_input_scores, scores_all, ids_quality_pass'),
  ('post_NMS_range_and_final_lineage','DERIVABLE','AVAILABLE','CSV has final anchor list/count. Full range mask and exact final geometry/score link exist.','range_mask, ids_final_range, final_corners, final_scores'),
  ('unique_GT_match_for_each_final_prediction','DERIVABLE','AVAILABLE','GT-stage rows provide matched anchor; all FP/unmatched final predictions are completely recoverable from saved final prediction-to-GT maps and were independently re-matched.','prediction_gt_idx50/70, prediction_fp50/70, gt_tp50/70')]
 write('R5A_GT_NMS_FIELD_AVAILABILITY.json',{'categories':['AVAILABLE','DERIVABLE','MISSING','AMBIGUOUS'],'fields':[{'field':a,'CSV_status':b,'scientific_shards_status':c,'definition':d,'existing_source':e} for a,b,c,d,e in items],'core_missing_evidence':[],'unresolved_ambiguity':[],'R5C_server_side_evidence_sufficient_without_forward':True,'bundle_does_not_include_large_scientific_npz':True})
 lines=['# 逐GT与NMS证据就绪审计','','R5A只检查证据及字段定义，不开展R5B负边际分类或R5C机制归因。服务器现有32份真实科学分片足以支持后续离线取证，无需新forward。四CSV本身不是完整逐候选几何数据集；缺失的展开列均可从已验Hash的NPZ无损提取。','','| 字段类别 | CSV | 32份科学分片 | 既有来源与定义 |','|---|---|---|---|']
 for a,b,c,d,e in items:lines.append(f'| {a} | {b} | {c} | {d} `{e}` |')
 lines+=['','## GT身份与排序','','GT稳定键是sample_idx与已验证的canonical GT映射。PER_GT_TRANSITIONS的GT_hash是dtype/shape/角点字节Hash，不是独立物体ID；NMS表的gt_idx也不是物体ID。已从实际冻结Tensor归档读取object_ids和object_bbx_center，在不导入torch的条件下重建GT角点并与canonical框做1e-5内唯一几何双射；四条件GT角点、物理object_id、输入列表及顺序全部一致。显式映射保存为R5A_GT_IDENTITY_MAP.csv。原生generate_gt_bbx使用set去重，不能未经核验把原始object_ids顺序当作canonical GT顺序。','','## 原生处理顺序与数值语义','','raw decoded 12800 anchors → score>0.2 → large_mask AND z_mask → score.argsort()[::-1][:1000] → rotated BEV NMS（float32 IoU>0.15才抑制）→ 最终三维range过滤（8角点均在边界内）→ 按np.argsort(-final_score)进行贪心唯一GT匹配（IoU>=0.5/0.7）。AP在冻结8帧所有预测Score上全局排序并作VOC2010 PR积分。匹配和NMS使用前4角点BEV面积IoU，不能称为三维体积IoU。原生NMS排序与评价排序的同分处理表达式不同，审计均保留原式。','','remove_large_pred_bbx原有实现的z_len实际重复使用y坐标跨度，且末次logical_and把该跨度按非零值处理；本轮按实际源码复算并确认捕获mask一致，没有修复或改变这一历史算法。另一个z_mask使用真正z最小值>=−3、最大值<=1。stage名称size_or_z_filter概括这两个已定义mask，不能据列名假定额外体积筛选。','','candidate_availability及loss*只描述高IoU候选在阶段的存在性，仍可能survives_final_but_unmatched；不是机制因果结论。to_TP_anchor_from_*是在SOURCE条件按TARGET匹配anchor索引查询，不能自动解释为同一物理预测对象。','','## 后续范围与上传包','','核心字段没有无法恢复的MISSING；字段定义经实际生成脚本核对，没有未解决的AMBIGUOUS。CSV未展开的原始几何、全部分数、抑制者排序及FP匹配均为DERIVABLE，32份NPZ中为AVAILABLE。上传包包含完整四CSV、32条计数和56个GT映射，可用于R5B逐样本算术分析；为控制体积没有打包全部科学NPZ。若在纯上传环境开展R5C几何/NMS机制分析，应按R5A_SOURCE_INVENTORY.csv的已验证路径另行导出相应原始数组，不能把CSV摘要冒充完整几何证据。无需新增GPU forward。']
 (ROOT/'R5A_GT_NMS_EVIDENCE_READINESS.md').write_text('\n'.join(lines)+'\n')
 return items


def create_report(aggregates,total,preserved_count):
 csvpaths=[R4/'analysis'/n for n in CORE];lines=['# RD3-R5A 原始证据与独立科学一致性审计','','最终判定：`R5A_PASS`。可以在不新增GPU forward的条件下进入RD3-R5B；本轮没有启动R5B/R5C。','','## 1. 数据来源与版本','','选择由封存R4 analyze_cpu.py生成、最终报告明确引用、且与32条独立验收记录逐键逐字段一致的R4/analysis四CSV。未根据文件名直接选取。同名旧RD3/R2文件已逐个列入R5A_SOURCE_INVENTORY.csv和R5A_CSV_SCHEMA_AUDIT.json：它们属于失败轮次、NOT_ESTIMABLE占位或空表，未混入科学数据。没有缺失的核心CSV，不需要DERIVED_RECONSTRUCTION。四原始CSV已逐字节复制至original_csv/，内容和所有行不变。','','| 原始CSV绝对路径 | 行 | 列 | 粒度 | SHA256 |','|---|---:|---:|---|---|']
 for p in csvpaths:
  s=SCHEMAS[str(p)];lines.append(f"| {p} | {s['data_row_count']} | {s['column_count']} | {s['data_granularity']} | `{s['sha256']}` |")
 lines+=['','源文件生成关系、全部字段定义、复合键唯一性、逐类型结构性空值及条件覆盖见R5A_CSV_SCHEMA_AUDIT.json。边际表44行=32个sample×contrast加12个scene×contrast；交互11行=8样本加3scene；转移672行=56GT×6条件对×2阈值；NMS480行=32条件摘要加448个GT×condition×threshold。它们不是四份同粒度的32行表。','','## 2. 冻结身份及32条完整性','','冻结8样本/scene/frame/GT均与原始manifest和任务清单一致，GT合计56。manifest SHA256：`'+FH+'`；epoch19 checkpoint SHA256：`'+CH+'`。实际文件存在并已核对。32个(sample,condition)唯一且恰好覆盖8×4，无重复、重命名、互换或遗漏。组成是R2已接受324四条、R3恢复424/F一条、R4新增27条。','','每条均独立校验科学数组、meta、inputs_after及原始候选五文件Hash，核对源执行代码seal、真实entered/returned账本、runtime/device来源、真实backend和独立接受证书。诊断候选的UNACCEPTED状态是原始保存层状态：接受分片通过独立manifest引用它；424/F单独通过R3恢复引用，原R2状态仍MODEL_RETURNED/UNACCEPTED、committed=0，不能因此将其错误视为缺失科学记录。','','本轮用受限标准库pickle/ZIP解析器和NumPy直接读取8份冻结输入与32份实际inputs_after.pt的Tensor存储，未导入torch；数值digest、RNG、R1逐leaf合同、原始类型/顺序/shape/dtype/device标签均重算核对。仅允许原生label_dict.offset新增且精确等于各slot GT offset拼接。历史原始三字段没有新增派生字段或数值变化。归档cuda:0仅作为字符串核验，未分配设备。详见R5A_INPUT_LEAF_AND_GT_MAPPING.json及R5A_OFFLINE_READER_CHECK.json。','','A/B/C保持物理slot0/1/2与ID1/2/3。每条实际四Block、6400query、8head、9level、8point全量Attention被核验，被屏蔽slot严格0，归一化及选定query softmax符合原容差；原始输入、feature Hash、MetaFlow来源/offset/reference、参数/buffer身份在四条件间一致。','','## 3. 独立复算核心指标','','以下均从实际最终框/Score/GT几何重做原生等价的float32 BEV贪心唯一匹配；NMS计数来自quality-pass原始候选的独立NMS重放，不是读取已有汇总后声称复算。','','| 条件 | 预测数 | NMS overlap删除 | TP50/FP50/FN50 | TP70/FP70/FN70 | AP50 | AP70 |','|---|---:|---:|---|---|---:|---:|']
 for c,v in aggregates.items():lines.append(f"| {c} | {v['final_predictions']} | {v['NMS_overlap_removed']} | {v['TP50']}/{v['FP50']}/{v['FN50']} | {v['TP70']}/{v['FP70']}/{v['FN70']} | {v['local_AP50']:.10f} | {v['local_AP70']:.10f} |")
 lines+=['','每组GT=56，Precision/Recall50/70及上表所有指标在R5A_FOUR_GROUP_RECONCILIATION.csv中逐项列出expected/observed/difference/source/evidence_level/status，全部PASS。AP从完整最终预测、Score及GT匹配重新全局排序并作VOC2010 PR积分；不是逐帧AP求和或平均。与用户10位小数基线的比较只容纳其舍入误差（AP 1e-9），与R4全精度AP比较1e-12；原生几何1e-5、Score/softmax1e-6、权重归一化1e-5及所有匹配阈值、NMS keep/GT匹配精确相同的标准没有放宽。','','证据层级明确区分：①原始框/Score/GT独立重算（本报告主结果）；②已保存逐样本计数或标志求和（另行交叉核对，非替代主结果）；③历史报告值对照（预期基线，非独立证据）。R5A_INDEPENDENT_GLOBAL_PR.json保存从几何重算的PR输入/排序/包络，R5A_SCIENTIFIC_RECORD_AUDIT.json保存32条来源与核验。','','## 4. 边际与交互算术','','以下仅作分量算术验收，没有开展负贡献样本分类或机制归因。','','| 分量差分 | ΔTP70 | ΔFP70 | ΔFN70 |','|---|---:|---:|---:|']
 for label,prefix in [('F−G-B（B|C）','Delta_B_given_C'),('F−G-C（C|B）','Delta_C_given_B'),('G-C−E（B|A）','Delta_B_given_A'),('G-B−E（C|A）','Delta_C_given_A')]:lines.append(f"| {label} | {total[prefix+'_TP70']:+d} | {total[prefix+'_FP70']:+d} | {total[prefix+'_FN70']:+d} |")
 lines+=['',f"交互F−G-C−G-B+E：I_BC_TP70={total['I_BC_TP70']:+d}，I_BC_FP70={total['I_BC_FP70']:+d}，I_BC_FN70={total['I_BC_FN70']:+d}。",'', '8样本、3scene及全8帧共12个聚合层级行见R5A_INTERACTION_RECONCILIATION.csv。逐样本和每组TP+FN=GT，所有边际ΔFN=−ΔTP；样本、scene行与原R4 CSV逐项相同，8帧边际及交互精确复现任务预期。FP/FN正差表示错误分量增加，没有加权效用。','','## 5. GT身份与NMS证据就绪','','四条件canonical GT角点逐字节一致，GT_hash重新计算；actual frozen object_ids与center/clean transform独立重建到canonical角点的唯一几何映射在原1e-5内，56个物理GT已保存R5A_GT_IDENTITY_MAP.csv。不能把GT索引或set去重前的object_ids列表顺序直接当物理身份。672个转移行的状态、anchor、IoU、Score和GT_hash以及480个NMS行均与原始数据逐行复核。','','NMS完整原始候选/Score/几何、掩码、keep/suppressor/overlap、top1000排序、最终range和唯一匹配均在32份实际NPZ中可用，本轮核验其完整性并重放处理顺序。四CSV只提供摘要和部分ID链接，未展开字段为DERIVABLE，不伪装成CSV直接AVAILABLE。逐类状态见R5A_GT_NMS_FIELD_AVAILABILITY.json与R5A_GT_NMS_EVIDENCE_READINESS.md。','','定义澄清：原生匹配是BEV IoU，不是三维体积IoU；stage availability不是TP/FN或因果贡献；anchor索引不是物理物体ID；NMS_overlap_removed不包含score、quality、top1000或range删除。原生large-box实现中z_len重复y跨度的历史行为已按源码记录并精确镜像核验，没有修改算法。','','## 6. FAIL、NOT_VERIFIABLE和范围限制','','核心身份/32组合/GT/计数/AP/边际/交互：FAIL=0，NOT_VERIFIABLE=0。没有发现需要调整数据才能解决的冲突，未发现无法恢复的核心缺失或未解决字段歧义。CSV中的variant空白和scene行空sample_idx是结构定义，非记录缺失；完整raw几何/Score不在CSV中但在分片中AVAILABLE，已记录DERIVABLE来源。所有原始CSV和已有结果均未改变。','','R5A没有推断哪个具体GT受到NMS影响，也没有执行R5B负边际分类。8帧只来自3个scene，56个GT不是56个独立实验单位。上传包不含体积较大的原始NPZ；纯上传环境的后续R5C几何分析需要依据现有路径追加相应原始数组，服务器已有全部信息，无需新forward。','','## 7. 只读与执行证明','','新forward：NO；CUDA初始化：NO；历史证据修改：NO。审计进程禁止torch/opencood等模型或设备模块导入、禁止设备动态库及子进程启动，写入被限制为本新目录。只使用标准库、NumPy、Shapely、PyYAML作离线解析/计算。'+str(preserved_count)+'份历史/模型/数据文件在本轮前后SHA256一致。证据见R5A_HISTORY_PRESERVATION_VALIDATION.json、R5A_EXECUTION_GUARD.json。','','## 8. 下一步判定及交付','','现有证据足以在不新增GPU forward的情况下进入RD3-R5B逐样本Remote负边际贡献分析：YES。依据是完整32条件配对、56个跨四条件稳定物理GT、从最终几何独立重算的计数与AP、四种边际和交互的逐样本/scene/总量精确一致，以及四CSV全部行的可信来源和核验。','','本轮仅执行R5A并停止。可上传包为`'+str(ROOT/'R5A_UPLOAD_BUNDLE.zip')+'`，内含四份原始CSV全部行、规定审计材料和SHA256SUMS.txt；附加GT身份映射及执行证明。ZIP自身Hash另见R5A_UPLOAD_BUNDLE.zip.sha256与R5A_BUNDLE_VALIDATION.json，避免将ZIP自身Hash放入其内容形成循环。']
 (ROOT/'RD3_R5A_FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')

def build_bundle():
 names=['original_csv/'+n for n in CORE]+['R5A_SOURCE_INVENTORY.csv','R5A_CSV_SCHEMA_AUDIT.json','R5A_FOUR_GROUP_RECONCILIATION.csv','R5A_SAMPLE_CONDITION_COUNTS.csv','R5A_INTERACTION_RECONCILIATION.csv','R5A_GT_NMS_EVIDENCE_READINESS.md','RD3_R5A_FINAL_REPORT.md','R5A_GT_IDENTITY_MAP.csv','R5A_GT_NMS_FIELD_AVAILABILITY.json','R5A_SCIENTIFIC_RECORD_AUDIT.json','R5A_INPUT_LEAF_AND_GT_MAPPING.json','R5A_INDEPENDENT_GLOBAL_PR.json','R5A_HISTORY_PRESERVATION_VALIDATION.json','R5A_EXECUTION_GUARD.json','R5A_OFFLINE_READER_CHECK.json','audit_r5a.py','offline_storage_reader.py','offline_science.py']
 checks={n:sha(ROOT/n) for n in names};(ROOT/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in checks.items()));names.append('SHA256SUMS.txt')
 dest=ROOT/'R5A_UPLOAD_BUNDLE.zip'
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as z:
  for n in names:z.write(ROOT/n,n)
 with zipfile.ZipFile(dest) as z:
  require(z.testzip() is None,'ZIP_CRC_FAILURE');require(set(z.namelist())==set(names),'ZIP_MEMBER_SET')
  for n,h in checks.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'ZIP_MEMBER_HASH '+n)
  actual={}
  for n in CORE:
   copied=z.read('original_csv/'+n);require(hashlib.sha256(copied).hexdigest()==sha(R4/'analysis'/n),'ZIP_ORIGINAL_BYTES '+n)
   rows=list(csv.DictReader(copied.decode().splitlines()));expected=SCHEMAS[str(R4/'analysis'/n)]['data_row_count'];require(len(rows)==expected,'ZIP_ALL_ORIGINAL_ROWS '+n);actual[n]=len(rows)
 h=sha(dest);(ROOT/'R5A_UPLOAD_BUNDLE.zip.sha256').write_text(h+'  R5A_UPLOAD_BUNDLE.zip\n')
 write('R5A_BUNDLE_VALIDATION.json',{'passed':True,'absolute_path':str(dest),'file_size_bytes':dest.stat().st_size,'sha256':h,'original_CSV_all_rows_included':actual,'original_bytes_unchanged':True,'member_count':len(names),'uncompressed_member_bytes':sum((ROOT/n).stat().st_size for n in names),'SHA256SUMS_all_members_verified':True,'silently_truncated':False,'split_required':False,'raw_scientific_NPZ_included':False})
 return dest,h


def run():
 require(not any(k=='torch' or k.startswith('torch.') or k=='opencood' or k.startswith('opencood.') for k in sys.modules),'FORBIDDEN_MODEL_IMPORT')
 require(sha(FROZEN)==FH and sha(CKPT)==CH,'FROZEN_MANIFEST_OR_CHECKPOINT_HASH_MISMATCH')
 frozen=load(FROZEN);require([r['sample_idx'] for r in frozen['samples']]==list(REFERENCE),'FROZEN_ORDER_IDENTITY')
 for row in frozen['samples']:require((row['scene_id'],row['scene_frame'],row['GT_count'])==REFERENCE[row['sample_idx']],'FROZEN_SCENE_FRAME_GT_IDENTITY')
 require(sum(r['GT_count'] for r in frozen['samples'])==56,'FROZEN_GT_SUM')
 config=yaml.safe_load((CKPT.parent/'config.yaml').read_text())['postprocess'];require(config['order']=='hwl' and config['target_args']['score_threshold']==.2 and config['nms_thresh']==.15,'NATIVE_ORIGINAL_CONFIG')
 final=discover_sources();print('R5A_SOURCES_DISCOVERED',len(INVENTORY),'CSV_versions',len(SCHEMAS),flush=True)
 audit_records(final,frozen,config);audit_core_csvs(frozen)
 # Preserve all eight sample identities and the canonical physical GT join.
 order={r['sample_idx']:i for i,r in enumerate(frozen['samples'])};COUNTS.sort(key=lambda r:(order[r['sample_idx']],list(CONDS).index(r['condition'])))
 table('R5A_SAMPLE_CONDITION_COUNTS.csv',COUNTS);table('R5A_GT_IDENTITY_MAP.csv',GT_MAP)
 aggregates,reconciliation=reconcile_groups();interactions,total=reconcile_interactions(frozen);readiness_report()
 write('R5A_SCIENTIFIC_RECORD_AUDIT.json',{'status':'PASS','record_count':32,'source_counts':dict(collections.Counter(r['source_stage'] for r in RECORD_AUDIT)),'records':RECORD_AUDIT,'CPU_Mock_records_used':0,'new_forward':False,'CUDA_initialization':False})
 write('R5A_INPUT_LEAF_AND_GT_MAPPING.json',{'status':'PASS','method':'restricted ZIP/pickle storage reader plus NumPy; not torch.load, not a model/pipeline','frozen_inputs_read_and_full_R1_snapshot_verified':8,'actual_inputs_after_read_and_values_devices_verified':32,'GT_physical_mappings':56,'records':INPUT_AUDIT,'reader_code_sha256':sha(ROOT/'offline_storage_reader.py')})
 write('R5A_CSV_SCHEMA_AUDIT.json',{'status':'PASS','selection_evidence':'sealed R4 generator + actual final report + authenticated final 32-record mapping + all original rows independently reconciled','selected_paths':[str(R4/'analysis'/n) for n in CORE],'version_count':len(SCHEMAS),'versions':list(SCHEMAS.values()),'selected_CSV_row_checks':CSV_CHECKS,'all_required_fields_defined_from_actual_sources':True,'core_missing_sources':[],'core_unresolved_ambiguities':[],'reconstructed_core_csv_files':[]})
 # Inventory only science and provenance used by this audit. Full old-tree seal is separate.
 for item in INVENTORY.values():require(sha(item['absolute_path'])==item['sha256'],'AUDITED_SOURCE_CHANGED '+item['absolute_path'])
 table('R5A_SOURCE_INVENTORY.csv',list(INVENTORY.values()))
 protected=load(ROOT/'R5A_HISTORY_READ_ONLY_SEAL.json')['files'];changed=[]
 for p,h in protected.items():
  if not Path(p).is_file() or sha(p)!=h:changed.append(p)
 require(not changed,'HISTORICAL_EVIDENCE_MUTATED '+str(changed))
 write('R5A_HISTORY_PRESERVATION_VALIDATION.json',{'status':'PASS','historical_files_verified_unchanged':len(protected),'changed_files':changed,'new_forward':False,'CUDA_initialization':False,'history_modified':False})
 require(not DENIED,'GUARD_REJECTED_UNEXPECTED_ACTION '+str(DENIED));require(not any(k.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml') for k in sys.modules),'DEVICE_MODEL_MODULE_PRESENT')
 write('R5A_EXECUTION_GUARD.json',{'status':'PASS','new_forward':'NO','CUDA_initialized':'NO','history_modified':'NO','torch_or_model_module_imported':False,'native_pipeline_called':False,'subprocess_or_evaluation_inference_launched':False,'model_checkpoint_loaded':False,'checkpoint_bytes_SHA256_verified':True,'serialized_tensor_device_tags_read_as_strings_only':True,'allowed_output_root':str(ROOT),'outside_root_writes_blocked':True,'device_library_loads_blocked':True,'denied_action_events':DENIED,'CPU_modules':['standard_library','numpy','shapely','PyYAML'],'local_time_Asia_Shanghai':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
 create_report(aggregates,total,len(protected));dest,h=build_bundle()
 write('R5A_RESULT.json',{'state':'R5A_PASS','new_forward':'NO','CUDA_initialized':'NO','history_modified':'NO','records_independently_audited':32,'source_counts':{'R2_accepted':4,'R3_recovered':1,'R4_new_real':27},'original_core_CSVs':[{k:SCHEMAS[str(R4/'analysis'/n)][k] for k in ('absolute_path','data_row_count','column_count','sha256','status')} for n in CORE],'aggregate_metrics':aggregates,'all8_interaction':total,'core_FAIL':[],'core_NOT_VERIFIABLE':[],'ready_for_R5B_without_new_GPU_forward':True,'R5B_or_R5C_executed':False,'report_directory':str(ROOT),'bundle_path':str(dest),'bundle_sha256':h,'bundle_size_bytes':dest.stat().st_size})
 print('R5A_PASS REAL_RECORDS=32 CSV_ALL_ROWS_VERIFIED=4 I_TP70=+2 I_FP70=-3 NO_TORCH_NO_FORWARD_NO_HISTORY_WRITE',flush=True);print('BUNDLE',str(dest),h,flush=True)

if __name__=='__main__':
 try:run()
 except Exception as e:
  write('R5A_EXECUTION_STOPPED.json',{'status':'STOPPED_PENDING_CAUSE_AUDIT','error':str(e),'traceback':traceback.format_exc(),'valid_record_count_before_stop':len(COUNTS),'denied_action_events':DENIED,'source_files_modified_by_audit':False,'new_forward':False,'CUDA_initialization':False})
  if COUNTS:table('R5A_PARTIAL_VALID_SAMPLE_COUNTS.csv',COUNTS)
  raise
