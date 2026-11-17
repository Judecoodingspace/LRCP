"""Read-only CPU evidence validation and reproducible P4 sample selection. No LRCP/torch import."""
from pathlib import Path
import ast,collections,csv,hashlib,json,pickle,re,subprocess,sys
ROOT=Path(__file__).resolve().parent
BASE=Path('/workspace/lrcp_d2')
SRC=Path('/workspace/LRCP')
B=BASE/'codex_reports/P3B_20261020_103959_980233'
C=BASE/'codex_reports/P3C1_20261020_132232_276238'
INFO=Path('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl')
MODEL=SRC/'opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def dump(name,x):
 p=ROOT/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvwrite(name,rows):
 with (ROOT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks=[];inputs={}
for directory,manifest in [(B,'ARTIFACTS_FINAL.sha256'),(C,'ARTIFACTS.sha256')]:
 lines=(directory/manifest).read_text().splitlines()
 for line in lines:
  expected,name=line.split('  ',1);p=directory/name
  actual=sha(p);assert actual==expected,str(p);inputs[str(p)]=actual
 checks.append({'check':str(directory/manifest),'passed':True,'files':len(lines)})
 inputs[str(directory/manifest)]=sha(directory/manifest)
 for path,expected in json.loads((directory/'input_hashes.json').read_text()).items():
  if path not in inputs:inputs[path]=sha(path)
  assert inputs[path]==expected,path
 checks.append({'check':str(directory/'input_hashes.json'),'passed':True})
# P3-A audit metadata protects checkpoint/config/capture script and old result records.
audit=BASE/'codex_reports/audit_20261018_215046_447565/audit_results.json'
a=json.loads(audit.read_text());n=0
for path,meta in a['files'].items():
 if path.endswith(('.py','.yaml','.pth','.json')) and Path(path).is_file():
  actual=sha(path);assert actual==meta['sha256'],path;inputs[path]=actual;n+=1
inputs[str(audit)]=sha(audit)
checks.append({'check':'P3-A tracked relevant code/config/checkpoint/JSON','passed':True,'files':n})
for p in [INFO,MODEL/'config.yaml',MODEL/'net_epoch_bestval_at19.pth',BASE/'P1_common910_indices.json',BASE/'p3a_capture_k3.py',BASE/'P3A_20261014_120929.json']:
 inputs[str(p)]=sha(p)
# Source and historical artifact difference guard. No inference, no import of source.
protected={str(p):sha(p) for part in ['opencood','data_preprocess_tools'] for p in (SRC/part).rglob('*') if p.is_file() and p.suffix in ('.py','.pyx','.cpp','.cu','.h')}
protected.update(inputs)
dump('PROTECTED_INPUT_HASHES_BEFORE.json',protected)
status=subprocess.run(['git','-C',str(SRC),'status','--short'],check=True,capture_output=True,text=True).stdout
head=subprocess.run(['git','-C',str(SRC),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
dump('WORKTREE_BEFORE.json',{'head':head,'status_short':status})
info=pickle.load(INFO.open('rb'))
rows=list(csv.DictReader((B/'per_gt.csv').open()))
assert len(rows)==8164
manifest=json.loads((BASE/'P1_common910_indices.json').read_text())
assert set(int(x) for x in manifest['sample_ids'])==set(int(x['sample_idx']) for x in rows)
by_group=collections.defaultdict(list)
for original in rows:
 x=dict(original);sid=int(x['sample_idx']);gt=int(x['gt_idx']);i=info[sid]
 m=re.search(r'/scene_(\d+)_(\d+)\.pcd\.bin$',i['lidar_path_1']);assert m
 scene,frame=map(int,m.groups());assert frame==i['timestamp']
 assert i['prev_samples'][3] is not None
 x.update(scene_id=scene,scene_frame_counter=frame,sample_token=i['token'],agent_num=i['agent_num'],lidar_path_1=i['lidar_path_1'])
 t0,t1=int(x['tp70_a0']),int(x['tp70_a1'])
 g={(1,0):'loss_TP70_to_FN70',(0,1):'gain_FN70_to_TP70',(1,1):'control_TP70_to_TP70'}.get((t0,t1))
 if g:
  x['group']=g;x['severity']=abs(float(x['delta_best_iou']));x['distance_m']=(float(x['gt_center_x'])**2+float(x['gt_center_y'])**2)**.5
  x['distance_band']='near_lt20m' if x['distance_m']<20 else 'far_ge20m'
  x['selection_hash']=hashlib.sha256(f"P4-Precheck-v1|{sid}|{gt}".encode()).hexdigest();by_group[g].append(x)
assert [len(by_group[g]) for g in ['loss_TP70_to_FN70','gain_FN70_to_TP70','control_TP70_to_TP70']]==[278,169,6476]
# rank thirds avoid arbitrary severity cutpoints. Each group gets 2 rows per third.
for g,pool in by_group.items():
 pool.sort(key=lambda x:(x['severity'],int(x['sample_idx']),int(x['gt_idx'])))
 for rank,x in enumerate(pool):x['severity_third']=min(2,rank*3//len(pool));x['severity_rank']=rank
selected=[];scene_counts=collections.Counter();band_counts=collections.Counter()
groups=['loss_TP70_to_FN70','gain_FN70_to_TP70','control_TP70_to_TP70']
for repeat in range(2):
 for third in range(3):
  for group in groups:
   eligible=[x for x in by_group[group] if x['severity_third']==third and all(int(x['sample_idx'])!=int(y['sample_idx']) and (x['scene_id']!=y['scene_id'] or abs(x['scene_frame_counter']-y['scene_frame_counter'])>=10) for y in selected)]
   assert eligible,(group,third)
   x=min(eligible,key=lambda x:(scene_counts[x['scene_id']],band_counts[(group,x['distance_band'])],x['selection_hash'],int(x['sample_idx']),int(x['gt_idx'])))
   selected.append(x);scene_counts[x['scene_id']]+=1;band_counts[(group,x['distance_band'])]+=1
out=[]
for order,x in enumerate(selected,1):
 out.append({k:x[k] for k in ['sample_idx','gt_idx','group','scene_id','scene_frame_counter','sample_token','agent_num','lidar_path_1','gt_center_x','gt_center_y','distance_m','distance_band','best_iou_a0','best_iou_a1','delta_best_iou','tp70_a0','tp70_a1','matched_pred_idx70_a0','matched_pred_idx70_a1','best_pred_score_a0','best_pred_score_a1','severity','severity_third','severity_rank','selection_hash']}|{'selection_order':order,'reason':f"{x['group']}; abs(delta_IoU) rank third {x['severity_third']+1}/3; scene/band balance, min 10 frame-counter separation; fixed SHA tie-break"})
csvwrite('SAMPLE_SELECTION.csv',out)
selected_sids={int(x['sample_idx']) for x in selected}
# All GT in selected frames, not merely focal rows, are required in pilot.
all_gt=[x for x in rows if int(x['sample_idx']) in selected_sids]
csvwrite('SELECTED_SAMPLES_ALL_GT.csv',all_gt)
# Validate selected historical capture contents; no IoU/AP recomputation.
import numpy as np
for sid in sorted(selected_sids):
 with np.load(BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_0.0.npz',allow_pickle=False) as z0,np.load(BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_1.0.npz',allow_pickle=False) as z1:
  assert np.array_equal(z0['gt_boxes'],z1['gt_boxes'])
  rr=[x for x in all_gt if int(x['sample_idx'])==sid];assert len(rr)==len(z0['gt_boxes'])
  for x in rr:
   idx=int(x['gt_idx']);xy=z0['gt_boxes'][idx,:,:2].mean(axis=0)
   assert np.allclose(xy,[float(x['gt_center_x']),float(x['gt_center_y'])],atol=1e-5,rtol=0)
checks.append({'check':'18 selected samples / every saved GT index and center / alpha equality','passed':True})
summary={'rule_version':'P4-Precheck-v1','groups_population':{g:len(v) for g,v in by_group.items()},'group_selections':dict(collections.Counter(x['group'] for x in selected)),'focal_gt':len(selected),'unique_samples':len(selected_sids),'all_gt_in_selected_samples':len(all_gt),'scenes_selected':dict(scene_counts),'near_far_by_group':{g:dict(collections.Counter(x['distance_band'] for x in selected if x['group']==g)) for g in groups},'sample_indices':sorted(selected_sids),'min_same_scene_frame_separation':min(abs(x['scene_frame_counter']-y['scene_frame_counter']) for i,x in enumerate(selected) for y in selected[i+1:] if x['scene_id']==y['scene_id']),'pilot_forwards':{'paired_core':36,'one_sample_instrumentation_equivalence':2,'one_sample_reverse_order_repeat':2,'maximum_total':40},'torch_imported':'torch' in sys.modules}
dump('SELECTION_SUMMARY.json',summary)
dump('INPUT_VALIDATION.json',{'passed':True,'checks':checks,'input_files_hashed':len(inputs),'source_files_locked':len(protected)-len(inputs),'torch_imported':'torch' in sys.modules})
# Source evidence: line-numbered exact excerpts, plus AST definitions, no execution.
source_files=['opencood/tools/inference_utils.py','opencood/tools/train_utils.py','opencood/models/point_pillar_deformable.py','opencood/models/meta_flow.py','opencood/models/sub_modules/defor_bev_backbone_resnet.py','opencood/models/sub_modules/defor_encoder_multi_scale.py','opencood/data_utils/datasets/intermediate_latency_fusion_dataset.py','opencood/data_utils/datasets/basedataset/v2xsim_basedataset.py','opencood/data_utils/post_processor/voxel_postprocessor.py','opencood/data_utils/post_processor/base_postprocessor.py','opencood/utils/box_utils.py','opencood/utils/eval_utils.py','data_preprocess_tools/flow_label_gen_v2x-smi.py']
index=[]
with (ROOT/'SOURCE_EVIDENCE.txt').open('x') as f:
 for relative in source_files:
  p=SRC/relative;txt=p.read_text();tree=ast.parse(txt)
  for n in ast.walk(tree):
   if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
    index.append({'file':str(p),'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'sha256':sha(p)})
  f.write('\nFILE '+str(p)+' SHA256 '+sha(p)+'\n')
  f.writelines(f'{i:5d} {line}\n' for i,line in enumerate(txt.splitlines(),1))
csvwrite('SOURCE_SYMBOL_INDEX.csv',index)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('VALIDATION',len(inputs),'inputs; torch imported:', 'torch' in sys.modules)
