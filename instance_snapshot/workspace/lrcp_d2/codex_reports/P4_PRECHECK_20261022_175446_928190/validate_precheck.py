"""Static/CPU checks only; AST-extract native helpers without importing torch/LRCP."""
from pathlib import Path
import ast,csv,hashlib,importlib.util,json,re,subprocess,sys
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Polygon
R=Path(__file__).resolve().parent
S=Path('/workspace/LRCP')
checks=[]
def check(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def extract(path,names,env):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 return env
for p in R.glob('*.py'):
 tree=ast.parse(p.read_text());compile(tree,str(p),'exec')
 check('compile_and_AST_'+p.name,True)
 check('no_torch_or_LRCP_import_'+p.name,not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(v.startswith(('torch','opencood')) for v in ([n.module or ''] if isinstance(n,ast.ImportFrom) else [a.name for a in n.names])) for n in ast.walk(tree)))
spec=importlib.util.spec_from_file_location('draft',R/'diagnostic_contract_draft.py');draft=importlib.util.module_from_spec(spec);spec.loader.exec_module(draft)
# 7 GT: each first loss stage, plus survives_final; include exact 0.7 threshold.
iou=np.zeros((7,6),np.float32)
for gi in range(1,7):iou[gi,gi-1]=np.float32(.7)
ids={'raw_decoded':np.arange(6),'score_pass':np.arange(1,6),'quality_pass':np.arange(2,6),'nms_top1000':np.arange(3,6),'nms_keep':np.arange(4,6),'final_range':np.arange(5,6)}
res=draft.attribute_gt(iou,ids)
check('six_distinct_stage_causes_plus_survivor',[x['availability_class'] for x in res]==list(draft.CAUSES)+['high_iou_survives_final'])
# Multiple strict candidates deleted at different stages remain separately attributed.
x=np.array([[.8,.9,0,0,0,0]],np.float32);rr=draft.attribute_gt(x,ids)[0]
check('multi_candidate_different_deletion_stages',rr['high_iou_deleted_ids']['score_filter']==[0] and rr['high_iou_deleted_ids']['size_or_z_filter']==[1] and rr['availability_class']=='size_or_z_filter')
bad={**ids,'nms_keep':np.array([0])}
try:draft.attribute_gt(x,bad);raise AssertionError('accepted nonnested')
except ValueError:check('reject_non_nested_lineage',True)
try:draft.attribute_gt(np.array([[float('nan')]*6]),ids);raise AssertionError('accepted NaN')
except ValueError:check('nonfinite_IoU_not_called_candidate_absence',True)
# Exact repository native polygon helpers and NMS, loaded as AST only.
env=extract(S/'opencood/utils/common_utils.py',{'compute_iou','convert_format'},{'np':np,'Polygon':Polygon})
common=SimpleNamespace(**{n:env[n] for n in ['compute_iou','convert_format']})
native=extract(S/'opencood/utils/box_utils.py',{'nms_rotated'},{'np':np,'common_utils':common})['nms_rotated']
class CpuTensor:
 def __init__(self,x):self.arr=x;self.shape=x.shape
 def cpu(self):return self
 def detach(self):return self
 def numpy(self):return self.arr
base=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],np.float32)
boxes=np.repeat(base[None],1002,axis=0)
scores=np.ones(1002,np.float32)*.8
nkeep=native(CpuTensor(boxes),CpuTensor(scores),.15)
r=draft.replay_native_nms(boxes,scores,.15,common.convert_format,common.compute_iou)
check('native_NMS_float32_ties_top1000_exact_keep',np.array_equal(nkeep,r['keep']))
check('NMS_separate_top1000_and_overlap',len(r['top1000_excluded'])==2 and len(r['suppressed'])==999 and len(r['keep'])==1)
check('NMS_suppressor_provenance',all(v['suppressor_nms_input_idx']==int(r['keep'][0]) and v['bev_iou_float32']==1. for v in r['suppressed']))
# Historical post-NMS boxes exercise actual geometry/dtypes, without recomputing AP.
for alpha in ['0.0','1.0']:
 p=Path('/workspace/lrcp_d2/P3A_20261014_120929_boxes')/f'sample_821_alpha_{alpha}.npz'
 with np.load(p,allow_pickle=False) as z:
  boxes=z['pred_boxes'];scores=z['pred_scores']
  nk=native(CpuTensor(boxes),CpuTensor(scores),.15)
  out=draft.replay_native_nms(boxes,scores,.15,common.convert_format,common.compute_iou)
  check('saved_geometry_CPU_NMS_exact_keep_alpha_'+alpha,np.array_equal(nk,out['keep']))
# CPU hook-double exercises explicit alpha/raw copy and cleanup contract.
class DummyModule:
 def __init__(self):self.hooks=[]
 def register_forward_hook(self,cb):self.hooks.append(cb);return SimpleNamespace(remove=lambda:None)
 def register_forward_pre_hook(self,cb):return SimpleNamespace(remove=lambda:None)
model=SimpleNamespace(meta_flow=DummyModule(),backbone=SimpleNamespace(defor_encoder=DummyModule()),cls_head=DummyModule(),reg_head=DummyModule(),dir_head=DummyModule())
emitted=[];state={'alpha':0.,'remote_slot':2}
handles=draft.paired_capture_hooks(model,state,lambda n,v,m:emitted.append((n,v.copy(),m.copy())))
x=np.ones((1,80,80,2),np.float32)
y=model.meta_flow.hooks[0](None,None,x)
check('hook_raw_effective_alpha0_and_source_unchanged',np.all(x==1) and np.all(y==0) and np.all(emitted[0][1]==1) and np.all(emitted[1][1]==0))
check('hook_metadata_slot_and_handle_contract',emitted[0][2]['slot']==2 and len(handles)==6)
# Validate sample quotas without rerunning data selection/source stats.
rows=list(csv.DictReader((R/'SAMPLE_SELECTION.csv').open()))
check('18_distinct_samples',len(rows)==18 and len({x['sample_idx'] for x in rows})==18)
for g in {x['group'] for x in rows}:
 group=[x for x in rows if x['group']==g]
 check('each_group_6_with_2_per_severity_third_'+g,len(group)==6 and all(sum(int(x['severity_third'])==t for x in group)==2 for t in range(3)))
check('all_scene_gaps_ge10',all(x['scene_id']!=y['scene_id'] or abs(int(x['scene_frame_counter'])-int(y['scene_frame_counter']))>=10 for i,x in enumerate(rows) for y in rows[i+1:]))
check('all162_GT',len(list(csv.DictReader((R/'SELECTED_SAMPLES_ALL_GT.csv').open())))==162)
B=Path('/workspace/lrcp_d2/codex_reports/P3B_20261020_103959_980233');C=Path('/workspace/lrcp_d2/codex_reports/P3C1_20261020_132232_276238')
b=json.loads((B/'summary.json').read_text());c=json.loads((C/'summary.json').read_text())
expected={'0.0':{'50':.8665290892796786,'70':.8071592298880969},'1.0':{'50':.8768414234452961,'70':.785423020395363}}
check('historical_four_AP_records_exact_no_recalculation',all(b['metrics'][a][t]['ap']==v for a,tt in expected.items() for t,v in tt.items()) and all(c['baseline_exact'].values()))
# Check each document source line target exists / no invalid source line reference.
links=[]
for p in R.glob('*.md'):
 for target in re.findall(r'\]\((/workspace/[^)]+)\)',p.read_text()):
  match=re.match(r'(.+):(\d+)$',target)
  f=Path(match[1] if match else target)
  assert f.is_file(),target
  if match:assert 1<=int(match[2])<=len(f.read_text().splitlines()),target
  links.append(target)
check('all_absolute_document_file_line_links_resolve',len(links)>40)
check('torch_not_imported','torch' not in sys.modules)
protected=json.loads((R/'PROTECTED_INPUT_HASHES_BEFORE.json').read_text())
changed=[p for p,d in protected.items() if not Path(p).exists() or sha(p)!=d]
work=json.loads((R/'WORKTREE_BEFORE.json').read_text())
current=subprocess.run(['git','-C',str(S),'status','--short'],check=True,capture_output=True,text=True).stdout
check('protected_source_checkpoint_config_history_unchanged',not changed)
check('git_worktree_status_unchanged',current==work['status_short'])
(R/'STATIC_CHECKS.json').write_text(json.dumps({'passed':True,'checks_count':len(checks),'checks':checks,'torch_imported':'torch' in sys.modules,'PPU_execution':False,'note':'CPU tests validate contracts only; full collector runtime equivalence not yet tested.'},ensure_ascii=False,indent=2)+'\n')
(R/'DIFF_AUDIT.json').write_text(json.dumps({'passed':not changed and current==work['status_short'],'protected_files':len(protected),'changed':changed,'git_status_before':work['status_short'],'git_status_after':current,'mutation_scope':'new P4_PRECHECK directory only; no source or historical mutation'},ensure_ascii=False,indent=2)+'\n')
required=['SOURCE_AUDIT.md','MECHANISM_HYPOTHESES.md','PILOT_PROTOCOL.md','SAMPLE_SELECTION.csv','IMPLEMENTATION_PLAN.md','STATIC_CHECKS.json','DIFF_AUDIT.json']
assert all((R/p).is_file() for p in required)
(R/'COMPLETION.json').write_text(json.dumps({'phase':'P4-Precheck','status':'COMPLETE_PRECHECK_ONLY','source_audit_complete':True,'CPU_selection_complete':True,'PPU_started':False,'pilot_approval_required':True,'pilot_samples':18,'pilot_all_GT':162,'pilot_forward_max':40,'static_checks':len(checks),'protected_files_unchanged':len(protected),'main_report':str(R/'REPORT.md')},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'protected_unchanged':len(protected),'torch_imported':'torch' in sys.modules,'main_report':str(R/'REPORT.md')},ensure_ascii=False))
