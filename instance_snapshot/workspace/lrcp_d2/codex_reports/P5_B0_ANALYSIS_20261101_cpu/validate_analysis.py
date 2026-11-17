"""Independent CPU cross-check: native TP/FP, manual PR integration, identity tables."""
import os,sys,json,csv,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,torch
O=Path(__file__).parent;R=O.parent;C2=R/'P5_B0_C2_20261031_cpu';RUN=R/'P5_B0_C3_20261031_cpu/real_run';P5=R/'P5_A_CPU_FIX_20261028_storage'
sys.path.insert(0,str(C2));sys.path.insert(0,'/workspace/LRCP')
from pilot_common import sha_file
from opencood.utils import eval_utils
sys.path.insert(0,str(R/'P4_C2_20261027_233036_301795'))
from c1_reused import cpu_tensor_inputs
rows=list(csv.DictReader((O/'per_GT.csv').open()));focal=list(csv.DictReader((O/'focal_results.csv').open()));metrics=json.loads((O/'AP_METRICS.json').read_text());summary=json.loads((O/'summary.json').read_text())
assert len(rows)==167 and len(focal)==18 and len({(x['sample_idx'],x['gt_idx']) for x in rows})==167
accepted=json.loads((RUN/'CPU_VALIDATION_COMPLETE.json').read_text());assert accepted['passed'] and not accepted['mock_only'];assert sha_file(RUN/'manifest.json')==accepted['manifest_sha256']
assert {(int(x['sample_idx']),int(x['gt_idx']),int(x['TP70_alpha1']),int(x['TP70_B0'])) for x in rows}=={(x['sample_idx'],x['gt_idx'],x['native_tp70'],x['intervened_tp70']) for x in accepted['result']['per_GT_transitions']}
identity={(int(x['sample_idx']),int(x['gt_idx'])):x for x in rows}
for x in csv.DictReader((R/'P4_FULL_PREP_20261028_113714_782940/per_GT_full.csv').open()):
 key=(int(x['sample_idx']),int(x['gt_idx']))
 if key not in identity:continue
 label='alpha'+str(int(x['alpha']));row=identity[key]
 assert int(x['TP70'])==int(row['TP70_'+label]);assert int(x['TP50'])==int(row['TP50_'+label])
manifest=json.loads((RUN/'manifest.json').read_text());paths={int(x['sample_idx']):Path(x['path'])/'payload.bin' for x in manifest['records'] if x['condition']=='core'}
ids=list(dict.fromkeys(int(x['sample_idx']) for x in rows));all_stats={}
for label in ('alpha0','alpha1','B0'):
 for t in (.5,.7):
  dest={'score':[],'tp':[],'fp':[],'gt':0}
  for sid in ids:
   path=paths[sid] if label=='B0' else P5/f'capture/sample_{sid}/alpha_{int(label[-1])}.0.npz'
   with np.load(path,allow_pickle=False) as z:
    result={t:{'score':[],'tp':[],'fp':[],'gt':0}}
    eval_utils.caluclate_tp_fp(*cpu_tensor_inputs(z['final_corners'],z['final_scores'],z['GT_canonical']),result,t)
   for k in ('score','tp','fp'):dest[k].extend(result[t][k])
   dest['gt']+=result[t]['gt']
  scores=np.asarray(dest['score']);order=np.argsort(-scores);tp=np.cumsum(np.asarray(dest['tp'])[order]);fp=np.cumsum(np.asarray(dest['fp'])[order]);rec=tp/dest['gt'];prec=tp/(tp+fp)
  rr=np.r_[0.,rec,1.];pp=np.r_[0.,prec,0.];envelope=np.maximum.accumulate(pp[::-1])[::-1];i=np.flatnonzero(rr[1:]!=rr[:-1]);ap=float(np.sum((rr[i+1]-rr[i])*envelope[i+1]))
  m=metrics[label][str(int(t*100))];assert m['TP']==int(tp[-1]) and m['FP']==int(fp[-1]) and m['GT']==dest['gt'];assert abs(m['AP_selected18']-ap)<1e-12
  assert m['TP']==sum(int(x['TP'+str(int(t*100))+'_'+label]) for x in rows)
  all_stats[label+str(t)]={'TP':int(tp[-1]),'FP':int(fp[-1]),'AP_manual_PR_integral':ap}
for group in csv.DictReader((O/'focal_groups.csv').open()):
 subset=[x for x in focal if x['category']==group['category']];assert len(subset)==3
 for label in ('alpha0','alpha1','B0'):assert sum(int(x['TP70_'+label]) for x in subset)==int(group['TP70_'+label])
for scene in csv.DictReader((O/'per_scene.csv').open()):
 subset=[x for x in rows if x['scene_id']==scene['scene_id']];assert len(subset)==int(scene['GT'])
 for label in ('alpha1','B0'):assert sum(int(x['TP70_'+label]) for x in subset)==int(scene['TP70_'+label])
for p,h in json.loads((O/'SOURCE_HASHES.json').read_text())['files'].items():assert sha_file(p)==h
assert not torch.cuda.is_initialized();assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30
out={'passed':True,'all_GT':167,'focal_GT':18,'metrics_conditions':3,'thresholds':2,'manual_PR_integration_verified':True,'P4_Full_native_GT_labels_verified':True,'C3_independent_GT_transitions_verified':True,'source_hashes_verified':True,'metrics':all_stats,'new_model_forward':0,'CUDA_initialized':False}
(O/'ANALYSIS_VALIDATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
