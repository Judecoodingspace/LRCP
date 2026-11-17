import os,json,csv,hashlib,pickle,re,collections,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
R=Path(__file__).resolve().parent;C=R.parent/'P4_FULL_CF_20261028_cpu_audit';F=R.parent/'P4_FULL_PREP_20261028_113714_782940';B=R.parent/'P3B_20261020_103959_980233'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
assert json.loads((C/'validation.json').read_text())['passed'] and json.loads((F/'RESEARCH_ACCEPTED.json').read_text())['passed']
for name in ['validation.json','summary.json','per_GT_mixed.csv','P5_SAMPLE_PROPOSAL.csv','P5_PROTOCOL.md']:
 entries={line.split('  ',1)[1].strip():line[:64] for line in (C/'ARTIFACTS.sha256').read_text().splitlines()};assert sha(C/name)==entries[name],name
cap=json.loads((F/'PPU_CAPTURE_COMPLETE.json').read_text());acc=json.loads((F/'RESEARCH_ACCEPTED.json').read_text());assert sha(F/'PPU_CAPTURE_COMPLETE.json')==acc['PPU_capture_sha256'] and sha(F/'CPU_VALIDATION_COMPLETE.json')==acc['CPU_validation_sha256']
inputs={int(x['sample_idx']):x for x in json.loads((F/'INPUT_MANIFEST.json').read_text())['samples']};assert len(inputs)==910
records={}
for sh in range(29):
 p=F/'capture'/f'shard_{sh:03d}'/'COLLECTION_MANIFEST.json';assert sha(p)==cap['shard_manifest_hashes'][str(sh)]
 for z in json.loads(p.read_text())['records']:records[(int(z['sample_idx']),int(z['alpha']))]=z
assert len(records)==1820
info=pickle.load(open('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl','rb'))
def scene(s):
 x=info[s];m=re.search(r'/scene_(\d+)_(\d+)\.pcd\.bin$',x['lidar_path_1']);assert m and int(m[2])==x['timestamp'] and x['token']==inputs[s]['sample_token'];return int(m[1]),int(m[2])
rows=[x for x in csv.DictReader((C/'per_GT_mixed.csv').open()) if x['geometry']=='1'];assert len(rows)==8164
cats=[('nms_recovered',lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='nms_overlap_suppression' and x['transition']=='FN_to_TP'),('nms_not_recovered',lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='nms_overlap_suppression' and x['transition']=='FN_to_FN'),('score_loss',lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='score_filter'),('raw_absent',lambda x:x['historic_transition']=='TP_to_FN' and x['native_stage']=='no_high_iou_raw_candidate'),('recovered',lambda x:x['historic_transition']=='FN_to_TP'),('stable_tp',lambda x:x['historic_transition']=='TP_to_TP')]
used=set();scenes=collections.Counter();selected=[];pop={}
for label,fn in cats:
 pool=[x for x in rows if fn(x)];pop[label]={'GT':len(pool),'samples':len({int(x['sample_idx']) for x in pool}),'scene_counts':dict(sorted(collections.Counter(str(scene(int(x['sample_idx']))[0]) for x in pool).items()))}
 for rep in range(3):
  cand=[x for x in pool if int(x['sample_idx']) not in used];assert cand,label
  def key(x):
   sid=int(x['sample_idx']);g=int(x['gt_idx']);sc,_=scene(sid);h=hashlib.sha256(f'P5-v2:{label}:{sid}:{g}'.encode()).hexdigest();return (sc in scenes,scenes[sc],h,sid,g)
  x=min(cand,key=key);sid=int(x['sample_idx']);gi=int(x['gt_idx']);sc,fr=scene(sid);used.add(sid);scenes[sc]+=1
  selected.append({'category':label,'sample_idx':sid,'gt_idx':gi,'scene_id':sc,'scene_frame':fr,'sample_token':inputs[sid]['sample_token'],'native_stage':x['native_stage'],'mixed_transition':x['transition'],'selection_sha256':key(x)[2],'input_sha256':inputs[sid]['sha256'],'input_semantic_hash':inputs[sid]['input_hash']})
assert len(selected)==len(used)==18 and [pop[k]['GT'] for k,_ in cats]==[69,136,64,9,169,6476]
for x in selected:
 sid=x['sample_idx'];inp=inputs[sid];assert sha(inp['path'])==inp['sha256'];payload=torch.load(inp['path'],map_location='cpu',weights_only=False,mmap=False);ego=payload['batch']['ego'];assert tuple(ego['anchor_box'].shape)==(80,80,2,7) and tuple(ego['transformation_matrix'].shape)==(4,4)
 assert all(not t.is_cuda for t in (ego['anchor_box'],ego['transformation_matrix']))
 for g in (0,1):
  z=records[(sid,g)];assert sha(z['path'])==z['sha256'] and sha(z['meta_path'])==z['meta_sha256'];assert z['input_hash']==inp['input_hash']
  with np.load(z['path'],allow_pickle=False) as f:
   assert f['cls_preds'].size==12800 and f['reg_preds'].size==12800*7 and f['dir_preds'].size==12800*2 and f['GT_canonical'].shape[0]>x['gt_idx']
   if g==0:gt=f['GT_canonical'].copy()
   else:assert np.array_equal(gt,f['GT_canonical'])
 del payload
assert not torch.cuda.is_initialized()
with (R/'P5_SAMPLE_PROPOSAL_V2.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=selected[0]);w.writeheader();w.writerows(selected)
(R/'SELECTION_EVIDENCE.json').write_text(json.dumps({'passed':True,'population':pop,'selected':18,'unique_samples':len(used),'scene_counts':dict(sorted(scenes.items())),'hash_rule':'P5-v2:<category>:<sample_idx>:<gt_idx>','frozen_input_and_pair_hashes':True,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
