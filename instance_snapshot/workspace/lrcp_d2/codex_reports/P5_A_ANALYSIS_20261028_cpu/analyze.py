import csv,json,hashlib
from pathlib import Path
import numpy as np
D=Path('/workspace/lrcp_d2/codex_reports/P5_A_CPU_FIX_20261028_storage');O=Path(__file__).parent
P=Path('/workspace/lrcp_d2/codex_reports/P4_FULL_PREP_20261028_113714_782940/per_GT_full.csv')
M=Path('/workspace/lrcp_d2/codex_reports/P4_FULL_CF_20261028_cpu_audit/per_GT_mixed.csv')
assert json.loads((D/'RESEARCH_ACCEPTED.json').read_text())['passed']
sel=list(csv.DictReader((D/'P5_SAMPLE_PROPOSAL_V2.csv').open())); keys={(int(x['sample_idx']),int(x['gt_idx'])) for x in sel}
gt={(int(x['sample_idx']),int(x['gt_idx']),int(x['alpha'])):x for x in csv.DictReader(P.open()) if (int(x['sample_idx']),int(x['gt_idx'])) in keys}
mix={(int(x['sample_idx']),int(x['gt_idx'])):x for x in csv.DictReader(M.open()) if int(x['geometry'])==1 and (int(x['sample_idx']),int(x['gt_idx'])) in keys}
manifest=json.loads((D/'CAPTURE_MANIFEST.json').read_text())['records']; idx={(x['sample_idx'],int(x['alpha'])):x for x in manifest}; rows=[]; focal=[]; bounds={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):return {'equal':bool(np.array_equal(a,b)),'mean_abs':float(np.mean(np.abs(a-b))),'max_abs':float(np.max(np.abs(a-b)))}
for x in sel:
 sid=int(x['sample_idx']);gid=int(x['gt_idx']); cases=[];metadata=[]
 for alpha in (0,1):
  rec=idx[sid,alpha];p=Path(rec['path']);assert sha(p)==rec['sha256'] and sha(p.with_suffix('.json'))==rec['meta_sha256'];cases.append(np.load(p,allow_pickle=False));metadata.append(json.loads(p.with_suffix('.json').read_text()))
 a,b=cases
 for i in range(4):
  row={'sample_idx':sid,'category':x['category'],'scene_id':x['scene_id'],'block':i,'query_count':len(metadata[0]['query_ids'])}
  for k in ['reference','offset','location','weights','sampling_offsets_input','sampling_offsets_output','attention_weights_output']:
   z=diff(a[f'cross_{i}_{k}'],b[f'cross_{i}_{k}']);row.update({f'{k}_{n}':v for n,v in z.items()})
  rows.append(row)
  for alpha in (0,1):
   for e in metadata[alpha]['attention']['blocks'][i]['summary']:
    key=(alpha,i,'ego' if e['agent_slot']=='ego' else ('valid_remote' if e['valid_history'] else 'invalid_remote'))
    z=bounds.setdefault(key,{'count':0,'oob':0,'unsupported':0});z['count']+=e['count'];z['oob']+=e['normalized_oob'];z['unsupported']+=e['kernel_unsupported']
 rawkeys=sorted(k for k in a.files if k.startswith('metaflow_raw_'));assert all(np.array_equal(a[k],b[k]) for k in rawkeys)
 for k in rawkeys:
  effective=k.replace('raw_','effective_');assert np.array_equal(a[effective],a[k]*0) and np.array_equal(b[effective],b[k])
 f={'sample_idx':sid,'gt_idx':gid,'category':x['category'],'scene_id':x['scene_id'],'TP70_alpha0':gt[sid,gid,0]['TP70'],'TP70_alpha1':gt[sid,gid,1]['TP70'],'alpha1_stage':gt[sid,gid,1]['last_strict_disappearance'],'rank_swap_alpha1':mix[sid,gid]['transition'],'raw_candidates_alpha1':gt[sid,gid,1]['count_raw_decoded'],'score_candidates_alpha1':gt[sid,gid,1]['count_score_pass'],'nms_candidates_alpha1':gt[sid,gid,1]['count_nms_keep'],'query_count':len(metadata[0]['query_ids']),'metaflow_raw_slot_count':len(rawkeys)}
 focal.append(f)
 for c in cases:c.close()
assert len(rows)==72 and len(focal)==18
for name,data in [('block_changes.csv',rows),('focal_results.csv',focal)]:
 with (O/name).open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
b=[]
for (alpha,block,group),z in sorted(bounds.items()):b.append({'alpha':alpha,'block':block,'group':group,**z,'oob_fraction':z['oob']/z['count'],'unsupported_fraction':z['unsupported']/z['count']})
with (O/'sampling_boundaries.csv').open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(b[0]));w.writeheader();w.writerows(b)
summary={'samples':18,'selected_queries_total':sum(x['query_count'] for x in focal),'block':{},'boundary':b}
for i in range(4):
 rr=[x for x in rows if x['block']==i];summary['block'][str(i)]={k:sum(not x[k+'_equal'] for x in rr) for k in ['reference','offset','location','weights','sampling_offsets_input','sampling_offsets_output','attention_weights_output']}
 summary['block'][str(i)]['sample_mean_abs_location_difference']=float(np.mean([x['location_mean_abs'] for x in rr]))
(O/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
