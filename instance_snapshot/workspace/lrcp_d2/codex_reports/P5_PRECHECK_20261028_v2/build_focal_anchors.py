import csv,json,collections
from pathlib import Path
R=Path(__file__).resolve().parent;F=R.parent/'P4_FULL_PREP_20261028_113714_782940'
sel=list(csv.DictReader((R/'P5_SAMPLE_PROPOSAL_V2.csv').open()));keys={(int(x['sample_idx']),int(x['gt_idx'])):x for x in sel};agg=collections.defaultdict(list)
for x in csv.DictReader((F/'strict_anchor_lineage_full.csv').open()):
 k=(int(x['sample_idx']),int(x['gt_idx']))
 if x['alpha']=='1' and k in keys:agg[k].append(x)
fields=['sample_idx','gt_idx','category','anchor_id','anchor_IoU_GT','score','first_deletion','suppressor','suppressor_IoU_GT','suppressor_score','anchor_query_index','suppressor_query_index']
with (R/'P5_FOCAL_ANCHORS.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for k,s in keys.items():
  for x in agg[k]:
   a=int(x['anchor_id']);sup=int(x['suppressor'])
   w.writerow({'sample_idx':k[0],'gt_idx':k[1],'category':s['category'],'anchor_id':a,'anchor_IoU_GT':x['IoU_GT'],'score':x['score'],'first_deletion':x['first_deletion'],'suppressor':sup,'suppressor_IoU_GT':x['suppressor_IoU_GT'],'suppressor_score':x['suppressor_score'],'anchor_query_index':a//2,'suppressor_query_index':sup//2 if sup>=0 else ''})
summary={'selected_gt':18,'strict_candidate_rows':sum(map(len,agg.values())),'zero_strict_gt':[{'sample_idx':k[0],'gt_idx':k[1]} for k in keys if not agg[k]],'anchor_to_query':'anchor flat index // 2 is head grid index only; physical GT-to-query mapping remains unvalidated'}
(R/'FOCAL_ANCHOR_SUMMARY.json').write_text(json.dumps(summary,indent=2))
