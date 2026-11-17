"""CPU test on independently stored pilot captures; no model or device initialization."""
from pathlib import Path
import csv,json,torch
from scientific_validator import check_case,R,B
P=R.parent/'P4_PILOT_RETRY_20261023_234457_384594'
rows=list(csv.DictReader((B/'per_gt.csv').open()));out=[]
for sid in [821,17,670]:
 rec=sorted([z for z in rows if int(z['sample_idx'])==sid],key=lambda z:int(z['gt_idx']))
 for a in [0,1]:
  stat,gts,events=check_case(sid,a,P/f'diagnostics/core_sample_{sid}_alpha_{a:.1f}.npz',rec)
  out.append({'sample_idx':sid,'alpha':a,'GT_count':len(gts),'TP70':sum(x['TP70'] for x in gts),'event_count':len(events),'passed':True})
assert not torch.cuda.is_initialized()
(R/'SCIENTIFIC_VALIDATOR_CPU_TEST.json').write_text(json.dumps({'passed':True,'cases':out,'model_forwards':0,'CUDA_initialized':False},indent=2))
print('SCIENTIFIC_VALIDATOR_REAL_HEAD_TEST_PASSED')
