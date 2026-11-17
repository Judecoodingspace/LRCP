import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
from INDEPENDENT_VALIDATOR import verify_attention,sha
R=Path(__file__).parent;P=R.parent/'P5_A_CPU_FIX_20261028_storage';manifest=json.loads((P/'CAPTURE_MANIFEST.json').read_text())['records'];out=[]
for rec in manifest:
 p=Path(rec['path']);assert sha(p)==rec['sha256'] and sha(p.with_suffix('.json'))==rec['meta_sha256'];m=json.loads(p.with_suffix('.json').read_text());q=m['query_ids'];active=m['attention']['active_slots']
 with np.load(p,allow_pickle=False) as z:
  a={k:z[k].copy() for k in z.files if k.startswith('cross_')};a['block0_output_selected']=np.zeros((1,len(q),128),np.float32)
  # History stores only selected-query reference; placeholders are not historical full reference.
  for b in range(4):
   ref=a[f'cross_{b}_reference'];full=np.zeros((1,6400,ref.shape[2],2),np.float32);full[:,q,:,:]=ref;a[f'cross_{b}_direct_reference']=full
  stats=verify_attention(a,{'query_ids':q,'active_slots':active})
 out.append({'sample_idx':rec['sample_idx'],'alpha':rec['alpha'],'recorded_queries':len(q),'passed':True})
assert len(out)==36 and not torch.cuda.is_initialized()
(R/'ATTENTION_HISTORICAL_REGRESSION.json').write_text(json.dumps({'passed':True,'cases':out,'case_count':36,'scope':'Only saved query values; full reference and block0 output placeholders used solely to invoke shared checker; not historical evidence for either.','real_model_forward':0,'CUDA_initialized':False},indent=2))
