import os,sys,json,csv,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent;F=R.parent/'P4_FULL_PREP_20261028_113714_782940'
sys.path.insert(0,str(F));sys.path.insert(0,'/workspace/LRCP')
import torch
from pilot_common import digest
inp={int(x['sample_idx']):x for x in json.loads((F/'INPUT_MANIFEST.json').read_text())['samples']};results=[]
for x in csv.DictReader((R/'P5_SAMPLE_PROPOSAL_V2.csv').open()):
 sid=int(x['sample_idx']);item=inp[sid];payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False)
 d=digest(payload['batch']);rng=digest(payload['rng']);assert d==item['input_hash'] and rng==item['RNG_hash'];results.append({'sample_idx':sid,'input_hash':d,'RNG_hash':rng})
 del payload
assert len(results)==18 and not torch.cuda.is_initialized()
(R/'SELECTED_INPUT_SEMANTIC_HASHES.json').write_text(json.dumps({'passed':True,'samples':results,'RNG_states_match':True,'CUDA_initialized':False,'model_forwards':0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
