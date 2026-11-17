import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from types import SimpleNamespace
from pathlib import Path
from pilot_common import MODEL,digest,sha_file
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
p=MODEL/'net_epoch_bestval_at19.pth';h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(MODEL)));m=train_utils.create_model(h);state=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(state,strict=True)
params=dict(m.named_parameters());buffers=dict(m.named_buffers())
assert set(state)==set(params)|set(buffers) and not torch.cuda.is_initialized()
z={'checkpoint_sha256':sha_file(p),'parameter_digest':digest(params),'buffer_digest':digest(buffers),'parameter_count':len(params),'buffer_count':len(buffers),'strict_load':True,'model_forward':0,'CUDA_initialized':False}
Path(__file__).with_name('CHECKPOINT_IDENTITY.json').write_text(json.dumps(z,indent=2));print(json.dumps(z))
