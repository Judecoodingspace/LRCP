#!/usr/bin/env python3
"""CPU-only independent verification of one committed control shard.
Invoked by C3 atomic-commit adapter; never initializes CUDA or runs a model.
"""
import os,sys,json,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
C2='/workspace/lrcp_d2/codex_reports/P5_B0_C2_20261031_cpu'
sys.path.insert(0,C2)
sys.path.insert(0,'/workspace/LRCP')
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import INDEPENDENT_VALIDATOR as v
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor

def validate(shard,sid,kind,mock=False,peer=None):
 p=Path(shard).resolve();sid=int(sid)
 v.require((sid,kind) in {(590,'off'),(590,'pass'),(918,'off'),(918,'pass')},'CONTROL_ID')
 man=json.loads((p/'manifest.json').read_text());v.require(man['sample']==sid*3+{'off':0,'pass':1}[kind],'SHARD_ID')
 for k in ('payload.bin','meta.json'):v.require(v.sha(p/k)==man['files'][k],'FILE_HASH')
 meta=json.loads((p/'meta.json').read_text());v.require(meta['condition']==kind and meta['backend']==('mock' if mock else 'real'),'CONDITION')
 selected={int(x['sample_idx']):x for x in csv.DictReader((v.R/'P5_SAMPLE_PROPOSAL_V2.csv').open())}
 items={int(x['sample_idx']):x for x in json.loads((v.FULL/'INPUT_MANIFEST.json').read_text())['samples']}
 v.verify_identity(meta,sid,selected,items);item=items[sid];v.require(v.sha(item['path'])==item['sha256'],'INPUT_HASH')
 payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False)
 v.require(v.digest(payload['batch'])==item['input_hash'] and v.digest(payload['rng'])==item['RNG_hash'],'INPUT_SEMANTICS')
 ego=payload['batch']['ego'];v.require(meta['active_slots']==list(item['slots'])[:int(ego['record_len'][0])-1],'ACTIVE_SLOT_COUNT')
 for key,slot in ego['calibrate_data'].items():v.same(np.asarray(slot['valid_mask'].cpu()),np.asarray(item['slots'][str(key)]['valid_mask']),'VALID_MASK')
 history_meta=json.loads((v.P5/f'capture/sample_{sid}/alpha_1.0.json').read_text())
 v.require(meta['query_ids']==history_meta['query_ids'],'QUERY_ALIGNMENT')
 integrity=meta['runtime_integrity']
 v.require(integrity.get('online_reference')=={'all_6400_reference_formula':True,'ego_and_invalid_unchanged':True,'later_reference_unchanged':True,'cross_call_count':4},'ONLINE_EVIDENCE')
 ident=json.loads((v.R/'CHECKPOINT_IDENTITY.json').read_text())
 v.require(v.sha(v.CHECKPOINT)==v.CH,'CHECKPOINT_HASH')
 for k in ('parameter','buffer'):
  v.require(integrity[k+'_before']==integrity[k+'_after'],'RUNTIME_'+k)
  if not mock:v.require(integrity[k+'_before']==ident[k+'_digest'],'CHECKPOINT_'+k)
 v.require(integrity['frozen_before']==integrity['frozen_after']==item['input_hash'],'FROZEN_MUTATED')
 h=yaml_utils.load_yaml(None,SimpleNamespace(model_dir=str(v.MODEL)));pp=build_postprocessor(h['postprocess'],False)
 with np.load(p/'payload.bin',allow_pickle=False) as z:
  a={k:z[k] for k in z.files};v.verify_reference(a,meta);v.verify_attention(a,meta);v.verify_heads(a,meta,payload['batch'],pp)
  if not mock:
   v.verify_historical_control(a,sid)
   with np.load(v.P5/f'capture/sample_{sid}/alpha_1.0.npz',allow_pickle=False) as old:
    v.same(a['GT_canonical'],old['GT_canonical'],'CONTROL_GT',1e-5)
    for key in ('suppressor','first_deletion_code'):v.same(a[key],old[key],'CONTROL_'+key)
    v.same(a['suppressor_overlap'],old['suppressor_overlap'],'CONTROL_OVERLAP',1e-6)
    for key in old.files:
     if key.startswith('metaflow_raw_'):v.same(a[key],old[key],'CONTROL_RAW_OFFSET',1e-6)
    for t in (.5,.7):
     _,matched,flags=v.eval_native(old['final_corners'],old['final_scores'],old['GT_canonical'],t)
     v.same(a[f'gt_tp{int(t*100)}'],flags,'CONTROL_TP')
     v.same(a[f'prediction_gt_idx{int(t*100)}'],matched,'CONTROL_MATCH')
  if peer:
   q=Path(peer);m=json.loads((q/'manifest.json').read_text())
   for key in ('payload.bin','meta.json'):v.require(v.sha(q/key)==m['files'][key],'PEER_HASH')
   pm=json.loads((q/'meta.json').read_text());v.require(int(pm['sample_idx'])==sid and pm['condition']!=kind,'PEER_IDENTITY')
   with np.load(q/'payload.bin',allow_pickle=False) as z:
    from runner_engine import close_enough
    close_enough(a,{k:z[k] for k in z.files})
    v.same(a['suppressor'],z['suppressor'],'PAIR_SUPPRESSOR')
 v.require(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
 return {'passed':True,'sample':sid,'condition':kind,'mock_only':mock,'historical_control_checked':not mock,'cuda_initialized':False,'real_model_forward':0}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('shard');ap.add_argument('sample',type=int);ap.add_argument('condition',choices=['off','pass']);ap.add_argument('--mock',action='store_true');ap.add_argument('--peer');args=ap.parse_args()
 try:print(json.dumps(validate(args.shard,args.sample,args.condition,args.mock,args.peer)))
 except Exception as e:print(json.dumps({'passed':False,'error':type(e).__name__+': '+str(e)}));sys.exit(1)
