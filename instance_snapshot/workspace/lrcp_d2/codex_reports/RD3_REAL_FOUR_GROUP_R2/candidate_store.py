"""All candidates remain UNACCEPTED; accepted science has a separate manifest.
Raw model return is persisted first. No producer self-passed mark promotes data.
"""
from rd3_common import *
from input_integrity import snapshot
import torch

def save_npz(path,a):
 path=Path(path);temp=path.with_name(path.name+'.tmp')
 with temp.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def raw_candidate(p,heads,offsets,references,obs,history):
 root=Path(p['run_root']);ordinal=p['attempt']['ordinal'];target=root/'diagnostics'/f'candidate_{ordinal:03d}';temp=target.with_name(target.name+'.tmp');temp.mkdir(parents=True)
 a={k:array(v) for k,v in heads.items()}
 for i,v in enumerate(offsets):a[f'metaflow_slot_{i+1}']=v
 for ev in obs.events:
  for k in ('reference','offset','location','weights','spatial_shapes'):a[f'cross_{ev["block"]}_{k}']=ev[k]
 if ordinal==1:
  for i,w in obs.full_weights.items():a[f'cross_{i}_full_weights']=w
 save_npz(temp/'raw_arrays.npz',a)
 with (temp/'inputs_after.pt').open('xb') as f:torch.save(p['batch'],f);f.flush();os.fsync(f.fileno())
 write(temp/'raw_meta.json',{'state':'UNACCEPTED','stage':'MODEL_RETURNED','attempt':p['attempt'],'model_returned':True,'sample_idx':p['row']['sample_idx'],'checkpoint_sha256':CH,'source_seal_sha256':sha_file(R/'SOURCE_AND_CHECKPOINT_SEAL.json') if (R/'SOURCE_AND_CHECKPOINT_SEAL.json').exists() else None,'input_leaf_before':p['leaf_before'],'input_leaf_after':snapshot(p['batch']),'history_receipts':history.receipts,'metaflow_events':history.meta_events})
 write(temp/'manifest.json',{'state':'UNACCEPTED','stage':'MODEL_RETURNED','files':{n:sha_file(temp/n) for n in ('raw_arrays.npz','raw_meta.json','inputs_after.pt')}})
 os.rename(temp,target);fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd);p['candidate_path']=str(target)
def complete_candidate(p,a,meta):
 target=Path(p['candidate_path']);save_npz(target/'arrays.npz',a);write(target/'meta.json',meta)
 write(target/'manifest.json',{'state':'UNACCEPTED','stage':'COLLECTED','science_accepted':False,'files':{n:sha_file(target/n) for n in ('raw_arrays.npz','raw_meta.json','inputs_after.pt','arrays.npz','meta.json')}})
def commit_candidate(root,p,result_path):
 root=Path(root);source=Path(p['candidate_path']);result=load(result_path)
 require(result['passed'] and result['candidate_manifest_sha256']==sha_file(source/'manifest.json'),'INDEPENDENT_CANDIDATE_GATE')
 target=root/'capture'/f"sample_{p['row']['sample_idx']}_{p['condition']}";temp=target.with_name(target.name+'.tmp');temp.mkdir(parents=True)
 for name in ('arrays.npz','meta.json','inputs_after.pt'):os.link(source/name,temp/name)
 write(temp/'manifest.json',{'state':'ACCEPTED','backend':result['backend'],'candidate':str(source.relative_to(root)),'candidate_manifest_sha256':sha_file(source/'manifest.json'),'validation_path':str(result_path),'validation_sha256':sha_file(result_path),'files':{n:sha_file(temp/n) for n in ('arrays.npz','meta.json','inputs_after.pt')}})
 os.rename(temp,target);fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 return {'sample_idx':p['row']['sample_idx'],'condition':p['condition'],'relative_path':str(target.relative_to(root)),'manifest_sha256':sha_file(target/'manifest.json')}
