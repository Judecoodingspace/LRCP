"""CPU-only R1 producer entry: no real constructor/worker/device path exists."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_only
cpu_only.install()
from rd3_common import *
import torch,fcntl,argparse,gc,traceback
from mock_backend import build,NativeCallTrace
from diagnostics import accepted_record

def atomic_case(root,name,a,meta,batch):
 target=Path(root)/'capture'/name;target.parent.mkdir(exist_ok=True);temp=target.with_name(name+'.tmp');temp.mkdir()
 with (temp/'arrays.npz').open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 with (temp/'inputs_after.pt').open('xb') as f:torch.save(batch,f);f.flush();os.fsync(f.fileno())
 write(temp/'meta.json',meta);write(temp/'manifest.json',{'kind':'MOCK_ONLY_CPU_R1','files':{k:sha_file(temp/k) for k in ('arrays.npz','inputs_after.pt','meta.json')}})
 os.rename(temp,target);fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 return {'relative_path':str(target.relative_to(root)),'manifest_sha256':sha_file(target/'manifest.json'),'sample_idx':meta['sample_idx'],'condition':meta['condition'],'mask_installed':meta['mask_installed']}

def launch(root,*,fault=None):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 with (root/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  require(not (root/'claim.json').exists(),'PERSISTENT_CPU_CLAIM_NO_REENTRY')
  sources={str(R/name):sha_file(R/name) for name in ('runner.py','backend.py','mock_backend.py','input_integrity.py','intervention.py','diagnostics.py','cpu_only.py','rd3_common.py')}
  write(root/'claim.json',{'pid':os.getpid(),'kind':'CPU_MOCK_ONLY','real_forward_increment':0,'source_hashes':sources},exclusive=True)
  backend=build();row=load(FROZEN)['samples'][0]
  ledger={'kind':'CPU_MOCK_ONLY','attempts':[],'records':[],'real_forward_increment':0};write(root/'ledger.json',ledger)
  try:
   for condition,mask in [('F',False),('F',True),('G-B',True),('G-C',True),('E',True)]:
    cpu_guard();p=backend.prepare(row);ordinal=len(ledger['attempts'])+1
    attempt={'ordinal':ordinal,'sample_idx':row['sample_idx'],'condition':condition,'mask_installed':mask,'status':'RESERVED_CPU_MOCK'}
    ledger['attempts'].append(attempt);write(root/'ledger.json',ledger)
    p.update(run_root=root,attempt=attempt,fault=fault)
    if fault:backend.model.fault=fault
    def event(status):attempt['status']=status;write(root/'ledger.json',ledger)
    p['on_model_event']=event
    with NativeCallTrace() as trace:a,meta=backend.forward_collect(p,condition,install_mask=mask)
    meta['native_calls']=dict(trace.counts);meta['run_source_hashes']=sources
    require(trace.counts['native_get_batch_pillar_features']==2 and trace.counts['native_PillarVFE']==3 and trace.counts['native_Scatter']==3 and trace.counts['native_history_projection']==2,'NATIVE_HISTORY_PATH_NOT_EXECUTED')
    rec=atomic_case(root,('F_unhooked' if not mask else condition),a,meta,p['batch']);ledger['records'].append(rec);write(root/'ledger.json',ledger)
    print('CPU_CASE',condition,'mask',mask,'native_calls',meta['native_calls'],flush=True)
    del p,a,meta;gc.collect()
   write(root/'producer.json',{'passed':True,'kind':'CPU_MOCK_ONLY','records':len(ledger['records']),**cpu_only.evidence()},exclusive=True)
  except BaseException as e:
   write(root/'failure.json',{'state':'CPU_TEST_REJECTED','reason':str(e),'traceback':traceback.format_exc(),'diagnostic_path':p.get('diagnostic_path') if 'p' in locals() else None,**cpu_only.evidence()});raise
 return ledger
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--fault');args=p.parse_args();launch(args.root,fault=args.fault)
