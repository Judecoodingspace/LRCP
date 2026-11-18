"""Read-only deployment/candidate readiness report. Does not import torch/LRCP."""
from pathlib import Path
import argparse,ast,json,sys,platform,os,hashlib,csv,importlib.metadata as md,subprocess
CHECKPOINT='666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu','model'],default='cpu');p.add_argument('--workspace',type=Path,default=Path('/workspace'));a=p.parse_args();w=a.workspace;checks=[];warnings=[]
 def ck(name,ok,detail=''):checks.append({'name':name,'pass':bool(ok),'detail':detail})
 ck('Python_3_10',sys.version_info[:2]==(3,10),sys.version);ck('x86_64',platform.machine()=='x86_64',platform.machine())
 v=os.statvfs('/dev/shm');shm=v.f_frsize*v.f_blocks
 if shm<8*2**30:warnings.append('dev_shm_below_8GiB: keep audited DataLoader workers=0; increasing RAM/disk does not enlarge shm')
 repo=w/'LRCP';ck('repo_present',(repo/'.git').exists())
 loc=w/'lrcp_d2/codex_reports/RD3_R5B_SAMPLE_UTILITY/R5B_RAW_SHARD_LOCATOR.csv';raw=[]
 if loc.is_file():
  for r in csv.DictReader(loc.open()):
   path=w/Path(r['absolute_path']).relative_to('/workspace');raw.append(path);ck('raw_source_present_'+r['sample_idx']+'_'+r['condition'],path.is_file())
 ck('raw32_locator',len(raw)==32)
 path=repo/'opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth';ck('checkpoint_SHA',path.is_file() and sha(path)==CHECKPOINT)
 z=w/'lrcp_d2/codex_reports/RD3_R5D_GEOMETRY_SCORE_COUNTERFACTUAL_v2_20261116/R5D_UPLOAD_BUNDLE.zip';ck('portable_R5D_bundle_present',z.is_file())
 loader=[]
 for name in ['train_w_delay.py','inference_w_delay.py','train.py','inference.py']:
  f=repo/'opencood/tools'/name
  if not f.is_file():continue
  for n in ast.walk(ast.parse(f.read_text())):
   if isinstance(n,ast.Call) and (isinstance(n.func,ast.Name) and n.func.id=='DataLoader' or isinstance(n.func,ast.Attribute) and n.func.attr=='DataLoader'):
    vals={k.arg:ast.literal_eval(k.value) for k in n.keywords if k.arg in ['num_workers','prefetch_factor','persistent_workers','pin_memory'] and isinstance(k.value,ast.Constant)};loader.append({'file':name,'line':n.lineno,**vals})
 for x in loader:
  if x['file'] in ['train_w_delay.py','inference_w_delay.py']:ck('audited_loader_'+x['file']+'_'+str(x['line']),x.get('num_workers')==0 and x.get('prefetch_factor') is None and not x.get('persistent_workers',False),x)
 versions={}
 for name in ['numpy','Shapely','PyYAML','torch','torchvision','spconv','MultiScaleDeformableAttention']:
  try:versions[name]=md.version(name)
  except md.PackageNotFoundError:versions[name]=None
 if a.mode=='cpu':
  for n,exp in [('numpy','1.23.5'),('Shapely','1.8.5.post1'),('PyYAML','6.0')]:ck('CPU_package_'+n,versions[n]==exp,versions[n])
 else:
  for n,exp in [('torch','2.6.0'),('torchvision','0.21.0'),('spconv','1.2.1'),('MultiScaleDeformableAttention','1.0')]:ck('runtime_metadata_'+n,versions[n]==exp,versions[n])
  release=Path('/usr/local/PPU_SDK/release.yaml');ck('PPU_SDK',release.is_file() and '1.6.1-14620b' in release.read_text())
  for n in ['sweeps','maps','v2.0','offset_maps']:ck('dataset_'+n,(w/'datasets/V2X-Sim-2.0'/n).is_dir())
  arr=list((w/'datasets/V2X-Sim-2.0/offset_maps').glob('*.npy'));ck('offset_count_bytes',len(arr)==10000 and sum(x.stat().st_size for x in arr)==37505280000)
  for n in ['train','val','test']:ck('index_'+n,(w/'datasets/v2xsim2_info_generated'/('v2xsim_infos_'+n+'.pkl')).is_file())
 warnings.extend(['Metadata/preflight PASS does not prove accelerator operator execution or full historical read-only validation.','Some generic train/inference entrypoints still use workers=4; choose the audited delay entrypoint or perform a separately reviewed runtime adaptation.'])
 out={'state':'PREFLIGHT_PASS' if all(c['pass'] for c in checks) else 'PREFLIGHT_INCOMPLETE','mode':a.mode,'checks':checks,'warnings':warnings,'dev_shm_bytes':shm,'package_versions':versions,'loader_settings':loader,'torch_imported':'torch' in sys.modules,'new_forward':0,'CUDA_initializations':0}
 print(json.dumps(out,ensure_ascii=False,indent=2));sys.exit(0 if out['state']=='PREFLIGHT_PASS' else 2)
if __name__=='__main__':main()
