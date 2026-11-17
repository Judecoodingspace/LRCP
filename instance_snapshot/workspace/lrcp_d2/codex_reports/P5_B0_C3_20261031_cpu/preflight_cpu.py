#!/usr/bin/env python3
"""Complete frozen-source/file and CPU semantic preflight; no real forward."""
import csv,json,sys,os,shutil,importlib
from pathlib import Path
from runtime_common import ROOT,C2,C1,P5,FULL,BUDGET,sha,read

def inspect(semantic=True):
 errors=[];checks={}
 try:
  for stage in (C1,C2):
   count=0
   for line in (stage/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1);p=Path(name);p=p if p.is_absolute() else stage/p
    if sha(p)!=expected:raise RuntimeError('HISTORICAL_SOURCE_HASH '+str(p))
    count+=1
   checks[stage.name+'_checksum_files']=count
  if not read(C1/'BACKEND_STATIC_VALIDATION.json')['passed'] or not read(C2/'C2_VALIDATION.json')['passed']:raise RuntimeError('C1_C2_ACCEPTANCE')
  ready=read(P5/'P5_READY.json');locked=ready['locked_files']
  for p,h in locked.items():
   if sha(p)!=h:raise RuntimeError('LOCKED_SOURCE_HASH '+p)
  checks['P5A_locked_files']=len(locked)
  selected=list(csv.DictReader((C2/'P5_SAMPLE_PROPOSAL_V2.csv').open()));ids=[int(x['sample_idx']) for x in selected]
  if len(ids)!=18 or len(set(ids))!=18 or set(ids)!={int(x['sample_idx']) for x in ready['samples']}:raise RuntimeError('SAMPLE_SET')
  if sha(C2/'P5_SAMPLE_PROPOSAL_V2.csv')!=sha(P5/'P5_SAMPLE_PROPOSAL_V2.csv'):raise RuntimeError('SAMPLE_SELECTION_HASH')
  for x in ready['samples']:
   if sha(x['path'])!=x['sha256']:raise RuntimeError('INPUT_FILE_HASH '+str(x['sample_idx']))
  checks['sample_count']=18
  acceptance=read(P5/'RESEARCH_ACCEPTED.json');capture=read(P5/'PPU_CAPTURE_COMPLETE.json')
  if not acceptance['passed'] or sha(P5/'CPU_VALIDATION_COMPLETE.json')!=acceptance['CPU_validation_sha256'] or sha(P5/'PPU_CAPTURE_COMPLETE.json')!=acceptance['PPU_capture_sha256'] or sha(P5/'CAPTURE_MANIFEST.json')!=capture['manifest_sha256']:raise RuntimeError('P5A_SCIENTIFIC_ACCEPTANCE')
  manifest=read(P5/'CAPTURE_MANIFEST.json');selected_records=[r for r in manifest['records'] if int(r['sample_idx']) in ids]
  if len(selected_records)!=36:raise RuntimeError('HISTORICAL_CASE_COUNT')
  for r in selected_records:
   p=Path(r['path'])
   if sha(p)!=r['sha256'] or sha(p.with_suffix('.json'))!=r['meta_sha256']:raise RuntimeError('P5A_BASELINE_HASH')
  checks['historical_records_hashed']=len(selected_records)
  resource_evidence=read(P5/'P5_READY.json')['budget']
  for ours,theirs in [('cpu_rss_bytes','CPU_RSS_bytes'),('min_disk_bytes',None),('min_free_device_mib','PPU_min_free_bytes'),('wall_seconds','PPU_wall_seconds'),('capture_max_bytes','capture_bytes'),('device_reserved_max_bytes','PPU_reserved_peak_bytes'),('device_allocated_max_bytes','PPU_allocated_peak_bytes')]:
   if theirs:
    value=resource_evidence[theirs]//2**20 if ours=='min_free_device_mib' else resource_evidence[theirs]
    if BUDGET[ours]!=value:raise RuntimeError('RESOURCE_PROTOCOL '+ours)
  checks['resource_policy']='P5-A validated limits; disk10GiB from p5_control.verify_ready'
  sys.path.insert(0,str(C2));sys.path.insert(0,'/workspace/LRCP')
  import torch
  from pilot_common import digest
  import real_backend as rb
  if sha(rb.CHECKPOINT)!=rb.CHECKPOINT_SHA:raise RuntimeError('CHECKPOINT_HASH')
  for module in ('runner_engine','intervention','real_backend','attention_observer','minimal_collector','opencood.tools.train_utils','opencood.data_utils.datasets','opencood.hypes_yaml.yaml_utils'):
   importlib.import_module(module)
  checks['dependency_imports']=True
  if semantic:
   for x in ready['samples']:
    payload=torch.load(x['path'],map_location='cpu',weights_only=False,mmap=False)
    if digest(payload['batch'])!=x['input_hash'] or digest(payload['rng'])!=x['RNG_hash']:raise RuntimeError('INPUT_SEMANTIC_HASH')
    del payload
   checks['input_and_RNG_semantic_hashes']=18
  if torch.cuda.is_initialized():raise RuntimeError('CUDA_INITIALIZED')
  checks['checkpoint_sha256']=rb.CHECKPOINT_SHA
  meminfo={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
  headroom=meminfo['MemAvailable']
  for lim,cur in [('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes'),('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current')]:
   if Path(lim).exists() and Path(lim).read_text().strip()!='max':headroom=min(headroom,int(Path(lim).read_text())-int(Path(cur).read_text()))
  if headroom<BUDGET['cpu_rss_bytes']:raise RuntimeError('CPU_MEMORY_HEADROOM')
  checks['cpu_memory_headroom_bytes']=headroom
  checks['cpu_headroom_note']='8GiB headroom and runtime 8GiB process-tree RSS limit'
  if shutil.disk_usage(ROOT).free<BUDGET['min_disk_bytes']:raise RuntimeError('DISK_LOW')
  checks['disk_free_bytes']=shutil.disk_usage(ROOT).free
 except Exception as e:errors.append(type(e).__name__+': '+str(e))
 return {'passed':not errors,'errors':errors,'checks':checks,'CUDA_initialized':False,'real_model_forward':0,'PPU_started':False,'ready':read(ROOT/'P5_B0_READY.json',{}).get('passed',False)}
if __name__=='__main__':
 result=inspect();print(json.dumps(result,indent=2,ensure_ascii=False));sys.exit(0 if result['passed'] else 1)
