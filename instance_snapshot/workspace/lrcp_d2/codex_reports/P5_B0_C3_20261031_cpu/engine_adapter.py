"""Runtime guards around UNMODIFIED C2 runner_engine.run in one worker process."""
import os,sys,json,time,subprocess,shutil
from pathlib import Path
from runtime_common import ROOT,C2,PYTHON,BUDGET,atomic,read,sha,rss,child_environment
from device_guard import query_live,validate_telemetry,UUID

def resource_check(mock,root,fault='',own_pid=None):
 disk=shutil.disk_usage(root).free;used=rss(os.getpid())
 if mock:
  gpu=f'0, {UUID}, PPU-ZW810E, 98304 MiB, 2 MiB, 0 %\n';procs=''
  if fault=='device_identity':gpu=gpu.replace(UUID,'GPU-WRONG')
  if fault=='device_busy':procs=f'{UUID}, 999, fake, 1 MiB\n'
  if fault=='device_query':raise RuntimeError('DEVICE_QUERY_FAILED')
  if fault=='disk':disk=0
  if fault=='rss':used=BUDGET['cpu_rss_bytes']+1
  return validate_telemetry(gpu,procs,disk,used,own_pid)
 return query_live(disk,used,own_pid)

class Execution:
 def __init__(self,root,backend,mock,fault=''):
  self.root=Path(root);self.backend=backend;self.mock=mock;self.fault_name=fault;self.current=None;self.records=[];self.controls=[];self.started=0;self.finished=0
  self.progress={'state':'MOCK_RUNNING_CONTROLS' if mock else 'PPU_RUNNING_CONTROLS','mock_only':mock,'run_id':self.root.name,'started':0,'completed':0,'core_completed':0,'committed':0,'controls_verified':0}
 def event(self,event,**fields):
  record={'epoch':time.time(),'event':event,**fields}
  with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  self.progress.update(fields);self.progress['last_event']=event;self.progress['updated_epoch']=record['epoch'];atomic(self.root/'progress.json',self.progress)
 def check(self):
  fault=self.fault_name
  if fault.startswith('core_') and fault in ('core_disk','core_rss'):fault=fault[5:] if self.current and self.current[1]=='core' else ''
  telemetry=resource_check(self.mock,self.root,fault,os.getpid() if not self.mock else None)
  self.event('resources',device=telemetry,cpu_rss_bytes=rss(os.getpid()),disk_free_bytes=shutil.disk_usage(self.root).free)
  return telemetry
 def run(self):
  sys.path.insert(0,str(C2))
  import runner_engine as engine
  original_reserve,original_commit=engine.reserve,engine.commit
  outer=self
  class Proxy:
   kind=outer.backend.kind
   def prepare(self,sid,kind):
    outer.current=(sid,kind)
    if kind=='core' and len(outer.controls)!=4:raise RuntimeError('CORE_BEFORE_CONTROLS')
    outer.check()
    outer.event('prepare',sample_idx=sid,condition=kind,state=('MOCK_' if outer.mock else 'PPU_')+('RUNNING_CORE' if kind=='core' else 'RUNNING_CONTROLS'))
    return outer.backend.prepare(sid,kind)
   def forward(self,p,sid,kind):
    outer.started+=1;outer.event('forward_enter',started=outer.started)
    if outer.mock and outer.fault_name=='forward_crash':os._exit(83)
    out=outer.backend.forward(p,sid,kind)
    outer.finished+=1;outer.event('forward_return',completed=outer.finished)
    if not outer.mock:
     import torch
     allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
     outer.event('device_allocator',PPU_allocated_bytes=allocated,PPU_reserved_bytes=reserved)
     if reserved>BUDGET['device_reserved_max_bytes']:raise RuntimeError('DEVICE_RESERVED_LIMIT')
     if allocated>BUDGET['device_allocated_max_bytes']:raise RuntimeError('DEVICE_ALLOCATED_LIMIT')
    return out
   def collect(self,*args):return outer.backend.collect(*args)
   def cleanup(self,p):
    outer.backend.cleanup(p)
    mods=p.get('modules',[])
    if any(m._forward_hooks or m._forward_pre_hooks for m in mods):raise RuntimeError('HOOKS_NOT_CLEANED')
    outer.event('hooks_cleaned')
   def baseline(self,sid):return outer.backend.baseline(sid)
   def fault(self,stage):
    if not outer.mock:return
    if stage=='after_reserve' and outer.fault_name=='core_reserved_crash' and outer.current[1]=='core':os._exit(84)
    if stage=='during_write' and outer.fault_name=='write_crash':os._exit(85)
    if stage=='after_commit' and outer.fault_name=='commit_crash':os._exit(86)
  def reserved(root,sid,kind):
   outer.check()
   n=original_reserve(root,sid,kind);outer.event('reserved',reserved=n)
   return n
  def committed(root,shard_id,data,meta,fault=None):
   folder=original_commit(root,shard_id,data,meta,fault=fault)
   sid,kind=int(meta['sample_idx']),meta['condition']
   rec={'sample_idx':sid,'condition':kind,'path':str(folder),'manifest_sha256':sha(folder/'manifest.json')}
   outer.records.append(rec);atomic(outer.root/'progress_manifest.json',{'kind':outer.backend.kind,'records':outer.records})
   outer.event('shard_committed',committed=len(outer.records),core_completed=sum(x['condition']=='core' for x in outer.records))
   if kind in ('off','pass'):
    if outer.mock and outer.fault_name=='control_mismatch':
     # Modify a scientific field AND rehash. The independent checker must reject it.
     import numpy as np,io
     with np.load(folder/'payload.bin',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
     a['cls_preds'][0,0,0,0]=8.;b=io.BytesIO();np.savez_compressed(b,**a);(folder/'payload.bin').write_bytes(b.getvalue())
     m=read(folder/'manifest.json');m['files']['payload.bin']=sha(folder/'payload.bin');atomic(folder/'manifest.json',m)
    cmd=[PYTHON,'-B',str(ROOT/'control_validate_cpu.py'),str(folder),str(sid),kind]
    if outer.mock:cmd.append('--mock')
    peer=next((x for x in outer.controls if x['sample_idx']==sid),None)
    if peer:cmd+=['--peer',peer['path']]
    res=subprocess.run(cmd,capture_output=True,text=True,env=child_environment(True),timeout=300)
    (outer.root/f'control_{sid}_{kind}.log').write_text(res.stdout+'\n'+res.stderr)
    if res.returncode:raise RuntimeError('CONTROL_INDEPENDENT_FAILED '+res.stdout.strip()[-800:])
    check=json.loads(res.stdout.strip().splitlines()[-1])
    if not check.get('passed'):raise RuntimeError('CONTROL_INDEPENDENT_REJECTED')
    outer.controls.append(rec);outer.event('control_verified',controls_verified=len(outer.controls))
    atomic(outer.root/'independent_control_gate.json',{'passed':len(outer.controls)==4,'mock_only':outer.mock,'controls':outer.controls})
    if outer.mock and outer.fault_name=='control_crash' and len(outer.controls)==4:os._exit(87)
   elif len(outer.controls)!=4:raise RuntimeError('CORE_BEFORE_CONTROLS')
   return folder
  engine.reserve=reserved;engine.commit=committed
  try:
   records=engine.run(self.root,C2/'P5_SAMPLE_PROPOSAL_V2.csv',Proxy())
   self.event('capture_complete',state='MOCK_CAPTURE_COMPLETE' if self.mock else 'PPU_CAPTURE_COMPLETE')
   return records
  except BaseException as e:
   phase='MOCK_FAILED' if self.mock else ('PPU_STOPPED_CONTROL_FAILED' if len(self.controls)<4 else 'PPU_CAPTURE_FAILED')
   self.event('capture_failed',state=phase,error=str(e));raise
  finally:
   engine.reserve=original_reserve;engine.commit=original_commit
