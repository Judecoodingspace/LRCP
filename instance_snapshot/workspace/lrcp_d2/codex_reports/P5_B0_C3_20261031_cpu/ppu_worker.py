#!/usr/bin/env python3
"""One C3 worker, real and MOCK_ONLY share Execution -> C2 runner_engine.run."""
import os,sys,json,argparse
from pathlib import Path
from runtime_common import ROOT,C2,BUDGET,read,atomic,identity,alive,child_environment

def worker(root,mock,token,fault=''):
 root=Path(root).resolve();owner=read(root/'owner.json',{})
 if owner.get('token')!=token or owner.get('mock_only')!=mock or not alive(owner.get('supervisor')):raise RuntimeError('OWNER_CLAIM_MISMATCH')
 if os.getppid()!=owner['supervisor']['pid']:raise RuntimeError('WORKER_PARENT_MISMATCH')
 with (root/'worker_started_once.json').open('x') as f:
  json.dump(identity(os.getpid()),f);f.flush();os.fsync(f.fileno())
 if not mock and (root!=ROOT/'real_run' or fault or os.environ.get('P5_B0_OWNER_TOKEN')!=token):raise RuntimeError('REAL_WORKER_AUTHORIZATION')
 atomic(root/'worker_identity.json',identity(os.getpid()))
 sys.path.insert(0,str(C2))
 from engine_adapter import Execution
 if mock:
  os.environ['CUDA_VISIBLE_DEVICES']=''
  from mock_factory import build_mock
  backend=build_mock()
 else:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('DEVICE_MAPPING')
  from real_backend import LRCPBackend
  backend=LRCPBackend.build_real('cuda:0')
  checkpoint=read(C2/'CHECKPOINT_IDENTITY.json')
  if backend.parameter_digest!=checkpoint['parameter_digest'] or backend.buffer_digest!=checkpoint['buffer_digest']:raise RuntimeError('CHECKPOINT_RUNTIME_IDENTITY')
 execution=Execution(root,backend,mock,fault)
 if mock and fault=='budget_exhausted':
  from cpu_components import atomic_json
  atomic_json(root/'budget.json',{'max':22,'attempts':[{'sample':i,'kind':'core'} for i in range(22)]})
 execution.run()
 if mock:
  import torch
  if torch.cuda.is_initialized():raise RuntimeError('MOCK_CUDA_INITIALIZED')
 return 0
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--token',required=True);a.add_argument('--mock',action='store_true');a.add_argument('--fault',default='');args=a.parse_args()
 try:sys.exit(worker(args.root,args.mock,args.token,args.fault))
 except BaseException as e:
  if isinstance(e,SystemExit):raise
  import traceback;traceback.print_exc();sys.exit(1)
