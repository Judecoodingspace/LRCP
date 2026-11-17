"""Read-only OS/dependency audit. No torch import, data replay, inference or mount change."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,sys,json,hashlib,subprocess,importlib.util,shutil,ast
R=Path(__file__).resolve().parent
P=R.parent/'P4_PILOT_20261022_193114_346841'
PRE=R.parent/'P4_PRECHECK_20261022_175446_928190'
SOURCE=Path('/workspace/LRCP')
def dump(name,x):
 with (R/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
protected=json.loads((P/'PROTECTED_HASHES_BEFORE.json').read_text())
for directory,manifest in [(P,'ARTIFACTS.sha256'),(PRE,'ARTIFACTS.sha256')]:
 for line in (directory/manifest).read_text().splitlines():
  d,name=line.split('  ',1);file=directory/name
  assert sha(file)==d,str(file);protected[str(file)]=d
 protected[str(directory/manifest)]=sha(directory/manifest)
for path,d in protected.items():assert Path(path).is_file() and sha(path)==d,path
dump('PROTECTED_BEFORE.json',protected)
commands={
 'shm_mount':['findmnt','-T','/dev/shm','-o','TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS'],
 'filesystems_bytes':['df','-B1','/dev/shm','/workspace/lrcp_d2'],
 'workspace_mount':['findmnt','-T','/workspace/lrcp_d2','-o','TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,OPTIONS'],
 'tmp_mount':['findmnt','-T','/tmp','-o','TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL'],
 'host_memory':['free','-b'],
 'device_memory_readonly':['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],
 'git_status':['git','-C',str(SOURCE),'status','--short']}
outputs={}
for name,cmd in commands.items():
 p=subprocess.run(cmd,capture_output=True,text=True)
 outputs[name]={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
dump('COMMAND_OUTPUTS.json',outputs)
with (R/'COMMAND_OUTPUTS.txt').open('x') as f:
 for name,v in outputs.items():f.write(name+'\n$ '+' '.join(v['command'])+'\n'+v['stdout']+v['stderr']+'\n')
for name,path in [('self_cgroup','/proc/self/cgroup'),('mountinfo','/proc/self/mountinfo'),('meminfo','/proc/meminfo'),('limits','/proc/self/limits')]:
 (R/(name+'.txt')).write_text(Path(path).read_text())
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
 k,v=line.split(':',1)
 try:mem[k]=int(v.split()[0])*1024
 except ValueError:pass
cgroot=Path('/sys/fs/cgroup/memory')
names=['memory.limit_in_bytes','memory.usage_in_bytes','memory.max_usage_in_bytes','memory.memsw.limit_in_bytes','memory.memsw.usage_in_bytes','memory.soft_limit_in_bytes','memory.failcnt','memory.oom_control','memory.stat','memory.use_hierarchy']
cg={name:(cgroot/name).read_text().strip() for name in names if (cgroot/name).exists()}
stat={k:int(v) for k,v in (line.split() for line in cg['memory.stat'].splitlines())}
shm_files=[]
for p in sorted(Path('/dev/shm').iterdir()):
 s=p.stat();shm_files.append({'name':p.name,'logical_bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_epoch':s.st_mtime,'mtime_Asia_Shanghai':datetime.fromtimestamp(s.st_mtime,ZoneInfo('Asia/Shanghai')).isoformat(),'uid':s.st_uid})
usage=shutil.disk_usage('/dev/shm');wusage=shutil.disk_usage('/workspace/lrcp_d2')
processes=[];holders=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  comm=(d/'comm').read_text().strip();status=(d/'status').read_text()
  children=(d/'task'/d.name/'children').read_text().strip()
  if any(x in comm.lower() for x in ['python','torch','resource_track']):
   pp=next((s.split()[1] for s in status.splitlines() if s.startswith('PPid:')),'')
   processes.append({'pid':int(d.name),'ppid':pp,'comm':comm,'children':children})
  for fd in (d/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if '/dev/shm/' in target or 'memfd:' in target or '/SYSV' in target:holders.append({'pid':int(d.name),'comm':comm,'fd':fd.name,'target':target})
   except (OSError,PermissionError):pass
 except (OSError,PermissionError):pass
roots={}
for pkg in ['torch','numpy','spconv','cumm']:
 spec=importlib.util.find_spec(pkg)
 roots[pkg]=None if spec is None else {'origin':spec.origin,'locations':list(spec.submodule_search_locations or [])}
dump('PACKAGE_LOCATIONS.json',roots)
sources={
 'torch_dataloader':Path(roots['torch']['locations'][0])/'utils/data/dataloader.py',
 'torch_collate':Path(roots['torch']['locations'][0])/'utils/data/_utils/collate.py',
 'torch_serialization':Path(roots['torch']['locations'][0])/'serialization.py',
 'torch_reductions':Path(roots['torch']['locations'][0])/'multiprocessing/reductions.py',
 'torch_fetch':Path(roots['torch']['locations'][0])/'utils/data/_utils/fetch.py',
 'numpy_format':Path(roots['numpy']['locations'][0])/'lib/format.py',
 'numpy_npyio':Path(roots['numpy']['locations'][0])/'lib/npyio.py',
 'spconv_voxel':Path(roots['spconv']['locations'][0])/'utils/__init__.py',
 'runner':P/'runner.py','candidate_runner':P/'runner_revision_candidate.py','common':P/'pilot_common.py','collector':P/'collector.py','memory_fix':P/'bounded_offset_cache_proposal.py',
 'data_base':SOURCE/'opencood/data_utils/datasets/basedataset/v2xsim_basedataset.py','data_intermediate':SOURCE/'opencood/data_utils/datasets/intermediate_latency_fusion_dataset.py','preprocess':SOURCE/'opencood/data_utils/pre_processor/sp_voxel_preprocessor.py'}
terms=['num_workers','pin_memory','_get_iterator','_SingleProcessDataLoaderIter','get_worker_info','_new_shared','multiprocessing_context.Process','multiprocessing_context.Queue','mmap','write_record(name, storage','get_storage_from_record','_share_fd_cpu_','_share_filename_cpu_','np.load(','numpy.fromfile','np.fromfile','torch.from_numpy','np.zeros','np.full','__getitem__','__getitems__','offset_map','params\'] = cav_content','share_memory','shared_memory','/dev/shm','torch.save','torch.load','tempfile','untyped_storage','clone()','subprocess','RUSAGE']
hits=[]
with (R/'SOURCE_EVIDENCE.txt').open('x') as f:
 for label,path in sources.items():
  txt=path.read_text();protected[str(path)]=sha(path)
  f.write('\nFILE '+str(path)+' SHA256 '+sha(path)+'\n')
  lines=txt.splitlines();selected=set()
  for i,s in enumerate(lines):
   if any(term in s for term in terms):
    hits.append({'label':label,'path':str(path),'line':i+1,'text':s})
    selected.update(range(max(0,i-3),min(len(lines),i+5)))
  for i in sorted(selected):f.write(f'{i+1:5d} {lines[i]}\n')
dump('SOURCE_HITS.json',hits)
dump('PROTECTED_BEFORE.json.updated',protected)
summary={'snapshot_Asia_Shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'cgroup_version':1,'cgroup_v2_memory_max_exists':Path('/sys/fs/cgroup/memory.max').exists(),'cgroup_v2_memory_current_exists':Path('/sys/fs/cgroup/memory.current').exists(),'cgroup_v1':cg,'cgroup_stat':stat,'ordinary_memory_limit_bytes':int(cg['memory.limit_in_bytes']),'cgroup_memory_usage_bytes':int(cg['memory.usage_in_bytes']),'cgroup_nominal_headroom_before_reclaim_bytes':int(cg['memory.limit_in_bytes'])-int(cg['memory.usage_in_bytes']),'host_meminfo_bytes':mem,'shm':{'mount_type':'tmpfs','total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free,'files':shm_files,'logical_files_total_bytes':sum(x['logical_bytes'] for x in shm_files),'allocated_files_total_bytes':sum(x['allocated_bytes'] for x in shm_files)},'workspace':{'realpath':str(Path('/workspace/lrcp_d2').resolve()),'filesystem':'overlay','total_bytes':wusage.total,'free_bytes':wusage.free,'used_bytes':wusage.used},'relevant_processes':processes,'shared_memory_open_fd_holders':holders,'old_runner_pid_4025436_exists':Path('/proc/4025436').exists(),'old_runner_state':json.loads((P/'STATE.json').read_text()),'environment':{k:os.environ.get(k) for k in ['TMPDIR','TEMP','TMP','TORCH_SHM_MANAGER','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'audit_process_imported_torch':'torch' in sys.modules}
dump('ENVIRONMENT_SNAPSHOT.json',summary)
print(json.dumps({k:summary[k] for k in ['snapshot_Asia_Shanghai','cgroup_version','ordinary_memory_limit_bytes','cgroup_memory_usage_bytes','cgroup_nominal_headroom_before_reclaim_bytes','old_runner_pid_4025436_exists']},ensure_ascii=False))
print('SHM',usage,'logical_bytes',summary['shm']['logical_files_total_bytes'],'allocated_bytes',summary['shm']['allocated_files_total_bytes'])
