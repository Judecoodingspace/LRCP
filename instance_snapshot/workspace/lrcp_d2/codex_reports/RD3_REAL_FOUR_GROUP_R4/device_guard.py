"""Fail-closed parser for P5-A's nvidia-smi-compatible PPU telemetry.
Pure CPU: callers provide text captured by a separately authorized monitor.
Connected to C3 supervisor and per-attempt adapter; vendor queries only.
"""
import csv, io
UUID='GPU-019e1215-c270-091c-0000-0000c0610973'
NAME='PPU-ZW810E'
MIN_FREE_MIB=4096

def validate_telemetry(gpu_csv,process_csv,free_disk_bytes,cpu_rss_bytes,own_pid=None):
 rows=list(csv.reader(io.StringIO(gpu_csv),skipinitialspace=True))
 selected=[r for r in rows if len(r)==6 and r[1].strip()==UUID]
 if len(selected)!=1:raise RuntimeError('DEVICE_IDENTITY_MISMATCH')
 r=selected[0]
 if r[0].strip()!='0' or r[2].strip()!=NAME:raise RuntimeError('DEVICE_IDENTITY_MISMATCH')
 try: total,used,util=(int(r[i].strip().split()[0]) for i in (3,4,5))
 except (ValueError,IndexError):raise RuntimeError('DEVICE_TELEMETRY_INVALID')
 if min(total,used,util)<0 or used>total or util>100:raise RuntimeError('DEVICE_TELEMETRY_INVALID')
 if total-used<MIN_FREE_MIB:raise RuntimeError('DEVICE_MEMORY_LOW')
 processes=list(csv.reader(io.StringIO(process_csv),skipinitialspace=True))
 for p in processes:
  if len(p)<2:raise RuntimeError('PROCESS_TELEMETRY_INVALID')
  if p[0].strip()==UUID:
   if own_pid is None or p[1].strip()!=str(own_pid):raise RuntimeError('DEVICE_BUSY')
   if len(p)!=4:raise RuntimeError('PROCESS_TELEMETRY_INVALID')
   try:process_mib=int(p[3].strip().split()[0])
   except (ValueError,IndexError):raise RuntimeError('PROCESS_MEMORY_UNAVAILABLE')
   if process_mib>12288:raise RuntimeError('DEVICE_PROCESS_MEMORY_LIMIT')
 if util>0 and own_pid is None:raise RuntimeError('DEVICE_BUSY')
 if free_disk_bytes<10*2**30:raise RuntimeError('DISK_LOW')
 if cpu_rss_bytes>8*2**30:raise RuntimeError('RSS_HIGH')
 return {'uuid':UUID,'name':NAME,'total_mib':total,'used_mib':used,'utilization_percent':util}

def query_live(free_disk_bytes,cpu_rss_bytes,own_pid=None):
 """Read vendor compatibility monitor without importing torch or initializing CUDA."""
 import subprocess
 tool='/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi'
 def call(args):
  try:r=subprocess.run([tool,*args],capture_output=True,text=True,timeout=10,check=True)
  except (OSError,subprocess.SubprocessError) as e:raise RuntimeError('DEVICE_QUERY_FAILED') from e
  return r.stdout
 gpu=call(['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
 proc=call(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])
 return validate_telemetry(gpu,proc,free_disk_bytes,cpu_rss_bytes,own_pid)
