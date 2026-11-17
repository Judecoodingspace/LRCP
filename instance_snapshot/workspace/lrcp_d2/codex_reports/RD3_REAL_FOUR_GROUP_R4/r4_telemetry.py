"""Pure read-only vendor telemetry, no torch and no budget/lock mutations."""
from control import *
from runtime_helpers import tree_rss
from device_guard import UUID,NAME,validate_telemetry
import csv,io,datetime
TOOL='/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi'
def observe():
 def call(args):
  r=subprocess.run([TOOL,*args],capture_output=True,text=True,timeout=10)
  require(r.returncode==0,'DEVICE_QUERY_FAILED '+r.stderr);return r.stdout
 gpu=call(['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
 processes=call(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])
 rows=[r for r in csv.reader(io.StringIO(gpu),skipinitialspace=True) if len(r)==6 and r[1].strip()==UUID]
 require(len(rows)==1 and rows[0][0].strip()=='0' and rows[0][2].strip()==NAME,'DEVICE_IDENTITY_MISMATCH')
 row=rows[0];total,used,util=[int(row[i].split()[0]) for i in (3,4,5)]
 require(0<=used<=total and 0<=util<=100,'DEVICE_TELEMETRY_INVALID')
 st=status();pid=st['worker_pid'] or st['owner_pid'];rss=tree_rss(pid) if pid and Path('/proc',str(pid)).exists() else 0
 return {'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':{'index':0,'uuid':UUID,'name':NAME,'total_mib':total,'used_mib':used,'utilization_percent':util},'experiment_pid':pid,'process_tree_RSS_bytes':rss,'disk_free_bytes':shutil.disk_usage(ROOT).free,'output_unique_inode_bytes':used_bytes(REAL_ROOT),'runtime':st,'display_state':'WAITING_FOR_MANUAL_START' if not (REAL_ROOT/'claim.json').exists() else st['state'],'raw_gpu_csv':gpu,'raw_process_csv':processes}
