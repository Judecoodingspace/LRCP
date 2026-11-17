"""Stat-only logical I/O estimate for future legacy CPU input replay; no torch."""
from pathlib import Path
import pickle,json
R=Path(__file__).resolve().parent
x=pickle.load(open('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl','rb'))
by={i['token']:i for i in x}
total=0;reads=0;unique={};accepted={}
def add(path):
 global total,reads
 p=Path(path);size=p.stat().st_size;total+=size;reads+=1;unique[str(p)]=size
for delay in range(4):
 count=0
 for sid,i in enumerate(x):
  if delay==3 and sid>941:break
  # retrieve_base_data loads current agent scans even before history-boundary rejection.
  for agent in range(1,i['agent_num']+1):add(i[f'lidar_path_{agent}'])
  if any(i['prev_samples'][k] is None for k in range(1,delay+5)):continue
  count+=1
  for agent in range(2,i['agent_num']+1):
   for k in [delay]+list(range(delay,delay+5)):
    frame=i if k==0 else by[i['prev_samples'][k]]
    add(frame[f'lidar_path_{agent}'])
   add('/workspace/datasets/V2X-Sim-2.0/offset_maps/'+i['token']+'.npy')
 accepted[str(delay)]=count
p={'note':'Literal retrieve_base_data reads before comm-range filtering; includes repeats and current scans before invalid-history rejection. OS cache may reduce physical I/O. Stat only; no preprocessing run.', 'literal_file_reads':reads,'logical_bytes':total,'unique_files':len(unique),'unique_bytes':sum(unique.values()),'valid_history_rows_by_delay':accepted}
(R/'RESOURCE_ESTIMATE.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps(p))
