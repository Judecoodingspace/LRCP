"""Read-only watchdog for one CPU process; no data workers or accelerator access."""
import os,time,json,signal
from pathlib import Path
R=Path(__file__).resolve().parent
item=json.loads((R/'CPU_STARTED.json').read_text());pid=int(item['pid']);path=Path('/proc')/str(pid);hard=8*2**30
with (R/'cpu_watchdog.jsonl').open('x') as log:
 while path.exists():
  try:
   lines=(path/'status').read_text().splitlines();d=dict(x.split(':',1) for x in lines if ':' in x)
   rss=int(d['VmRSS'].split()[0])*1024;peak=int(d['VmHWM'].split()[0])*1024
   row={'epoch':time.time(),'pid':pid,'RSS_bytes':rss,'peak_RSS_bytes':peak}
   if max(rss,peak)>hard:
    row['event']='RSS_HARD_LIMIT';log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
    (R/'WATCHDOG_FAILURE.json').write_text(json.dumps({'passed':False,**row},indent=2));os.kill(pid,signal.SIGTERM);break
  except (FileNotFoundError,ProcessLookupError):break
  time.sleep(.25)
