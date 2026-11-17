from control import *
from telemetry import observe
import argparse

def main():
 p=argparse.ArgumentParser(description='Read-only PPU monitor. Ctrl+C only stops this observer.');p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=5);a=p.parse_args();require(5<=a.interval<=10,'INTERVAL_5_TO_10_SECONDS')
 while True:
  result=observe();print(json.dumps(result,ensure_ascii=False),flush=True)
  if a.once or result['display_state'] in ('COMPLETED','FAILED','STOPPED'):return
  time.sleep(a.interval)
if __name__=='__main__':
 try:main()
 except KeyboardInterrupt:print('MONITOR_STOPPED; inference process unchanged.',flush=True)
