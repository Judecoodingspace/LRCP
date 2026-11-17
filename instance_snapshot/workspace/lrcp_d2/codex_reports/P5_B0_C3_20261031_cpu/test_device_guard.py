import json
from device_guard import validate_telemetry,UUID
ok=f'0, {UUID}, PPU-ZW810E, 98304 MiB, 18 MiB, 0 %\n'
def run():
 cases={}
 def test(name,fn,expected):
  try:fn(); actual='OK'
  except RuntimeError as e:actual=str(e)
  cases[name]={'expected':expected,'actual':actual,'passed':actual==expected}
 test('idle',lambda:validate_telemetry(ok,'',20*2**30,1),'OK')
 test('busy',lambda:validate_telemetry(ok,f'{UUID}, 42, python, 1 MiB\n',20*2**30,1),'DEVICE_BUSY')
 test('identity_changed',lambda:validate_telemetry(ok.replace(UUID,'GPU-other'),'',20*2**30,1),'DEVICE_IDENTITY_MISMATCH')
 test('insufficient_disk',lambda:validate_telemetry(ok,'',1,1),'DISK_LOW')
 test('rss_limit',lambda:validate_telemetry(ok,'',20*2**30,9*2**30),'RSS_HIGH')
 test('memory_limit',lambda:validate_telemetry(ok.replace('18 MiB','98000 MiB'),'',20*2**30,1),'DEVICE_MEMORY_LOW')
 test('invalid_metrics',lambda:validate_telemetry(ok.replace('18 MiB','bad'),'',20*2**30,1),'DEVICE_TELEMETRY_INVALID')
 print(json.dumps({'passed':all(v['passed'] for v in cases.values()),'pure_synthetic':True,'device_queried':False,'runner_wired':False,'tests':cases},indent=2))
if __name__=='__main__':run()
