import os, sys, json, subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from test_multiline import Toy, Observer, q,p,value,ref,sh
import p5_control as ctl
mods=[Toy().eval() for _ in range(4)]
try:
 with Observer(mods,[0]) as ob:
  with torch.no_grad():
   mods[0](q,p,value,ref,sh)
   mods[0](q,p,value,ref,sh)
except RuntimeError as e:
 assert str(e)=='DUPLICATE_CROSS_CALL'
else:raise AssertionError('actual duplicate module call was not rejected')
assert sys.gettrace() is None and all(not m.sampling_offsets._forward_hooks for m in mods)
class Child:
 def __init__(self,rc):self.rc=rc;self.waited=False
 def poll(self):return self.rc
 def wait(self,timeout):self.waited=True;return self.rc
c=Child(1)
assert ctl.watchdog_rss('State:\tZ (zombie)\n',c) is None and c.waited
assert ctl.watchdog_rss('State:\tS (sleeping)\nVmRSS:\t1234 kB\n',Child(None))==1234*1024
assert ctl.watchdog_rss('State:\tS (sleeping)\n',Child(1)) is None
try:ctl.watchdog_rss('State:\tS (sleeping)\n',Child(None))
except RuntimeError as e:assert str(e)=='CPU_RSS_WATCHDOG_UNAVAILABLE'
else:raise AssertionError('missing RSS for live child accepted')
assert not torch.cuda.is_initialized()
Path(__file__).with_name('REGRESSION_TEST.json').write_text(json.dumps({'passed':True,'multiline_observer':True,'real_duplicate_rejected':True,'observer_cleanup':True,'watchdog_zombie_reaped':True,'watchdog_live_missing_rss_rejected':True,'CUDA_initialized':False,'real_model_forwards':0},indent=2))
