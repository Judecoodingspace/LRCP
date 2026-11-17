"""Read-only input locks and old failure-state check. No torch import."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
inputs=json.loads((R/'inputs.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
for p,h in {**inputs['history_hashes'],**inputs['protected_hashes']}.items():assert sha(p)==h,p
p=Path(inputs['pilot_directory']);assert json.loads((p/'WORKER_FINAL.json').read_text())['state']=='FAILED';assert json.loads((p/'EXIT_CODE.json').read_text())['exit_code']==1
print('INPUT_HASHES_AND_OLD_FAILED_RECORD_VERIFIED')
