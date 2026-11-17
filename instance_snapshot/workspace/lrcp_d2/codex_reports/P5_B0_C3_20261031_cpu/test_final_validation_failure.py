"""Rehashed GT tampering rejected through formal controller validate command."""
import json,shutil,subprocess,io
from pathlib import Path
import numpy as np
from runtime_common import ROOT,PYTHON,sha,atomic,read,child_environment
source=ROOT/'mock_runs'/'e2e_release';root=ROOT/'mock_runs'/'fault_final_replay';shutil.copytree(source,root)
m=read(root/'manifest.json')
for r in m['records']:r['path']=str(root/'capture'/Path(r['path']).name)
p=Path(m['records'][0]['path'])
with np.load(p/'payload.bin',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
a['gt_tp70'][0]=1-a['gt_tp70'][0];b=io.BytesIO();np.savez_compressed(b,**a);(p/'payload.bin').write_bytes(b.getvalue())
f=read(p/'manifest.json');f['files']['payload.bin']=sha(p/'payload.bin');atomic(p/'manifest.json',f);m['records'][0]['manifest_sha256']=sha(p/'manifest.json');atomic(root/'manifest.json',m)
reserved=read(root/'budget.json')
proc=subprocess.run([PYTHON,'-B',str(ROOT/'controller.py'),'validate','--mock','--root',str(root)],capture_output=True,text=True,env=child_environment(True),timeout=300)
(ROOT/'final_validation_failure.log').write_text(proc.stdout+'\n'+proc.stderr)
state=read(root/'progress.json')['state'];result={'passed':proc.returncode!=0 and 'GT_TP70' in proc.stderr and state=='MOCK_VALIDATION_FAILED' and read(root/'budget.json')==reserved,'rehashed_semantic_error':'GT_TP70','formal_command':'controller.py validate --mock','state':state,'budget_unchanged':read(root/'budget.json')==reserved,'MOCK_ONLY':True}
atomic(ROOT/'C3_FINAL_VALIDATION_FAILURE_TEST.json',result);print(json.dumps(result));assert result['passed']
