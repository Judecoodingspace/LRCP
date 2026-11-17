"""Idempotent, read-only completed-job acceptance entry. No torch or device APIs."""
import sys,json
import launch_control as c

def check():
 state=c.status()
 if state['state']!='VALIDATED':
  print(json.dumps({'passed':False,'state':state,'reason':'作业尚未通过最终验收；此检查不会启动或重跑任务。'},ensure_ascii=False,indent=2));return 2
 c.verify_ready(full=True,require_pointer=False)
 manifest=c.R/'PPU_FINAL_ARTIFACTS.sha256'
 if not manifest.exists():raise RuntimeError('FINAL_SEALED_MANIFEST_MISSING')
 count=0
 for line in manifest.read_text().splitlines():
  h,n=line.split(None,1);p=c.R/n.strip()
  if c.sha(p)!=h:raise RuntimeError('FINAL_ARTIFACT_CHANGED '+str(p))
  count+=1
 print(json.dumps({'passed':True,'state':'VALIDATED','checked_final_artifacts':count,'protected_inputs_and_18_batches_unchanged':True,'exit_code':0,'model_forwards_in_this_check':0,'run_directory':str(c.R)},ensure_ascii=False,indent=2));return 0
if __name__=='__main__':
 try:sys.exit(check())
 except Exception as e:print(json.dumps({'passed':False,'reason':str(e)},ensure_ascii=False));sys.exit(1)
