import os,tempfile,json,pathlib,time,sys,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
import p5_worker as w
rows=list(csv.DictReader((w.R/'P5_SAMPLE_PROPOSAL_V2.csv').open()));ids=[int(x['sample_idx']) for x in rows];assert len(ids)==len(set(ids))==18 and ids[0]==590
with tempfile.TemporaryDirectory(prefix='p5_schedule_') as d:
 old=w.R;w.R=pathlib.Path(d);(w.R/'capture').mkdir();w.state.update(attempted=0,completed=0,core=0,guards=0,completed_samples=0)
 try:
  def fake(sid,alpha,kind):
   w.guard(time.monotonic());w.update(attempted=w.state['attempted']+1);w.state['completed']+=1;w.state['core' if kind=='core' else 'guards']+=1
   return {'head':np.array([sid,alpha])}
  for a in (0.,1.):fake(590,a,'guard')
  for a in (0.,1.):w.save_case(590,a,fake(590,a,'core'),{'sid':590,'alpha':a})
  for a in (1.,0.):fake(590,a,'guard')
  for sid in ids[1:]:
   for a in (0.,1.):w.save_case(sid,a,fake(sid,a,'core'),{'sid':sid,'alpha':a})
  assert (w.state['attempted'],w.state['completed'],w.state['core'],w.state['guards'])==(40,40,36,4)
  assert len(list((w.R/'capture').glob('*/alpha_*.npz')))==36
  try:fake(590,0.,'guard')
  except RuntimeError as e:assert str(e)=='FORWARD_LIMIT'
  else:raise AssertionError('41st call allowed')
 finally:w.R=old
assert not torch.cuda.is_initialized()
(old/'SCHEDULE_TEST.json').write_text(json.dumps({'passed':True,'synthetic_calls':40,'core':36,'extra':4,'41st_rejected':True,'sample_count':18,'CUDA_initialized':False,'model_forwards':0},indent=2))
