import csv,os,tempfile,shutil,json
from pathlib import Path
from cpu_components import reserve,commit,verify_shard
rows=list(csv.DictReader(Path(__file__).with_name('P5_SAMPLE_PROPOSAL_V2.csv').open()))
ids=[int(r['sample_idx']) for r in rows];assert len(ids)==len(set(ids))==18 and 590 in ids and 918 in ids

def run(root,fail=False):
 controls=((590,'off'),(590,'pass'),(918,'pass'),(918,'off'));result={}
 for sample,kind in controls:
  reserve(root,sample,kind)
  # Synthetic values, not a test of model predictions.
  result[(sample,kind)]=('bad' if fail and sample==918 and kind=='off' else 'same')
 for sample in (590,918):
  if result[(sample,'off')]!=result[(sample,'pass')]:raise RuntimeError('CONTROL_FAILED')
 for sample in ids:
  reserve(root,sample,'core');commit(root,sample,('sample='+str(sample)).encode(),{'mock':True,'sample':sample})
 assert len(json.loads((root/'budget.json').read_text())['attempts'])==22
 assert all(verify_shard(root/'capture'/f'sample_{sample}') for sample in ids)
 return True
roots=[Path(tempfile.mkdtemp(prefix='b0_flow_')) for _ in range(2)]
try:
 assert run(roots[0]);
 try:run(roots[1],True)
 except RuntimeError as e:assert str(e)=='CONTROL_FAILED'
 else:raise AssertionError('failed control accepted')
 assert len(json.loads((roots[1]/'budget.json').read_text())['attempts'])==4 and not (roots[1]/'capture').exists()
 Path(__file__).with_name('MOCK_WORKFLOW_TEST.json').write_text(json.dumps({'passed':True,'mock_only':True,'selected_samples':18,'control_order':[[590,'off'],[590,'pass'],[918,'pass'],[918,'off']],'control_failure_blocks_core':True,'synthetic_forward_reservations':22,'real_model_forwards':0,'CUDA_initialized':False},indent=2))
finally:
 for r in roots:shutil.rmtree(r)
