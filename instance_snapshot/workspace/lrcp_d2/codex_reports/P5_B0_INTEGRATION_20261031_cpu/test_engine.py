import os,tempfile,shutil,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from runner_engine import run
from mock_backend import MockBackend
from cpu_components import verify_shard,sha
from mock_integrity_validator import validate
import csv
import torch
r=Path(__file__).parent;roots=[Path(tempfile.mkdtemp(prefix='b0_e2e_')) for _ in range(2)]
try:
 b=MockBackend();records=run(roots[0],r/'P5_SAMPLE_PROPOSAL_V2.csv',b)
 assert b.forward_count==22 and len(records)==22 and all(verify_shard(z['path']) and sha(Path(z['path'])/'manifest.json')==z['manifest_sha256'] for z in records)
 assert json.loads((roots[0]/'state.json').read_text())['state']=='MOCK_COMPLETE'
 ids=[int(x['sample_idx']) for x in csv.DictReader((r/'P5_SAMPLE_PROPOSAL_V2.csv').open())]
 assert validate(roots[0],ids)['passed']
 victim=Path(records[0]['path'])/'payload.bin';old=victim.read_bytes();victim.write_bytes(old+b'bad')
 try:validate(roots[0],ids)
 except RuntimeError as e:assert str(e)=='HASH'
 else:raise AssertionError('corruption accepted')
 victim.write_bytes(old)
 man=roots[0]/'manifest.json';m=json.loads(man.read_text());m['records'][0]=m['records'][1];man.write_text(json.dumps(m))
 try:validate(roots[0],ids)
 except RuntimeError as e:assert str(e)=='IDENTITY'
 else:raise AssertionError('duplicate accepted')
 try:run(roots[0],r/'P5_SAMPLE_PROPOSAL_V2.csv',MockBackend())
 except FileExistsError:pass
 else:raise AssertionError('restart accepted')
 bad=MockBackend(fail_control=True)
 try:run(roots[1],r/'P5_SAMPLE_PROPOSAL_V2.csv',bad)
 except RuntimeError as e:assert str(e).startswith('CONTROL_MISMATCH')
 else:raise AssertionError('control failure accepted')
 assert bad.forward_count==4 and len(json.loads((roots[1]/'budget.json').read_text())['attempts'])==4 and not (roots[1]/'control_gate.json').exists()
 assert not torch.cuda.is_initialized()
 (r/'ENGINE_MOCK_TEST.json').write_text(json.dumps({'passed':True,'scientific_schema_complete':False,'uses_real_LRCP_model':False,'real_model_forwards':0,'CUDA_initialized':False,'controls_and_core':22,'failed_control_blocks_core':True,'duplicate_restart_rejected':True,'independent_mock_integrity_corruption_and_duplicate_rejected':True,'synthetic_query_shape':[2,3]},indent=2))
finally:
 for x in roots:shutil.rmtree(x)
