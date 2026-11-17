import os,json,tempfile,subprocess,sys
from pathlib import Path
from cpu_components import reserve,commit,verify_shard
root=Path(tempfile.mkdtemp(prefix='b0mock_'))
try:
 assert reserve(root,590,'off')==1
 try:reserve(root,590,'off')
 except RuntimeError as e:assert str(e)=='DUPLICATE_FORWARD'
 else:raise AssertionError('duplicate accepted')
 p=commit(root,590,b'synthetic data',{'synthetic':True});assert verify_shard(p)
 try:commit(root,590,b'other',{})
 except RuntimeError as e:assert str(e)=='DUPLICATE_SAMPLE'
 else:raise AssertionError('overwrite accepted')
 assert verify_shard(p)
 for i in range(21):reserve(root,i,'core')
 try:reserve(root,99,'core')
 except RuntimeError as e:assert str(e)=='FORWARD_LIMIT'
 else:raise AssertionError('23rd accepted')
 assert len(json.loads((root/'budget.json').read_text())['attempts'])==22
 (p/'payload.bin').write_bytes(b'corrupted')
 try:verify_shard(p)
 except RuntimeError as e:assert str(e)=='HASH_MISMATCH'
 else:raise AssertionError('corruption accepted')
 out={'passed':True,'pure_mock':True,'failed_attempt_reserved':True,'persistent_22_limit':True,'duplicate_forward_rejected':True,'atomic_sample_commit':True,'duplicate_sample_rejected':True,'corrupted_payload_rejected':True,'real_model_forwards':0,'CUDA_initialized':False}
 Path(__file__).with_name('COMPONENT_TESTS.json').write_text(json.dumps(out,indent=2))
finally:
 import shutil;shutil.rmtree(root)
