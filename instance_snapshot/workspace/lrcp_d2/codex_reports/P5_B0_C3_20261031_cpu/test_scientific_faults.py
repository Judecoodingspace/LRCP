# C2's independently accepted 17 injections reused against NEW C3 same-entry output.
import sys,shutil
from pathlib import Path
sys.path.insert(0,'/workspace/lrcp_d2/codex_reports/P5_B0_C2_20261031_cpu')
import json
_c3=Path(__file__).parent
_fixture=_c3/'mock_runs'/'scientific_fault_fixture'
shutil.copytree(_c3/'mock_runs'/'e2e_release',_fixture)
_m=json.loads((_fixture/'manifest.json').read_text())
for _rec in _m['records']:_rec['path']=str(_fixture/'capture'/Path(_rec['path']).name)
(_fixture/'manifest.json').write_text(json.dumps(_m))
import os,json,io,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from INDEPENDENT_VALIDATOR import validate,sha,verify_historical_control
R=Path(__file__).parent;root=R/'mock_runs'/'scientific_fault_fixture';mp=root/'manifest.json';base_manifest=mp.read_bytes();record=json.loads(base_manifest)['records'][0];p=Path(record['path']);mf=p/'manifest.json';data=p/'payload.bin';meta=p/'meta.json';original={f:f.read_bytes() for f in [mp,mf,data,meta]}
results={}
def restore():
 for f,b in original.items():f.write_bytes(b)
def rehash():
 z=json.loads(mf.read_text());z['files']['payload.bin']=sha(data);z['files']['meta.json']=sha(meta);mf.write_text(json.dumps(z))
 z=json.loads(mp.read_text());z['records'][0]['manifest_sha256']=sha(mf);mp.write_text(json.dumps(z))
def change_array(key,func):
 with np.load(data,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 func(a[key]);buf=io.BytesIO();np.savez_compressed(buf,**a);data.write_bytes(buf.getvalue());rehash()
def rejection(name,fn,expected,mock=True):
 restore();fn()
 try:validate(root,mock=mock)
 except Exception as e:
  got=type(e).__name__+': '+str(e);assert expected in got,(name,got,expected);results[name]={'rejected':True,'reason':got}
 else:raise AssertionError('undetected '+name)
def meta_change(f):
 z=json.loads(meta.read_text());f(z);meta.write_text(json.dumps(z));rehash()
def add(x,index,delta=.05):x[index]+=delta
try:
 rejection('wrong_sample',lambda:meta_change(lambda m:m.__setitem__('sample_idx',999)), 'SAMPLE_IDENTITY')
 rejection('wrong_physical_cav',lambda:meta_change(lambda m:m['slots']['1']['physical_cav_ids'].__setitem__(0,'forged')),'PHYSICAL_SLOT_MANIFEST')
 rejection('invalid_slot_activated',lambda:meta_change(lambda m:m['valid_slots'].append('4')),'VALID_SLOT_SET')
 rejection('active_slot_order',lambda:meta_change(lambda m:m['active_slots'].reverse()),'ACTIVE_SLOT_ORDER')
 rejection('remote_shift',lambda:change_array('reference_original_full',lambda x:add(x,(0,0,3,0))),'SHIFT_FORMULA')
 rejection('ego_reference',lambda:change_array('cross_0_direct_reference',lambda x:add(x,(0,0,0,0))),'BLOCK0_REFERENCE')
 rejection('block1_reference',lambda:change_array('cross_1_direct_reference',lambda x:add(x,(0,0,0,0))),'LATER_REFERENCE')
 rejection('sampling_location',lambda:change_array('cross_0_location',lambda x:add(x,(0,0,0,0,0,0))),'LOCATION_FORMULA')
 rejection('attention_weights',lambda:change_array('cross_0_weights',lambda x:add(x,(0,0,0,0,0))),'WEIGHT_SUM')
 rejection('head_cls',lambda:change_array('cls_preds',lambda x:x.__setitem__((0,0,0,0),8)),'POSTPROCESS_score_mask')
 rejection('nms_keep',lambda:change_array('native_nms_keep',lambda x:x.__setitem__(0,999)),'POSTPROCESS_native_nms_keep')
 rejection('suppressor',lambda:change_array('suppressor',lambda x:x.__setitem__(0,0)),'POSTPROCESS_suppressor')
 rejection('gt_match',lambda:change_array('gt_tp70',lambda x:x.__setitem__(0,1-x[0])),'GT_TP70')
 restore()
 with np.load(data,allow_pickle=False) as z:sample={k:z[k].copy() for k in z.files}
 try:verify_historical_control(sample,590)
 except Exception as e:
  assert 'CONTROL_cls_preds' in str(e);results['historical_control']={'rejected':True,'reason':str(e),'scope':'synthetic output explicitly compared against historical baseline'}
 else:raise AssertionError('historical control mismatch accepted')
 rejection('missing_payload',lambda:data.unlink(),'FileNotFoundError')
 def duplicate():
  z=json.loads(mp.read_text());z['records'][0]=z['records'][1];mp.write_text(json.dumps(z))
 rejection('duplicate_condition',duplicate,'RECORD_IDENTITY')
 rejection('payload_old_hash',lambda:data.write_bytes(original[data]+b'x'),'FILE_HASH')
 require_no_cuda=not torch.cuda.is_initialized();assert require_no_cuda
 (R/'SCIENTIFIC_FAULT_INJECTION.json').write_text(json.dumps({'passed':len(results)==17 and all(v['rejected'] for v in results.values()),'rehashed_semantic_faults_rejected':True,'tests':results,'remaining_real_device_checks':['actual PPU control output not available'],'real_model_forward':0,'CUDA_initialized':False},indent=2))
finally:restore()
