import os,json,csv,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
from pathlib import Path
D=Path('/workspace/lrcp_d2/codex_reports/P5_A_CPU_FIX_20261028_storage');O=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((D/'RESEARCH_ACCEPTED.json').read_text());assert a['passed'] and sha(D/'CPU_VALIDATION_COMPLETE.json')==a['CPU_validation_sha256'] and sha(D/'PPU_CAPTURE_COMPLETE.json')==a['PPU_capture_sha256']
ready=json.loads((D/'P5_READY.json').read_text());manifest=json.loads((D/'CAPTURE_MANIFEST.json').read_text())['records']; rows=list(csv.DictReader((D/'P5_SAMPLE_PROPOSAL_V2.csv').open()));ids={int(x['sample_idx']) for x in rows};assert len(ids)==18 and len(manifest)==36
r={(x['sample_idx'],int(x['alpha'])):x for x in manifest};out=[]
for s in sorted(ids):
 meta=[];zz=[]
 for alpha in (0,1):
  item=r[s,alpha];p=Path(item['path']);assert sha(p)==item['sha256'] and sha(p.with_suffix('.json'))==item['meta_sha256'];meta.append(json.loads(p.with_suffix('.json').read_text()));zz.append(np.load(p,allow_pickle=False))
 a,b=zz;assert meta[0]['query_ids']==meta[1]['query_ids'] and meta[0]['attention']['active_slots']==meta[1]['attention']['active_slots']
 q=meta[0]['query_ids'];active=meta[1]['attention']['active_slots'];valid=meta[1]['valid_slot_call_order'];ref0=a['cross_0_reference'];ref1=b['cross_0_reference'];assert ref1.shape==(1,len(q),3*(1+len(active)),2)
 base=ref1[:,:,:1,:];assert np.array_equal(ref0[:,:,:3,:],ref1[:,:,:3,:]);counts=0
 for j,slot in enumerate(active):
  sl=slice(3*(j+1),3*(j+2));
  if slot in valid:
   off=b[f'metaflow_effective_{valid.index(slot)}'].reshape(6400,2)[q][None,:,None,:]/80
   assert np.allclose(ref1[:,:,sl,:],base+off,atol=1e-6,rtol=0)
   assert np.array_equal(ref0[:,:,sl,:],np.broadcast_to(base,ref0[:,:,sl,:].shape));counts+=1
 for i in range(4):assert np.array_equal(b[f'cross_{i}_reference'],ref1)
 out.append({'sample_idx':s,'selected_queries':len(q),'active_slots':active,'valid_slots':valid,'validated_valid_active_slots':counts,'alpha0_base_exact_selected':True,'alpha1_shift_formula_selected':True,'all_cross_reference_equal_selected':True})
 for z in zz:z.close()
for x in ready['samples']:
 assert x['sample_idx'] in ids and sha(x['path'])==x['sha256']
locked=ready['locked_files'];checkpoint=[(k,v) for k,v in locked.items() if 'net_epoch_bestval_at19.pth' in k];assert checkpoint
for p,h in checkpoint:assert sha(p)==h
assert not torch.cuda.is_initialized()
(O/'SAVED_DATA_AUDIT.json').write_text(json.dumps({'passed':True,'cases':36,'samples':18,'model_forwards':0,'CUDA_initialized':False,'checkpoint':checkpoint,'records':out},indent=2))
print('passed 18 samples; checkpoint',checkpoint)
