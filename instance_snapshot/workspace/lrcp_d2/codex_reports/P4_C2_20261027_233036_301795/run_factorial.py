from c2_common import *
import traceback

def run_phase(mixed):
 marker='mixed_execution.json' if mixed else 'baseline_validation.json'
 assert not (R/marker).exists(),'REFUSE_RESTART'
 if mixed:assert json.loads((R/'baseline_validation.json').read_text())['passed']
 start=time.monotonic();pp=make_pp();checks=[]
 for sid in sorted(int(x['sample_idx']) for x in chosen()):
  inp=anchor_input(sid,pp);pil={g:load_pilot(sid,g) for g in [0,1]};reference={g:c1case(sid,g,0)[0] for g in [0,1]}
  assert np.array_equal(pil[0]['GT_canonical'],pil[1]['GT_canonical'])
  for g in [0,1]:
   for adm in [0,1]:
    rank=1-adm if mixed else adm
    data,stat=factorial_postprocess(pp,inp,pil[g],pil[adm],pil[rank]);data['IoU_GT_raw']=reference[g]['IoU_GT_raw'].copy()
    data['anchor_definition']=array(inp['ego']['anchor_box']);data['transform']=array(inp['ego']['transformation_matrix'])
    assert np.array_equal(data['raw_corners'],reference[g]['raw_corners'])
    donorA=c1case(sid,adm,adm)[0];donorR=c1case(sid,rank,rank)[0]
    assert np.array_equal(data['scores_all'],donorA['scores_all']) and np.array_equal(data['rank_scores_all'],donorR['scores_all'])
    q=data['ids_quality_pass'];assert np.array_equal(data['ids_nms_top1000'],q[data['rank_scores_all'][q].argsort()[::-1][:1000]])
    assert np.array_equal(data['final_scores'],data['scores_all'][data['ids_final_range']])
    if not mixed:
     old,ost=c1case(sid,g,adm)
     for key in old:
      if key in data:assert np.array_equal(data[key],old[key]),(sid,g,adm,key,'C1_BASELINE_ARRAY_MISMATCH')
     for th in ['50','70']:assert stat[th]==ost[th],(sid,g,adm,th,'NATIVE_MATCHING_MISMATCH')
    save_c2(sid,g,adm,rank,data,stat)
    checks.append({'sample_idx':sid,'G':g,'A':adm,'R':rank,'passed':True,'C1_all_shared_arrays_bitwise_equal':not mixed,'score_provenance_exact':True,'fixed_geometry_exact':True,'native_NMS_shadow_exact':True,'top1000_truncated':max(0,len(q)-1000),'TP70':int(data['GT_TP70'].sum())})
  guard();log('MIXED_SAMPLE_PASS' if mixed else 'BASELINE_SAMPLE_PASS',sample_idx=sid,elapsed=time.monotonic()-start)
 assert len(checks)==72
 save(R/marker,{'passed':True,'phase':'mixed' if mixed else 'homogeneous_baseline','case_count':72,'checks':checks,'strict_classification_and_matching_exact':True,'coordinate_tolerance':0,'score_tolerance':0,'C1_tolerance_note':'bitwise stricter than C1 1e-5m/1e-6','elapsed_seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0})
 log('MIXED_GATE_COMPLETE' if mixed else 'BASELINE_GATE_COMPLETE',elapsed=time.monotonic()-start)
if __name__=='__main__':
 try:run_phase('--mixed' in sys.argv)
 except BaseException:
  save(R/'FAILURE.json',{'phase':'mixed' if '--mixed' in sys.argv else 'baseline','traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0});raise
