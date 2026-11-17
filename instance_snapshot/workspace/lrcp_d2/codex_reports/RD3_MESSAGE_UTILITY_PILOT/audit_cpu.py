"""Read every historical frozen CPU batch and its raw 1000-entry index; no model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from rd3_common import *
import pickle,re,torch,gc,collections,traceback

def run():
 started=time.monotonic();source=Path('/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl')
 info=pickle.load(source.open('rb'));bytoken={x['token']:x for x in info}
 m=load(FULL/'INPUT_MANIFEST.json');common=list(map(int,load(BASE/'P1_common910_indices.json')['sample_ids']))
 require([x['sample_idx'] for x in m['samples']]==common and len(common)==910,'COMMON910_IDENTITY')
 require(sha_file(CKPT)==CH,'CHECKPOINT_SHA')
 history=full_records();rows=[]
 for no,item in enumerate(m['samples']):
  cpu_guard();sid=item['sample_idx'];raw=info[sid];require(item['sample_token']==raw['token'],'RAW_TOKEN')
  require(sha_file(item['path'])==item['sha256'],'BATCH_SHA '+str(sid))
  payload=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=False);b=payload['batch'];e=b['ego']
  require(digest(b)==item['input_hash'] and digest(payload['rng'])==item['RNG_hash'],'BATCH_SEMANTICS')
  ids=list(map(str,e['cav_id_list']));n=int(e['record_len'][0]);require(n==len(ids)==raw['agent_num'],'AGENT_COUNT')
  require(ids==list(map(str,range(1,n+1))) and ids[0]=='1','PHYSICAL_SLOT_ORDER')
  match=re.search(r'scene_(\d+)_(\d+)\.pcd',raw['lidar_path_1']);scene,frame=map(int,match.groups())
  histories=[];available=True
  for k in range(3,8):
   token=raw['prev_samples'][k];h=bytoken.get(token);good=h is not None and item['history_tokens'][str(k)]==token
   if good:
    hm=re.search(r'scene_(\d+)_(\d+)\.pcd',h['lidar_path_1']);good=int(hm[1])==scene
    good=good and all(Path(h[f'lidar_path_{c}']).is_file() and np.isfinite(h[f'lidar_pose_{c}']).all() for c in ids)
   available=available and good
   histories.append({'lag':k,'token':token,'timestamp_raw':h['timestamp'] if h else None,'scene_frame':int(hm[2]) if h else None,'available':good})
  slots={};history_inputs=True
  for slot,v in e['calibrate_data'].items():
   valid=bool(v['valid_mask'][0]);expected=int(slot)<n
   require(valid==expected,'SLOT_VALIDITY');require(v['physical_cav_ids']==item['slots'][str(slot)]['physical_cav_ids'],'PHYSICAL_ID')
   if valid:
    hist=v['lidar_history'][0];good=hist is not None and all(k in hist for k in ('voxel_features','voxel_coords','voxel_num_points'))
    if good:
     coords=array(hist['voxel_coords']);good=set(coords[:,0].astype(int).tolist())==set(range(5)) and np.isfinite(array(hist['voxel_features'])).all()
    good=bool(good and tuple(v['offset'].shape)==(1,80,80,2) and torch.isfinite(v['offset']).all().item() and float(v['time_delay'][0])==300)
    history_inputs=history_inputs and good
    slots[str(slot)]={'physical_id':v['physical_cav_ids'][0],'history_valid':good,'history_input_hash':digest(hist),'offset_GT_hash':digest(v['offset']),'time_delay':float(v['time_delay'][0])}
   else:require(v['lidar_history']==[None],'PADDED_HISTORY')
  pair=array(e['pairwise_t_matrix']);goodpair=pair.shape==(1,5,5,4,4) and np.isfinite(pair).all()
  goodpair=bool(goodpair and np.allclose(pair[0,:n,:n]@pair[0,:n,:n].swapaxes(0,1),np.eye(4),atol=1e-4))
  require(goodpair,'PAIRWISE_COMPLETENESS')
  require(torch.isfinite(e['lidar_pose']).all().item() and e['lidar_pose'].shape==(n,6),'POSE_COMPLETENESS')
  gt=int(e['object_bbx_mask'].sum().item());require(gt==len(item['GT_mapping']['old_gt_idx_to_new_gt_idx']),'GT_COUNT')
  eligible=n==3 and available and history_inputs and goodpair
  row={'sample_idx':sid,'scene_id':scene,'scene_frame':frame,'ego_physical_id':ids[0],'remote_physical_ids':ids[1:],'real_agent_count':n,'GT_count':gt,'ego_timestamp_raw':raw['timestamp'],'remote_timestamp_raw':histories[0]['timestamp_raw'],'time_delay':[0]+[300]*(n-1),'time_semantics':'native lag index 3 * 100; raw timestamp/frame counter kept separately; no unit reinterpretation','history':histories,'slots':slots,'history_complete':available,'metaflow_inputs_complete':history_inputs,'pairwise_complete':goodpair,'pairwise_hash':digest(e['pairwise_t_matrix']),'poses_hash':digest(e['lidar_pose']),'raw_pose_hashes':{c:digest(raw[f'lidar_pose_{c}']) for c in ids},'input':item,'historical_full':history[sid],'eligible':eligible}
  rows.append(row);del payload,b,e;gc.collect()
  if no%50==0:print('AUDIT',no+1,'/910','RSS',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,flush=True)
 counts=collections.Counter(x['real_agent_count'] for x in rows);eligible=[x for x in rows if x['eligible']];scenes=sorted({x['scene_id'] for x in eligible})
 write(R/'AUDIT_MANIFEST_910.json',{'samples':rows},exclusive=True)
 csvwrite(R/'ELIGIBLE_SAMPLES.csv',[{k:x[k] for k in ('sample_idx','scene_id','scene_frame','ego_physical_id','remote_physical_ids','GT_count','history_complete','metaflow_inputs_complete','pairwise_complete')} for x in eligible])
 # Round robin sorted scenes. Within each scene, choose midpoint of each of three
 # chronological strata and closest-to-stratum-median GT count, tie by index.
 # No RD3 predictions, P5 selection or failure GT enter this rule.
 groups={s:sorted([x for x in eligible if x['scene_id']==s],key=lambda x:x['scene_frame']) for s in scenes};selected=[]
 for round_ in range(8):
  for scene in scenes:
   g=groups[scene];parts=np.array_split(np.arange(len(g)),3);part=parts[round_%3]
   used={x['sample_idx'] for x in selected};candidates=[g[int(i)] for i in part if g[int(i)]['sample_idx'] not in used]
   if not candidates:continue
   midpoint=float(np.median([x['scene_frame'] for x in candidates]));target=[min(x['GT_count'] for x in candidates),max(x['GT_count'] for x in candidates),float(np.median([x['GT_count'] for x in candidates]))][round_%3]
   selected.append(min(candidates,key=lambda x:(abs(x['GT_count']-target),abs(x['scene_frame']-midpoint),x['sample_idx'])))
   if len(selected)==8:break
  if len(selected)==8:break
 sufficient=len(selected)>=4 and len({x['scene_id'] for x in selected})>=2
 freeze={'schema':'RD3_FROZEN_V1','samples':selected if sufficient else [],'eligible_count':len(eligible),'eligible_scenes':scenes,'selection_rule':'sorted-scene round robin; three chronological strata; GT min/max/median then stratum midpoint then sample_idx','target':8,'preferred_scenes':4,'limitation':'only '+str(len(scenes))+' eligible scenes exist' if len(scenes)<4 else None,'checkpoint_sha256':CH,'configuration_sha256':sha_file(MODEL/'config.yaml'),'source_manifest_sha256':sha_file(FULL/'INPUT_MANIFEST.json'),'raw_index_sha256':sha_file(source),'lag_index':3,'time_encoding':300,'conditions':CONDITIONS,'Q_components':['TP50','FP50','FN50','TP70','FP70','FN70'],'utility_orientation':'signed raw count differences; TP positive is better, FP/FN positive is worse; no scalar weights','sufficient':sufficient}
 write(R/'FROZEN_MANIFEST.json',freeze,exclusive=True);(R/'FROZEN_MANIFEST.sha256').write_text(sha_file(R/'FROZEN_MANIFEST.json')+'\n')
 result={'passed':True,'sample_count':910,'counts':{str(k):counts[k] for k in (1,2,3,4,5)},'ego_only':counts[1],'ego_plus_one':counts[2],'ego_plus_two':counts[3],'ego_plus_three_or_more':sum(v for k,v in counts.items() if k>=4),'eligible_count':len(eligible),'eligible_scenes':scenes,'frozen_samples':[x['sample_idx'] for x in selected],'frozen_scenes':sorted({x['scene_id'] for x in selected}),'frozen_sha256':sha_file(R/'FROZEN_MANIFEST.json'),'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'torch_version':torch.__version__,'numpy_version':np.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'real_forwards':0}
 write(R/'CPU_DATA_AUDIT.json',result,exclusive=True);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(R/'CPU_AUDIT_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'real_forwards':0,'CUDA_initialized':torch.cuda.is_initialized()});raise
