"""Validate all12800 saved 3D corners incl. native direction+projection on CPU.
Validation only: never replaces the captured geometry used in the experiments.
"""
from cpu_common import *
def main():
 sources,meta,proof=load_sources(export=False);frozen=load(REPORTS/'RD3_MESSAGE_UTILITY_PILOT/FROZEN_MANIFEST.json');transforms={};rr=[]
 for (sid,c),a in sources.items():
  if sid not in transforms:
   rec=next(x for x in frozen['samples'] if x['sample_idx']==sid);payload,_=load_storage_archive(rec['input']['path']);transforms[sid]=np.asarray(payload['batch']['ego']['transformation_matrix'])
  decoded=a['decoded_before_dir'].reshape(-1,7).copy();direction=a['dir_preds'].transpose(0,2,3,1).reshape(-1,CONFIG['dir_args']['num_bins']).argmax(-1).astype(np.float32);period=2*np.pi/CONFIG['dir_args']['num_bins'];offset=CONFIG['dir_args']['dir_offset'];yaw=decoded[:,6];yaw=yaw-offset-np.floor((yaw-offset)/period)*period+offset+period*direction;yaw=yaw-np.floor(yaw/(2*np.pi)+.5)*(2*np.pi);decoded[:,6]=yaw
  box=decoded[:,[0,1,2,5,4,3,6]];template=np.asarray([[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,1],[1,1,1],[-1,1,1],[-1,-1,1]],np.float32)/2;local=box[:,None,3:6]*template;co=np.cos(yaw);si=np.sin(yaw);z=np.zeros_like(co);one=np.ones_like(co);rot=np.stack([co,si,z,-si,co,z,z,z,one],axis=1).reshape(-1,3,3);corners=np.matmul(local,rot)+box[:,None,:3];hom=np.concatenate([corners.transpose(0,2,1),np.ones((len(box),1,8),np.float32)],axis=1);projected=np.matmul(transforms[sid],hom)[:,:3].transpose(0,2,1);error=same(projected,a['raw_corners'],'ALL_RAW_NATIVE_3D_DIR_PROJECTION',1e-5)
  sig=(1/(1+np.exp(-a['cls_preds'].transpose(0,2,3,1).reshape(-1)))).astype(np.float32);serror=same(sig,a['scores_all'],'SOURCE_ALL_RAW_NATIVE_SIGMOID',1e-6)
  rr.append({'sample_idx':sid,'condition':c,'full12800_direction_corrected_3D_corners_max_abs_error':error,'existing_corner_tolerance':1e-5,'original_score_sigmoid_max_abs_error':serror,'existing_score_tolerance':1e-6,'captured_geometry_used_unchanged':True,'source_path':LOC[sid,c]['absolute_path'],'source_sha256':LOC[sid,c]['sha256']})
 write('R5D_FULL_GEOMETRY_PACKAGE_VALIDATION.json',{'state':'FULL_GEOMETRY_PACKAGE_PASS','source_records':rr,'all12800_corners_direction_projection_verified':True,'geometry_replaced_by_CPU_approximation':False,'new_model_forward':0,'CUDA_initializations':0});print('FULL_GEOMETRY_PACKAGE_PASS',len(rr),flush=True)
if __name__=='__main__':main()
