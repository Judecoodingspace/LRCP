"""Read-only independent R1 attention re-computation, original tolerances."""
from rd3_common import *
from INDEPENDENT_VALIDATOR import same
def verify_attention(a,meta):
 keep=CONDITIONS[meta['condition']];require(meta['keep']==keep,'MASK_IDENTITY')
 q=QUERIES;ref=a['reference_full'];require(ref.shape==(1,6400,9,2),'FULL_REFERENCE_SHAPE')
 for slot in (1,2):
  off=a[f'metaflow_slot_{slot}'];require(off.shape==(1,80,80,2),'OFFSET_SHAPE')
  same(ref[:,:,slot*3:(slot+1)*3,:],np.broadcast_to(ref[:,:,:1,:]+off.reshape(1,6400,1,2)/np.array([80,80],np.float32),(1,6400,3,2)),'METAFLOW_SLOT_FORMULA',1e-6)
  same(a['pred_offset'][:,:,slot-1,:],off.reshape(1,6400,2),'OFFSET_SLOT_IDENTITY')
 for i in range(4):
  w=a[f'cross_{i}_full_weights'];require(w.ndim==5 and w.shape[:2]==(1,6400) and w.shape[3]==9,'FULL_WEIGHTS_SHAPE')
  require(np.isfinite(w).all() and (w>=0).all(),'NONFINITE_OR_NEGATIVE_WEIGHT')
  same(w.sum((-1,-2)),np.ones(w.shape[:3],np.float32),'GLOBAL_SUM',1e-5)
  for slot,enabled in enumerate(keep):
   if not enabled:require(np.count_nonzero(w[:,:,:,slot*3:(slot+1)*3,:])==0,'BLOCKED_WEIGHT_NONZERO')
  heads,points=w.shape[2],w.shape[-1]
  require(heads==8 and points==8,'NATIVE_HEAD_POINT_COVERAGE')
  raw=a[f'cross_{i}_raw_logits_selected'];require(np.isfinite(raw).all(),'RAW_LOGITS_NONFINITE')
  v=raw.reshape(1,len(q),heads,5,3,points)[:,:,:,:3,:,:].copy()
  for slot,enabled in enumerate(keep):
   if not enabled:v[:,:,:,slot,:,:]=-np.inf
  flat=v.reshape(1,len(q),heads,-1);ex=np.exp(flat-flat.max(-1,keepdims=True));expect=(ex/ex.sum(-1,keepdims=True)).reshape(1,len(q),heads,9,points)
  same(w[:,q],expect,'MASKED_SOFTMAX_RECOMPUTE',1e-6);same(a[f'cross_{i}_weights'],w[:,q],'SELECTED_WEIGHTS')
  sp=a[f'cross_{i}_spatial_shapes'];same(sp,np.tile(np.array([[80,80],[40,40],[20,20]]),(3,1)),'SPATIAL_ORDER')
  r=a[f'cross_{i}_reference'];same(r,ref[:,q],'REFERENCE_UNCHANGED')
  off=a[f'cross_{i}_offset'];proj=a[f'cross_{i}_sampling_offsets_output'];same(off,proj.reshape(1,len(q),heads,15,points,2)[:,:,:,:9],'OFFSET_PROJECTION')
  same(a[f'cross_{i}_location'],r[:,:,None,:,None,:]+off/sp[:,::-1].astype(np.float32)[None,None,None,:,None,:],'LOCATION_RECOMPUTE',1e-6)
 return True
