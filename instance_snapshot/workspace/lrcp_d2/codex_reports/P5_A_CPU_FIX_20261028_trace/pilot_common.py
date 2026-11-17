"""Shared P4 safety, identity, exact CPU IoU, and resource utilities."""
from pathlib import Path
import os,sys,json,hashlib,time,shutil,csv,subprocess,resource
import numpy as np
R=Path(__file__).resolve().parent
PRE=R.parent/'P4_PRECHECK_20261022_175446_928190'
SRC=Path('/workspace/LRCP')
BASE=Path('/workspace/lrcp_d2')
MODEL=SRC/'opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18'
STAGES=('raw_decoded','score_pass','quality_pass','nms_top1000','nms_keep','final_range')
sys.path.insert(0,str(SRC))
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def jsonable(x):
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 return x

def write_json(p,x,exclusive=True):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 if exclusive and p.exists():raise FileExistsError(p)
 with tmp.open('x') as f:json.dump(jsonable(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 os.replace(tmp,p)

def event(name,**kwargs):
 x={'event':name,'epoch':time.time(),**kwargs}
 line=json.dumps(jsonable(x),ensure_ascii=False,allow_nan=False)
 print(line,flush=True)
 with (R/'events.jsonl').open('a') as f:f.write(line+'\n')

def array(x):
 if hasattr(x,'detach'):return x.detach().cpu().numpy().copy()
 return np.array(x,copy=True)

def digest(x):
 h=hashlib.sha256()
 def visit(v):
  if hasattr(v,'detach'):v=v.detach().cpu().numpy()
  if isinstance(v,np.ndarray):
   h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(np.ascontiguousarray(v).tobytes())
  elif isinstance(v,dict):
   h.update(b'dict')
   for k,value in v.items():h.update(repr(k).encode());visit(value)
  elif isinstance(v,(list,tuple)):
   h.update(type(v).__name__.encode())
   for value in v:visit(value)
  else:h.update(repr(v).encode())
 visit(x);return h.hexdigest()

def tensor_inventory(x,prefix=''):
 out=[]
 if hasattr(x,'detach') or isinstance(x,np.ndarray):
  a=array(x);out.append({'path':prefix,'shape':list(a.shape),'dtype':str(a.dtype),'nbytes':a.nbytes,'sha256':digest(a)})
 elif isinstance(x,dict):
  for k,v in x.items():out.extend(tensor_inventory(v,prefix+'/'+str(k)))
 elif isinstance(x,(list,tuple)):
  for k,v in enumerate(x):out.extend(tensor_inventory(v,prefix+'/'+str(k)))
 return out

def bytes_in(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())
def budget(start=None,seconds=None):
 if start is not None and time.monotonic()-start>seconds:raise RuntimeError('TIME_LIMIT_EXCEEDED')
 if shutil.disk_usage(R).free<5*2**30:raise RuntimeError('DISK_RESERVE_EXCEEDED')
 if bytes_in(R/'batches')>35*2**30:raise RuntimeError('INPUT_CACHE_LIMIT_EXCEEDED')
 if bytes_in(R/'diagnostics')>2**30:raise RuntimeError('DIAGNOSTIC_LIMIT_EXCEEDED')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('CPU_RSS_LIMIT_EXCEEDED')

def gpu_status():
 import torch
 if not torch.cuda.is_available():raise RuntimeError('FRAMEWORK_CUDA_UNAVAILABLE')
 props=torch.cuda.get_device_properties(0)
 device_uuid=str(getattr(props,'uuid',''))
 if not device_uuid:raise RuntimeError('FRAMEWORK_DEVICE_UUID_UNAVAILABLE; cannot safely map cuda:0 to process-query device')
 framework={'torch':torch.__version__,'device':str(torch.device('cuda:0')),'name':torch.cuda.get_device_name(0),'uuid':device_uuid,'free_total_bytes':list(torch.cuda.mem_get_info(0))}
 vendor=shutil.which('nvidia-smi')
 if vendor is None:raise RuntimeError('NO_SUPPORTED_READ_ONLY_DEVICE_PROCESS_QUERY; cannot prove cuda:0 conflict-free')
 cmd=[vendor,'--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 if p.returncode:raise RuntimeError('DEVICE_QUERY_UNSUPPORTED: '+p.stderr)
 normalize_uuid=lambda x:x.strip().lower().replace('gpu-','').replace('-','')
 gpu_lines=p.stdout.strip().splitlines();gpu0=next((x for x in gpu_lines if len(x.split(','))>1 and normalize_uuid(x.split(',')[1])==normalize_uuid(device_uuid)),None)
 if gpu0 is None:raise RuntimeError('CUDA0_DEVICE_NOT_IDENTIFIED')
 uuid=gpu0.split(',')[1].strip()
 q=subprocess.run([vendor,'--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
 if q.returncode:raise RuntimeError('DEVICE_PROCESS_QUERY_UNSUPPORTED: '+q.stderr)
 conflicts=[]
 for line in q.stdout.splitlines():
  vals=[x.strip() for x in line.split(',')]
  if vals and vals[0]==uuid and len(vals)>1 and vals[1]!=str(os.getpid()):conflicts.append(line)
 if conflicts:raise RuntimeError('CUDA0_COMPUTE_CONFLICT: '+repr(conflicts))
 return {'framework':framework,'device_query':p.stdout,'compute_processes':q.stdout,'pid':os.getpid(),'device_query_backend':'detected nvidia-smi compatibility interface'}

def selections():return sorted(list(csv.DictReader((R/'SAMPLE_SELECTION.csv').open())),key=lambda x:int(x['selection_order']))
def historical(sid,alpha):
 p=BASE/f'P3A_20261014_120929_boxes/sample_{sid}_alpha_{alpha:.1f}.npz'
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def gt_mapping(old,new):
 if old.shape!=new.shape:raise RuntimeError('GT_COUNT_MISMATCH')
 error=np.max(np.abs(old[:,None]-new[None]),axis=(2,3))
 hits=error<=1e-5
 if not np.all(hits.sum(0)==1) or not np.all(hits.sum(1)==1):raise RuntimeError('GT_BIJECTION_FAILED')
 mapping=np.argmax(hits,axis=1)
 return mapping,{'max_abs_error':float(error[np.arange(len(old)),mapping].max(initial=0.)),'exact_after_mapping':bool(np.array_equal(old,new[mapping])),'old_gt_idx_to_new_gt_idx':mapping.tolist()}

def iou_matrix(gt,boxes):
 from opencood.utils import common_utils
 from shapely.geometry import Polygon
 if not np.isfinite(gt).all() or not np.isfinite(boxes).all():raise RuntimeError('NONFINITE_GEOMETRY')
 low=boxes[:,:4,:2].min(1);high=boxes[:,:4,:2].max(1)
 polygons=[Polygon(b[:4,:2]) for b in boxes]
 if any(not p.is_valid or p.area<=0 for p in polygons):raise RuntimeError('INVALID_RAW_POLYGON')
 out=np.zeros((len(gt),len(boxes)),np.float32)
 for gi,g in enumerate(gt):
  lo=g[:4,:2].min(0);hi=g[:4,:2].max(0)
  ix=np.flatnonzero(np.all(high>lo,axis=1)&np.all(low<hi,axis=1))
  poly=Polygon(g[:4,:2])
  if not poly.is_valid or poly.area<=0:raise RuntimeError('INVALID_GT_POLYGON')
  out[gi,ix]=common_utils.compute_iou(poly,[polygons[k] for k in ix])
 return out

def greedy(iou,scores):
 remaining=list(range(iou.shape[0]));matched=np.full(len(scores),-1,np.int32)
 for pi in np.argsort(-scores):
  if remaining:
   vals=iou[remaining,pi]
   if vals.max()>=.7:
    pos=int(vals.argmax());matched[pi]=remaining.pop(pos)
 flags=np.zeros(iou.shape[0],np.int8)
 flags[matched[matched>=0]]=1
 return flags,matched

def verify_historical(sid,alpha,pred,scores,gt):
 old=historical(sid,alpha);mapping,meta=gt_mapping(old['gt_boxes'],gt)
 cp=gt[mapping]
 if pred.shape!=old['pred_boxes'].shape or scores.shape!=old['pred_scores'].shape:raise RuntimeError(f'HISTORICAL_PRED_COUNT_MISMATCH sid={sid} alpha={alpha} old={old["pred_boxes"].shape} new={pred.shape}')
 pe=float(np.max(np.abs(pred-old['pred_boxes']),initial=0.));se=float(np.max(np.abs(scores-old['pred_scores']),initial=0.))
 meta.update(pred_max_abs_error=pe,score_max_abs_error=se,pred_exact=bool(np.array_equal(pred,old['pred_boxes'])),score_exact=bool(np.array_equal(scores,old['pred_scores'])))
 if pe>1e-5 or se>1e-6:raise RuntimeError(f'HISTORICAL_PRED_OUTPUT_MISMATCH sid={sid} alpha={alpha} pred_error={pe} score_error={se}')
 mat=iou_matrix(cp,pred);flags,matched=greedy(mat,scores)
 records=[x for x in csv.DictReader((R/'SELECTED_SAMPLES_ALL_GT.csv').open()) if int(x['sample_idx'])==sid]
 records.sort(key=lambda x:int(x['gt_idx']))
 oldflags=np.array([int(x['tp70_a'+str(int(alpha))]) for x in records])
 oldbest=np.array([float(x['best_iou_a'+str(int(alpha))]) for x in records])
 best=mat.max(1) if len(scores) else np.zeros(len(cp))
 if not np.array_equal(flags,oldflags) or not np.array_equal(best>=.7,oldbest>=.7):raise RuntimeError('HISTORICAL_GT_THRESHOLD_CLASS_MISMATCH')
 meta.update(all_GT_TP70_equal=True,all_GT_best_IoU_threshold_equal=True,GT_count=len(cp))
 return cp,mapping,meta

def write_npz(p,**arrays):
 total=sum(np.asarray(a).nbytes for a in arrays.values())
 if total>64*2**20:raise RuntimeError('CASE_ARRAY_BUDGET_EXCEEDED')
 if bytes_in(R/'diagnostics')+total>2**30:raise RuntimeError('DIAGNOSTIC_LIMIT_EXCEEDED')
 p=Path(p)
 if p.exists():raise FileExistsError(p)
 tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('xb') as f:np.savez_compressed(f,**arrays)
 os.replace(tmp,p)
