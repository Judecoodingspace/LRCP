"""R5B report/package builder: consumes independently accepted result tables.
No model imports, new inference or R5C attribution. Writes only this directory.
"""
import os,sys,csv,json,hashlib,zipfile,shutil,itertools,collections,datetime,importlib.abc
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent;A=R.parent/'RD3_R5A_EVIDENCE_AUDIT';OLD=R.parent/'RD3_REAL_FOUR_GROUP_R4/analysis'
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
class OfflineOnly(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','opencood','cupy','pycuda','pynvml'):raise ImportError('REPORT_BUILDER_NO_MODEL_DEVICE_IMPORT')
sys.meta_path.insert(0,OfflineOnly())
def allowed(p):
 if isinstance(p,int):return True
 try:return Path(os.fsdecode(p)).resolve().is_relative_to(R)
 except (TypeError,ValueError):return False
def guard(event,args):
 if event=='subprocess.Popen':raise RuntimeError('NO_SUBPROCESS_IN_REPORT_BUILDER')
 if event=='open':
  p,mode,flags=args
  if ((isinstance(mode,str) and any(x in mode for x in 'wa+x')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) and not allowed(p):raise PermissionError('WRITE_ONLY_NEW_R5B_DIRECTORY')
 if event in ('os.mkdir','os.remove','os.rmdir','os.utime','os.chmod') and not allowed(args[0]):raise PermissionError('MUTATE_ONLY_NEW_R5B_DIRECTORY')
sys.addaudithook(guard)
CATS=['CLEAR_BENEFICIAL','CLEAR_HARMFUL','TRADEOFF','COUNT_NEUTRAL']
SHORT={'U_B_given_C':'B|C（G-B→F）','U_C_given_B':'C|B（G-C→F）','U_B_given_A':'B|A（E→G-C）','U_C_given_A':'C|A（E→G-B）'}
def j(p):return json.loads(Path(p).read_text())
def csvrows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(n,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with (R/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sign(x):return f'{int(x):+d}'
def boolcol(x,k):return x[k]=='True'

def main():
 v=j(R/'R5B_INDEPENDENT_VALIDATION.json');assert v['state']=='R5B_PASS' and v['passed'] and v['failed_check_count']==0
 assert v['producer_code_sha256']==sha(R/'analyze_r5b.py') and v['source_code_sha256']==sha(R/'validate_r5b_cpu.py')
 assert v['passed_check_count']==len(v['checks']) and all(x['passed'] for x in v['checks'])
 p=j(R/'R5B_PRODUCER_RESULT.json');utility=csvrows(R/'R5B_SAMPLE_AGENT_UTILITY.csv');trans=csvrows(R/'R5B_GT_TRANSITION_AUDIT.csv');summary=csvrows(R/'R5B_GT_TRANSITION_SUMMARY.csv');inter=csvrows(R/'R5B_SAMPLE_SCENE_INTERACTION.csv');patterns=csvrows(R/'R5B_GT_FOUR_CONDITION_PATTERNS.csv');dist=csvrows(R/'R5B_PATTERN_DISTRIBUTION.csv');concentration=csvrows(R/'R5B_INTERACTION_CONCENTRATION.csv');sensitivity=csvrows(R/'R5B_THRESHOLD_SENSITIVITY.csv');patternchanges=csvrows(R/'R5B_GT_PATTERN_THRESHOLD_COMPARISON.csv')
 harm=[x for x in utility if x['class70']=='CLEAR_HARMFUL'];trade=[x for x in utility if x['class70']=='TRADEOFF'];s70=[x for x in inter if x['iou_threshold']=='0.7' and x['aggregation_level']=='sample'];sc70=[x for x in inter if x['iou_threshold']=='0.7' and x['aggregation_level']=='scene'];loss70=[x for x in trans if x['iou_threshold']=='0.7' and x['transition_state']=='TP_TO_FN'];loss50=[x for x in trans if x['iou_threshold']=='0.5' and x['transition_state']=='TP_TO_FN'];conditional70=[x for x in patterns if x['iou_threshold']=='0.7' and boolcol(x,'CONDITIONAL_DETECTION_CHANGE')]
 # Exact raw shard locator for future offline work; no candidate-level mechanisms examined.
 locators=[]
 for x in csvrows(R/'R5B_SOURCE_INVENTORY.csv'):
  if x['role']=='necessary_existing_real_result_shard':locators.append({'sample_idx':x['sample_idx'],'condition':x['condition'],'source_record_key':json.dumps({'sample_idx':int(x['sample_idx']),'condition':x['condition']}),'absolute_path':x['path'],'sha256':x['sha256'],'used_members':'GT_canonical, final_corners, final_scores, ids_final_range, gt_tp50/70, prediction_gt_idx50/70, prediction_fp50/70','not_used_for_mechanism':'raw candidates/Score-filter/NMS/attention/MetaFlow mechanism arrays not analyzed'})
 assert len(locators)==32;csvsave('R5B_RAW_SHARD_LOCATOR.csv',locators)
 candidates=[]
 for x in harm+trade:
  candidates.append({'sample_idx':x['sample_idx'],'scene_id':x['scene_id'],'contrast':x['contrast'],'physical_GT_id':'','iou_threshold':.7,'result_level_reason':x['class70'],'delta_TP70':x['delta_TP70'],'delta_FP70':x['delta_FP70'],'source_before':x['source_before'],'source_after':x['source_after'],'mechanism_not_yet_verified':True})
 for x in patterns:
  if x['REMOTE_ADDITION_LOSS']=='True' or x['CONDITIONAL_DETECTION_CHANGE']=='True':candidates.append({'sample_idx':x['sample_idx'],'scene_id':x['scene_id'],'contrast':'FOUR_CONDITION_GT_PATTERN','physical_GT_id':x['physical_object_id'],'iou_threshold':x['iou_threshold'],'result_level_reason':'/'.join(k for k in ['FULL_ONLY_TP','REMOTE_ADDITION_LOSS','CONDITIONAL_DETECTION_CHANGE'] if x[k]=='True'),'pattern_bits_E_GB_GC_F':x['pattern_bits'],'I_GT':x['I_GT'],'source_files':x['source_files'],'mechanism_not_yet_verified':True})
 csvsave('R5B_R5C_CANDIDATE_CASES.csv',candidates)
 primary_names=['R5B_SAMPLE_CONDITION_COUNTS.csv','R5B_SAMPLE_AGENT_UTILITY.csv','R5B_NEGATIVE_AND_TRADEOFF_CASES.csv','R5B_GT_TRANSITION_AUDIT.csv','R5B_GT_TRANSITION_SUMMARY.csv','R5B_SAMPLE_SCENE_INTERACTION.csv','R5B_GT_FOUR_CONDITION_PATTERNS.csv','R5B_PATTERN_DISTRIBUTION.csv']
 schema=[]
 for n in primary_names:
  data=csvrows(R/n);schema.append({'file':n,'rows':len(data),'columns':list(data[0]) if data else [],'sha256':sha(R/n)})
 save('R5B_OUTPUT_SCHEMA.json',{'files':schema,'pattern_order':'D_E,D_GB(A+C),D_GC(A+B),D_F','event_key':'sample×conditional contrast×physical object ID×IoU threshold','frame_GT_key':'sample_idx + physical_object_id','distinct_physical_GT_key':'scene_id + original physical_object_id','56_GT_is_frame_observation_denominator':True,'distinct_scene_object_ID_count':25,'classification_uses_delta_TP_delta_FP_only':True,'I_TP_and_I_FP_absolute_shares_have_separate_denominators':True})
 lines=['# RD3-R5B 逐样本Remote边际贡献、GT迁移与交互分解','','最终状态：`R5B_PASS`。B1–B6已完成；独立CPU验收'+str(v['passed_check_count'])+'项通过、0失败，任务范围内缺失/不可验证项为0。','','新forward：NO；CUDA初始化：NO；torch/opencood导入：NO；历史数据修改：NO；R5C机制分析：未执行。所有数据来自R5A已验收的32条真实科学记录。','','## 数据、身份及证据单位','','4份R4原始CSV在分析前逐一核对SHA256，与R5A一致；schema、GT映射、source inventory均按实际文件读取。32个sample-condition完整，8样本、3scene、每条件GT合计56。B1又从已有最终框/Score/GT在CPU上重做唯一匹配，计数与R5A原始几何验收精确一致；没有重新计算或定义AP。','','“56个GT”是8个冻结帧内的56条GT观测，其物理ID已跨四条件核验；跨帧按(scene_id, object_id)去重为25个ID。所有事件同时报告事件数、帧内GT数、去重物理ID、样本数和scene数。32条边际是同8帧上的重复条件对照，不是32个独立实验样本；不同模式内去重ID可能因同一对象跨帧模式变化而重叠，不能将各模式去重ID数相加当作全局不同对象数。','','F=ABC，G-B=AC，G-C=AB，E=A。所有边际符号都是加入Remote后−加入之前，物理slot不变。GT identity来自R5A物理ID+canonical框Hash，未用anchor代替物理身份。','','## B1 四条件计数','','| 条件 | TP50 | FP50 | FN50 | TP70 | FP70 | FN70 | 最终预测数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for x in csvrows(R/'R5B_FOUR_GROUP_COUNTS.csv'):lines.append('| '+x['condition']+' | '+' | '.join(x[k] for k in ['TP50','FP50','FN50','TP70','FP70','FN70','final_prediction_count'])+' |')
 lines+=['','逐帧TP+FN=GT、TP+FP=最终预测数在两个阈值下均通过。GT_count是待检测目标数量，和固定3个感知Agent数量不同。32条原始计数、来源路径与记录键见R5B_SAMPLE_CONDITION_COUNTS.csv。','','## RQ1：存在明确不利的Remote边际吗？','','存在。IoU70共9条CLEAR_HARMFUL，涉及4个不同样本（324、354、384、725）和2个scene（28、93）。7条表现为TP不变但FP增加；另外2条是sample725分别单独加入B/C时TP下降且FP增加。32条对照四分类互斥且完整，不用TP/FP权重合并效用。','','| 分类 | IoU70对照数 | IoU50对照数 |','|---|---:|---:|']
 for cat in CATS:lines.append(f"| {cat} | {p['B1_B2']['IoU70_category_counts'].get(cat,0)} | {p['B1_B2']['IoU50_category_counts'].get(cat,0)} |")
 lines+=['','| 条件边际 | 有利 | 不利 | 权衡 | 计数中性 |','|---|---:|---:|---:|---:|']
 for contrast in SHORT:
  got=[x for x in utility if x['contrast']==contrast];lines.append('| '+SHORT[contrast]+' | '+' | '.join(str(sum(x['class70']==cat for x in got)) for cat in CATS)+' |')
 lines+=['','全部IoU70明确不利案例：','','| sample | scene | 加入Remote的条件边际 | ΔTP70 | ΔFP70 | ΔFN70 |','|---:|---:|---|---:|---:|---:|']
 for x in harm:lines.append(f"| {x['sample_idx']} | {x['scene_id']} | {SHORT[x['contrast']]} | {sign(x['delta_TP70'])} | {sign(x['delta_FP70'])} | {sign(x['delta_FN70'])} |")
 lines+=['','所有IoU70权衡案例（TP与FP同向变化，不归入CLEAR_HARMFUL）：','','| sample | scene | 条件边际 | ΔTP70 | ΔFP70 |','|---:|---:|---|---:|---:|']
 for x in trade:lines.append(f"| {x['sample_idx']} | {x['scene_id']} | {SHORT[x['contrast']]} | {sign(x['delta_TP70'])} | {sign(x['delta_FP70'])} |")
 lines+=['','32条边际原始before/after计数、两阈值差分及分类见R5B_SAMPLE_AGENT_UTILITY.csv；逐阈值不利、权衡及隐藏退化的不同标签维度见R5B_NEGATIVE_AND_TRADEOFF_CASES.csv（23行，含IoU50/70，不能当作23个不同样本）。','','## RQ2：净TP非下降是否掩盖了具体GT退化？','','在本冻结集合没有观察到。按用户定义delta_TP>=0且TP_TO_FN>0，HIDDEN_GT_REGRESSION在IoU70和IoU50均为0；更严格的净TP增加却损失GT也为0。GT_IDENTITY_TURNOVER同样两阈值均为0。不是把损失删掉：实际损失发生在sample725，加入B或C时样本净TP各下降1，不属于“隐藏”条件。','','| 阈值 | FN→TP事件 | 涉及帧内GT/物理ID | TP→FN事件 | 涉及帧内GT/物理ID | 隐藏退化case | GT替换case |','|---|---:|---|---:|---|---:|---:|']
 for t in ['70','50']:
  g=p['GT_events'][t];lines.append(f"| {int(t)/100} | {g['FN_TO_TP']['event_count']} | {g['FN_TO_TP']['unique_frame_GT_count']}/{g['FN_TO_TP']['distinct_scene_object_count']} | {g['TP_TO_FN']['event_count']} | {g['TP_TO_FN']['unique_frame_GT_count']}/{g['TP_TO_FN']['distinct_scene_object_count']} | {g['hidden_case_count']} | {g['turnover_case_count']} |")
 lines+=['','IoU70的14个FN→TP事件涉及7条帧内GT、4个去重物理ID、4个样本、2个scene；2个TP→FN事件仅涉及1个GT、1个样本、1个scene。逐个损失事件为：','','| sample | scene | frame | 物理GT ID | 阈值 | 条件边际 | 状态 |','|---:|---:|---:|---|---:|---|---|']
 for x in loss70+loss50:lines.append(f"| {x['sample_idx']} | {x['scene_id']} | {x['scene_frame']} | {x['physical_object_id']} | {x['iou_threshold']} | {SHORT[x['contrast']]} | TP→FN |")
 lines+=['','同一GT在B|A、C|A两条对照中各出现一次损失事件，不能计为两个不同物理目标。完整迁移表448行=56帧内GT×4边际×2阈值；每条边际/阈值有56个配对，64个sample×contrast×threshold汇总均满足gain−loss=ΔTP、loss−gain=ΔFN及四状态总数=GT。原CSV存储方向通常与“加入Remote”相反，输出保留原始键及反向标记，状态按加入前→加入后重新核对。','','## RQ3：TP/FP交互+2/−3来自哪里？','','| sample | scene | I_TP70 | I_FP70 | I_FN70 | TP绝对份额 | FP绝对份额 |','|---:|---:|---:|---:|---:|---:|---:|']
 for x in s70:lines.append(f"| {x['sample_idx']} | {x['scene_id']} | {sign(x['I_BC_TP'])} | {sign(x['I_BC_FP'])} | {sign(x['I_BC_FN'])} | {float(x['abs_I_TP_share']):.0%} | {float(x['abs_I_FP_share']):.0%} |")
 lines+=['','| 聚合层级 | scene | I_TP70 | I_FP70 | I_FN70 |','|---|---|---:|---:|---:|']
 for x in sc70+[x for x in inter if x['iou_threshold']=='0.7' and x['aggregation_level']=='all_8_frames']:lines.append(f"| {x['aggregation_level']} | {x['scene_id']} | {sign(x['I_BC_TP'])} | {sign(x['I_BC_FP'])} | {sign(x['I_BC_FN'])} |")
 lines+=['','TP交互仅sample324和725非零，各+1，各占绝对TP交互总量2的50%；对应scene28和93。TP正分量+2、负分量0，没有正负抵消。FP交互有4个非零样本、3个非零scene；负分量−4（324:−2，725:−1，384:−1）与正分量+1（424）抵消后为−3，绝对FP交互总量为5。TP与FP份额分别归一化，未做加权合并。','','两种条件差分恒等式对TP/FP/FN、每样本与两阈值均成立：I=U_B|C−U_B|A=U_C|B−U_C|A。FP还满足I_FP=I_最终预测数−I_TP；总体I_最终预测数=−1，所以−3=−1−2。这个FP计数恒等式不识别或追踪跨条件的同一“假阳性物体”，不能据此声称消除了3个固定物理FP。','','## RQ4：哪些GT状态体现条件依赖？','','状态顺序严格为(D_E,D_GB,D_GC,D_F)，其中GB=G-B=AC，GC=G-C=AB。16模式（包括零频模式）完整保存于R5B_PATTERN_DISTRIBUTION.csv；下列是IoU70有观测的模式：','','| 模式E,GB,GC,F | 帧内GT数 | 不同scene-object ID数 | 每GT I | TP交互贡献 |','|---|---:|---:|---:|---:|']
 for x in dist:
  if x['iou_threshold']=='0.7' and int(x['GT_frame_observation_count']):lines.append(f"| {x['pattern_bits']} | {x['GT_frame_observation_count']} | {x['distinct_scene_object_count']} | {sign(x['I_GT_per_observation'])} | {sign(x['TP_interaction_contribution'])} |")
 lines+=['','IoU70逐GT交互非零的完整来源：','','| sample | scene/frame | 物理GT ID | E,GB,GC,F | I_GT | 结果级标记 |','|---:|---|---|---|---:|---|']
 for x in conditional70:lines.append(f"| {x['sample_idx']} | {x['scene_id']}/{x['scene_frame']} | {x['physical_object_id']} | {x['pattern_bits']} | {sign(x['I_GT'])} | {', '.join(k for k in ['FULL_ONLY_TP','REMOTE_ADDITION_LOSS','CONDITIONAL_DETECTION_CHANGE'] if x[k]=='True')} |")
 lines+=['','**正交互不等于联合融合有益。** sample324/29118是0001：只有Full检测成功；sample725/14014是1000：仅Ego成功、两种单Remote及Full均失败，却同样I_GT=+1，因为公式中E项为+1。总体+2因此不是“多检出了两个互补GT”。两者都是可复核的最终检测状态条件依赖，不能由此断言特征互补、Attention竞争、MetaFlow或NMS机制。','','GT级I在每样本、每scene与全8帧求和严格等于TP交互。CONDITIONAL_DETECTION_CHANGE等价于两种Remote响应差分不相等，即I_GT≠0；B/C两式是同一个交互恒等式，不能再当成两份独立证据。','','## RQ5：IoU50与IoU70观察是否一致？','','CLEAR_HARMFUL的存在及sample/scene范围一致；32条边际中28条四分类不变，4条变化，其中只有sample324的C|A跨越明确不利标签：','','| sample | 边际 | IoU50分类（ΔTP/ΔFP） | IoU70分类（ΔTP/ΔFP） |','|---:|---|---|---|']
 for x in utility:
  if x['classification_changed_at_IoU50_vs70']=='True':lines.append(f"| {x['sample_idx']} | {SHORT[x['contrast']]} | {x['class50']}（{sign(x['delta_TP50'])}/{sign(x['delta_FP50'])}） | {x['class70']}（{sign(x['delta_TP70'])}/{sign(x['delta_FP70'])}） |")
 lines+=['','HIDDEN_GT_REGRESSION和GT_IDENTITY_TURNOVER两阈值均为0。共有4条帧内GT四条件模式发生变化：','','| sample/物理ID | 模式50 | 模式70 | I_GT50→70 |','|---|---|---|---|']
 for x in patternchanges:
  if x['pattern_changed']=='True':lines.append(f"| {x['frame_GT_key']} | {x['pattern50']} | {x['pattern70']} | {sign(x['I_GT50'])}→{sign(x['I_GT70'])} |")
 lines+=['','IoU50损失对象是sample725的4071；IoU70则是同帧14014，尽管两阈值样本TP损失和CLEAR_HARMFUL标签相同。不能用计数标签稳定替代物理GT身份稳定。TP非零交互样本在IoU50只有725，在IoU70为324和725；全局I_TP/I_FP/I_FN由IoU50的+1/−2/−1变为IoU70的+2/−3/−2。没有扫描其他阈值，也没有用逐帧AP均值解释变化。','','## 独立验收与输出追溯','','分析代码analyze_r5b.py与独立验收validate_r5b_cpu.py分离；验收器不导入Producer，从R5A原始计数、真实NPZ唯一匹配状态与物理GT映射重新计算全部核心不变量。共'+str(v['passed_check_count'])+'项检查通过，失败0、缺失0、任务范围内不可验证0。8份核心CSV主键唯一，32计数/32边际/448迁移/64迁移汇总/24分层交互/112四条件GT/32模式均完整。R5C机制未做，不能以“不可验证项0”解读为机制已验证。','','各原始CSV路径、SHA256与R5A比对见R5B_INITIAL_SOURCE_HASH_CHECK.json及R5B_INDEPENDENT_VALIDATION.json。32个NPZ的精确路径、Hash、sample/condition记录键见R5B_RAW_SHARD_LOCATOR.csv；未把大体积原始NPZ打进上传包。原始四CSV及必要R5A schema/验证材料会按原始字节完整打包。3523份历史文件Hash保持不变。程序导入受CPU guard限制，无torch/opencood/设备动态库，未启动子进程推理；源代码修订草稿与语法兼容日志保留在source_revisions/，不修改已有科学记录。','','## 可支持的结论、限制及R5C候选','','本冻结模型与固定输入/slot/time300的8帧集合确有结果级明确不利Remote边际；不利并非只发生在TP下降时，FP增加而TP不变同样按指定偏序规则不利。本集合未观察到被净TP非下降掩盖的GT退化；这不是对其他数据的不存在证明。TP非加性交互集中在2帧/2scene；FP有跨4帧/3scene的正负抵消。正交互既可能来自Full-only成功，也可能来自Ego-only成功下的差分结构。','','值得后续独立授权R5C核验的案例：','','1. sample725，scene93/frame31：E→G-C与E→G-B，IoU70 GT14014与IoU50 GT4071的TP→FN；负计数两阈值都存在但物理目标切换。应追踪已有最终几何/Score/唯一匹配及后处理证据，当前不指定原因。','2. sample324，scene28/frame30，GT29118：IoU70仅Full为TP、IoU50为0101；同帧C|A由CLEAR_BENEFICIAL转CLEAR_HARMFUL，是条件和阈值依赖的结果例。','3. sample354/384的FP-only不利边际、sample424的正FP交互+1以及sample484的C边际由有利变权衡，可用于对比结果级FP变化和交互抵消。完整候选键见R5B_R5C_CANDIDATE_CASES.csv；这只是候选清单，未执行机制实验。','','限制：8帧聚类于3scene，56条GT观测及32条边际不是独立scene；不能外推为common910总体规律。固定time300下只改变融合集合，不能解释为AoI/时延变化的因果效应，也没有验证无线通信负载降低。未训练、未扩样、未开发效用预测器，未进行Score/NMS/Attention/MetaFlow机制归因。完成R5B后停止，不自动启动R5C。']
 (R/'RD3_R5B_SCIENTIFIC_ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
 # Immutable exact input copies for the upload bundle, no truncation.
 (R/'original_csv').mkdir(exist_ok=True);(R/'input_r5a').mkdir(exist_ok=True)
 raw_names=['AGENT_MARGINAL_UTILITY.csv','AGENT_INTERACTION.csv','PER_GT_TRANSITIONS.csv','NMS_STAGE_ANALYSIS.csv']
 for n in raw_names:shutil.copy2(OLD/n,R/'original_csv'/n);assert sha(OLD/n)==sha(R/'original_csv'/n)
 inputnames=['R5A_GT_IDENTITY_MAP.csv','R5A_CSV_SCHEMA_AUDIT.json','R5A_SOURCE_INVENTORY.csv','R5A_SAMPLE_CONDITION_COUNTS.csv','RD3_R5A_FINAL_REPORT.md','R5A_RESULT.json','R5A_DELIVERY_VERIFIED.json','R5A_SCIENTIFIC_RECORD_AUDIT.json']
 for n in inputnames:shutil.copy2(A/n,R/'input_r5a'/n);assert sha(A/n)==sha(R/'input_r5a'/n)
 save('R5B_EXECUTION_GUARD.json',{'state':'R5B_PASS','new_forward':'NO','CUDA_initialized':'NO','historical_data_modified':'NO','torch_imported':False,'opencood_imported':False,'R5C_executed':False,'AP_recomputed_or_redefined':False,'single_weighted_utility_used':False,'output_root':str(R),'source_and_result_hashes':{str(p.relative_to(R)):sha(p) for p in sorted(R.glob('*.py'))},'independent_validation_passed':True,'independent_check_count':v['passed_check_count'],'completed_time_Asia_Shanghai':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
 # Required output completeness and independent-input hashes checked before packaging.
 required=primary_names+['R5B_INDEPENDENT_VALIDATION.json','RD3_R5B_SCIENTIFIC_ASSESSMENT.md']
 for n in required:assert (R/n).is_file()
 for h in v['CSV_source_SHA256']:assert sha(h['path'])==h['expected_R5A_sha256']
 names=[str(p.relative_to(R)) for p in sorted(R.glob('*.csv'))]+[str(p.relative_to(R)) for p in sorted(R.glob('*.json')) if p.name not in ('R5B_HISTORY_READ_ONLY_SEAL.json','R5B_UPLOAD_BUNDLE_VALIDATION.json','R5B_FINAL_RESULT.json')]+['RD3_R5B_SCIENTIFIC_ASSESSMENT.md','analyze_r5b.py','validate_r5b_cpu.py','report_and_bundle.py']
 names += [str(p.relative_to(R)) for p in sorted((R/'source_reference').glob('*.py'))]+['original_csv/'+n for n in raw_names]+['input_r5a/'+n for n in inputnames]+['logs/producer_cpu_final.log','logs/independent_validation_cpu.log']
 names=list(dict.fromkeys(names));hashes={n:sha(R/n) for n in names};(R/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in hashes.items()));names.append('SHA256SUMS.txt')
 archive=R/'R5B_UPLOAD_BUNDLE.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n in names:z.write(R/n,n)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
  counts={}
  for n in raw_names:
   content=z.read('original_csv/'+n);assert content==(OLD/n).read_bytes();counts[n]=len(list(csv.DictReader(content.decode().splitlines())))
  assert counts=={'AGENT_MARGINAL_UTILITY.csv':44,'AGENT_INTERACTION.csv':11,'PER_GT_TRANSITIONS.csv':672,'NMS_STAGE_ANALYSIS.csv':480}
 archivehash=sha(archive);(R/'R5B_UPLOAD_BUNDLE.zip.sha256').write_text(archivehash+'  R5B_UPLOAD_BUNDLE.zip\n')
 result={'state':'R5B_PASS','bundle_absolute_path':str(archive),'bundle_sha256':archivehash,'bundle_size_bytes':archive.stat().st_size,'required_csv_rows':{n:len(csvrows(R/n)) for n in primary_names},'full_original_CSV_rows_packed':counts,'zip_members':len(names),'all_SHA256SUMS_verified':True,'raw_npz_not_packed_precise_locator_provided':True,'new_forward':'NO','CUDA_initialized':'NO','historical_data_modified':'NO','independent_passed_checks':v['passed_check_count'],'failed_checks':0,'not_verifiable':[],'IoU70_category_counts':p['B1_B2']['IoU70_category_counts'],'IoU50_category_counts':p['B1_B2']['IoU50_category_counts'],'hidden_GT_regression_cases50':p['GT_events']['50']['hidden_case_count'],'hidden_GT_regression_cases70':p['GT_events']['70']['hidden_case_count'],'GT_identity_turnover_cases50':p['GT_events']['50']['turnover_case_count'],'GT_identity_turnover_cases70':p['GT_events']['70']['turnover_case_count'],'R5C_started':False}
 save('R5B_UPLOAD_BUNDLE_VALIDATION.json',result);save('R5B_FINAL_RESULT.json',result);print('R5B_PASS_REPORT_AND_BUNDLE_COMPLETE',archivehash,archive.stat().st_size,flush=True)
if __name__=='__main__':main()
