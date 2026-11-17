"""Render R5C accepted numerical observations; never runs an experiment."""
from stage_common import *
from datetime import datetime,timezone
import shutil,collections

def mdtable(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x).replace('|','∣').replace('\n',' ') for x in r)+' |' for r in rows])
def main():
 v=load(ROOT/'R5C_INDEPENDENT_VALIDATION.json');require(v['state']=='R5C_PASS','NO_REPORT_PASS_WITHOUT_INDEPENDENT_PASS')
 trace=csvrows(ROOT/'R5C_GT_STAGE_TRACE.csv');lineage=csvrows(ROOT/'R5C_CANDIDATE_LINEAGE.csv');fp=csvrows(ROOT/'R5C_FP_PROVENANCE.csv');summ=csvrows(ROOT/'R5C_FP_MARGINAL_ACCOUNTING.csv');inter=csvrows(ROOT/'R5C_INTERACTION_MECHANISM_MATRIX.csv');c0=load(ROOT/'R5C_C0_REPLAY_VALIDATION.json');scope=load(ROOT/'R5C_CANDIDATE_SELECTION_AND_PACKAGE_SCOPE.json');prod=load(ROOT/'R5C_PRODUCER_RESULT.json')
 inv=csvrows(A/'R5A_SOURCE_INVENTORY.csv');sourcecopy=[];folder=ROOT/'source_reference/native_text';folder.mkdir(exist_ok=True)
 native=[x for x in inv if x['kind']=='source_definition'];model=Path('/workspace/LRCP/opencood/models/point_pillar_deformable.py');native.append({'absolute_path':str(model),'sha256':sha(model),'kind':'native_model_output_definition'})
 native.append({'absolute_path':c0['source_config_path'],'sha256':c0['source_config_sha256'],'kind':'frozen_config'})
 for x in native:
  src=Path(x['absolute_path']);target=folder/(src.name+'.txt');require(sha(src)==x['sha256'],'NATIVE_REFERENCE_SHA');shutil.copyfile(src,target);sourcecopy.append({**x,'copied_to':str(target.relative_to(ROOT)),'copied_sha256':sha(target),'imported':False})
 write('R5C_NATIVE_PIPELINE_CONTRACT.json',{'actual_sources':sourcecopy,'candidate_count':12800,'head_order':'CHW to HWC then grid y,x,anchor rotation; 80x80x2; query_index=anchor_id//2','candidate_identity_scope':'immutable head grid row within frozen sample, never physical prediction identity','score':'sigmoid(cls_logits), strict score >0.2','regression':'actual frozen anchors hwl; diagonal sqrt(w^2+l^2); xy,z offsets; exp(h,w,l) and yaw delta; no head algorithm changes','direction':'2 bins; native limit_period and dir_offset; source-preserved final raw corners, no regenerated predictions','quality':'No iou_preds emitted, no learned IoU rescoring. large/z masks active before NMS, not disabled. Historical large-box y-span-as-z boolean quirk preserved.','topK':'After score/large/z, native score.argsort()[::-1][:1000] inside NMS, not evaluator argsort(-score)','NMS':'float32 Shapely BEV IoU, suppress strictly >0.15; retain first highest-score active candidate','final_range':'native gt_range; all eight corners inside x,y,z limits; after NMS','GT_match':'scores np.argsort(-score), each prediction greedily chooses maximum BEV IoU among remaining GT; >= evaluation threshold; each GT used once','head_numeric_checks':'original existing tolerance: sigmoid 1e-6, reg/direction 1e-5; final corners/scores/keep/suppressors exact; no tolerance relaxed','native_replay32':True,'quality_size_z_removed_total':sum(r['large_or_z_removed'] for r in c0['records']),'top1000_removed_total':sum(r['top1000_removed'] for r in c0['records']),'NMS_suppressed_total':sum(r['NMS_suppressed'] for r in c0['records']),'final_range_removed_total':sum(r['final_range_removed'] for r in c0['records']),'analysis_is_replay_of_recorded_outputs_not_new_inference':True})
 write('R5C_SOURCE_AND_EXECUTION_SEAL.json',{'state':'R5C_PASS','created_utc':datetime.now(timezone.utc).isoformat(),'frozen_manifest_sha256':sha(REPORTS/'RD3_MESSAGE_UTILITY_PILOT/FROZEN_MANIFEST.json'),'epoch19_checkpoint_sha256':sha('/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth'),'current_scripts':{n:sha(ROOT/n) for n in ['analyze_r5c.py','stage_common.py','validate_r5c_independent.py','validate_upload_cpu.py','write_report_and_scope.py']},'derived_counts':prod,'new_real_forward_attempts':0,'new_GPU_PPU_model_calls':0,'CUDA_initializations':0,'historical_file_modifications':0,'accepted_original_source_result_count':32,'scope':'read-only result-stage audit; no model execution, new algorithm, counterfactual experiment or R5D'})
 lookup={(int(r['sample_idx']),r['condition'],int(r['candidate_id'])):r for r in lineage}
 def core_table(sid,oid):
  rr=[]
  for c in STATE_ORDER:
   for t in (50,70):
    r=next(r for r in trace if r['sample_idx']==str(sid) and r['physical_GT_id']==oid and r['condition']==c and float(r['iou_threshold'])==t/100)
    rr.append([c,t,f"{float(r['raw_max_IoU']):.8f} / {r['raw_max_IoU_candidate_id']} / {float(r['raw_max_IoU_candidate_score']):.8f}",'/'.join(r['high_IoU_count_'+s] for s in STAGES),r['final_max_IoU_candidate_id'],f"{float(r['final_max_IoU']):.8f}",r['TP_FN'],r['matched_candidate_id'] or 'NULL',f"{float(r['matched_IoU']):.8f}" if r['matched_IoU'] else 'NULL',f"{float(r['matched_score']):.8f}" if r['matched_score'] else 'NULL',r['observed_final_bottleneck']])
  return mdtable(['条件','IoU阈值%','raw maxIoU / ID / Score','高IoU数 raw/Score/quality/topK/NMS/range','最终maxIoU候选ID','最终maxIoU','TP/FN','匹配ID','匹配IoU','匹配Score','终点证据'],rr)
 report=f'''# RD3-R5C 科学审计：已记录检测候选的阶段路径

状态：**R5C_PASS**。本轮仅进行CPU离线复算，读取已验收32条真实模型结果；新forward=0，CUDA初始化=0，GPU/PPU模型调用=0，历史修改=0。未执行R5D或任何新的控制变量实验。

独立验收：{v['passed_check_count']:,}项通过，{v['failed_check_count']}项失败，关键候选链UNRESOLVED={v['UNRESOLVED_required_count']}。结果级/阶段级通过不表示已识别Attention、MetaFlow等模块的因果贡献，也不表示可识别跨条件的物理FP目标。

## 证据、单位与原生口径

8个冻结帧，3个scene，56条帧内GT观测，25个跨帧scene-object身份；四个Remote边际来自同一帧，不能作为四个独立实验单位。条件F=ABC、G-B=AC、G-C=AB、E=A，物理slot0/1/2保持固定。全部32条NPZ按R5B_RAW_SHARD_LOCATOR.csv定位并核对SHA256；R5A/R5B报告、CSV和物理GT映射按既有封存Hash复核。再次核对3,589个历史文件，包括原生源码、epoch19 checkpoint和既有运行/预算记录，全部未变。

原生head为cls/reg/dir；实际输出无iou_preds，不启用学习式IoU质量分数重标定。先sigmoid(cls)并严格Score>0.2，再执行large/z几何过滤，再在NMS内部按原生argsort逆序截取top1000，以BEV IoU>0.15抑制，最后按gt_range过滤全部8个角点。large-box历史实现以y跨度充当第三个逻辑量的细节原样保留，未修正算法。评价为Score降序对尚未匹配的GT贪心唯一匹配，BEV IoU≥0.5/0.7；不是3D体积IoU。完整配置、源码副本及Hash见R5C_NATIVE_PIPELINE_CONTRACT.json。

32条最终框、Score、NMS keep和suppressor均精确重放；head sigmoid误差≤1e-6、原生reg/方向解码≤1e-5（既有容差，未放宽）。全部32条large/z实际剔除0，top1000截断0，NMS抑制884，最终range剔除2。不能用后两项总数替代某个GT的候选链。

IoU70汇总原样复现：F=45/18/11、G-B=40/18/16、G-C=42/16/14、E=39/13/17（TP/FP/FN）。IoU50为47/16/9、43/15/13、43/15/13、40/12/16。本轮不重定义AP，也不进行逐帧AP平均。

## RQ1：725的两个Ego-only检出为何不同

以下表中的raw最大IoU候选、最终最大IoU候选和最终唯一匹配候选是三个不同概念。FN匹配字段为NULL；最终最大IoU为0可以是真实几何零交叠，但绝不能填入FN的匹配字段。完整448行覆盖56 GT×4条件×2阈值，核心三GT共24行。

### GT14014，scene93/frame31

{core_table(725,'14014')}

IoU70模式E/GB/GC/F=`1000`，IoU50=`1111`。最终候选6288四条件均保留、Score均远高于0.2，但GT IoU从E的0.71411580降至GB 0.69406426、GC 0.69263726、F 0.69118702。因此它在0.5仍为TP，0.7则成为FP，同时GT成为FN。

三个Remote条件**不是没有raw IoU70候选**：GB/GC/F分别有4/3/1个，但这些候选全部在Score阶段消失，最高Score分别为0.14720663/0.16721418/0.07323053，均≤0.2。不存在可进入NMS的IoU70候选，也不存在最终高IoU候选被唯一GT匹配排挤。E有11个raw高IoU候选，4个通过Score，NMS留下1个，最终与该GT唯一匹配。可直接支持“SCORE_OR_QUALITY_FILTERING（本例实际为Score）”和同一存活anchor跨IoU阈值的几何变化；不能把多分支共同变化解释成单独Score模块的因果效应。

### GT4071，scene93/frame31

{core_table(725,'4071')}

IoU50模式=`1000`，IoU70=`0000`。所有条件raw maxIoU都高于0.7（E/GB/GC/F约0.75334/0.77420/0.77866/0.78552），但IoU70候选全部Score≤0.2：最高分别0.03436889/0.04246713/0.05345231/0.05810392，所以四条件在0.7均FN。

0.5下路径不同：E最终保留7028，IoU=0.50790995、Score=0.63845110，成功匹配；Remote条件仍有经过Score和quality的IoU50候选，GB/GC/F进入NMS的数量为3/4/4，但**全部被7030抑制**。7028在这些条件仍有IoU=0.50105757/0.50581896/0.50777972，满足0.5，却不在最终结果中。7030最终保留，其对目标IoU只有0.44413048/0.44115186/0.43618697。

最直接的排序和NMS证据：E中7028 Score 0.63845110 >7030 0.63362074，7028抑制7030，两者BEV IoU 0.88983822。GB中7030 0.63632566 >7028 0.61736965，7030抑制7028，IoU 0.89145982；GC为0.62347806>0.60308087、IoU0.87680626；F为0.59578764>0.56727004、IoU0.86597055。其余全部高IoU候选的suppressor边同样保存。可支持NMS候选生存/Score排序现象，不能宣称原生NMS算法发生改变；输入候选几何、方向数值和Score本已共同变化。

因此，同帧两个GT的主要瓶颈不同：14014的IoU70损失为高IoU候选Score全失，并且存活6288几何低于0.7；4071的IoU50损失为已有高IoU候选在NMS全失。不能将二者统称同一种损失机制。

## RQ2：324/GT29118 Full-only的候选与阶段

{core_table(324,'29118')}

IoU70模式=`0001`，IoU50=`0101`。Full最终匹配ID **6512**，Score=0.71313167、BEV IoU=0.70197785。其query_index=3256，grid_y=40、grid_x=56、rotation_index=0。相同6512在G-B也最终保留，Score=0.69673079、IoU=0.69980067，因此在IoU50匹配、IoU70不匹配。E和G-C中6512的Score分别0.02186639、0.05360840，未通过Score门槛，目标IoU分别0.44693932、0.45949039。

G-B并非raw几何不足：8个raw IoU70候选，其中6514（GT IoU0.71320564、Score0.65535086）和6516（GT IoU0.75688696、Score0.47011870）进入NMS；二者均被6512抑制，pair IoU分别0.96695858、0.91044128，6512被保留但与GT只有0.69980067。Full则有4个高IoU候选通过Score/NMS入口，6512本身跨过0.7并保留。E/GC在两个阈值均为高IoU候选Score全失。

6512四条件dir_bin均为0，不能声称dir分类翻转；原始yaw回归值为E −0.01883421、GB −0.04221465、GC −0.01139762、F −0.02581332弧度，reg/dir logits/角点随条件改变。GB/F的原生quality-pass排序均为0基rank13。分数和回归几何变化、NMS竞争都能从记录核验，但该比较没有将它们逐一控制固定。Full-only不是“另外生成一个可稳定物理识别的新目标”的证据；它与同一固定head格点候选的跨阈值几何和生存路径相符。

## RQ3：全部9条不利边际的FP来源

以IoU70审计，方向始终加入Remote之后减之前。以下anchor仅为固定head格点ID，不是跨条件物理FP对象ID；所有对应的原始角点、Score、最大GT IoU、before/after阶段、NMS suppressor及原始分片Hash见R5C_FP_PROVENANCE.csv。新增/消失以“该格点候选是否进入最终预测”为准。

'''
 notes={
 (324,'U_B_given_A'):'10709由Score未过转为通过（0.189618→0.200640）；最终FP。',
 (324,'U_C_given_A'):'6512由Score未过转为最终FP；IoU0.699801<0.7；此候选在0.5为TP。',
 (354,'U_C_given_B'):'6514跨Score门槛形成FP；7086由原NMS抑制转为保留，同时6926被7086抑制而消失，后两者净差0。',
 (354,'U_C_given_A'):'6514跨Score门槛形成FP（0.016020→0.635403）。',
 (384,'U_B_given_C'):'11019跨Score门槛形成FP（0.007686→0.457840）。',
 (384,'U_B_given_A'):'11019、12137跨Score门槛形成FP；7072从NMS抑制转保留，6912反被7072抑制而消失，交换净差0。',
 (725,'U_B_given_C'):'6254跨Score门槛形成FP（0.195276→0.211814）。',
 (725,'U_B_given_A'):'6254跨Score门槛；6288由GT14014的TP转为FP；7030替代7028成为NMS存活FP，后两者FP计数净差0。',
 (725,'U_C_given_A'):'6288由TP转为FP；7030替代7028成为NMS存活FP，后两者净差0。'}
 harmful=[x for x in summ if x['analysis_role']=='MANDATED_CLEAR_HARMFUL'];harmful.sort(key=lambda x:([324,354,384,725].index(int(x['sample_idx'])),list(CONTRASTS).index(x['contrast'])))
 report+=mdtable(['sample','加入Remote边际','FP before→after','新增最终FP','TP→FP','消失最终FP','FP→TP','已核验路径'],[[r['sample_idx'],r['contrast'].replace('U_',''),r['FP_before']+'→'+r['FP_after'],r['ADDED_FINAL_FP_AT_ANCHOR'],r['TP_TO_FP_AT_ANCHOR'],r['REMOVED_FINAL_FP_AT_ANCHOR'],r['FP_TO_TP_AT_ANCHOR'],notes[int(r['sample_idx']),r['contrast']]] for r in harmful])
 report+='''

每条均满足ΔFP=新增最终FP+TP→FP−消失最终FP−FP→TP。9个边际合计13次新增最终FP格点事件、2次TP→FP、4次消失FP、10次持续FP、0次FP→TP，净ΔFP合计+11。13次新增中9次来自该anchor原先Score未过、4次原先被NMS抑制；4次消失均为NMS抑制。4次NMS新增分别与消失抵消，因此净差可写成9次Score门槛进入加2次几何阈值TP→FP。它是这些重叠边际的记账恒等式，不是11个独立对象或某模块的因果效用估计。

全部9条不利边际的最终FP均不满足任何GT的0.7 IoU（25个after-FP格点事件的max GT IoU<0.7）；没有证据把这些FP增加归因于“满足IoU阈值但GT已被更高Score候选占据”的唯一匹配竞争。7028在725/E是IoU50 TP，但在本节IoU70计数为原有FP；不得混用阈值。所有重点新增FP均经过Score/quality、NMS和range；不能只因NMS总剔除数改变便归因为该FP的来源。

跨条件几何代理采用BEV IoU≥0.5且两侧候选均仅有一个相互符合对象，134条最终预测侧记录满足唯一代理，15条无达到阈值的代理，0条多义代理。唯一几何代理仍不等于物理身份认证；无代理保持未知，不强行对应。部分同anchor角点也有较大变化，例如384 B|A的11019 before/after BEV IoU≈0.49597，明确标记几何位移，不将其称为同一物理虚假物体。

424正FP交互对照：E唯一FP anchor7046 Score0.21263678；GB/GC/F同anchor均Score未过（GB0.14644839、GC0.14291371，F值保存在lineage），三条件最终FP都为0。因此两种Remote单独加入各减少1个FP，合用并不再额外减少第二个FP，二阶FP交互为+1。这一正交互不代表Full比E产生更多FP。

## RQ4：交互项对应什么事件

交互定义I=F−(G-C)−(G-B)+E；对TP、FP、FN分别计算，不合成加权单一效用。四条件顺序下表统一为E、G-B、G-C、F。FP项是在相同head格点上FP指示变量的精确代数分解，候选空间可比不代表物理对象可比。

'''
 inter70=[next(r for r in inter if r['sample_idx']==str(s) and float(r['iou_threshold'])==.7) for s in (324,725,384,424)]
 report+=mdtable(['sample','TP四条件','FP四条件','FN四条件','I_TP / I_FP / I_FN','关键GT模式'],[[r['sample_idx'],','.join(str(json.loads(r['four_condition_counts'])[c]['TP']) for c in STATE_ORDER),','.join(str(json.loads(r['four_condition_counts'])[c]['FP']) for c in STATE_ORDER),','.join(str(json.loads(r['four_condition_counts'])[c]['FN']) for c in STATE_ORDER),r['I_TP']+' / '+r['I_FP']+' / '+r['I_FN'],'29118:0001' if r['sample_idx']=='324' else '14014:1000' if r['sample_idx']=='725' else 'GT二阶项均0'] for r in inter70])
 report+='\n\n'
 for r in inter70:
  report+=f"**sample{r['sample_idx']} FP逐anchor非零交互：**\n\n"
  nz=json.loads(r['nonzero_FP_anchor_interactions']);report+=mdtable(['anchor','FP位串 E/GB/GC/F','该格点I_FP','阶段信息'],[[x['anchor_id'],''.join(str(x[n]) for n in ['FP_E','FP_GB','FP_GC','FP_F']),x['I_FP_anchor'],str(x['stage_codes'])] for x in nz])+'\n\n'
 report+='''阶段码1=Score淘汰、2=large/z、3=top1000、4=NMS淘汰、5=最终range、6=最终保留。完整0.5/0.7双阈值分解、GT模式与每一层候选数量见R5C_INTERACTION_MECHANISM_MATRIX.csv及R5C_FP_ANCHOR_INTERACTION.csv。

- 324：TP+1来自GT29118的0001，即只有Full检出；FP−2来自6512在GB为FP而Full跨0.7成为TP（−1），以及10709仅在GC作为FP保留（−1；其余条件Score未过）。
- 725：TP+1来自GT14014的1000，即E检出而全部Remote条件不检出；不能称为联合融合收益。FP−1分解为6288在Remote条件均变FP（−1），7028仅E保留且在0.7为FP（+1），7030仅Remote保留为FP（−1）；7028/7030为有确切suppressor的生存交换，不是两个已验证的物理对象。
- 384：TP交互0；6912仅E存活为FP（+1），7072仅Remote存活为FP（−1），12137仅GC过Score且存活为FP（−1），净−1。11019在GC和Full均为FP，二阶项为0，虽然它出现在不利边际中。
- 424：TP交互0；7046仅E为FP（+1），Remote均被Score剔除，产生FP正交互+1。

其他四样本的IoU70二阶TP/FP/FN项为0，所以全8帧仍为+2/−3/−2；IoU50总交互为+1/−2/−1，725的正TP项来自另一个GT4071的1000。不能把两个阈值的正项拼成同一目标机制，也不能将8帧或56 GT当作相互独立的重复实验。

## RQ5：证据等级、标签边界与科学判断

**直接计算事实。** 原始32条head/候选、sigmoid Score、角点BEV IoU、全部处理阶段成员集合、NMS keep/suppressor/Score/pair IoU、最终Score序列的唯一GT匹配、四条件计数和交互恒等式均可逐ID复查。每个GT记录首个“全部可用高IoU候选消失”的阶段，并另存“至少一个被NMS抑制”，二者没有混淆。FN匹配Score/IoU/ID缺失是正确语义，不是数据损坏。

**与阶段解释一致的现象。** Remote条件存在条件性检测损失：725/14014在0.7、725/4071在0.5，以及9个IoU70不利计数边际均有直接结果证据。近端表现集中在Score门槛、Score排序/NMS生存及存活框几何跨评价阈值；本任务选定核心链未见top1000或large/z瓶颈、最终range瓶颈或唯一GT竞争导致的核心失检。对给定已保存输出，原生下游流水线足以重现这些TP/FP/FN，但无法仅靠“后处理算法改变”解释条件间差异，因为同一原生算法始终未变，输入head几何/Score本已改变。

**尚不能验证的上游模块假设。** Attention、MetaFlow、特征融合、时间历史及检测头内部哪一个因素独立导致了候选变差，现有保存结果不能分离验证。部分事件伴随几何、分数、NMS竞争共同变化，应保留MULTIPLE_STAGE_CHANGES及raw_geometry_changed；不能将NMS下游观察当作其独立因果干预效果。跨条件FP物理目标身份同样未测量。这两类未知是本研究的识别边界，不是伪造候选链或忽略FAIL的理由。

标签规则：RAW_GEOMETRY_DIFFERENCE只指raw高IoU存在性翻转，不能因任意角点数值变化就贴此标签；原始几何数值是否变化另列。SCORE_OR_QUALITY_FILTERING区分实际Score与large/z；本轮核心实际为Score。NMS_SUPPRESSION要求FN侧有preNMS高IoU候选且所有这些候选均有实际suppressor、NMS后无高IoU候选。MULTIPLE_STAGE_CHANGES要求至少两种有数值证据的材料变化（成功anchor跨阈值、Score成员、NMS成员、raw存在性），不充当无法解释的兜底标签。全部有状态变化的GT边际及核心三GT双阈值稳定对照共50行；不把这些重叠标签频次解释为独立机制发生率。

已有证据足以提出可证伪、局部的控制变量问题，但不足以直接声称某个上游模块有害或发展新的效用算法。**下一步最小实验仅建议、不执行：** 在相同冻结帧/固定head身份上，先经单独授权作保存候选的固定几何/固定Score双向控制：725/14014检验6288的几何跨0.7与高IoU候选Score门槛的相对作用；725/4071固定角点只控制7028/7030的Score排序，验证NMS竞争路径；324/29118固定候选集合分别控制6512几何和Score，检验0.69980/0.70198的边界与6514/6516竞争。必须预先定义对照及阈值，不能以调阈值寻找满意结果。该建议不证明模块因果；若进一步识别Attention/MetaFlow，需另行设计单因素干预并获得真实forward授权。本轮没有执行这些离线反事实或新forward实验。

## 独立验收、可恢复工程记录与交付

生产分析与独立验收分别由两个CPU进程运行。独立验收器没有导入生产脚本：重新打开32个原始NPZ，以独立active-mask NMS循环和独立GT贪心循环复核，复用的只是Hash核验的R5A NumPy/Shapely浮点几何基本运算。核验32条框/Score/TPFPFN、884条NMS边、448行GT迹线与R5B状态、4,660条候选主键/原始数据/缺失语义、50行标签判据、13个FP对照（9个不利+424四种边际）、8行焦点交互及全部8帧总交互。验收结果见R5C_INDEPENDENT_VALIDATION.json。

生产阶段曾有NumPy int64的JSON导出错误，独立验收器首版曾使用R5A不存在的final_prediction_count字段。分别以显式转Python int、按实际R5A final_predictions与R5B final_prediction_count字段映射修复；未改动原始数据、数值算法或容差。失败源码和日志保存在source_revisions/及logs/，最终完整运行均通过，不隐藏失败尝试。

主要文件：

- R5C_CASE_MANIFEST.csv：45个预先指定案例（24核心GT条件阈值、9不利边际、424四种边际、8交互行）。
- R5C_GT_STAGE_TRACE.csv：448行，覆盖全部56 GT×4条件×2阈值；含GT身份/Hash、raw极值、所有stage高IoU ID、匹配字段和源Hash。
- R5C_CANDIDATE_LINEAGE.csv：4,660行，包含全部Score通过、全部raw IoU≥0.5候选、极值、最终候选及suppressor闭包；四条件统一格点集合。导出选择不影响完整12,800候选的原始重放，未删候选以迎合结论。
- R5C_GT_TRANSITION_MECHANISMS.csv：50行有数值支持的标签；关键链UNRESOLVED=0。
- R5C_FP_PROVENANCE.csv：31行格点事件，包含9不利边际与424对照的全部FP并集；FP_MARGINAL_ACCOUNTING.csv逐对精确相加。
- R5C_INTERACTION_MECHANISM_MATRIX.csv及FP_ANCHOR_INTERACTION.csv：焦点样本双阈值四条件及FP二阶分解。
- R5C_NMS_SUPPRESSOR_EDGES.csv：全部884条NMS边；R5C_GT_MATCH_ORDER.csv：462个最终预测×阈值的贪心匹配事件。
- evidence/：32份压缩关键候选实测数组，包含原始角点/head/Score、全12,800行Score及GT IoU矩阵、全部stage ID和最终匹配；所有导出数组由独立验收器与原始NPZ核对。
- historical_reference/：必要R5A/B物理GT映射、候选清单、定位表、计数及报告的Hash一致副本；source_reference/：纯CPU几何/受限存储读取实现及原生源码/配置文本，仅阅读，不导入模型。
- analyze_r5c.py、stage_common.py、validate_r5c_independent.py：服务器上完整CPU重现；validate_upload_cpu.py：无需原始NPZ的压缩证据重放，原生NMS、最终匹配及关键候选路径可复查。

**打包范围。** R5C_UPLOAD_BUNDLE.zip包含当前报告、全部CSV、JSON验收、脚本、关键证据、必要来源副本与SHA256SUMS.txt；不打包整套大型原始科学NPZ、输入/权重和attention等未使用中间张量。低Score且与任何GT IoU<0.5、亦非所选极值的候选角点未全部导出。全量Score/GT IoU矩阵已保存；压缩包可重放所有Score通过候选的NMS、最终框和匹配，并重算每个关键高IoU候选几何。对未导出角点的全量矩阵原始几何计算、完整head、冻结输入/模型与历史运行预算身份，仍须在服务器按原始NPZ路径/Hash运行独立验收器，不能以便携验收替代。SHA256SUMS.txt不包含自身及ZIP以避免循环Hash；ZIP独立Hash另存交付文件。

至此停止：未启动新forward、训练、扩样、AoI实验或效用预测器开发。
'''
 (ROOT/'RD3_R5C_SCIENTIFIC_ASSESSMENT.md').write_text(report)
 readme='''# R5C CPU离线证据复核

最终科学结论与限制见 RD3_R5C_SCIENTIFIC_ASSESSMENT.md。当前状态以R5C_INDEPENDENT_VALIDATION.json为准；source_revisions中的FAIL是已保留的工程调试记录。

服务器完整独立复核（已实际运行，不调用任何模型）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/RD3_R5C_STAGE_MECHANISM_AUDIT/validate_r5c_independent.py
```

压缩包解压后，在解压目录运行 `python -B validate_upload_cpu.py`，需要NumPy、Shapely；该入口只读重放并将结果写到stdout，不导入torch/opencood。便携复核不替代服务器原始Hash身份和省略角点的原始几何计算；详见主报告打包范围。包中native_text是原生源码文本参考，不能作为本轮允许执行模型的授权。

表格空字符串表示未匹配/未评估/无suppressor，不可填0；IoU50和70严格分开。same anchor仅表示固定head网格格点。跨条件FP物理身份未测量。
'''
 (ROOT/'README_CPU_REPLAY.md').write_text(readme)
 print('REPORT_AND_SCOPE_WRITTEN',v['state'],flush=True)
if __name__=='__main__':main()
