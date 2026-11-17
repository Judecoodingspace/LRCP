"""Render accepted R5D observations and scientific limits; no experiment launch."""
from cpu_common import *
from datetime import datetime,timezone,timedelta
import platform,shapely

def mt(head,rr):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(str(x).replace('|','∣') for x in r)+' |' for r in rr])
def main():
 v=load(ROOT/'R5D_INDEPENDENT_VALIDATION.json');geo=load(ROOT/'R5D_FULL_GEOMETRY_PACKAGE_VALIDATION.json');require(v['state']=='R5D_PASS' and geo['state']=='FULL_GEOMETRY_PACKAGE_PASS','ALL_VALIDATORS_MUST_PASS')
 results=rows(ROOT/'R5D_FACTORIAL_RESULTS.csv');focal=rows(ROOT/'R5D_FOCAL_GT_OUTCOMES.csv');sw=rows(ROOT/'R5D_NMS_SCORE_SWAP.csv');eff=rows(ROOT/'R5D_GEOMETRY_SCORE_INTERACTION.csv');off=rows(ROOT/'R5D_OFF_TARGET_EFFECTS.csv');cand=rows(ROOT/'R5D_CANDIDATE_STAGE_CHANGES.csv');pg=rows(ROOT/'R5D_PER_GT_FACTORIAL.csv');allrun=rows(ROOT/'R5D_ALL_RUN_RESULTS.csv')
 def get(rid,oid,t):return next(r for r in focal if r['run_id']==rid and r['physical_GT_id']==oid and float(r['iou_threshold'])==t/100)
 def short(r,t):return '/'.join(r[n+str(t)] for n in ['TP','FP','FN'])
 def status(r):
  sid=int(r['sample_idx']);rid=r['run_id'];return '/'.join(get(rid,oid,t)['TP_FN'] for oid,t in ([('14014',70),('4071',50)] if sid==725 else [('29118',70)]))
 body=f'''# RD3-R5D：冻结候选Geometry–Score的2×2离线反事实审计

**R5D_PASS**。本轮完成且仅完成四个预先冻结pair的16个Geometry–Score组合，以及725的G-B/G-C/F三个原始条件下7028/7030的两Score交换诊断。全部从每条完整12,800候选开始重做原生后处理。新模型forward=0，GPU/PPU调用=0，CUDA初始化=0，历史修改=0，推理账本attempt增量=0。没有训练、扩样、修改网络或自动开展下一阶段。

最终独立验收：{v['passed_check_count']:,}项通过、{v['failed_check_count']}项失败、{v['unverifiable_required_items']}项必需项目无法验证。6份原始完整候选来源的全部3D角点、方向修正与坐标投影另经CPU核验：最大角点误差{max(x['full12800_direction_corrected_3D_corners_max_abs_error'] for x in geo['source_records']):.10g}，沿用1e-5容差；sigmoid Score沿用1e-6容差，未放宽。实验使用捕获角点和Score原值，没有用CPU近似角点替换。

本次为独立版本重执行：原有R5D目录保持只读；复用其已审查CPU脚本，从实际locator原始NPZ重新生成全部19条结果，没有复制既有结果代替运行。版本准备及脚本来源见R5D_VERSION_PROVENANCE.json。

## 冻结证据、实验单位及口径

实际locator来自R5B_RAW_SHARD_LOCATOR.csv，32条原始NPZ均再次核对SHA256。研究重放仅涉及其中6份完整来源：725的E/G-B/G-C/F，324的G-B/F，不以R5C压缩包的部分角点替代。R5C主报告、合同、GT迹线、candidate lineage、suppressor和匹配顺序、独立验收，以及R5A物理GT映射均按既有Hash核验。历史{v['historical_files_verified_unchanged']:,}个文件再次核对保持不变，包括历史科学分片、源码、checkpoint、日志与预算证据。

本阶段实际分析2个冻结帧（scene93/frame31的sample725、scene28/frame30的sample324），涉及11个不同帧内GT，其中3个重点GT。相同E输出在P1/P2/P3重复使用，不是三次独立模型重复；四种组合也不是四个独立样本。R5A/B/C的8帧背景不等于本轮扩大了研究样本。

Geometry包固定原始全部三维角点（含方向修正及投影），并保留decoded_before_dir、reg_preds、dir_preds及其来源Hash；Score为原始sigmoid分类分数。对齐检查覆盖全候选0…12799、80×80×2 head格点、真实冻结anchor、相同输入/slot物理身份、GT角点与physical ID、坐标变换和配置。所有原始数组只读，并在执行后核对原始leaf Hash未变。

流水线严格保留：Score>0.2；无iou_preds学习式重标定；large/z几何过滤（包括历史large-box第三个逻辑量实际使用y跨度的细节）；NMS内部原生argsort逆序top1000；float32 BEV IoU>0.15抑制；最后8角点range过滤；按np.argsort(-Score)对剩余GT贪心唯一匹配，阈值分别≥0.5/≥0.7。没有修改阈值、NMS、匹配规则或既有容差，没有计算单帧AP或加权效用。

## 完整2×2结果

P1：725，条件0=E，条件1=G-B。P2：725，E→G-C。P3：725，E→F。P4：324，G-B→F。G0_S1表示固定条件0的完整几何、换用条件1全部Score；G1_S0反之。所有8条原始组合先完成逐candidate、逐stage、逐GT的精确验收，才执行混合。重复原始E数组/结果一致。

'''
 body+=mt(['pair','组合','TP50/FP50/FN50','TP70/FP70/FN70','最终预测数','重点GT状态'],[[r['pair_id'],r['combination'],short(r,50),short(r,70),r['final_prediction_count'],status(r)] for r in results])
 body+='''

重点GT状态：P1–P3为“14014@IoU70 / 4071@IoU50”；P4为“29118@IoU70”。所有组合同时保存两阈值全部GT结果。分母始终为该帧5或6个GT；TP+FN=GT数，TP+FP=最终预测数均通过。

## RQ1：725/GT14014的IoU70损失

'''
 rr=[]
 for p in PAIRS[:3]:
  for combo in COMBOS:
   r=get(p['pair_id']+'_'+combo,'14014',70);rr.append([r['run_id'],r['TP_FN'],'/'.join(r['high_count_'+s] for s in STAGES),r['matched_candidate_id'] or 'NULL',r['matched_IoU'] or 'NULL',r['matched_score'] or 'NULL',r['bottleneck']])
 body+=mt(['组合','GT14014@70','高IoU数 raw/Score/quality/topK/NMS/range','匹配候选','匹配IoU','匹配Score','终点'],rr)
 body+='''

三个pair的GT14014状态按G0S0、G0S1、G1S0、G1S1均为 **1100**：

- 固定E几何，使用任一Remote Score，均仍由6288成功匹配；IoU保持0.7141157985，Score分别变为0.9626621604、0.9593778253、0.9462591410。
- 固定Remote几何，换回E Score，均**没有恢复**；6288仍最终保留，但IoU固定为G-B 0.6940642595、G-C 0.6926372647、F 0.6911870241，均低于0.7。无其他候选替代成为该GT的匹配。
- 三种Remote几何的raw IoU70候选数为4/3/1；不论使用E还是相应Remote Score，全部未过Score门槛。相反，E几何的11个raw高IoU候选在Remote Score下有5/5/6个通过Score（E Score下4个），NMS均保留高IoU的6288。因此R5C观察到的“高IoU候选Score全失”并不意味着把E分数整体换回来就能救回：高IoU候选落在哪些固定格点，随几何包改变。

就本轮两个给定Score向量而言，几何来源替换能改变该GT的IoU70状态，而Score整体替换不能。它是有限候选层控制的结果，不能推导“Score普遍无关”“Geometry分支是独立网络原因”或对其他Score赋值的必然结论。GT14014在所有2×2组合的IoU50均仍为TP。

其他GT状态：整体Score从E换为Remote时，GT4071在IoU50同步由TP变FN；换回E Score则恢复。除此之外没有非重点GT的TP/FN变化。完整双基线比较及匹配anchor变化见OFF_TARGET_EFFECTS。

## RQ2：725/GT4071的IoU50与NMS排序

P1/P2/P3的2×2模式均为 **1010**：固定任一几何，E Score对应TP，相应Remote Score对应FN。固定Remote几何换回E Score时，7028恢复最终保留并匹配，IoU分别0.5010575652、0.5058189631、0.5077797174，Score均为E原值0.6384510994；7030被抑制。固定E几何换成Remote Score则保留7030，全部可用IoU50候选被抑制，目标变FN。

为避免把全Score向量变化误当局部排序效应，按授权仅交换每个原始Remote条件的7028、7030两个Score；其他12,798个Score与全部几何逐元素保持不变，两原Score都>0.2。重新运行完整流水线后的实际结果：

'''
 body+=mt(['原始条件','7028 Score 原→交换后','7030 Score 原→交换后','7028保留/7030被其抑制','4071匹配IoU@50','匹配Score@50','ΔTP50/FP50/FN50','ΔTP70/FP70/FN70'],[[r['condition'],f"{float(r['candidate_7028_score_before']):.9f}→{float(r['candidate_7028_score_after']):.9f}",f"{float(r['candidate_7030_score_before']):.9f}→{float(r['candidate_7030_score_after']):.9f}",r['candidate_7028_NMS_keep_after']+'/'+str(r['candidate_7030_suppressor_after']=='7028'),r['GT4071_match_IoU_after50'],r['GT4071_match_score_after50'],'/'.join(r['delta_'+n+'50'] for n in ['TP','FP','FN']),'/'.join(r['delta_'+n+'70'] for n in ['TP','FP','FN'])] for r in sw])
 body+='''

三次均恢复7028，7030均由7028抑制，GT4071重新成为IoU50 TP；所有其他GT状态不变，IoU70仍FN。交换后并未产生其他NMS keep新增或减少：keep集合仅7028加入、7030移除，最终预测数分别仍为5/6/6。IoU50因将最终FP7030替换为TP7028，TP+1/FP−1/FN−1；IoU70两者仍均为FP，FP计数不变但FP格点身份由7030换为7028。

完整记录同时保留被抑制者suppressor的连带变化：G-B涉及7028、7030、7032、7186、7188、7190、7192共7个ID；G-C和F各再含7026，共8个ID。除两主候选外，它们仍被抑制，只是suppressor从7030改为7028。几何固定时的NMS排序敏感性因此得到直接支持；没有验证模型如何学习这些Score，也没有触及Attention/MetaFlow内部因果。

## RQ3：324/GT29118 Full-only检测

'''
 rr=[]
 for combo in COMBOS:
  rid='P4_'+combo;r=get(rid,'29118',70);cc=next(x for x in cand if x['run_id']==rid and x['candidate_id']=='6512');rr.append([combo,r['TP_FN'],json.loads(cc['geometry_IoU_by_physical_GT'])['29118'],cc['score'],cc['NMS_keep'],r['matched_candidate_id'] or 'NULL',r['matched_IoU'] or 'NULL','/'.join(r['high_count_'+s] for s in STAGES)])
 body+=mt(['组合','GT29118@70','6512的GT IoU','6512 Score','6512 NMS keep','GT匹配ID','GT匹配IoU','高IoU数 raw/Score/quality/topK/NMS/range'],rr)
 body+='''

固定G-B几何，仅使用F Score，目标仍FN；固定F几何，使用G-B Score，目标成为TP。6512在全部四组合均保留，没有其他候选替代它。几何固定时6512 IoU逐元素所依赖角点不变；Score固定时分数不变。6514、6516在全部四组合都通过Score进入NMS，但均被6512抑制；源几何和各自Score、实际suppressor/抑制pair IoU都在候选表和完整NPZ中。

6512的G-B IoU为0.6998006701，F为0.7019778490，差约 **0.00217718**；距0.7分别−0.00019933和+0.00197785。它是评价阈值附近的离散检出敏感性，不是稳健的大幅定位改善。在这两个已给定Score向量上，选择Full几何足以改变该GT的IoU70状态；不能据此将其归因某个网络几何分支的独立机制。四组合IoU50均TP。

其他GT没有TP/FN状态变化；换用Full Score时GT34709的匹配候选由7124改为7126，仍为TP。选择Full几何时6512由IoU70 FP变TP，因此在最终预测数不变时TP+1、FP−1、FN−1。

## RQ4：分量差分及Geometry–Score交互

所有差分均为替换之后减之前。I_GS=G1S1−G1S0−G0S1+G0S0；这不是Remote B/C的二阶交互，也不是网络分类/回归分支的因果分解。下表每个三元组依次TP/FP/FN。

'''
 def triple(r,prefix):return '/'.join(r[prefix+'_'+m] for m in ['TP','FP','FN'])
 body+=mt(['pair','IoU','ΔG∣S0','ΔG∣S1','ΔS∣G0','ΔS∣G1','I_GS'],[[r['pair_id'],r['iou_threshold'],triple(r,'Delta_G_given_S0'),triple(r,'Delta_G_given_S1'),triple(r,'Delta_S_given_G0'),triple(r,'Delta_S_given_G1'),triple(r,'I_GS')] for r in eff])
 body+='''

8个pair×阈值单元的TP/FP/FN与最终预测数I_GS均为0；42个GT×pair×阈值记录的I_GT也均为0，没有被总数相消掩盖的GT非加性。对本轮有限替换及这些最终计数，ΔG不随所选S来源改变，ΔS不随所选G来源改变。未观察到所问的“显著”非加性计数变化；这里不作统计显著性检验。

这不证明输出因素普遍独立，也不表示中间阶段全都线性。例如GT14014@70通过Score的高IoU候选数，P1为4/5/0/0、P2为4/5/0/0、P3为4/6/0/0，按同一代数有−1/−1/−2的中间数量交互，但其最终TP指示仍为1/1/0/0。NMS、阈值与候选集合依然是非线性的；只有本次所测最终计数未产生非零二阶项。R5B/C的Remote B/C交互+2/−3/−2未被改动，也没有被本轮不同因素、不同研究单元的零交互否定。

## 全部额外GT、最终预测及FP影响

R5D_OFF_TARGET_EFFECTS.csv保存54条run/原始参考/阈值比较；每个2×2组合与该pair两个原始基线比较（排除自比较），三个局部交换分别与其原始条件比较。全量记录GT状态变化、稳定状态但匹配anchor变化、最终预测与FP新增/消失、TP↔FP、全部NMS keep和suppressor变化，阈值不混用。

没有任何非重点GT的TP/FN改变；但不能说“其他预测完全不变”。相对各pair的G0S0：

'''
 impact=[]
 for r in off:
  if r['reference_run_id'].endswith('G0_S0') and r['run_id'].endswith(('G0_S1','G1_S0','G1_S1')) and float(r['iou_threshold'])==.7:
   impact.append([r['run_id'],r['final_candidate_added'],r['final_candidate_removed'],r['FP_anchor_added'],r['FP_anchor_removed'],r['GT_match_anchor_changes_without_status_change']])
 body+=mt(['组合 vs G0S0','最终ID加入','最终ID移除','FP70 ID加入','FP70 ID移除','保持检出但匹配anchor变化'],impact)
 body+='''

725的主要额外FP来源6254：只有G-C或F Score激活它，所以P2/P3的G0S1和G1S1在两个IoU阈值均新增该格点FP，最终预测数+1；E或G-B Score不激活。P3的Full Score还使物理GT7870的匹配从6478变为6318，两个阈值仍TP；P4的Full Score使GT34709从7124变为7126，同样不改变检出状态。6288在Remote几何下只于IoU70转FP；7028/7030的NMS交换在IoU50是TP损失/恢复，在IoU70是FP换格点、计数净差0。所有局部交换除4071@50外没有其他GT状态或稳定匹配anchor改变。

每个FP差分都通过ΔFP=新增最终FP+TP→FP−消失最终FP−FP→TP核验。固定head格点是计算对应关系，不是FP物理目标身份；不声称新增了可跨条件认证的虚假物体。完整0.5/0.7清单以CSV为准，不把IoU50的TP在IoU70当作TP使用。

## RQ5：解释层次与尚缺证据

1. **原始模型输出事实。** 全部来源为R5A/B/C已独立验收的真实结果，8个重复原始组合精确重放，冻结输入/GT/anchor/坐标与Hash对齐，原始几何/Score的联合结果没有改写。
2. **离线混合反事实结果。** 在同一固定候选ID空间，替换完整几何或完整Score会得到表列结果；仅交换7028/7030两分数能恢复4071@50。全候选流水线保留其他候选竞争，没有手工设置keep或局部拼凑GT结果。
3. **支持的后处理敏感性。** GT14014@70和29118@70在所选两Score来源下随几何来源翻转；GT4071@50随Score来源及局部两候选排序翻转；324的约0.00218 IoU差是阈值边界效应。换回E Score不能激活725 Remote几何中任何raw IoU70候选，说明R5C的过滤事实需与几何—候选ID对齐共同理解。
4. **尚未识别的模型内部机制。** 同一模型的Geometry和Score自然联合产生，混合输出可能偏离自然联合分布。本轮没有干预Attention、MetaFlow或任何网络内部变量，不能把候选层替换效应说成这些模块、AoI或真实通信网络的独立因果效应。也未对其他帧、其他Score或新时延给出泛化结论。

下一阶段备选仅提出两项、不执行：（一）在固定324帧和原始0.7阈值下，预注册定位/数值扰动幅度的稳健性检验，量化6512跨阈值结论的局部性，不能通过调阈值制造改善；（二）在同一冻结帧/slot上预注册单一上游因素干预，以6288几何和7028/7030排序为不同终点，单独申请真实forward授权，检验输出变化来源。当前结果不足以省略上游控制实验。

## 独立验收及交付复核

生产脚本与独立验收器分进程执行，验收器不导入生产脚本。从实际原始NPZ重新读取全候选，用独立active-mask NMS与独立Score贪心GT循环，复核全部19条输入/阶段/输出数组、来源替换、仅两Score变化、原始基线、候选与GT主键、两阈值缺失值、所有54条影响比较、8条计数交互及42条GT分解。共40,927项通过，FAIL=0，必需UNRESOLVED=0。另有全12800候选方向/3D角点/投影验收，不用近似值替换实际几何。额外独立R5C导出证据门禁在混合前通过16,157项核查；生产后再次核对lineage、全NMS suppressor边、GT逐步匹配顺序和8条原始组合，详见R5D_R5C_EXPORT_PRE_HYBRID_VALIDATION.json与R5D_R5C_EXPORT_POST_PRODUCER_VALIDATION.json。所有新输出仅写入本R5D目录。

必需CSV均已交付，另有全GT、全run、逐GT交互表，完整原始baseline验收与输入对齐、几何包验收、只读历史封存。运行日志保存在logs/。SHA256SUMS.txt记录包内文件Hash（不含自身、ZIP与外部ZIP校验回执，避免循环Hash）。

上传包包含6份完整原始候选层来源（evidence/source/，保留全部12,800角点、Score、reg/dir/cls与原始阶段/GT结果），以及19份实际使用的完整输入角点/Score和全部后处理数组（evidence/runs/）；对关键结果没有R5C式低Score角点省略。validate_upload_cpu.py可在解压后的独立目录只用NumPy/Shapely重放全部19条。不打包与本轮无关的原始attention/feature大数组、冻结输入batch及checkpoint；验证整个原始NPZ文件身份及历史原文件不变仍需服务器原路径，不能将subset文件Hash冒充原NPZ Hash。六份精确来源如下，全部32条定位另见historical_reference/R5B_RAW_SHARD_LOCATOR.csv：

'''
 body+=mt(['sample','condition','原始完整NPZ绝对路径','原始完整NPZ SHA256'],[[sid,c,LOC[sid,c]['absolute_path'],LOC[sid,c]['sha256']] for sid,c in source_keys()])
 body+=f'''

服务器完整独立复核（已实际运行）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /workspace/venvs/lrcp/bin/python -B {ROOT}/validate_r5d_independent.py
```

压缩包解压后在其目录运行 `python -B validate_upload_cpu.py`，无需原始模型、GPU或原始attention张量，仅输出CPU重放验收至stdout。此便携脚本不验证未打包的完整原始NPZ字节及checkpoint身份；完整溯源由服务器独立验收与上述原始SHA完成。

本轮工作完成后停止，不启动任何新forward或后续阶段。
'''
 (ROOT/'RD3_R5D_SCIENTIFIC_ASSESSMENT.md').write_text(body)
 write('R5D_SOURCE_AND_EXECUTION_SEAL.json',{'state':'R5D_PASS','created_client_time':datetime.now(timezone(timedelta(hours=8))).isoformat(),'client_timezone':'Asia/Shanghai','study_design_sha256':sha(ROOT/'R5D_DESIGN_FROZEN.json'),'scripts':{p.name:sha(p) for p in ROOT.glob('*.py')},'scientific_result_files':{n:sha(ROOT/n) for n in ['R5D_FACTORIAL_RESULTS.csv','R5D_FOCAL_GT_OUTCOMES.csv','R5D_NMS_SCORE_SWAP.csv','R5D_GEOMETRY_SCORE_INTERACTION.csv','R5D_OFF_TARGET_EFFECTS.csv']},'python':platform.python_version(),'numpy':np.__version__,'shapely':shapely.__version__,'factorial_runs':16,'score_swap_runs':3,'real_forward_increment':0,'CUDA_initializations':0,'GPU_PPU_calls':0,'history_modifications':0,'inference_budget_attempt_increment':0,'study_complete_no_next_stage_started':True})
 write('R5D_RESULT.json',{'state':'R5D_PASS','independent_checks_passed':v['passed_check_count'],'FAIL':[],'UNRESOLVED_required':[],'factorial_runs':16,'score_swap_runs':3,'focal_GT14014_70_bits_00_01_10_11':{p:'1100' for p in ['P1','P2','P3']},'focal_GT4071_50_bits':{p:'1010' for p in ['P1','P2','P3']},'focal_GT29118_70_bits':'0011','all_TP_FP_FN_and_final_count_I_GS_zero':all(int(r['I_GS_'+n])==0 for r in eff for n in ['TP','FP','FN','final_prediction_count']),'all42_GT_I_zero':all(int(r['I_GT'])==0 for r in pg),'non_focal_GT_detection_status_changes':sum(len(json.loads(r['other_GT_status_changes'])) for r in off),'new_forward':0,'CUDA_initializations':0,'GPU_PPU_calls':0,'history_modifications':0,'upstream_module_causal_effects_identified':False})
 (ROOT/'README_CPU_REPLAY.md').write_text(f'''# R5D复核入口

完整结论：RD3_R5D_SCIENTIFIC_ASSESSMENT.md；状态：R5D_INDEPENDENT_VALIDATION.json。

16个冻结2×2组合和3个仅两Score交换均是CPU后处理，不是模型forward。几何和Score可能来自不同原始条件，不属于自然模型联合输出；不能解释为上游模块独立因果。

服务器完整独立验证：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /workspace/venvs/lrcp/bin/python -B {ROOT}/validate_r5d_independent.py
```

解压包独立复核：在解压目录运行 `python -B validate_upload_cpu.py`，需要NumPy与Shapely。stdout输出验收，全部12,800角点及Score均随包保存；不调用torch/opencood。原始大NPZ文件身份、原始输入与checkpoint完整性仍需服务器验收，精确路径及SHA在PAIR_MANIFEST、原始locator和报告中。

G0/G1与S0/S1以PAIR_MANIFEST定义。Focal表FN的matched字段为空，不能填0。固定anchor/grid仅是候选空间计算对应，不是FP物理目标身份。所有阈值、原生NMS与GT匹配均冻结。
''')
 print('R5D_REPORT_COMPLETE',v['state'],len(results),'factorial rows',flush=True)
if __name__=='__main__':main()
