from c2_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo
import difflib

def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def run():
 assert json.loads((R/'validation.json').read_text())['passed']
 m=json.loads((R/'factorial_metrics.json').read_text());v=json.loads((R/'validation.json').read_text());b=json.loads((R/'baseline_validation.json').read_text());mix=json.loads((R/'mixed_execution.json').read_text())
 rows=list(csv.DictReader((R/'per_GT_factorial.csv').open()));effects=list(csv.DictReader((R/'per_GT_effects.csv').open()));selections=chosen();lookup={(int(z['sample_idx']),int(z['gt_idx']),z['condition']):z for z in rows}
 figures=R/'figures';figures.mkdir(exist_ok=True)
 grid=np.array([[int(lookup[(int(z['sample_idx']),int(z['gt_idx']),c)]['TP70']) for c in CONDITIONS] for z in selections])
 fig,ax=plt.subplots(figsize=(13,10));ax.imshow(grid,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
 ax.set_xticks(range(8));ax.set_xticklabels(CONDITIONS,rotation=25,ha='right');ax.set_yticks(range(18));ax.set_yticklabels([f"{z['sample_idx']}/GT{z['gt_idx']} ({z['group'].split('_')[0]})" for z in selections]);ax.set_title('P4-C2 focal GTs: actual native rematching, all eight conditions')
 short={'score_filter':'score','nms_overlap_suppression':'NMS','no_high_iou_raw_candidate':'raw','size_or_z_filter':'quality','nms_top1000_truncation':'top1000','post_nms_range_filter':'range','high_iou_survives_final':'match'}
 for i,z in enumerate(selections):
  for j,c in enumerate(CONDITIONS):
   x=lookup[(int(z['sample_idx']),int(z['gt_idx']),c)];ax.text(j,i,('TP #'+x['matched_anchor']) if grid[i,j] else 'FN '+short[x['last_strict_disappearance']],ha='center',va='center',fontsize=7)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(figures/('focal_GT_factorial.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(9,4))
 for g,ax in enumerate(axs):
  data=np.array([[m['group_metrics']['all_GT'][f'G{g}A{a}R{r}']['TP70'] for r in [0,1]] for a in [0,1]])
  ax.imshow(data,cmap='YlGnBu',vmin=130,vmax=138);ax.set_xticks([0,1]);ax.set_xticklabels(['R0','R1']);ax.set_yticks([0,1]);ax.set_yticklabels(['A0','A1']);ax.set_title(f'Fixed G{g}: TP70 / 162 GT')
  for a in [0,1]:
   for r in [0,1]:ax.text(r,a,str(data[a,r]),ha='center',va='center',fontsize=20)
 fig.suptitle('Descriptive selected-18 factorial result; not a population estimate');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(figures/('all_GT_factorial_counts.'+ext),dpi=170)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));labels=['G0/S0','G0/S1','G1/S0','G1/S1'];cnt=[m['alignment'][f'G{g}S{s}_raw_positive_overlap']['maxscore_is_maxIoU'] for g in [0,1] for s in [0,1]]
 ax.bar(labels,cnt,color=['#3274a1','#e1812c','#3274a1','#e1812c']);ax.set_ylabel('GTs / 162');ax.set_ylim(0,40);ax.set_title('Within positive-overlap anchors: highest score also highest IoU')
 for i,n in enumerate(cnt):ax.text(i,n+.5,str(n),ha='center')
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(figures/('score_geometry_alignment.'+ext),dpi=170)
 plt.close(fig)
 out=[]
 out.append('# LRCP P4-C2：Score准入与NMS排序分离实验及P4-Full决策\n')
 out.append('**P4-C2已完成，独立CPU验收通过。** 四个同源条件共72组逐位复现C1后，执行四个混合条件72组。新增模型forward=0，未初始化CUDA/启动PPU。原Pilot FAILED及exit1保持不变；本轮没有实施Full预处理、全量采集或内部干预。\n')
 out.append('新目录：`'+str(R)+'`。完成时间：'+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()+'。\n')
 out.append('## 协议与首要基线\n')
 out.append('G固定原reg/dir/anchor几何。A的全12800分类分数经严格>0.2决定准入，并作为最终置信度与eval全局排序；R只决定top1000与原polygon NMS排序（阈值严格>0.15）。通过原post_process的临时内存NMS入口适配实施，finally恢复函数引用，不写模型/数据集源码。混合条件不得解释为原生LRCP推理。每条件真实调用原NMS，并以独立shadow keep逐项核对；最终重新调用原eval贪心匹配。\n')
 out.append('72组同源基线的所有共享数组、阶段mask/ordered anchor IDs、NMS keep、最终框/score及GT匹配与C1逐位一致（实际误差0，严于既有1e-5m/1e-6上限）。其通过后才执行混合条件。所有144组质量后候选最多71，top1000截断0组，局部R排序无同分；未改sort稳定性。R值即使≤0.2也不能再次过滤：只要A通过即可参与排序。例如941/GT9的G1A1R0保留#2295，其A=.241111、R=.016480，这是预设人工反事实，而不是更改score阈值。\n')
 out.append('## A：八条件结果，覆盖全部GT\n')
 hdr=['条件','全部162GT TP70/FN70','FP70','18焦点TP','6损失TP','6恢复TP','6控制TP','描述AP70']
 t=[]
 for c in CONDITIONS:
  z=m['metrics'][c]['70'];t.append([c,f"{z['TP']}/{z['FN']}",z['FP'],*[m['group_metrics'][grp][c]['TP70'] for grp in ['all_focal','loss','gain','control']],f"{z['AP_descriptive_selected18']:.10f}"])
 out.append(table(hdr,t)+'\n')
 out.append('![全部162GT因子表](figures/all_GT_factorial_counts.png)\n')
 out.append('这些AP与匹配数量来自历史结局分层选出的18样本，不能当910无偏总体估计，不能作为训练后模型改进。G1A1R0的136TP高于G1A1R1的134TP只是诊断结果，不是可部署性能结论。\n')
 out.append('### 全部18焦点，TP及实际保留anchor\n')
 focal=[]
 for z in selections:
  sid,gi=int(z['sample_idx']),int(z['gt_idx']);items=[]
  for c in CONDITIONS:
   x=lookup[(sid,gi,c)];items.append('TP#'+x['matched_anchor'] if int(x['TP70']) else 'FN('+short[x['last_strict_disappearance']]+')')
  focal.append([f'{sid}/GT{gi}',z['group'].split('_')[0],*items])
 out.append(table(['sample/GT','组',*CONDITIONS],focal)+'\n![全部18焦点](figures/focal_GT_factorial.png)\n')
 out.append('## B：单独改变A、单独改变R及交互\n')
 out.append('y为每GT实际TP70，I=y(A1,R1)−y(A1,R0)−y(A0,R1)+y(A0,R0)。固定G0有2/162个GT、固定G1有3/162个GT非零，共5个(G,GT)观察、4个唯一GT。所有非零阶段数量交互首次出现在NMS，而非raw/score/quality/top1000。**存在明确个例非加性，但不能声称普遍或统计显著；正I也可能是两种损失叠加的二元下限，不等于协同增益。**\n')
 inter=[z for z in effects if int(z['interaction_TP'])]
 out.append(table(['sample/GT','G','A0R0','A0R1','A1R0','A1R1','I'],[[f"{z['sample_idx']}/GT{z['gt_idx']}",z['G'],*[z['TP_A'+a+'R'+r] for a,r in [('0','0'),('0','1'),('1','0'),('1','1')]],z['interaction_TP']] for z in inter])+'\n')
 out.append('- **683/GT7，G0**：只有A1R1为TP。#9593 IoU=.748468，S0=.083707、S1=.460556；A0直接删除，A1R0又被#9433（IoU=.634642；R0=.320192）抑制，A1R1才保留。准入与排序均必要。\n- **360/GT5，G1**：A0R0为FN，另三组为TP。#6524 IoU=.792284已经通过A0，但被#6522（IoU=.560342）抑制；A1把#6522分数从.602266变为.033353而删除竞争者，或R1直接反转排序，均可恢复。不是“改变A只作用于焦点候选本身”。\n- **683/GT7，G1**：类似替代路径；A1删除竞争者#9273（IoU=.655218、S0=.419156→S1=.077194），或R1优先#9433（IoU=.720607）均可恢复。\n- **941/GT9，G1**：仅A1R0恢复。#2295 IoU=.900669与#2455 .845246在A0均低分被删；A1让它们及低IoU竞争者#2775同时入场。R1下#2775（IoU=.531803，score .349474）压制两个严格候选，R0下#2295优先。\n- **918/GT6，G0**：A0R0 TP，任一改变均FN；A1删尽严格候选，A0R1则#925（IoU=.715276）被#1245（.415984，R1=.767974）抑制。I=+1体现损失不可重复扣两次，并非正向协同。\n')
 out.append('### 全部单因子迁移，包括新增退化\n')
 for x in m['single_factor_transitions']:
  targets='；'.join(f"{z['sample_idx']}/GT{z['gt_idx']} {'FN→TP' if z['to_TP'] else 'TP→FN'} ({short[z['from_stage']]}→{short[z['to_stage']]})" for z in x['targets']) or '无'
  out.append(f"- {x['from']}→{x['to']}：恢复{x['gains']}、退化{x['losses']}；{targets}。\n")
 out.append('## 六个原α1损失：A与R分开后能恢复多少？\n')
 loss=[z for z in selections if z['group'].startswith('loss_')];lt=[]
 for z in loss:
  sid,gi=int(z['sample_idx']),int(z['gt_idx']);rs=[lookup[(sid,gi,f'G1A{a}R{r}')] for a in [0,1] for r in [0,1]]
  lt.append([f'{sid}/GT{gi}',int(rs[0]['n70_raw_decoded']),*[('TP#'+q['matched_anchor'] if int(q['TP70']) else 'FN '+short[q['last_strict_disappearance']]) for q in rs],'/'.join(q['n70_score_pass'] for q in rs)])
 out.append(table(['sample/GT','G1 raw≥.7','A0R0','A0R1','A1R0','A1R1','score后严格数量(同列序)'],lt)+'\n')
 out.append('821/GT2与941/GT8只需R0，在A0/A1都恢复：score后严格候选集分别3个/2个相同，恢复框#10433/#7166与C1一致。**单改准入A1→A0、保留R1，六例恢复0个；单改排序R1→R0、保留A1，恢复2个。** 17/GT5与918/GT6在两种A下均score删尽；486/GT2两种A/R均NMS删尽；670/GT3仅A0让#6028（.734370，S0=.314553/S1=.158407）入场，但两种R都被#6186（IoU=.672138）抑制。其raw严格候选均存在，不能把未恢复称raw缺候选。G0下360/GT5则四条件都没有raw严格候选，分数不能创造几何。\n')
 out.append('## C：分数与定位几何一致性，明确分母\n')
 out.append('主统计单位为唯一(sample,GT,G,S)，候选集为对该GT BEV IoU>0的全部raw anchors（非按结果挑选），以及其quality后、S>.2准入后子集。最高score按原argsort()[::-1]，是否同达最高IoU使用1e-6绝对容差；并列数另存，没有更改真实NMS。不同GT可共享候选，GT-anchor事件重复按关联计，不能当独立物体。全12800含零重叠的敏感性另存，但其全场最高分往往是别的目标，不作主结论。\n')
 al=[]
 for g in [0,1]:
  for s in [0,1]:
   z=m['alignment'][f'G{g}S{s}_raw_positive_overlap'];ad=m['alignment'][f'G{g}S{s}_admitted_positive_overlap'];al.append([f'G{g}/S{s}',f"{z['maxscore_is_maxIoU']}/{z['nonempty_GT']}",z['highest_score_below70_despite_strict_available'],f"{z['mean_IoU_regret']:.6f}",f"{ad['maxscore_is_maxIoU']}/{ad['nonempty_GT']}"])
 out.append(table(['固定G/score','raw重叠集最高Score亦最高IoU','raw有严格框但最高分IoU<.7的GT','均值IoU regret','准入后非空集同达最大IoU'],al)+'\n![分数与几何一致性](figures/score_geometry_alignment.png)\n')
 out.append('原生G0A0R0中6个GT出现低IoU保留框压制严格候选，共10个GT-anchor事件；原生G1A1R1为7个GT、14事件。两组这些抑制者的A分数都更高。并不意味着这些GT全是FN：有的还存在其他严格保留框。完整8条件低IoU抑制者计数及A/R分数见factorial_metrics.json和per_GT_candidate_lineage.csv。最高score≠最高IoU本身不证明分类头错误，分类目标也不等同精确定位目标；真实NMS事件与阈值附近几何变化才给出更直接的机制证据。\n')
 out.append('## D：证据边界\n')
 out.append('结论支持保存head层面的准入/竞争集合与排序耦合，反驳“只换A或只换R在所有GT上同向改善”和“全量S0移植可恢复全部α1损失”。不能将人为混合分解成MetaFlow内部分类/回归的独立因果贡献，不证明offset预测错，不保证训练后可实现这些混合输出。18样本按历史结局分层，不能外推278病例的阶段占比、910恢复率/AP或真实异步系统/MVMOT指标；相邻GT共享场景/候选，非独立统计样本。\n')
 out.append('## E：优先级与P4-Full独立准备\n')
 out.append('如果下一目标仍是定位910中278损失的阶段分布，建议优先**P4-Full最小候选追踪**，而非立刻给offset错误定因。当前18例有选择偏差，不能支持决定全局改哪一模块；Full只增加raw heads/lineage与验证，不扩大到k0..5或全特征。随后按raw缺失/score/NMS各层、恢复/稳定对照选择更小内部MetaFlow干预。若研究问题直接改为内部offset因果作用，则小干预才是必要证据，Full阶段统计不能替代。预算不足可先238个loss sample定向追踪，但须明确不等同Full。\n')
 out.append('278个loss GT在238sample；P3-A最终框/score/GT可复用，raw heads/低分anchors及lineage需新增。全量核心1820前向；复用18pair则新增1784核心，完整821 native/instrumented/reverse六次新检查后共1790；完全新采则核心1820+额外检查4=1824。Pilot设备采集阶段40次42.378秒只是部分流水线基准，Full外推约31.6分钟，不是整个作业耗时。冻结910batch均值约22.85GiB磁盘，需另批缓存上限，CPU仍逐sample且RSS8GiB。详见P4_FULL_PLAN.md、P4_FULL_RESOURCE_BUDGET.md。\n')
 out.append('本轮只规划，未生成Full PPU READY、未更新READY_CURRENT、未执行910预处理。下一步须用户明确批准范围、1790或1824预算、缓存/时间上限；新隔离runner完成CPU READY后，用户亲自在终端启动PPU。PPU采集状态与CPU独立验收分开，绝不因CPU脚本错误自动重采。\n')
 out.append('## F：真实异步通信与MVMOT的机制线索\n')
 out.append('可检验线索：时间补偿可能共同改变候选定位、分数准入及NMS竞争；检测漏失或保留低IoU框可能进一步影响后续关联/轨迹，但本实验没有轨迹、真实异步通信、身份或跟踪评价。后续需独立记录每CAV真实时间戳、发送/到达/融合时间、时钟误差、延迟分布/抖动/丢包/乱序，以及运动、遮挡和外参；使用冻结或受控通信轨迹，避免把k3的约600ms标签当完整异步网络模型。MVMOT需要有跨帧真实ID的连续序列，独立验证检测变化→关联/ID切换/轨迹连续性路径，比较同一检测输入下的关联与同一关联下的检测变体；anchor ID不是真实ID，不能从18个离散样本推断跟踪改善。\n')
 out.append('## 验收、交付与复现\n')
 out.append(f"独立验收{v['case_count']}条件、{v['GT_rows_verified']}GT行、{v['all_anchor_rows_verified']}全anchor行、{v['strict_candidate_lineage_verified']}严格候选链、{v['nms_priority_rows_verified']}排序变化行、{v['alignment_rows_verified']}一致性行。分别重新native NMS与GT匹配、重算所有阶段/因子差分及CSV读回；历史{v['history_files_unchanged']}文件及{v['protected_files_unchanged']}保护文件hash不变。原worker FAILED/exit1保留。第一次验收收尾读取旧状态时误用worker_phase字段，已改为真实state，首轮错误记录保留，完整CPU验收重新通过；没有重跑模型或反事实实验。\n")
 out.append('per_GT_factorial.csv含每阶段best IoU/严格数量/全部anchor IDs和TP/FN；per_GT_effects.csv含固定G四条件和所有A/R条件效应及交互；per_GT_candidate_lineage.csv含所有raw严格候选S0/S1/A/R分数、首删因、抑制者/GT IoU/分数/overlap；all_anchor_factorial.csv覆盖全部12800 anchors的四A/R删除状态，包括原本低分者；nms_priority_changes.csv保留固定A下两R的keep及suppressor变化（含未变）；score_geometry_alignment.csv逐GT明确集合/分母；cases/保存144组完整CPU数组和原生匹配。没有只保留成功案例。\n')
 out.append('复现时复制本目录*.py、inputs.json、两个选择CSV、PROTOCOL.md、两份P4_FULL规划MD及run_cpu.sh到全新空目录，运行其CPU入口。脚本拒绝覆盖基线/实验结果、隐藏CUDA，依次基线→混合→分析→独立验收→只读规划→报告；不在当前目录重复执行。源码/历史来源hash锁定在inputs.json及P4_FULL_PLAN_EVIDENCE.json。\n')
 out.append(f"资源：同源CPU {b['elapsed_seconds']:.3f}s，混合{mix['elapsed_seconds']:.3f}s，独立验收{v['elapsed_seconds']:.3f}s；三者最高RSS={max(b['peak_RSS_bytes'],mix['peak_RSS_bytes'],v['peak_RSS_bytes'])/2**30:.3f}GiB。全程未依赖大tensor共享内存，未改变阈值、样本或原始数据。\n")
 (R/'REPORT.md').write_text('\n'.join(out))
 # Static audit is separate from actual runtime validation.
 static=[]
 for f in R.glob('*.py'):
  text=f.read_text();tree=ast.parse(text);compile(tree,str(f),'exec');static.append(f.name)
 for field in ['history_hashes','protected_hashes']:
  for path,h in INPUT[field].items():assert sha_file(path)==h,path
 (R/'IMPLEMENTATION_DIFF.diff').write_text(''.join(difflib.unified_diff((R/'c1_reused.py').read_text().splitlines(True),(R/'c2_common.py').read_text().splitlines(True),fromfile='C1_reused_readonly_copy.py',tofile='C2_factorial_adapter.py')))
 save(R/'STATIC_CHECKS.json',{'passed':True,'compiled_files':static,'model_created':False,'CUDA_initialized':torch.cuda.is_initialized(),'model_forwards':0,'NMS_adapter_finally_restores_reference':True,'same_source_baseline_bitwise_cases':72,'mixed_native_NMS_cases':72})
 save(R/'COMPLETION.json',{'phase':'P4-C2','completed':True,'independent_validation_passed':True,'sample_count':18,'GT_count':162,'CPU_postprocess_conditions':144,'model_forwards':0,'CUDA_initialized':False,'full_plan_only':True,'P4_FULL_READY_published':False,'original_worker_state_preserved':'FAILED','original_exit_code_preserved':1,'run_directory':str(R),'peak_RSS_report_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
 guard();log('P4_C2_REPORT_COMPLETE',model_forwards=0)
 print('P4_C2_COMPLETED_CPU_ONLY')
if __name__=='__main__':run()
