"""Render P4 precheck reports from frozen CPU selection and source evidence. No model import."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
S=Path('/workspace/LRCP')
D=Path('/workspace/lrcp_d2')
sel=json.loads((R/'SELECTION_SUMMARY.json').read_text())
validation=json.loads((R/'INPUT_VALIDATION.json').read_text())
rows=list(csv.DictReader((R/'SAMPLE_SELECTION.csv').open()))
def link(relative,line):
 p=S/relative
 return f'[{relative}:{line}]({p}:{line})'
def put(name,text):
 with (R/name).open('x') as f:f.write(text.strip()+'\n')
P='opencood/models/point_pillar_deformable.py'
M='opencood/models/meta_flow.py'
BB='opencood/models/sub_modules/defor_bev_backbone_resnet.py'
E='opencood/models/sub_modules/defor_encoder_multi_scale.py'
V='opencood/data_utils/post_processor/voxel_postprocessor.py'
U='opencood/utils/box_utils.py'
DS='opencood/data_utils/datasets/intermediate_latency_fusion_dataset.py'
BASE='opencood/data_utils/datasets/basedataset/v2xsim_basedataset.py'
FLOW='data_preprocess_tools/flow_label_gen_v2x-smi.py'
CFG='opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/config.yaml'
put('SOURCE_AUDIT.md',f'''
# P4-Precheck：真实执行路径与源码证据

**源码审计和 CPU 样本选择已完成；PPU 先导实验未运行，机制尚未定位。** 当前可以可靠区分和记录多个候选删除阶段，不能从历史 post-NMS 框判断损失首次发生在哪里。

## 输入与审计边界

本目录全部工作仅 CPU / 标准库与 NumPy，未导入 torch、实例化模型、加载 checkpoint 到设备或运行数据集预处理。P3-B 最终清单 43 个文件、P3-C1 清单 27 个文件通过哈希校验，两阶段 input_hashes 中的实际输入均匹配；另按 P3-A audit_results.json 核验其记录的相关脚本、配置、checkpoint、JSON。总计 {validation['input_files_hashed']} 个已验证输入路径。源码和历史产物前后哈希见 PROTECTED_INPUT_HASHES_BEFORE.json、DIFF_AUDIT.json。

检查 /AGENTS.md、/workspace/AGENTS.md、LRCP 和 lrcp_d2 及 codex_reports 的 AGENTS.md，并搜索两个项目的下级 AGENTS.md，均未找到已安装规则；AGENTS.md.draft 未当作生效规则。阅读了 P3-A 审计、P3-B REPORT_FINAL.md、P3-C1 REPORT.md 及其实现和历史采集脚本。没有重复 AP/IoU 全量计算和 P3-A 目录审计。

冻结 checkpoint SHA256：666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724。k=3、内部时间编码 300、约 600 ms 的物理间隔沿用已验收实验定义。当前模型/数据集源码纳入本次哈希锁；P3-A 旧清单未锁定每个模型内部文件，不能单凭本次读取证明所有内部源码与采集时逐字相同。git HEAD/status 保存在 WORKTREE_BEFORE.json；已有 untracked formal_run_logs/ 原样保留。

**以下形状是当前配置和源码静态推导，不是本次 PPU 实测。** 先导运行必须逐项断言并记录实际 dtype/shape。B=1，N 为通信范围内实际 CAV 数，N≤5；T=5 历史帧；M 为非空 voxel 数。所有源码引用均为本次快照的绝对路径与起始行号；SOURCE_EVIDENCE.txt、SOURCE_SYMBOL_INDEX.csv 保存原文及 AST 定义区间。

## 真实调用链

```mermaid
flowchart TD
    A[同一处理后 ego batch] --> B[PointPillarDeformable.forward]
    B --> C[每有效远端槽：5帧 VFE / Scatter / ResNet / pose warp]
    C --> D[MetaFlow：历史特征 + SineTimeEncoding 300]
    D --> E[raw offset → alpha scaling → padded pred_offset]
    B --> F[当前及延迟点云 VFE / Scatter / ResNet]
    F --> G[DeforEncoderMultiScale]
    E --> G
    G --> H[融合特征 → cls_head / reg_head / dir_head]
    H --> I[VoxelPostprocessor：解码 → score → direction → corners / projection]
    I --> J[size / z masks → top1000 → overlap NMS → range]
    J --> K[最终框和分数 → eval_utils]
```

实际入口是 [p3a_capture_k3.py](/workspace/lrcp_d2/p3a_capture_k3.py:122)：create_model + load_saved_model，model.eval；135–141 的 forward hook 返回 MetaFlow 输出乘 α；189–206 同一已处理 batch 做多个 α。{link('opencood/tools/train_utils.py',113)} 按 core_method 动态实例化 PointPillarDeformable。{link('opencood/tools/inference_utils.py',156)} 的 intermediate_fusion 实际转调 inference_early_fusion；后者 140–146 只对 batch['ego'] 调模型，再调 dataset.post_process。不是对每个 CAV 分别生成独立检测结果后做 late fusion。

配置分支由 {link(CFG,230)} resnet=true、multi_scale=true，{link(CFG,239)} calibrate=true、stage2、head_embed_dims=128，{link(CFG,351)} core_method=point_pillar_deformable 确定。非 ResNet 的 DeforBEVBackbone 存在但本次未走；不能用其实现代替当前链。

## A：MetaFlow 生成、坐标和尺度

| 节点/函数 | 代码证据 | 输入 / 输出形状与定义 |
|---|---|---|
| 测试历史选择 frame_select | {link(BASE,272)} | k=3 时历史索引 [3,4,5,6,7]；远端主输入为第3帧历史，ego 保持当前帧；缺历史返回 None |
| time / GT offset 装载 | {link(BASE,222)} | time_index*100=300；offset_map[ego_id-1,cav_id-1,2,:,:,:] → (1,80,80,2) |
| 远端槽组装 collate_batch_train | {link(DS,495)}、{link(DS,638)} | 槽1..4按 cav_id_list 对齐，不等于物理 CAV ID；offset (B,80,80,2)、time_delay (B,)、valid_mask (B,)、5帧 voxel 字典列表；无效槽填0 |
| get_batch_pillar_features | {link(P,59)} | 每个有效远端的历史 (5,64,160,160) scatter；ResNet层为 (5,64,80,80)、(5,128,40,40)、(5,256,20,20)，pose warp 后堆叠成 (Bv,5,C_l,H_l,W_l) |
| CNNBackbone.forward | {link(M,96)} | 3层分别反卷积到 (Bv*5,128,80,80)，concat 384通道、1×1 shrink 到128，reshape (Bv,5,128,80,80) |
| MetaFlow.forward | {link(M,167)} | 5帧 value (Bv,32000,128)，query / time / pos (Bv,6400,128)，reference (Bv,6400,5,2)；blocks 后 Linear(128,2) 输出 (Bv,80,80,2) |
| α干预和槽恢复 | {link(P,108)}；历史脚本135–141 | 有效远端调用 MetaFlow，hook 返回 α*raw；index_copy 回 B 维。最终 (B,6400,4,2) 包含4个槽，无效槽0；output_dict['pred_offset']是乘α后但尚未 /80 的值 |

Pillar 输入/输出证据：{link('opencood/models/sub_modules/pillar_vfe.py',105)}，voxel_features (M,32,4)、coords (M,4) b/z/y/x、num_points (M,) → pillar_features (M,64)；{link('opencood/models/sub_modules/point_pillar_scatter.py',15)} → (sum N,64,160,160)。80/40/20 分辨率来自 {link('opencood/models/sub_modules/resblock.py',162)} 的 [2,2,2] stride，不采用注释中的旧尺寸。

MetaFlow.classifier 虽然叫 classifier，实际是两通道连续 offset 回归，没有 sigmoid/tanh/clamp，也没有再乘 time_delay；time 通过 {link(M,27)} 的 sin/cos 编码加到 query。α=0 仍会计算历史 CNN 和 MetaFlow，仍使用内部时间300；仅把其输出置0，不关闭融合或注意力自身的 learned sampling_offsets。

坐标证据：{link(FLOW,80)} 将 ego 坐标 x/y 映射到 80×80 网格；{link(FLOW,169)} 对 boxes[0]当前 → boxes[i]历史做变换，{link(FLOW,222)} 把 transformed_points - points 写入 map[y,x,:]。{link(FLOW,240)} 将各时刻框投影到当前 ego 坐标后生成图。因此这是一种**当前输出网格点 → 历史特征采样位置的反向采样位移**，不是“把历史框向前平移”的世界坐标速度向量。

通道0为 x/W，通道1为 y/H；1格名义为64/80=0.8m，offset 本身单位是80网格单元。注意 anchor 中心由 linspace(-31.6,31.6,80) 生成，其步距不是精确0.8；不能用 anchor index 直接反推 MetaFlow 像素世界坐标，应使用对应的特征网格定义。GT map 的生成脚本提供单位/方向证据，不提供每个已有 npy 的生成版本证明。没有独立有效流 mask；背景零、静止零、没有覆盖的零混在一起，不能把全图零值直接视为可靠真实运动标签来计算 offset EPE。

## B：offset 怎样进入融合

{link(BB,141)} 当前主输入的 ResNet 特征经 input_proj 变成三层 (N,128,80,80)/(N,128,40,40)/(N,128,20,20)，163–166 先单车分支再 cooperative encoder。单车分支不使用 pred_offset，仍是可采集的负控制特征。

{link(E,259)} DeforEncoderMultiScale.forward 第280/281行先将 pred_offsets 的 x/W、y/H 分别除80。各层 pose warp 到ego，flatten并加 level/agent embedding，value 为 (N,8400,128)。基础reference为 (1,6400,1,2)，远端补偿后 reference 为 (1,6400,3N,2)：ego三层保持基础点，远端每槽三层都加同一个归一化 offset。**这里没有先用 MetaFlow 对整幅 BEV 做一次独立 grid_sample warp**；它移动的是后续 deformable attention 的参考采样点。已有的 warp_affine_simple 是 pose 对齐，与该补偿有别。

{link(E,54)} DeforAttn.forward 仍从 query 预测自己的 learned sampling_offsets，公式为：

`sampling_location = reference(base + alpha*MetaFlow/[80,80]) + learned_sampling_offsets/[W_level,H_level]`

cross attention 的 location (1,6400,8,3N,8,2)、weight (1,6400,8,3N,8)，kernel 输入 value (1,N*8400,8,16)。两个 offset 不是同一个参数或同一个归一化尺度。{link(E,178)} 的 self attention 用 ref_2d[:,:,:1,:] 即ego基础参考，不直接用远端 MetaFlow，但输入query可已受先前cross attention影响。当前配置4个cross block + 2个self block；最后输出 (B,128,80,80)。

{link('opencood/models/sub_modules/torch_transformation_utils.py',322)} pose warp 使用 affine_grid/grid_sample、align_corners=False。自定义采样 kernel {link('opencood/utils/ms_deform_attn_ops/src/cuda/ms_deform_im2col_cuda.cuh',281)} 按 loc*size-0.5 插值并检查边界；完全越界位置不累加有效值，未发现对 MetaFlow / reference 预先clamp。可记录 normalized out-of-range比例作为诊断，但 [0,1] 外不等于 kernel 支持区外：必须同时计算 kernel 的 -1<h_im<H、-1<w_im<W 边界，不能把两者混称“零采样率”。本次未测量这些比例。

## C、D：融合到检测头

{link(P,146)} backbone返回 fused_features (1,128,80,80)；152–156 的1×1 cls_head、reg_head分别输出 cls_preds (1,2,80,80)、reg_preds (1,14,80,80)。两个分类通道是两个朝向anchor，各自一个目标logit，不是两个语义类别。dir_head 在 {link(P,181)} 输出 (1,4,80,80)，每anchor两个方向bin。

supervise_single=true、use_seperate_head=true 时另有 single_* 输出，来源single_features (N,128,80,80)；当前 dataset.post_process 只用不带_single的 fused输出。**没有单独的 IoU预测头 / iou_preds 输出**；VoxelPostprocessor中的IoU分数重加权分支存在，但本模型未触发。MetaFlow offset 不直接加到检测框中心/回归输出；对检测的影响经融合特征再到head。

## E：后处理真实顺序（不能省略的阶段）

{link(DS,685)} 调 VoxelPostprocessor.post_process，再 generate_gt_bbx。{link(V,243)} 按如下顺序执行：

| 阶段 | 行号 | 形状 / 精确规则 |
|---|---|---|
| sigmoid / flatten | 292–294 | logits permute NHWA → prob (1,12800)，保留float32；无softmax |
| 全anchor框解码 | 299–303；{link(V,405)} | reg permute后 (1,12800,7)，anchor (80,80,2,7)，输出xyzhwlr；xy×anchor平面对角线，z×anchor h，hwl exp(delta)，yaw+anchor |
| score过滤 | 305–314 | prob **>0.2**；12800个候选 → Ks个；原生是先解码，再按score取框 |
| 方向纠正 | 317–333 | dir logits reshape (1,12800,2)，用同一score mask，修正yaw；num_bins=2，dir_offset=0.7853 |
| 角点与ego投影 | 342–361 | (Ks,7) → (Ks,8,3)，转换standup仅用来组织score；测试变换矩阵是identity ({link(DS,664)}) |
| 尺寸及z过滤 | 372–377 | 两个独立bool mask AND，Ks→Kq，保留原索引顺序 |
| NMS内部前1000 | {link(U,719)} | scores.argsort()[::-1][:1000]；超过1000的其余框是**top截断**，不叫overlap抑制 |
| rotated polygon NMS | {link(U,724)} | 被已选框压制的条件为 BEV IoU **>0.15**；返回相对Kq索引，分数降序 |
| NMS后range过滤 | 391–398；{link(U,384)} | x/y∈[-32,32]、z∈[-3,2]，默认8个角点都在范围内；不是中心点检查 |
| 最终输出 | 400–402 | (Kfinal,8,3)、(Kfinal,)；可为空/None |

尺寸/z实现需要原样记录，不能在诊断时静默修正：{link(U,840)} remove_large_pred_bbx 用x/y轴对齐包围跨度≤6m；其 z_len **实际再次读取y通道**，且 logical_and(index,z_len) 把该跨度转为bool，并非检查真正z长度。{link(U,872)} remove_bbx_abnormal_z 则要求所有角点 z_min≥-3、z_max≤1。这个上界1比最终range上界2严格。它们是当前后处理的真实行为，不是本次发现的实测损失来源；本阶段不修bug。须分开保存两个mask和AND结果。

配置的 postprocess.max_num=100 用于GT padding等；在这个 post_process 路径未找到“最多100个预测框”的截断。真实预测top限制来自 nms_rotated 内硬编码1000，且其默认排序与 eval_utils 的 np.argsort(-score) 写法不同，不能统一替换为stable排序。

## F：进入 eval_utils 的框已经筛选和NMS

{link('opencood/tools/inference_utils.py',140)} → dataset.post_process返回 → [p3a_capture_k3.py:198](/workspace/lrcp_d2/p3a_capture_k3.py:198) 保存NPZ → 206–214 调 {link('opencood/utils/eval_utils.py',41)}。因此历史1820份NPZ与AP计算使用的是**score、尺寸/z、top1000、overlap NMS、最终range之后**的框。eval_utils 自身只按分数排序、做BEV贪心匹配，不再解码/score过滤/NMS。GT由 {link('opencood/data_utils/post_processor/base_postprocessor.py',46)} 生成、按object_id去重并做range过滤；GT索引必须以保存NPZ为准。

## 可采集性与当前缺口

可用临时 module hooks 采集：MetaFlow原输出/有效输出、fusion encoder输入pred_offsets、encoder/blocks输出摘要、fused cls/reg/dir原输出；同时记录slot→physical_cav_id、valid_mask、time_delay、pairwise matrices。heads输出不能取single_*冒充fused。

全12800解码框在原 post_process 中已存在，低分框却没有原生corner/方向纠正后的记录。需要在隔离诊断副本上对**全部anchor**执行相同方向纠正、corners、ego投影，并证明score通过子集与原生结果完全一致。不能只记录score>0.2之后的“pre-NMS”集合后声称原始阶段无高IoU框。

原生函数没有返回逐anchor删除原因、top1000名单或NMS suppressor。可在新进程对精确函数帧做受限只读trace，并对NMS输入CPU重放产生lineage，再与原keep列表逐项核对；不需永久修改源码。没有记录这些张量前，历史P3结果不能支持阶段归因。完整的待批准采集实现尚未编写或运行，diagnostic_contract_draft.py只提供hooks及CPU归因接口草案。
''')
put('MECHANISM_HYPOTHESES.md',f'''
# P4：机制假设与证伪条件

当前证据只定位到最终候选集合/排序发生变化，未定位到错误产生阶段。P3-C1 的配对分数移植不恢复AP70，不能证明MetaFlow offset错误。此先导以阶段候选可用性和逐anchor去向为主要终点，不比较总体AP，不报告总体因果占比。

| 假设 | 支持它所需的新证据 | 什么会反驳 / 限制它 |
|---|---|---|
| H1 原始候选生成/回归质量不足 | 同batch下，损失GT在α0全12800解码框存在IoU≥0.7，在α1同阶段不存在；score尚未筛选 | α1原始全候选存在高IoU框，则该GT的“完全缺候选”解释被反驳；仍可能几何质量分布变差 |
| H2 分数门限删掉严格候选 | α1 raw集合有高IoU候选，但全部score≤0.2；逐anchor score mask可重现 | 任一高IoU框通过score则不能把该GT最终缺失只归于score；单调校准可能影响过滤，却与P3-C1固定框排序诊断不同 |
| H3 尺寸/z检查放大差异 | 高IoU候选通过score，却被实际remove_large / abnormal_z mask删光 | 高IoU候选通过两mask则该阶段非最后可用性丢失点；尤其查z上界1和真实代码的y/z实现，不将可疑实现当成已证实原因 |
| H4a NMS top1000截断 | α1高IoU候选通过quality，但所有候选排名>1000 | Kq≤1000则此假设在该样本直接被排除 |
| H4b overlap NMS抑制 | 候选进入top1000，实际keep列表不含它；CPU重放给出真实较高优先级suppressor、IoU>0.15及分数 | 高IoU框仍在native NMS keep集合则不能归于overlap抑制；同分排序必须沿用native argsort |
| H5 NMS之后range删除 | NMS保留的严格候选被8角点range mask删除 | 最终range通过则此阶段不是候选消失点；范围检查是3D，评价是BEV，两者不同 |
| H6 采样位移/融合特征是上游中介 | raw MetaFlow在两α一致，effective=α*raw，进入reference/采样位置差异可复现，fusion摘要和head回归差异随之出现 | 若head完全一致则此样本不能支持该链产生候选差异；若offset原值/输入不一致则配对实验失效，不能解释机制 |
| H7 offset方向、尺度或CAV槽映射不符 | 记录slot/physical ID、current→history ego网格定义、归一化前后值及真实采样位置，对有独立可信运动标签区域做核查 | 未有有效运动mask或可靠GT流就不能把全图差异、向量大小或EPE称为offset预测错误；非零offset并不意味着错误 |
| H8 边界采样造成特征衰减 | kernel支持区越界权重/比例增加，与相关GT ROI特征和候选下降共现，控制样本提供对照 | 仅有[0,1]外比例不足以支持零采样，kernel插值支持有半像素边界；相关不证明因果 |

对全部162个GT同时保存每阶段 best IoU、IoU≥0.7候选数及其原始anchor IDs，记录每个高IoU候选的首次删除阶段。一个GT的多个高IoU框可以在不同阶段被删除，不能只挑最佳框去向就给整个GT贴单一机制标签。另报告“最后一个高IoU候选在哪步消失”，它是集合可用性的描述，不是独立因果贡献。若最终有高IoU框却是FN，另查原协议贪心竞争，不能把可用性当成TP。

样本是明确按结果分层的机制案例，不能用18个目标外推910样本或不同checkpoint。6个恢复目标和6个双TP控制必须与6个损失目标一起报告；无变化/反证案例不可剔除。原始anchor索引只是计算结构身份，不是真实物体身份，跨α匹配仍必须说明规则。

本轮只观察两个α条件，并没有分别干预回归头、分类头或NMS；可以定位高IoU可用性在哪步丢失或被放大，不能独立确定每个内部模块的因果责任。若需进一步因果验证，应根据采集结果另行提出最小干预，不自动扩展本轮范围。
''')
selection_table='\n'.join(f"| {x['sample_idx']} | {x['gt_idx']} | {x['group'].split('_')[0]} | {x['scene_id']} / {x['scene_frame_counter']} | {float(x['best_iou_a0']):.6f} → {float(x['best_iou_a1']):.6f} |" for x in rows)
put('PILOT_PROTOCOL.md',f'''
# P4 最小 PPU 先导协议（待批准，尚未运行）

## 目的、固定项与终点

固定原checkpoint、配置、k=3、time code=300、B=1、eval/no_grad，沿原 intermediateLatency / ResNet / stage2 路径。只在MetaFlow返回值乘α，α∈{{0,1}}，不关闭MetaFlow，不改变阈值、不重训练、不将内部时间改为600。物理约600ms不是写入模型的时间编码。

主要终点为每GT在全解码、score后、quality后、NMS top1000、NMS后、range后的严格IoU可用性/候选去向及head/融合摘要。保留原生最终框和原贪心匹配作核对，不把本先导作为新总体AP比较。

## 可复现选择

prepare_precheck.py读取P3-B per_gt.csv。候选总体为损失278、恢复169、双TP控制6476。每组按 abs(delta_best_iou)、sample_idx、gt_idx 排序；rank*3//组大小分成三个等秩层，每层选2行。依次repeat=0,1、third=0,1,2、group=loss,gain,control。每一步排除已用sample，及同scene frame counter差<10的候选，再最小化：(该scene已选数，该组近/远band已选数，SHA256('P4-Precheck-v1|sample_idx|gt_idx')，sample_idx，gt_idx)。近/远界20m；无可行候选即报错，不静默放宽规则。

scene/frame来自test.pkl的真实lidar_path_1中的scene_<id>_<counter>.pcd.bin，并与timestamp核对；不拿sample_idx//100猜场景。当前选择覆盖10场景，每组近/远各3个，同场景最小间隔13帧；既覆盖贴近0.7边界的小变化，也覆盖较大变化和稳定控制，不只挑严重失效例子。

| sample_idx | 历史GT索引 | 组 | 场景 / 帧计数 | 最佳IoU α0 → α1 |
|---:|---:|---|---|---|
{selection_table}

共18个不同sample、18个焦点GT，但采集后必须分析其全部162个已有GT（SELECTED_SAMPLES_ALL_GT.csv）。原始/历史点云和offset文件450份已检查存在，合计294559384字节；存在性不是内容哈希，正式采集前为实际使用文件补SHA256。

## 配对输入与历史可比性

1. 同一样本**只预处理一次**，将collate_batch_test输出的CPU batch去除派生中间量后冻结。点云shuffle仍存在，不能分别dataset[sid]两次当作配对输入。保存递归tensor/array值、形状、dtype哈希、Python/NumPy/torch RNG状态、cav_id_list、valid_mask、physical_cav_ids、record_len、anchor、poses、历史tokens和time_delay。原始RNG状态不能从P3-A NPZ恢复。
2. 为尽量恢复历史910中的既定条件，首选CPU重放原seed303及delay [0,1,2,3]的数据准备顺序，前三个delay走完整原loader，k3走到sample941；包括原collate后才过滤公共910的顺序，不能直接跳读18个样本改变NumPy RNG。只缓存所选18个batch，不运行原采集脚本（它会启动大量推理）。应在隔离进程中按已审计的数据路径实现；库初始化/RNG消耗也纳入日志。
3. CPU重放只恢复**数据准备轨迹**，不保证原PPU输出逐位一致。首先选 selection_order=1 的sample821做验证。α0/1在新batch中的输出须与其历史NPZ核对：GT角点优先逐位；若只是行序不同，必须唯一逐角点bijection后显式保存 old_gt_idx→new_gt_idx，不能按索引强行对应。数值不一致/非唯一则停止历史案例归因。
4. 历史post-NMS框必须数量、score/角点及相应GT分类复核。目标为逐位一致；仅机器数值差时允许预先固定 atol=1e-5（角点m）、score atol=1e-6、rtol=0，同时严格要求所有162个GT的TP70/FN70和best-IoU跨0.7关系一致；任何匹配/阈值跨越不一致即停止，不悄悄替换样本。若无法恢复历史，需要报告复现失败并另行批准“新预处理条件”的小实验，不能冒用历史损失组。
5. 每个冻结batch建立两个值完全相同、无共享可写tensor/array存储的深副本送同一设备。源batch永久只读逻辑，递归哈希校验。允许隔离原forward对历史dict添加pillar_features/spatial_features、对label_dict添加offset的已知行为；不可把第一次前向遗留派生状态带到另一α。记录model buffers前后哈希，eval/no_grad，无autocast、无新确定性开关改变原数值协议。
6. 一半样本先α1后α0，另一半相反（按selection_order奇偶预定）；记录并恢复每个pair前的RNG状态。α0/1的**raw MetaFlow**、上游multi-scale features应一致；effective各自等于α*raw。额外逆序重复一个样本检查顺序/缓存效应。

## 预算：18样本，最多40次完整模型forward

- 核心：18×2=36次带诊断的完整模型前向。
- sample821先做不带采集的α0/1基准，2次；其带采集α0/1已计入36次。比较raw heads与最终框，证明instrumentation等价。
- 同一sample821再逆序重复α0/1，2次，验证配对顺序。总上限40次，不包括无需模型的CPU后处理重放。
- 无单独warmup推理，不调用P1/P3整套运行脚本；某一验收失败立即停，未消耗的预算不转用其他样本。每完整forward内部最多4次MetaFlow调用，各用5历史帧；不能把40次前向误报成40次子模块调用。

## 每条件必须采集

| 数据 | 定义 / 精度 / 追溯 |
|---|---|
| 输入batch | 每样本一次冻结、完整model-relevant处理后输入，FP32/int原精度；两α输入hash一致 |
| raw/effective offset | 每有效槽 (1,80,80,2) FP32；4槽padded offset和GT offset含valid信息；原值/α乘值在归一化前记录，避免归一化in-place别名 |
| 融合输入及特征 | 3层投影/pose对齐特征摘要、single_features摘要、6个cooperative block输出摘要、最终fused摘要；均值/std/L2 map、每channel统计、确定性抽样hash，另记录FP32逐GT ROI；不可只存均值就宣称特征完全一致 |
| 采样诊断 | 参考点与最终sampling_locations的[0,1]外比例、真实kernel支持区外的权重质量及各level/agent分组；只存聚合和GT邻域，不存全attention数组 |
| 检测头 | fused cls/reg/dir原logits/tensor，全尺寸FP32；dir不是可选遗漏项，单车head须分开命名 |
| 全anchor候选 | 12800 raw anchor IDs、decoded centers、同公式完整方向纠正/角点投影、scores；包括score≤0.2；原方向纠正前副本另外保留以供重放 |
| 过滤lineage | score mask；large mask、z mask、AND；NMS输入ID顺序；native top1000、keep及排除理由；NMS后range mask；最终ID到raw anchor映射 |
| NMS抑制证据 | CPU复现的suppressor ID、浮点IoU、原分数/排序；逐项等于native keep才可解释抑制；超top1000单列 |
| 每GT | canonical历史sample_idx/gt_idx、运行GT映射、每阶段bestIoU/全部高IoU数/高IoU IDs及其删除阶段、最终原greedy状态与matched anchor |

每一候选ID为(sample_idx,alpha,anchor_flat_idx)，flat_idx=((y*80)+x)*2+anchor_orientation。跨α相同位置只表示同anchor结构，不等于真实车辆。原GT索引是NPZ数组行；物体自然身份若需讨论必须另核对数据集object_ids，不能从预测索引推断。

## 可靠区分候选不足和删除

设Q0为**全部**经方向纠正、ego投影的decoded boxes；Q1为score后，Q2为size/z后，Q3为top1000，Q4为NMS keep，Q5为final range。所有集合用raw anchor ID嵌套追踪。

对每GT计算 c_j=|{{q∈Qj: BEV_IoU(q,GT)≥0.7}}|，保存bestIoU和ID列表。c0=0才能说原始预测没有严格候选；c0>0但c1=0是score删尽；依次区分quality、top1000、NMS overlap、range。各高IoU候选的删因允许同时涉及多个阶段；“最后可用性丢失点”与“所有候选首次删除分布”分别报告。最终c5>0但贪心为FN另记matching competition。

角点、score、方向和lineage缺一不可。若只拿native nms输入，没有低分解码角点，就无法区分raw候选不足与score删除；若只有nms keep、没有top1000和抑制重放，就无法区分截断与overlap。非有限回归/exp溢出/无效polygon是数值故障，必须单列并停止可靠归因，不能塞进“没有高IoU框”。

## 保护和批准范围

本协议不是启动授权。本次不提供可直接运行PPU的launcher。后续需用户批准：实现隔离进程的临时采集runner（仅新P4_PILOT目录）、加载原checkpoint到cuda:0，至多18样本/40完整forward，CPU历史预处理轨迹重放与CPU后处理trace/replay，按IMPLEMENTATION_PLAN的磁盘/时间上限写诊断数据。不得扩大到全910或另换参数，失败不自动补跑。hook/trace/临时wrapper须finally撤销，不写回源码、配置、checkpoint和旧报告。
''')
put('IMPLEMENTATION_PLAN.md',f'''
# P4：最小采集实现、资源与验收

**本次交付为可审阅方案和CPU接口草案；没有完整PPU runner，也没有执行模型前向。** 用户批准下一阶段后，再在新的 P4_PILOT_<run_id>/ 实现runner，先完成静态审查和CPU自检再触发已批准的有限前向。

## 最小改动方法

1. 独立runner沿历史 train_utils/yaml_utils/build_dataset 读取原文件，运行时只复现已授权实验的batch_size=1、validate_dir=test_dir、k=3覆盖；不写配置、不改checkpoint/time code。先校验本次源码/配置/权重hash，有变化就停止并报告。
2. 模块 hooks：MetaFlow返回hook同步detach/copy raw，并返回output*α；fusion encoder pre-hook读取归一化前padded pred_offsets；encoder及6个block的post-hook做摘要/ROI。fused cls/reg/dir hooks获取全量head输出；不要挂single_*误取。用get_batch_pillar_features的临时bound wrapper记录agent_index/valid批行，为下次MetaFlow回调确定remote_slot，结合calibrate_data metadata映射到physical CAV ID；不把调用序号视为agent身份。
3. offset GT不是模型使用的stage2预测offset。保存它只供单位/方向/局部标签质量核查，零背景不作运动真值。本轮不训练、不改loss。
4. 原post_process保留为正式返回。仅对其精确code object安装受限只读trace，记录：在305行之前的全解码副本；317行方向分支前的score mask/所选scores；335处方向纠正后副本；363处投影角点；381行NMS调用前的质量mask及其输入；386处native NMS keep（相对质量后数组）；400处range mask。Python line trace在执行该行**之前**触发，必须用AST语句边界和运行hash断言，不靠未审核的手写行号。空框提前return也要记录，不能沿用上一个样本的缓存。
5. nms_rotated临时wrapper只复制输入并记录原函数返回，不替换native运算；CPU shadow使用相同float32分数、scores.argsort()[::-1]、top1000、同Shapely/common_utils float32 IoU和严格>0.15，记录suppressor。重放keep和native不同即停止NMS归因。不得改为torchvision NMS、stable sort或standup IoU。
6. 在独立副本上对全部12800 decoded anchors用native方向公式/角点/ego投影，保留低分框；不改原返回对象。每个score通过的anchor须与trace角点一致。方向/几何重放可CPU完成，允许1e-5m且任何IoU0.7边界差异都不能接受为成功分类。
7. stage归因读取Q0..Q5原anchor lineage；对所有162GT逐一计算相同BEV polygon IoU。用AABB无重叠剔除减少Shapely计算，但阈值附近必须精确复核，不用近似IoU。保存原greedy结果以区分候选存在与实际匹配。
8. 草案 diagnostic_contract_draft.py 提供三类接口：attribute_gt（嵌套集合/删因）、replay_native_nms（CPU原协议重放及suppressor）、paired_capture_hooks（依赖调用者提供slot和有界emit）。它不创建模型、不导入torch、不启动PPU，缺少完整trace/runner，**不可将静态通过称为采集器运行验收**。STATIC_CHECKS.json记录AST/编译和CPU单元验证；DIFF_AUDIT.json证明保护文件无改动。

## 数据体积预算（MiB=2^20，未压缩估计）

80×80、12800 anchors、最大4远端、FP32：

| 内容 | 每α估算 | 约束 |
|---|---:|---|
| cls/reg/dir全部原输出 (2+14+4)×6400×4 | 0.488 MiB | FP32，不压成FP16 |
| decoded centers (12800×7×4) | 0.342 MiB | 保留方向纠正前副本以重放 |
| 全候选角点 (12800×8×3×4) | 1.172 MiB | 包含低分框 |
| scores + masks/anchor IDs/NMS原因 | ≤0.6 MiB | 删因用数组/紧凑表，不重复角点到每一阶段 |
| raw/effective offset、padded核对、GT | ≤0.8 MiB | 有效槽原值0.195 MiB，避免重复存每层normalized值 |
| 特征摘要 + 每GT ROI | ≤2.5 MiB | ≤16个GT ROI，每ROI 9×9×128 FP32约40.5KiB；若GT>16，按gt_idx均匀选16个仅裁剪特征，所有GT的候选分析不裁剪 |
| 采样位置/attention摘要 | ≤0.5 MiB | 只聚合或固定GT小邻域，禁止全量location/attention落盘 |
| 小型元数据/最终框 | ≤0.2 MiB | 支持raw anchor ID追溯 |

计划每α约6–7MiB、每pair≤14MiB；18对+4次校验约280MiB，诊断输出软限512MiB、硬限1GiB（额外余量覆盖日志和NMS lineage）。不依赖压缩率。可选完整 fused feature 每α3.125MiB，本协议默认不保存全图；若摘要不足以解释，申请后再扩展，不能自动加保存量。

**禁止**保存所有attention arrays：N=5时单cross层location约46.875MiB、weights约23.438MiB，4层两α仅这些就约562.5MiB；历史多次MetaFlow还会放大。摘要应在当前层完成即释放，不在Python列表里保留设备引用。需比较两α精确特征差时，将第一侧FP32流转CPU暂存，第二侧做L2/cos/maxabs等配对摘要，之后立即丢弃，只保存摘要和ROI；max跨层同时保留一个stage，禁止保留完整计算图。

处理后配对输入缓存另算：Mtest≤70000、每voxel32×4浮点，单agent/单帧仅voxel_features理论上限约34.18MiB。最大5个主agent加4×5历史=25份，FP32 voxels约854.5MiB，再加coords/counts/labels后**每batch硬限1GiB**，不能把全部18batch常驻内存。磁盘一次归档每样本一份，只存原model-relevant处理字段，不存派生features和冗余可视化原点云，18份硬限18GiB；内存流式至多一个batch及两设备副本的CPU源，建议CPU RSS硬限8GiB。每次写入先按nbytes估算，超过限额即中止，不通过删字段或降精度继续。

总磁盘：输入归档≤18GiB + 诊断≤1GiB，预留至少25GiB可用空间，实时保留5GiB安全余量，文件用独占新目录及临时文件原子rename。现有450份raw文件约281MiB只是18例所需raw文件体积，**不是**历史预处理轨迹重放I/O：轨迹可能读取全test集多个delay，远大于此值。按estimate_resources.py仅stat估算，原数据准备轨迹约93466次文件读取、逻辑读取86.38GiB；去重原文件约5.68GiB，OS缓存会影响实际物理I/O。这里尚未运行这些预处理。

## 时间和设备预算

本次未做设备benchmark，不声称实测显存或精确速度。建议单 cuda:0、B=1、原FP32/eval/no_grad，核心36 + 校验4 = 最多40完整forward，内部每forward最多4次MetaFlow、每次5帧历史编码。额外设备常驻仅hooks瞬时拷贝/摘要，目标≤256MiB；不将完整中间层长期留在显存。原模型峰值未知，启动前只读检查可用内存；OOM就停止，不自动降精度/改batch/改配置。

保守审批预算：CPU历史预处理重放≤30分钟、PPU与拷贝阶段≤15分钟、CPU精确IoU/NMS重放及验证≤30分钟；这些是超时上限，不是性能预测。实际耗时逐段记录，超过任一上限停止并报告；不自动扩展到全910，不为追求历史一致而无限重试。

## 强制验收 / 失败保护

- 启动前：本次锁定输入和源码哈希一致，所选18sample/GT映射唯一；PPU明确批准；输出目录全新；空间与预算通过。原源码/config/checkpoint零修改。
- 首个样本先native无采集2次，再instrumented2次（已计核心36）；heads原输出与最终结果逐位一致为目标。若仅数值容差需记录最大差、保证没有阈值/匹配迁移；失败则停止所有后续样本。
- 配对：所有原始输入tensor/array值哈希一致、time300、history[3..7]、有效槽/物理CAV映射一致；raw MetaFlow和上游特征一致，effective=α*raw，归一化x/80 y/80一致；处理dict派生字段隔离，模型buffers无意外改动。违反则记实验失效，不做机制解释。
- 全anchor覆盖12800；score严格>0.2，所有mask、top1000、suppression、range lineage无重复、子集嵌套、无漏掉删除项。NMS input相对索引映射到raw anchor，最终框/score回连原结果。
- CPU重放native后处理、NMS keep和最终结果一致；为每个GT保留所有高IoU候选去向，跨阶段不修改其几何。原始非有限值、invalid polygon、无法确认的trace边界、错误索引一律终止归因，禁止记为缺候选。
- 历史复现：每例与P3-A保存GT/框/score和P3-B类别核对，超容差或0.7跨阈值即停止；明确处理set(object_ids)带来的跨进程行序风险，禁止按当前gt_idx直接代替历史GT。
- sample821逆序重复后应与其已采集结果一致；不一致则全pilot不可作为机制证据，不自动替换控制例。
- 每样本原子写入+hash+完成标记；异常finally撤销hook/trace/wrapper并释放设备，保留失败日志、不覆盖半成品、不自动重启。终端中断/OOM/磁盘超限停止，不影响原实验。
- 结束对源码、checkpoint、配置及历史产物做前后哈希/工作区差异审计；新P4_PILOT完成标记仅在全部验证后写入。

## 待用户批准的具体操作

允许在隔离进程中临时安装采集hooks/只读trace/wrapper；读取并CPU重放既定历史预处理轨迹，保存18份冻结batch；加载同一checkpoint到cuda:0；18个指定sample做至多40次完整forward；按上述限额写入全新的P4_PILOT目录。没有修改模型/数据集源码、checkpoint、配置文件或时间编码的请求。任何增加样本、forward、磁盘上限或干预条件都需另行批准。
''')
put('REPORT.md',f'''
# P4-Precheck 研究审计结论

**P4-Precheck已完成；源码证据、CPU抽样和待批准先导协议已交付。没有启动PPU或训练，尚不能判定严格IoU损失的首发阶段。**

1. 真实路径：历史采集脚本 → inference_intermediate_fusion（转调early接口） → PointPillarDeformable → 每远端5帧BEV历史 → MetaFlow原offset×α → DeforEncoderMultiScale移动远端reference points → fused feature → cls/reg/dir heads → VoxelPostprocessor。MetaFlow并非直接平移预测框，α0仍保留注意力自己的learned sampling offsets。
2. 后处理确切顺序：全anchor解码 → score>0.2 →方向纠正/角点投影 →尺寸/z检查 →NMS内部top1000 →overlap IoU>0.15抑制 →NMS后8角点range过滤 →eval_utils。旧NPZ已经经历全部步骤。源码另有size检查误读y/z及z上界1的行为，本次只记录、不修正，也不预断它是损失来源。
3. 可采集：原/有效offset、槽/物理CAV映射、fusion/blocks特征摘要与小ROI、完整cls/reg/dir、全部12800decoded anchors及每步mask、排序和NMS抑制来源、最终框。必须补全低分框的方向/角点诊断副本，并用native通过子集验证；只存score后pre-NMS不足以区分生成与过滤。
4. 先导：18个不同sample，损失/恢复/双TP控制各6焦点GT，覆盖10真实scene，同场景间隔至少13帧；每组近/远各3。检查这18例全部162GT。核心36次前向，加2次无采集对照和2次逆序重放，**上限40次**。同一次预处理batch缓存后配对，不分别随机预处理。历史逐帧GT/输出复现失败就停止归因。
5. 证据判据：逐GT在raw、score、quality、top1000、NMS、range各阶段保存bestIoU/高IoU数及anchor lineage；只在全部raw没有IoU≥0.7框时记候选生成不足，其余按真实mask和NMS suppressor区分删除。还要报告全部候选删因，不能把单个索引当真实物体身份。该方案定位处理阶段，不直接证明offset错误或给出模块因果贡献。
6. 待批准操作：隔离runner及临时hooks/trace；CPU预处理轨迹重放并归档18份冻结batch；同checkpoint在cuda:0至多40完整前向；向全新P4_PILOT目录保存诊断。诊断硬限1GiB、输入缓存≤18GiB，至少25GiB空闲，CPU重放/设备/CPU分析分别30/15/30分钟超时即停。原源码、配置、checkpoint和历史文件不改。

详见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)、[MECHANISM_HYPOTHESES.md](MECHANISM_HYPOTHESES.md)、[PILOT_PROTOCOL.md](PILOT_PROTOCOL.md)、[SAMPLE_SELECTION.csv](SAMPLE_SELECTION.csv)、[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)。验证及差异结果见 INPUT_VALIDATION.json、STATIC_CHECKS.json、DIFF_AUDIT.json。草案没有可直接运行推理的入口，未把静态检查当作PPU运行验收。
''')
print('Reports written:',len(list(R.glob('*.md'))))
