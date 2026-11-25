# RD4-D1-SMOKE｜V1 协议收口与首次真实模型运行验收

2026-11-25 · `RD4D1SMOKE_Sgm9MD` · **PASS_READY_FOR_PILOT** · [Issue #8](https://github.com/Judecoodingspace/LRCP/issues/8)

## 修正 ENV-R2 的历史扫解释

固定提交与 Notebook 中的生产 `intermediateLatency` loader 都在历史点云体素化前执行自车点过滤。样本 324 的原生 CPU `__getitem__` 追踪记录了 10/10 个历史扫调用，进入体素器前自车区域内点为 0。ENV-R2 直接调用体素器并向历史分支传入未过滤点云，先前把人工对照分支错误标注为生产路径。按实际过滤规则重读对照，全体 168 组逐字节相同。继续使用 **RD4_DETERMINISTIC_PREPROCESSING_V1**，重建资产 0，CPU 输入 72/72 通过。 [固定源码](https://github.com/Judecoodingspace/LRCP/blob/95e3bb36456542d8c24e37a463537a756efc63e8/opencood/data_utils/datasets/intermediate_latency_fusion_dataset.py#L120-L129) · [ENV-R2 原报告](rd4-env-r2-runtime-build.md)

训练时保存的 loader 源码与本次运行源码哈希一致。原 RD3 八个冻结样本的 80 个历史槽扫共检查 1,086,173 个真实非 padding 本地坐标点，区域内命中 0。源码、P4 构造路径及输入统计支持过滤口径，逐步训练随机点序仍无独立记录。

## 真实运行结果

单张 PPU-ZW810E 执行样本 324 的全部 8 项，调用 8 次，返回 8 次，门禁 8/8 通过。使用已验收供应商 PyTorch、SDK、原生扩展及固定 epoch19。

| 记录 | 输入与验收 |
|---|---|
| S01 | 原 RD3 F 复现。检测头、分数、几何和参考点对照最大绝对差均为 0，融合输入特征指纹、候选、NMS、最终 ID 和两阈值匹配一致。 |
| S02 | B3+C3 无 mask 控制通过。 |
| S03 | B3+C3 全 True，和 S02 的检测输出、参考点与四层 Attention 权重一致。 |
| S04 | B0+C3，实际模型时码 0/300，MetaFlow 两槽历史绑定通过。 |
| S05 | 重新读入 S04 混龄输入，输出和匹配一致。 |
| S06 | B0，C 缺席 lag3 占位。四层屏蔽权重精确为零。 |
| S07 | C 占位替换为真实 lag0。C 一级融合输入特征最大变化 3.68023，A/B 活跃特征、四层 Attention 权重和最终检测最大变化均为 0。 |
| S08 | Ego-only 通过，B/C 两槽四层屏蔽权重均精确为零。 |

原始 offset NPY 读取 0 次。lag3 使用历史真实监督 slice，lag0 使用原生零监督约定。Stage-2 参考点对照确认融合使用 MetaFlow 预测位移。训练和时间补偿干预均为 0。

## Pilot 准入

S03、S04、S06、S08 共 4 条可纳入正式 72 条。S01、S02、S05、S07 作为独立复现及控制保留。剩余 68 条尚未运行。本结果仅验收单样本工作流、运行复现和指定缺席槽扰动，无 AoI 科学效应结论。
