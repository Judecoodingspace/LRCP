# RD4-D0-R3｜确定性预处理协议冻结与生产接口验收

状态 **PARTIAL** · 运行 `RD4D0R3_20261124_222236_ZMlfqV`

## 实际通过

- 已冻结 `RD4_DETERMINISTIC_PREPROCESSING_V1`。A、B、C 主扫、历史及缺席占位用同一 NumPy 1.23.5 first-hit 协议。以固定命名空间、原始 sample_data token 和 main/history 用途通过 SHA256 派生 MT19937 种子。保留空间范围、容量、五帧主扫 warp 约定及原生时码 0/300。名义 0/600 ms 按 5 Hz 换算。
- 使用原始点云检查 168 个扫与用途组合。698 个重新体素化对照逐数组完全相等，其中 42 个由独立进程重建。72/72 条冻结候选通过 NumPy CPU 输入验收，固定 Ego、当前 GT 及顺序、anchor、评价坐标系，非干预 Remote 的主扫、历史、时码及双向变换不变。
- 未来 Pilot 包含 32 条新协议 lag3 基线和 40 条新增组合。RD3 历史 32 条仅作独立对照。Offset 原始 NPY 仍为 0/8；本方案使用真实封存 lag3 slice 与原生 lag0 零监督分支。

## 尚待验收

当前 CPU 环境无可导入的 PyTorch / spconv。原生体素接口与 Torch CPU 封装尚未实测。历史逐字节复现已经退出准入要求，新 NumPy 实现的确定性也不能代替原生接口验证。

目前不建议授权模型 Smoke。先完成 CPU 生产组件及封装验收，再考虑样本 324 的 **8 次**最小计划：1 次历史环境复现、4 次正式候选、3 次额外控制。须验证无掩码/全 True 等价、固定三槽缺席隔离、混龄条件、offset 接线、后处理及重复运行稳定性。

新增真实模型 forward、GPU/PPU 调用、训练均为 **0**。D1 尚未启动。此公开记录只含结果和研究边界。

[Issue #5](https://github.com/Judecoodingspace/LRCP/issues/5) · [实验索引](README.md)
