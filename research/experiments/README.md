# LRCP 实验发布索引

从[最新一轮](rd4-d4-r3-source-slot-disentanglement-prep.md)或[机器可读索引](index.json)开始。各报告记录结论、证据与复现限制，GitHub Issues 记录对应验收。

- [RD3｜324 帧定位稳健性收口](rd3-324-localization-robustness.md) · **PASS** · `start.V3JpuM` · [Issue #1](https://github.com/Judecoodingspace/LRCP/issues/1)
- [RD4-D0｜逐 Agent 观测年龄可行性](rd4-d0-independent-age-feasibility.md) · **PARTIAL** · `RD4D0_20261124_173647_KKSrQo` · [Issue #2](https://github.com/Judecoodingspace/LRCP/issues/2)
- [RD4-D0-R1｜最小原始数据恢复与时间核验](rd4-d0-r1-source-recovery.md) · **PARTIAL** · `RD4D0R1_20261124_203814_wB5LsL` · [Issue #3](https://github.com/Judecoodingspace/LRCP/issues/3)
- [RD4-D0-R2｜独立年龄加载与 CPU 输入等价](rd4-d0-r2-loader-equivalence.md) · **PARTIAL** · `RD4D0R2_20261124_211604_MoklD9` · [Issue #4](https://github.com/Judecoodingspace/LRCP/issues/4)
- [RD4-D0-R3｜确定性预处理协议冻结与生产接口验收](rd4-d0-r3-deterministic-protocol.md) · **PARTIAL** · `RD4D0R3_20261124_222236_ZMlfqV` · [Issue #5](https://github.com/Judecoodingspace/LRCP/issues/5)

当前准备完成 **RD4-D4-R3-PREP**，208 条输入通过 CPU 预检，等待手动正式运行。

- [RD4-D0-R3 补充 CPU 接口验收](rd4-d0-r3-cpu-interface-supplement.md) · **PARTIAL** · Torch CPU 72/72 · [Issue #5](https://github.com/Judecoodingspace/LRCP/issues/5)

- [RD4-ENV-R1｜历史运行环境恢复与生产组件审计](rd4-env-r1-runtime-recovery.md) · **NEED_EXTENSION_BUILD** · [Issue #6](https://github.com/Judecoodingspace/LRCP/issues/6) · 等待兼容环境授权，D1 未启动。

- [RD4-ENV-R2｜运行环境重建与原生扩展验收](rd4-env-r2-runtime-build.md) · **PARTIAL** · `ENV_R2_hR5FJq` · [Issue #7](https://github.com/Judecoodingspace/LRCP/issues/7)

- [RD4-D1-SMOKE｜V1 协议收口与首次真实模型运行验收](rd4-d1-smoke.md) · **PASS_READY_FOR_PILOT** · `RD4D1SMOKE_Sgm9MD` · [Issue #8](https://github.com/Judecoodingspace/LRCP/issues/8) · 含 ENV-R2 历史过滤补充说明。

- [RD4-D1-PILOT-PREP｜正式 Pilot 终端运行准备](rd4-d1-pilot-prep.md) · **READY_FOR_MANUAL_PILOT** · `RD4PILOT_apyoxH` · 新增完整 forward 0。

- [RD4-D1 Pilot｜正式运行验收与两档名义年龄分析](rd4-d1-pilot-final.md) · **PASS** · `RD4PILOT_apyoxH` · [Issue #9](https://github.com/Judecoodingspace/LRCP/issues/9) · 72/72，新增 forward 68。

- [RD4-D2-R1｜观测年龄效用的 GT 级机制分析](rd4-d2-r1-age-mechanism.md) · **PASS** · `RD4D2R1_eUmvYD` · [本轮 Issue](https://github.com/Judecoodingspace/LRCP/issues/10)。

- [RD4-D4-R1-PREP｜独立场景复验准备](rd4-d4-r1-independent-validation-prep.md) · **READY_FOR_MANUAL_VALIDATION** · `RD4D4R1PREP_epZeZe` · [Issue #11](https://github.com/Judecoodingspace/LRCP/issues/11) · CPU 输入 108/108，正式科学记录 0/108。

- [RD4-D4-R1｜独立场景复验与观测年龄效用](rd4-d4-r1-independent-validation-final.md) · **PASS** · `RD4D4R1PREP_epZeZe` · [Issue #12](https://github.com/Judecoodingspace/LRCP/issues/12) · 108/108 条正式捕获。

- [RD4-D4-R2｜规模化年龄效用与物理槽位控制准备](rd4-d4-r2-scale-validation-prep.md) · **PARTIAL（技术验收通过，Notebook 到期时间待核实）** · `RD4D4R2PREP_7nNcY2` · [Issue #13](https://github.com/Judecoodingspace/LRCP/issues/13)。

- [RD4-D4-R2｜规模化观测年龄效用与物理槽位控制](rd4-d4-r2-scale-validation-final.md) · **PASS** · `RD4D4R2PREP_7nNcY2` · [Issue #14](https://github.com/Judecoodingspace/LRCP/issues/14) · 924/924 条正式捕获，CPU 收口新增 forward 0。

- [RD4-D4-R3-PREP｜物理来源与模型槽位析因准备](rd4-d4-r3-source-slot-disentanglement-prep.md) · **READY_FOR_MANUAL_SLOT_VALIDATION** · `RD4D4R3PREP_trGF8O` · [Issue #15](https://github.com/Judecoodingspace/LRCP/issues/15) · 200 条原序复用，正式新增 forward 0。
