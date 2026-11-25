# RD4-ENV-R2｜运行环境重建与原生扩展验收

2026-11-25 · `ENV_R2_hR5FJq` · **PARTIAL** · [Issue #7](https://github.com/Judecoodingspace/LRCP/issues/7)

## 环境与算子

目标环境为独立 Python 3.10 虚拟环境，继承供应商 PyTorch `2.4.0+ppu1.5.0.oe`、CUDA build 11.8、PPU SDK `1.6.1-51ad16` 和 NumPy 2.0.2。安装了 Open3D 0.19.0、pyquaternion 0.9.9、icecream 2.2.0、固定历史提交的 pypcd 0.1.1 及解析出的传递依赖。CPU Cython 组件使用 Cython 3.0.12。安装前后 `pip check` 保留同样 4 个平台既有冲突。

重建了原生 MultiScaleDeformableAttention、spconv 1.2.1 和 CPU box_overlaps。供应商运行时报告两张 PPU-ZW810E，兼容目标为 8.0+PTX。独立 MSDA 小张量对照在第 0 张 PPU 通过两次原生前向和一组 FP32 梯度测试，FP32 与 FP64 相对作者参考实现的最大绝对误差分别为 4.66e−10、8.67e−19。Shapely rotated NMS、CPU box overlaps 与原生 anchor 生成通过固定检查。

## 生产预处理对照

在固定同一过滤规则、点序和参数时，原生 VoxelGeneratorV2 与 168/168 份冻结数组逐字节相同。测试覆盖首 32 点、容量截断、边界、ZYX 坐标、dtype 和零填充。按生产 intermediateLatency loader 的实际过滤规则，主扫 40/40 一致，历史扫 128/128 有真实点集或坐标差异。原 loader 的历史分支保留自车命中点，冻结 RD4 协议统一过滤这些点。按 128 组历史用途求和，原生路径多出 3,297 个体素坐标，914 个共有体素的保留点集改变，净多保留 89,121 个点。计数按用途分组汇总。

## Checkpoint 与门禁

原 epoch19 在 CPU 严格加载。580 个 state_dict 项全部匹配键名、shape 和 dtype。固定八项 Smoke 输入、mask 挂钩注册与移除、监督 offset 接线通过 CPU 接口检查。按原生 loader 出现历史点集差异，因此 E7 等待人工决定是否保留冻结的统一过滤协议。原定 8 次 Smoke 尚未具备启动条件，完整 LRCP forward、训练均为 0。

本轮保持原始模型源码、权重、72 条冻结输入及既有科学证据。恢复脚本、锁定依赖、构建日志和本地哈希记录在独立运行产物中。 [前序环境审计](rd4-env-r1-runtime-recovery.md)
