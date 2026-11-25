# RD4-D0-R3｜已有环境与 CPU 张量补充验收

**总体验收 PARTIAL。72/72 条冻结候选通过实际 Torch CPU 封装，生产 voxelizer 仍阻塞。**

原 R3 仅用隔离 NumPy venv 检查，范围限于该解释器。当前平台 Python 3.10.13 实际导入 torch 2.4.0（发行包 2.4.0+ppu1.5.0.oe，CUDA build 11.8），FreeSurfer Python 3.8.13 实际导入 torch 2.0.0+cu117。两者均通过 CPU 运算；各候选解释器均无法导入 spconv。历史 R4 使用独立 Python 3.10 venv，搭配 torch 2.6.0、spconv 1.2.1；该历史 venv 在当前实例未找到。

以当前平台 Torch 使用历史张量类型模板物化冻结输入，核验 2,664 个 CPU 张量的内容、shape、dtype、设备以及物理槽、BZYX、pairwise、GT、anchor、offset 和 mask。原 72 条冻结输入未改变。原生 LRCP 预处理入口导入还遇到 open3d 缺失，尚未执行 native voxel generator 或与 NumPy 做语义对照。当前平台的 CPU 成功结果不能作为历史 PPU 环境兼容证据。

**保留 PARTIAL，暂不授权 Smoke。** 已关闭 Torch CPU 张量封装缺口，仍需确认生产体素化和兼容运行环境。新增前向、GPU/PPU 调用和训练均为 0。未安装或升级依赖，未修改核心源码或历史证据。D1 未启动。

[原 R3 报告](rd4-d0-r3-deterministic-protocol.md) · [实验索引](README.md) · [Issue #5](https://github.com/Judecoodingspace/LRCP/issues/5)
