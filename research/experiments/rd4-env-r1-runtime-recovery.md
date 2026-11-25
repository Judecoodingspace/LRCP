# RD4-ENV-R1｜历史运行环境恢复与生产组件审计

**NEED_EXTENSION_BUILD，另需安装依赖授权。D1 Smoke 未启动。**

历史供应商 Torch 2.6.0 环境未能安全恢复。旧构件存在，完整历史 venv 和匹配发行包缺失。当前平台 Python 3.10 的供应商 Torch 2.4.0+ppu1.5.0.oe 可运行 CPU，保留已通过的 72 条 Torch CPU 输入验收。

历史 MSDeformAttn、spconv 和 pcdet GPU 扩展需要 CUDA 12 运行时 SONAME。当前 SDK 提供 CUDA 11.8；历史 libspconv 的一个 c10 强符号在当前 Torch 导出库扫描中也未找到。检查是静态的，未加载历史二进制。

固定 Stage-2 模型的原生 Deformable Attention 是主要前向阻塞，冻结体素可绕过 spconv 原始预处理。Open3D 由原生产入口的顶层依赖引入，走此入口仍需满足真实导入。后处理还需验证 box_overlaps 及 Shapely 口径。spconv 仍是生产体素化对照门禁。checkpoint 已核对哈希，尚未实例化或 strict 加载。

推荐以当前供应商 Torch 为基础建立单独继承环境，分开审批依赖安装和原生扩展重建；如可获取历史供应商镜像，可另行复验原环境。新增模型前向、GPU/PPU、训练、安装和编译均为 0。

[本轮 Issue](https://github.com/Judecoodingspace/LRCP/issues/6) · [索引](README.md)
