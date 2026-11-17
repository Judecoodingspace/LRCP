# LRCP 实例清理前恢复说明（2026-11-17，Asia/Shanghai）

本目录是待审核的公开代码候选，尚未推送。代码快照包含原始研究脚本和部分报告生成模板，可能揭示研究个案与数值；请确认可公开后再推送。凭据模式扫描无发现不代表人工保密审核已完成。原始数据、研究结果表、模型权重和完整运行账本另在私有备份中。

## 代码身份

- GitHub目标为公开仓库 https://github.com/Judecoodingspace/LRCP 。扫描时远端仅main，SHA为809ba62aad7c3e45c919e088579e60d94643c9bd。
- 当前实例HEAD为44c9aaf8fa62ed3cb07d156696e52136efcf50fb，分支sens/20260918-stage2-paper-lr-5e-4。此分支与main的科学配置差异为Stage-2学习率。独立eval分支cba5d37eaa033b96fed6f875ab068148336644c8改变评价后处理；两种配置不能混用。
- 提案备份分支以当前HEAD为基线，另保存eval和sens相对main的补丁；本地LRCP_ALL_REFS.bundle保存全部已有分支和历史。新实例可以先从该bundle克隆当前sens分支，或在分支发布后按明确commit克隆GitHub。
- instance_snapshot/workspace/下是从工作区外置研究目录逐字节复制的源代码；SOURCE_SNAPSHOT_MANIFEST.csv列出原始路径、目标路径和SHA256。所有文件原始内容保持不变，包含历史版本和测试代码。

## 恢复顺序

1. 在实例清理前，将Git分支推送并回读验证，同时把私有备份传到实例之外。仅保存在/workspace的文件仍会丢失。保存平台镜像ID/快照和持久磁盘信息。
2. 新实例恢复到/workspace布局；先克隆固定commit的LRCP，保持/workspace/LRCP。研究快照不是可直接在仓库子目录运行的新包，原脚本大量使用/workspace绝对路径。
3. 使用restore_archives.py对PRIVATE_RD3_MINIMUM_CONTINUATION.tar先dry-run再加--apply，恢复原始候选、8个冻结batch、指定权重和handoff内容。随后对handoff/PRIVATE_project_text_snapshot.tar.gz同样恢复全部文本状态。脚本拒绝覆盖不同内容；先完成干净恢复，不通过改Hash清单绕过冲突。
4. 需要完整历史服务器验收时，另外恢复完整lrcp_d2、模型日志/权重、数据索引及历史依赖。最低续研包未包含全部历史二进制；R5D的历史只读封存验收不能仅靠最低包通过。保留真实运行budget/claim/manifest，不将mock账本冒充真实账本，也不重置attempt。
5. 仅复核R5D：解压R5D_UPLOAD_BUNDLE.zip到独立目录，建立CPU虚拟环境，安装requirements.cpu-audit.txt，用该环境python -B validate_upload_cpu.py。该命令只做保存候选CPU后处理。没有必要初始化CUDA或重新forward。
6. 后续真实模型工作须恢复平台PPU环境和数据，再另行按新任务授权执行。迁移过程不自动启动训练、推理或下一阶段研究。

## 实际环境与安装边界

- Ubuntu 22.04.5，x86_64，Python3.10.13；现有/workspace/venvs/lrcp使用include-system-site-packages=true。
- 系统继承：平台PPU适配torch2.6.0、torchvision0.21.0、spconv1.2.1、MultiScaleDeformableAttention1.0、NumPy1.23.5；Shapely1.8.5.post1位于venv。PPU SDK1.6.1-14620b。完整当前包清单另存私有环境快照。
- 仓库environment.yml的Python3.7和未固定依赖不能复原该环境。pip版本清单不能替代平台专用PyTorch、PPU SDK、驱动和镜像。不要用普通PyPI torch覆盖平台包。
- 应选择相同平台镜像，执行其/usr/local/PPU_SDK/envsetup.sh设置环境；镜像ID/摘要目前尚未从项目文件确定，需要平台保全。SDK的CUDA接口名称不代表NVIDIA硬件。
- spconv源码固定fad3000249d27ca918f2655ff73c41f39b0f3127，4个本地PPU修改在spconv_1.2.1_PPU.patch。子模块：cutlass c2b80ad4e4f8b60a65500bd04c8fecddff2ba355；mp11 29764aad4881fde809af6a025c12012e47a55515；pybind11 3b1dbebabc801c9cf6f0953a4c20b904d444f879。旧.so/build不能作为跨镜像可移植安装包；按相同ABI重建，保留已有构建日志供参考。
- LRCP原生扩展源码在opencood/utils/ms_deform_attn_ops、opencood/pcdet_utils和opencood/utils。本轮没有执行编译、安装或设备验证；此说明不是新实例部署已通过的证明。

## 数据与科学记录

RD3指定权重为/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth，SHA256=666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724。不要用同目录普通epoch19或敏感性/E1权重替代。

32份原始NPZ按R5B_RAW_SHARD_LOCATOR.csv保存，8个冻结输入按FROZEN_MANIFEST.json保存；所有原始路径和SHA在私有CRITICAL_EVIDENCE_MANIFEST.csv。便携R5D包能独立复核19条CPU重放，但不能替代整个原始实验与未打包历史的身份验证。

后续完整模型工作还需/workspace/datasets/V2X-Sim-2.0和/workspace/datasets/v2xsim2_info_generated。offset_maps为生成资产，不能仅以重新下载原始数据替代。backup_staging同时有完整offset tar和分片，属于实例本地副本；先验证一份实例外副本，再考虑是否还需冗余文件，不在本任务中删除。
