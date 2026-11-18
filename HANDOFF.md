# LRCP 新实例 Handoff：保全、部署与 Codex 接手

更新：2026-11-18（Asia/Shanghai）。适用于本项目从旧智算 Notebook 迁移到新实例。本文及工具会更新在 `preserve/instance-20261117` 分支。公开仓库不保存私有原始数据、PKL、权重或二进制扩展备份。

## 0. 先回答：需要下载到连接 Notebook 的电脑吗？

**可以，而且在没有已确认保留的持久盘/对象存储时，应先下载到电脑或另一处实例外存储，校验后再清理旧实例。创建新实例后重新上传。** 电脑只是中转；连接着 Notebook 不会自动备份服务器文件。`/workspace/backup_staging` 也仍是旧实例本地目录。

只保存 GitHub 代码和1.33 GiB续研包，可以恢复本轮冻结候选与CPU离线审计；**不能恢复整个数据集、全部历史二进制或所有训练状态**。后续真实模型任务还需保存数据与平台环境。源/目标文件名相同不代表传输成功，必须比较SHA256。

### 清理前需要拿走什么

| 文件/范围 | 旧实例位置 | 用途与优先级 |
|---|---|---|
| `PRIVATE_RD3_MINIMUM_CONTINUATION.tar`，1,425,346,560字节，约1.33 GiB | `/workspace/lrcp_migration_audit_20261117/` | 必存。32条完整原始NPZ、8个冻结batch、指定RD3权重、报告/账本/源码文本、Git bundle、R5上传包等 |
| `PRIVATE_HANDOFF_KIT_20261118.zip`，约100 MiB，以实际文件大小为准 | `/workspace/lrcp_handoff_20261118/` | 建议与最低包一起下载。含本文、恢复工具、离线CPU wheel、数据索引备份、同ABI扩展备份及校验清单；不含下面约80 GiB数据归档 |
| 原始数据六个ZIP，合计约44.6 GiB | `/workspace/datasets/downloads/V2X-Sim-2.0/` | `lidar1.zip`、`lidar2.zip`、`lidar3.zip`、`lidarseg.zip`、`maps.zip`、`v2.0.zip`。若电脑已有同Hash文件，无需重复下载 |
| `offset_maps_20260902.tar`，约34.94 GiB | `/workspace/backup_staging/` | LRCP生成资产，10,000个`.npy`，数组总计37,505,280,000字节。原始数据ZIP不含它；不能漏存 |
| 完整历史目录，`lrcp_d2`约37.1 GiB逻辑大小（磁盘占用约35 GiB），另有模型日志等 | `/workspace/lrcp_d2`、`/workspace/LRCP/opencood/logs`等 | 需要完整历史服务器验收、其他阶段或全部权重时必须另备份；最低包不替代此项 |
| 平台镜像/持久盘信息 | 平台管理页 | 记录镜像名称、ID/摘要、PPU套餐、持久卷ID及挂载点。若支持保存当前Notebook为自定义镜像，清理前保存；镜像是否包含`/workspace`须单独确认 |

最低续研包SHA256：

```text
26b3d6a4e9ee6cf4feba2d7949e3632eed181db9c631f8daa143c312b53c78a4
```

数据ZIP、offset归档和补充包的精确大小/Hash在 [`recovery/handoff_20261118/ASSET_MANIFEST.json`](recovery/handoff_20261118/ASSET_MANIFEST.json)。数据传输校验文本为 [`DATASET_TRANSPORT_SHA256SUMS.txt`](recovery/handoff_20261118/DATASET_TRANSPORT_SHA256SUMS.txt)。工具包自身的SHA放在其旁边的`DOWNLOAD_SHA256SUMS.txt`，避免循环Hash。

**最低包没有训练/验证/测试PKL。** 本次补充的`PRIVATE_DATASET_METADATA_20261118.tar.gz`已收入工具包，保存原有3个PKL、生成脚本和日志等10个文件。不要到新实例后直接重建PKL而丢失既有数据身份。

## 1. 平台上选什么环境

用户提供的是`zhisuan.com`域名的Notebook详情页；当前工具读取该页面时连接被重置，未核验登录后的控件、套餐名称或镜像列表。下面是实例内查证的配置要求，不能据此虚构平台按钮。私有工具包中的`PLATFORM_PRIVATE_NOTES.json`保留用户给出的详情页地址；公开文档不公开原Notebook标识。

### 选择分成两种，不必为纯CPU审计租加速卡

| 目标 | 建议选择 | 说明 |
|---|---|---|
| 继续R5D保存候选的CPU分析/重放 | x86_64，Ubuntu22.04，Python3.10；规划4–8 vCPU、16–32 GiB RAM、至少40 GiB可用空间 | 这是容量规划建议，非新实例实测最低门槛；不需要PPU/GPU和完整数据集 |
| 恢复完整历史后做CPU服务器验收 | 同上，额外预留历史文件与解包空间，建议至少100 GiB | 需恢复所有被历史Hash封存引用的文件，不能用便携验收代替完整验收 |
| 后续真实模型工作 | 优先恢复同款PPU-ZW810E、相同供应商镜像和SDK/驱动组合；完整数据工作区建议500 GiB持久存储 | 历史采集记录为2张PPU-ZW810E，每张98,304 MiB（96 GiB）；容器配额实测32 vCPU、256 GiB RAM。它们是旧环境事实，**不是已经测得的最低需求**。单设备任务可先申请1张同款卡进行环境恢复；双卡任务按原配置恢复，缩容需重新审计 |

原环境关键版本：

- Ubuntu22.04.5、x86_64、glibc2.35、Python3.10.13。
- 平台PPU版`torch==2.6.0`、`torchvision==0.21.0`；CUDA接口版本历史记录为12.6，但设备实际为PPU。
- PPU SDK `1.6.1-14620b`；旧采集驱动 `1.5.1-1d747a`，PPU-SMI1.18。驱动是历史采集值，新套餐需平台确认兼容性。
- `spconv==1.2.1`（有4处本地PPU源码修改）、`MultiScaleDeformableAttention==1.0`。
- `numpy==1.23.5`、`Shapely==1.8.5.post1`、`PyYAML==6.0`；其他原环境版本见最低包内环境清单。
- 原`/workspace/venvs/lrcp`是`include-system-site-packages=true`的venv，torch等实际来自系统。单独拷贝venv不能带走系统包、驱动和SDK。

优先保存当前可工作Notebook的**自定义镜像**，其次才是选择同供应商基础镜像并恢复自定义依赖。项目文件未提供可核实的镜像ID/摘要，应在平台清理前记录，不能猜一个通用“PyTorch2.6 CUDA12.6”镜像替代。仓库`environment.yml`仍是Python3.7时代的配置，不应当作本次部署锁文件。普通PyPI版torch不能直接替代PPU适配版。

如果只能选择NVIDIA GPU，这是一次新的兼容迁移：PPU扩展包不能直接使用，需重新编译并另行验证数值/算子；不能把它描述为原环境已恢复。

### 创建Notebook前逐项核对

1. 工作盘是否在删除Notebook后保留；记录卷ID、挂载路径和保留政策。看到大容量`overlay`不等于有持久盘。
2. 确保项目可使用`/workspace`路径，或将已确认的持久卷按原布局挂到该路径。脚本和数据索引大量使用绝对路径。
3. 确认Python3.10、架构x86_64以及相同PPU镜像；记录镜像摘要。先分配恢复所需资源，研究任务另行启动。
4. 创建时设置共享内存；若页面没有该项，向平台支持提供下一节的Docker/Kubernetes配置要求。
5. 确认是否有独立SSH/SFTP入口、主机和端口。Notebook网页的`:32082`不是可推定的SSH端口。

## 2. 共享内存：之前踩坑的地方

旧实例`df -h /dev/shm`实际只有**64 MiB**。它与256 GiB容器RAM、PPU显存和工作盘容量是不同资源。加RAM/显存/磁盘不会自动扩大`/dev/shm`。

该项目历史上多worker数据加载因共享内存受限失败，正式延迟入口已采用`num_workers=0`；训练同时保留`prefetch_factor=None`。恢复时先保持已验收的单进程方案：

```python
num_workers=0
prefetch_factor=None
persistent_workers=False
```

不要给`num_workers=0`配`prefetch_factor=2`或启用persistent workers。不要为绕过共享内存问题改变原科学batch size、数据语义或阈值。`pin_memory`沿用入口原值：已验收延迟训练入口为True，延迟推理入口为False；不要全局乱改。

**并非仓库所有入口都已修复。** `train_w_delay.py`、`inference_w_delay.py`采用workers0；`train.py`、`inference.py`等通用入口仍可能是workers4，其他入口甚至workers8。新任务必须先确认实际入口，不能因为克隆了备份分支就假设所有DataLoader安全。`preflight.py`会列出这些设置，不会修改源码。

新实例建议先预留`/dev/shm=8 GiB`作为起点，32+ GiB RAM时规划；仍从workers0恢复，是否增大并行度须另做资源验证。8 GiB不是已证明适合所有训练批次的充分条件。如果平台只能保留64 MiB，仍可做CPU离线审计；模型恢复先保持原workers0路径并单独验证，不启动未审计的4/8-worker入口。自己有容器启动权限时，对应Docker参数是`--shm-size=8g`；平台托管Notebook应在创建配置/后端设置，而不是假定Notebook内能运行Docker。Docker缺省共享内存为64m。[Docker官方说明](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)

若平台支持Pod配置，管理员可把内存型`emptyDir`挂到`/dev/shm`，例如：

```yaml
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
# 挂到目标容器：
volumeMounts:
  - name: dshm
    mountPath: /dev/shm
```

内存型卷会计入内存资源，需与RAM配额一起规划；Notebook中的`emptyDir`仍不是持久备份。[Kubernetes官方说明](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir) 单进程加载和worker/prefetch参数语义参考[PyTorch2.6 DataLoader文档](https://docs.pytorch.org/docs/2.6/data.html)。

创建后先在Terminal执行：

```bash
python3 --version
df -h /workspace /dev/shm
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
cat /sys/fs/cgroup/memory.max 2>/dev/null || true
```

cgroup v1/v2文件不一定都存在；不要只看宿主机`free -h`便认定拥有全部RAM。不要在不清楚平台权限/内存上限时强行remount共享内存或使用host IPC。

## 3. 电脑下载与新实例上传

### 3.1 从旧实例下载

可用JupyterLab/平台文件管理器下载以上文件，或使用平台**实际提供**的SSH/SFTP入口。若浏览器不支持超过1 GiB文件，先在旧实例用标准Python按512 MiB分片：

```bash
python3 /workspace/lrcp_handoff_20261118/transfer.py split \
  /workspace/lrcp_migration_audit_20261117/PRIVATE_RD3_MINIMUM_CONTINUATION.tar \
  --out /workspace/transfer_minimum --chunk-mib 512
```

将所有`.part-00000`等分片及`.parts.json`一起下载。该脚本也适用于约20 GiB的`lidar1/2.zip`和35 GiB offset归档，不必重复打包数据。保留原文件，先完成实例外验证。

下载后校验：Linux用`sha256sum 文件`，macOS用`shasum -a 256 文件`；Windows PowerShell：

```powershell
Get-FileHash .\PRIVATE_RD3_MINIMUM_CONTINUATION.tar -Algorithm SHA256
```

分片可在电脑或新实例用同一个Python工具合并（Windows可用`py -3`代替`python3`）：

```bash
python3 transfer.py join PRIVATE_RD3_MINIMUM_CONTINUATION.tar.parts.json \
  --output PRIVATE_RD3_MINIMUM_CONTINUATION.tar
```

脚本逐片核对大小/Hash，并核对整体Hash；缺片、损坏或已有不同输出会失败。失败后保留的`.partial`仅是未完成输出，检查后可单独移走再重试。**不要用PowerShell文本拼接或编辑器合并二进制分片。**

### 3.2 完整历史另行归档（需要完整服务器验收时）

最低包不含所有历史二进制。如需完整保存，清理前另行生成并下载完整历史包；以下只给命令，本轮未运行这项约40 GiB的额外归档：

```bash
tar --sparse -cf /workspace/LRCP_FULL_HISTORY_20261118.tar -C /workspace \
  lrcp_d2 lrcp_d1 lrcp_b0 lrcp_e1_configs lrcp_e1_logs lrcp_recovery \
  LRCP/opencood/logs third_party
sha256sum /workspace/LRCP_FULL_HISTORY_20261118.tar \
  > /workspace/LRCP_FULL_HISTORY_20261118.tar.sha256
```

该包还要与Git代码、数据索引/数据归档和平台镜像配套，先核对历史封存引用的所有路径都被覆盖。源历史包与Hash文件都要拿到实例外；不能为了省空间先删除原历史。新实例解包到临时目录检查后按原路径恢复；有冲突先比Hash，不盲目覆盖。

### 3.3 上传到新实例

先创建`/workspace/restore_uploads`；通过Notebook文件上传、平台文件管理或SSH/SFTP把最低包和工具包传进去。数据归档放`/workspace/restore_uploads/dataset_archives`。

若平台提供SSH入口，可从电脑执行（替换所有尖括号字段）：

```bash
scp -P <实际SSH端口> PRIVATE_RD3_MINIMUM_CONTINUATION.tar \
  PRIVATE_HANDOFF_KIT_20261118.zip <用户>@<SSH主机>:/workspace/restore_uploads/
```

几十GiB的数据建议使用支持续传的SFTP或rsync工具；先核对平台支持的入口。浏览器传大文件中断时可改用分片方案。新实例重新计算SHA256，与电脑/本清单一致后才解包。不是仅比较文件名或显示“上传完成”。

建议电脑保留：最低包约1.33 GiB+工具包约100 MiB；如果要保全完整模型数据，六个ZIP+offset归档另约79.6 GiB，再加完整历史备份和空间余量。电脑已有同Hash六ZIP时，只下载缺失的生成资产与历史。完整保存数据和历史时，电脑/外置盘建议留至少150 GiB空闲；使用支持大文件的文件系统，避免FAT32单文件4 GiB限制。清理前至少确认一份实例外副本可读且Hash正确，最好另留第二份副本。

## 4. 新Notebook快速恢复：先完成CPU路线

以下命令在**新实例的Terminal**执行。克隆前若`/workspace/LRCP`已存在，先检查，禁止盲目覆盖；恢复工具也会拒绝覆盖不同内容。

```bash
mkdir -p /workspace/restore_uploads /workspace/handoff_checks
cd /workspace
git clone --branch preserve/instance-20261117 \
  https://github.com/Judecoodingspace/LRCP.git LRCP
cd /workspace/LRCP
git log -1 --oneline
```

已发布源码保全提交为`e12c82ebee0d22d0cdfac2aaf3a5bd9f41b91e9d`，后续Handoff提交仅增加恢复材料。原始32条模型结果对应其冻结配置/权重，不是新Handoff提交又运行的模型。完整scientific源码对比：

```bash
git diff e12c82ebee0d22d0cdfac2aaf3a5bd9f41b91e9d -- opencood
```

新建工作区应为空差异。记录新HEAD，但不得为“匹配旧结果”修改历史Hash或阈值。

若GitHub暂时无法访问：先在上传目录校验并解开工具包，再从其中的完整Git bundle克隆，同样包含本版Handoff：

```bash
cd /workspace/restore_uploads
sha256sum -c DOWNLOAD_SHA256SUMS.txt
python3 -m zipfile -e PRIVATE_HANDOFF_KIT_20261118.zip handoff_kit
git clone --branch preserve/instance-20261117 \
  /workspace/restore_uploads/handoff_kit/git_backup/LRCP_HANDOFF_20261118.bundle \
  /workspace/LRCP
```

两种克隆方式只选一种，目标目录必须是新的。工具包本身已带恢复脚本，支持离线接手。新实例缺少`git`、`python3.10`或venv支持时，先换到匹配的Notebook镜像，或请平台管理员补齐这些基础工具；不要用Python3.11强装本包cp310 wheel。

### 4.1 解开工具包、恢复最低包

```bash
cd /workspace/restore_uploads
# 先用旁边的DOWNLOAD_SHA256SUMS.txt核对两个下载文件
sha256sum -c DOWNLOAD_SHA256SUMS.txt
test -d handoff_kit || python3 -m zipfile -e PRIVATE_HANDOFF_KIT_20261118.zip handoff_kit
cd /workspace/LRCP
python3 recovery/handoff_20261118/bootstrap_minimum.py \
  --archive /workspace/restore_uploads/PRIVATE_RD3_MINIMUM_CONTINUATION.tar
# dry-run通过后实际恢复
python3 recovery/handoff_20261118/bootstrap_minimum.py \
  --archive /workspace/restore_uploads/PRIVATE_RD3_MINIMUM_CONTINUATION.tar --apply
```

工具自动完成外层包与其内部文本快照两次恢复，校验固定SHA并拒绝不同内容覆盖。外层`handoff/`恢复到`/workspace/_lrcp_handoff`；原实验内容按原`/workspace/...`路径恢复。默认dry-run还检查内部文本恢复，**不是只解开外层便结束**。

`PRIVATE_project_text_snapshot.tar.gz`已在最低包内，不必另行下载第二份。源码快照`instance_snapshot/workspace/...`只是仓库内保全副本；科学脚本仍从恢复后的原绝对路径运行，不直接把快照子目录当新的项目根。

### 4.2 创建CPU审计环境

工具包的`cpu_wheels/`包含Python3.10/Linux x86_64适用的3个固定版本wheel。优先离线安装：

```bash
cd /workspace/LRCP
bash recovery/handoff_20261118/setup_cpu.sh \
  /workspace/restore_uploads/handoff_kit/cpu_wheels
/workspace/venvs/lrcp_cpu_audit/bin/python -B \
  recovery/handoff_20261118/preflight.py --mode cpu \
  > /workspace/handoff_checks/cpu_preflight.json
```

若不带wheelhouse参数，脚本会从PyPI安装同3个版本；不安装torch，不导入LRCP模型，不查询设备。不要用`pip install -r requirements.txt`整体替换平台依赖。CPU环境是独立venv，避免改动供应商系统环境。

Notebook的Python内核可能仍是系统Python。初次恢复优先在Terminal使用上面的完整解释器路径。如果确需Notebook内核，单独安装ipykernel并注册对应venv，选择该内核后核对`sys.executable`；不要误以为终端激活venv会自动切换已有Notebook内核。

### 4.3 用保存候选验收，禁止为恢复而重新forward

```bash
replay_dir="$(mktemp -d /workspace/handoff_checks/r5d_portable.XXXXXX)"
/workspace/venvs/lrcp_cpu_audit/bin/python -m zipfile -e \
  /workspace/lrcp_d2/codex_reports/RD3_R5D_GEOMETRY_SCORE_COUNTERFACTUAL_v2_20261116/R5D_UPLOAD_BUNDLE.zip \
  "$replay_dir"
CUDA_VISIBLE_DEVICES='' /workspace/venvs/lrcp_cpu_audit/bin/python -B \
  "$replay_dir/validate_upload_cpu.py"
```

预期输出`R5D_UPLOAD_REPLAY_PASS`，19条CPU重放、16个2×2组合、3个Score交换、8条原始组合精确复现，`torch_imported=false`。这是便携证据验收，**不是完整历史服务器验收，也不是PPU算子验收**。新实例未实际创建前，本文只能声明旧实例工具核验通过，不能声明新实例已部署成功。

## 5. 数据集如何上传、解包与核验

### 5.1 必须恢复的布局

```text
/workspace/datasets/
├── V2X-Sim-2.0/
│   ├── sweeps/          # 六ZIP中的lidar1/2/3解出同一目录树
│   ├── maps/
│   ├── v2.0/            # 本项目使用的元数据版本
│   ├── lidarseg/        # 保存原完整数据；当前检测任务是否使用另依入口
│   └── offset_maps/     # 10,000个原有生成.npy
└── v2xsim2_info_generated/
    ├── v2xsim_infos_train.pkl
    ├── v2xsim_infos_val.pkl
    ├── v2xsim_infos_test.pkl
    └── 原生成脚本与日志...
```

六ZIP实际检查到的顶层分别为`sweeps/`、`maps/`、`v2.0/`、`lidarseg/`，**没有外层`V2X-Sim-2.0/`**。它们应解到`/workspace/datasets/V2X-Sim-2.0`。offset tar顶层为`offset_maps/`，也解到同一位置。不要多解一层，造成`.../V2X-Sim-2.0/V2X-Sim-2.0/...`。

元数据PKL内可能引用原绝对路径，保留`/workspace/datasets/...`最省事。如果平台挂载点不同，优先按原布局挂载/建立经验证的映射；不要批量改写历史PKL、改Hash或无审计地重新生成。只读目录符号链接可以作为适配方案，但恢复工具为防路径逃逸会拒绝某些链接目标，先确认映射再选择恢复路径。

### 5.2 先上传归档，后解压

把六ZIP和完整offset tar上传到`/workspace/restore_uploads/dataset_archives`。无需同时上传旧`offset_maps_20260902.tar.part-*`和完整tar两套；已有分片时按原校验清单重组出完整tar并核对Hash。

```bash
cd /workspace/restore_uploads/dataset_archives
sha256sum -c /workspace/LRCP/recovery/handoff_20261118/DATASET_TRANSPORT_SHA256SUMS.txt
cd /workspace/LRCP
python3 recovery/handoff_20261118/restore_dataset.py \
  --uploads /workspace/restore_uploads/dataset_archives \
  --metadata /workspace/restore_uploads/handoff_kit/PRIVATE_DATASET_METADATA_20261118.tar.gz
# 传输SHA与布局dry-run通过后恢复
python3 recovery/handoff_20261118/restore_dataset.py \
  --uploads /workspace/restore_uploads/dataset_archives \
  --metadata /workspace/restore_uploads/handoff_kit/PRIVATE_DATASET_METADATA_20261118.tar.gz \
  --apply
```

脚本先核对所有传输SHA和归档路径，再解包；不加载pickle，不重新生成数据，不覆盖不同内容。解包要消耗压缩归档+展开数据+临时/日志空间，建议工作盘500 GiB而不是刚好够放45 GiB ZIP。当前数据展开约126 GiB，归档与offset包会额外占空间。先核验恢复结果并确认实例外副本，再由用户决定是否清理上传归档；工具不自动删除。

若暂时只需3个PKL，可加`--metadata-only`恢复索引；这不意味着原始点云和offset已恢复。CPU离线R5D本身不需要完整数据集。

恢复后核验3个PKL的SHA（原值）：

```text
train 963a66da43dcd3b9e4e89987a0e671b3e7ffd2f139c93f9ca06087f2e1a5d8c5
val   8c59b916e41757da90017253ca0da742a9ee7607e516ce4e4fe989f1baf727b7
test  96c3dd51bca68947d57eff94cda2da2c811c1301e510b8cc3e1dfec50adcbd16
```

还需确认offset数量10,000、数组字节总计37,505,280,000；后续数据读取审计可验证PKL路径与若干文件内容。归档Hash/数量检查通过不能替代完整样本语义审计。不要把生成脚本重新跑一遍当作旧索引/offset的身份恢复。

### 5.3 如果旧实例和电脑都没有原始ZIP

可从[V2X-Sim官方项目](https://github.com/ai4ce/V2X-Sim#dataset)找原始数据入口。目前官方README链接到原始数据镜像和预处理数据，务必区分。LRCP当前使用的是本表布局的原始数据+自己的PKL/offset，不能把其他任务预处理版当作同一数据集。重新下载的包可能重打包、Hash不同，应记录来源并做内容对齐，不能改本清单强行通过。官方原始数据也不会替你恢复本项目生成的PKL、offset和完整实验历史。

## 6. 后续要恢复PPU模型环境时

**先完成CPU恢复，再做这一节。** 当前没有授权新模型forward/训练；下面安装/导入步骤不等于自动开始下阶段研究。

### 6.1 相同供应商镜像优先

```bash
source /usr/local/PPU_SDK/envsetup.sh
python3.10 -m venv --system-site-packages /workspace/venvs/lrcp
source /workspace/venvs/lrcp/bin/activate
export PYTHONPATH="/workspace/LRCP:${PYTHONPATH:-}"
```

先用`importlib.metadata`核对版本、SDK文本和架构；不要先升级torch或照旧environment.yml安装。可将工具包中的三个固定CPU依赖安装到这个专用venv，保持原NumPy/Shapely/PyYAML口径，不替换供应商torch/SDK：

```bash
/workspace/venvs/lrcp/bin/python -m pip install --no-index \
  --find-links /workspace/restore_uploads/handoff_kit/cpu_wheels \
  -r /workspace/LRCP/recovery/handoff_20261118/requirements.cpu-audit.txt
```

工具包`PRIVATE_PPU_COMPAT_20261118.tar.gz`保存原`spconv1.2.1`wheel、5个项目扩展、MultiScaleDeformableAttention扩展及兼容要求。**它不含供应商torch、驱动或PPU SDK。** 仅同Python3.10/x86_64/PPU SDK/供应商torch与ABI环境可考虑复用：

```bash
/workspace/venvs/lrcp/bin/python -B /workspace/LRCP/recovery/handoff_20261118/install_ppu_compat.py \
  --archive /workspace/restore_uploads/handoff_kit/PRIVATE_PPU_COMPAT_20261118.tar.gz
# 检查同平台、同镜像及dry-run结果后实际安装
/workspace/venvs/lrcp/bin/python -B /workspace/LRCP/recovery/handoff_20261118/install_ppu_compat.py \
  --archive /workspace/restore_uploads/handoff_kit/PRIVATE_PPU_COMPAT_20261118.tar.gz --apply
```

工具核对包SHA、Python/架构、SDK、torch/numpy版本和torch源码指纹，向专用venv安装spconv，复制已核对的扩展；不安装torch、驱动或SDK，不调用设备。版本/指纹一致仍不是ABI或算子执行的完整证明，后续还需单独导入/算子验收。不匹配时停止此捷径，从源码重建，不通过删除兼容检查来强装。

### 6.2 二进制不匹配时从源码重建

spconv源码固定`fad3000249d27ca918f2655ff73c41f39b0f3127`，按备份补丁恢复4处PPU适配。子模块的固定版本见`recovery/instance_20261117/README.md`。原构建记录为Release、PPU SDK提供的`/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc`，不是默认NVIDIA工具链。

一般顺序为：同平台SDK环境 → 固定spconv源码/子模块与PPU补丁 → 构建wheel → LRCP本地扩展。记录编译日志，控制并发（如`MAX_JOBS=2`）防止编译内存激增。下面是待新实例验证的构建入口，不能视作已在新实例通过：

```bash
# 在固定且已应用PPU补丁的spconv目录
MAX_JOBS=2 /workspace/venvs/lrcp/bin/python setup.py bdist_wheel
# 在/workspace/LRCP/opencood/utils/ms_deform_attn_ops目录
MAX_JOBS=2 /workspace/venvs/lrcp/bin/python setup.py build install
# 下列两个setup.py按项目根目录拼接源码路径，必须从/workspace/LRCP执行
cd /workspace/LRCP
MAX_JOBS=2 /workspace/venvs/lrcp/bin/python opencood/pcdet_utils/setup.py build_ext --inplace
/workspace/venvs/lrcp/bin/python opencood/utils/setup.py build_ext --inplace
```

不要用CUDA接口名称推断硬件，不要把旧`.so`复制到不匹配的NVIDIA/其他Python环境，也不要用`spconv2.x`替代1.2.1后宣称原环境相同。安装完整Python依赖时以原环境清单逐项补缺，保留平台核心包；不要全量强制重装325个观察到的包。

完成数据和依赖恢复后可运行：

```bash
/workspace/venvs/lrcp/bin/python -B /workspace/LRCP/recovery/handoff_20261118/preflight.py --mode model
```

此命令只检查元数据/文件/共享内存，不导入torch或执行算子。需要检查实际设备/导入/forward时，下一位Codex应明确区分“安装检查”“设备/算子检查”和“真实模型forward”，遵守新任务的预算与授权。恢复环境不消耗科学forward账本。

## 7. Codex接手要知道的研究状态

- 最近完成：`RD3-R5D`，正式重执行目录为`/workspace/lrcp_d2/codex_reports/RD3_R5D_GEOMETRY_SCORE_COUNTERFACTUAL_v2_20261116/`，状态`R5D_PASS`。原R5D目录保留为历史，不覆盖合并。
- 主要入口：`RD3_R5D_SCIENTIFIC_ASSESSMENT.md`、`R5D_RESULT.json`、`R5D_INDEPENDENT_VALIDATION.json`。旧实例主独立验收41,023项通过；便携包1,202项检查是另一种范围，不能混为同一验收。
- R5D只做16条2×2和3条Score交换CPU重放，从完整12,800候选开始。没有新forward，没有开启下一阶段。候选替换不识别Attention、MetaFlow、AoI或通信网络的独立因果效应。
- 原始真实模型基线为R5A/B/C已验收的32条结果，使用`RD3_R5B_SAMPLE_UTILITY/R5B_RAW_SHARD_LOCATOR.csv`定位；GT身份用R5A映射；输入用`RD3_MESSAGE_UTILITY_PILOT/FROZEN_MANIFEST.json`。不能为恢复而重跑32次forward。
- 指定权重是`/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/net_epoch_bestval_at19.pth`，SHA256 `666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724`。同目录普通epoch19、E1或敏感性权重都不能代替。
- 使用同目录实际冻结`config.yaml`及R5C合同。当前sens分支的训练学习率配置、eval分支paper-style后处理不能替换冻结实验配置。源码保全不等于更改既有科学协议。
- 原生流程：sigmoid后严格Score>0.2；无iou_preds质量重标定；保留历史large/z过滤；原生top1000；BEV IoU>0.15 NMS；最后range；Score降序贪心唯一GT匹配，阈值≥0.5/≥0.7。沿用float32、原生排序与既有容差，不“修复”历史large-box细节。
- 真实运行budget/claim/manifest保留原值，不重置attempt，不将mock账本当真实账本。`.pid`和锁持有者状态不代表新实例仍有旧进程，不据旧PID直接kill或启动重复任务。
- 原训练主配置与纸面学习率敏感性配置不同；时间编码内部`time_index*100`与V2X-Sim5 Hz物理报告刻度200 ms不同，不能混用。需要时读原冻结说明，不凭旧Notebook标题推断。
- 最低包保存了指定检测权重，不保证含完整优化器、scheduler、随机数和所有epoch状态；不能据此宣称训练可从任意旧step精确续跑。

### 给新实例Codex的首条消息，可直接复制

```text
请先读取 /workspace/LRCP/HANDOFF.md。
这是LRCP迁移恢复任务，先只做文件恢复、环境安装检查和CPU离线验收；不要启动新的模型forward、CUDA初始化、训练或后续研究。
GitHub备份分支是 preserve/instance-20261117，科学源码保全基线e12c82e。
最低续研包和PRIVATE_HANDOFF_KIT位于 /workspace/restore_uploads；请先核对SHA，再dry-run并恢复最低包的外层和内部文本。
请使用恢复工具建立独立CPU环境，检查解释器、/dev/shm、冻结GT/候选来源与指定checkpoint，复核R5D便携包19条重放。
先报告：代码、最低证据、完整历史、数据集、CPU环境、PPU环境分别恢复到什么程度；无法验证的项列为缺失，不修改历史Hash或阈值强行通过。
如果已上传数据集归档，再按Handoff恢复六ZIP、offset_maps和原PKL；不要重新生成旧实验数据。
环境恢复和CPU验收结束后停止，等待下一阶段的具体研究指令。
```

## 8. 最终验收与停止条件

分别记录以下状态，不能笼统写“恢复成功”：

1. **代码**：新分支能克隆，冻结源码与e12c82e相比未改变；新Handoff提交可单独追溯。
2. **最低证据**：外层与内部文本Hash正确，32条原始NPZ、8冻结输入、指定权重及R5D包存在；便携CPU重放PASS。
3. **完整历史**：所有历史封存引用文件已恢复才可执行服务器全历史验收。少了旧捕获/早期batch应报告缺失，不能把便携PASS替代它。
4. **数据集**：七个运输归档Hash正确、布局正确、三PKL原Hash正确、offset数量/字节正确；实际读取语义审计另列。
5. **CPU环境**：专用解释器及3依赖正确，无torch导入/设备初始化。
6. **PPU环境**：同平台镜像/SDK/版本/扩展已恢复；元数据检查与实际算子/forward验收分开，后者需明确新授权。
7. **实例外备份**：电脑/持久存储保存的包已独立核验。GitHub不含大型私有包；上传到新实例也不应删除最后一份外部备份。

本轮不会实际替用户点击平台创建实例，也未验证新实例的计费套餐、镜像ID、持久卷删除策略或SSH入口。网页不可读的具体UI项保留为人工确认，其他步骤已按现有文件/版本/归档结构写明。

## 9. 本版工具验证记录

在旧实例新建临时、隔离的Python3.10 CPU环境，从工具包wheel离线安装3个固定依赖，完成R5D全部19条保存候选重放（1,202项便携检查通过）。恢复工具另有47项检查通过，涵盖分片合并、损坏/缺片拒绝、不同索引拒绝覆盖、补充归档逐成员Hash、最低包外层+内部文本dry-run和PPU扩展恢复dry-run。没有创建新平台实例，没有实际覆盖完整数据集，没有验证PPU算子执行，没有新模型forward/CUDA初始化。实际六ZIP、offset归档和PKL补充包还通过了运输SHA与路径布局dry-run，未实际解压整个数据集。验证回执见[HANDOFF_TOOL_VALIDATION.json](recovery/handoff_20261118/HANDOFF_TOOL_VALIDATION.json)。
