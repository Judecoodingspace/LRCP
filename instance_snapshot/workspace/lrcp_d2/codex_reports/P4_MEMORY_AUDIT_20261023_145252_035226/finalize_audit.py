"""Render memory audit and verify protected files; no replay or model execution."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,ast,csv,subprocess
R=Path(__file__).resolve().parent
P=R.parent/'P4_PILOT_20261022_193114_346841'
j=json.loads((R/'ENVIRONMENT_SNAPSHOT.json').read_text())
probe=json.loads((R/'TINY_CPU_PROBE.json').read_text());assert probe['passed']
protected=json.loads((R/'PROTECTED_BEFORE.json.updated').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
changed=[p for p,d in protected.items() if not Path(p).is_file() or sha(p)!=d]
assert not changed,changed
calls=[];risky=[]
paths=[P/n for n in ['runner.py','runner_revision_candidate.py','pilot_common.py','collector.py','bounded_offset_cache_proposal.py']]
for path in paths:
 tree=ast.parse(path.read_text())
 for n in ast.walk(tree):
  if isinstance(n,ast.Call):
   f=ast.unparse(n.func)
   if f.endswith(('DataLoader','torch.save','torch.load','np.load','clone','clone()')) or f in ['subprocess.run']:
    calls.append({'file':str(path),'line':n.lineno,'call':f,'keywords':{k.arg:ast.unparse(k.value) for k in n.keywords}})
   if f.split('.')[-1] in ['share_memory_','share_memory','SharedMemory','_new_shared','_share_fd_cpu_','_share_filename_cpu_','Process','Pool','Queue','spawn','fork','mmap']:
    risky.append({'file':str(path),'line':n.lineno,'call':f})
assert not risky,risky
(R/'CALL_AUDIT.json').write_text(json.dumps({'checked_files':[str(p) for p in paths],'memory_related_calls':calls,'explicit_shared_memory_or_multiprocessing_launch_calls':risky,'scope':'Static direct calls only; not proof of every native dependency syscall.'},ensure_ascii=False,indent=2)+'\n')
for p in R.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
GiB=2**30;TiB=2**40
limit=j['ordinary_memory_limit_bytes'];usage=j['cgroup_memory_usage_bytes'];stat=j['cgroup_stat'];disk=j['workspace'];shm=j['shm']
checks={'protected_files_unchanged':not changed,'old_runner_still_stopped':not j['old_runner_pid_4025436_exists'] and j['old_runner_state']['status']=='STOPPED','cgroup_v1_identified':j['cgroup_version']==1,'shared_tmpfs_exact_64MiB':shm['total_bytes']==64*2**20,'workspace_not_tmpfs':disk['filesystem']=='overlay','single_process_tiny_probe_passed':probe['passed'],'no_explicit_shared_IPC_calls_in_runner':not risky,'no_PPU_initialization_in_probe':probe['checks']['CUDA_not_initialized'],'no_historical_replay':not probe['real_dataset_replay'],'RSS_limit_not_changed':"8*2**30" in (P/'pilot_common.py').read_text()}
assert all(checks.values())
summary={'audit_complete':True,'pilot_retry_paused':True,'snapshot_Asia_Shanghai':j['snapshot_Asia_Shanghai'],'ordinary_memory_cgroup_version':1,'ordinary_memory_hard_limit_bytes':limit,'ordinary_memory_cgroup_current_usage_bytes':usage,'ordinary_memory_cgroup_cache_bytes':stat['cache'],'ordinary_memory_cgroup_RSS_bytes':stat['rss'],'shared_memory_total_bytes':shm['total_bytes'],'shared_memory_used_bytes':shm['used_bytes'],'shared_memory_logical_file_bytes':shm['logical_files_total_bytes'],'shared_memory_allocated_file_bytes':shm['allocated_files_total_bytes'],'workspace_filesystem':'overlay','workspace_free_bytes':disk['free_bytes'],'direct_evidence_linking_old_failure_to_shm':False,'old_failure_direct_cause':'Explicit process peak-RSS guard >8GiB','shared_memory_blocks_current_single_process_disk_cache_plan':False,'memory_fix_compatible_with_64MiB_shm':True,'real_18_sample_memory_bound_or_equivalence_validated':False,'synthetic_CPU_checks':len(probe['checks']),'protected_files_unchanged':len(protected),'PPU_model_forwards_this_audit':0,'historical_replay_this_audit':False,'container_configuration_changed':False}
(R/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(R/'VERIFICATION.json').write_text(json.dumps({'passed':True,'checks':checks,'protected_file_count':len(protected),'changed_files':changed,'tiny_probe_checks':probe['checks'],'bounded_offset_proposal_modified':False,'existing_runner_modified':False},ensure_ascii=False,indent=2)+'\n')
(R/'NEXT_CPU_STRATEGY.md').write_text('''# 下一次受控 CPU 重试策略（本次未执行）

重试继续暂停。需要用户明确解除暂停并批准一次新的CPU重试；不改共享内存大小，不重启容器，不提高8GiB RSS保护。

1. 全新P4_PILOT目录沿用18样本/历史seed303、delay[0,1,2,3]、原读取/随机shuffle/过滤顺序。num_workers=0、batch_size=1、shuffle=False、pin_memory=False；不加预取线程/worker，不经multiprocessing Queue/Pool/共享tensor。库内部CPU线程不等于数据worker；不为省内存随意修改线程配置或原算法。
2. 采用已审阅BoundedOffsetDataset代理，仅对持久scene_database offset视图clone成同值小tensor；原生getitem恰好一次，当前返回item不改。验证shape/dtype/device/value以及RNG不变。确认实际DataLoader未通过__getitems__批量捷径绕过代理；现有LRCP路径没有该捷径。
3. 记录逻辑tensor大小和去重后的底层storage大小。released_storage_bytes是解除持久引用的估计，当前batch仍可能持有旧storage，不能当作即时RSS下降。逐batch释放不再需要的引用；每个delay结束释放旧loader/dataset/iterator引用，不把4份dataset同时保留。只做与数值/RNG无关的对象生命周期管理，仍需历史等价验收。
4. 每个所选样本在同一次原预处理结果上落盘一份冻结batch，不将18份batch全部常驻RAM。保存到该新目录的batches/sample_*.pt.tmp，再原子rename；不使用/dev/shm、POSIX SharedMemory或共享tensor搬运。archive总计≤18GiB、每batch≤1GiB，预计大小优先按unique storage估算，torch.save可能保存视图的整块storage。
5. 后续仅在获准阶段逐样本读取，torch.load(map_location='cpu',weights_only=False,mmap=False)，仅处理本实验自行写入并已核验SHA256的归档，恢复其RNG状态。同一已处理batch的两个α隔离可写派生状态，不重新随机预处理。明确weights_only=False是为NumPy RNG状态兼容，不是共享内存开关。
6. 保留ru_maxrss峰值>8GiB即停止；补充每batch当前VmRSS、峰值、sample_idx/delay、cache storage计数；约6GiB预警但不调高硬限。同步记录cgroup v1 usage/stat(cache/rss/shmem)、/dev/shm容量/用量，区分应用常驻与page cache。绝不清空系统page cache或删除已有共享文件。
7. 预检磁盘≥25GiB并保留5GiB余量，CPU重放≤30分钟；任何资源/等价性失败停止、不自动重试。真实18样本是否低于8GiB尚待验证。磁盘文件读写会进入cgroup文件缓存，不能把“落盘”当作完全不占内存。
8. 本次不恢复PPU。解除暂停后仍须先完成CPU门和sample821既有smoke门；此前有限PPU授权不会被本审计扩大。共享内存扩容/容器重启/改workers/RSS限额均不在方案中，也无必要为本单进程路径申请。
''')
report=f'''# P4-Pilot：64 MiB共享内存环境审计

**审计完成，重试继续暂停。64 MiB是/dev/shm这个tmpfs挂载的容量，不是普通CPU内存、进程RSS、实验目录容量或PPU显存上限。当前单进程DataLoader+普通磁盘冻结batch方案没有已知的64 MiB共享内存阻碍；仍未证明完整18样本CPU重放低于8GiB。**

快照时间：{j['snapshot_Asia_Shanghai']}。本次未执行历史数据重放、模型前向、PPU初始化、容器重启、共享内存扩容或RSS上限变更。旧runner已退出，其STOPPED状态和0次forward记录保持不变。

## 真实容量与关键命令

| 对象 | 观测值 | 正确含义 |
|---|---:|---|
| /dev/shm | tmpfs，总67,108,864字节=64MiB；使用0、可用64MiB | 该挂载的共享内存容量 |
| 普通内存硬限 | {limit:,}字节={limit/GiB:.0f}GiB | cgroup v1 memory.limit_in_bytes，不是64MiB |
| cgroup当前用量 | {usage:,}字节≈{usage/GiB:.3f}GiB | 多进程/页缓存等容器计费，不能等同当前runner RSS |
| cgroup cache / rss / shmem | {stat['cache']/GiB:.3f} / {stat['rss']/GiB:.3f} / {stat['shmem']/GiB:.3f}GiB | 当前用量主要是文件缓存；memory.stat与usage读取有时间差且usage本身为近似计数 |
| limit−usage | {j['cgroup_nominal_headroom_before_reclaim_bytes']/GiB:.3f}GiB | 名义计费余量，非保证可分配空间；缓存可回收、其他进程也可变化 |
| /workspace/lrcp_d2 | overlay；总{disk['total_bytes']:,}字节≈{disk['total_bytes']/TiB:.3f}TiB；可用{disk['free_bytes']:,}字节≈{disk['free_bytes']/TiB:.3f}TiB | 位于根overlay可写文件系统，不是/dev/shm或64MiB tmpfs；宿主底层具体介质未由此证明 |
| /proc/meminfo MemTotal | {j['host_meminfo_bytes']['MemTotal']/TiB:.3f}TiB | 主机视角，不能作为容器可用内存承诺 |
| P4进程RSS保护 | 8GiB，检查ru_maxrss | runner应用层峰值保护，未改变 |
| PPU显存 | 每卡98304MiB=96GiB；只读快照每卡使用2MiB、GPU利用率0% | 独立设备内存，不是CPU共享内存；查询没有启动推理 |

实际是cgroup **v1**：/proc/self/cgroup有memory控制器，/sys/fs/cgroup/memory挂载类型为cgroup。v2的memory.max、memory.current不存在，应读取其v1对应项：

```text
findmnt -T /dev/shm
/dev/shm  shm  tmpfs  64M  used=0  avail=64M  size=65536k

cat /sys/fs/cgroup/memory/memory.limit_in_bytes
{limit}
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
{usage}

findmnt -T /workspace/lrcp_d2
/  overlay  overlay
```

完整未压缩命令输出在COMMAND_OUTPUTS.txt/JSON，原始mountinfo、cgroup、meminfo、limits亦已保存。当前memory.stat中的hierarchical_memory_limit同为256GiB；更外层不可见的资源竞争不能由此排除。swap限额与memory限额相同，快照swap为0，不把swap当可用缓冲。

memory.max_usage_in_bytes={j['cgroup_v1']['memory.max_usage_in_bytes']}、memory.failcnt={j['cgroup_v1']['memory.failcnt']}是容器生命周期历史值；本次oom_control显示under_oom=0、oom_kill=0。缺少上次退出时同步快照，不能把这些历史计数归因于P4或64MiB shm。

## RSS / cgroup / shm / 显存的区别

- RSS：一个进程驻留的物理页，包括匿名堆和驻留文件映射等；ru_maxrss是该进程历史峰值，释放对象后也不会降低历史峰值。上次runner正是这个峰值越8GiB触发Python异常。
- cgroup内存：容器进程及其页缓存、shmem等的内核计费；不是把进程RSS简单相加。普通磁盘读写进入page cache，也可能增加cgroup计费。
- /dev/shm：特定tmpfs挂载；POSIX共享对象通常在此分配。tmpfs页也会计入cgroup，不能理解为完全独立的额外内存池。memfd、System V shm或其他挂载不必受这个/dev/shm容量控制。
- PPU显存：设备侧容量。CPU pinned memory仍是主机内存，不是显存，也不等于/dev/shm；当前pin_memory=False。/proc/self/limits的Max locked memory=65536字节=64KiB，亦与64MiB /dev/shm不同。

## /dev/shm旧文件

观察到10个torch_*对象，逻辑大小合计{shm['logical_files_total_bytes']:,}字节≈{shm['logical_files_total_bytes']/2**20:.3f}MiB，但每个st_blocks为0，分配总量0；这与df使用0一致。逻辑st_size不是已分配页，不能相加后声称64MiB已爆满。mtime位于2026-09-01/02，早于2026-10-22本轮P4运行；不能从文件名或存在性认定为P4创建。

扫描可访问/proc进程fd未找到当前持有这些shm/memfd/SYSV对象的fd。这是当前可见范围，不足以证明没有已关闭fd但仍映射的其他对象；微型测试进程自己的maps没有共享内存映射。没有删除任何旧共享文件，也没有进行64MiB容量压测。

## 当前runner和依赖是否使用共享内存

| 路径 | 已核对行为 | 64MiB相关性 |
|---|---|---|
| runner DataLoader | num_workers=0、pin_memory=False、shuffle=False、B=1；真实安装torch dataloader.py:418返回_SingleProcessDataLoaderIter，744断言workers=0 | 不启动tensor worker/IPC队列，不把冻结batch放进shm |
| torch multiprocessing | torch导入时可加载该模块；其reductions有_share_fd_cpu_/_share_filename_cpu_功能 | 模块可用不等于使用。当前runner没有Process/Pool/Queue/share_memory_调用；如未来改workers>0或通过队列传tensor会进入共享storage路径 |
| 默认collate | 安装源码collate.py:266仅在get_worker_info()!=None时分配_new_shared | 主进程worker_info=None；LRCP自定义collate使用NumPy/torch常规拼接 |
| torch.save | serialization.py:1216将CPU storage写入目标ZIP记录；runner写入本目录batches/*.pt.tmp再rename | 普通overlay文件写入，不使用/dev/shm临时文件；磁盘页缓存可能增加cgroup用量 |
| torch.load | 安装源码1417–1418中mmap未指定时为False；map_location=cpu。旧runner还没走到此步 | 本次不启用mmap；weights_only=False只是自建NumPy RNG归档的兼容开关，非共享内存开关 |
| NumPy / torch.from_numpy | 运行路径np.load(offset_path)未指定mmap_mode，numpy_format.py:790为普通fromfile；from_numpy在同进程共享NumPy缓冲区 | 这是进程内别名，不是跨进程POSIX共享内存；完整数组被视图保留会增加普通RSS |
| 审计代码中的np.load(mmap_mode=r) | 仅先前只读NPY元数据/别名核查使用；不是P4 runner的数据读取路径 | 磁盘文件映射不自动占/dev/shm，驻留页仍可计入RSS/cgroup |
| spconv VoxelGeneratorV2 | 已安装spconv utils源码用np.full坐标表并经generate→points_to_voxel；当前SpVoxelPreprocessor调用generate | Python路径未见IPC/shm创建。环境没有可发现的cumm包；未对本地C++扩展做全syscall追踪，不宣称其所有内部行为均已穷尽 |
| 子进程 | runner中的subprocess.run只执行资源查询命令，经普通pipe传文本 | 是短命令子进程，不是数据worker，不传大tensor；未见隐式历史数据多进程 |
| CPU线程 | 微型进程见OS线程{probe['OS_threads']}；torch线程设置{probe['torch_CPU_threads']}/{probe['torch_interop_threads']} | 线程共享同进程地址空间不等于POSIX shm。本次不改线程设置，以免影响既定轨迹/数值条件 |

实际安装torch路径为/usr/local/lib/python3.10/site-packages/torch，版本{probe['torch_version']}；NumPy为{probe['numpy_version']}。venv的sysconfig纯Python目录不是这些包的真实来源，审计使用find_spec定位并保存源码行号/哈希，未假定安装位置。SOURCE_EVIDENCE.txt和CALL_AUDIT.json提供追溯。

不能用静态搜索证明每个native依赖绝无共享内存调用。完整18样本及PPU后端未在本次运行，依赖的隐藏IPC需求仍应在恢复后的既有smoke门观察；当前没有需要扩容shm的证据。

## 微型单进程CPU验证

仅使用合成数组，总payload<128KiB，没有导入LRCP或调用真实dataset。20项检查通过：

- 真实DataLoader迭代器为_SingleProcessDataLoaderIter，worker_info=None；无新增子进程。
- 原tensor、offset.clone、DataLoader结果及磁盘加载结果的is_shared()均为False。
- offset底层storage从40960字节压缩到512字节，数值/返回item不变，clone不消耗NumPy/torch RNG。
- 保存5213字节tiny_frozen_batch.pt到本overlay目录，torch.load(...,mmap=False)数值一致；普通NumPy load结果不是memmap。
- /dev/shm前后used均0，无新增文件，无本进程shm/memfd映射；CUDA未初始化。
- 此微型进程峰值RSS为{probe['peak_RSS_bytes']:,}字节≈{probe['peak_RSS_bytes']/2**20:.3f}MiB，在8GiB内。其RSS本来大于64MiB但shm仍0，不能把两种限制混为一谈。

这些结果支持已测小路径与64MiB配置兼容，不代表真实18样本的峰值内存或历史等价已验收。原微型代码、日志、机器可读检查均已保存；测试小文件保留为审计产物，不写到/dev/shm。

## 上次8.014GiB失败的原因

FAILURE.json记录的明确触发点是pilot_common.py:72：ru_maxrss*1024>8*2**30，随后抛CPU_RSS_LIMIT_EXCEEDED。峰值8,605,294,592字节，超限15,360,000字节。不是观察到SIGBUS、ENOSPC或内核OOM kill后反推；上次共享内存使用曲线没有采集，**不存在证明它与64MiB共享内存限制直接相关的证据**。

最有源码依据的内存累积来源是V2XSIMBaseDataset:211的params字典别名，加237/241把完整多agent offset_map的视图存回持久scene_database。sample553的小切片51,200字节会保留4,096,000字节数组，sample821则保留6,400,000字节，放大80/125倍。delay0存zeros，delay1改存视图，与失败阶段一致。它涉及普通进程内存保留；未有堆快照，不能认定唯一来源或量化其峰值贡献。

## BoundedOffsetDataset审阅与下一步

**可以沿用其修复方向，并与64MiB shm兼容；不能保证单靠它就完成18样本。** 它只在主进程将持久offset视图clone为相同CPU tensor，不调用share_memory_、multiprocessing、shm_open或mmap；当前返回item不变、原生getitem一次。旧合成8项及本次20项检查提供数值/RNG/storage证据。本次未修改或重新执行旧runner。

注意clone瞬间旧view和新tensor可同时存在，当前batch仍可持有旧storage，Python/CPU allocator也未必立即把页还给OS；released_storage_bytes不是RSS即时减少量。torch.save会保留storage别名语义，逻辑tensor大小小于整个底层storage时可能写入更大数据，下一轮应按unique storage估算缓存与序列化开销。代理的__getattr__未来可能暴露原dataset.__getitems__批量捷径，当前LRCP无该捷径，但下轮应断言没有绕过代理。

下一轮策略详见NEXT_CPU_STRATEGY.md：单进程原顺序准备、compact持久offset、每样本一份磁盘归档、逐样本加载，不把18份batch常驻；保留8GiB峰值硬保护，增加当前RSS/cgroup/shm同步日志，原seed/样本/数据/模型协议不变。磁盘page cache仍需监测；不通过改sharing strategy、workers、精度或扩容tmpfs掩盖普通RSS问题。

需要额外授权的只有恢复真实CPU重试：当前用户明确要求暂停，故本次不启动。共享内存扩容、容器重启、提高8GiB限额均不需要也不申请；此前PPU授权不因本审计被扩大或自动恢复执行。任何后续真实重放仍需新的明确恢复指令及原验收门。

## 交付及保护

ENVIRONMENT_SNAPSHOT.json、COMMAND_OUTPUTS.txt/JSON、SOURCE_EVIDENCE.txt、CALL_AUDIT.json、TINY_CPU_PROBE.json、SUMMARY.json、VERIFICATION.json保存证据。{len(protected)}个受保护文件前后哈希一致，旧P4 runner/修复草案、checkpoint、源码和历史产物未修改。审计代码和本目录文件由ARTIFACTS.sha256封存。P4-Pilot重试继续暂停。
'''
(R/'REPORT.md').write_text(report)
(R/'COMPLETION.json').write_text(json.dumps({'phase':'P4 memory/shm audit','audit_complete':True,'pilot_retry_paused':True,'historical_replay_executed':False,'PPU_model_inference_executed':False,'container_changes':False,'protected_files_unchanged':len(protected),'report':str(R/'REPORT.md'),'completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()},ensure_ascii=False,indent=2)+'\n')
files=sorted(p for p in R.iterdir() if p.is_file() and p.name!='ARTIFACTS.sha256')
with (R/'ARTIFACTS.sha256').open('x') as f:
 for p in files:f.write(sha(p)+'  '+p.name+'\n')
for line in (R/'ARTIFACTS.sha256').read_text().splitlines():
 d,n=line.split('  ',1);assert sha(R/n)==d
print(json.dumps({'summary':summary,'protected_files':len(protected),'sealed_files':len(files)},ensure_ascii=False))
