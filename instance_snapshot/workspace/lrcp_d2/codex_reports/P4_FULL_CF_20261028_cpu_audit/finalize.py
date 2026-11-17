import json,csv,hashlib,collections,time
from pathlib import Path
R=Path(__file__).resolve().parent
j=lambda n:json.loads((R/n).read_text())
a=j('INPUT_AUDIT.json');p=j('PILOT_GATE.json');b=j('baseline_validation.json');m=j('factorial_metrics.json');i=j('independent_validation.json');s=j('summary.json')
assert all(x['passed'] for x in [a,p,b,m,i,s]) and a['conditions']==1820 and a['inputs']==910 and p['cases']==72 and b['cases']==m['cases']==i['cases']==1820
rows=list(csv.DictReader((R/'per_GT_mixed.csv').open()));assert len(rows)==16328
assert len({(x['sample_idx'],x['gt_idx'],x['geometry']) for x in rows})==16328
for g in ('0','1'):
 rr=[x for x in rows if x['geometry']==g];assert len(rr)==8164
 assert sum(int(x['native_TP70']) for x in rr)==b['metrics'][f'G{g}_70']['TP']
 assert sum(int(x['mixed_TP70']) for x in rr)==m['metrics'][f'G{g}_70']['TP']
 assert sum(int(x['top1000_truncated'])>0 for x in csv.DictReader((R/'case_metrics.csv').open()))==0
for x in rows:
 nums=[int(x[k]) for k in ['raw','score','quality','top1000','nms','range']];assert all(u>=v for u,v in zip(nums,nums[1:])),x
assert not (R/'FAILURE.json').exists() and not (R/'INDEPENDENT_FAILURE.json').exists()
(R/'validation.json').write_text(json.dumps({'passed':True,'input_hashes_passed':True,'pilot_72_exact':True,'native_1820_exact':True,'mixed_1820_independently_replayed':True,'GT_rows':len(rows),'GT_identity_unique':True,'GT_stage_monotonic':True,'top1000_truncation_cases':0,'peak_RSS_bytes':max(a['peak_rss_bytes'],b['peak_RSS_bytes'],m['peak_RSS_bytes'],i['CPU_RSS_peak']),'model_forwards':0,'CUDA_initialized':False},indent=2))
report='''# P4-Full-CF：全量 NMS 排序来源反事实

**结论：固定 G1 几何和原生准入、将 NMS 排序从 S1 换为 S0 后，205 个 NMS 阶段损失 GT 中恢复 69 个；全体 GT 同时新增 36 个 TP70→FN70，净增加 63 个 TP70。** 这是固定 head 输出下的人为后处理反事实，不能直接当作模型真实性能改善或证明 MetaFlow offset 的对错。

## 基线和验收

910样本、8164 GT、两种α共1820条件。原始1820采集及元数据和910冻结输入哈希通过；18样本72条件与P4-C2的数组、抑制者及匹配完全一致。再重算原生1820条件，框和score分别以1e-5与1e-6容差核对，score/quality/top1000/NMS/range索引和两阈值GT匹配一致；四个AP精确等于P3-A。混合1820条件完成后，另以完整采集框、原生准入ID独立重放NMS、range并重新匹配GT，两个阈值的AP、TP、FP一致至1e-12。主实验与验收均未初始化CUDA、forward为0；RSS峰值767 MiB。score/quality候选、几何、GT在混合条件固定；两个混合条件均无top1000截断。

| 条件 | AP50 | TP50 / FP50 | AP70 | TP70 / FP70 | 最终P70 | 最终R70 |
|---|---:|---:|---:|---:|---:|---:|
| G0A0R0 | 0.8665290893 | 7111 / 1064 | 0.8071592299 | 6754 / 1421 | 0.826177 | 0.827291 |
| G0A0R1 | 0.8706375145 | 7142 / 1032 | 0.8106398440 | 6770 / 1404 | 0.828236 | 0.829250 |
| G1A1R1 | 0.8768414234 | 7209 / 1329 | 0.7854230204 | 6645 / 1893 | 0.778285 | 0.813939 |
| G1A1R0 | 0.8771831117 | 7209 / 1361 | 0.7951724607 | 6708 / 1862 | 0.782730 | 0.821656 |

G1 AP70上升0.0097494403（0.9749个百分点），TP70净+63、FP70净−31；AP50略升但FP50增加32，表明TP变化不能替代完整PR分析。G0换S1排序亦上升：AP70 +0.0034806141，TP70净+16，而不是出现镜像下降。完整PR见global_PR.csv；原生PR与P3-B对应排序一致。

## 全部GT与原损失组

- G1A1R1→G1A1R0：FN→TP 99，TP→FN 36，TP→TP 6609，FN→FN 1420。99个恢复包括原278损失中的69个与原双FN中的30个；36个新退化包括原169恢复组的31个与原双TP的5个。
- 原278损失中，205个最后在NMS消失的GT有69个恢复、136个仍FN；64个score删尽及9个raw无严格候选均仍FN。这些属于固定G1、固定A1下的观察，并不意味着所有205个可通过排序恢复。
- 原169个α1恢复GT中，138个保持TP、31个变FN。
- G0A0R0→G0A0R1：FN→TP 48，TP→FN 32，TP→TP 6722，FN→FN 1362。48个恢复中40个属于历史α1恢复组、8个属于历史双FN；32个退化中29个属于历史α1损失组、3个为历史双TP。

完整六阶段迁移与逐GT结果分别见stage_migration.csv和per_GT_mixed.csv；逐anchor保留/抑制变化共15198行，包含两个排序源、抑制者、分数和重叠；GT严格候选抑制事件见GT_nms_events.csv。计数按GT去重，不能把同一anchor对多个GT的记录当作多个真实目标。

## 得分与定位质量

在每个GT的正IoU、已通过原生score和quality的候选集合中，G1的7382个非空GT中，原生排序分数最高候选也是最高IoU候选的有1839个，换S0排序后有1995个；G0分别为7280个非空GT中的1995和2138。此统计与真实物体身份无关，也不能保证更高IoU候选能通过所有后续步骤。对于G1原生NMS，739个严格候选事件中，抑制者的排序分数更高但对该GT的IoU<0.7，涉及392个去重GT；换排序后为647事件、330个GT。事件可以在GT间重复，属于质量与排名耦合的描述，不是可恢复目标的数量。

## 样本和场景稳健性

数据覆盖10个可从原始scene路径核验的scene。G1排序交换在85个sample净改善、30个净退化、795个净不变；10个scene中7个净改善、2个净退化、1个不变。G0为45个sample净改善、29个退化、836个不变；scene为6改善、3退化、1不变。统计是该数据集逐帧描述，10个scene不能支持对其他场景的独立总体推断，连续帧或同一车辆的GT不可当作独立样本。sample_summary.csv和scene_summary.csv提供完整分组。

## 解释与下一步

固定G和准入时，R替换改变实际NMS选择并可恢复部分严格检测，支持排序与几何质量耦合参与结果；新增退化和两种几何均可能改善说明这不是单一方向的普遍分数优劣。score已删尽与raw缺失的候选不可能靠仅替换NMS排序复活；不能据此区分这些缺失是上游融合、回归、分类筛选或其他耦合因素的根因。实验没有以GT排序，不搜索阈值；不能推断offset正确或错误，也不等于模型训练后或真实在线运行能直接得到这些分数。

P5_PROTOCOL.md和P5_SAMPLE_PROPOSAL.csv提出15个样本、10个scene的受控内部测量及单block移植方案，建议上限49次forward；仅为待审方案，需新的用户PPU授权、CPU READY验收并由用户亲自从终端启动。本次没有启动PPU。

## 产物

INPUT_AUDIT.json、PILOT_GATE.json、baseline_validation.json、factorial_metrics.json、independent_validation.json和validation.json记录逐层门禁；本目录Python脚本及日志可复现CPU处理。per_GT_mixed.csv、stage_migration.csv、nms_priority_changes.csv、GT_nms_events.csv、score_geometry_alignment.csv、global_PR.csv、sample_summary.csv、scene_summary.csv为证据。ARTIFACTS.sha256覆盖新目录产物，不覆盖或改写旧实验数据。
'''
(R/'REPORT.md').write_text(report)
files=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACTS.sha256' and '__pycache__' not in p.parts)
with (R/'ARTIFACTS.sha256').open('x') as f:
 for p in files:
  h=hashlib.sha256()
  with p.open('rb') as inp:
   for block in iter(lambda:inp.read(1024*1024),b''):h.update(block)
  f.write(h.hexdigest()+'  '+str(p.relative_to(R))+'\n')
print('validated',len(files),'artifacts')
