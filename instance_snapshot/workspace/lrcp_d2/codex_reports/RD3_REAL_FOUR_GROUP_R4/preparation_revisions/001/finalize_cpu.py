"""User-run CPU finalizer. Incomplete records yield NOT_ESTIMABLE, no metrics."""
import cpu_only
cpu_only.install()
from rd3_common import *
from dataset_cpu import validate_collection
import argparse,fcntl,traceback

def main():
 p=argparse.ArgumentParser(description='CPU-only independent 32-record verification and global PR analysis.');p.parse_args()
 out=R/'analysis'
 with (out/'finalize.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  result=validate_collection();write(out/'INDEPENDENT_REAL_VALIDATION.json',result)
  if not result['passed']:
   write(out/'NOT_ESTIMABLE.json',{'state':'NOT_ESTIMABLE','accepted_real_records':result['accepted_record_count'],'missing':result['missing'],'metrics_generated':False,'CUDA_initialized':False,'real_model_forward_calls':0})
   (out/'FINAL_SCIENTIFIC_REPORT.md').write_text('# NOT_ESTIMABLE\n\n32条真实科学记录尚不完整。当前独立接受'+str(result['accepted_record_count'])+'条，缺失'+str(len(result['missing']))+'条。未计算TP/FP/AP或消息效用结论；缺失记录见NOT_ESTIMABLE.json。\n')
   print('NOT_ESTIMABLE',json.dumps(result['missing'],ensure_ascii=False),flush=True);return 3
  from analyze_cpu import run
  run(R/'real_run');summary=load(out/'ANALYSIS_SUMMARY.json')
  report='# RD3-R4 独立科学报告\n\n32条真实科学记录已独立重验，包括4条R2接受记录、1条R3恢复引用和27条R4新结果。\n\n四条件全局Score排序PR积分结果：\n\n'+json.dumps(summary['metrics'],ensure_ascii=False,indent=2)+'\n\n逐样本TP/FP/FN正负及混合变化见AGENT_MARGINAL_UTILITY.csv；双Remote交互见AGENT_INTERACTION.csv；逐GT和NMS阶段证据见PER_GT_TRANSITIONS.csv、NMS_STAGE_ANALYSIS.csv。\n\n无加权效用；FP/FN为错误分量，正差代表更多错误。8帧仅来自3个scene，56个GT不是56个独立实验单位。阶段关联不能直接解释为严格因果贡献。该报告仅描述冻结局部集合。\n'
  (out/'FINAL_SCIENTIFIC_REPORT.md').write_text(report);cpu_guard();print('R4_REAL_32_INDEPENDENTLY_ACCEPTED_CPU_ANALYSIS_COMPLETE',flush=True);return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:
  write(R/'analysis/FINALIZE_FAILED.json',{'state':'SCIENTIFIC_VALIDATION_FAILED','error':str(e),'traceback':traceback.format_exc(),'CUDA_initialized':torch.cuda.is_initialized(),'real_forward_calls':0});raise
