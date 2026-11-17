"""Exercise the unchanged C2 persistent reservation with concurrent CPU processes."""
import sys,json,multiprocessing as mp
from pathlib import Path
from runtime_common import ROOT,C2,atomic
sys.path.insert(0,str(C2))
from cpu_components import reserve

def attempt(root,n,q):
 try:reserve(root,n,'core');q.put('accepted')
 except RuntimeError:q.put('rejected')
if __name__=='__main__':
 root=ROOT/'mock_runs'/'budget_concurrency';root.mkdir(exist_ok=False)
 ctx=mp.get_context('fork');q=ctx.Queue();workers=[ctx.Process(target=attempt,args=(root,n,q)) for n in range(30)]
 for p in workers:p.start()
 for p in workers:p.join(20)
 values=[q.get(timeout=5) for _ in workers]
 result={'passed':values.count('accepted')==22 and values.count('rejected')==8 and all(p.exitcode==0 for p in workers),'accepted':values.count('accepted'),'rejected':values.count('rejected'),'component':'unchanged C2 cpu_components.reserve also wrapped by formal C3 engine adapter','MOCK_ONLY':True}
 atomic(ROOT/'C3_BUDGET_CONCURRENCY.json',result);print(json.dumps(result))
