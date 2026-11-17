import sys,tempfile,subprocess,os,json,shutil
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='b0_parallel_'))
try:
 script='from pathlib import Path;import sys;from cpu_components import reserve;reserve(Path(sys.argv[1]),int(sys.argv[2]),"core")'
 env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent),PYTHONDONTWRITEBYTECODE='1')
 ps=[subprocess.Popen([sys.executable,'-B','-c',script,str(root),str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env) for i in range(30)]
 status=[p.communicate()[1] for p in ps];codes=[p.returncode for p in ps]
 assert codes.count(0)==22 and len(json.loads((root/'budget.json').read_text())['attempts'])==22
 assert all(code==0 or b'FORWARD_LIMIT' in err for code,err in zip(codes,status))
 Path(__file__).with_name('CONCURRENCY_TEST.json').write_text(json.dumps({'passed':True,'processes':30,'accepted':22,'rejected':8,'CUDA_initialized':False,'real_model_forwards':0},indent=2))
finally:shutil.rmtree(root)
