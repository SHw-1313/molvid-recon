import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/flow_static_to_md_20261023';r=RUN/'auxiliary_execution.json'
if r.exists():raise FileExistsError(r)
s=dict(status='running',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),budget_seconds=3600,seconds=0.,commands=[]);r.write_text(json.dumps(s,indent=2)+'\n')
start=time.monotonic()
with (RUN/'process_logs/cuda_checks.log').open('x') as f:
 try:p=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_static_replay_cuda.py'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT,timeout=600);code=p.returncode
 except subprocess.TimeoutExpired:code=124
s.update(status='completed' if code==0 else 'failed',seconds=time.monotonic()-start,exit_code=code);r.write_text(json.dumps(s,indent=2)+'\n');raise SystemExit(code)
