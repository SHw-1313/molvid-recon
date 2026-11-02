import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_hq_ratio_20261102'
p=RUN/'sampling_checks_execution.json'
if p.exists():raise FileExistsError('No automatic retry')
s=dict(status='running',pid=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),seconds=0,cap_seconds=600)
def save():p.write_text(json.dumps(s,indent=2)+'\n')
save();start=time.monotonic()
with (RUN/'process_logs/sampling_checks.log').open('x') as f:
 child=subprocess.Popen([sys.executable,'-u',str(TASK/'check_sampling.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);s['child_pid']=child.pid;save()
 try:code=child.wait(timeout=570)
 except subprocess.TimeoutExpired:
  child.terminate()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.kill();child.wait()
  code=124
s.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.monotonic()-start);save();raise SystemExit(code)
