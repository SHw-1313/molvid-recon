"""One bounded targeted-check process, with every failure retained."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_q4_align_20261030';receipt=RUN/'checks_execution.json'
if receipt.exists():raise FileExistsError('Preserve checks; no auto retry')
previous=json.loads((RUN/'checks_execution_failed_directory.json').read_text())
s=dict(status='running',pid=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),budget_seconds=3600.,seconds=previous['seconds'],commands=previous['commands'].copy())
def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
cmd=[sys.executable,'-u',str(TASK/'check_q4_alignment.py')];item=dict(command=cmd,status='running');s['commands'].append(item);save();start=time.monotonic()
with (RUN/'process_logs/checks.log').open('x') as f:
 child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
 try:code=child.wait(timeout=3570-previous['seconds'])
 except subprocess.TimeoutExpired:
  child.terminate()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.kill();child.wait()
  code=124
elapsed=time.monotonic()-start;s['seconds']+=elapsed;item.update(seconds=elapsed,exit_code=code,status='completed' if code==0 else 'failed');s['status']=item['status'];save();raise SystemExit(code)
