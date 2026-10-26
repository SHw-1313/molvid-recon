"""Time a single explicitly requested auxiliary command against the shared8h cap."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/flow_ligand_4096_20261026'
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
receipt=RUN/'auxiliary_execution.json'
s=json.loads(receipt.read_text()) if receipt.exists() else dict(budget_seconds=28800.,seconds=0.,commands=[])
if any(x['label']==a.label for x in s['commands']):raise ValueError('Use explicit new label for a necessary retry; never overwrite evidence')
if any(x['status']=='running' for x in s['commands']):raise RuntimeError('Auxiliary command already active')
left=s['budget_seconds']-s['seconds']-10
if left<=0:raise RuntimeError('Auxiliary budget exhausted')
item=dict(label=a.label,argv=a.command,status='running',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'));s['commands'].append(item)
def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
save();start=time.monotonic()
with (RUN/'process_logs'/(a.label+'.log')).open('x') as f:
 child=subprocess.Popen(a.command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
 try:code=child.wait(timeout=left)
 except subprocess.TimeoutExpired:
  child.terminate()
  try:child.wait(timeout=8)
  except subprocess.TimeoutExpired:child.kill();child.wait()
  code=124
elapsed=time.monotonic()-start;s['seconds']+=elapsed;item.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=elapsed);save();sys.exit(code)
