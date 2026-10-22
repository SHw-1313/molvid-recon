"""Serial finite checks/probes; shared3200s allocation, no automatic retries."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/static_ref_20261022';CAP=3200.
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.command[1:] if a.command[:1]==['--'] else a.command
if not argv:raise ValueError('Explicit command required')
RUN.mkdir(parents=True,exist_ok=True);(RUN/'process_logs').mkdir(exist_ok=True)
receipt=RUN/'auxiliary_execution.json'
state=json.loads(receipt.read_text()) if receipt.exists() else dict(budget_seconds=CAP,seconds=0.,commands=[])
if any(x['status']=='running' or x['label']==a.label for x in state['commands']):raise RuntimeError('No concurrent auxiliary command or reused label')
remaining=CAP-state['seconds']-15.
if remaining<=0:raise RuntimeError('Shared check/probe allocation exhausted')
log=RUN/'process_logs'/f'{a.label}.log';item=dict(label=a.label,argv=argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),status='running',log=str(log),timeout_seconds=remaining)
state['commands'].append(item)
def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
save();start=time.monotonic()
with log.open('w') as f:
    child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT)
    item['pid']=child.pid;save()
    try:code=child.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        code=124
seconds=time.monotonic()-start;state['seconds']+=seconds
item.update(exit_code=code,seconds=seconds,status='completed' if code==0 else 'failed');save()
print(json.dumps(item),flush=True);sys.exit(code)
