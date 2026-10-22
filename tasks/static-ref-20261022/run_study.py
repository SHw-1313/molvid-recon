"""Two finite static runs and retained 4096/8192 probes; no automatic retry."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/static-ref-20261022';RUN=ROOT/'runs/static_ref_20261022'
CAP=20000. # Two arms plus 3200 shared checks/old probe =43200 process GPU seconds.
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('control','concat'),required=True);a=p.parse_args()
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
receipt=RUN/(a.arm+'_execution.json')
if receipt.exists():raise FileExistsError(receipt)
state=dict(arm=a.arm,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),status='running',budget_seconds=CAP,seconds=0.,commands=[],allocator_GiB=20)
def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
def command(label,argv):
    remaining=CAP-state['seconds']-15.
    if remaining<=0:raise RuntimeError('Cumulative allocation exhausted')
    log=RUN/'process_logs'/f'{a.arm}_{label}.log'
    item=dict(label=label,argv=argv,log=str(log),status='running',timeout_seconds=remaining)
    state['commands'].append(item);state['current']=label;save();start=time.monotonic()
    with log.open('w') as f:
        child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT)
        item['pid']=child.pid;save()
        try:code=child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            code=124
    elapsed=time.monotonic()-start;state['seconds']+=elapsed
    item.update(exit_code=code,seconds=elapsed,status='completed' if code==0 else 'failed');save()
    if code:raise RuntimeError(f'{label} exited {code}; raw retained, no automatic retry')
save()
try:
    output=RUN/a.arm
    base=[sys.executable,'-u','tools/train_coordinate_state.py','--config',str(TASK/'configs'/f'{a.arm}.json'),'--output',str(output),'--seed','71','--allocator-GiB','20']
    command('first32',base+['--stop-after','32'])
    shutil.copy2(output/'checkpoint_latest.pt',output/'checkpoint_step32.pt')
    command('to4096',base+['--resume',str(output/'checkpoint_step32.pt'),'--stop-after','4096'])
    shutil.copy2(output/'checkpoint_latest.pt',output/'checkpoint_step4096.pt')
    command('to8192',base+['--resume',str(output/'checkpoint_step4096.pt')])
    for step in (4096,8192):
        command(f'probe{step}',[sys.executable,'-u',str(TASK/'static_probe.py'),'generate','--stage',a.arm+str(step),'--allocator-GiB','20'])
    state['status']='completed';save()
except Exception as error:
    state.update(status='failed_or_budget_stopped',error=repr(error));save();raise
