"""Finite authorized static -> MD -> six Q50 commands; no retries or scheduler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
TASK=ROOT/'tasks/flow-capacity-20261020'
RUN=ROOT/'runs/flow_capacity_20261020'
CASES=('atlas_1whz_A','atlas_3vnb_A','atlas_6e5y_A','4J8T','2XSB','5TV1')
# Sum=94 GPUh, plus a separate 2h shared checks/profile reserve: total <=96h.
CAP_HOURS=dict(control=9,observed=11,query_pre=14,temporal=11,collective=14,query_post=14,wide=21)

def run_arm(arm):
    receipt=RUN/(arm+'_execution.json')
    if receipt.exists(): raise FileExistsError(receipt)
    used=0.; cap=CAP_HOURS[arm]*3600
    state=dict(arm=arm,status='running',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),
        budget_seconds=cap,seconds=0.,commands=[],scope='4096 static + 4096 MD + 12 Q50; includes retries (none automatic)')
    def save():
        receipt.write_text(json.dumps(state,indent=2)+'\n')
    def command(label,args):
        nonlocal used
        remaining=cap-used
        if remaining<=0: raise RuntimeError('Arm allocation exhausted; no budget transfer automatic')
        log=RUN/'process_logs'/(arm+'_'+label+'.log')
        item=dict(label=label,argv=args,log=str(log),timeout_seconds=remaining,status='running')
        state['commands'].append(item);state['current']=label;save()
        started=time.monotonic()
        with log.open('w') as f:
            process=subprocess.Popen(args,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT)
            item['pid']=process.pid;save()
            try: code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill();process.wait()
                code=124
        elapsed=time.monotonic()-started;used+=elapsed
        item.update(exit_code=code,seconds=elapsed,status='completed' if code==0 else 'failed')
        state.update(seconds=used,remaining_seconds=cap-used);save()
        if code: raise RuntimeError(f'{label} exited {code}; no retry')
    save()
    try:
        for phase in ('static','md'):
            output=RUN/arm/phase
            base=[sys.executable,'-u','tools/train_coordinate_state.py','--config',str(TASK/'configs'/f'{arm}_{phase}.json'),
                '--output',str(output),'--seed','71']
            initial=base+(['--warm-start',str(RUN/arm/'static/coordinate_final.pt')] if phase=='md' else [])
            command(phase+'_first32',initial+['--stop-after','32'])
            command(phase+'_remaining',base+['--resume',str(output/'checkpoint_latest.pt')])
        for case in CASES:
            command('generate_'+case,[sys.executable,'-u',str(TASK/'run_flow_short.py'),'--arm',arm,
                '--record',f'/workspace/MOLVID-MAIN-20261016/runs/main_integration_20261016/short50_inputs/records/{case}.pt',
                '--checkpoint',str(RUN/arm/'md/coordinate_final.pt'),'--output',str(RUN/'generation'/arm/case)])
        state['status']='completed';save()
    except Exception as e:
        state.update(status='failed_or_budget_stopped',error=repr(e));save();raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=CAP_HOURS,required=True)
    p.add_argument('--then-control',action='store_true');a=p.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito': raise RuntimeError('Use existing torch-ito')
    sys.path.insert(0,str(ROOT))
    import molvid
    if Path(molvid.__file__).resolve().parent != ROOT/'molvid':
        raise RuntimeError(f'Wrong local package: {molvid.__file__}')
    print(json.dumps({'package':molvid.__file__,'root':str(ROOT),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}),flush=True)
    run_arm(a.arm)
    if a.then_control: run_arm('control')
if __name__=='__main__': main()
