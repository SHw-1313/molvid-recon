"""Finite first32/exact-resume512 runner; requires an explicit assigned time cap.

This entrypoint never starts itself, retries, or changes the fixed recipe.
Separate --phase first32/remaining permits the common CPU pair check between
arms; --phase all permits an already-authorized continuous lane.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
TASK=ROOT/'tasks/flow-hq-ratio-20261102'
RUN=ROOT/'runs/flow_hq_ratio_20261102'
PARENT='/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('h4q4','h4q8','h8q4','h8q8','h12q12'), required=True)
    p.add_argument('--budget-seconds', type=float, required=True, help='Assigned cap for this arm, including previous phases')
    p.add_argument('--allocator-GiB', type=float, default=32)
    p.add_argument('--phase', choices=('first32','remaining','all'), default='all')
    a=p.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito': raise RuntimeError('Existing torch-ito required')
    RUN.mkdir(parents=True,exist_ok=True); (RUN/'process_logs').mkdir(exist_ok=True)
    receipt=RUN/f'train_{a.arm}_{a.phase}_execution.json'
    if receipt.exists(): raise FileExistsError(receipt)
    previous=0.
    if a.phase=='remaining':
        first=json.loads((RUN/f'train_{a.arm}_first32_execution.json').read_text())
        if first['status']!='completed': raise ValueError('First32 phase did not finish')
        previous=first['seconds']
    elif (RUN/a.arm).exists():
        raise FileExistsError('Preserve existing training output; use the explicit remaining phase')
    s=dict(status='running',pid=os.getpid(),arm=a.arm,phase=a.phase,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        budget_seconds=a.budget_seconds,previous_phase_seconds=previous,seconds=0.,commands=[],
        parent=PARENT,planned_slots=256,planned_optimizer_updates=2*len(json.loads((RUN/'data/manifest.json').read_text())['rows']))
    def save(): receipt.write_text(json.dumps(s,indent=2)+'\n')
    def command(label,argv):
        left=a.budget_seconds-previous-s['seconds']-31.
        if left<=0: raise RuntimeError('Assigned training cap exhausted')
        log=RUN/'process_logs'/f'{a.arm}_{label}.log'
        item=dict(label=label,argv=argv,status='running',log=str(log),timeout_seconds=left)
        s['commands'].append(item);save();start=time.monotonic()
        with log.open('x') as f:
            child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT)
            item['pid']=child.pid;save()
            try: code=child.wait(timeout=left)
            except subprocess.TimeoutExpired:
                child.terminate()
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: child.kill();child.wait()
                code=124
        sec=time.monotonic()-start;s['seconds']+=sec
        item.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=sec);save()
        if code: raise RuntimeError(f'{label} failed; partial preserved without retry')
    save()
    try:
        out=RUN/a.arm
        base=[sys.executable,'-u',str(TASK/'train_hq.py'),'--config',str(TASK/'configs'/f'{a.arm}.json'),
              '--output',str(out),'--seed','71','--allocator-GiB',str(a.allocator_GiB)]
        if a.phase!='remaining':
            command('first32',base+['--warm-start',PARENT,'--stop-after','32'])
            shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_step32.pt')
        if a.phase!='first32':
            command('remaining256',base+['--resume',str(out/'checkpoint_latest.pt')])
        s['status']='completed';save()
    except Exception as error:
        s.update(status='failed_or_budget_stopped',error=repr(error));save();raise


if __name__=='__main__': main()
