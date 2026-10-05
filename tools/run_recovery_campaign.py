"""Bounded sequential Gcoord GPU queue after this campaign's PVB job releases it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started=time.monotonic()
    while not (args.wait_for/'execution_end.json').exists():
        if time.monotonic()-started>21600: raise TimeoutError('six-hour prerequisite budget')
        atomic_write_json(args.output/'queue.json',dict(status='waiting_for_own_PVB_evaluation',
            wait_for=str(args.wait_for),seconds=time.monotonic()-started))
        time.sleep(20)
    summary=json.loads((args.wait_for/'summary.json').read_text())
    if summary['status']!='completed': raise RuntimeError('PVB prerequisite incomplete; inspect before requeue')
    for arm,seed,steps,name in [('mixed',71,16,'pilot_mixed71')]+[
            (arm,seed,2048,f'{arm}_{seed}') for seed in (71,72) for arm in ('static','mixed')]:
        atomic_write_json(args.output/'queue.json',dict(status='training',active=name,
            arm=arm,seed=seed,updates=steps))
        command=[sys.executable,'-m','tools.train_current_recovery','--cache',str(args.cache),
            '--output',str(args.output/name),'--arm',arm,'--seed',str(seed),'--steps',str(steps)]
        with (args.output/f'{name}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=22000)
        if result.returncode:
            atomic_write_json(args.output/'queue.json',dict(status='failed',active=name,exit_code=result.returncode))
            raise RuntimeError(f'{name} failed; preserved log, no formal launch from failed pilot')
    atomic_write_json(args.output/'queue.json',dict(status='completed',seconds=time.monotonic()-started,
        formal_models=4,pilot_models=1,parent_generator_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-for',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
