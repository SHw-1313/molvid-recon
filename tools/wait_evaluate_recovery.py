"""Use the recovery GPU for its evaluation, independently of capacity inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json


def run(args):
    queue=args.root/'gcoord_evaluation_queue'
    if queue.exists(): raise FileExistsError(queue)
    queue.mkdir(parents=True);started=time.monotonic()
    while True:
        if time.monotonic()-started>86400: raise TimeoutError('24h prerequisite bound')
        state=json.loads((args.training/'queue.json').read_text())
        if state['status']=='failed': raise RuntimeError('training failed; inspect before requeue')
        if state['status']=='completed' and (args.root/'expanded_a1/execution_end.json').exists(): break
        atomic_write_json(queue/'status.json',dict(status='waiting_for_training_and_A1',training=state))
        time.sleep(20)
    atomic_write_json(queue/'status.json',dict(status='GPU_evaluation'))
    command=[sys.executable,'-m','tools.evaluate_current_recovery','--cache',str(args.root/'development48_v2'),
        '--generated',str(args.root/'expanded_a1'),'--training',str(args.training),
        '--output',str(args.root/'expanded_gcoord')]
    with (queue/'evaluation.log').open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=22000)
    atomic_write_json(queue/'status.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);run(p.parse_args())
