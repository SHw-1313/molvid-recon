"""Use the recovery GPU for its evaluation, independently of capacity inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json


def run(args):
    queue=args.root/args.queue_name
    if queue.exists(): raise FileExistsError(queue)
    queue.mkdir(parents=True);started=time.monotonic()
    while True:
        if time.monotonic()-started>86400: raise TimeoutError('24h prerequisite bound')
        state=json.loads((args.training/'queue.json').read_text())
        if state['status']=='failed': raise RuntimeError('training failed; inspect before requeue')
        released=args.wait_for_release is None or args.wait_for_release.exists()
        if state['status']=='completed' and (args.root/'expanded_a1/execution_end.json').exists() and released: break
        atomic_write_json(queue/'status.json',dict(status='waiting_for_training_A1_and_GPU_release',training=state,
            wait_for_release=None if args.wait_for_release is None else str(args.wait_for_release)))
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
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--queue-name',default='gcoord_evaluation_queue')
    p.add_argument('--wait-for-release',type=Path);run(p.parse_args())
