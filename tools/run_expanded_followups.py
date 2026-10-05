"""Own-GPU sequential followups, using complete upstream artifacts only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json


def await_complete(path,*,seconds=21600):
    start=time.monotonic()
    while not path.exists():
        if time.monotonic()-start>seconds: raise TimeoutError(f'waiting for {path}')
        time.sleep(20)
    value=json.loads(path.read_text())
    if value.get('exception') is not None: raise RuntimeError(f'upstream failed: {path}')
    return value


def run(args):
    output=args.root/args.queue_name
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    atomic_write_json(output/'status.json',dict(status='waiting_for_own_A1_evaluation'))
    await_complete(args.root/'expanded_a1/execution_end.json')
    if json.loads((args.root/'expanded_a1/summary.json').read_text())['status']!='completed':
        raise RuntimeError('A1 benchmark incomplete')
    capacity=Path('/workspace/molvid-recon-gm-capacity-v3/runs/gm_motion_memory_v3/C/r10/configs/formal')
    for arm,step,checkpoint in [
        ('c4',11000,capacity/'C4/frame_joint_step_00011000.pt'),
        ('c8',11000,capacity/'C8/frame_joint_step_00011000.pt'),
        ('c4',13575,capacity/'C4/frame_joint_step_00013575.pt'),
        ('c8',13575,args.root/'c8_continuation_v2/formal/frame_joint_step_00013575.pt')]:
        if arm=='c8' and step==13575:
            await_complete(args.root/'c8_continuation_v2/execution_end.json',seconds=64800)
        name=f'expanded_{arm}_{step}'
        atomic_write_json(output/'status.json',dict(status='evaluating_capacity',active=name))
        command=[sys.executable,'-m','tools.evaluate_expanded_native','--cache',str(args.root/'development48_v2'),
            '--output',str(args.root/name),'--arm',arm,'--checkpoint',str(checkpoint)]
        with (output/f'{name}.log').open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=22000)
    atomic_write_json(output/'status.json',dict(status='waiting_for_Gcoord_training'))
    for name in ('static_71','mixed_71','static_72','mixed_72'):
        await_complete(args.recovery_root/name/'execution_end.json',seconds=86400)
    atomic_write_json(output/'status.json',dict(status='evaluating_Gcoord'))
    command=[sys.executable,'-m','tools.evaluate_current_recovery','--cache',str(args.root/'development48_v2'),
        '--generated',str(args.root/'expanded_a1'),'--training',str(args.recovery_root),
        '--output',str(args.root/'expanded_gcoord')]
    with (output/'gcoord.log').open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=22000)
    atomic_write_json(output/'status.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recovery-root',type=Path,required=True)
    p.add_argument('--queue-name',default='expanded_followup_queue');run(p.parse_args())
