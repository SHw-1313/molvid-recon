"""Per-seed own-GPU evaluations after both fixed-budget source arms finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json,sha256_file


def run(args):
    training=args.root/'source_training';queue=args.root/f'source_evaluation_queue_{args.seed}'
    if queue.exists(): raise FileExistsError(queue)
    queue.mkdir(parents=True);started=time.monotonic()
    prerequisite=training/f'queue_{args.seed}/status.json'
    atomic_write_json(queue/'status.json',dict(status='waiting_for_both_own_training_arms',prerequisite=str(prerequisite)))
    while True:
        if time.monotonic()-started>129600: raise TimeoutError('36h training prerequisite')
        status=json.loads(prerequisite.read_text())
        if status['status']=='failed': raise RuntimeError('training failed, no evaluation')
        if status['status']=='completed': break
        time.sleep(30)
    outputs=[]
    def execute(name,command,timeout=22000):
        atomic_write_json(queue/'status.json',dict(status='running',active=name,command=command))
        with (queue/f'{name}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:
            atomic_write_json(queue/'status.json',dict(status='failed',active=name,exit_code=result.returncode))
            raise RuntimeError(f'{name} failed; no downstream comparison')
    for arm in ('unit','v02'):
        name=f'{arm}_{args.seed}';directory=training/name
        result=json.loads((directory/'complete.json').read_text());checkpoint=Path(result['final_checkpoint'])
        if result['steps']!=2048 or sha256_file(checkpoint)!=result['final_sha256']: raise ValueError('only exact final2048')
        execute(f'audit_{name}',[sys.executable,'-m','tools.audit_source_checkpoint','--checkpoint',str(checkpoint),
            '--config',str(training/f'{name}.json'),'--output',str(args.root/f'source_training_audit_{name}_final')],600)
        output=args.root/f'expanded_source_{name}';outputs.append(output)
        execute(f'generate_{name}',[sys.executable,'-m','tools.evaluate_expanded_native',
            '--cache',str(args.root/'development48_v2'),'--output',str(output),
            '--arm','checkpoint','--variant',f'source_{name}','--checkpoint',str(checkpoint)])
        execute(f'chemistry_{name}',[sys.executable,'-m','tools.audit_cached_stereochemistry',
            '--cache',str(args.root/'development48_v2'),'--generated',str(output),
            '--output',str(args.root/f'stereochemistry_source_{name}')],600)
    execute('paired',[sys.executable,'-m','tools.compare_expanded_native','--generated',
        str(args.root/'expanded_a1'),str(args.root/'expanded_pvb'),*map(str,outputs),
        '--exposure',str(args.root/'external_overlap_v3/summary.json'),
        '--output',str(args.root/f'paired_source_{args.seed}'),
        '--pair',f'source_v02_{args.seed}:source_unit_{args.seed}'],600)
    atomic_write_json(queue/'status.json',dict(status='completed',seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[71,72],required=True);run(p.parse_args())
