"""Actual largest scheduled-query updates, disposable and not efficacy runs."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from molvid.runtime import atomic_write_json, sha256_file


def run(args):
    manifest=json.loads((args.root/'manifest.json').read_text())
    queue=args.root/'profile_queue'
    if queue.exists():
        raise FileExistsError(queue)
    queue.mkdir()
    completed=[]
    try:
        for entry in manifest['profiles']:
            config=Path(entry['configuration']); output=Path(entry['output'])
            if sha256_file(config) != entry['configuration_sha256'] or output.exists():
                raise ValueError('profile config changed or destination already exists')
            command=[sys.executable,'-u','-m','tools.train_coordinate_flow',
                '--config',str(config),'--output',str(output),'--seed',str(entry['seed']),
                '--profile-largest','--max-steps','1']
            atomic_write_json(queue/'status.json',dict(status='running',seed=entry['seed'],command=command))
            with (queue/f'profile_{entry["seed"]}.log').open('x') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            summary=json.loads((output/'summary.json').read_text())
            rows=[json.loads(line) for line in (output/'updates.jsonl').read_text().splitlines()]
            if (summary['status'] != 'completed' or summary['updates_this_execution'] != 1
                    or len(rows) != 1 or rows[0]['handedness_applied'] is not True
                    or not all(math.isfinite(rows[0][key]) for key in ('loss','rf_loss_A2','handedness_loss','grad_norm'))):
                raise ValueError('actual auxiliary largest-query update did not qualify')
            if list(output.glob('coordinate_step_*.pt')):
                raise ValueError('disposable profile unexpectedly saved training weights')
            completed.append(dict(seed=entry['seed'],profile_updates=1,peak_cuda_bytes=rows[0]['peak_cuda_bytes'],
                valid_atom_queries=rows[0]['valid_atom_queries'],samples=rows[0]['samples'],seconds=summary['seconds']))
        atomic_write_json(queue/'status.json',dict(status='completed',profiles=completed,
            optimizer_updates_discarded=2,formal_training_started=False,effectiveness_tested=False))
    except BaseException as error:
        atomic_write_json(queue/'status.json',dict(status='failed',error=repr(error),completed=completed))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    run(p.parse_args())
