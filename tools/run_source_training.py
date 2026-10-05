"""One GPU/seed lane, with actual full-model16-step smoke then exact same-arm resume."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json,sha256_file
from molvid.provenance import freeze_run_identity
from molvid.training.execution_evidence import ExecutionEvidence


def run(args):
    queue=args.root/f'queue_{args.seed}'
    if queue.exists(): raise FileExistsError(queue)
    queue.mkdir(parents=True);start=time.monotonic()
    if args.wait_for:
        atomic_write_json(queue/'status.json',dict(status='waiting_for_own_GPU_release',wait_for=str(args.wait_for)))
        while not args.wait_for.exists():
            if time.monotonic()-start>86400: raise TimeoutError('24h GPU prerequisite')
            time.sleep(20)
        if json.loads(args.wait_for.read_text()).get('exception') is not None: raise RuntimeError('GPU prerequisite failed')
    manifest=json.loads((args.root/'manifest.json').read_text());protocol=manifest['protocol']
    lane_order=['v02','unit'] if args.seed==71 else ['unit','v02']
    runs=sorted([r for r in manifest['runs'] if r['seed']==args.seed],key=lambda r:lane_order.index(r['arm']))
    for run in runs:
        if shutil.disk_usage(args.root).free<10*1024**3: raise RuntimeError('need10GiB free before a full-model arm')
        config=Path(run['configuration']);out=Path(run['output']);out.mkdir()
        if sha256_file(config)!=run['configuration_sha256']: raise ValueError('frozen config changed')
        freeze_run_identity(out,inputs=[args.root/'manifest.json',config,Path(protocol['parent'])],
            purpose='Full-model source distribution training',protocol=dict(protocol,arm=run['arm'],seed=args.seed,lane_order=lane_order))
        command=[sys.executable,'-u','-m','molvid.cli.train_frame_joint','--config',str(config),'--checkpoint-every','1024']
        started=time.monotonic()
        for phase,extra in [('initial16',['--warm-start',protocol['parent'],'--warm-start-sha256',protocol['parent_sha256'],
                '--max-steps','16']),('remaining2032',None)]:
            if extra is None:
                parent=out/'train/frame_joint_step_00000016.pt'
                extra=['--resume',str(parent),'--resume-sha256',sha256_file(parent),'--max-steps','2048']
            atomic_write_json(queue/'status.json',dict(status='training',active=run['name'],phase=phase,command=command+extra))
            with ExecutionEvidence(out/phase),(out/f'{phase}.log').open('x') as log:
                result=subprocess.run(command+extra,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1,protocol['max_seconds_per_arm']-(time.monotonic()-started)))
                if result.returncode:
                    atomic_write_json(queue/'status.json',dict(status='failed',active=run['name'],phase=phase,exit_code=result.returncode))
                    raise RuntimeError('training failed; preserve diagnostics and do not continue')
        final=out/'train/frame_joint_step_00002048.pt'
        atomic_write_json(out/'complete.json',dict(status='completed',steps=2048,final_checkpoint=str(final),
            final_sha256=sha256_file(final),seconds=time.monotonic()-started,parent=protocol['parent'],
            parent_sha256=protocol['parent_sha256'],warm_start=True,exact_resume_within_arm_at_step=16))
    atomic_write_json(queue/'status.json',dict(status='completed',seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[71,72],required=True);p.add_argument('--wait-for',type=Path)
    run(p.parse_args())
