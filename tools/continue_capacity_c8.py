"""Actual C8 training: prove source parity at11001, then finish original epoch.

All outputs are isolated here. Original source/checkpoints and schedules remain
read-only. This is exact objective/schedule continuation, not a new architecture.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file
from molvid.training.execution_evidence import ExecutionEvidence
from tools.check_gm_capacity_resume import _assert_equal


def run(output):
    if output.exists():raise FileExistsError(output)
    root=Path('/workspace/molvid-recon-gm-capacity-v3')
    config=root/'runs/gm_motion_memory_v3/C/r10/configs/formal_C8.json'
    parent=root/'runs/gm_motion_memory_v3/C/r10/configs/formal/C8/frame_joint_step_00011000.pt'
    sha=sha256_file(parent)
    if sha!='e6ed57fee02781ce6303fe56a079226fb32502c94c675c8e9181c20d6ff8239d':raise ValueError('C8 parent changed')
    if shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free<10*1024**3:
        raise RuntimeError('need10GiB free before starting training')
    freeze_run_identity(output,inputs=[config,parent],purpose='Exact C8 resume with original-source next-update parity',
        protocol=dict(parent_step=11000,final_step=13575,expected_new_updates=2575,
            parity_step=11001,old_source=str(root),checkpoint_every=1000,test_opened=False,
            changed_objective=False,changed_schedule=False,max_seconds=64800))
    started=time.monotonic()
    def train(name,cwd,resume,limit):
        target=output/name
        command=[sys.executable,'-u','-m','molvid.cli.train_frame_joint','--config',str(config),
            '--resume',str(resume),'--resume-sha256',sha256_file(resume),'--max-steps',str(limit),
            '--output-root',str(target.resolve()),'--checkpoint-every','1000']
        atomic_write_json(output/f'{name}_command.json',dict(command=command,cwd=str(cwd)))
        with (output/f'{name}.log').open('x') as handle:
            subprocess.run(command,cwd=cwd,check=True,stdout=handle,stderr=subprocess.STDOUT,
                           timeout=max(1,64800-(time.monotonic()-started)))
        return target/f'frame_joint_step_{limit:08d}.pt'
    with ExecutionEvidence(output):
        old=train('original_next_update',root,parent,11001)
        new=train('current_next_update',Path.cwd(),parent,11001)
        left=torch.load(old,map_location='cpu',weights_only=False)
        right=torch.load(new,map_location='cpu',weights_only=False)
        # Compare numerical training state, not source provenance/creation timestamps.
        atomic_write_json(output/'checkpoint_keys.json',dict(old=sorted(left),new=sorted(right)))
        keys=[k for k in ('model','optimizer','scheduler','step','successful_updates','rank_states','rng_state') if k in left]
        if 'model' not in keys or 'optimizer' not in keys:raise ValueError('unknown checkpoint state schema')
        counts={k:_assert_equal(left[k],right[k],k) for k in keys}
        atomic_write_json(output/'next_update_parity.json',dict(status='passed',exact=True,checked=counts,
            old_sha256=sha256_file(old),new_sha256=sha256_file(new),step=11001))
        del left,right
        final=train('formal',Path.cwd(),new,13575)
        atomic_write_json(output/'complete.json',dict(status='completed',final_checkpoint=str(final),
            final_sha256=sha256_file(final),final_step=13575,additional_updates=2575,
            parity_only_duplicate_updates=1,seconds=time.monotonic()-started,test_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
