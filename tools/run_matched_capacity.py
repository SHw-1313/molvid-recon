"""Re-evaluate C4 at11000, paired to existing immutable C8 at11000."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file
from molvid.training.execution_evidence import ExecutionEvidence
from tools.run_gm_capacity import _verify_paired_checkpoint


def run(output):
    if output.exists():raise FileExistsError(output)
    root=Path('/workspace/molvid-recon-gm-capacity-v3/runs/gm_motion_memory_v3/C/r10/configs')
    checkpoints={arm:root/'formal'/arm/'frame_joint_step_00011000.pt' for arm in ('C4','C8')}
    paired=_verify_paired_checkpoint(checkpoints['C4'],checkpoints['C8'])
    c8=json.loads((root/'evaluation/C8/ordinary/protocol.json').read_text())
    if sha256_file(checkpoints['C8'])!=c8['checkpoint_sha256']:raise ValueError('old C8 evaluation is not11000')
    c4_sha=sha256_file(checkpoints['C4']);recipe=root/'evaluation_recipe.json'
    families={
        'ordinary':(Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views'), 'paired_view_manifest.json'),
        'fixed_history':(Path('/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views'),'fixed_history_manifest.json')}
    freeze_run_identity(output,inputs=[*checkpoints.values(),recipe],purpose='Exposure-matched11000 C4/C8 capacity evaluation',
        protocol=dict(paired_exposure=paired,training_updates=11000,seeds=[0,1,2],steps=16,
            c8_reuse=str(root/'evaluation/C8'),new_optimizer_updates=0,test_opened=False))
    atomic_write_json(output/'paired_exposure.json',paired)
    with ExecutionEvidence(output):
        for family,(views,manifest) in families.items():
            command=[sys.executable,'-m','tools.evaluate_gm_a','--checkpoint',str(checkpoints['C4']),
                '--checkpoint-sha256',c4_sha,'--recipe',str(recipe),'--store',str(views/'clip_store/valid'),
                '--manifest',str(views/manifest),'--output',str(output/'C4'/family)]
            with (output/f'C4_{family}.log').open('x') as handle:
                subprocess.run(command,check=True,stdout=handle,stderr=subprocess.STDOUT)
        atomic_write_json(output/'complete.json',dict(status='completed',c4_sha256=c4_sha,
            c8_sha256=c8['checkpoint_sha256'],paired_exposure=paired,test_opened=False,
            c8_reused=str(root/'evaluation/C8'),new_training_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
