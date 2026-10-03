"""One real A0 CUDA step exercising every applicable gradient diagnostic."""

import argparse
import json
from pathlib import Path

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.runtime import atomic_write_json, configure_device, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.joint import FrameJointTrainer, JointLossConfig
from tools.gm_a_evidence import PARENT, PARENT_SHA, SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = configure_device('cuda', deterministic=True)
    seed_all(SEED)
    raw = load_config('configs/frame_gm_p2_formal_GM_260923.yaml', schema='molvid.frame_joint.train.v1')
    train, _, data_hash, _, _, close = _open_data(raw['data'])
    try:
        selected = min(
            (s for s in get_clip_specs(train) if s.time_bucket_id == 'dt_100ps'),
            key=lambda s: (s.atoms, s.sample_id),
        )
        loaded = load_frame_joint_inference(
            PARENT, expected_sha256=PARENT_SHA,
            codec_path=raw['codec']['checkpoint'], codec_sha256=raw['codec']['sha256'],
            device=device,
        )
        model = loaded['model']
        batch = prepare_frame_joint_batch(
            model.target_teacher, collate_clip_records([train[selected.index]]),
            device=device, normalizer=model, history_frames=4,
        )
        trainer = FrameJointTrainer(
            model, loss_config=JointLossConfig.resolve(loaded['payload']['contracts']['loss']),
            learning_rates={'history_encoder': 1e-4, 'dit': 2e-4, 'decoder': 5e-5},
            stage='continuation', amp=True, data_hash=data_hash,
            teacher_artifact_sha256=raw['codec']['sha256'],
            schedule_contract={'diagnostic_smoke': True},
        )
        trainer.load_warm_start(PARENT, expected_sha256=PARENT_SHA)
        metrics = trainer.train_step(
            batch, generator=torch.Generator(device=device).manual_seed(SEED + 812),
            diagnose_gradients=True,
        )
        gradients = metrics['gradient_diagnostics']
        required = ('flow', 'flow_h', 'flow_v', 'clean_coordinate', 'clean_bond')
        passed = metrics['step'] == 1 and all(
            gradients[name]['active'] and gradients[name]['dit_raw_norm'] >= 0
            for name in required
        )
        atomic_write_json(args.output/'diagnostic_cuda.json', {
            'schema': 'molvid.A.diagnostic_cuda.v1', 'sample_id': selected.sample_id,
            'required': required, 'gradient_diagnostics': gradients,
            'loss': metrics['loss'], 'passed': passed, 'test_opened': False,
        })
        if not passed:
            raise RuntimeError('A0 gradient diagnostics incomplete')
    finally:
        close()


if __name__ == '__main__':
    main()
