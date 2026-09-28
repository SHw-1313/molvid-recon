"""Compare full K=2 and streamed K=2 CUDA updates on one real train clip."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.runtime import atomic_write_json, configure_device, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig
from tools.gm_a_evidence import PARENT, PARENT_SHA, SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-atoms', type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = configure_device('cuda', deterministic=True)
    seed_all(SEED)
    raw = load_config('configs/frame_gm_p2_formal_GM_260923.yaml', schema='molvid.frame_joint.train.v1')
    recipe = json.loads(args.recipe.read_text())
    train, _, data_hash, _, _, close = _open_data(raw['data'])
    try:
        specs = [s for s in get_clip_specs(train) if s.time_bucket_id == 'dt_100ps']
        selected = min(specs, key=lambda s: (abs(s.atoms - args.target_atoms), s.sample_id))
        record = train[selected.index]
        cpu_batch = collate_clip_records([record])
        results = {}
        final_states = {}
        final_gradients = {}
        for label, threshold in (('full', 0), ('streamed', 1)):
            loaded = load_frame_joint_inference(
                PARENT, expected_sha256=PARENT_SHA,
                codec_path=raw['codec']['checkpoint'], codec_sha256=raw['codec']['sha256'],
                device=device,
            )
            model = loaded['model']
            batch = prepare_frame_joint_batch(
                model.target_teacher, cpu_batch, device=device,
                normalizer=model, history_frames=4,
            )
            config = SampledAuxiliaryConfig.resolve({
                'enabled': True, 'euler_steps': recipe['euler_steps'],
                'physical': recipe['physical'],
                'streamed_atom_frames_threshold': threshold,
            })
            trainer = FrameJointTrainer(
                model, loss_config=JointLossConfig.resolve(loaded['payload']['contracts']['loss']),
                learning_rates={'history_encoder': 1e-4, 'dit': 2e-4, 'decoder': 5e-5},
                stage='continuation', amp=True, data_hash=data_hash,
                teacher_artifact_sha256=raw['codec']['sha256'],
                schedule_contract={'streamed_parity': True}, sampled_config=config,
            )
            trainer.load_warm_start(PARENT, expected_sha256=PARENT_SHA)
            main_rng = torch.Generator(device=device).manual_seed(SEED + 410)
            aux_rng = torch.Generator(device=device).manual_seed(SEED + 411)
            metrics = trainer.train_step(
                batch, generator=main_rng, sampled_generator=aux_rng,
                auxiliary_active=True,
            )
            results[label] = metrics
            final_states[label] = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            final_gradients[label] = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            }
            del trainer, batch, model, loaded
            torch.cuda.empty_cache()
        keys = sorted(key for key in results['full'] if key.startswith(('raw_', 'weighted_')))
        metric_error = {
            key: abs(results['full'][key] - results['streamed'][key]) for key in keys
        }
        parameter_error = max(
            float((final_states['full'][name] - final_states['streamed'][name]).abs().max())
            for name in final_states['full']
        )
        gradient_error = abs(results['full']['grad_norm'] - results['streamed']['grad_norm'])
        gradient_delta_sq = sum(
            float((final_gradients['full'][name] - final_gradients['streamed'][name]).double().square().sum())
            for name in final_gradients['full']
        )
        gradient_reference_sq = sum(
            float(value.double().square().sum()) for value in final_gradients['full'].values()
        )
        relative_gradient_l2 = math.sqrt(gradient_delta_sq / gradient_reference_sq)
        gradient_max_abs = max(
            float((final_gradients['full'][name] - final_gradients['streamed'][name]).abs().max())
            for name in final_gradients['full']
        )
        tolerance = 2e-4
        passed = (
            all(math.isfinite(value) and value <= tolerance for value in metric_error.values())
            and relative_gradient_l2 <= 3e-3 and gradient_error <= tolerance
            and parameter_error <= 4.1e-4
            and results['streamed'].get('sampled_streamed_backward') is True
        )
        evidence = {
            'schema': 'molvid.A.streamed_gradient_parity.v1',
            'sample_id': selected.sample_id, 'atoms': selected.atoms,
            'euler_steps': recipe['euler_steps'], 'draws': 2,
            'metric_max_abs_difference': max(metric_error.values()),
            'per_metric_difference': metric_error,
            'parameter_max_abs_difference': parameter_error,
            'gradient_relative_l2_difference': relative_gradient_l2,
            'gradient_max_abs_difference': gradient_max_abs,
            'grad_norm_abs_difference': gradient_error,
            'tolerance': tolerance, 'passed': passed,
            'gradient_relative_l2_tolerance': 3e-3,
            'parameter_max_abs_tolerance': 4.1e-4,
            'gpu_hours': (time.time() - started) / 3600,
            'test_opened': False,
        }
        atomic_write_json(args.output/'streamed_parity.json', evidence)
        if not passed:
            raise RuntimeError('streamed and full K=2 updates differ')
    finally:
        close()


if __name__ == '__main__':
    main()
