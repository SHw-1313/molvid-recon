"""Bounded validation diagnosis of A1 solver error and observed-memory use.

No optimizer updates, test records, checkpoint writes, or model monkeypatches.
The optional memory removals are out-of-distribution sensitivity interventions,
not trained ablations. Selection is frozen before generation and covers each
validation system equally at the requested lags, R1/anchor0/H4.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import molvid
from molvid.checkpoints import load_frame_joint_inference
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.geometry import bond_errors
from molvid.evaluation.physical import sampled_clash_metrics
from molvid.evaluation.temporal import temporal_metrics_v3
from molvid.flow.sampling import euler_sample_frames
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.model import FrameJointCondition
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch
from tools.evaluate_frame_joint_multitime import meta_for
from tools.recompute_frame_gm_metrics_v3 import _jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path('/workspace/molvid-gm-memory-v3-coordination/recipe_release.json'))
    parser.add_argument('--views', type=Path, default=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--systems', type=int, default=8)
    parser.add_argument('--lags', type=int, nargs='+', default=[100, 400])
    parser.add_argument('--steps', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--ablate-memory', action='store_true')
    args = parser.parse_args()
    if args.systems < 1 or any(n < 1 for n in args.steps + args.lags):
        raise ValueError('system count, lags and solver steps must be positive')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('seeds must be unique nonnegative integers')
    if args.output.exists():
        raise ValueError('use a fresh diagnostic output directory')
    release = json.loads(args.release.read_text())
    if release['selected_arm'] != 'A1' or release['test_opened'] is not False:
        raise ValueError('diagnostic requires the evaluated validation-only A1 release')
    config_path = Path(release['configuration']['path'])
    if sha256_file(config_path) != release['configuration']['sha256']:
        raise ValueError('released configuration hash differs')
    raw = load_config(config_path, schema='molvid.frame_joint.train.v1')
    checkpoint = release['selected_checkpoint']
    recipe_path = Path('/workspace/molvid-recon-gm-motion-v3') / release['auxiliary_recipe']['path']
    if sha256_file(recipe_path) != release['auxiliary_recipe']['sha256']:
        raise ValueError('released physical recipe hash differs')
    physical = PhysicalAuxiliaryConfig.resolve(json.loads(recipe_path.read_text())['physical'])
    store = args.views / 'clip_store/valid'
    manifest = args.views / 'paired_view_manifest.json'
    reference_protocol = Path('/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A1/ordinary/protocol.json')
    expected = json.loads(reference_protocol.read_text())
    if sha256_file(store / 'index.txt') != expected['store_index_sha256'] or sha256_file(manifest) != expected['manifest_sha256']:
        raise ValueError('validation inputs differ from A1 formal evaluation')
    dataset = ClipMMapDataset(store)
    try:
        by_id = {str(row[0]): i for i, row in enumerate(dataset._index)}
        systems = sorted({sid.rsplit('_R', 1)[0] for sid in by_id})[:args.systems]
        ids = [f'{system}_R1_a000000_dt_{lag}ps_H4' for system in systems for lag in args.lags]
        if not ids or any(sid not in by_id for sid in ids):
            raise ValueError('balanced validation selection is incomplete')
        source = Path(molvid.__file__).resolve().parents[1]
        protocol = {
            'schema': 'molvid.abcd.generation_diagnostic.v1', 'sample_ids': ids,
            'checkpoint_sha256': checkpoint['sha256'], 'release_sha256': sha256_file(args.release),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
            'source_path': str(source), 'precision': 'fp32', 'steps': args.steps, 'seeds': args.seeds,
            'ablate_memory': args.ablate_memory, 'split': 'valid', 'test_opened': False,
            'store_index_sha256': expected['store_index_sha256'], 'manifest_sha256': expected['manifest_sha256'],
            'interpretation': 'balanced short-clip diagnosis; memory removal is OOD sensitivity, not a trained ablation',
        }
        args.output.mkdir(parents=True)
        atomic_write_json(args.output / 'protocol.json', protocol)
        started = time.monotonic()
        device = configure_device('cuda', deterministic=True)
        loaded = load_frame_joint_inference(checkpoint['path'], expected_sha256=checkpoint['sha256'],
            codec_path=raw['codec']['checkpoint'], codec_sha256=raw['codec']['sha256'], device=device)
        model = loaded['model'].eval()
        rows = []
        with torch.no_grad():
            for sid in ids:
                record = dataset[by_id[sid]]
                meta = meta_for(record)
                batch = prepare_frame_joint_batch(model.target_teacher, collate_clip_records([record]),
                    device=device, normalizer=model, history_frames=4)
                target = batch.coordinate_batch.x.float()
                frames = [4 + i for i in torch.where(batch.query.frame_mask[0])[0].tolist()]
                variants = [('persistence', None, None), ('oracle', None, None)]
                variants += [('generated', steps, seed) for steps in args.steps for seed in args.seeds]
                if args.ablate_memory:
                    variants += [(name, 16, seed) for name in ('zero_history', 'zero_motion') for seed in args.seeds]
                for name, steps, seed in variants:
                    tick = time.monotonic()
                    latent = None
                    if name == 'persistence':
                        future = batch.observed_context.coordinates[-1:].expand_as(batch.target_coordinates).float()
                    elif name == 'oracle':
                        future = model.decoder(batch.observed_context, batch.target_future, batch.query).coordinates.float()
                    else:
                        kwargs = {}
                        if name != 'generated':
                            condition = model.prepare_condition(batch.observed_context)
                            memory, motion = condition.history_memory, condition.motion_context
                            if name == 'zero_history':
                                memory = replace(memory, h=torch.zeros_like(memory.h), v=torch.zeros_like(memory.v))
                            else:
                                if motion is None:
                                    raise ValueError('A1 must retain its motion branch')
                                motion = replace(motion, h=torch.zeros_like(motion.h), v=torch.zeros_like(motion.v))
                            kwargs['prepared_condition'] = FrameJointCondition(memory, motion)
                        latent, _ = euler_sample_frames(model, context=batch.observed_context,
                            query=batch.query, steps=steps, seed=seed, **kwargs)
                        future = model.decoder(batch.observed_context, latent, batch.query).coordinates.float()
                    torch.cuda.synchronize()
                    generation_seconds = time.monotonic() - tick
                    prediction = torch.cat((batch.observed_context.coordinates.float(), future))
                    temporal = temporal_metrics_v3(prediction, target, batch.coordinate_batch, history_frames=4)
                    geometry = bond_errors(prediction, target, batch.coordinate_batch, frames)
                    physical_metrics = sampled_clash_metrics(future, batch, physical)
                    row = dict(sample_id=sid, system=meta['system'], lag_ps=meta['lag_ps'], variant=name,
                        steps=steps, seed=seed, generation_seconds=generation_seconds,
                        metrics=_jsonable({**temporal.metrics, **geometry,
                            **{key: value for key, value in physical_metrics.items() if key != 'samples'}}))
                    if latent is not None:
                        normalized = model.normalize(latent)
                        row['normalized_latent_mse_h'] = float((normalized.h - batch.normalized_target.h).square().mean())
                        row['normalized_latent_mse_v'] = float((normalized.v - batch.normalized_target.v).square().mean())
                    rows.append(row)
                    if sid == ids[0] and name == 'generated' and steps == 16 and seed == args.seeds[0]:
                        coordinates = future.cpu().numpy()
                        np.savez_compressed(args.output / 'parity_coordinates.npz', coordinates=coordinates)
                        atomic_write_json(args.output / 'parity_identity.json', {
                            'sample_id': sid, 'steps': steps, 'seed': seed,
                            'coordinate_sha256': hashlib.sha256(coordinates.tobytes()).hexdigest(),
                        })
                    atomic_write_json(args.output / 'rows.json', rows)
                print(json.dumps({'completed_records': len(rows) // len(variants), 'total_records': len(ids), 'sample_id': sid}), flush=True)
        atomic_write_json(args.output / 'summary.json', {
            'status': 'completed', 'records': len(ids), 'rows': len(rows),
            'gpu_hours': (time.monotonic() - started) / 3600,
            'protocol_sha256': sha256_file(args.output / 'protocol.json'),
            'test_opened': False,
        })
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
