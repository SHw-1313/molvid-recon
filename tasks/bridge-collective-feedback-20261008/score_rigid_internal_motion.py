"""Decompose cached ligand motion; never alter generated coordinates.

All displacements are in Angstrom in the last-observed protein frame. Fit a
proper rotation between successive ligand shapes with fixed atom identities.
Rotation and shape residual squared norms are NOT additive: report their
cross term too. This is a diagnostic of finite sampled paths, not kinetics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molvid.data.store import ClipMMapDataset
from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.complex_motion import aligned_future
from molvid.runtime import atomic_write_json


def observables(draws, record):
    observed = np.asarray(record['x'][3], dtype=np.float64)
    ligand = np.asarray(record['component_id']) == 1
    alignment = np.asarray(record['align_mask'], dtype=bool)
    future = np.stack([aligned_future(x, observed, alignment)[:, ligand] for x in draws])
    initial = np.broadcast_to(observed[ligand], (len(future), 1, ligand.sum(), 3))
    path = np.concatenate((initial, future), axis=1)
    center = path.mean(axis=2, keepdims=True)
    shape = path - center
    left, right = shape[:, :-1], shape[:, 1:]
    u, singular, vh = np.linalg.svd(left.swapaxes(-1, -2) @ right)
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    correction[..., 2, 2] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    rotated = left @ rotation
    rotation_step, residual = rotated - left, right - rotated
    total_step = np.diff(path, axis=1)
    translation_step = np.diff(center, axis=1)
    centered_step = right - left
    # Mean squared per-atom displacement for each of the12 native80ps steps.
    square = lambda x: np.square(x).sum(axis=-1).mean(axis=-1)
    total2, trans2, centered2 = map(square, (total_step, translation_step, centered_step))
    rot2, residual2 = map(square, (rotation_step, residual))
    cross = 2 * (rotation_step * residual).sum(axis=-1).mean(axis=-1)
    # Arithmetic identity on the actual analyzed trajectories, not a model test.
    if not (np.allclose(total2, trans2 + centered2, atol=1e-8, rtol=1e-7)
            and np.allclose(centered2, rot2 + residual2 + cross, atol=1e-8, rtol=1e-7)):
        raise ArithmeticError('Motion decomposition identity failed')
    result = {}
    for name, value in [('total', total2), ('translation', trans2),
                        ('centered', centered2), ('rotation_fit', rot2),
                        ('internal_residual', residual2)]:
        # First summarize a complete path, then average independent draws.
        result[f'{name}_step_rms_A'] = float(np.sqrt(value.mean(axis=1)).mean())
    result['rotation_internal_cross_A2'] = float(cross.mean())
    result['near_rank1_step_fraction'] = float((singular[..., 1] <
        1e-6 * np.maximum(singular[..., 0], 1e-12)).mean())
    angle = np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1)-1)/2, -1, 1))
    result['rotation_angle_mean_deg'] = float(np.degrees(angle).mean())
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    models = {}
    for item in args.generated:
        name, path = item.split('=', 1)
        root = Path(path)
        summary = json.loads((root/'summary.json').read_text())
        source_rows = json.loads((root/'rows.json').read_text())
        index = {r['sample_id']: i for i, r in enumerate(source_rows)}
        counts = {}
        for row in source_rows:
            counts[row['system']] = counts.get(row['system'], 0) + 1
        if (summary['status'] != 'completed' or len(index) != 2*args.systems
                or len(source_rows) != len(index) or len(counts) != args.systems
                or set(counts.values()) != {2} or name in models or name == 'MD'):
            raise ValueError(f'Complete matched panel required: {root}')
        models[name] = (root, index)
    shared = next(iter(models.values()))[1].keys()
    if any(index.keys() != shared for _, index in models.values()):
        raise ValueError('Panel conditions differ')
    data = ClipMMapDataset(args.data/'valid')
    lookup = {f[0]: i for i, f in enumerate(data._clip_index_fields)}
    rows = []
    try:
        for number, sample in enumerate(sorted(shared), 1):
            record = data[lookup[sample]]
            values = {'MD': observables(np.asarray(record['x'])[None, 4:], record)}
            for name, (root, index) in models.items():
                saved = torch.load(root/f'coordinates_{index[sample]:04d}.pt',
                                   map_location='cpu', weights_only=False)
                if saved['sample_id'] != sample or len(saved['coordinates']) != 8:
                    raise ValueError('Expected original eight-draw ensemble')
                values[name] = observables(saved['coordinates'].numpy(), record)
            rows.append(dict(sample_id=sample, system=record['system_id'], values=values))
            if number % 16 == 0:
                print(json.dumps(dict(conditions=number, model_calls=0)), flush=True)
    finally:
        data.close()
    strata = {'all': {r['system'] for r in rows}}
    if args.systems == 64:
        inventory = json.loads((args.data/'summary.json').read_text())
        strata.update({k: set(inventory[k]) for k in ['old16', 'new48']})
    estimates = {}
    for stratum, systems in strata.items():
        selected = [r for r in rows if r['system'] in systems]
        estimates[stratum] = {name: {metric: system_estimate([
            (r['system'], r['values'][name][metric]) for r in selected])
            for metric in selected[0]['values'][name]} for name in ['MD', *models]}
    atomic_write_json(args.output, dict(status='completed', model_calls=0,
        systems=args.systems, conditions=len(rows), forecasts_per_model=8*len(rows),
        sources={name: str(root) for name, (root, _) in models.items()},
        estimates=estimates, rows=rows,
        interpretation=[
            'Descriptive amplitudes, not lower-is-better or proper predictive scores. One MD future per condition.',
            'Translation is heavy-atom centroid motion, not mass-weighted center of mass.',
            'Protein alignment removes reference-frame rigid motion. Fit includes observed-to-first-future transition.',
            'Internal residual is best-rigid-fit shape change, not exclusively bond jitter; includes torsions and legitimate deformations.',
            'Fixed atom correspondence; molecular symmetries are not remapped. Near-collinear geometry can make rotation angles nonunique.',
            'Rotation and residual contributions have a nonzero cross term; do not claim an additive energy fraction.',
            'All64, old16 and new48 are reused development data, not independent confirmation or equilibrium sampling.',
            'Marginal system-bootstrap intervals; paired comparisons require per-system differences, not subtracting CI endpoints.'
        ], default_promotion=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--generated', action='append', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--systems', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
