"""Score cached ligand internal paths, separately from rigid-body motion.

Whole-path energy scores retain temporal correlations within each draw. Bond
and nonbonded distances are scored separately, in Angstrom, without fitting
scales to these validation results. This is conditional predictive scoring,
not an equilibrium distribution, chemical validity, or chirality test.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from molvid.data.store import ClipMMapDataset
from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.complex_motion import energy_score
from molvid.runtime import atomic_write_json


def internal_pairs(record):
    ligand = np.flatnonzero(np.asarray(record['component_id']) == 1)
    left, right = np.triu_indices(len(ligand), k=1)
    edges = np.asarray(record['bond_index'])
    covalent = {tuple(sorted((int(a), int(b)))) for a, b in edges.T}
    bonded = np.array([(int(a), int(b)) in covalent
                      for a, b in zip(ligand[left], ligand[right])], dtype=bool)
    return {
        'bond': (ligand[left[bonded]], ligand[right[bonded]]),
        # Includes 1-3, 1-4, longer-range, and any interfragment pairs. This is
        # an internal conformation observable, not a nonbonded-energy term.
        'nonbonded_distance': (ligand[left[~bonded]], ligand[right[~bonded]]),
    }


def scores(coordinates, record, pairs):
    xyz = np.asarray(record['x'], dtype=np.float64)
    observed, truth = xyz[3], xyz[4:]
    draws = np.asarray(coordinates, dtype=np.float64)
    result = {}
    for name, (left, right) in pairs.items():
        if not len(left):
            continue
        pred = np.linalg.norm(draws[:, :, left] - draws[:, :, right], axis=-1)
        target = np.linalg.norm(truth[:, left] - truth[:, right], axis=-1)
        initial = np.linalg.norm(observed[left] - observed[right], axis=-1)
        pred_increment = np.diff(np.concatenate((
            np.broadcast_to(initial, (len(pred), 1, len(initial))), pred), axis=1), axis=1)
        true_increment = np.diff(np.concatenate((initial[None], target)), axis=0)
        result[f'{name}_path_energy_score_A'] = energy_score(pred, target)
        result[f'{name}_increment_path_energy_score_A'] = energy_score(pred_increment, true_increment)
        result[f'{name}_time_std_A'] = float(pred.std(axis=1).mean())
        result[f'{name}_MD_time_std_A'] = float(target.std(axis=0).mean())
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    models = {}
    for root in args.generated:
        summary = json.loads((root / 'summary.json').read_text())
        if summary['status'] != 'completed' or summary['conditions'] != 32:
            raise ValueError(f'Completed32-condition coordinates required: {root}')
        variant = summary['variant']
        if variant in models:
            raise ValueError(f'Duplicate model: {variant}')
        rows = json.loads((root / 'rows.json').read_text())
        models[variant] = (root, {row['sample_id']: i for i, row in enumerate(rows)})
    shared = next(iter(models.values()))[1].keys()
    if any(index.keys() != shared for _, index in models.values()):
        raise ValueError('Models must predict the same32 conditions')
    store = ClipMMapDataset(args.data / 'valid')
    lookup = {fields[0]: i for i, fields in enumerate(store._clip_index_fields)}
    rows = []
    try:
        for sample in sorted(shared):
            record = store[lookup[sample]]
            pairs = internal_pairs(record)
            persistence = np.repeat(np.asarray(record['x'])[3:4], 12, axis=0)
            rows.append(dict(system=record['system_id'], sample_id=sample,
                variant='persistence', metrics=scores(np.stack((persistence, persistence)), record, pairs)))
            for variant, (root, index) in models.items():
                saved = torch.load(root / f'coordinates_{index[sample]:04d}.pt',
                                   map_location='cpu', weights_only=False)
                if saved['sample_id'] != sample or len(saved['coordinates']) != 8:
                    raise ValueError('Expected original eight-draw coordinate ensemble')
                rows.append(dict(system=record['system_id'], sample_id=sample,
                    variant=variant, metrics=scores(saved['coordinates'].numpy(), record, pairs)))
            print(json.dumps(dict(completed_conditions=len(rows)//(len(models)+1), model_calls=0)), flush=True)
    finally:
        store.close()
    grouped = defaultdict(list)
    for row in rows:
        for metric, value in row['metrics'].items():
            grouped[row['variant'], metric].append((row['system'], value))
    estimates = {'|'.join(key): system_estimate(values) for key, values in grouped.items()}
    paired = {}
    for candidate, reference in (pair.split(':') for pair in args.pair):
        left = {r['sample_id']: r for r in rows if r['variant'] == candidate}
        right = {r['sample_id']: r for r in rows if r['variant'] == reference}
        if len(left) != 32 or left.keys() != right.keys():
            raise ValueError(f'Unknown or unmatched pair: {candidate}:{reference}')
        metrics = [key for key in next(iter(left.values()))['metrics'] if 'energy_score' in key]
        paired[f'{candidate}_minus_{reference}'] = {
            metric: system_estimate([(left[s]['system'], left[s]['metrics'][metric]
                                      - right[s]['metrics'][metric]) for s in left
                                     if metric in left[s]['metrics'] and metric in right[s]['metrics']])
            for metric in metrics}
    atomic_write_json(args.output / 'rows.json', rows)
    atomic_write_json(args.output / 'summary.json', dict(status='completed', model_calls=0,
        systems=16, conditions=32, forecasts_reused_per_model=256,
        sources={name: str(root) for name, (root, _) in models.items()},
        estimates=estimates, candidate_minus_reference=paired,
        uncertainty='System-clustered2000 bootstrap; seeds not pooled; marginal95percent CIs',
        limitations=[
            'Exploratory follow-up on reused development data, not independent confirmation',
            'One reference future per condition; proper conditional predictive scores, not equilibrium evidence',
            'Pairwise distances do not identify ligand orientation or chirality; retain existing pocket/clash scores',
            'Nonbonded-distance set means all ligand atom pairs without a direct covalent edge, not force-field energy',
            'Time standard deviations are descriptive amplitudes, not lower-is-better objectives',
            'Unbiased finite-ensemble energy-score estimates can be negative; do not clip or interpret as perfect prediction',
        ]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--generated', type=Path, nargs='+', required=True)
    parser.add_argument('--pair', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(
        '/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2'))
    run(parser.parse_args())
