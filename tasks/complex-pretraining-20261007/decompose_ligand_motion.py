"""Separate coherent ligand translation from internal bond/shape fluctuation.

Read cached forecasts only. Interim output includes all commonly completed
systems with both windows, not a validation-selected subset. No hypothesis test
or model-selection claim is made from this running snapshot.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molvid.data.store import ClipMMapDataset
from molvid.evaluation.complex_motion import aligned_future
from molvid.runtime import atomic_write_json


def observables(draws, record):
    reference = np.asarray(record['x'][3], dtype=np.float64)
    mask = np.asarray(record['align_mask'], dtype=bool)
    ligand = np.asarray(record['component_id']) == 1
    x = np.stack([aligned_future(d, reference, mask) for d in draws])
    center = x[:, :, ligand].mean(2)
    relative = x[:, :, ligand] - center[:, :, None]
    edges = np.asarray(record['bond_index'])
    i, j = edges[:, (edges[0] < edges[1]) & ligand[edges[0]] & ligand[edges[1]]]
    lengths = np.linalg.norm(x[:, :, i] - x[:, :, j], axis=-1)
    def rmsf(a):
        return np.sqrt(np.square(a - a.mean(1, keepdims=True)).sum(-1).mean(1)).mean()
    return dict(
        atom_rmsf_A=float(rmsf(x[:, :, ligand])),
        center_rmsf_A=float(rmsf(center)),
        relative_atom_rmsf_A=float(rmsf(relative)),
        mean_radius_gyration_A=float(np.sqrt(np.square(relative).sum(-1).mean(-1)).mean()),
        bond_temporal_std_A=float(lengths.std(1).mean()),
        mean_bond_length_A=float(lengths.mean()),
    )


def run(args):
    roots = {k: Path(v) for k, v in (entry.split('=', 1) for entry in args.generated)}
    rows = {k: json.loads((v / 'rows.json').read_text()) for k, v in roots.items()}
    maps = {k: {r['sample_id']: (i, r) for i, r in enumerate(v)} for k, v in rows.items()}
    common = set.intersection(*(set(v) for v in maps.values()))
    first = next(iter(maps.values()))
    systems = {first[s][1]['system'] for s in common}
    selected = sorted(s for s in common if sum(first[t][1]['system'] == first[s][1]['system']
                      for t in common) == 2)
    if not selected:
        raise ValueError('No common complete system yet')
    dataset = ClipMMapDataset(args.data / 'valid')
    indices = {fields[0]: i for i, fields in enumerate(dataset._clip_index_fields)}
    result = []
    try:
        for sample in selected:
            record = dataset[indices[sample]]
            values = {'MD': observables(np.asarray(record['x'])[None, 4:], record)}
            for name, mapping in maps.items():
                index, _ = mapping[sample]
                saved = torch.load(roots[name] / f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
                if saved['sample_id'] != sample:
                    raise ValueError('Cached forecast sample mismatch')
                values[name] = observables(saved['coordinates'].numpy(), record)
            result.append(dict(sample_id=sample, system=record['system_id'], observables=values))
    finally:
        dataset.close()
    means = {name: {metric: float(np.mean([r['observables'][name][metric] for r in result]))
                   for metric in result[0]['observables'][name]} for name in result[0]['observables']}
    atomic_write_json(args.output, dict(
        status='completed_cached_diagnostic', full_validation=len(selected) == 32,
        systems=len({r['system'] for r in result}), conditions=len(selected),
        input_completed_conditions={k: len(v) for k, v in rows.items()},
        inputs={k: str(v) for k, v in roots.items()}, model_calls=0,
        note='Descriptive running snapshot; no significance/promotion. Relative RMSF includes rigid ligand rotation; bond std is rotation invariant.',
        mean_observables=means, rows=result,
    ))
    print(json.dumps(dict(conditions=len(selected), means=means), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--generated', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args)
