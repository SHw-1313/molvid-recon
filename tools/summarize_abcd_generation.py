"""Summarize the frozen, balanced generation diagnosis; never select a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from molvid.runtime import atomic_write_json, sha256_file


def metrics(row):
    m = row['metrics']
    return {
        'bond_rmse_A': m['bond_rmse'],
        'clashes_per_1000_atoms': m['clashes_per_1000_atoms'],
        'vdw_severity_A2_per_atom': m['vdw_severity_A2_per_atom'],
        'residue_rmsf_mae_A': m['rmsf']['residue_profile']['mae_A'],
        'mean_rmsf_prediction_A': m['rmsf']['system_mean_prediction_A'],
        'mean_rmsf_target_A': m['rmsf']['system_mean_target_A'],
        'msd_curve_mae_A2': m['msd']['curve_mae_A2'],
        'distance_increment_mae_A': m['internal']['distance_increment']['mae_A'],
        'torsion_increment_mae_radian': m['internal']['torsion_increment_radian']['mae_radian'],
        'generation_seconds': row['generation_seconds'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / 'protocol.json').read_text())
    completion = json.loads((args.run / 'summary.json').read_text())
    rows = json.loads((args.run / 'rows.json').read_text())
    if completion['status'] != 'completed' or protocol['test_opened'] is not False:
        raise ValueError('only completed validation diagnosis may be summarized')
    variants = [('oracle', None, None), ('persistence', None, None)]
    variants += [('generated', step, seed) for step in protocol['steps'] for seed in protocol['seeds']]
    if protocol['ablate_memory']:
        variants += [(name, 16, seed) for name in ('zero_history', 'zero_motion') for seed in protocol['seeds']]
    expected = {(sid, *v) for sid in protocol['sample_ids'] for v in variants}
    actual = {(r['sample_id'], r['variant'], r['steps'], r['seed']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('duplicate, missing, or unexpected diagnostic rows')
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = row['variant'] + (f"_euler{row['steps']}" if row['steps'] is not None else '')
        groups[label][row['system']].append(metrics(row))
    systems = sorted(next(iter(groups.values())))
    per_system, means = {}, {}
    for label, by_system in groups.items():
        if sorted(by_system) != systems:
            raise ValueError('systems differ between diagnostic variants')
        per_system[label] = {
            system: {key: float(np.mean([r[key] for r in records])) for key in records[0]}
            for system, records in by_system.items()
        }
        means[label] = {key: float(np.mean([r[key] for r in per_system[label].values()]))
                        for key in next(iter(per_system[label].values()))}
    baseline = 'generated_euler16'
    comparisons = {}
    if baseline in means:
        rng = np.random.default_rng(20261004)
        indices = rng.integers(0, len(systems), size=(10000, len(systems)))
        for label in means:
            if label == baseline:
                continue
            comparisons[label] = {}
            for key in means[label]:
                delta = np.array([per_system[label][s][key] - per_system[baseline][s][key] for s in systems])
                comparisons[label][key] = {
                    'paired_mean_delta': float(delta.mean()),
                    'relative_percent': (100 * float(delta.mean()) / means[baseline][key]
                                         if means[baseline][key] != 0 else None),
                    'exploratory_system_bootstrap_ci95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
                }
    result = {'schema': 'molvid.abcd.generation_diagnostic_summary.v1',
        'systems': systems, 'records': len(protocol['sample_ids']), 'rows': len(rows),
        'aggregation': 'mean seeds and lags within each system, then equal system mean',
        'limitations': 'R1/anchor0/H4 short clips, one training seed; zero-memory is OOD sensitivity, not trained ablation; bootstrap exploratory',
        'source_protocol_sha256': sha256_file(args.run / 'protocol.json'),
        'source_rows_sha256': sha256_file(args.run / 'rows.json'),
        'means': means, 'per_system': per_system, 'comparisons_to_euler16': comparisons,
        'test_opened': False}
    atomic_write_json(args.run / 'analysis.json', result)
    print(json.dumps({'means': means, 'comparisons_to_euler16': comparisons}, indent=2))


if __name__ == '__main__':
    main()
