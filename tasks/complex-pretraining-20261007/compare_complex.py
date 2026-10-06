"""System-paired comparison of completed, cached full-complex forecasts.

No model calls and no re-generation. Each row already scores the eight-draw
ensemble. The two windows remain clustered within their biological system.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from molvid.evaluation.ca_detail import system_estimate
from molvid.runtime import atomic_write_json


METRICS = (
    'ligand_bond_rmse_A', 'ligand_rmsf_mae_A', 'ligand_center_msd_mae_A2',
    'ligand_center_path_energy_score_A', 'pocket_distance_path_energy_score_A',
    'pocket_contact_brier', 'all_residue_contact_occupancy_mae',
)


def load(root):
    summary = json.loads((root / 'summary.json').read_text())
    if summary['status'] != 'completed' or summary['conditions'] != 32:
        raise ValueError(f'Full 32-condition evaluation is not completed: {root}')
    rows = json.loads((root / 'rows.json').read_text())
    result = {r['sample_id']: r for r in rows}
    if len(rows) != 32 or len(result) != 32:
        raise ValueError('Duplicate or missing condition')
    systems = [r['system'] for r in rows]
    if len(set(systems)) != 16 or any(systems.count(s) != 2 for s in systems):
        raise ValueError('Expected two windows per system')
    return summary, result


def compare(candidate, reference, output):
    candidate_summary, a = load(candidate)
    if reference == Path('persistence'):
        reference_summary = dict(variant='last_observed_persistence')
        b = {key: dict(row, metrics=row['persistence']) for key, row in a.items()}
    else:
        reference_summary, b = load(reference)
    if a.keys() != b.keys():
        raise ValueError('Candidate/reference conditions differ')
    results = {}
    for metric in METRICS:
        values = {key: [] for key in (
            'candidate', 'reference', 'persistence', 'candidate_minus_reference',
            'candidate_minus_persistence', 'reference_minus_persistence')}
        for sample in sorted(a):
            x, y = a[sample], b[sample]
            if x['system'] != y['system']:
                raise ValueError('System identity mismatch')
            av, bv = x['metrics'][metric], y['metrics'][metric]
            pv = x['persistence'][metric]
            if av is None or bv is None or pv is None:
                if not (av is None and bv is None and pv is None and y['persistence'][metric] is None):
                    raise ValueError('Metric availability differs across models')
                continue
            if not np.isfinite([av, bv, pv]).all():
                raise ValueError('Non-finite score')
            if not np.isclose(pv, y['persistence'][metric], rtol=1e-7, atol=1e-9):
                raise ValueError('Different reference future or persistence calculation')
            for key, value in zip(values, (av, bv, pv, av-bv, av-pv, bv-pv)):
                values[key].append((x['system'], value))
        results[metric] = {key: system_estimate(v) for key, v in values.items()}
        results[metric]['conditions'] = len(values['candidate'])
        results[metric]['systems'] = len({s for s, _ in values['candidate']})
    result = dict(
        status='completed', candidate=str(candidate), reference=str(reference),
        candidate_variant=candidate_summary['variant'],
        reference_variant=reference_summary['variant'],
        systems=16, conditions=32, draws_per_condition=8, model_calls=0,
        metrics=results, lower_is_better=list(METRICS),
        uncertainty='Paired system bootstrap, 2000 resamples; seeds not pooled; marginal CIs, no multiplicity correction',
        limitations=[
            'Official validation only; family/scaffold independence and pretraining absence not established',
            'Single reference MD future per condition; short-horizon predictive scores, not equilibrium evidence',
            'PVB uses released MISATO weights and native H1; Molvid uses H4; training domains/budgets may differ',
            'Bond error does not cover all stereochemistry; low COM error alone can hide distorted or motionless ligands',
        ], default_promotion=False,
    )
    atomic_write_json(output, result)
    print(json.dumps({m: results[m]['candidate_minus_reference'] for m in METRICS}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    compare(args.candidate, args.reference, args.output)
