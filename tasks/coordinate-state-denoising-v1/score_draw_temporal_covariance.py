"""Conditional stochastic temporal coupling from paired cached source draws.

For two independent draws, (X1-X2)/sqrt(2) cancels their conditional mean.
Its uncentered second moments estimate conditional covariance. Normalized
finite-sample ratios are descriptive, not unbiased correlation estimators.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.complex_motion import aligned_future
from molvid.runtime import atomic_write_json


def score(a, b):
    residual = (a-b)/np.sqrt(2.)
    mean = (a+b)/2.
    half_difference = (a-b)/2.
    energy = lambda x: float(np.square(x).sum(-1).mean())
    metrics = {'draw_difference_second_moment_A2': energy(residual),
               'ensemble_mean_second_moment_A2': energy(mean),
               'mean_signal_second_moment_unbiased_A2': energy(mean)-energy(half_difference),
               'single_draw_second_moment_A2': (energy(a)+energy(b))/2.}
    # No per-window time centering: that would itself induce anticorrelation.
    for lag in (1, 2, 4, 8):
        left, right = residual[:-lag], residual[lag:]
        cross = float((left*right).sum(-1).mean())
        norm = np.sqrt(energy(left)*energy(right))
        metrics[f'lag{lag}.residual_cross_moment_A2'] = cross
        metrics[f'lag{lag}.residual_cosine'] = cross/norm if norm > 1e-12 else None
    # Only future-to-future increments; no artificial zero at the observation.
    steps = np.diff(residual, axis=0)
    left, right = steps[:-1], steps[1:]
    cross = float((left*right).sum(-1).mean())
    norm = np.sqrt(energy(left)*energy(right))
    metrics['residual_increment_lag1_cosine'] = cross/norm if norm > 1e-12 else None
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--generated', type=Path, nargs='+', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    models, reference = {}, None
    for root in args.generated:
        summary = json.loads((root/'summary.json').read_text())
        protocol = json.loads((root/'protocol.json').read_text())
        if summary['status'] != 'completed' or summary['systems'] != 48 or summary['rows'] != 192:
            raise ValueError('Complete48-system192-draw panels required')
        if len(protocol['records']) != 96 or protocol['seeds'] != [17, 29]:
            raise ValueError('Expected96 conditions and both registered draws')
        if reference and any(reference[k] != protocol[k] for k in ('records', 'query_dt_ps', 'seeds')):
            raise ValueError('Different conditions/time grids/draws')
        variants = {row['variant'] for row in json.loads((root/'rows.json').read_text())}
        if len(variants) != 1 or next(iter(variants)) in models:
            raise ValueError('Distinct single variants required')
        models[next(iter(variants))] = root
        reference = protocol
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, entry in enumerate(reference['records']):
        record = torch.load(args.cache/entry['path'], map_location='cpu', weights_only=False)['record']
        observed = np.asarray(record['x'], dtype=np.float64)[3]
        align = np.asarray(record['align_mask'], dtype=bool)
        valid = np.asarray(record['loss_mask'], dtype=bool)
        for variant, root in models.items():
            saved = torch.load(root/f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
            draws = [d for d in saved['predictions'] if d['variant'] == variant]
            if saved['sample_id'] != entry['sample_id'] or [d['seed'] for d in draws] != [17, 29]:
                raise ValueError('Draw/condition mismatch')
            arrays = [np.asarray(d['coordinates'], dtype=np.float64) for d in draws]
            if any(a.shape != (12, len(observed), 3) or not np.isfinite(a).all() for a in arrays):
                raise ValueError('Complete finite12-frame raw forecasts required')
            displacement = [(aligned_future(a, observed, align)-observed)[..., valid, :] for a in arrays]
            rows.append(dict(system=entry['system'], sample_id=entry['sample_id'], variant=variant,
                             metrics=score(*displacement)))
        print(json.dumps(dict(conditions=index+1, model_calls=0)), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        for key, value in row['metrics'].items():
            if value is not None:
                grouped[row['variant']+'|'+key].append((row['system'], value))
    atomic_write_json(args.output/'rows.json', rows)
    atomic_write_json(args.output/'summary.json', dict(status='completed', systems=48, conditions=96,
        query_dt_ps=reference['query_dt_ps'], model_calls=0, optimizer_updates=0,
        sources={k:str(v) for k,v in models.items()},
        estimates={k:system_estimate(v) for k,v in grouped.items()},
        definition='Residual=(draw17-draw29)/sqrt(2); no time centering. Cross moments estimate conditional stochastic covariance assuming independent draws. Ratios are descriptive pooled-atom finite-window cosines.',
        mean_signal='Mean((X1+X2)/2)^2 minus Mean((X1-X2)/2)^2, with squared vector norm: unbiased mean-signal energy assuming independent draws; may be negative and is not clipped.',
        alignment='Established whole-protein per-frame fit to fixed last observed frame; same valid atom and fit masks as shared evaluation',
        limitations=['Two draws per condition and reused development48; broad exploratory uncertainty',
            'One MD future cannot identify true conditional stochastic covariance; no MD covariance target or error claimed',
            'Higher persistence is not necessarily correct; read with geometry/motion/distribution scores',
            'Conditional covariance after fixed-reference alignment, not equilibrium covariance or physical velocity',
            'Noise-free ratios are undefined and omitted with estimator counts retained',
            'No conditional-mean accuracy, mechanism causality, or thermodynamic claim'], promotion=False))


if __name__ == '__main__':
    main()
