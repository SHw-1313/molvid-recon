"""Describe online fitting by native task progress; never rank A/B losses."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from molvid.runtime import atomic_write_json


def summarize(path):
    status = json.loads((path / 'status.json').read_text())
    recipe = json.loads((path / 'recipe.json').read_text())['settings']
    rows = [json.loads(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    if (status['status'] != 'completed' or len(rows) != recipe['updates']
            or [r['step'] for r in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('Require a complete nonduplicated fixed-budget training log')
    scheme = recipe['scheme']
    bounds = np.linspace(0., 1., 5)
    if scheme == 'edm':
        scales = recipe['scales']
        bounds = np.exp(np.linspace(math.log(scales['sigma_min_A']),
                                   math.log(scales['sigma_max_A']), 5))
    sections = {}
    for phase, selected in [('first512', rows[:512]), ('last512', rows[-512:])]:
        sections[phase] = {}
        for domain in ['all', 'protein', 'complex']:
            bins = []
            for index in range(4):
                group = [r for r in selected if (domain == 'all' or r['domain'] == domain)
                         and int(np.clip(np.searchsorted(bounds, r['progress_mean'], side='right') - 1, 0, 3)) == index]
                stats = dict(progress_bounds=bounds[index:index+2].tolist(), examples=len(group))
                for metric in ['loss', 'raw_coordinate_MSE_A2', 'grad_norm']:
                    values = np.asarray([r[metric] for r in group])
                    if len(values) == 0 or not np.isfinite(values).all():
                        raise ValueError('Empty or nonfinite online diagnostic bin')
                    stats[metric] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                         p90=float(np.quantile(values, .9)))
                stats['gradient_clipped_fraction'] = float(np.mean([
                    r['grad_norm'] > recipe['clip_grad'] for r in group]))
                bins.append(stats)
            sections[phase][domain] = bins
    return dict(source=str(path), scheme=scheme, updates=len(rows),
                progress='s, dimensionless' if scheme == 'flow' else 'sigma, Angstrom',
                sections=sections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--flow', type=Path, required=True)
    parser.add_argument('--edm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_write_json(args.output, dict(status='completed', model_calls=0,
        flow=summarize(args.flow), edm=summarize(args.edm),
        interpretation=[
            'Descriptive online fitting only: changing weights, randomly sampled progress/noise, different training windows; not a fixed-checkpoint denoising benchmark.',
            'Flow bins are linear s quartiles; EDM bins are log-sigma quartiles. They are not equal noise states and losses or raw target errors cannot rank the recipes.',
            'Flow raw error is vector-field error; EDM raw error is clean-displacement error. Neither is full-source generated trajectory error.',
            'Gradient norm is measured before clipping. Frequent clipping alone does not prove a failed optimizer or justify changing the ongoing paired recipe.',
            'First and last512 updates each include two full256-system cycles, but history, clock and actual window exposure differ; improvements are not controlled paired-window estimates.'
        ]))
