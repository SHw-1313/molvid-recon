"""Spatially coordinated increments from complete cached protein forecasts.

Diagnostic only: no model calls, training, future-selected pairs or new samples.
Reuses the established fixed-observed-reference alignment and fair path score.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.complex_motion import aligned_future, energy_score
from molvid.geometry.ca_detail import condition_from_record
from molvid.runtime import atomic_write_json


def pairs(record):
    valid = np.asarray(record['loss_mask'], dtype=bool)
    bonds = np.asarray(record['bond_index'], dtype=int).T
    bonds = np.unique(np.sort(bonds, axis=1), axis=0)
    bonds = bonds[(bonds[:, 0] != bonds[:, 1]) & valid[bonds].all(1)]
    ca = condition_from_record(record).ca_index.numpy()
    ca = ca[valid[ca]]
    i, j = np.triu_indices(len(ca), k=1)
    observed = np.asarray(record['x'])[3]
    distance = np.linalg.norm(observed[ca[i]] - observed[ca[j]], axis=-1)
    keep = (distance >= 4.5) & (distance <= 8.)
    contacts = np.stack((ca[i[keep]], ca[j[keep]]), axis=1)
    if len(contacts) > 4096:
        selection = np.random.default_rng(4109).choice(len(contacts), 4096, replace=False)
        contacts = contacts[np.sort(selection)]
    return {'covalent': bonds, 'observed_CA_shell4p5_8A': contacts}


def increments(xyz, observed, align):
    future = aligned_future(np.asarray(xyz, dtype=np.float64), observed, align)
    return np.diff(np.concatenate((observed[None], future)), axis=0)


def edge_values(step, edges):
    a, b = step[:, edges[:, 0]], step[:, edges[:, 1]]
    denominator = np.sqrt(np.square(a).sum((0, 2)) * np.square(b).sum((0, 2)))
    # Numerical floor in A^2 summed across the finite window, not clipping motion.
    valid = denominator > 1e-12
    cosine = np.full(len(edges), np.nan)
    cosine[valid] = (a*b).sum((0, 2))[valid] / denominator[valid]
    return dict(increment_cosine=cosine, relative=a-b,
                common_step_rms_A=float(np.sqrt(np.square((a+b)/2).sum(-1).mean())),
                relative_step_rms_A=float(np.sqrt(np.square(a-b).sum(-1).mean())))


def score(steps, true_step, edge_sets):
    metrics = {}
    for name, edges in edge_sets.items():
        metrics[f'{name}.pairs'] = len(edges)
        if not len(edges):
            continue
        truth = edge_values(true_step, edges)
        values = [edge_values(s, edges) for s in steps]
        cosines = np.stack([v['increment_cosine'] for v in values])
        valid = np.isfinite(cosines).all(0) & np.isfinite(truth['increment_cosine'])
        metrics[f'{name}.defined_pair_fraction'] = float(valid.mean())
        metrics[f'{name}.increment_cosine'] = float(cosines[:, valid].mean()) if valid.any() else None
        metrics[f'{name}.increment_cosine_absolute_MD_error'] = float(np.abs(
            cosines[:, valid].mean(0)-truth['increment_cosine'][valid]).mean()) if valid.any() else None
        for key in ('common_step_rms_A', 'relative_step_rms_A'):
            metrics[f'{name}.{key}'] = float(np.mean([v[key] for v in values]))
        metrics[f'{name}.relative_increment_path_fair_ES_A'] = energy_score(
            np.stack([v['relative'] for v in values]), truth['relative'])
    return metrics


def run(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    models, reference = {}, None
    for root in args.generated:
        protocol = json.loads((root/'protocol.json').read_text())
        summary = json.loads((root/'summary.json').read_text())
        if summary['status'] != 'completed' or summary['systems'] != 48 or summary['rows'] != 192:
            raise ValueError('Complete48-system192-draw panels required')
        if len(protocol['records']) != 96 or protocol['seeds'] != [17, 29]:
            raise ValueError('Expected96 conditions and both declared draws')
        if reference and any(reference[k] != protocol[k] for k in ('records', 'query_dt_ps', 'seeds')):
            raise ValueError('Different forecast conditions')
        variants = {r['variant'] for r in json.loads((root/'rows.json').read_text())}
        if len(variants) != 1 or next(iter(variants)) in models:
            raise ValueError('One distinct raw variant per panel required')
        models[next(iter(variants))] = root
        reference = protocol
    rows = []
    for index, entry in enumerate(reference['records']):
        record = torch.load(args.cache/entry['path'], map_location='cpu', weights_only=False)['record']
        xyz = np.asarray(record['x'], dtype=np.float64)
        observed, align = xyz[3], np.asarray(record['align_mask'], dtype=bool)
        edge_sets = pairs(record)
        true_step = increments(xyz[4:], observed, align)
        # MD and zero-motion values are diagnostics, not predictive ensembles.
        for variant, steps in [('MD', [true_step, true_step]),
                               ('persistence', [np.zeros_like(true_step)]*2)]:
            rows.append(dict(system=entry['system'], sample_id=entry['sample_id'], variant=variant,
                             metrics=score(steps, true_step, edge_sets)))
        for variant, root in models.items():
            saved = torch.load(root/f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
            draws = [p for p in saved['predictions'] if p['variant'] == variant]
            if saved['sample_id'] != entry['sample_id'] or [d['seed'] for d in draws] != [17, 29]:
                raise ValueError('Wrong identities or missing draws')
            arrays = [np.asarray(d['coordinates'], dtype=np.float64) for d in draws]
            if any(a.shape != xyz[4:].shape or not np.isfinite(a).all() for a in arrays):
                raise ValueError('Incomplete/nonfinite future')
            steps = [increments(a, observed, align) for a in arrays]
            rows.append(dict(system=entry['system'], sample_id=entry['sample_id'], variant=variant,
                             metrics=score(steps, true_step, edge_sets)))
        print(json.dumps(dict(conditions=index+1, model_calls=0)), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        for key, value in row['metrics'].items():
            if value is not None:
                grouped[row['variant']+'|'+key].append((row['system'], value))
    paired = {}
    for pair in args.pair:
        left_name, right_name = pair.split(':')
        left = {r['sample_id']: r for r in rows if r['variant'] == left_name}
        right = {r['sample_id']: r for r in rows if r['variant'] == right_name}
        if len(left) != 96 or left.keys() != right.keys():
            raise ValueError('Incomplete paired panels')
        paired[pair] = {key: system_estimate([(left[s]['system'],
            left[s]['metrics'][key]-right[s]['metrics'][key]) for s in left
            if left[s]['metrics'].get(key) is not None and right[s]['metrics'].get(key) is not None])
            for key in next(iter(left.values()))['metrics']
            if key.endswith(('absolute_MD_error', 'fair_ES_A'))}
    atomic_write_json(args.output/'rows.json', rows)
    atomic_write_json(args.output/'summary.json', dict(status='completed', systems=48, conditions=96,
        model_calls=0, query_dt_ps=reference['query_dt_ps'], sources={k:str(v) for k,v in models.items()},
        estimates={k:system_estimate(v) for k,v in grouped.items()}, paired=paired,
        definition='Edge cosine=sum_t(dx_i dot dx_j)/sqrt(sum_t|dx_i|^2 sum_t|dx_j|^2);12 increments including observed-to-first-future;not mean-subtracted covariance',
        alignment='Each whole protein frame fitted to last observation with existing align_mask;no separate residue/component fits',
        pairs='All valid unique covalent edges;observed CA distance shell4.5..8A,max4096 fixed-seed pairs;never future-selected',
        limitations=['Finite-window uncentered increment coupling,not an equilibrium DCCM or physical velocity at finer time resolution',
            'Cosine absolute MD error is descriptive,not a proper distribution score;relative-increment fair ES is a proper score only for that transformed observable',
            'Larger coherence is not automatically better;compare to MD and motion/geometry jointly',
            'Rigid protein fitting can induce correlations;both MD and generated paths use the same fixed observed reference',
            'Zero-motion correlations undefined,coverage retained;no favorable imputation',
            'Two draws and one MD future per condition;development-only marginal system-bootstrap intervals',
            'MD self-scores and duplicated persistence paths are diagnostics,not fair competing learned ensembles'],
        default_promotion=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--generated', type=Path, nargs='+', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pair', action='append', default=[])
    run(parser.parse_args())
