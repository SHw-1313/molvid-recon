"""Spatial motion coupling from complete cached protein forecasts; no model calls.

Signed residue cross-correlations distinguish concerted and opposing motion.
Covariances retain amplitude. Joint two-residue full-path energy scores also
check the sampled joint distribution rather than only a mean statistic.
Twelve frames / two draws are diagnostics, not equilibrium covariance estimates.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.complex_motion import aligned_future
from molvid.geometry.ca_detail import condition_from_record
from molvid.runtime import atomic_write_json


def coupling(path, left, right):
    centered = path - path.mean(0, keepdims=True)
    increments = np.diff(path, axis=0)
    result = {}
    for name, x in [('position', centered), ('increment', increments)]:
        covariance = (x[:, left] * x[:, right]).sum(-1).mean(0)
        power = np.square(x).sum(-1).mean(0)
        denominator = np.sqrt(power[left] * power[right])
        result[name+'_covariance_A2'] = covariance
        # Zero variance carries no estimated directional correlation, not +1.
        result[name+'_correlation'] = np.divide(covariance, denominator,
            out=np.zeros_like(covariance), where=denominator > 1e-12)
    return result


def joint_pair_path_es(predictions, target, left, right):
    # Each feature vector is the whole 12-frame path of two residues.
    # RMS normalization changes only the overall constant of Euclidean ES.
    p = np.concatenate((predictions[:, :, left], predictions[:, :, right]), -1)
    y = np.concatenate((target[:, left], target[:, right]), -1)
    first = np.sqrt(np.square(p-y[None]).mean(axis=(1, 3))).mean(0)
    # Standard empirical energy score, including diagonal zero distances.
    second = np.zeros_like(first)
    for a in p:
        for b in p:
            second += np.sqrt(np.square(a-b).mean(axis=(0, 2)))
    return first - .5 * second / len(p)**2


def run(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    models, reference = {}, None
    for root in args.generated:
        protocol = json.loads((root/'protocol.json').read_text())
        summary = json.loads((root/'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError('Full completed panels required')
        if reference is not None and any(protocol[k] != reference[k] for k in ('records','query_dt_ps','seeds')):
            raise ValueError('Same observations, physical clocks and forecast draws required')
        reference = protocol
        variants = {row['variant'] for row in json.loads((root/'rows.json').read_text())}
        if len(variants) != 1 or next(iter(variants)) in models:
            raise ValueError('Each root must contain one distinct model')
        models[next(iter(variants))] = root
    rows = []
    for index, entry in enumerate(reference['records']):
        record = torch.load(args.cache/entry['path'], map_location='cpu', weights_only=False)['record']
        xyz = np.asarray(record['x'], dtype=np.float64)
        ca = condition_from_record(record).ca_index.numpy()
        observed = xyz[3]
        align = np.asarray(record['align_mask'], dtype=bool)
        target = aligned_future(xyz[4:], observed, align)[:, ca] - observed[ca][None]
        left, right = np.triu_indices(len(ca), 1)
        separation = np.linalg.norm(observed[ca[left]] - observed[ca[right]], axis=-1)
        strata = {'all': np.ones(len(left), dtype=bool), 'near_lt8A': separation < 8,
                  'mid_8to16A': (separation >= 8) & (separation < 16), 'far_ge16A': separation >= 16}
        pairs = {}
        for stratum, mask in strata.items():
            selected = np.flatnonzero(mask)
            if len(selected) > 4096:
                selected = selected[np.linspace(0, len(selected)-1, 4096, dtype=int)]
            if len(selected):
                pairs[stratum] = (left[selected], right[selected])
        for variant, root in models.items():
            saved = torch.load(root/f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
            predictions = [p for p in saved['predictions'] if p['variant'] == variant]
            if saved['sample_id'] != entry['sample_id'] or [p['seed'] for p in predictions] != reference['seeds']:
                raise ValueError('Incomplete or mismatched condition')
            paths = np.stack([aligned_future(np.asarray(p['coordinates'], dtype=np.float64),
                observed, align)[:, ca] - observed[ca][None] for p in predictions])
            if paths.shape[1:] != target.shape or not np.isfinite(paths).all():
                raise ValueError('Incomplete full path')
            metrics = {}
            for stratum, (i, j) in pairs.items():
                truth = coupling(target, i, j)
                draws = [coupling(p, i, j) for p in paths]
                for key, y in truth.items():
                    mean = np.stack([d[key] for d in draws]).mean(0)
                    metrics[f'{stratum}.{key}.MAE_MD'] = float(np.abs(mean-y).mean())
                    metrics[f'{stratum}.{key}.generated_mean'] = float(mean.mean())
                    metrics[f'{stratum}.{key}.MD_mean'] = float(y.mean())
                metrics[f'{stratum}.joint_pair_full_path_ES_A'] = float(joint_pair_path_es(paths, target, i, j).mean())
            rows.append(dict(system=entry['system'], sample_id=entry['sample_id'], variant=variant, metrics=metrics))
        print(json.dumps(dict(conditions=index+1, models=len(models), model_calls=0)), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        for key, value in row['metrics'].items():
            grouped[row['variant']+'|'+key].append((row['system'],value))
    paired = {}
    for pair in args.pair:
        a, b = pair.split(':')
        lhs = {r['sample_id']:r for r in rows if r['variant'] == a}
        rhs = {r['sample_id']:r for r in rows if r['variant'] == b}
        if set(lhs) != set(rhs) or not lhs:
            raise ValueError('A requested matched model pair is absent')
        paired[pair] = {key: system_estimate([(r['system'],r['metrics'][key]-rhs[s]['metrics'][key])
            for s,r in lhs.items()]) for key in next(iter(lhs.values()))['metrics'] if 'MAE_MD' in key or 'ES_A' in key}
    atomic_write_json(args.output/'rows.json', rows)
    atomic_write_json(args.output/'summary.json', dict(status='completed', conditions=len(reference['records']),
        sources={k:str(v) for k,v in models.items()}, model_calls=0, query_dt_ps=reference['query_dt_ps'],
        estimates={k:system_estimate(v) for k,v in grouped.items()}, paired=paired,
        limitations=['CA-based protein coupling, not all-atom or ligand stereochemistry',
            'One short MD future and two forecast draws; not equilibrium or stable DCCM estimation',
            'Correlation MAE is diagnostic and is not a proper distributional score',
            'Pair-path ES is empirical with only two draws; training-seed variability requires separate replication',
            'Common rigid alignment can induce correlations; identical alignment is applied to all models and MD',
            'Pair selection depends on observed geometry only; at most4096 pairs per distance stratum',
            'Do not maximize positive correlation: genuine opposing motions must also be reproduced'] ))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--generated',type=Path,nargs='+',required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pair',action='append',default=[])
    run(p.parse_args())
