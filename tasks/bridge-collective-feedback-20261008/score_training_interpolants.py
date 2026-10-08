"""TRAIN-only geometry experiment for straight versus block-rotation paths.

No neural inference/training, no source-encoder sample or bridge noise. This
isolates the deterministic interpolation path and is not a generative score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from molvid.data.bridge_mixture import BridgeMixture
from molvid.runtime import atomic_write_json


def pairs_below_1A(x):
    tree = cKDTree(x)
    return int((tree.count_neighbors(tree, 1.) - len(x)) // 2)


def rotation_path(source, target, group, time):
    result = np.empty_like(source)
    for key in np.unique(group):
        mask = group == key
        a, b = source[mask], target[mask]
        ca, cb = a.mean(0), b.mean(0)
        ra, rb = a-ca, b-cb
        rotation = np.eye(3)
        if len(a) >= 3:
            u, singular, vh = np.linalg.svd(ra.T @ rb)
            if singular[1] > 1e-8 * max(singular[0], 1.):
                sign = np.array([1., 1., np.linalg.det(u @ vh)])
                rotation = ((u * sign) @ vh).T
        fraction = Rotation.from_rotvec(time * Rotation.from_matrix(rotation).as_rotvec()).as_matrix()
        shape = (1-time)*ra + time*(rb @ rotation)
        result[mask] = (1-time)*ca + time*cb + shape @ fraction.T
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    settings = json.loads(args.config.read_text())
    mixture = BridgeMixture(settings['data'], 71)
    selected = {'protein': 0, 'complex': 0}
    rows = []
    try:
        while min(selected.values()) < args.systems_per_domain:
            batch, task, _, _, identity = mixture.next()
            domain = identity['domain']
            if task != 'dynamic' or selected[domain] >= args.systems_per_domain:
                continue
            selected[domain] += 1
            x = batch.x.numpy().astype(np.float64)
            edges = batch.bond_index.numpy()
            components = batch.component_id.numpy()
            keys = np.stack([components, batch.block_id.numpy()], -1)
            _, groups = np.unique(keys, axis=0, return_inverse=True)
            within = groups[edges[0]] == groups[edges[1]]
            lengths = np.linalg.norm(x[:, edges[0]]-x[:, edges[1]], axis=-1)
            for history in (4, 8):
                source = x[history-1]
                for query in (history, 15):
                    target = x[query]
                    endpoint_count = max(pairs_below_1A(source), pairs_below_1A(target))
                    for flow_time in (.25, .5, .75):
                        ideal_lengths = (1-flow_time)*lengths[history-1] + flow_time*lengths[query]
                        paths = {
                            'straight': (1-flow_time)*source+flow_time*target,
                            'component_rotation': rotation_path(source, target, components, flow_time),
                            'block_rotation': rotation_path(source, target, groups, flow_time),
                        }
                        for name, state in paths.items():
                            error2 = (np.linalg.norm(state[edges[0]]-state[edges[1]], axis=-1)-ideal_lengths)**2
                            row = dict(system=identity['system'], sample=batch.sample_id[0], domain=domain,
                                clock=identity['time_bucket_id'], history=history, query=query,
                                physical_lag_ps=float(batch.time_ps[0,query]-batch.time_ps[0,history-1]),
                                flow_time=flow_time, path=name,
                                close_pairs_excess_per1000=1000*(pairs_below_1A(state)-endpoint_count)/len(state),
                                bond_length_interpolation_rmse_A=float(np.sqrt(error2.mean())),
                                within_block_bond_rmse_A=float(np.sqrt(error2[within].mean())) if within.any() else None,
                                cross_block_bond_rmse_A=float(np.sqrt(error2[~within].mean())) if (~within).any() else None)
                            rows.append(row)
            print(json.dumps(dict(completed_systems=selected, rows=len(rows))), flush=True)
    finally:
        mixture.close()
    summaries = {}
    for domain in selected:
        summaries[domain] = {}
        for path in ('straight', 'component_rotation', 'block_rotation'):
            subset = [r for r in rows if r['domain'] == domain and r['path'] == path]
            summaries[domain][path] = {key: float(np.mean([r[key] for r in subset if r[key] is not None]))
                for key in ('close_pairs_excess_per1000', 'bond_length_interpolation_rmse_A',
                            'within_block_bond_rmse_A', 'cross_block_bond_rmse_A')}
    atomic_write_json(args.output, dict(status='completed', model_calls=0, training_updates=0,
        selection='first fixed32 dynamic TRAIN systems per domain from seed71 mixture; same clips/queries/flow times across paths',
        systems=selected, summaries=summaries, rows=rows,
        limitations=['Clean deterministic paths only; no learned source, source perturbation or bridge noise',
            'All unique atom pairs below1A including bonded pairs; not a full van-der-Waals clash score',
            'Block-wise rotation does NOT preserve bonds between blocks or prevent interblock collisions',
            'Geometry of training interpolants is not generated quality or evidence of correct thermodynamics',
            'No change to actual training, reference models, held-out data or sampling protocol']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--systems-per-domain', type=int, default=32)
    run(parser.parse_args())
