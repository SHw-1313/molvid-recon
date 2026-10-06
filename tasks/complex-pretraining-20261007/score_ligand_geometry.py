"""CPU geometry diagnostics of completed cached full-complex forecasts.

Uses the existing internal-angle/torsion definitions and frozen physical table.
No model calls, fitting, hidden-frame conditioning or prediction-chosen cutoffs.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from molvid.data.store import ClipMMapDataset
from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.review_diagnostics import internal_geometry, chemistry_indices
from molvid.losses.sampled_physical import _topological_distances
from molvid.runtime import atomic_write_json
from molvid.training.predictive_data import RELEASE


def topology(record, physical):
    roles = np.asarray(record['component_id'])
    ligand, protein = np.flatnonzero(roles == 1), np.flatnonzero(roles == 0)
    bonds = torch.as_tensor(record['bond_index'], dtype=torch.long)
    excluded, one_four = _topological_distances(SimpleNamespace(
        num_atoms=len(roles), covalent_bond_index=bonds))
    radii = np.asarray([physical['vdw_radii_A'][str(int(t))] for t in record['atype']])
    def thresholds(target, internal):
        table = radii[ligand, None] + radii[None, target] - physical['allowed_overlap_A']
        for i, atom in enumerate(ligand):
            table[i, np.isin(target, list(one_four[atom]))] -= (
                physical['one_four_overlap_A'] - physical['allowed_overlap_A'])
            table[i, np.isin(target, list(excluded[atom]))] = 0
        if internal:
            table = np.triu(table, k=1)
        return np.maximum(table, 0)
    local = np.full(len(roles), -1, dtype=np.int64)
    local[ligand] = np.arange(len(ligand))
    edges = bonds.numpy()
    selected = (roles[edges[0]] == 1) & (roles[edges[1]] == 1)
    ligand_bonds = torch.as_tensor(local[edges[:, selected]], dtype=torch.long)
    return ligand, protein, ligand_bonds, thresholds(ligand, True), thresholds(protein, False)


def score(draw, truth, observed, topology_data):
    ligand, protein, bonds, internal_threshold, cross_threshold = topology_data
    pred = torch.as_tensor(draw[:, ligand], dtype=torch.float32)
    target = torch.as_tensor(truth[:, ligand], dtype=torch.float32)
    geometry = internal_geometry(pred, target, bonds)
    result = {k: geometry[k] for k in ('angle_cosine_rmse', 'torsion_circular_error',
        'torsion_increment_circular_error') if k in geometry}
    _, volumes, _ = chemistry_indices(bonds, len(ligand))
    if volumes:
        center, a, b, c = np.asarray(volumes).T
        def volume(x):
            return (np.cross(x[..., a, :] - x[..., center, :], x[..., b, :] - x[..., center, :])
                    * (x[..., c, :] - x[..., center, :])).sum(-1)
        reference = volume(np.asarray(observed)[ligand])
        valid = np.abs(reference) > .2  # Same observed-volume threshold as existing CA diagnostic.
        if valid.any():
            v = volume(np.asarray(draw)[:, ligand])[:, valid]
            result['observed_nonplanar_volume_sign_flip_fraction'] = float((v * reference[valid] < 0).mean())
        result['observed_nonplanar_centers'] = int(valid.sum())
    internal_counts, cross_counts, internal_severity, cross_severity = [], [], [], []
    for frame in np.asarray(draw):
        li = frame[ligand]
        internal = np.maximum(internal_threshold - np.linalg.norm(li[:, None] - li[None, :], axis=-1), 0)
        cross = np.maximum(cross_threshold - np.linalg.norm(li[:, None] - frame[None, protein], axis=-1), 0)
        internal_counts.append(float((internal > 0).sum()) * 1000 / len(ligand))
        cross_counts.append(float((cross > 0).sum()) * 1000 / len(ligand))
        internal_severity.append(float(np.square(internal).sum()) / len(ligand))
        cross_severity.append(float(np.square(cross).sum()) / len(ligand))
    result.update(ligand_internal_clashes_per_1000_ligand_atoms=float(np.mean(internal_counts)),
        protein_ligand_clashes_per_1000_ligand_atoms=float(np.mean(cross_counts)),
        ligand_internal_overlap_A2_per_ligand_atom=float(np.mean(internal_severity)),
        protein_ligand_overlap_A2_per_ligand_atom=float(np.mean(cross_severity)))
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.generated/'summary.json').read_text())
    if summary['status'] != 'completed' or summary['conditions'] != 32:
        raise ValueError('Completed full-complex generation required')
    records = json.loads((args.generated/'rows.json').read_text())
    release = json.loads(RELEASE.read_text())
    recipe = Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    physical = json.loads(recipe.read_text())['physical']
    dataset = ClipMMapDataset(args.data/'valid')
    index = {f[0]: i for i, f in enumerate(dataset._clip_index_fields)}
    rows = []
    try:
        for position, entry in enumerate(records):
            record = dataset[index[entry['sample_id']]]
            saved = torch.load(args.generated/f'coordinates_{position:04d}.pt', map_location='cpu', weights_only=False)
            if saved['sample_id'] != entry['sample_id']:
                raise ValueError('Cached coordinate identity mismatch')
            truth, observed = np.asarray(record['x'])[4:], np.asarray(record['x'])[3]
            topo = topology(record, physical)
            values = [('MD', -1, truth), ('persistence', -1, np.repeat(observed[None], len(truth), axis=0))]
            values += [(summary['variant'], seed, draw.numpy()) for seed, draw in zip(saved['seeds'], saved['coordinates'])]
            for variant, seed, draw in values:
                rows.append(dict(system=entry['system'], sample_id=entry['sample_id'], variant=variant, seed=seed,
                    metrics=score(draw, truth, observed, topo)))
            atomic_write_json(args.output/'rows.json', rows)
            print(json.dumps(dict(conditions=position+1, rows=len(rows), model_calls=0)), flush=True)
    finally:
        dataset.close()
    groups = defaultdict(list)
    for row in rows:
        for metric, value in row['metrics'].items():
            groups[(row['variant'], metric)].append((row['system'], value))
    atomic_write_json(args.output/'summary.json', dict(status='completed', model_calls=0, systems=16, conditions=32,
        generated=str(args.generated), physical_recipe=str(recipe), physical=physical,
        estimates={'|'.join(k): system_estimate(v) for k, v in groups.items()},
        limitations=[
            'Angle/torsion differences to one MD future are conformation errors, not necessarily invalidity',
            'Ordered nonplanar-center sign flips are not complete R/S assignments; planar and unrecognized centers omitted',
            'Clashes use the existing diagnostic radii/tolerances, not a force-field energy or thermodynamic estimate',
            'No hydrogen geometry or complete aromaticity/bond-order stereochemistry assessment',
            'System-equal aggregation; inference draws are not independent biological replicates',
        ]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2'))
    run(parser.parse_args())
