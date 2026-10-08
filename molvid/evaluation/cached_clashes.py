"""Exact uncapped CPU clash scoring for cached coordinates, not a model path.

Uses the existing frozen radius/overlap table and graph-distance exclusions.
KD-tree search only eliminates pairs outside every possible clash threshold.
Chunked queries bound memory even for catastrophically collapsed structures.
"""
import numpy as np
from scipy.spatial import cKDTree

from ..losses.sampled_physical import _topological_distances


class CachedClashScorer:
    def __init__(self, topology, loss_mask, config):
        self.nodes = np.flatnonzero(np.asarray(loss_mask, dtype=bool))
        self.n = len(self.nodes)
        if not self.n:
            raise ValueError('Valid atom set is empty')
        self.component = topology.component_id.cpu().numpy()[self.nodes]
        types = topology.atom_type.cpu().numpy()[self.nodes]
        self.radii = np.asarray([config.vdw_radii_A[str(t)] for t in types], dtype=np.float32)
        self.allowed = np.float32(config.allowed_overlap_A)
        self.one_four_allowed = np.float32(config.one_four_overlap_A)
        if min(self.allowed, self.one_four_allowed) < 0:
            raise ValueError('Overlap allowances must be nonnegative')
        excluded, one_four = _topological_distances(topology)
        local = {int(node): i for i, node in enumerate(self.nodes)}
        self.excluded = [np.asarray([local[b] for b in excluded[int(a)] if b in local], dtype=np.int64)
                         for a in self.nodes]
        self.one_four = [np.asarray([local[b] for b in one_four[int(a)] if b in local], dtype=np.int64)
                         for a in self.nodes]
        # Search performed in double precision on the same FP32 positions as
        # the reference metric. Margin protects FP32 threshold roundoff only.
        self.radius = float(2 * self.radii.max()) + 1e-5

    def frame(self, coordinates):
        xyz = np.asarray(coordinates, dtype=np.float32)[self.nodes]
        if not np.isfinite(xyz).all():
            raise ValueError('Nonfinite generated coordinates; cannot score as zero clashes')
        tree = cKDTree(xyz)
        counts = dict(all=0, protein=0, ligand=0, cross_component=0)
        severity = {key: 0. for key in counts}
        affected = {key: np.zeros(self.n, dtype=bool) for key in counts}
        for start in range(0, self.n, 128):
            neighbors = tree.query_ball_point(xyz[start:start+128], self.radius, eps=0., workers=1)
            for i, values in enumerate(neighbors, start):
                j = np.asarray(values, dtype=np.int64)
                j = j[(j > i) & ~np.isin(j, self.excluded[i])]
                if not len(j):
                    continue
                threshold = np.maximum(self.radii[i]+self.radii[j]-np.where(
                    np.isin(j, self.one_four[i]), self.one_four_allowed, self.allowed), 0.)
                distance = np.linalg.norm(xyz[i]-xyz[j], axis=-1)
                overlap = np.maximum(threshold-distance, 0.)
                j, overlap = j[overlap > 0], overlap[overlap > 0]
                categories = dict(all=np.ones(len(j), dtype=bool),
                    protein=(self.component[i] == 0) & (self.component[j] == 0),
                    ligand=(self.component[i] == 1) & (self.component[j] == 1),
                    cross_component=self.component[i] != self.component[j])
                for name, mask in categories.items():
                    chosen = j[mask]
                    counts[name] += len(chosen)
                    severity[name] += float(np.square(overlap[mask]).sum(dtype=np.float64))
                    if len(chosen):
                        affected[name][i] = True
                        affected[name][chosen] = True
        result = {}
        for name in counts:
            prefix = '' if name == 'all' else name + '.'
            result.update({prefix+'clashes_per_1000_atoms': counts[name]*1000/self.n,
                prefix+'affected_atom_fraction': float(affected[name].mean()),
                prefix+'vdw_severity_A2_per_atom': severity[name]/self.n})
        return result

    def trajectory(self, coordinates):
        rows = [self.frame(frame) for frame in coordinates]
        return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
