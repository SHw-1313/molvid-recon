"""Observed-only learned chemical-token memory (B private backend).

Coordinates are Angstrom; clocks and analysis support are physical ps. No
teacher, target, flow-time, optimizer, or detached feature cache lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..equivariant import AxisPreservingLinear
from ..geometry.types import StaticTopologyMetadata
from ..time import pairwise_time_features


@dataclass(frozen=True)
class ChemicalMapping:
    """Fixed atom -> group indices, with isolated packed systems.

    Build once from topology, not once per history frame. Protein atoms share
    a residue group; an explicitly supplied ligand mask retains atom tokens.
    """
    atom_to_group: Tensor  # [N]
    group_system: Tensor  # [R]
    counts: Tensor  # [R]

    @classmethod
    def from_topology(cls, topology: StaticTopologyMetadata, ligand_mask: Tensor | None = None):
        n = topology.num_atoms
        ligand = torch.zeros(n, dtype=torch.bool, device=topology.abid.device) if ligand_mask is None else ligand_mask.bool()
        if ligand.shape != (n,):
            raise ValueError("ligand_mask must be [N]")
        atom = torch.arange(n, device=topology.abid.device)
        keys = torch.stack((topology.abid, topology.component_id, ligand.long(),
                            torch.where(ligand, atom, topology.block_id)), dim=-1)
        unique, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
        return cls(inverse, unique[:, 0], counts)

    def pool(self, value: Tensor) -> Tensor:
        """Mean over atoms; retain frame and optional xyz/channel axes."""
        result = value.new_zeros((value.shape[0], self.counts.numel(), *value.shape[2:]))
        result.index_add_(1, self.atom_to_group, value)
        divisor = self.counts.to(value.dtype).reshape(1, -1, *([1] * (value.ndim - 2)))
        return result / divisor


@dataclass(frozen=True)
class RawHistoryObservation:
    coordinates: Tensor  # [Th,N,3], Angstrom; padding may contain arbitrary values
    physical_time_ps: Tensor  # [B,Th], relative to last valid observed frame = 0
    frame_mask: Tensor  # [B,Th], valid prefix
    topology: StaticTopologyMetadata
    mapping: ChemicalMapping
    anchor_coordinates: Tensor  # [N,3], last observed geometry, Angstrom

    def __post_init__(self):
        x, time, mask = self.coordinates, self.physical_time_ps, self.frame_mask
        if x.ndim != 3 or x.shape[1:] != (self.topology.num_atoms, 3):
            raise ValueError("history coordinates must be [Th,N,3]")
        if time.shape != (self.topology.batch_size, x.shape[0]) or mask.shape != time.shape or mask.dtype != torch.bool:
            raise ValueError("history clock/mask must be [B,Th], mask boolean")
        if self.anchor_coordinates.shape != x.shape[1:]:
            raise ValueError("last observed anchor must be [N,3]")
        if any(t.device != x.device for t in (time, mask, self.topology.abid, self.mapping.atom_to_group, self.anchor_coordinates)):
            raise ValueError("all observed fields must share a device")
        if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("each history must be a nonempty valid prefix")
        if not bool(torch.isfinite(time[mask]).all()) or bool((time[mask] > 0).any()):
            raise ValueError("history must contain only finite observed times <= 0 ps")
        last = mask.sum(1) - 1
        if not bool((time.gather(1, last[:, None]) == 0).all()):
            raise ValueError("last observed time must be 0 ps")
        if bool(((time[:, 1:] <= time[:, :-1]) & mask[:, 1:]).any()):
            raise ValueError("valid history times must increase strictly")
        valid_atoms = mask[self.topology.abid].T
        if not bool(torch.isfinite(x[valid_atoms]).all()):
            raise ValueError("nonfinite valid observed coordinates")
        atom = torch.arange(x.shape[1], device=x.device)
        if not torch.equal(x[last[self.topology.abid], atom], self.anchor_coordinates):
            raise ValueError("anchor must be the actual last observed geometry")
        mapping = self.mapping
        if mapping.atom_to_group.shape != (x.shape[1],) or not torch.equal(mapping.group_system[mapping.atom_to_group], self.topology.abid):
            raise ValueError("chemical mapping mixes packed systems")


class LocalAtomUpdate(nn.Module):
    """Invariant messages and channel-only equivariant vector updates."""
    def __init__(self, scalar_width: int, vector_width: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * scalar_width + 3, scalar_width), nn.SiLU(),
                                     nn.Linear(scalar_width, scalar_width + 2 * vector_width))
        self.scalar_update = nn.Sequential(nn.Linear(scalar_width, scalar_width), nn.SiLU())
        self.vector_update = AxisPreservingLinear(vector_width, vector_width)
        self.scalar_width = scalar_width

    def forward(self, h, v, x, anchor, edges):
        receiver, sender = edges
        displacement = x[:, sender] - x[:, receiver]
        distance = displacement.float().square().sum(-1).clamp_min(1e-12).sqrt()
        anchor_distance = (anchor[sender] - anchor[receiver]).float().square().sum(-1).clamp_min(1e-12).sqrt()
        geometry = torch.stack((distance, distance.square() / 10, anchor_distance.expand_as(distance)), -1)
        message = self.message(torch.cat((h[:, receiver], h[:, sender], geometry.to(h.dtype)), -1))
        mh = message[..., :self.scalar_width]
        direction_gate, vector_gate = message[..., self.scalar_width:].chunk(2, -1)
        mv = displacement.unsqueeze(-1) * direction_gate.unsqueeze(-2) + v[:, sender] * torch.sigmoid(vector_gate).unsqueeze(-2)
        sum_h, sum_v = torch.zeros_like(h), torch.zeros_like(v)
        sum_h.index_add_(1, receiver, mh.to(h.dtype))
        sum_v.index_add_(1, receiver, mv.to(v.dtype))
        degree = torch.bincount(receiver, minlength=h.shape[1]).clamp_min(1).to(h.dtype)
        return h + self.scalar_update(sum_h / degree[None, :, None]), v + self.vector_update(sum_v / degree[None, :, None, None])


class LightHistoryEncoder(nn.Module):
    """Two shared covalent message layers before residue pooling.

    Initial implementation uses covalent edges only. It makes no claim of
    dynamic nonbond neighbor coverage; G remains the downstream spatial path.
    Atom chunks are unnecessary: frames can be chunked without changing the
    network or introducing any CPU model execution.
    """
    def __init__(self, scalar_width=64, vector_width=32, layers=2, frame_chunk_size=8):
        super().__init__()
        self.element = nn.Embedding(128, scalar_width)
        self.block_type = nn.Embedding(128, scalar_width)
        self.displacement_gate = nn.Linear(scalar_width, vector_width)
        self.layers = nn.ModuleList(LocalAtomUpdate(scalar_width, vector_width) for _ in range(layers))
        self.frame_chunk_size = int(frame_chunk_size)
        if self.frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")

    def forward(self, observation: RawHistoryObservation) -> tuple[Tensor, Tensor]:
        topology, mapping = observation.topology, observation.mapping
        # Canonical directed covalent union: input schemas may contain one or both directions.
        edges = topology.covalent_bond_index
        edges = torch.unique(torch.cat((edges, edges.flip(0)), dim=1), dim=1)
        base = self.element(topology.atom_type) + self.block_type(topology.block_type)
        outputs_h, outputs_v = [], []
        for start in range(0, observation.coordinates.shape[0], self.frame_chunk_size):
            stop = start + self.frame_chunk_size
            mask = observation.frame_mask[:, start:stop][topology.abid].T
            x = torch.where(mask[..., None], observation.coordinates[start:stop], observation.anchor_coordinates[None])
            h = base[None].expand(x.shape[0], -1, -1)
            displacement = x - observation.anchor_coordinates[None]
            v = displacement[..., None] * self.displacement_gate(base)[None, :, None, :]
            for layer in self.layers:
                h, v = layer(h, v, x, observation.anchor_coordinates, edges)
            outputs_h.append(mapping.pool(h * mask[..., None]))
            outputs_v.append(mapping.pool(v * mask[..., None, None]))
        return torch.cat(outputs_h), torch.cat(outputs_v)


@dataclass(frozen=True)
class AnalysisTokens:
    h: Tensor  # [S,R,C]
    v: Tensor  # [S,R,3,Cv]
    valid: Tensor  # [B,S]
    labels: Tensor  # [B,S,5]: center, scale, coverage, detail flag, motion-known
    support_ps: Tensor  # [B,S,2], actual observed support, not a synthetic timestamp
    detail_h: Tensor  # [D,R,C], signed, exact zero for constant features
    detail_v: Tensor  # [D,R,3,Cv]


class PhysicalScaleAnalysis(nn.Module):
    """Local quadrature/Haar analysis, not an orthogonal wavelet transform.

    Width means total support width in ps. Kernels need two observations on
    each half, max adjacent gap <= width/2, and >=75% temporal coverage.
    M-index uses 100 * frame index for analysis only; query clocks stay ps.
    """
    def __init__(self, scales_ps=(400, 800, 1600, 3200, 6400), max_centers=4, recent=4, mode="physical"):
        super().__init__()
        if mode not in ("physical", "index") or max_centers < 1 or recent < 1 or any(s <= 0 for s in scales_ps):
            raise ValueError("invalid analysis configuration")
        self.scales_ps, self.max_centers, self.recent, self.mode = tuple(scales_ps), max_centers, recent, mode

    def forward(self, h: Tensor, v: Tensor, observation: RawHistoryObservation) -> AnalysisTokens:
        physical, mask = observation.physical_time_ps.float(), observation.frame_mask
        batch, frames = mask.shape
        groups = observation.mapping.group_system
        last = mask.sum(1) - 1
        clock = physical if self.mode == "physical" else (torch.arange(frames, device=h.device)[None] - last[:, None]).float() * 100
        weights, valid, labels, support, detail_flags = [], [], [], [], []
        for kind in ("level", *range(self.recent), *[(s, c) for s in self.scales_ps for c in range(self.max_centers)]):
            weight = clock.new_zeros(batch, frames)
            okay = mask.new_zeros(batch)
            label = clock.new_zeros(batch, 5)
            bounds = clock.new_zeros(batch, 2)
            detail = isinstance(kind, tuple)
            for b in range(batch):
                indices = torch.nonzero(mask[b]).flatten()
                t = clock[b, indices]
                q = torch.ones_like(t)
                if t.numel() > 1:
                    delta = t[1:] - t[:-1]
                    q = torch.cat((delta[:1] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1:] / 2))
                if kind == "level":
                    chosen, w = indices, q / q.sum()
                    center, scale, coverage = (t[0] + t[-1]) / 2, t[-1] - t[0], t.new_tensor(1.)
                    usable = True
                elif isinstance(kind, int):
                    selected = indices.numel() - 1 - kind
                    usable = selected >= 0
                    chosen = indices[max(0, selected):max(0, selected) + 1]
                    w = q.new_ones(1)
                    center, scale, coverage = clock[b, chosen[0]], t.new_tensor(0.), t.new_tensor(float(usable))
                else:
                    width, center_index = kind
                    # Spread centers across all observed past, not only the last few ns.
                    available = (t[-1] - t[0] - width).clamp_min(0)
                    center = t[-1] - width / 2 - available * center_index / max(1, self.max_centers - 1)
                    local = (t >= center - width / 2) & (t <= center + width / 2)
                    chosen = indices[local]
                    lt, lq = t[local], q[local]
                    left = lt < center
                    right = ~left
                    coverage = (lt[-1] - lt[0]) / width if lt.numel() else t.new_tensor(0.)
                    usable = lt.numel() >= 4 and int(left.sum()) >= 2 and int(right.sum()) >= 2
                    usable = usable and bool(coverage >= .75) and bool(((lt[1:] - lt[:-1]) <= width / 2).all())
                    # +/- halves each integrate to 1/2; weighted mean is zero.
                    w = .5 * (lq * right / (lq * right).sum().clamp_min(1e-12) - lq * left / (lq * left).sum().clamp_min(1e-12))
                    scale = t.new_tensor(float(width))
                if usable:
                    weight[b, chosen] = w
                    okay[b] = True
                if chosen.numel():
                    bounds[b] = physical[b, chosen[[0, -1]]]
                label[b] = torch.stack((torch.as_tensor(center, device=h.device), torch.as_tensor(scale, device=h.device), coverage, t.new_tensor(float(detail)), t.new_tensor(float(indices.numel() > 1))))
            weights.append(weight)
            valid.append(okay)
            labels.append(label)
            support.append(bounds)
            detail_flags.append(detail)
        weights = torch.stack(weights, 1)  # B,S,Th
        atom_weights = weights[groups].permute(1, 0, 2)  # S,R,Th
        # Subtract one observed state before detail accumulation. This makes
        # constant details exactly zero even under floating reduction error.
        reference_h = h[0:1]
        reference_v = v[0:1]
        scalar, vector = [], []
        for s, is_detail in enumerate(detail_flags):
            scalar.append(torch.einsum("rt,trc->rc", atom_weights[s], (h - reference_h if is_detail else h).float()))
            vector.append(torch.einsum("rt,trdc->rdc", atom_weights[s], (v - reference_v if is_detail else v).float()))
        sh, sv = torch.stack(scalar), torch.stack(vector)
        flags = torch.tensor(detail_flags, device=h.device)
        return AnalysisTokens(sh.to(h.dtype), sv.to(v.dtype), torch.stack(valid, 1), torch.stack(labels, 1), torch.stack(support, 1), sh[flags], sv[flags])


@dataclass(frozen=True)
class GroupMemory:
    h: Tensor  # [K,R,256], projected from scalar128 slots
    v: Tensor  # [K,R,3,128], projected from vector64 slots
    mapping: ChemicalMapping
    source_support_ps: Tensor  # [B,S,2], slot attends a distribution of supports
    source_valid: Tensor  # [B,S]
    slot_weights: Tensor  # [R,K,S], audit weights, retain autograd
    analysis: AnalysisTokens

    def gather_atoms(self) -> tuple[Tensor, Tensor]:
        """Local group gather only, never Q*N*R*K attention."""
        return self.h[:, self.mapping.atom_to_group], self.v[:, self.mapping.atom_to_group]

    def query_time_features(self, query_time_ps: Tensor) -> tuple[Tensor, Tensor]:
        """Expected endpoint time features over each slot's actual support.

        Slots have a distribution over observation intervals, not a fabricated
        representative frame time. Output [N,Q,K,4] and [N,Q,K].
        """
        lower, upper = self.source_support_ps.unbind(-1)
        features = (pairwise_time_features(query_time_ps, lower) + pairwise_time_features(query_time_ps, upper)) / 2
        distance = (((query_time_ps[:, :, None] - lower[:, None]).abs() / 100).sqrt()
                    + ((query_time_ps[:, :, None] - upper[:, None]).abs() / 100).sqrt()) / 2
        group = self.mapping.group_system
        features = torch.einsum('rks,rqsd->rqkd', self.slot_weights.float(), features[group].float())
        distance = torch.einsum('rks,rqs->rqk', self.slot_weights.float(), distance[group].float())
        atom = self.mapping.atom_to_group
        return features[atom], distance[atom]


class LearnedHistoryMemory(nn.Module):
    """Prepare once per observed prefix; callers must preserve this graph."""
    def __init__(self, scalar_width=64, vector_width=32, slots=8, memory_scalar=128, memory_vector=64, analysis_mode="physical", output_scalar=256, output_vector=128):
        super().__init__()
        self.light_encoder = LightHistoryEncoder(scalar_width, vector_width)
        self.analysis = PhysicalScaleAnalysis(mode=analysis_mode)
        self.queries = nn.Parameter(torch.randn(slots, memory_scalar) / math.sqrt(memory_scalar))
        self.anchor_query = nn.Linear(scalar_width, memory_scalar)
        self.key = nn.Linear(scalar_width, memory_scalar)
        self.label_key = nn.Sequential(nn.Linear(5, memory_scalar), nn.SiLU(), nn.Linear(memory_scalar, memory_scalar))
        self.scalar_value = nn.Linear(scalar_width, memory_scalar)
        self.vector_value = AxisPreservingLinear(vector_width, memory_vector)
        self.scalar_interaction = nn.Sequential(nn.Linear(memory_scalar, memory_scalar), nn.SiLU(), nn.Linear(memory_scalar, memory_scalar))
        self.vector_gate = nn.Linear(memory_scalar, memory_vector)
        self.scalar_output = nn.Linear(memory_scalar, output_scalar)
        self.vector_output = AxisPreservingLinear(memory_vector, output_vector)
        self.channels = 128  # frozen target width; light widths are independent

    def set_trainable(self, enabled: bool):
        self.requires_grad_(bool(enabled))

    def contract(self):
        return {"backend": "learned_memory", "light_scalar": self.light_encoder.element.embedding_dim,
                "light_vector": self.light_encoder.displacement_gate.out_features,
                "light_layers": len(self.light_encoder.layers), "graph": "covalent_only",
                "slots": self.queries.shape[0], "memory_scalar": self.queries.shape[1],
                "memory_vector": self.vector_value.output_width,
                "analysis_mode": self.analysis.mode, "scales_ps": list(self.analysis.scales_ps),
                "max_centers": self.analysis.max_centers, "recent": self.analysis.recent,
                "teacher_observed_frames": 2, "teacher_anchor_times_ps": [-800, 0]}

    def forward(self, observation: RawHistoryObservation) -> GroupMemory:
        h, v = self.light_encoder(observation)
        analysis = self.analysis(h, v, observation)
        labels = analysis.labels.clone()
        labels[..., :2] /= 1000  # fixed ps -> ns numeric scale, no trajectory amplitude normalization
        groups = observation.mapping.group_system
        keys = self.key(analysis.h).permute(1, 0, 2) + self.label_key(labels)[groups]
        last = observation.frame_mask.sum(1) - 1
        anchor_h = h[last[groups], torch.arange(groups.numel(), device=h.device)]
        queries = self.queries[None] + self.anchor_query(anchor_h)[:, None]
        scores = torch.einsum("rkc,rsc->rks", queries, keys) / math.sqrt(keys.shape[-1])
        scores = scores.float().masked_fill(~analysis.valid[groups, None], -torch.inf)
        weights = scores.softmax(-1)
        scalar = torch.einsum("rks,src->krc", weights, self.scalar_value(analysis.h).float())
        vector = torch.einsum("rks,srdc->krdc", weights, self.vector_value(analysis.v).float())
        scalar = scalar + self.scalar_interaction(scalar)
        vector = vector * torch.sigmoid(self.vector_gate(scalar))[:, :, None]
        return GroupMemory(self.scalar_output(scalar), self.vector_output(vector), observation.mapping, analysis.support_ps, analysis.valid, weights, analysis)
