"""Sparse observed-geometry messages for selected Frame DiT blocks."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..equivariant import AxisPreservingLinear
from ..geometry.types import StaticTopologyMetadata
from ..latent.types import FrameLatentBatch, ObservedContext


@dataclass(frozen=True)
class LocalTopology:
    source: Tensor
    destination: Tensor
    hop: Tensor


class LocalTopologyCache:
    """Bounded host/device cache for covalent 1-hop plus 1-3 edges."""

    def __init__(self, capacity: int = 512) -> None:
        if int(capacity) < 1:
            raise ValueError("local topology cache capacity must be positive")
        self.capacity = int(capacity)
        self._host: OrderedDict[tuple[Any, ...], tuple[Tensor, Tensor, Tensor]] = OrderedDict()
        self._device: OrderedDict[tuple[Any, ...], LocalTopology] = OrderedDict()

    @staticmethod
    def _key(topology: StaticTopologyMetadata) -> tuple[Any, ...]:
        atom_ptr = tuple(int(value) for value in topology.atom_ptr.detach().cpu().tolist())
        return (tuple(topology.topology_id), atom_ptr, int(topology.covalent_bond_index.shape[1]))

    def _build_host(self, topology: StaticTopologyMetadata) -> tuple[Tensor, Tensor, Tensor]:
        key = self._key(topology)
        cached = self._host.get(key)
        if cached is not None:
            self._host.move_to_end(key)
            return cached
        abid = topology.abid.detach().cpu().tolist()
        bonds = topology.covalent_bond_index.detach().cpu().transpose(0, 1).tolist()
        adjacency: dict[int, set[int]] = {index: set() for index in range(topology.num_atoms)}
        one_hop: set[tuple[int, int]] = set()
        for first, second in bonds:
            first, second = int(first), int(second)
            if first == second:
                continue
            if abid[first] != abid[second]:
                raise ValueError("covalent topology contains a cross-system edge")
            edge = (min(first, second), max(first, second))
            one_hop.add(edge)
            adjacency[first].add(second)
            adjacency[second].add(first)
        two_hop: set[tuple[int, int]] = set()
        for center, neighbors in adjacency.items():
            ordered = sorted(neighbors)
            for position, first in enumerate(ordered):
                for second in ordered[position + 1:]:
                    if first == second or abid[first] != abid[second]:
                        continue
                    edge = (min(first, second), max(first, second))
                    if edge not in one_hop:
                        two_hop.add(edge)
        directed: list[tuple[int, int, int]] = []
        for hop, edges in ((1, one_hop), (2, two_hop)):
            for first, second in sorted(edges):
                directed.extend(((first, second, hop), (second, first, hop)))
        if directed:
            source, destination, hop = zip(*directed)
            value = (
                torch.tensor(source, dtype=torch.long),
                torch.tensor(destination, dtype=torch.long),
                torch.tensor(hop, dtype=torch.long),
            )
        else:
            value = tuple(torch.empty(0, dtype=torch.long) for _ in range(3))
        self._host[key] = value
        self._host.move_to_end(key)
        while len(self._host) > self.capacity:
            evicted, _ = self._host.popitem(last=False)
            for device_key in [item for item in self._device if item[0] == evicted]:
                del self._device[device_key]
        return value

    def materialize(self, topology: StaticTopologyMetadata, device: torch.device) -> LocalTopology:
        key = self._key(topology)
        device_key = (key, str(device))
        cached = self._device.get(device_key)
        if cached is not None:
            self._device.move_to_end(device_key)
            return cached
        source, destination, hop = self._build_host(topology)
        result = LocalTopology(
            source=source.to(device=device),
            destination=destination.to(device=device),
            hop=hop.to(device=device),
        )
        self._device[device_key] = result
        self._device.move_to_end(device_key)
        while len(self._device) > self.capacity * 2:
            self._device.popitem(last=False)
        return result


class LocalGeometryMessage(nn.Module):
    """Equivariant local message with raw-vector invariant scalar features."""

    def __init__(
        self,
        scalar_width: int,
        vector_width: int,
        *,
        max_atom_type: int = 128,
        rbf_count: int = 8,
        cache_capacity: int = 512,
    ) -> None:
        super().__init__()
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.hop_embedding = nn.Embedding(3, 8)
        self.atom_embedding = nn.Embedding(int(max_atom_type), 8)
        centers = torch.linspace(0.0, 10.0, int(rbf_count))
        self.register_buffer("rbf_centers_A", centers)
        self.rbf_gamma = 1.0 / max(float(centers[1] - centers[0]) ** 2, 1.0e-6)
        invariant_width = 4 * self.vector_width
        static_width = 8 + 2 * 8 + int(rbf_count)
        input_width = 2 * self.scalar_width + invariant_width + static_width
        self.edge_hidden = nn.Sequential(
            nn.Linear(input_width, self.scalar_width),
            nn.SiLU(),
        )
        self.scalar_out = nn.Linear(self.scalar_width, self.scalar_width)
        self.vector_gate = nn.Linear(self.scalar_width, 3 * self.vector_width)
        self.vector_neighbor = AxisPreservingLinear(self.vector_width, self.vector_width)
        self.vector_difference = AxisPreservingLinear(self.vector_width, self.vector_width)
        self.scalar_gate = nn.Parameter(torch.zeros(self.scalar_width))
        self.vector_residual_gate = nn.Parameter(torch.zeros(self.vector_width))
        self.cache = LocalTopologyCache(cache_capacity)

    @staticmethod
    def _last_observed_centered(context: ObservedContext) -> Tensor:
        last = context.latent.frame_mask.sum(dim=1).long() - 1
        atom_last = last.index_select(0, context.latent.abid)
        atom = torch.arange(context.latent.num_atoms, device=context.coordinates.device)
        coordinate = context.coordinates[atom_last, atom]
        origin = context.sample_origin.index_select(0, context.latent.abid)
        return coordinate - origin

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        noisy: FrameLatentBatch,
        context: ObservedContext,
    ) -> tuple[Tensor, Tensor]:
        topology = self.cache.materialize(noisy.topology, h.device)
        source, destination = topology.source, topology.destination
        if source.numel() == 0:
            return torch.zeros_like(h), torch.zeros_like(v)
        if torch.any(noisy.abid.index_select(0, source) != noisy.abid.index_select(0, destination)):
            raise RuntimeError("local geometry cache produced a cross-system edge")
        reference = self._last_observed_centered(context).float()
        displacement = reference.index_select(0, source) - reference.index_select(0, destination)
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-6)
        direction = displacement / distance.unsqueeze(-1)
        rbf = torch.exp(
            -self.rbf_gamma * (distance.unsqueeze(-1) - self.rbf_centers_A.float()) ** 2
        )
        atom_type = noisy.atom_type.remainder(self.atom_embedding.num_embeddings)
        static = torch.cat((
            self.hop_embedding(topology.hop),
            self.atom_embedding(atom_type.index_select(0, source)),
            self.atom_embedding(atom_type.index_select(0, destination)),
            rbf.to(dtype=h.dtype),
        ), dim=-1)
        source_v = v.index_select(1, source)
        destination_v = v.index_select(1, destination)
        source_norm = torch.log1p(source_v.float().square().sum(dim=2))
        destination_norm = torch.log1p(destination_v.float().square().sum(dim=2))
        inner = (source_v.float() * destination_v.float()).sum(dim=2)
        inner = inner.sign() * torch.log1p(inner.abs())
        difference_norm = torch.log1p(
            (source_v.float() - destination_v.float()).square().sum(dim=2)
        )
        invariant = torch.cat((source_norm, destination_norm, inner, difference_norm), dim=-1)
        edge_input = torch.cat((
            h.index_select(1, source),
            h.index_select(1, destination),
            invariant.to(dtype=h.dtype),
            static.unsqueeze(0).expand(h.shape[0], -1, -1),
        ), dim=-1)
        hidden = self.edge_hidden(edge_input)
        scalar_message = self.scalar_out(hidden)
        neighbor_gate, difference_gate, direction_gate = self.vector_gate(hidden).chunk(3, dim=-1)
        vector_message = (
            torch.tanh(neighbor_gate).unsqueeze(2) * self.vector_neighbor(source_v)
            + torch.tanh(difference_gate).unsqueeze(2)
            * self.vector_difference(source_v - destination_v)
            + torch.tanh(direction_gate).unsqueeze(2)
            * direction.to(dtype=v.dtype).view(1, -1, 3, 1)
        )
        atom_mask = noisy.atom_frame_mask()
        edge_mask = atom_mask.index_select(1, source) & atom_mask.index_select(1, destination)
        scalar_message = scalar_message * edge_mask.unsqueeze(-1)
        vector_message = vector_message * edge_mask.unsqueeze(-1).unsqueeze(-1)
        scalar = torch.zeros((*h.shape[:2], h.shape[-1]), device=h.device, dtype=torch.float32)
        vector = torch.zeros((*v.shape[:3], v.shape[-1]), device=v.device, dtype=torch.float32)
        scalar.index_add_(1, destination, scalar_message.float())
        vector.index_add_(1, destination, vector_message.float())
        degree = h.new_zeros((h.shape[1],), dtype=torch.float32)
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=torch.float32))
        scalar = scalar / degree.clamp_min(1.0).view(1, -1, 1)
        vector = vector / degree.clamp_min(1.0).view(1, -1, 1, 1)
        return (
            torch.tanh(self.scalar_gate) * scalar.to(dtype=h.dtype),
            torch.tanh(self.vector_residual_gate).view(1, 1, 1, -1)
            * vector.to(dtype=v.dtype),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "edge_set": "deduplicated_bidirectional_covalent_1hop_plus_1-3",
            "reference_geometry": "last_observed_coordinates",
            "rbf_count": int(self.rbf_centers_A.numel()),
            "raw_vector_invariants": ["node_log_norm2", "edge_inner", "edge_difference_log_norm2"],
            "vector_operations": ["channel_mix_neighbor", "channel_mix_difference", "observed_direction"],
            "aggregation": "destination_degree_mean_fp32",
            "outer_gate_initialization": "zero",
        }


__all__ = ["LocalGeometryMessage", "LocalTopology", "LocalTopologyCache"]
