"""Shared TorchMD frame encoder over isolated trajectory-frame graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..data.batch import ClipBatch
from ..data.chemistry import NUM_ATOM_TYPE, NUM_BLOCK_TYPE
from ..geometry.frames import build_frame_graph, pack_frame_nodes, unpack_frame_features
from ..geometry.neighbors import CudaRadiusNeighborList
from ..geometry.topology import BoundedTopologyCache, DistanceOnlyBondCache
from ..geometry.types import FrameGraphBatch
from .torchmd import TorchMDEncoder


@dataclass
class FrameEncoderOutput:
    h: Tensor
    v: Tensor
    graph: FrameGraphBatch

    @property
    def scalar(self) -> Tensor:
        return self.h

    @property
    def vector(self) -> Tensor:
        return self.v

    @property
    def backend_used(self) -> str:
        return self.graph.backend_used

    @property
    def graph_mode(self) -> str:
        return self.graph.graph_mode

    @property
    def spatial_backbone(self) -> str:
        return self.graph.spatial_backbone

    def __iter__(self):
        yield self.h
        yield self.v


class FrameEncoder(nn.Module):
    """Encode all frames in a single externally supplied TorchMD graph batch."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_rbf: int = 50,
        num_heads: int = 8,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        *,
        dtype: torch.dtype = torch.float32,
        bond_construction: Mapping[str, Any] | str | None = None,
        topology_cache_capacity: int = 4096,
        topology_device_cache_capacity: int = 4096,
        distance_bond_min: float = 0.5,
        distance_bond_max: float = 2.2,
        distance_bond_max_num_neighbors: int = 64,
        distance_bond_cache_capacity: int = 4096,
    ) -> None:
        super().__init__()
        if isinstance(bond_construction, str):
            bond_config = {"mode": bond_construction}
        else:
            bond_config = dict(bond_construction or {"mode": "topology"})
        bond_mode = str(bond_config.get("mode", "topology"))
        if bond_mode not in {"topology", "distance_only"}:
            raise ValueError("bond_construction.mode must be topology or distance_only")
        self.bond_construction_mode = bond_mode
        self.graph_mode = "external"
        self.spatial_backbone = "torchmd_et"
        self.neighbor_builder = CudaRadiusNeighborList(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            loop=True,
        )
        self.topology_cache = BoundedTopologyCache(
            max_canonical_entries=topology_cache_capacity,
            max_device_entries=topology_device_cache_capacity,
        )
        self.distance_bond_cache = (
            DistanceOnlyBondCache(
                min_distance_angstrom=distance_bond_min,
                max_distance_angstrom=distance_bond_max,
                max_num_neighbors=distance_bond_max_num_neighbors,
                capacity=distance_bond_cache_capacity,
            )
            if bond_mode == "distance_only"
            else None
        )
        self.backbone = TorchMDEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=NUM_ATOM_TYPE,
            max_b=NUM_BLOCK_TYPE,
            dtype=dtype,
        )

    def prepare_batch(self, batch: ClipBatch) -> None:
        if self.bond_construction_mode == "topology":
            self.topology_cache.register_packed_batch(batch)

    def prepare_distance_bonds(
        self, references: Mapping[str, Mapping[str, Any]], *, device: torch.device
    ) -> list[dict[str, Any]]:
        if self.distance_bond_cache is None:
            raise RuntimeError("distance references require distance_only mode")
        manifest = []
        for topology_id in sorted(references):
            item = references[topology_id]
            manifest.append(
                self.distance_bond_cache.register_reference(
                    topology_id,
                    torch.as_tensor(item["coordinates"], dtype=torch.float32),
                    device=device,
                    atom_identity_sha256=item.get("atom_identity_sha256"),
                    sample_id=str(item.get("sample_id", "")),
                    source_split=str(item.get("source_split", "unknown")),
                )
            )
            expected = item.get("coordinate_sha256")
            if expected not in (None, "") and expected != manifest[-1]["coordinate_sha256"]:
                raise RuntimeError(f"distance reference hash mismatch for {topology_id!r}")
        return manifest

    def forward(self, batch: ClipBatch) -> FrameEncoderOutput:
        nodes = pack_frame_nodes(batch)
        graph = build_frame_graph(
            nodes,
            batch,
            neighbor_builder=self.neighbor_builder,
            topology_cache=self.topology_cache,
            distance_bond_cache=self.distance_bond_cache,
            bond_construction_mode=self.bond_construction_mode,
            spatial_backbone=self.spatial_backbone,
        )
        result = self.backbone(
            z=graph.z,
            b=graph.b,
            pos=graph.pos,
            batch=graph.batch,
            edge_index=graph.edge_index,
            edge_weight_t=graph.edge_weight,
            edge_vec_t=graph.edge_vec,
            bond_type=graph.bond_type,
        )
        h, v = unpack_frame_features(result, nodes)
        return FrameEncoderOutput(h=h, v=v, graph=graph)

    encode = forward
