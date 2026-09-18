"""Coordinate-independent topology and packed per-frame graph types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor

STATIC_TOPOLOGY_SCHEMA = "pvb.codec.state_detail.static_topology.v1"
@dataclass
class FrameNodeBatch:
    """Packed frame nodes independent of any spatial graph construction."""

    pos: Tensor  # [T*N_total, 3]
    z: Tensor  # [T*N_total]
    b: Tensor  # [T*N_total]
    batch: Tensor  # [T*N_total], graph id = frame_id * B + sample_id
    graph_id: Tensor  # explicit graph-id alias
    frame_id: Tensor  # [T*N_total]
    sample_id: Tensor  # [T*N_total]
    frame_mask: Tensor  # [B, T]
    atom_counts: tuple[int, ...]
    frames: int
    atoms: int
    batch_size: int
    graph_count: int

    @property
    def num_nodes(self) -> int:
        return int(self.pos.shape[0])

@dataclass
class FrameGraphBatch:
    """Flattened graph inputs and topology used by one spatial encoder call."""

    pos: Tensor  # [T*N_total, 3]
    z: Tensor  # [T*N_total]
    b: Tensor  # [T*N_total]
    batch: Tensor  # [T*N_total], graph id = frame_id * B + sample_id
    graph_id: Tensor  # alias with an explicit name for callers/tests
    frame_id: Tensor  # [T*N_total]
    sample_id: Tensor  # [T*N_total]
    edge_index: Tensor  # [2, E], union of distance and covalent edges
    edge_weight: Tensor  # [E]
    edge_vec: Tensor  # [E, 3]
    bond_type: Tensor  # [E], 1 for a replicated covalent edge
    bond_index: Tensor  # [2, T*E_bond], replicated and frame-offset
    distance_edge_index: Tensor  # [2, E_distance]
    distance_edge_weight: Tensor  # [E_distance]
    distance_edge_vec: Tensor  # [E_distance, 3]
    frame_mask: Tensor  # [B, T]
    backend_used: str
    bond_construction_mode: str
    graph_mode: str = "external"
    spatial_backbone: str = "torchmd_et"

    @property
    def num_nodes(self) -> int:
        return int(self.pos.shape[0])

    @property
    def num_graphs(self) -> int:
        if self.batch.numel() == 0:
            return 0
        return int(self.batch.max().item()) + 1

@dataclass(frozen=True)
class StaticTopologyMetadata:
    """Coordinate-independent topology aligned to the latent ``[N, ...]`` atom axis.

    This deliberately does not mirror :class:`FrameGraphBatch`: radius edges, edge vectors,
    distances, positions, and frame-expanded indices are spatial-runtime data and are not valid
    latent metadata.  ``covalent_bond_type`` is currently a binary indicator because the clip
    schema stores covalent connectivity without a bond-order field.
    """

    atom_type: Tensor
    block_type: Tensor
    abid: Tensor
    block_id: Tensor
    component_id: Tensor
    atom_ptr: Tensor
    covalent_bond_index: Tensor
    covalent_bond_type: Tensor
    topology_id: tuple[str, ...] = ()
    sample_id: tuple[str, ...] = ()
    schema_version: str = STATIC_TOPOLOGY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_TOPOLOGY_SCHEMA:
            raise ValueError(f"unsupported static topology schema {self.schema_version!r}")
        vectors = {
            "atom_type": self.atom_type,
            "block_type": self.block_type,
            "abid": self.abid,
            "block_id": self.block_id,
            "component_id": self.component_id,
        }
        for name, value in vectors.items():
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must have shape [N]")
        atom_count = int(self.atom_type.numel())
        if any(int(value.numel()) != atom_count for value in vectors.values()):
            raise ValueError("static topology atom fields must share the latent N axis")
        if not isinstance(self.atom_ptr, Tensor) or self.atom_ptr.ndim != 1:
            raise ValueError("atom_ptr must have shape [B+1]")
        if self.atom_ptr.numel() < 2:
            raise ValueError("atom_ptr must describe at least one sample")
        if self.atom_ptr.device != self.abid.device:
            raise ValueError("static topology tensors must share a device")
        atom_ptr = self.atom_ptr.to(dtype=torch.long)
        if int(atom_ptr[0]) != 0 or int(atom_ptr[-1]) != atom_count:
            raise ValueError("atom_ptr must start at zero and end at N")
        if torch.any(atom_ptr[1:] < atom_ptr[:-1]):
            raise ValueError("atom_ptr must be nondecreasing")
        batch_size = int(atom_ptr.numel() - 1)
        if self.abid.numel() and (
            torch.any(self.abid < 0) or torch.any(self.abid >= batch_size)
        ):
            raise ValueError("static topology abid contains an invalid sample id")
        if not isinstance(self.covalent_bond_index, Tensor):
            raise ValueError("covalent_bond_index must be a tensor")
        if self.covalent_bond_index.ndim != 2 or self.covalent_bond_index.shape[0] != 2:
            raise ValueError("covalent_bond_index must have shape [2, E_static]")
        if self.covalent_bond_index.device != self.abid.device:
            raise ValueError("covalent bond metadata must share the latent device")
        edge_count = int(self.covalent_bond_index.shape[1])
        if not isinstance(self.covalent_bond_type, Tensor) or self.covalent_bond_type.shape != (edge_count,):
            raise ValueError("covalent_bond_type must have shape [E_static]")
        if self.covalent_bond_type.device != self.abid.device:
            raise ValueError("covalent bond types must share the latent device")
        if edge_count:
            edge_index = self.covalent_bond_index.to(dtype=torch.long)
            if torch.any(edge_index < 0) or torch.any(edge_index >= atom_count):
                raise ValueError("covalent bond index exceeds the latent N atom axis")
            if torch.any(self.abid.index_select(0, edge_index[0]) != self.abid.index_select(0, edge_index[1])):
                raise ValueError("static topology contains a cross-sample covalent bond")
        if self.topology_id and len(self.topology_id) != batch_size:
            raise ValueError("topology_id must contain one entry per latent sample")
        if self.sample_id and len(self.sample_id) != batch_size:
            raise ValueError("sample_id must contain one entry per latent sample")

    @property
    def num_atoms(self) -> int:
        return int(self.atom_type.numel())

    @property
    def batch_size(self) -> int:
        return int(self.atom_ptr.numel() - 1)

    @property
    def z(self) -> Tensor:
        return self.atom_type

    @property
    def b(self) -> Tensor:
        return self.block_type

    @property
    def bond_index(self) -> Tensor:
        return self.covalent_bond_index

    @classmethod
    def from_batch(cls, batch: Any) -> "StaticTopologyMetadata":
        """Extract only static chemical fields from a ``ClipBatch``-like object."""

        def field(name: str) -> Any:
            if isinstance(batch, Mapping):
                if name not in batch:
                    raise ValueError(f"batch is missing static topology field {name!r}")
                return batch[name]
            if not hasattr(batch, name):
                raise ValueError(f"batch is missing static topology field {name!r}")
            return getattr(batch, name)

        atom_type = torch.as_tensor(field("atype"), dtype=torch.long)
        block_type = torch.as_tensor(field("btype"), device=atom_type.device, dtype=torch.long)
        abid = torch.as_tensor(field("abid"), device=atom_type.device, dtype=torch.long).flatten()
        block_id = torch.as_tensor(field("block_id"), device=atom_type.device, dtype=torch.long).flatten()
        component_id = torch.as_tensor(field("component_id"), device=atom_type.device, dtype=torch.long).flatten()
        atom_ptr = torch.as_tensor(field("atom_ptr"), device=atom_type.device, dtype=torch.long).flatten()
        bond_index = torch.as_tensor(field("bond_index"), device=atom_type.device, dtype=torch.long)
        if bond_index.numel() == 0:
            bond_index = torch.empty((2, 0), device=atom_type.device, dtype=torch.long)
        elif bond_index.ndim != 2 or bond_index.shape[0] != 2:
            raise ValueError("bond_index must have shape [2, E_static]")
        bond_type = torch.ones(
            (int(bond_index.shape[1]),), device=atom_type.device, dtype=torch.long
        )
        if isinstance(batch, Mapping):
            topology_values = batch.get("topology_id", ())
            sample_values = batch.get("sample_id", ())
        else:
            topology_values = getattr(batch, "topology_id", ())
            sample_values = getattr(batch, "sample_id", ())
        topology_id = tuple(str(value) for value in topology_values)
        sample_id = tuple(str(value) for value in sample_values)
        return cls(
            atom_type=atom_type,
            block_type=block_type,
            abid=abid,
            block_id=block_id,
            component_id=component_id,
            atom_ptr=atom_ptr,
            covalent_bond_index=bond_index,
            covalent_bond_type=bond_type,
            topology_id=topology_id,
            sample_id=sample_id,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "StaticTopologyMetadata":
        return replace(
            self,
            atom_type=self.atom_type.to(device, non_blocking=non_blocking),
            block_type=self.block_type.to(device, non_blocking=non_blocking),
            abid=self.abid.to(device, non_blocking=non_blocking),
            block_id=self.block_id.to(device, non_blocking=non_blocking),
            component_id=self.component_id.to(device, non_blocking=non_blocking),
            atom_ptr=self.atom_ptr.to(device, non_blocking=non_blocking),
            covalent_bond_index=self.covalent_bond_index.to(device, non_blocking=non_blocking),
            covalent_bond_type=self.covalent_bond_type.to(device, non_blocking=non_blocking),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atom_count": self.num_atoms,
            "batch_size": self.batch_size,
            "atom_type_shape": list(self.atom_type.shape),
            "block_type_shape": list(self.block_type.shape),
            "abid_shape": list(self.abid.shape),
            "block_id_shape": list(self.block_id.shape),
            "component_id_shape": list(self.component_id.shape),
            "atom_ptr_shape": list(self.atom_ptr.shape),
            "covalent_bond_index_shape": list(self.covalent_bond_index.shape),
            "covalent_bond_type_shape": list(self.covalent_bond_type.shape),
            "bond_index_space": "latent_atom_axis_N",
            "bond_encoding": "binary_covalent_connectivity",
            "coordinate_independent": True,
            "frame_invariant": True,
            "contains_radius_edges": False,
            "contains_distance_or_edge_vectors": False,
            "contains_target_coordinates": False,
        }
