"""Frame packing, graph construction and feature restoration."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from ..data.batch import ClipBatch
from .neighbors import NeighborList
from .topology import BoundedTopologyCache, DistanceOnlyBondCache
from .types import FrameGraphBatch, FrameNodeBatch


def _get_field(batch: ClipBatch | Mapping[str, Any], name: str) -> Any:
    if isinstance(batch, Mapping):
        if name not in batch:
            raise ValueError(f"clip batch is missing required field: {name}")
        return batch[name]
    if not hasattr(batch, name):
        raise ValueError(f"clip batch is missing required field: {name}")
    return getattr(batch, name)


def _as_tensor(value: Any, *, name: str, dtype: torch.dtype | None = None) -> Tensor:
    result = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if result.numel() == 0 and name not in {"bond_index", "delta_time_ps"}:
        raise ValueError(f"clip field {name} must not be empty")
    return result


def _assert_device_condition(condition: Tensor, message: str) -> None:
    """Raise without synchronizing the host on the CUDA graph hot path."""

    condition = condition if condition.ndim == 0 else condition.all()
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


def _host_atom_counts(batch: ClipBatch | Mapping[str, Any], atoms: int) -> tuple[int, ...]:
    if isinstance(batch, ClipBatch):
        counts = batch.host_atom_counts
    elif "atom_counts" in batch:
        counts = tuple(int(value) for value in batch["atom_counts"])
    else:
        atom_ptr = _as_tensor(_get_field(batch, "atom_ptr"), name="atom_ptr", dtype=torch.long)
        if atom_ptr.device.type != "cpu":
            raise RuntimeError(
                "CUDA graph construction requires collated host atom_counts; "
                "prepare the CPU ClipBatch before transfer"
            )
        counts = tuple(
            int(atom_ptr[index + 1] - atom_ptr[index])
            for index in range(int(atom_ptr.numel() - 1))
        )
    if not counts or any(int(value) < 1 for value in counts):
        raise ValueError("atom_counts must contain positive per-sample counts")
    if sum(int(value) for value in counts) != int(atoms):
        raise ValueError("atom_counts must span the packed atom axis")
    return tuple(int(value) for value in counts)


def _host_atom_identity_hashes(
    batch: ClipBatch | Mapping[str, Any], batch_size: int
) -> tuple[str | None, ...]:
    """Read immutable atom identity metadata without a CUDA synchronization."""

    if isinstance(batch, ClipBatch):
        hashes = batch.host_atom_identity_hashes
    elif "atom_identity_sha256" in batch:
        hashes = tuple(str(value) for value in batch["atom_identity_sha256"])
    else:
        atom_ptr = _as_tensor(_get_field(batch, "atom_ptr"), name="atom_ptr", dtype=torch.long)
        if atom_ptr.device.type != "cpu":
            raise RuntimeError(
                "CUDA distance-only graph construction requires collated host "
                "atom_identity_sha256; prepare the CPU ClipBatch before transfer"
            )
        hashes = ()
    if hashes and len(hashes) != int(batch_size):
        raise ValueError("atom_identity_sha256 must contain one entry per packed sample")
    if not hashes:
        return tuple(None for _ in range(int(batch_size)))
    return tuple(str(value) if value not in (None, "") else None for value in hashes)


def _expand_atom_field(value: Tensor, *, frames: int, atoms: int, name: str) -> Tensor:
    """Expand a per-atom field without materializing an unnecessary copy."""

    if value.ndim == 1:
        if value.shape[0] != atoms:
            raise ValueError(f"{name} must have shape [N_total]")
        return value.unsqueeze(0).expand(frames, -1).reshape(-1)
    if value.ndim >= 2 and value.shape[0] == frames and value.shape[1] == atoms:
        return value.reshape(frames * atoms, *value.shape[2:])
    raise ValueError(
        f"{name} must have shape [N_total, ...] or [T, N_total, ...], "
        f"got {tuple(value.shape)}"
    )
def pack_frame_nodes(batch: ClipBatch | Mapping[str, Any]) -> FrameNodeBatch:
    """Pack atom metadata into frame-isolated nodes without building edges."""

    x = _as_tensor(_get_field(batch, "x"), name="x")
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must have shape [T, N_total, 3]")
    frames, atoms = int(x.shape[0]), int(x.shape[1])
    if frames < 1 or atoms < 1:
        raise ValueError("x must contain at least one frame and one atom")
    if x.dtype != torch.float32:
        raise RuntimeError(
            "frame graph geometry requires original FP32 coordinates; "
            "do not cast coordinates to BF16"
        )

    atom_ptr = _as_tensor(
        _get_field(batch, "atom_ptr"), name="atom_ptr", dtype=torch.long
    ).flatten()
    if atom_ptr.ndim != 1 or atom_ptr.numel() < 2:
        raise ValueError("atom_ptr must have shape [B+1]")
    atom_counts = _host_atom_counts(batch, atoms)
    batch_size = len(atom_counts)
    if atom_ptr.numel() != batch_size + 1:
        raise ValueError("atom_ptr and host atom_counts disagree")
    if atom_ptr.device.type == "cpu":
        if int(atom_ptr[0]) != 0 or int(atom_ptr[-1]) != atoms:
            raise ValueError("atom_ptr must span the packed atom axis")
        if torch.any(atom_ptr[1:] <= atom_ptr[:-1]):
            raise ValueError("atom_ptr must contain strictly increasing sample offsets")

    abid = _as_tensor(_get_field(batch, "abid"), name="abid", dtype=torch.long).flatten()
    if abid.numel() != atoms:
        raise ValueError("abid must have shape [N_total]")
    count_tensor = torch.tensor(atom_counts, device=abid.device, dtype=torch.long)
    expected_abid = torch.repeat_interleave(
        torch.arange(batch_size, device=abid.device, dtype=torch.long),
        count_tensor,
    )
    _assert_device_condition(
        abid == expected_abid,
        "abid must agree with host atom_counts and packed sample order",
    )

    frame_mask = _as_tensor(
        _get_field(batch, "frame_mask"), name="frame_mask", dtype=torch.bool
    )
    if frame_mask.shape != (batch_size, frames):
        raise ValueError("frame_mask must have shape [B, T]")
    _assert_device_condition(
        torch.any(frame_mask),
        "a frame-encoder chunk must contain at least one valid frame",
    )
    if frames > 1:
        _assert_device_condition(
            (~frame_mask[:, 1:]) | frame_mask[:, :-1],
            "frame_mask must be a valid prefix, including within a sliced chunk",
        )

    dense_pos = x.reshape(frames * atoms, 3)
    dense_z = _expand_atom_field(
        _as_tensor(_get_field(batch, "atype"), name="atype", dtype=torch.long),
        frames=frames,
        atoms=atoms,
        name="atype",
    )
    dense_b = _expand_atom_field(
        _as_tensor(_get_field(batch, "btype"), name="btype", dtype=torch.long),
        frames=frames,
        atoms=atoms,
        name="btype",
    )
    dense_sample_id = abid.repeat(frames)
    dense_frame_id = torch.arange(
        frames, device=dense_pos.device, dtype=torch.long
    ).repeat_interleave(atoms)
    dense_graph_id = dense_frame_id * batch_size + dense_sample_id
    dense_valid = frame_mask.index_select(0, abid).transpose(0, 1).reshape(-1)
    dense_index = torch.nonzero(dense_valid, as_tuple=False).flatten()
    dense_to_compact = torch.full(
        (frames * atoms,), -1, device=dense_pos.device, dtype=torch.long
    )
    dense_to_compact.index_copy_(
        0,
        dense_index,
        torch.arange(dense_index.numel(), device=dense_pos.device, dtype=torch.long),
    )
    pos = dense_pos.index_select(0, dense_index)
    z = dense_z.index_select(0, dense_index)
    b = dense_b.index_select(0, dense_index)
    sample_id = dense_sample_id.index_select(0, dense_index)
    frame_id = dense_frame_id.index_select(0, dense_index)
    graph_id = dense_graph_id.index_select(0, dense_index)
    return FrameNodeBatch(
        pos=pos,
        z=z,
        b=b,
        batch=graph_id,
        graph_id=graph_id,
        frame_id=frame_id,
        sample_id=sample_id,
        dense_index=dense_index,
        dense_to_compact=dense_to_compact,
        frame_mask=frame_mask,
        atom_counts=tuple(int(value) for value in atom_counts),
        frames=frames,
        atoms=atoms,
        batch_size=batch_size,
        graph_count=frames * batch_size,
    )
def _union_edges(
    distance_edge_index: Tensor,
    distance_edge_weight: Tensor,
    distance_edge_vec: Tensor,
    bond_index: Tensor,
    *,
    pos: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Union distance and bond edges without sorting the full edge set."""

    if bond_index.numel() == 0:
        bond_type = torch.zeros(
            distance_edge_index.shape[1], dtype=torch.long, device=pos.device
        )
        return (
            distance_edge_index,
            distance_edge_weight,
            distance_edge_vec,
            bond_type,
        )
    bond_codes = bond_index[0] * int(pos.shape[0]) + bond_index[1]
    bond_order = torch.argsort(bond_codes)
    sorted_bond_codes = bond_codes[bond_order]
    if distance_edge_index.numel():
        distance_codes = (
            distance_edge_index[0] * int(pos.shape[0]) + distance_edge_index[1]
        )
        insertion = torch.searchsorted(sorted_bond_codes, distance_codes)
        keep_distance = insertion >= sorted_bond_codes.numel()
        if sorted_bond_codes.numel():
            safe_insertion = insertion.clamp(max=sorted_bond_codes.numel() - 1)
            keep_distance = keep_distance | (
                sorted_bond_codes[safe_insertion] != distance_codes
            )
        kept_distance_index = distance_edge_index[:, keep_distance]
        kept_distance_weight = distance_edge_weight[keep_distance]
        kept_distance_vec = distance_edge_vec[keep_distance]
    else:
        keep_distance = torch.empty((0,), dtype=torch.bool, device=pos.device)
        kept_distance_index = distance_edge_index
        kept_distance_weight = distance_edge_weight
        kept_distance_vec = distance_edge_vec
    bond_vec = pos[bond_index[0]] - pos[bond_index[1]]
    bond_weight = torch.linalg.vector_norm(bond_vec, dim=-1)
    edge_index = torch.cat([kept_distance_index, bond_index], dim=1)
    edge_weight = torch.cat([kept_distance_weight, bond_weight], dim=0)
    edge_vec = torch.cat([kept_distance_vec, bond_vec], dim=0)
    bond_type = torch.cat(
        [
            torch.zeros(
                kept_distance_index.shape[1],
                dtype=torch.long,
                device=pos.device,
            ),
            torch.ones(bond_index.shape[1], dtype=torch.long, device=pos.device),
        ],
        dim=0,
    )
    return edge_index, edge_weight, edge_vec, bond_type
def _replicate_sample_bonds(
    local_bonds: Tensor,
    *,
    frame_indices: Tensor,
    sample_start: int,
    sample_atoms: int,
    packed_atoms: int,
    dense_to_compact: Tensor,
) -> Tensor:
    if local_bonds.numel() == 0 or frame_indices.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=local_bonds.device)
    _assert_device_condition(
        (local_bonds >= 0) & (local_bonds < sample_atoms),
        "cached bond topology exceeds its sample atom count",
    )
    offsets = (
        frame_indices.to(device=local_bonds.device, dtype=torch.long) * int(packed_atoms)
        + int(sample_start)
    )
    dense = (
        local_bonds.unsqueeze(1) + offsets.view(1, -1, 1)
    ).permute(0, 1, 2).reshape(2, -1)
    compact = dense_to_compact.index_select(0, dense.reshape(-1)).reshape_as(dense)
    _assert_device_condition(
        compact >= 0,
        "padded frame nodes must not receive replicated covalent bonds",
    )
    return compact
def build_frame_graph(
    nodes: FrameNodeBatch,
    batch: ClipBatch | Mapping[str, Any],
    *,
    neighbor_builder: Any,
    topology_cache: BoundedTopologyCache | None,
    distance_bond_cache: DistanceOnlyBondCache | None = None,
    bond_construction_mode: str = "topology",
    graph_mode: str = "external",
    spatial_backbone: str = "torchmd_et",
) -> FrameGraphBatch:
    """Prepare one isolated graph per ``(sample, frame)`` pair."""

    if graph_mode != "external":
        raise RuntimeError(
            "visnet_radius uses native_radius; external graph construction "
            "is intentionally unavailable"
        )
    pos = nodes.pos
    frames = nodes.frames
    atoms = nodes.atoms
    atom_counts = nodes.atom_counts
    batch_size = nodes.batch_size
    frame_mask = nodes.frame_mask
    z = nodes.z
    b = nodes.b
    sample_id = nodes.sample_id
    frame_id = nodes.frame_id
    graph_id = nodes.graph_id

    topology_ids = tuple(str(value) for value in _get_field(batch, "topology_id"))
    if len(topology_ids) != batch_size:
        raise ValueError("topology_id must contain one entry per packed sample")
    identity_hashes = (
        _host_atom_identity_hashes(batch, batch_size)
        if bond_construction_mode == "distance_only"
        else tuple(None for _ in range(batch_size))
    )
    replicated_parts: list[Tensor] = []
    sample_starts: list[int] = []
    cursor = 0
    for sample_atoms in atom_counts:
        sample_starts.append(cursor)
        cursor += int(sample_atoms)
    for sample_index, topology_id in enumerate(topology_ids):
        sample_start = sample_starts[sample_index]
        sample_atoms = int(atom_counts[sample_index])
        if bond_construction_mode == "topology":
            local_bonds = topology_cache.materialize(
                topology_id,
                device=pos.device,
                atom_count=sample_atoms,
                allow_cpu_test=neighbor_builder.backend_used == "dense_test",
            )
        else:
            if distance_bond_cache is None:
                raise RuntimeError("distance_only cache is unavailable")
            local_bonds = distance_bond_cache.materialize(
                topology_id,
                device=pos.device,
                atom_count=sample_atoms,
                atom_identity_sha256=identity_hashes[sample_index],
            )
        replicated_parts.append(
            _replicate_sample_bonds(
                local_bonds,
                frame_indices=torch.nonzero(
                    frame_mask[sample_index], as_tuple=False
                ).flatten(),
                sample_start=sample_start,
                sample_atoms=sample_atoms,
                packed_atoms=atoms,
                dense_to_compact=nodes.dense_to_compact,
            )
        )
    if replicated_parts:
        replicated_bonds = torch.cat(replicated_parts, dim=1)
    else:
        replicated_bonds = torch.empty((2, 0), dtype=torch.long, device=pos.device)

    neighbors: NeighborList = neighbor_builder(
        pos, graph_id, batch_size=frames * batch_size
    )
    distance_edges = neighbors.edge_index
    distance_weight = neighbors.edge_weight
    distance_vec = neighbors.edge_vec
    edge_index, edge_weight, edge_vec, bond_type = _union_edges(
        distance_edges,
        distance_weight,
        distance_vec,
        replicated_bonds,
        pos=pos,
    )
    if edge_index.numel():
        _assert_device_condition(
            graph_id[edge_index[0]] == graph_id[edge_index[1]],
            "constructed a cross-frame or cross-sample edge",
        )
    return FrameGraphBatch(
        pos=pos,
        z=z,
        b=b,
        batch=graph_id,
        graph_id=graph_id,
        frame_id=frame_id,
        sample_id=sample_id,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_vec=edge_vec,
        bond_type=bond_type,
        bond_index=replicated_bonds,
        distance_edge_index=distance_edges,
        distance_edge_weight=distance_weight,
        distance_edge_vec=distance_vec,
        frame_mask=frame_mask,
        backend_used=neighbor_builder.backend_used,
        bond_construction_mode=bond_construction_mode,
        graph_mode=graph_mode,
        spatial_backbone=spatial_backbone,
    )

def unpack_frame_features(result: Any, nodes: FrameNodeBatch) -> tuple[Tensor, Tensor]:
    """Restore time-major scalar/vector features from the packed node axis."""

    if isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise ValueError("spatial encoder must return scalar and vector features")
        h, v = result[0], result[1]
    else:
        h = getattr(result, "h", getattr(result, "scalar", None))
        v = getattr(result, "v", getattr(result, "vector", None))
    if not isinstance(h, Tensor) or not isinstance(v, Tensor):
        raise ValueError("spatial encoder must return tensor scalar/vector features")
    expected_nodes = nodes.pos.shape[0]
    if h.ndim != 2 or h.shape[0] != expected_nodes:
        raise ValueError(
            "spatial scalar output must have shape [T*N_total, C], "
            f"got {tuple(h.shape)}"
        )
    if v.ndim != 3 or v.shape[0] != expected_nodes or v.shape[1] != 3:
        raise ValueError(
            "spatial vector output must have shape [T*N_total, 3, C], "
            f"got {tuple(v.shape)}"
        )
    dense_h = h.new_zeros((nodes.frames * nodes.atoms, h.shape[-1]))
    dense_v = v.new_zeros((nodes.frames * nodes.atoms, 3, v.shape[-1]))
    dense_h.index_copy_(0, nodes.dense_index, h)
    dense_v.index_copy_(0, nodes.dense_index, v)
    return (
        dense_h.reshape(nodes.frames, nodes.atoms, h.shape[-1]),
        dense_v.reshape(nodes.frames, nodes.atoms, 3, v.shape[-1]),
    )
