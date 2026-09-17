"""Batched spatial encoding for the multi-frame codec path.

The legacy PVB models consume one packed structure at a time.  This module
keeps that path untouched and provides the small adapter needed by the clip
contract: a time-major ``[T, N_total, 3]`` tensor is flattened into one graph
batch with one graph id per ``(sample, frame)`` pair.  The spatial backbone is
called exactly once and its scalar/vector outputs are restored to the
time-major layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from data.clip_dataset import ClipBatch
from utils.bio_utils import NUM_ATOM_TYPE, NUM_BLOCK_TYPE

from .bond_sources import DistanceOnlyBondCache
from .neighbor_graph import NeighborList, make_neighbor_list
from .topology_cache import BoundedTopologyCache
from .torchmd_et import TorchMD_VQ_ET
from .visnet import (
    MolViSNetEncoder,
    SpatialEncoderOutput,
    SUPPORTED_SPATIAL_BACKBONES,
    make_spatial_backbone,
)


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
        torch.any(frame_mask, dim=1),
        "every sample must contain at least one valid frame",
    )

    pos = x.reshape(frames * atoms, 3)
    z = _expand_atom_field(
        _as_tensor(_get_field(batch, "atype"), name="atype", dtype=torch.long),
        frames=frames,
        atoms=atoms,
        name="atype",
    )
    b = _expand_atom_field(
        _as_tensor(_get_field(batch, "btype"), name="btype", dtype=torch.long),
        frames=frames,
        atoms=atoms,
        name="btype",
    )
    sample_id = abid.repeat(frames)
    frame_id = torch.arange(
        frames, device=pos.device, dtype=torch.long
    ).repeat_interleave(atoms)
    graph_id = frame_id * batch_size + sample_id
    return FrameNodeBatch(
        pos=pos,
        z=z,
        b=b,
        batch=graph_id,
        graph_id=graph_id,
        frame_id=frame_id,
        sample_id=sample_id,
        frame_mask=frame_mask,
        atom_counts=tuple(int(value) for value in atom_counts),
        frames=frames,
        atoms=atoms,
        batch_size=batch_size,
        graph_count=frames * batch_size,
    )


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
class CheckpointLoadReport:
    """Auditable result of optional spatial-backbone checkpoint loading."""

    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = ()

    @property
    def matched_keys(self) -> tuple[str, ...]:
        return self.matched

    @property
    def missing_keys(self) -> tuple[str, ...]:
        return self.missing

    @property
    def unexpected_keys(self) -> tuple[str, ...]:
        return self.unexpected

    def __getitem__(self, key: str) -> Any:
        if key in {"matched", "matched_keys"}:
            return self.matched
        if key in {"missing", "missing_keys"}:
            return self.missing
        if key in {"unexpected", "unexpected_keys"}:
            return self.unexpected
        if key == "shape_mismatch":
            return self.shape_mismatch
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": list(self.matched),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "shape_mismatch": [
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint_shape),
                    "model_shape": list(model_shape),
                }
                for key, checkpoint_shape, model_shape in self.shape_mismatch
            ],
        }


@dataclass
class FrameEncoderOutput:
    """Time-major scalar/vector features plus auditable graph metadata."""

    h: Tensor  # [T, N_total, C], invariant scalar features
    v: Tensor  # [T, N_total, 3, C], equivariant vector features
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
        # Keep the common ``h, v = encoder(batch)`` spelling convenient while
        # retaining graph metadata for tests and downstream codec modules.
        yield self.h
        yield self.v


class PVBFrameEncoder(nn.Module):
    """Batch all frames through one shared selectable spatial-backbone call.

    ``ClipBatch.x`` is time-major and atoms are packed by sample.  The graph
    id formula is deliberately explicit and stable:

    ``graph_id[t, atom] = t * batch_size + abid[atom]``.

    The default backbone is the existing PVB TorchMD implementation.  ViSNet
    variants use the same node packing and expose the same representation
    contract. Tests and small downstream adapters may inject an equivalent module through
    ``spatial_encoder``; it must return scalar and vector features as its
    first two outputs.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_rbf: int = 50,
        num_heads: int = 8,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        spatial_backbone: str = "torchmd_et",
        lmax: int = 1,
        vertex: bool = True,
        trainable_rbf: bool = False,
        vecnorm_type: str | None = None,
        vertex_type: str | None = None,
        rbf_type: str | None = None,
        trainable_vecnorm: bool = False,
        *,
        spatial_encoder: nn.Module | None = None,
        encoder: nn.Module | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint: str | Path | None = None,
        neighbor_backend: str = "cuda_radius",
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
        if spatial_encoder is not None and encoder is not None:
            raise ValueError("pass only one of spatial_encoder and encoder")
        if checkpoint_path is not None and checkpoint is not None:
            raise ValueError("pass only one of checkpoint_path and checkpoint")
        if max_num_neighbors < 1:
            raise ValueError("max_num_neighbors must be positive")
        if neighbor_backend not in {"cuda_radius", "dense_test"}:
            raise ValueError(
                "neighbor_backend must be 'cuda_radius' for production or "
                "'dense_test' for explicit CPU tests"
            )
        selected_backbone = str(spatial_backbone).lower()
        if selected_backbone not in SUPPORTED_SPATIAL_BACKBONES:
            raise ValueError(
                f"unsupported spatial_backbone {spatial_backbone!r}; expected one of "
                f"{SUPPORTED_SPATIAL_BACKBONES}"
            )
        v2_backbone = selected_backbone in {
            "visnet_v2_radius",
            "visnet_v2_bonded",
        }
        if v2_backbone:
            if int(lmax) not in {1, 2}:
                raise ValueError(
                    f"v2 spatial lmax must be 1 or 2; got {lmax}"
                )
        elif int(lmax) != 1:
            raise ValueError(
                f"spatial lmax must be lmax=1 for the codec vector contract; got {lmax}"
            )
        if not v2_backbone and (
            vertex_type is not None
            or rbf_type is not None
            or bool(trainable_vecnorm)
        ):
            raise ValueError(
                "vertex_type, rbf_type, and trainable_vecnorm are v2-only "
                "options; legacy v1 backbones use vertex, Gaussian RBF, and "
                "their existing vector semantics"
            )
        resolved_vertex_type = (
            (
                str(vertex_type).lower()
                if vertex_type is not None
                else ("edge" if bool(vertex) else "none")
            )
            if v2_backbone
            else None
        )
        resolved_rbf_type = (
            (str(rbf_type).lower() if rbf_type is not None else "expnorm")
            if v2_backbone
            else None
        )
        if bond_construction is None:
            bond_config: dict[str, Any] = {"mode": "topology"}
        elif isinstance(bond_construction, str):
            bond_config = {"mode": bond_construction}
        elif isinstance(bond_construction, Mapping):
            bond_config = dict(bond_construction)
        else:
            raise TypeError("bond_construction must be a mode string or mapping")
        bond_mode = str(bond_config.get("mode", "topology"))
        if bond_mode not in {"topology", "distance_only"}:
            raise ValueError(
                "bond_construction.mode must be 'topology' or 'distance_only'"
            )
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_num_neighbors = int(max_num_neighbors)
        self.neighbor_backend = str(neighbor_backend)
        self.spatial_backbone = selected_backbone
        self.graph_mode = (
            "native_radius"
            if selected_backbone in {"visnet_radius", "visnet_v2_radius"}
            else "external"
        )
        self.bond_construction_mode = (
            "native_radius" if self.graph_mode == "native_radius" else bond_mode
        )
        self.bond_construction = bond_config
        self.vertex_type = resolved_vertex_type
        self.rbf_type = resolved_rbf_type
        self.trainable_vecnorm = bool(trainable_vecnorm)
        self.neighbor_builder = None
        self.topology_cache = None
        self.distance_bond_cache = None
        if self.graph_mode == "external":
            self.neighbor_builder = make_neighbor_list(
                self.neighbor_backend,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                max_num_neighbors=self.max_num_neighbors,
                loop=True,
            )
            self.topology_cache = BoundedTopologyCache(
                max_canonical_entries=int(topology_cache_capacity),
                max_device_entries=int(topology_device_cache_capacity),
            )
            self.distance_bond_cache = (
                DistanceOnlyBondCache(
                    min_distance_angstrom=float(distance_bond_min),
                    max_distance_angstrom=float(distance_bond_max),
                    max_num_neighbors=int(distance_bond_max_num_neighbors),
                    capacity=int(distance_bond_cache_capacity),
                )
                if bond_mode == "distance_only"
                else None
            )
        self.spatial_encoder = (
            spatial_encoder if spatial_encoder is not None else encoder
        )
        if self.spatial_encoder is None:
            self.spatial_encoder = make_spatial_backbone(
                selected_backbone,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                num_heads=num_heads,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=NUM_ATOM_TYPE,
                max_b=NUM_BLOCK_TYPE,
                max_num_neighbors=max_num_neighbors,
                neighbor_backend=neighbor_backend,
                lmax=lmax,
                vertex=vertex,
                trainable_rbf=trainable_rbf,
                vecnorm_type=vecnorm_type,
                vertex_type=resolved_vertex_type,
                rbf_type=resolved_rbf_type,
                trainable_vecnorm=trainable_vecnorm,
                dtype=dtype,
            )
        self.checkpoint_report: CheckpointLoadReport | None = None
        selected_checkpoint = checkpoint_path if checkpoint_path is not None else checkpoint
        if selected_checkpoint is not None:
            self.checkpoint_report = self.load_checkpoint(selected_checkpoint)

    @property
    def encoder_module(self) -> nn.Module:
        """Alias useful to callers that call the backbone an encoder."""

        return self.spatial_encoder

    def _distance_edges(
        self,
        pos: Tensor,
        graph_id: Tensor,
        *,
        graph_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Construct graph neighbors using host-derived graph cardinality."""

        neighbors: NeighborList = self.neighbor_builder(
            pos,
            graph_id,
            batch_size=int(graph_count),
        )
        return neighbors.edge_index, neighbors.edge_weight, neighbors.edge_vec

    @staticmethod
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

    def _register_topology_batch(self, batch: ClipBatch | Mapping[str, Any]) -> None:
        topology_ids = tuple(str(value) for value in _get_field(batch, "topology_id"))
        atom_ptr = _as_tensor(_get_field(batch, "atom_ptr"), name="atom_ptr", dtype=torch.long)
        abid = _as_tensor(_get_field(batch, "abid"), name="abid", dtype=torch.long).flatten()
        bond_index = _as_tensor(
            _get_field(batch, "bond_index"), name="bond_index", dtype=torch.long
        )
        if atom_ptr.device.type != "cpu" or abid.device.type != "cpu" or bond_index.device.type != "cpu":
            raise RuntimeError(
                "topology registration must happen on the CPU batch before CUDA transfer; "
                "GPU-to-CPU graph round trips are forbidden"
            )
        if len(topology_ids) != atom_ptr.numel() - 1:
            raise ValueError("topology_id must contain one entry per packed sample")
        if bond_index.numel() and torch.any(
            abid[bond_index[0]] != abid[bond_index[1]]
        ):
            raise ValueError("bond_index contains a cross-sample bond")
        for sample_index, topology_id in enumerate(topology_ids):
            start = int(atom_ptr[sample_index])
            stop = int(atom_ptr[sample_index + 1])
            if bond_index.numel():
                same_sample = abid[bond_index[0]] == sample_index
                local = bond_index[:, same_sample] - start
            else:
                local = torch.empty((2, 0), dtype=torch.long)
            self.topology_cache.register(
                topology_id,
                local,
                atom_count=stop - start,
            )

    def prepare_batch(self, batch: ClipBatch | Mapping[str, Any]) -> None:
        """Register topology from an untransferred CPU batch exactly once."""

        if self.bond_construction_mode != "topology":
            return
        atom_ptr = _as_tensor(
            _get_field(batch, "atom_ptr"), name="atom_ptr", dtype=torch.long
        )
        if atom_ptr.device.type != "cpu":
            raise RuntimeError(
                "topology registration requires the CPU batch before CUDA transfer; "
                "call model.prepare_batch(cpu_batch) before _to_device"
            )
        self._register_topology_batch(batch)

    def prepare_distance_bonds(
        self,
        references: Mapping[str, Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> list[dict[str, Any]]:
        if self.bond_construction_mode != "distance_only":
            raise RuntimeError("distance references are only valid in distance_only mode")
        if self.distance_bond_cache is None:
            raise RuntimeError("distance_only cache was not initialized")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "distance_only bond construction requires a CUDA device; "
                "refusing a CPU fallback"
            )
        manifest: list[dict[str, Any]] = []
        for topology_id in sorted(references):
            item = references[topology_id]
            registration = self.distance_bond_cache.register_reference(
                topology_id,
                torch.as_tensor(item["coordinates"], dtype=torch.float32),
                device=device,
                atom_identity_sha256=item.get("atom_identity_sha256"),
                sample_id=str(item.get("sample_id", "")),
                source_split=str(item.get("source_split", "unknown")),
            )
            expected_coordinate_hash = item.get("coordinate_sha256")
            if expected_coordinate_hash not in (None, "") and str(expected_coordinate_hash) != registration["coordinate_sha256"]:
                raise RuntimeError(
                    f"distance-only canonical coordinate hash mismatch for {topology_id!r}"
                )
            manifest.append(registration)
        return manifest

    def _replicate_sample_bonds(
        self,
        local_bonds: Tensor,
        *,
        frame_count: int,
        sample_start: int,
        sample_atoms: int,
        packed_atoms: int,
    ) -> Tensor:
        if local_bonds.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=local_bonds.device)
        _assert_device_condition(
            (local_bonds >= 0) & (local_bonds < sample_atoms),
            "cached bond topology exceeds its sample atom count",
        )
        offsets = (
            torch.arange(frame_count, device=local_bonds.device, dtype=torch.long)
            * int(packed_atoms)
            + int(sample_start)
        )
        return (
            local_bonds.unsqueeze(1) + offsets.view(1, -1, 1)
        ).permute(0, 1, 2).reshape(2, -1)

    def build_external_graph(
        self,
        nodes: FrameNodeBatch,
        batch: ClipBatch | Mapping[str, Any],
    ) -> FrameGraphBatch:
        """Prepare one isolated graph per ``(sample, frame)`` pair."""

        if self.graph_mode != "external":
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
            if self.bond_construction_mode == "distance_only"
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
            if self.bond_construction_mode == "topology":
                local_bonds = self.topology_cache.materialize(
                    topology_id,
                    device=pos.device,
                    atom_count=sample_atoms,
                    allow_cpu_test=self.neighbor_backend == "dense_test",
                )
            else:
                if self.distance_bond_cache is None:
                    raise RuntimeError("distance_only cache is unavailable")
                local_bonds = self.distance_bond_cache.materialize(
                    topology_id,
                    device=pos.device,
                    atom_count=sample_atoms,
                    atom_identity_sha256=identity_hashes[sample_index],
                )
            replicated_parts.append(
                self._replicate_sample_bonds(
                    local_bonds,
                    frame_count=frames,
                    sample_start=sample_start,
                    sample_atoms=sample_atoms,
                    packed_atoms=atoms,
                )
            )
        if replicated_parts:
            replicated_bonds = torch.cat(replicated_parts, dim=1)
        else:
            replicated_bonds = torch.empty((2, 0), dtype=torch.long, device=pos.device)

        distance_edges, distance_weight, distance_vec = self._distance_edges(
            pos,
            graph_id,
            graph_count=frames * batch_size,
        )
        edge_index, edge_weight, edge_vec, bond_type = self._union_edges(
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
            backend_used=self.neighbor_builder.backend_used,
            bond_construction_mode=self.bond_construction_mode,
            graph_mode=self.graph_mode,
            spatial_backbone=self.spatial_backbone,
        )

    def build_graph(self, batch: ClipBatch | Mapping[str, Any]) -> FrameGraphBatch:
        """Compatibility wrapper for callers that explicitly need external graphs."""

        if self.graph_mode != "external":
            raise RuntimeError(
                "visnet_radius uses native_radius and has no external graph "
                "compatibility path"
            )
        nodes = pack_frame_nodes(batch)
        return self.build_external_graph(nodes, batch)

    def _forward_external_spatial(
        self,
        nodes: FrameNodeBatch,
        graph: FrameGraphBatch,
    ) -> Any:
        forward_external = getattr(self.spatial_encoder, "forward_external", None)
        if callable(forward_external):
            return forward_external(nodes, graph)
        return self.spatial_encoder(
            z=graph.z,
            b=graph.b,
            pos=graph.pos,
            batch=graph.batch,
            edge_index=graph.edge_index,
            edge_weight_t=graph.edge_weight,
            edge_vec_t=graph.edge_vec,
            bond_type=graph.bond_type,
        )

    def _forward_native_spatial(self, nodes: FrameNodeBatch) -> Any:
        forward_native = getattr(self.spatial_encoder, "forward_native", None)
        if not callable(forward_native):
            raise TypeError(
                "visnet_radius requires a spatial encoder with forward_native(nodes)"
            )
        return forward_native(nodes)

    def _native_graph(
        self,
        nodes: FrameNodeBatch,
        result: Any,
    ) -> FrameGraphBatch:
        if not isinstance(result, SpatialEncoderOutput):
            raise TypeError(
                "native spatial encoders must return SpatialEncoderOutput with "
                "the native graph metadata"
            )
        if (
            result.edge_index is None
            or result.edge_weight is None
            or result.edge_vec is None
        ):
            raise ValueError(
                "native spatial output is missing edge_index/edge_weight/edge_vec"
            )
        edge_type = result.edge_type
        if edge_type is None:
            edge_type = torch.zeros(
                result.edge_index.shape[1],
                dtype=torch.long,
                device=result.edge_index.device,
            )
        empty_bonds = torch.empty(
            (2, 0), dtype=torch.long, device=nodes.pos.device
        )
        if result.edge_index.numel():
            _assert_device_condition(
                nodes.graph_id[result.edge_index[0]]
                == nodes.graph_id[result.edge_index[1]],
                "native radius graph produced a cross-frame or cross-sample edge",
            )
        return FrameGraphBatch(
            pos=nodes.pos,
            z=nodes.z,
            b=nodes.b,
            batch=nodes.batch,
            graph_id=nodes.graph_id,
            frame_id=nodes.frame_id,
            sample_id=nodes.sample_id,
            edge_index=result.edge_index,
            edge_weight=result.edge_weight,
            edge_vec=result.edge_vec,
            bond_type=edge_type,
            bond_index=empty_bonds,
            distance_edge_index=result.edge_index,
            distance_edge_weight=result.edge_weight,
            distance_edge_vec=result.edge_vec,
            frame_mask=nodes.frame_mask,
            backend_used=result.backend_used,
            bond_construction_mode=self.bond_construction_mode,
            graph_mode=result.graph_mode,
            spatial_backbone=self.spatial_backbone,
        )

    @staticmethod
    def _unpack_features(result: Any) -> tuple[Tensor, Tensor]:
        if isinstance(result, (tuple, list)):
            if len(result) < 2:
                raise ValueError("spatial encoder must return scalar and vector features")
            h, v = result[0], result[1]
        else:
            h = getattr(result, "h", getattr(result, "scalar", None))
            v = getattr(result, "v", getattr(result, "vector", None))
        if not isinstance(h, Tensor) or not isinstance(v, Tensor):
            raise ValueError("spatial encoder must return tensor scalar/vector features")
        return h, v

    def forward(self, batch: ClipBatch | Mapping[str, Any]) -> FrameEncoderOutput:
        nodes = pack_frame_nodes(batch)
        if self.graph_mode == "native_radius":
            result = self._forward_native_spatial(nodes)
            graph = self._native_graph(nodes, result)
        else:
            # CPU topology registration is an explicit caller-side phase.
            # Doing it here would be too late for a CUDA batch and would hide
            # a synchronizing pass on every forward.
            graph = self.build_external_graph(nodes, batch)
            result = self._forward_external_spatial(nodes, graph)
        h, v = self._unpack_features(result)
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
        frames = nodes.frames
        atoms = nodes.atoms
        return FrameEncoderOutput(
            h=h.reshape(frames, atoms, h.shape[-1]),
            v=v.reshape(frames, atoms, 3, v.shape[-1]),
            graph=graph,
        )

    encode = forward

    @staticmethod
    def _state_dict_from_checkpoint(payload: Any) -> Mapping[str, Tensor]:
        if isinstance(payload, Mapping):
            for key in ("state_dict", "model_state_dict", "model", "encoder"):
                candidate = payload.get(key)
                if isinstance(candidate, Mapping) and candidate:
                    payload = candidate
                    break
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint does not contain a state-dict mapping")
        state = {str(key): value for key, value in payload.items() if isinstance(value, Tensor)}
        if not state:
            raise ValueError("checkpoint state dict contains no tensor entries")
        return state

    @staticmethod
    def _candidate_keys(key: str) -> Sequence[str]:
        candidates = [key]
        prefixes = ("module.", "encoder.", "spatial_encoder.", "model.encoder.")
        changed = True
        while changed:
            changed = False
            for candidate in tuple(candidates):
                for prefix in prefixes:
                    if candidate.startswith(prefix):
                        stripped = candidate[len(prefix) :]
                        if stripped not in candidates:
                            candidates.append(stripped)
                            changed = True
        return candidates

    def load_checkpoint(self, path: str | Path) -> CheckpointLoadReport:
        """Load matching tensors and return a complete key compatibility report."""

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"encoder checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        source = self._state_dict_from_checkpoint(payload)
        target = self.spatial_encoder.state_dict()
        matched: list[str] = []
        missing = set(target)
        unexpected: list[str] = []
        shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        loadable: dict[str, Tensor] = {}
        for source_key, value in source.items():
            target_key = next(
                (candidate for candidate in self._candidate_keys(source_key) if candidate in target),
                None,
            )
            if target_key is None:
                unexpected.append(source_key)
                continue
            if tuple(value.shape) != tuple(target[target_key].shape):
                shape_mismatch.append(
                    (target_key, tuple(value.shape), tuple(target[target_key].shape))
                )
                continue
            loadable[target_key] = value
            matched.append(target_key)
            missing.discard(target_key)
        self.spatial_encoder.load_state_dict(loadable, strict=False)
        report = CheckpointLoadReport(
            matched=tuple(sorted(set(matched))),
            missing=tuple(sorted(missing)),
            unexpected=tuple(sorted(unexpected)),
            shape_mismatch=tuple(sorted(shape_mismatch)),
        )
        self.checkpoint_report = report
        return report


# Short aliases keep the adapter discoverable without duplicating the model.
FrameEncoder = PVBFrameEncoder
PVBFrameGraph = FrameGraphBatch
