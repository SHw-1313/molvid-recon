"""Coordinate-free latent fields, batch contract and exact contract hashes."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from ..codec.types import STATE_DETAIL_CODEC_SCHEMA
from ..geometry.types import StaticTopologyMetadata

DIT_ADAPTER_SCHEMA = "pvb.dit.state_detail.adapter.v2"
DIT_BATCH_SCHEMA = "pvb.dit.state_detail.batch.v2"
DIT_STATS_SCHEMA = "pvb.dit.state_detail.stats.v2"
DIT_MODEL_SCHEMA = "pvb.dit.state_detail.model.v2"
SUPPORTED_DIT_MODES = ("ratio2_state_detail", "ratio4_state_detail")
SUPPORTED_RATIOS = (2, 4)
FRAME_COUNT = 16
FRAME_LATENT_SCHEMA = "molvid.frame_joint.latent.v1"

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("contract contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def tensor_hash(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    payload = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes": tensor.numpy().tobytes().hex(),
    }
    return contract_hash(payload)


@dataclass(frozen=True)
class LatentFields:
    """The four fields predicted by the DiT or used by rectified flow."""

    state_h: Tensor
    detail_h: Tensor
    state_v: Tensor
    detail_v: Tensor

    def __post_init__(self) -> None:
        scalar = (self.state_h, self.detail_h)
        vector = (self.state_v, self.detail_v)
        if any(value.ndim != 3 for value in scalar):
            raise ValueError("scalar latent fields must have shape [K, N, C]")
        if any(value.ndim != 4 or value.shape[2] != 3 for value in vector):
            raise ValueError("vector latent fields must have shape [K, N, 3, C]")
        if self.detail_h.shape != self.state_h.shape or self.detail_v.shape != self.state_v.shape:
            raise ValueError("state and detail fields must have matching shapes")
        if self.state_v.shape[:2] != self.state_h.shape[:2]:
            raise ValueError("scalar and vector fields must share [K, N]")
        if self.state_v.shape[-1] != self.state_h.shape[-1]:
            raise ValueError("scalar and vector codec widths must match")

    def names(self) -> tuple[str, ...]:
        return ("state_h", "detail_h", "state_v", "detail_v")

    def as_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in self.names()}

    def map(self, function) -> "LatentFields":
        return LatentFields(*(function(getattr(self, name)) for name in self.names()))

    def detach(self) -> "LatentFields":
        return self.map(lambda value: value.detach())

    def clone(self) -> "LatentFields":
        return self.map(lambda value: value.clone())

    def to(self, *args, **kwargs) -> "LatentFields":
        return self.map(lambda value: value.to(*args, **kwargs))

    @classmethod
    def zeros_like(cls, value: "LatentFields") -> "LatentFields":
        return value.map(torch.zeros_like)


def _topology_field(
    topology: Any,
    name: str,
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if topology is not None and hasattr(topology, name):
        value = torch.as_tensor(getattr(topology, name), device=device, dtype=dtype).flatten()
        if value.numel() != n:
            raise ValueError(f"topology.{name} must have shape [N]")
        return value
    return torch.zeros(n, device=device, dtype=dtype)


def _topology_atom_ptr(topology: Any, batch_size: int, n: int, device: torch.device) -> Tensor:
    if topology is not None and hasattr(topology, "atom_ptr"):
        value = torch.as_tensor(topology.atom_ptr, device=device, dtype=torch.long).flatten()
        if value.numel() != batch_size + 1 or int(value[-1]) != n:
            raise ValueError("topology.atom_ptr does not match the latent batch")
        return value
    if batch_size == 1:
        return torch.tensor([0, n], device=device, dtype=torch.long)
    return torch.linspace(0, n, batch_size + 1, device=device, dtype=torch.long)


@dataclass
class LatentBatch:
    """Coordinate-free generative input/output contract.

    fields contains only the four encoded/generated banks. In particular, there
    are no raw detail members and no target coordinates.
    """

    fields: LatentFields
    token_mask: Tensor
    detail_valid: Tensor
    detail_component_mask: Tensor
    block_frame_mask: Tensor
    block_time_ps: Tensor
    frame_time_ps: Tensor
    abid: Tensor
    sample_origin: Tensor
    topology: Any
    ratio: int
    mode: str
    width: int
    observed_mask: Optional[Tensor] = None
    loss_mask: Optional[Tensor] = None
    atom_type: Optional[Tensor] = None
    block_type: Optional[Tensor] = None
    component_id: Optional[Tensor] = None
    block_id: Optional[Tensor] = None
    atom_ptr: Optional[Tensor] = None
    codec_hash: str = ""
    data_hash: str = ""
    stats_hash: str = ""
    schema_version: str = DIT_BATCH_SCHEMA
    adapter_schema: str = DIT_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        self.ratio = int(self.ratio)
        self.width = int(self.width)
        if self.ratio not in SUPPORTED_RATIOS:
            raise ValueError("DiT supports only ratios 2 and 4")
        expected_mode = f"ratio{self.ratio}_state_detail"
        if self.mode != expected_mode:
            raise ValueError(f"mode must be {expected_mode!r}")
        if self.schema_version != DIT_BATCH_SCHEMA:
            raise ValueError(f"unsupported DiT batch schema {self.schema_version!r}")
        if self.adapter_schema != DIT_ADAPTER_SCHEMA:
            raise ValueError(f"unsupported adapter schema {DIT_ADAPTER_SCHEMA!r}")
        k_expected = FRAME_COUNT // self.ratio
        self.abid = torch.as_tensor(self.abid, device=self.fields.state_h.device, dtype=torch.long).flatten()
        if self.fields.state_h.shape != (k_expected, self.abid.numel(), self.width):
            raise ValueError(
                f"state_h must have shape [K={k_expected}, N, C={self.width}], "
                f"got {tuple(self.fields.state_h.shape)}"
            )
        if self.fields.state_v.shape != (k_expected, self.abid.numel(), 3, self.width):
            raise ValueError("state_v has an invalid [K,N,3,C] shape")
        batch_size = int(self.token_mask.shape[0])
        if self.token_mask.shape != (batch_size, k_expected):
            raise ValueError("token_mask must have shape [B,K]")
        if self.detail_valid.shape != self.token_mask.shape:
            raise ValueError("detail_valid must have shape [B,K]")
        if self.block_frame_mask.shape != (batch_size, k_expected, self.ratio):
            raise ValueError("block_frame_mask must have shape [B,K,r]")
        if self.block_time_ps.shape != self.block_frame_mask.shape:
            raise ValueError("block_time_ps must have shape [B,K,r]")
        if self.frame_time_ps.ndim != 2 or self.frame_time_ps.shape[0] != batch_size:
            raise ValueError("frame_time_ps must have shape [B,T]")
        if self.sample_origin.shape != (batch_size, 3):
            raise ValueError("sample_origin must have shape [B,3]")
        if self.abid.numel() and (self.abid.min() < 0 or self.abid.max() >= batch_size):
            raise ValueError("abid contains an invalid sample id")
        self.token_mask = torch.as_tensor(self.token_mask, dtype=torch.bool, device=self.fields.state_h.device)
        self.detail_valid = torch.as_tensor(self.detail_valid, dtype=torch.bool, device=self.fields.state_h.device)
        self.detail_component_mask = torch.as_tensor(
            self.detail_component_mask, dtype=torch.bool, device=self.fields.state_h.device
        )
        if self.detail_component_mask.shape[:2] != (batch_size, k_expected):
            raise ValueError("detail_component_mask must start with [B,K]")
        if self.detail_component_mask.shape[-1] != self.ratio - 1:
            raise ValueError("detail_component_mask has the wrong coefficient count")
        if self.observed_mask is None:
            self.observed_mask = torch.zeros_like(self.token_mask, dtype=torch.bool)
        else:
            self.observed_mask = torch.as_tensor(
                self.observed_mask, device=self.token_mask.device, dtype=torch.bool
            )
            if self.observed_mask.shape != self.token_mask.shape:
                raise ValueError("observed_mask must have shape [B,K]")
        if self.loss_mask is not None:
            self.loss_mask = torch.as_tensor(
                self.loss_mask, device=self.abid.device, dtype=torch.bool
            ).flatten()
            if self.loss_mask.numel() != self.abid.numel():
                raise ValueError("loss_mask must have shape [N]")
        self.block_frame_mask = torch.as_tensor(
            self.block_frame_mask, dtype=torch.bool, device=self.fields.state_h.device
        )
        self.block_time_ps = torch.as_tensor(
            self.block_time_ps, dtype=self.fields.state_h.dtype, device=self.fields.state_h.device
        )
        self.frame_time_ps = torch.as_tensor(
            self.frame_time_ps, dtype=self.fields.state_h.dtype, device=self.fields.state_h.device
        )
        self.sample_origin = torch.as_tensor(
            self.sample_origin, dtype=self.fields.state_h.dtype, device=self.fields.state_h.device
        )
        if self.atom_type is None:
            self.atom_type = _topology_field(self.topology, "atom_type", self.abid.numel(), dtype=torch.long, device=self.abid.device)
        if self.block_type is None:
            self.block_type = _topology_field(self.topology, "block_type", self.abid.numel(), dtype=torch.long, device=self.abid.device)
        if self.component_id is None:
            self.component_id = _topology_field(self.topology, "component_id", self.abid.numel(), dtype=torch.long, device=self.abid.device)
        if self.block_id is None:
            self.block_id = _topology_field(self.topology, "block_id", self.abid.numel(), dtype=torch.long, device=self.abid.device)
        if self.atom_ptr is None:
            self.atom_ptr = _topology_atom_ptr(self.topology, batch_size, self.abid.numel(), self.abid.device)
        for name in ("atom_type", "block_type", "component_id", "block_id"):
            value = torch.as_tensor(getattr(self, name), device=self.abid.device, dtype=torch.long).flatten()
            if value.numel() != self.abid.numel():
                raise ValueError(f"{name} must have shape [N]")
            setattr(self, name, value)
        self.atom_ptr = torch.as_tensor(self.atom_ptr, device=self.abid.device, dtype=torch.long).flatten()
        if (
            self.atom_ptr.numel() != batch_size + 1
            or int(self.atom_ptr[0]) != 0
            or int(self.atom_ptr[-1]) != self.abid.numel()
            or torch.any(self.atom_ptr[1:] < self.atom_ptr[:-1])
        ):
            raise ValueError("atom_ptr must be monotone, start at zero, and end at N")
        expected_abid = torch.repeat_interleave(
            torch.arange(batch_size, device=self.abid.device, dtype=torch.long),
            self.atom_ptr[1:] - self.atom_ptr[:-1],
        )
        if not torch.equal(self.abid, expected_abid):
            raise ValueError("abid must follow the declared atom_ptr sample boundaries")
        if self.topology is not None:
            if hasattr(self.topology, "abid"):
                topology_abid = torch.as_tensor(
                    self.topology.abid, device=self.abid.device, dtype=torch.long
                ).flatten()
                if not torch.equal(topology_abid, self.abid):
                    raise ValueError("topology.abid does not match the latent batch")
            if hasattr(self.topology, "atom_ptr"):
                topology_ptr = torch.as_tensor(
                    self.topology.atom_ptr, device=self.abid.device, dtype=torch.long
                ).flatten()
                if not torch.equal(topology_ptr, self.atom_ptr):
                    raise ValueError("topology.atom_ptr does not match the latent batch")
        if any(not torch.isfinite(value).all() for value in self.fields.as_dict().values()):
            raise ValueError("all latent fields must be finite")

    @property
    def state_h(self) -> Tensor:
        return self.fields.state_h

    @property
    def detail_h(self) -> Tensor:
        return self.fields.detail_h

    @property
    def state_v(self) -> Tensor:
        return self.fields.state_v

    @property
    def detail_v(self) -> Tensor:
        return self.fields.detail_v

    @property
    def batch_size(self) -> int:
        return int(self.token_mask.shape[0])

    @property
    def tokens(self) -> int:
        return int(self.token_mask.shape[1])

    @property
    def num_atoms(self) -> int:
        return int(self.abid.numel())

    def state_atom_mask(self) -> Tensor:
        return self.token_mask.index_select(0, self.abid).transpose(0, 1)

    def detail_atom_mask(self) -> Tensor:
        return self.state_atom_mask() & self.detail_valid.index_select(0, self.abid).transpose(0, 1)

    def field_masks(self) -> dict[str, Tensor]:
        return {
            "state_h": self.state_atom_mask(),
            "detail_h": self.detail_atom_mask(),
            "state_v": self.state_atom_mask(),
            "detail_v": self.detail_atom_mask(),
        }

    def zero_invalid(self) -> "LatentBatch":
        masks = self.field_masks()
        fields = LatentFields(
            self.state_h * masks["state_h"].unsqueeze(-1),
            self.detail_h * masks["detail_h"].unsqueeze(-1),
            self.state_v * masks["state_v"].unsqueeze(-1).unsqueeze(-1),
            self.detail_v * masks["detail_v"].unsqueeze(-1).unsqueeze(-1),
        )
        return replace(self, fields=fields)

    def with_fields(self, fields: LatentFields) -> "LatentBatch":
        return replace(self, fields=fields)

    def with_observation(
        self, observed_mask: Tensor, *, sample_origin: Optional[Tensor] = None
    ) -> "LatentBatch":
        observed = torch.as_tensor(observed_mask, device=self.token_mask.device, dtype=torch.bool)
        if observed.shape != self.token_mask.shape:
            raise ValueError("observed_mask must have shape [B,K]")
        if torch.any(observed & ~self.token_mask):
            raise ValueError("observed_mask cannot include invalid tokens")
        origin = self.sample_origin if sample_origin is None else sample_origin.to(self.sample_origin)
        return replace(self, observed_mask=observed, sample_origin=origin)

    def to(self, *args, **kwargs) -> "LatentBatch":
        topology = self.topology
        if hasattr(topology, "to"):
            topology = topology.to(*args, **kwargs)
        return replace(
            self,
            fields=self.fields.to(*args, **kwargs),
            token_mask=self.token_mask.to(*args, **kwargs),
            detail_valid=self.detail_valid.to(*args, **kwargs),
            detail_component_mask=self.detail_component_mask.to(*args, **kwargs),
            block_frame_mask=self.block_frame_mask.to(*args, **kwargs),
            block_time_ps=self.block_time_ps.to(*args, **kwargs),
            frame_time_ps=self.frame_time_ps.to(*args, **kwargs),
            abid=self.abid.to(*args, **kwargs),
            sample_origin=self.sample_origin.to(*args, **kwargs),
            observed_mask=self.observed_mask.to(*args, **kwargs),
            loss_mask=None if self.loss_mask is None else self.loss_mask.to(*args, **kwargs),
            atom_type=self.atom_type.to(*args, **kwargs),
            block_type=self.block_type.to(*args, **kwargs),
            component_id=self.component_id.to(*args, **kwargs),
            block_id=self.block_id.to(*args, **kwargs),
            atom_ptr=self.atom_ptr.to(*args, **kwargs),
            topology=topology,
        )

    def contract(self) -> dict[str, Any]:
        topology_contract = self.topology.contract() if hasattr(self.topology, "contract") else None
        return {
            "schema_version": self.schema_version,
            "adapter_schema": self.adapter_schema,
            "codec_schema": STATE_DETAIL_CODEC_SCHEMA,
            "mode": self.mode,
            "ratio": self.ratio,
            "width": self.width,
            "state_h_shape": list(self.state_h.shape),
            "detail_h_shape": list(self.detail_h.shape),
            "state_v_shape": list(self.state_v.shape),
            "detail_v_shape": list(self.detail_v.shape),
            "token_mask_shape": list(self.token_mask.shape),
            "detail_valid_shape": list(self.detail_valid.shape),
            "detail_component_mask_shape": list(self.detail_component_mask.shape),
            "block_frame_mask_shape": list(self.block_frame_mask.shape),
            "block_time_ps_shape": list(self.block_time_ps.shape),
            "frame_time_ps_shape": list(self.frame_time_ps.shape),
            "abid_shape": list(self.abid.shape),
            "sample_origin_shape": list(self.sample_origin.shape),
            "observed_mask_shape": list(self.observed_mask.shape),
            "loss_mask_shape": None if self.loss_mask is None else list(self.loss_mask.shape),
            "loss_mask_hash": None if self.loss_mask is None else tensor_hash(self.loss_mask),
            "topology": topology_contract,
            "contains_raw_detail": False,
            "contains_target_coordinates": False,
            "contains_radius_edges": False,
            "contains_distances": False,
            "contains_frame_expanded_graph": False,
            "codec_hash": self.codec_hash,
            "data_hash": self.data_hash,
            "stats_hash": self.stats_hash,
        }

    @property
    def hash(self) -> str:
        return contract_hash(self.contract())


@dataclass(frozen=True)
class FrameLatentBatch:
    """Per-frame scalar/vector features on a packed atom axis.

    ``h`` is ``[T,N,C]`` and ``v`` is ``[T,N,3,C]``.  The xyz axis is an
    SO(3) vector axis; learned linear maps may only act on the final channel
    axis.  ``time_ps`` is physical time and is unrelated to rectified-flow
    progress.
    """

    h: Tensor
    v: Tensor
    time_ps: Tensor
    frame_mask: Tensor
    topology: StaticTopologyMetadata
    statistics_hash: str = ""
    schema_version: str = FRAME_LATENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FRAME_LATENT_SCHEMA:
            raise ValueError(f"unsupported frame latent schema {self.schema_version!r}")
        if self.h.ndim != 3:
            raise ValueError("frame scalar features must have shape [T,N,C]")
        if self.v.ndim != 4 or self.v.shape[:2] != self.h.shape[:2] or self.v.shape[2] != 3:
            raise ValueError("frame vector features must have shape [T,N,3,C]")
        if self.v.shape[-1] != self.h.shape[-1]:
            raise ValueError("frame scalar/vector channel widths must match")
        if self.topology.num_atoms != self.h.shape[1]:
            raise ValueError("frame features and topology disagree on packed atoms")
        if self.time_ps.shape != (self.topology.batch_size, self.h.shape[0]):
            raise ValueError("time_ps must have shape [B,T]")
        if self.frame_mask.shape != self.time_ps.shape:
            raise ValueError("frame_mask must have shape [B,T]")
        if self.h.device != self.v.device or self.h.device != self.topology.abid.device:
            raise ValueError("frame features and topology must share a device")
        if not torch.isfinite(self.h).all() or not torch.isfinite(self.v).all():
            raise ValueError("frame latent contains NaN or Inf")
        clock = self.time_ps.to(device=self.h.device, dtype=self.h.dtype)
        mask = self.frame_mask.to(device=self.h.device, dtype=torch.bool)
        object.__setattr__(self, "time_ps", clock)
        object.__setattr__(self, "frame_mask", mask)
        for sample in range(self.batch_size):
            valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
            if valid.numel() and not torch.equal(valid, torch.arange(valid.numel(), device=valid.device)):
                raise ValueError("frame_mask must be a valid prefix")
            values = clock[sample].index_select(0, valid)
            if values.numel() > 1 and not bool(torch.all(values[1:] > values[:-1])):
                raise ValueError("valid physical times must be strictly increasing")

    @property
    def frames(self) -> int:
        return int(self.h.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.h.shape[1])

    @property
    def width(self) -> int:
        return int(self.h.shape[-1])

    @property
    def batch_size(self) -> int:
        return self.topology.batch_size

    @property
    def abid(self) -> Tensor:
        return self.topology.abid

    @property
    def atom_ptr(self) -> Tensor:
        return self.topology.atom_ptr

    @property
    def atom_type(self) -> Tensor:
        return self.topology.atom_type

    @property
    def block_type(self) -> Tensor:
        return self.topology.block_type

    @property
    def component_id(self) -> Tensor:
        return self.topology.component_id

    @property
    def block_id(self) -> Tensor:
        return self.topology.block_id

    @property
    def token_mask(self) -> Tensor:
        return self.frame_mask

    @property
    def state_h(self) -> Tensor:
        """Compatibility name used only by the shared factorized layout."""

        return self.h

    def atom_frame_mask(self) -> Tensor:
        return self.frame_mask.index_select(0, self.abid).transpose(0, 1)

    def with_features(
        self,
        h: Tensor,
        v: Tensor,
        *,
        statistics_hash: str | None = None,
    ) -> "FrameLatentBatch":
        """Replace differentiable fields without rechecking static metadata.

        Flow integration calls this method many times with the same validated
        topology, clock, and mask.  Re-running the constructor there would
        synchronize CUDA merely to repeat finite/prefix checks.
        """

        if h.shape != self.h.shape or v.shape != self.v.shape:
            raise ValueError("replacement frame features must preserve shape")
        if h.device != self.h.device or v.device != self.v.device:
            raise ValueError("replacement frame features must preserve device")
        value = copy.copy(self)
        object.__setattr__(value, "h", h)
        object.__setattr__(value, "v", v)
        if statistics_hash is not None:
            object.__setattr__(value, "statistics_hash", str(statistics_hash))
        return value

    def slice_frames(self, start: int, stop: int) -> "FrameLatentBatch":
        if not 0 <= int(start) < int(stop) <= self.frames:
            raise ValueError("invalid frame latent slice")
        return replace(
            self,
            h=self.h[int(start):int(stop)],
            v=self.v[int(start):int(stop)],
            time_ps=self.time_ps[:, int(start):int(stop)],
            frame_mask=self.frame_mask[:, int(start):int(stop)],
        )

    def detach(self) -> "FrameLatentBatch":
        return self.with_features(self.h.detach(), self.v.detach())

    def to(self, *args, **kwargs) -> "FrameLatentBatch":
        device = self.h.to(*args, **kwargs).device
        return replace(
            self,
            h=self.h.to(*args, **kwargs),
            v=self.v.to(*args, **kwargs),
            time_ps=self.time_ps.to(device=device),
            frame_mask=self.frame_mask.to(device=device),
            topology=self.topology.to(device),
        )


@dataclass(frozen=True)
class QuerySpec:
    """Future query clock and validity; never contains target coordinates."""

    time_ps: Tensor
    frame_mask: Tensor

    def __post_init__(self) -> None:
        if self.time_ps.ndim != 2 or self.frame_mask.shape != self.time_ps.shape:
            raise ValueError("query time_ps/frame_mask must have shape [B,Q]")
        if not torch.isfinite(self.time_ps).all():
            raise ValueError("query time_ps contains NaN or Inf")
        object.__setattr__(self, "frame_mask", self.frame_mask.to(dtype=torch.bool))

    @property
    def batch_size(self) -> int:
        return int(self.time_ps.shape[0])

    @property
    def frames(self) -> int:
        return int(self.time_ps.shape[1])


@dataclass(frozen=True)
class ObservedContext:
    """Observed-only model condition with an observation-defined origin."""

    latent: FrameLatentBatch
    coordinates: Tensor
    sample_origin: Tensor
    loss_mask: Tensor

    def __post_init__(self) -> None:
        if self.coordinates.shape != (self.latent.frames, self.latent.num_atoms, 3):
            raise ValueError("observed coordinates must have shape [H,N,3]")
        if self.sample_origin.shape != (self.latent.batch_size, 3):
            raise ValueError("sample_origin must have shape [B,3]")
        if self.loss_mask.shape != (self.latent.num_atoms,):
            raise ValueError("loss_mask must have shape [N]")
        if self.latent.frames < 1 or not bool(torch.all(self.latent.frame_mask[:, 0])):
            raise ValueError("observed context requires a valid first frame")

    @property
    def topology(self) -> StaticTopologyMetadata:
        return self.latent.topology

    def last_features(self) -> tuple[Tensor, Tensor]:
        last = self.latent.frame_mask.sum(dim=1).to(dtype=torch.long) - 1
        atom_last = last.index_select(0, self.latent.abid)
        atom = torch.arange(self.latent.num_atoms, device=self.latent.h.device)
        return self.latent.h[atom_last, atom], self.latent.v[atom_last, atom]

    def reference_centered_coordinates(self) -> Tensor:
        atom_origin = self.sample_origin.index_select(0, self.latent.abid)
        return self.coordinates[0] - atom_origin


@dataclass(frozen=True)
class HistoryMemory:
    """Compressed observed history; detail fields are history-only diagnostics."""

    h: Tensor
    v: Tensor
    token_mask: Tensor
    group_time_ps: Tensor
    group_frame_mask: Tensor
    abid: Tensor
    detail_h: Tensor
    detail_v: Tensor

    def __post_init__(self) -> None:
        if self.h.ndim != 3 or self.v.ndim != 4 or self.v.shape[:2] != self.h.shape[:2] or self.v.shape[2] != 3:
            raise ValueError("history memory expects h=[M,N,C], v=[M,N,3,Cv]")
        batch, tokens = self.token_mask.shape
        if self.h.shape[0] != tokens or self.h.shape[1] != self.abid.numel():
            raise ValueError("history token/atom axes disagree")
        if self.group_time_ps.shape != self.group_frame_mask.shape or self.group_time_ps.shape[:2] != (batch, tokens):
            raise ValueError("history group time fields must have shape [B,M,G]")
        if self.detail_h.shape[:2] != self.h.shape[:2] or self.detail_v.shape[:3] != self.v.shape[:3]:
            raise ValueError("history detail diagnostics disagree with memory axes")

    @property
    def tokens(self) -> int:
        return int(self.h.shape[0])

    def representative_time_ps(self) -> Tensor:
        valid = self.group_frame_mask.to(dtype=self.group_time_ps.dtype)
        return (self.group_time_ps * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
