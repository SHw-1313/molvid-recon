"""State/detail latent contracts for the v1 molecular DiT probe.

This module is deliberately a boundary around the frozen state/detail codec. It
contains no coordinate processing and no graph construction. The generated
model sees one token per (latent time, atom) location; state and detail are
concatenated on channels rather than represented as separate sequence tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from .state_detail_codec_v2 import (
    RATIO_FOR_MODE,
    STATE_DETAIL_CODEC_SCHEMA,
    StateDetailLatent,
    compute_masked_centroid_origin,
)


DIT_ADAPTER_SCHEMA = "pvb.dit.state_detail.adapter.v2"
DIT_BATCH_SCHEMA = "pvb.dit.state_detail.batch.v2"
DIT_STATS_SCHEMA = "pvb.dit.state_detail.stats.v2"
DIT_MODEL_SCHEMA = "pvb.dit.state_detail.model.v2"
SUPPORTED_DIT_MODES = ("ratio2_state_detail", "ratio4_state_detail")
SUPPORTED_RATIOS = (2, 4)
FRAME_COUNT = 16


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


class AxisPreservingLinear(nn.Module):
    """A channel-only, bias-free linear map for vector features."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        if int(input_width) < 1 or int(output_width) < 1:
            raise ValueError("AxisPreservingLinear widths must be positive")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.weight = nn.Parameter(torch.empty(self.output_width, self.input_width))
        self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.input_width:
            raise ValueError(
                f"expected final channel width {self.input_width}, got {value.shape[-1]}"
            )
        return torch.matmul(value, self.weight.t())


@dataclass(frozen=True)
class LatentFieldSet:
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

    def map(self, function) -> "LatentFieldSet":
        return LatentFieldSet(*(function(getattr(self, name)) for name in self.names()))

    def detach(self) -> "LatentFieldSet":
        return self.map(lambda value: value.detach())

    def clone(self) -> "LatentFieldSet":
        return self.map(lambda value: value.clone())

    def to(self, *args, **kwargs) -> "LatentFieldSet":
        return self.map(lambda value: value.to(*args, **kwargs))

    @classmethod
    def zeros_like(cls, value: "LatentFieldSet") -> "LatentFieldSet":
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
class DiTLatentBatch:
    """Coordinate-free generative input/output contract.

    fields contains only the four encoded/generated banks. In particular, there
    are no raw detail members and no target coordinates.
    """

    fields: LatentFieldSet
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

    def zero_invalid(self) -> "DiTLatentBatch":
        masks = self.field_masks()
        fields = LatentFieldSet(
            self.state_h * masks["state_h"].unsqueeze(-1),
            self.detail_h * masks["detail_h"].unsqueeze(-1),
            self.state_v * masks["state_v"].unsqueeze(-1).unsqueeze(-1),
            self.detail_v * masks["detail_v"].unsqueeze(-1).unsqueeze(-1),
        )
        return replace(self, fields=fields)

    def with_fields(self, fields: LatentFieldSet) -> "DiTLatentBatch":
        return replace(self, fields=fields)

    def with_observation(
        self, observed_mask: Tensor, *, sample_origin: Optional[Tensor] = None
    ) -> "DiTLatentBatch":
        observed = torch.as_tensor(observed_mask, device=self.token_mask.device, dtype=torch.bool)
        if observed.shape != self.token_mask.shape:
            raise ValueError("observed_mask must have shape [B,K]")
        if torch.any(observed & ~self.token_mask):
            raise ValueError("observed_mask cannot include invalid tokens")
        origin = self.sample_origin if sample_origin is None else sample_origin.to(self.sample_origin)
        return replace(self, observed_mask=observed, sample_origin=origin)

    def to(self, *args, **kwargs) -> "DiTLatentBatch":
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


class StateDetailLatentAdapter(nn.Module):
    """Pack frozen codec latents and project them into DiT scalar/vector widths."""

    def __init__(
        self,
        codec_width: int = 128,
        scalar_width: int = 256,
        vector_width: int = 128,
        *,
        ratio: int,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.codec_width = int(codec_width)
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.ratio = int(ratio)
        self.mode = mode or f"ratio{self.ratio}_state_detail"
        if self.ratio not in SUPPORTED_RATIOS or self.mode not in SUPPORTED_DIT_MODES:
            raise ValueError("the DiT adapter supports only ratio2_state_detail and ratio4_state_detail")
        if RATIO_FOR_MODE.get(self.mode) != self.ratio:
            raise ValueError("ratio and mode disagree")
        if min(self.codec_width, self.scalar_width, self.vector_width) < 1:
            raise ValueError("all adapter widths must be positive")
        packed_width = 2 * self.codec_width
        self.scalar_in = nn.Linear(packed_width, self.scalar_width)
        self.vector_in = AxisPreservingLinear(packed_width, self.vector_width)
        self.scalar_out = nn.ModuleDict({
            "state_h": nn.Linear(self.scalar_width, self.codec_width),
            "detail_h": nn.Linear(self.scalar_width, self.codec_width),
        })
        self.vector_out = nn.ModuleDict({
            "state_v": AxisPreservingLinear(self.vector_width, self.codec_width),
            "detail_v": AxisPreservingLinear(self.vector_width, self.codec_width),
        })

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": DIT_ADAPTER_SCHEMA,
            "codec_schema": STATE_DETAIL_CODEC_SCHEMA,
            "ratio": self.ratio,
            "mode": self.mode,
            "codec_width": self.codec_width,
            "scalar_width": self.scalar_width,
            "vector_width": self.vector_width,
            "input_packing": "concat_state_detail_channel",
            "state_detail_token_count": FRAME_COUNT // self.ratio,
            "vector_maps": "bias_free_channel_only",
            "contains_raw_detail": False,
            "contains_target_coordinates": False,
        }

    def validate_latent(self, latent: StateDetailLatent) -> None:
        if not isinstance(latent, StateDetailLatent):
            raise TypeError("expected a StateDetailLatent from the frozen state/detail codec")
        if latent.mode != self.mode or int(latent.ratio) != self.ratio:
            raise ValueError(f"latent mode/ratio must be {self.mode}/{self.ratio}")
        if latent.frames != FRAME_COUNT:
            raise ValueError(f"DiT v1 requires T={FRAME_COUNT}")
        k_expected = FRAME_COUNT // self.ratio
        if latent.state_h.shape[0] != k_expected or latent.state_v.shape[0] != k_expected:
            raise ValueError("latent K does not match T/ratio")
        if latent.detail_h is None or latent.detail_v is None:
            raise ValueError("R2/R4 generated latents require state and detail fields")
        if latent.state_h.shape[-1] != self.codec_width or latent.detail_h.shape[-1] != self.codec_width:
            raise ValueError("latent codec width does not match the adapter")

    def pack(
        self,
        latent: StateDetailLatent,
        *,
        codec_hash: str = "",
        data_hash: str = "",
        origin_from_latent: bool = False,
        loss_mask: Optional[Tensor] = None,
    ) -> DiTLatentBatch:
        self.validate_latent(latent)
        batch_size = int(latent.block_frame_mask.shape[0])
        origin = latent.sample_origin if origin_from_latent else latent.sample_origin.new_zeros((batch_size, 3))
        fields = LatentFieldSet(latent.state_h, latent.detail_h, latent.state_v, latent.detail_v)
        return DiTLatentBatch(
            fields=fields,
            token_mask=latent.token_mask,
            detail_valid=latent.detail_valid,
            detail_component_mask=latent.detail_component_mask,
            block_frame_mask=latent.block_frame_mask,
            block_time_ps=latent.block_time_ps,
            frame_time_ps=latent.frame_time_ps,
            abid=latent.abid,
            sample_origin=origin,
            topology=latent.topology,
            ratio=self.ratio,
            mode=self.mode,
            width=self.codec_width,
            loss_mask=loss_mask,
            codec_hash=str(codec_hash),
            data_hash=str(data_hash),
        ).zero_invalid()

    from_latent = pack

    def packed_inputs(self, batch: DiTLatentBatch) -> tuple[Tensor, Tensor]:
        if batch.ratio != self.ratio or batch.mode != self.mode:
            raise ValueError("batch ratio/mode does not match the adapter")
        return (
            self.scalar_in(torch.cat((batch.state_h, batch.detail_h), dim=-1)),
            self.vector_in(torch.cat((batch.state_v, batch.detail_v), dim=-1)),
        )

    project_inputs = packed_inputs

    def project_outputs(self, h: Tensor, v: Tensor) -> LatentFieldSet:
        if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("model features must have shapes [K,N,Dh] and [K,N,3,Dv]")
        return LatentFieldSet(
            self.scalar_out["state_h"](h),
            self.scalar_out["detail_h"](h),
            self.vector_out["state_v"](v),
            self.vector_out["detail_v"](v),
        )

    def make_generated_latent(self, batch: DiTLatentBatch, fields: LatentFieldSet) -> StateDetailLatent:
        """Reconstruct a decoder-ready latent with raw diagnostics absent."""

        if fields.state_h.shape[-1] != self.codec_width:
            raise ValueError("generated fields must be inverse-projected to codec width")
        return StateDetailLatent(
            state_h=fields.state_h,
            state_v=fields.state_v,
            detail_h=fields.detail_h,
            detail_v=fields.detail_v,
            raw_detail_h=None,
            raw_detail_v=None,
            token_mask=batch.token_mask,
            detail_valid=batch.detail_valid,
            detail_component_mask=batch.detail_component_mask,
            block_frame_mask=batch.block_frame_mask,
            frame_time_ps=batch.frame_time_ps,
            block_time_ps=batch.block_time_ps,
            abid=batch.abid,
            sample_origin=batch.sample_origin,
            topology=batch.topology,
            ratio=batch.ratio,
            mode=batch.mode,
            coefficient_order=("D01",) if batch.ratio == 2 else ("Dmid", "D01", "D23"),
            width=batch.width,
        )

    generated_latent = make_generated_latent


def _validate_frame_mask(frame_mask: Tensor, *, batch_size: int, frames: int) -> Tensor:
    mask = torch.as_tensor(frame_mask, dtype=torch.bool)
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, frames):
        raise ValueError(f"frame_mask must have shape [{batch_size},{frames}]")
    for sample in range(batch_size):
        valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if valid.numel() and not torch.equal(valid, torch.arange(valid.numel(), device=mask.device)):
            raise ValueError("frame_mask must contain a valid prefix for each sample")
    return mask


def frame_prefix_observation_mask(frame_mask: Tensor, history_frames: int) -> Tensor:
    """Return the observed prefix for H=0,4,8 without touching future data."""

    mask = torch.as_tensor(frame_mask, dtype=torch.bool)
    if mask.ndim != 2 or mask.shape[1] != FRAME_COUNT:
        raise ValueError(f"frame_mask must have shape [B,{FRAME_COUNT}]")
    history = int(history_frames)
    if history not in (0, 4, 8):
        raise ValueError("v1 supports only H=0, H=4, or H=8")
    for sample in range(mask.shape[0]):
        valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if valid.numel() and not torch.equal(
            valid, torch.arange(valid.numel(), device=mask.device)
        ):
            raise ValueError("frame_mask must contain a valid prefix for each sample")
    if torch.any(mask.sum(dim=1) < history):
        raise ValueError("history length exceeds the valid frame count of a sample")
    prefix = torch.arange(FRAME_COUNT, device=mask.device).view(1, -1) < history
    return mask & prefix


@dataclass(frozen=True)
class ObservationCondition:
    frame_observation_mask: Tensor
    latent_observation_mask: Tensor
    sample_origin: Tensor
    history_frames: int

    def contract(self) -> dict[str, Any]:
        return {
            "history_frames": int(self.history_frames),
            "frame_observation_shape": list(self.frame_observation_mask.shape),
            "latent_observation_shape": list(self.latent_observation_mask.shape),
            "origin_shape": list(self.sample_origin.shape),
            "origin_source": "observed_frame0_centroid" if self.history_frames else "fixed_zero",
        }


def _observed_origins(
    coordinates: Optional[Tensor],
    batch: DiTLatentBatch,
    frame_observation_mask: Tensor,
    history_frames: int,
    atom_mask: Optional[Tensor],
) -> Tensor:
    origin = batch.sample_origin.new_zeros((batch.batch_size, 3))
    if history_frames == 0:
        return origin
    if coordinates is None:
        raise ValueError("coordinates are required only to derive the observed frame-0 origin for H>0")
    x = torch.as_tensor(coordinates, device=batch.state_h.device, dtype=batch.state_h.dtype)
    if x.ndim != 3 or x.shape[0] != FRAME_COUNT or x.shape[1] != batch.num_atoms or x.shape[2] != 3:
        raise ValueError("clean observed coordinates must have shape [T,N,3]")
    if not bool(frame_observation_mask[:, 0].all()):
        raise ValueError("H>0 requires an observed frame-0 for every sample")
    if atom_mask is None:
        raise ValueError("H>0 observation origin requires the static codec loss_mask")
    return compute_masked_centroid_origin(
        x,
        frame_mask=frame_observation_mask,
        abid=batch.abid,
        atom_mask=atom_mask,
    )


def build_observation_condition(
    batch: DiTLatentBatch,
    *,
    history_frames: int,
    coordinates: Optional[Tensor] = None,
    frame_mask: Optional[Tensor] = None,
    observed_frame_mask: Optional[Tensor] = None,
    loss_mask: Optional[Tensor] = None,
) -> ObservationCondition:
    """Build block-aligned observation state without retaining target coordinates."""

    frames = int(batch.frame_time_ps.shape[1])
    source_mask = batch.block_frame_mask.new_ones((batch.batch_size, frames))
    if frame_mask is not None and observed_frame_mask is not None:
        raise ValueError("pass frame_mask or observed_frame_mask, not both")
    if frame_mask is not None:
        source_mask = _validate_frame_mask(
            torch.as_tensor(frame_mask, device=source_mask.device, dtype=torch.bool),
            batch_size=batch.batch_size,
            frames=frames,
        )
    else:
        source_mask.zero_()
        flat = batch.block_frame_mask.reshape(batch.batch_size, -1)
        source_mask[:, : min(frames, flat.shape[1])] = flat[:, :frames]
    if observed_frame_mask is not None:
        observed_frames = torch.as_tensor(
            observed_frame_mask, device=source_mask.device, dtype=torch.bool
        )
        if observed_frames.shape != source_mask.shape:
            raise ValueError("observed_frame_mask must have shape [B,T]")
        if torch.any(observed_frames & ~source_mask):
            raise ValueError("observed_frame_mask cannot include invalid source frames")
    else:
        observed_frames = frame_prefix_observation_mask(source_mask, history_frames)
    valid = batch.block_frame_mask
    observed_padded = observed_frames.new_zeros((batch.batch_size, batch.tokens * batch.ratio))
    observed_padded[:, :frames] = observed_frames
    observed_blocks = observed_padded.reshape(batch.batch_size, batch.tokens, batch.ratio)
    partial = (observed_blocks & valid).any(dim=-1) & ((~observed_blocks) & valid).any(dim=-1)
    if torch.any(partial):
        locations = torch.nonzero(partial, as_tuple=False).tolist()
        raise ValueError(
            "partial observed codec block is not supported; choose H=0,4,8 on a shared "
            f"R2/R4 boundary (locations={locations[:4]})"
        )
    latent_observed = (observed_blocks | ~valid).all(dim=-1) & valid.any(dim=-1)
    static_loss_mask = batch.loss_mask if loss_mask is None else loss_mask
    if static_loss_mask is not None:
        static_loss_mask = torch.as_tensor(
            static_loss_mask, device=batch.state_h.device, dtype=torch.bool
        ).flatten()
        if static_loss_mask.numel() != batch.num_atoms:
            raise ValueError("loss_mask must have shape [N]")
    origin = _observed_origins(
        coordinates,
        batch,
        observed_frames,
        int(history_frames),
        static_loss_mask,
    )
    return ObservationCondition(
        frame_observation_mask=observed_frames.detach().clone(),
        latent_observation_mask=latent_observed.detach().clone(),
        sample_origin=origin.detach().clone(),
        history_frames=int(history_frames),
    )


@dataclass(frozen=True)
class LatentStatistics:
    """Mask-aware, ratio-specific train-split statistics."""

    ratio: int
    mode: str
    width: int
    state_h_mean: Tensor
    state_h_std: Tensor
    detail_h_mean: Tensor
    detail_h_std: Tensor
    state_v_rms: Tensor
    detail_v_rms: Tensor
    provenance: Mapping[str, Any]
    schema_version: str = DIT_STATS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DIT_STATS_SCHEMA:
            raise ValueError(f"unsupported statistics schema {self.schema_version!r}")
        if self.ratio not in SUPPORTED_RATIOS or self.mode != f"ratio{self.ratio}_state_detail":
            raise ValueError("statistics must be ratio2_state_detail or ratio4_state_detail")
        values = (
            self.state_h_mean,
            self.state_h_std,
            self.detail_h_mean,
            self.detail_h_std,
            self.state_v_rms,
            self.detail_v_rms,
        )
        if any(tuple(value.shape) != (self.width,) for value in values):
            raise ValueError("all statistics must have shape [C]")
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("statistics contain NaN or Inf")
        if torch.any(self.state_h_std <= 1e-8) or torch.any(self.detail_h_std <= 1e-8):
            raise ValueError("scalar standard deviations must be positive")
        if torch.any(self.state_v_rms <= 1e-8) or torch.any(self.detail_v_rms <= 1e-8):
            raise ValueError("vector RMS scales must be positive")

    @classmethod
    def fit(
        cls,
        batches: Iterable[DiTLatentBatch],
        *,
        ratio: int,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "LatentStatistics":
        ratio = int(ratio)
        mode = f"ratio{ratio}_state_detail"
        if ratio not in SUPPORTED_RATIOS:
            raise ValueError("statistics support only ratios 2 and 4")
        sums: dict[str, Tensor] = {}
        squares: dict[str, Tensor] = {}
        counts: dict[str, float] = {}
        width: Optional[int] = None
        seen = 0
        for batch in batches:
            if batch.ratio != ratio or batch.mode != mode:
                raise ValueError("statistics batches must share one ratio and mode")
            width = batch.width if width is None else width
            if width != batch.width:
                raise ValueError("statistics batches must share codec width")
            masks = batch.field_masks()
            for name in ("state_h", "detail_h"):
                value = getattr(batch, name).float()
                mask = masks[name].to(value.dtype).unsqueeze(-1)
                sums.setdefault(name, torch.zeros(batch.width, device=value.device))
                squares.setdefault(name, torch.zeros(batch.width, device=value.device))
                sums[name] += (value * mask).sum(dim=(0, 1))
                squares[name] += (value.square() * mask).sum(dim=(0, 1))
                counts[name] = counts.get(name, 0.0) + float(mask.sum())
            for name in ("state_v", "detail_v"):
                value = getattr(batch, name).float()
                mask = masks[name].to(value.dtype).unsqueeze(-1).unsqueeze(-1)
                sums.setdefault(name, torch.zeros(batch.width, device=value.device))
                squares.setdefault(name, torch.zeros(batch.width, device=value.device))
                sums[name] += (value * mask).sum(dim=(0, 1, 2))
                squares[name] += (value.square() * mask).sum(dim=(0, 1, 2))
                counts[name] = counts.get(name, 0.0) + float(mask.sum() * 3.0)
            seen += 1
        if not seen or width is None:
            raise ValueError("at least one training batch is required for statistics")
        state_h_mean = sums["state_h"] / counts["state_h"]
        detail_h_mean = sums["detail_h"] / counts["detail_h"]
        state_h_var = squares["state_h"] / counts["state_h"] - state_h_mean.square()
        detail_h_var = squares["detail_h"] / counts["detail_h"] - detail_h_mean.square()
        return cls(
            ratio=ratio,
            mode=mode,
            width=width,
            state_h_mean=state_h_mean,
            state_h_std=state_h_var.clamp_min(0).sqrt(),
            detail_h_mean=detail_h_mean,
            detail_h_std=detail_h_var.clamp_min(0).sqrt(),
            state_v_rms=(squares["state_v"] / counts["state_v"]).clamp_min(0).sqrt(),
            detail_v_rms=(squares["detail_v"] / counts["detail_v"]).clamp_min(0).sqrt(),
            provenance=dict(provenance or {}),
        )

    def _broadcast(self, value: Tensor, target: Tensor) -> Tensor:
        shape = (1,) * (target.ndim - 1) + (self.width,)
        return value.to(device=target.device, dtype=target.dtype).reshape(shape)

    def normalize_fields(self, fields: LatentFieldSet) -> LatentFieldSet:
        return LatentFieldSet(
            (fields.state_h - self._broadcast(self.state_h_mean, fields.state_h))
            / self._broadcast(self.state_h_std, fields.state_h),
            (fields.detail_h - self._broadcast(self.detail_h_mean, fields.detail_h))
            / self._broadcast(self.detail_h_std, fields.detail_h),
            fields.state_v / self._broadcast(self.state_v_rms, fields.state_v),
            fields.detail_v / self._broadcast(self.detail_v_rms, fields.detail_v),
        )

    def inverse_fields(self, fields: LatentFieldSet) -> LatentFieldSet:
        return LatentFieldSet(
            fields.state_h * self._broadcast(self.state_h_std, fields.state_h)
            + self._broadcast(self.state_h_mean, fields.state_h),
            fields.detail_h * self._broadcast(self.detail_h_std, fields.detail_h)
            + self._broadcast(self.detail_h_mean, fields.detail_h),
            fields.state_v * self._broadcast(self.state_v_rms, fields.state_v),
            fields.detail_v * self._broadcast(self.detail_v_rms, fields.detail_v),
        )

    def _check_batch(self, batch: DiTLatentBatch) -> None:
        if batch.ratio != self.ratio or batch.mode != self.mode or batch.width != self.width:
            raise ValueError("latent statistics contract does not match the batch")

    def normalize(self, batch: DiTLatentBatch) -> DiTLatentBatch:
        self._check_batch(batch)
        return batch.with_fields(self.normalize_fields(batch.fields)).zero_invalid()

    def inverse_normalize(
        self, value: DiTLatentBatch | LatentFieldSet
    ) -> DiTLatentBatch | LatentFieldSet:
        if isinstance(value, DiTLatentBatch):
            self._check_batch(value)
            return value.with_fields(self.inverse_fields(value.fields)).zero_invalid()
        return self.inverse_fields(value)

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ratio": self.ratio,
            "mode": self.mode,
            "width": self.width,
            "state_h_mean_hash": tensor_hash(self.state_h_mean),
            "state_h_std_hash": tensor_hash(self.state_h_std),
            "detail_h_mean_hash": tensor_hash(self.detail_h_mean),
            "detail_h_std_hash": tensor_hash(self.detail_h_std),
            "state_v_rms_hash": tensor_hash(self.state_v_rms),
            "detail_v_rms_hash": tensor_hash(self.detail_v_rms),
            "provenance": dict(self.provenance),
            "vector_mean_subtraction": False,
            "mask_policy": "state=token_mask, detail=token_mask_and_detail_valid",
        }

    @property
    def hash(self) -> str:
        return contract_hash(self.contract())

    def state_dict(self) -> dict[str, Any]:
        """Return a self-contained, CPU-portable statistics artifact."""

        return {
            "schema_version": self.schema_version,
            "ratio": int(self.ratio),
            "mode": self.mode,
            "width": int(self.width),
            "state_h_mean": self.state_h_mean.detach().to(device="cpu").clone(),
            "state_h_std": self.state_h_std.detach().to(device="cpu").clone(),
            "detail_h_mean": self.detail_h_mean.detach().to(device="cpu").clone(),
            "detail_h_std": self.detail_h_std.detach().to(device="cpu").clone(),
            "state_v_rms": self.state_v_rms.detach().to(device="cpu").clone(),
            "detail_v_rms": self.detail_v_rms.detach().to(device="cpu").clone(),
            "provenance": dict(self.provenance),
            "statistics_hash": self.hash,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LatentStatistics":
        required = (
            "ratio",
            "mode",
            "width",
            "state_h_mean",
            "state_h_std",
            "detail_h_mean",
            "detail_h_std",
            "state_v_rms",
            "detail_v_rms",
            "provenance",
        )
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(f"statistics state is missing fields: {missing}")
        value = cls(
            ratio=int(state["ratio"]),
            mode=str(state["mode"]),
            width=int(state["width"]),
            state_h_mean=torch.as_tensor(state["state_h_mean"]).detach().clone(),
            state_h_std=torch.as_tensor(state["state_h_std"]).detach().clone(),
            detail_h_mean=torch.as_tensor(state["detail_h_mean"]).detach().clone(),
            detail_h_std=torch.as_tensor(state["detail_h_std"]).detach().clone(),
            state_v_rms=torch.as_tensor(state["state_v_rms"]).detach().clone(),
            detail_v_rms=torch.as_tensor(state["detail_v_rms"]).detach().clone(),
            provenance=dict(state["provenance"]),
            schema_version=str(state.get("schema_version", DIT_STATS_SCHEMA)),
        )
        stored_hash = state.get("statistics_hash")
        if stored_hash is not None and str(stored_hash) != value.hash:
            raise ValueError("statistics artifact hash does not match its tensors and provenance")
        return value

    def to(self, *args, **kwargs) -> "LatentStatistics":
        return replace(
            self,
            state_h_mean=self.state_h_mean.to(*args, **kwargs),
            state_h_std=self.state_h_std.to(*args, **kwargs),
            detail_h_mean=self.detail_h_mean.to(*args, **kwargs),
            detail_h_std=self.detail_h_std.to(*args, **kwargs),
            state_v_rms=self.state_v_rms.to(*args, **kwargs),
            detail_v_rms=self.detail_v_rms.to(*args, **kwargs),
        )


__all__ = [
    "AxisPreservingLinear",
    "DIT_ADAPTER_SCHEMA",
    "DIT_BATCH_SCHEMA",
    "DIT_MODEL_SCHEMA",
    "DIT_STATS_SCHEMA",
    "DiTLatentBatch",
    "LatentFieldSet",
    "LatentStatistics",
    "ObservationCondition",
    "StateDetailLatentAdapter",
    "build_observation_condition",
    "canonical_json",
    "contract_hash",
    "frame_prefix_observation_mask",
    "tensor_hash",
]
