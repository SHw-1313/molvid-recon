"""Mask-aware train-split latent statistics with checked artifact hashes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor

from .types import (
    DIT_STATS_SCHEMA,
    SUPPORTED_RATIOS,
    FrameLatentBatch,
    LatentBatch,
    LatentFields,
    contract_hash,
    tensor_hash,
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
        batches: Iterable[LatentBatch],
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

    def normalize_fields(self, fields: LatentFields) -> LatentFields:
        return LatentFields(
            (fields.state_h - self._broadcast(self.state_h_mean, fields.state_h))
            / self._broadcast(self.state_h_std, fields.state_h),
            (fields.detail_h - self._broadcast(self.detail_h_mean, fields.detail_h))
            / self._broadcast(self.detail_h_std, fields.detail_h),
            fields.state_v / self._broadcast(self.state_v_rms, fields.state_v),
            fields.detail_v / self._broadcast(self.detail_v_rms, fields.detail_v),
        )

    def inverse_fields(self, fields: LatentFields) -> LatentFields:
        return LatentFields(
            fields.state_h * self._broadcast(self.state_h_std, fields.state_h)
            + self._broadcast(self.state_h_mean, fields.state_h),
            fields.detail_h * self._broadcast(self.detail_h_std, fields.detail_h)
            + self._broadcast(self.detail_h_mean, fields.detail_h),
            fields.state_v * self._broadcast(self.state_v_rms, fields.state_v),
            fields.detail_v * self._broadcast(self.detail_v_rms, fields.detail_v),
        )

    def _check_batch(self, batch: LatentBatch) -> None:
        if batch.ratio != self.ratio or batch.mode != self.mode or batch.width != self.width:
            raise ValueError("latent statistics contract does not match the batch")

    def normalize(self, batch: LatentBatch) -> LatentBatch:
        self._check_batch(batch)
        return batch.with_fields(self.normalize_fields(batch.fields)).zero_invalid()

    def inverse_normalize(
        self, value: LatentBatch | LatentFields
    ) -> LatentBatch | LatentFields:
        if isinstance(value, LatentBatch):
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


@dataclass(frozen=True)
class FrameLatentStatistics:
    """Train-split statistics for per-frame h/v; incompatible with R2/R4 stats."""

    h_mean: Tensor
    h_std: Tensor
    v_rms: Tensor
    provenance: Mapping[str, Any]
    schema_version: str = "molvid.frame_joint.stats.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "molvid.frame_joint.stats.v1":
            raise ValueError("unsupported frame latent statistics schema")
        if self.h_mean.ndim != 1 or self.h_std.shape != self.h_mean.shape or self.v_rms.shape != self.h_mean.shape:
            raise ValueError("frame latent statistics must have shape [C]")
        if any(not torch.isfinite(value).all() for value in (self.h_mean, self.h_std, self.v_rms)):
            raise ValueError("frame latent statistics contain NaN or Inf")
        if torch.any(self.h_std <= 0) or torch.any(self.v_rms <= 0):
            raise ValueError("frame latent statistics scales must be positive")

    @property
    def width(self) -> int:
        return int(self.h_mean.numel())

    @classmethod
    def fit(
        cls,
        batches: Iterable[FrameLatentBatch],
        *,
        provenance: Mapping[str, Any],
        epsilon: float = 1.0e-6,
    ) -> "FrameLatentStatistics":
        h_sum = h_square = v_square = None
        scalar_count = vector_count = 0.0
        seen = 0
        for batch in batches:
            mask = batch.atom_frame_mask()
            h = batch.h.float()
            v = batch.v.float()
            h_mask = mask.to(dtype=h.dtype).unsqueeze(-1)
            v_mask = h_mask.unsqueeze(-1)
            if h_sum is None:
                h_sum = torch.zeros(batch.width, device=h.device)
                h_square = torch.zeros_like(h_sum)
                v_square = torch.zeros_like(h_sum)
            if batch.width != int(h_sum.numel()):
                raise ValueError("statistics batches have different frame widths")
            h_sum += (h * h_mask).sum(dim=(0, 1))
            h_square += (h.square() * h_mask).sum(dim=(0, 1))
            v_square += (v.square() * v_mask).sum(dim=(0, 1, 2))
            scalar_count += float(mask.sum())
            vector_count += float(mask.sum()) * 3.0
            seen += 1
        if not seen or h_sum is None or scalar_count <= 0 or vector_count <= 0:
            raise ValueError("at least one valid training frame batch is required")
        mean = h_sum / scalar_count
        variance = (h_square / scalar_count - mean.square()).clamp_min(0.0)
        return cls(
            h_mean=mean,
            h_std=variance.sqrt().clamp_min(float(epsilon)),
            v_rms=(v_square / vector_count).sqrt().clamp_min(float(epsilon)),
            provenance=dict(provenance),
        )

    def _broadcast(self, value: Tensor, target: Tensor) -> Tensor:
        return value.to(device=target.device, dtype=target.dtype).reshape((1,) * (target.ndim - 1) + (self.width,))

    def normalize(self, batch: FrameLatentBatch) -> FrameLatentBatch:
        if batch.width != self.width:
            raise ValueError("frame statistics width differs from latent")
        h = (batch.h - self._broadcast(self.h_mean, batch.h)) / self._broadcast(self.h_std, batch.h)
        v = batch.v / self._broadcast(self.v_rms, batch.v)
        return batch.with_features(h, v, statistics_hash=self.hash)

    def inverse(self, batch: FrameLatentBatch) -> FrameLatentBatch:
        if batch.width != self.width:
            raise ValueError("frame statistics width differs from latent")
        h = batch.h * self._broadcast(self.h_std, batch.h) + self._broadcast(self.h_mean, batch.h)
        v = batch.v * self._broadcast(self.v_rms, batch.v)
        return batch.with_features(h, v, statistics_hash=self.hash)

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "width": self.width,
            "h_mean_hash": tensor_hash(self.h_mean),
            "h_std_hash": tensor_hash(self.h_std),
            "v_rms_hash": tensor_hash(self.v_rms),
            "vector_mean_subtraction": False,
            "vector_xyz_shared_scale": True,
            "fit_scope": "training_split_global",
            "provenance": dict(self.provenance),
        }

    @property
    def hash(self) -> str:
        return contract_hash(self.contract())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "h_mean": self.h_mean.detach().cpu(),
            "h_std": self.h_std.detach().cpu(),
            "v_rms": self.v_rms.detach().cpu(),
            "provenance": dict(self.provenance),
            "statistics_hash": self.hash,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FrameLatentStatistics":
        value = cls(
            h_mean=torch.as_tensor(state["h_mean"]).detach().clone(),
            h_std=torch.as_tensor(state["h_std"]).detach().clone(),
            v_rms=torch.as_tensor(state["v_rms"]).detach().clone(),
            provenance=dict(state["provenance"]),
            schema_version=str(state.get("schema_version", "")),
        )
        if state.get("statistics_hash") not in (None, value.hash):
            raise ValueError("frame statistics hash differs from its tensors/provenance")
        return value

    def to(self, *args, **kwargs) -> "FrameLatentStatistics":
        return replace(
            self,
            h_mean=self.h_mean.to(*args, **kwargs),
            h_std=self.h_std.to(*args, **kwargs),
            v_rms=self.v_rms.to(*args, **kwargs),
        )
