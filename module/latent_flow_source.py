"""Observed-only source construction for the state/detail rectified flow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor

from .state_detail_latent_adapter import (
    DiTLatentBatch,
    LatentFieldSet,
    LatentStatistics,
    StateDetailLatentAdapter,
)


SOURCE_SCHEMA = "pvb.dit.state_detail.source.v1"
SOURCE_MODES = ("gaussian", "conditional")
CENTER_KINDS = ("repeat_last_coordinate_encode", "block_state_zero_detail")


@dataclass(frozen=True)
class SourceContract:
    """Checkpoint-resumable description of a source arm."""

    source_mode: str
    center_kind: str
    sigma: float
    normalization_hash: str
    schema: str = SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.source_mode not in SOURCE_MODES:
            raise ValueError(f"unsupported source_mode {self.source_mode!r}")
        if self.center_kind not in CENTER_KINDS:
            raise ValueError(f"unsupported center_kind {self.center_kind!r}")
        if float(self.sigma) != 1.0:
            raise ValueError("source sigma is frozen at one per coefficient")
        if not str(self.normalization_hash):
            raise ValueError("source contract requires a normalization hash")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_mode": self.source_mode,
            "center_kind": self.center_kind,
            "sigma": float(self.sigma),
            "normalization_hash": self.normalization_hash,
            "history": "observed_prefix_only_H4_H8",
            "future_source": "eps" if self.source_mode == "gaussian" else "m_plus_eps",
            "noise_std_per_coefficient": 1.0,
        }


def _expand_mask(mask: Tensor, value: Tensor) -> Tensor:
    return mask.reshape(mask.shape + (1,) * (value.ndim - 2))


def future_field_masks(batch: DiTLatentBatch) -> dict[str, Tensor]:
    if batch.observed_mask is None:
        raise ValueError("source construction requires an observation mask")
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    return {
        name: value & ~observed for name, value in batch.field_masks().items()
    }


def future_only_fields(fields: LatentFieldSet, batch: DiTLatentBatch) -> LatentFieldSet:
    masks = future_field_masks(batch)
    return LatentFieldSet(
        fields.state_h * _expand_mask(masks["state_h"], fields.state_h).to(fields.state_h.dtype),
        fields.detail_h * _expand_mask(masks["detail_h"], fields.detail_h).to(fields.detail_h.dtype),
        fields.state_v * _expand_mask(masks["state_v"], fields.state_v).to(fields.state_v.dtype),
        fields.detail_v * _expand_mask(masks["detail_v"], fields.detail_v).to(fields.detail_v.dtype),
    )


def combine_source_fields(
    center: LatentFieldSet | None,
    noise: LatentFieldSet,
    *,
    source_mode: str,
) -> LatentFieldSet:
    """Return eps for Gaussian or m+eps for conditional source."""

    if source_mode == "gaussian":
        if center is not None:
            raise ValueError("Gaussian source must not receive a conditional center")
        return noise
    if source_mode != "conditional":
        raise ValueError(f"unsupported source_mode {source_mode!r}")
    if center is None:
        raise ValueError("conditional source requires an observed-only center")
    return LatentFieldSet(
        center.state_h + noise.state_h,
        center.detail_h + noise.detail_h,
        center.state_v + noise.state_v,
        center.detail_v + noise.detail_v,
    )


def repeat_last_coordinate_template(batch: Any, history_frames: int) -> Tensor:
    """Construct a template using only x[0:H] and the last observed frame."""

    history = int(history_frames)
    if history not in (4, 8):
        raise ValueError("source centers support only H=4 and H=8")
    if int(batch.frames) != 16:
        raise ValueError("source centers require the frozen 16-frame tokenizer")
    template = batch.x.clone()
    for sample in range(int(batch.batch_size)):
        start = int(batch.atom_ptr[sample].item())
        stop = int(batch.atom_ptr[sample + 1].item())
        if stop <= start:
            raise ValueError("source center cannot contain an empty sample")
        template[history:, start:stop] = batch.x[history - 1, start:stop].unsqueeze(0)
    return template


def _with_future_mask(
    center_batch: DiTLatentBatch,
    target_batch: DiTLatentBatch,
) -> LatentFieldSet:
    if center_batch.observed_mask is None:
        center_batch = center_batch.with_observation(target_batch.observed_mask)
    return future_only_fields(center_batch.fields, target_batch)


@torch.no_grad()
def repeat_last_coordinate_center(
    codec_model: Any,
    coordinate_batch: Any,
    target_batch: DiTLatentBatch,
    *,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    history_frames: int,
    codec_hash: str = "",
    data_hash: str = "",
) -> tuple[LatentFieldSet, Any, Tensor]:
    """Encode an observed-only repeated-coordinate template on the same GPU."""

    template = repeat_last_coordinate_template(coordinate_batch, history_frames)
    template_batch = replace(coordinate_batch, x=template)
    template_latent = codec_model.encode(template_batch)
    center_batch = adapter.pack(
        template_latent,
        codec_hash=codec_hash,
        data_hash=data_hash,
        origin_from_latent=True,
        loss_mask=coordinate_batch.loss_mask,
    )
    center_batch = center_batch.with_observation(
        target_batch.observed_mask,
        sample_origin=center_batch.sample_origin,
    )
    normalized = statistics.normalize(center_batch)
    return _with_future_mask(normalized, target_batch), template_latent, template


def _copy_last_observed_state(
    target_batch: DiTLatentBatch,
    history_frames: int,
) -> LatentFieldSet:
    ratio = int(target_batch.ratio)
    token = int(history_frames) // ratio - 1
    if token < 0 or token >= target_batch.tokens:
        raise ValueError("history does not select a complete observed token")
    if target_batch.observed_mask is None:
        raise ValueError("block-state center requires an observation mask")
    observed = target_batch.observed_mask.index_select(0, target_batch.abid).transpose(0, 1)
    future = {
        name: value & ~observed for name, value in target_batch.field_masks().items()
    }
    return LatentFieldSet(
        torch.where(
            _expand_mask(future["state_h"], target_batch.state_h),
            target_batch.state_h[token : token + 1].expand_as(target_batch.state_h),
            target_batch.state_h,
        ),
        torch.where(
            _expand_mask(future["detail_h"], target_batch.detail_h),
            torch.zeros_like(target_batch.detail_h),
            target_batch.detail_h,
        ),
        torch.where(
            _expand_mask(future["state_v"], target_batch.state_v),
            target_batch.state_v[token : token + 1].expand_as(target_batch.state_v),
            target_batch.state_v,
        ),
        torch.where(
            _expand_mask(future["detail_v"], target_batch.detail_v),
            torch.zeros_like(target_batch.detail_v),
            target_batch.detail_v,
        ),
    )


@torch.no_grad()
def block_state_zero_detail_center(
    target_batch: DiTLatentBatch,
    *,
    statistics: LatentStatistics,
    history_frames: int,
) -> LatentFieldSet:
    """Use the last observed raw state and raw-zero future detail."""

    raw_fields = _copy_last_observed_state(target_batch, history_frames)
    raw_batch = target_batch.with_fields(raw_fields)
    normalized = statistics.normalize(raw_batch)
    return future_only_fields(normalized.fields, target_batch)


@torch.no_grad()
def build_observed_center(
    center_kind: str,
    *,
    codec_model: Any,
    coordinate_batch: Any,
    target_batch: DiTLatentBatch,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    history_frames: int,
    codec_hash: str = "",
    data_hash: str = "",
) -> tuple[LatentFieldSet, dict[str, Any]]:
    """Build one center and return small provenance metadata."""

    if center_kind == "repeat_last_coordinate_encode":
        center, template_latent, template = repeat_last_coordinate_center(
            codec_model,
            coordinate_batch,
            target_batch,
            adapter=adapter,
            statistics=statistics,
            history_frames=history_frames,
            codec_hash=codec_hash,
            data_hash=data_hash,
        )
        return center, {
            "center_kind": center_kind,
            "history_frames": int(history_frames),
            "construction": "observed_prefix_plus_repeated_last_observed_coordinate",
            "template_latent": template_latent,
            "template_coordinates": template,
            "uses_future_coordinates": False,
        }
    if center_kind == "block_state_zero_detail":
        center = block_state_zero_detail_center(
            target_batch,
            statistics=statistics,
            history_frames=history_frames,
        )
        return center, {
            "center_kind": center_kind,
            "history_frames": int(history_frames),
            "construction": "last_complete_observed_state_plus_raw_zero_detail",
            "uses_future_coordinates": False,
        }
    raise ValueError(f"unsupported center_kind {center_kind!r}")


def source_contract(
    *,
    source_mode: str,
    center_kind: str,
    statistics: LatentStatistics,
    sigma: float = 1.0,
) -> dict[str, Any]:
    return SourceContract(
        source_mode=source_mode,
        center_kind=center_kind,
        sigma=float(sigma),
        normalization_hash=statistics.hash,
    ).as_dict()


__all__ = [
    "CENTER_KINDS",
    "SOURCE_MODES",
    "SOURCE_SCHEMA",
    "SourceContract",
    "block_state_zero_detail_center",
    "build_observed_center",
    "combine_source_fields",
    "future_field_masks",
    "future_only_fields",
    "repeat_last_coordinate_center",
    "repeat_last_coordinate_template",
    "source_contract",
]
