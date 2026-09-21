"""Observed-only Gaussian and conditional source construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import torch
from torch import Tensor

from ..latent.adapter import StateDetailLatentAdapter
from ..latent.statistics import LatentStatistics
from ..latent.types import FrameLatentBatch, LatentBatch, LatentFields, QuerySpec

SOURCE_SCHEMA = "pvb.dit.state_detail.source.v1"
SOURCE_MODES = ("gaussian", "conditional")
CENTER_KINDS = ("repeat_last_coordinate_encode", "block_state_zero_detail")
FIELD_NAMES = ("state_h", "detail_h", "state_v", "detail_v")


def sample_isotropic_noise(
    fields: LatentFields, *, generator: Optional[torch.Generator] = None
) -> LatentFields:
    def draw(value: Tensor) -> Tensor:
        return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
    return fields.map(draw)


def _mask_fields(fields: LatentFields, batch: LatentBatch) -> LatentFields:
    masks = batch.field_masks()
    return LatentFields(
        fields.state_h * masks["state_h"].to(fields.state_h.dtype).unsqueeze(-1),
        fields.detail_h * masks["detail_h"].to(fields.detail_h.dtype).unsqueeze(-1),
        fields.state_v * masks["state_v"].to(fields.state_v.dtype).unsqueeze(-1).unsqueeze(-1),
        fields.detail_v * masks["detail_v"].to(fields.detail_v.dtype).unsqueeze(-1).unsqueeze(-1),
    )


def _expand_mask(mask: Tensor, value: Tensor) -> Tensor:
    return mask.reshape(mask.shape + (1,) * (value.ndim - 2))


def future_field_masks(batch: LatentBatch) -> dict[str, Tensor]:
    if batch.observed_mask is None:
        raise ValueError("source construction requires an observation mask")
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    return {
        name: value & ~observed for name, value in batch.field_masks().items()
    }


def future_only_fields(fields: LatentFields, batch: LatentBatch) -> LatentFields:
    masks = future_field_masks(batch)
    return LatentFields(
        fields.state_h * _expand_mask(masks["state_h"], fields.state_h).to(fields.state_h.dtype),
        fields.detail_h * _expand_mask(masks["detail_h"], fields.detail_h).to(fields.detail_h.dtype),
        fields.state_v * _expand_mask(masks["state_v"], fields.state_v).to(fields.state_v.dtype),
        fields.detail_v * _expand_mask(masks["detail_v"], fields.detail_v).to(fields.detail_v.dtype),
    )


def sample_source(
    center: LatentFields | None,
    noise: LatentFields,
    *,
    source_mode: str,
) -> LatentFields:
    """Return eps for Gaussian or m+eps for conditional source."""

    if source_mode == "gaussian":
        if center is not None:
            raise ValueError("Gaussian source must not receive a conditional center")
        return noise
    if source_mode != "conditional":
        raise ValueError(f"unsupported source_mode {source_mode!r}")
    if center is None:
        raise ValueError("conditional source requires an observed-only center")
    return LatentFields(
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
    center_batch: LatentBatch,
    target_batch: LatentBatch,
) -> LatentFields:
    if center_batch.observed_mask is None:
        center_batch = center_batch.with_observation(target_batch.observed_mask)
    return future_only_fields(center_batch.fields, target_batch)


@torch.no_grad()
def repeat_last_coordinate_center(
    codec_model: Any,
    coordinate_batch: Any,
    target_batch: LatentBatch,
    *,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    history_frames: int,
    codec_hash: str = "",
    data_hash: str = "",
) -> tuple[LatentFields, Any, Tensor]:
    """Encode an observed-only repeated-coordinate template on the same GPU."""

    template = repeat_last_coordinate_template(coordinate_batch, history_frames)
    template_batch = replace(coordinate_batch, x=template)
    template_latent = codec_model.encode(template_batch)
    center_batch = adapter.from_codec_latent(
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
    target_batch: LatentBatch,
    history_frames: int,
) -> LatentFields:
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
    return LatentFields(
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
    target_batch: LatentBatch,
    *,
    statistics: LatentStatistics,
    history_frames: int,
) -> LatentFields:
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
    target_batch: LatentBatch,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    history_frames: int,
    codec_hash: str = "",
    data_hash: str = "",
) -> tuple[LatentFields, dict[str, Any]]:
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
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"unsupported source_mode {source_mode!r}")
    if center_kind not in CENTER_KINDS:
        raise ValueError(f"unsupported center_kind {center_kind!r}")
    if float(sigma) != 1.0:
        raise ValueError("source sigma is frozen at one per coefficient")
    if not str(statistics.hash):
        raise ValueError("source contract requires a normalization hash")
    return {
        "schema": SOURCE_SCHEMA,
        "source_mode": source_mode,
        "center_kind": center_kind,
        "sigma": float(sigma),
        "normalization_hash": statistics.hash,
        "history": "observed_prefix_only_H4_H8",
        "future_source": "eps" if source_mode == "gaussian" else "m_plus_eps",
        "noise_std_per_coefficient": 1.0,
    }


def repeat_last_frame_center(observed: FrameLatentBatch, query: QuerySpec) -> FrameLatentBatch:
    """Repeat only the last valid observed h/v into every future query frame."""

    if observed.batch_size != query.batch_size:
        raise ValueError("observed and query batch sizes differ")
    last = observed.frame_mask.sum(dim=1).long() - 1
    if torch.any(last < 0):
        raise ValueError("source center requires at least one observed frame per sample")
    atom_last = last.index_select(0, observed.abid)
    atom = torch.arange(observed.num_atoms, device=observed.h.device)
    last_h = observed.h[atom_last, atom]
    last_v = observed.v[atom_last, atom]
    h = last_h.unsqueeze(0).expand(query.frames, -1, -1).clone()
    v = last_v.unsqueeze(0).expand(query.frames, -1, -1, -1).clone()
    atom_mask = query.frame_mask.index_select(0, observed.abid).transpose(0, 1)
    return FrameLatentBatch(
        h=h * atom_mask.unsqueeze(-1),
        v=v * atom_mask.unsqueeze(-1).unsqueeze(-1),
        time_ps=query.time_ps,
        frame_mask=query.frame_mask,
        topology=observed.topology,
        statistics_hash=observed.statistics_hash,
    )


def sample_frame_source(
    center: FrameLatentBatch,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[FrameLatentBatch, FrameLatentBatch]:
    """Return ``repeat(last_observed)+N(0,1)`` and the sampled noise."""

    noise_h = torch.randn(center.h.shape, device=center.h.device, dtype=center.h.dtype, generator=generator)
    noise_v = torch.randn(center.v.shape, device=center.v.device, dtype=center.v.dtype, generator=generator)
    atom_mask = center.atom_frame_mask()
    noise = center.with_features(
        noise_h * atom_mask.unsqueeze(-1),
        noise_v * atom_mask.unsqueeze(-1).unsqueeze(-1),
    )
    source = center.with_features(center.h + noise.h, center.v + noise.v)
    return source, noise
