"""Prepare CPU topology before moving codec or DiT batches to the device."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Mapping, Protocol

import torch
from torch import Tensor, nn

from ..data.batch import ClipBatch

from ..flow.source import build_observed_center
from ..flow.source import repeat_last_frame_center
from ..latent.adapter import StateDetailLatentAdapter
from ..latent.conditioning import HistoryConditionedLatents, combine_clean_target_with_condition, make_history_corruption_views
from ..latent.statistics import LatentStatistics
from ..latent.types import (
    FrameLatentBatch,
    LatentBatch,
    LatentFields,
    ObservedContext,
    QuerySpec,
)
from ..codec.frame import FrozenFrameTeacher, slice_clip_frames


def _to_device(value: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    if isinstance(value, ClipBatch):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=non_blocking)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _to_device(
                    getattr(value, field.name),
                    device,
                    non_blocking=non_blocking,
                )
                for field in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {
            key: _to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _to_device(item, device, non_blocking=non_blocking) for item in value
        )
    if isinstance(value, list):
        return [
            _to_device(item, device, non_blocking=non_blocking) for item in value
        ]
    return value


def prepare_batch_then_to_device(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    """Run the CPU-only registration phase before any CUDA transfer."""

    prepare = getattr(model, "prepare_batch", None)
    if callable(prepare):
        prepare(batch)
    return _to_device(batch, device, non_blocking=non_blocking)


@dataclass(frozen=True)
class PreparedDiTBatch:
    coordinate_batch: ClipBatch
    target_view: ClipBatch
    observed: LatentBatch
    source_center: LatentFields | None
    corruption: HistoryConditionedLatents


@dataclass(frozen=True)
class PreparedFrameJointBatch:
    """Separated observed condition, query clock, and frozen future targets."""

    coordinate_batch: ClipBatch
    observed_context: ObservedContext
    query: QuerySpec
    target_future: FrameLatentBatch
    normalized_target: FrameLatentBatch
    source_center: FrameLatentBatch
    target_coordinates: Tensor


class FrameLatentNormalizer(Protocol):
    """Minimal normalization boundary used during Frame Joint preparation."""

    def normalize(self, latent: FrameLatentBatch) -> FrameLatentBatch: ...


@torch.no_grad()
def prepare_frame_joint_batch(
    teacher: FrozenFrameTeacher,
    batch: ClipBatch,
    *,
    device: torch.device,
    normalizer: FrameLatentNormalizer,
    history_frames: int,
) -> PreparedFrameJointBatch:
    """Encode a clip once while exposing only observed tensors to the model.

    The teacher call is intentionally the only no-grad region.  The trainable
    history encoder is invoked later by ``FrameJointModel.forward``.
    """

    history = int(history_frames)
    if history < 1 or history >= batch.frames:
        raise ValueError("history_frames must leave at least one future frame")
    coordinate = prepare_batch_then_to_device(teacher, batch, device)
    full_latent, origin = teacher(coordinate)
    observed_latent = full_latent.slice_frames(0, history)
    target_future = full_latent.slice_frames(history, full_latent.frames)
    observed = ObservedContext(
        latent=observed_latent,
        coordinates=coordinate.x[:history],
        sample_origin=origin,
        loss_mask=coordinate.loss_mask.to(dtype=torch.bool),
    )
    query = QuerySpec(
        time_ps=target_future.time_ps,
        frame_mask=target_future.frame_mask,
    )
    normalized_target = normalizer.normalize(target_future)
    normalized_observed = normalizer.normalize(observed_latent)
    source_center = repeat_last_frame_center(normalized_observed, query)
    return PreparedFrameJointBatch(
        coordinate_batch=coordinate,
        observed_context=observed,
        query=query,
        target_future=target_future,
        normalized_target=normalized_target,
        source_center=source_center,
        target_coordinates=coordinate.x[history:],
    )


@torch.no_grad()
def encode_batch(
    codec: nn.Module,
    adapter: StateDetailLatentAdapter,
    batch: ClipBatch,
    *,
    device: torch.device,
    codec_hash: str,
    data_hash: str,
) -> tuple[LatentBatch, ClipBatch]:
    """Register static topology on CPU, then encode a clean clip on device."""

    coordinate = prepare_batch_then_to_device(codec, batch, device)
    latent = codec.encode(coordinate)
    packed = adapter.from_codec_latent(
        latent,
        codec_hash=codec_hash,
        data_hash=data_hash,
        origin_from_latent=True,
        loss_mask=coordinate.loss_mask,
    )
    return packed, coordinate


@torch.no_grad()
def prepare_dit_batch(
    codec: nn.Module,
    adapter: StateDetailLatentAdapter,
    batch: ClipBatch,
    *,
    device: torch.device,
    statistics: LatentStatistics,
    history_frames: int,
    sigmas_angstrom: Tensor,
    epsilon: Tensor,
    source_mode: str,
    center_kind: str,
    codec_hash: str,
    data_hash: str,
) -> PreparedDiTBatch:
    """Keep clean target, noisy observed condition and source center separate."""

    clean_target, coordinate = encode_batch(
        codec, adapter, batch, device=device, codec_hash=codec_hash, data_hash=data_hash
    )
    views = make_history_corruption_views(
        coordinate,
        history_frames=history_frames,
        sigma_per_sample_angstrom=sigmas_angstrom,
        epsilon=epsilon,
    )
    if bool(torch.all(sigmas_angstrom == 0)):
        condition_latent = clean_target
    else:
        condition_latent = adapter.from_codec_latent(
            codec.encode(views.condition_view),
            codec_hash=codec_hash,
            data_hash=data_hash,
            origin_from_latent=True,
            loss_mask=views.condition_view.loss_mask,
        )
    corruption = combine_clean_target_with_condition(
        clean_target, condition_latent, views=views, history_frames=history_frames
    )
    if source_mode == "gaussian":
        center = None
    elif source_mode == "conditional":
        if history_frames == 0:
            raise ValueError("conditional source requires observed history")
        center, _ = build_observed_center(
            center_kind,
            codec_model=codec,
            coordinate_batch=views.condition_view,
            target_batch=corruption.observed,
            adapter=adapter,
            statistics=statistics,
            history_frames=history_frames,
            codec_hash=codec_hash,
            data_hash=data_hash,
        )
    else:
        raise ValueError(f"unsupported source_mode {source_mode!r}")
    return PreparedDiTBatch(
        coordinate_batch=coordinate,
        target_view=views.target_view,
        observed=corruption.observed,
        source_center=center,
        corruption=corruption,
    )
