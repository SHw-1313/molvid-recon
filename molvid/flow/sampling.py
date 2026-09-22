"""Observed-clamped Euler sampling and decoder-ready latent generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor, nn

from ..latent.adapter import StateDetailLatentAdapter
from ..latent.statistics import LatentStatistics
from ..latent.types import FrameLatentBatch, LatentBatch, LatentFields, ObservedContext, QuerySpec, contract_hash
from .source import repeat_last_frame_center, sample_frame_source
from .source import FIELD_NAMES, _mask_fields, sample_isotropic_noise, sample_source

if TYPE_CHECKING:
    from ..model import FrameJointModel


def apply_observation_clamp(
    fields: LatentFields, clean_fields: LatentFields, batch: LatentBatch
) -> LatentFields:
    """Restore observed fields and zero invalid fields."""
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    masks = batch.field_masks()
    result = []
    for name in FIELD_NAMES:
        value = getattr(fields, name)
        clean = getattr(clean_fields, name)
        obs = observed.reshape(observed.shape + (1,) * (value.ndim - 2))
        valid = masks[name].to(value.dtype).reshape(masks[name].shape + (1,) * (value.ndim - 2))
        result.append(torch.where(obs, clean, value) * valid)
    return LatentFields(*result)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    return generator.manual_seed(int(seed))


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    batch: LatentBatch,
    *,
    steps: int,
    seed: int,
    adapter: Optional[StateDetailLatentAdapter] = None,
    statistics: Optional[LatentStatistics] = None,
    source_center: Optional[LatentFields] = None,
    source_mode: str = "gaussian",
) -> tuple[LatentBatch, dict[str, Any]]:
    if int(steps) not in (8, 16):
        raise ValueError("v1 sampling supports exactly 8 or 16 Euler steps")
    if batch.observed_mask is None:
        raise ValueError("sampling requires a block observation mask")
    steps = int(steps)
    was_training = model.training
    model.eval()
    noise = sample_isotropic_noise(batch.fields, generator=_make_generator(batch.state_h.device, seed))
    source = sample_source(source_center, noise, source_mode=source_mode)
    source = _mask_fields(source, batch)
    current = apply_observation_clamp(source, batch.fields, batch)
    for step in range(steps):
        tau = torch.full(
            (batch.batch_size,), float(step) / steps,
            device=batch.state_h.device, dtype=batch.state_h.dtype,
        )
        prediction = model(batch.with_fields(current), tau)
        if not isinstance(prediction, LatentFields):
            raise TypeError("the DiT must return a LatentFields")
        current = LatentFields(
            *(getattr(current, name) + getattr(prediction, name) / steps for name in FIELD_NAMES)
        )
        current = apply_observation_clamp(current, batch.fields, batch)
    result = batch.with_fields(current).zero_invalid()
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    clamp_errors: list[Tensor] = []
    for name in FIELD_NAMES:
        value = getattr(current, name)
        clean = getattr(batch.fields, name)
        expanded = observed.reshape(observed.shape + (1,) * (value.ndim - 2))
        if bool(torch.any(expanded)):
            clamp_errors.append((value - clean).abs().masked_select(expanded).max())
    observed_clamp_max_abs = (
        float(torch.stack(clamp_errors).max().detach().cpu()) if clamp_errors else 0.0
    )
    if statistics is not None:
        result = statistics.inverse_normalize(result)
    if was_training:
        model.train()
    metadata = {
        "solver": "euler",
        "steps": steps,
        "seed": int(seed),
        "deterministic": True,
        "observed_clamp": True,
        "observed_clamp_exact": observed_clamp_max_abs == 0.0,
        "observed_clamp_max_abs": observed_clamp_max_abs,
        "stats_hash": "" if statistics is None else statistics.hash,
        "adapter_hash": "" if adapter is None else contract_hash(adapter.contract()),
        "source_mode": source_mode,
        "source_center": source_mode == "conditional",
    }
    return result, metadata


def generate_state_detail_latent(
    model: nn.Module,
    adapter: StateDetailLatentAdapter,
    batch: LatentBatch,
    statistics: LatentStatistics,
    *,
    steps: int,
    seed: int,
    source_center: Optional[LatentFields] = None,
    source_mode: str = "gaussian",
    ) -> tuple[Any, dict[str, Any]]:
    normalized = statistics.normalize(batch)
    sampled, metadata = euler_sample(
        model,
        normalized,
        steps=steps,
        seed=seed,
        adapter=adapter,
        statistics=statistics,
        source_center=source_center,
        source_mode=source_mode,
    )
    metadata["raw_detail_h"] = None
    metadata["raw_detail_v"] = None
    metadata["latent_contract_hash"] = sampled.hash
    return adapter.make_generated_latent(sampled, sampled.fields), metadata


@torch.no_grad()
def euler_sample_frames(
    model: "FrameJointModel",
    *,
    context: ObservedContext,
    query: QuerySpec,
    steps: int = 16,
    seed: int,
) -> tuple[FrameLatentBatch, dict[str, Any]]:
    """Jointly update every future frame at each Euler flow step."""

    if int(steps) < 1:
        raise ValueError("Euler steps must be positive")
    normalized_observed = model.normalize(context.latent)
    center = repeat_last_frame_center(normalized_observed, query)
    current, _ = sample_frame_source(
        center, generator=_make_generator(context.latent.h.device, seed)
    )
    prepared_condition = model.prepare_condition(context)
    was_training = model.training
    model.eval()
    for step in range(int(steps)):
        flow_time = torch.full(
            (context.latent.batch_size,),
            float(step) / float(steps),
            device=current.h.device,
            dtype=current.h.dtype,
        )
        output = model(
            context=context,
            query=query,
            noisy_future=current,
            flow_time=flow_time,
            prepared_condition=prepared_condition,
        )
        velocity = output.velocity
        current = current.with_features(
            current.h + velocity.h / float(steps),
            current.v + velocity.v / float(steps),
        )
    physical = model.inverse(current)
    if was_training:
        model.train()
    return physical, {
        "solver": "euler",
        "steps": int(steps),
        "seed": int(seed),
        "joint_query_update": True,
        "statistics_hash": model.statistics_hash,
        "future_fields": ["h", "v"],
    }
