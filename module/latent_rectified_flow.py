"""Normalized latent rectified flow for the state/detail DiT probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from .state_detail_latent_adapter import (
    DiTLatentBatch,
    LatentFieldSet,
    LatentStatistics,
    StateDetailLatentAdapter,
    contract_hash,
)
from .latent_flow_source import combine_source_fields

FIELD_NAMES = ("state_h", "detail_h", "state_v", "detail_v")


def broadcast_sample_time(
    tau: Tensor | float, target: Tensor, *, sample_ids: Optional[Tensor] = None
) -> Tensor:
    """Broadcast one tau per sample to any scalar/vector field rank."""
    value = torch.as_tensor(tau, device=target.device, dtype=target.dtype)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape((1,) * target.ndim)
    if value.ndim != 1:
        raise ValueError("tau must be a scalar or one-dimensional")
    if sample_ids is not None:
        ids = torch.as_tensor(sample_ids, device=target.device, dtype=torch.long).flatten()
        if target.ndim < 2 or ids.numel() != target.shape[1]:
            raise ValueError("sample_ids must align to the target atom axis")
        return value.index_select(0, ids).reshape(
            (1, ids.numel()) + (1,) * (target.ndim - 2)
        )
    if value.numel() != target.shape[0]:
        raise ValueError("per-sample tau must match the target leading axis")
    return value.reshape((value.numel(),) + (1,) * (target.ndim - 1))


def rectified_flow_interpolate(
    z_data: Tensor, eps: Tensor, tau: Tensor | float, *, sample_ids: Optional[Tensor] = None
) -> Tensor:
    """Compute z_tau=(1-tau)*eps+tau*z_data for arbitrary tensor ranks."""
    if z_data.shape != eps.shape:
        raise ValueError("z_data and eps must have identical shapes")
    t = broadcast_sample_time(tau, z_data, sample_ids=sample_ids)
    return (1.0 - t) * eps + t * z_data


def rectified_flow_velocity(z_data: Tensor, eps: Tensor) -> Tensor:
    """Compute u_target=z_data-eps."""
    if z_data.shape != eps.shape:
        raise ValueError("z_data and eps must have identical shapes")
    return z_data - eps


interpolate = rectified_flow_interpolate
velocity_target = rectified_flow_velocity


def sample_isotropic_noise(
    fields: LatentFieldSet, *, generator: Optional[torch.Generator] = None
) -> LatentFieldSet:
    def draw(value: Tensor) -> Tensor:
        return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
    return fields.map(draw)


@dataclass(frozen=True)
class FlowLoss:
    total: Tensor
    fields: Mapping[str, Tensor]
    valid_elements: Mapping[str, int]

    def as_dict(self) -> dict[str, Tensor | int]:
        result: dict[str, Tensor | int] = {"total": self.total}
        for name in FIELD_NAMES:
            result[f"{name}_loss"] = self.fields[name]
            result[f"{name}_valid_elements"] = self.valid_elements[name]
        return result


def _masked_field_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, int]:
    if prediction.shape != target.shape or tuple(mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("field prediction, target, and mask shapes disagree")
    expanded = mask.to(prediction.dtype)
    if prediction.ndim == 3:
        expanded = expanded.unsqueeze(-1)
        elements = prediction.shape[-1]
    elif prediction.ndim == 4 and prediction.shape[2] == 3:
        expanded = expanded.unsqueeze(-1).unsqueeze(-1)
        elements = 3 * prediction.shape[-1]
    else:
        raise ValueError("fields must have rank 3 or rank 4 with xyz axis")
    valid = int(mask.sum().item()) * elements
    if valid == 0:
        return prediction.sum() * 0.0, 0
    return ((prediction - target).square() * expanded).sum() / valid, valid


def four_field_loss(prediction: LatentFieldSet, target: LatentFieldSet, batch: DiTLatentBatch) -> FlowLoss:
    values: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    masks = batch.field_masks()
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    for name in FIELD_NAMES:
        masks[name] = masks[name] & ~observed
        values[name], counts[name] = _masked_field_mse(
            getattr(prediction, name), getattr(target, name), masks[name]
        )
    return FlowLoss(sum(values.values()) / 4.0, values, counts)


compute_four_field_loss = four_field_loss
masked_four_field_loss = four_field_loss


def _mask_fields(fields: LatentFieldSet, batch: DiTLatentBatch) -> LatentFieldSet:
    masks = batch.field_masks()
    return LatentFieldSet(
        fields.state_h * masks["state_h"].to(fields.state_h.dtype).unsqueeze(-1),
        fields.detail_h * masks["detail_h"].to(fields.detail_h.dtype).unsqueeze(-1),
        fields.state_v * masks["state_v"].to(fields.state_v.dtype).unsqueeze(-1).unsqueeze(-1),
        fields.detail_v * masks["detail_v"].to(fields.detail_v.dtype).unsqueeze(-1).unsqueeze(-1),
    )


def _zero_observed(fields: LatentFieldSet, batch: DiTLatentBatch) -> LatentFieldSet:
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    result = []
    for name in FIELD_NAMES:
        value = getattr(fields, name)
        result.append(value * (~observed).to(value.dtype).reshape(
            observed.shape + (1,) * (value.ndim - 2)
        ))
    return LatentFieldSet(*result)


def apply_observation_clamp(
    fields: LatentFieldSet, clean_fields: LatentFieldSet, batch: DiTLatentBatch
) -> LatentFieldSet:
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
    return LatentFieldSet(*result)


@dataclass(frozen=True)
class FlowSample:
    tau: Tensor
    noise: LatentFieldSet
    source: LatentFieldSet
    interpolated: LatentFieldSet
    target: LatentFieldSet


class RectifiedFlowObjective(nn.Module):
    def __init__(self, *, tau_min: float = 0.0, tau_max: float = 1.0) -> None:
        super().__init__()
        if not (0.0 <= tau_min < tau_max <= 1.0):
            raise ValueError("tau range must satisfy 0 <= tau_min < tau_max <= 1")
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)

    def sample(
        self,
        batch: DiTLatentBatch,
        *,
        generator: Optional[torch.Generator] = None,
        source_center: Optional[LatentFieldSet] = None,
        source_mode: str = "gaussian",
    ) -> FlowSample:
        tau = torch.rand(
            (batch.batch_size,), device=batch.state_h.device, dtype=batch.state_h.dtype, generator=generator
        )
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau
        noise = sample_isotropic_noise(batch.fields, generator=generator)
        source = combine_source_fields(
            source_center,
            noise,
            source_mode=source_mode,
        )
        source = _mask_fields(source, batch)
        interpolated = LatentFieldSet(
            rectified_flow_interpolate(batch.state_h, source.state_h, tau, sample_ids=batch.abid),
            rectified_flow_interpolate(batch.detail_h, source.detail_h, tau, sample_ids=batch.abid),
            rectified_flow_interpolate(batch.state_v, source.state_v, tau, sample_ids=batch.abid),
            rectified_flow_interpolate(batch.detail_v, source.detail_v, tau, sample_ids=batch.abid),
        )
        target = LatentFieldSet(
            rectified_flow_velocity(batch.state_h, source.state_h),
            rectified_flow_velocity(batch.detail_h, source.detail_h),
            rectified_flow_velocity(batch.state_v, source.state_v),
            rectified_flow_velocity(batch.detail_v, source.detail_v),
        )
        interpolated = _mask_fields(interpolated, batch)
        target = _zero_observed(_mask_fields(target, batch), batch)
        interpolated = apply_observation_clamp(interpolated, batch.fields, batch)
        return FlowSample(tau, noise, source, interpolated, target)

    def loss(self, prediction: LatentFieldSet, target: LatentFieldSet, batch: DiTLatentBatch) -> FlowLoss:
        return four_field_loss(prediction, target, batch)

    def forward(
        self, model: nn.Module, batch: DiTLatentBatch, *, generator: Optional[torch.Generator] = None
    ) -> tuple[FlowLoss, FlowSample, LatentFieldSet]:
        sample = self.sample(batch, generator=generator)
        prediction = model(batch.with_fields(sample.interpolated), sample.tau)
        if not isinstance(prediction, LatentFieldSet):
            raise TypeError("the DiT must return a LatentFieldSet")
        return self.loss(prediction, sample.target, batch), sample, prediction


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    return generator.manual_seed(int(seed))


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    batch: DiTLatentBatch,
    *,
    steps: int,
    seed: int,
    adapter: Optional[StateDetailLatentAdapter] = None,
    statistics: Optional[LatentStatistics] = None,
    source_center: Optional[LatentFieldSet] = None,
    source_mode: str = "gaussian",
) -> tuple[DiTLatentBatch, dict[str, Any]]:
    if int(steps) not in (8, 16):
        raise ValueError("v1 sampling supports exactly 8 or 16 Euler steps")
    if batch.observed_mask is None:
        raise ValueError("sampling requires a block observation mask")
    steps = int(steps)
    was_training = model.training
    model.eval()
    noise = sample_isotropic_noise(batch.fields, generator=_make_generator(batch.state_h.device, seed))
    source = combine_source_fields(source_center, noise, source_mode=source_mode)
    source = _mask_fields(source, batch)
    current = apply_observation_clamp(source, batch.fields, batch)
    for step in range(steps):
        tau = torch.full(
            (batch.batch_size,), float(step) / steps,
            device=batch.state_h.device, dtype=batch.state_h.dtype,
        )
        prediction = model(batch.with_fields(current), tau)
        if not isinstance(prediction, LatentFieldSet):
            raise TypeError("the DiT must return a LatentFieldSet")
        current = LatentFieldSet(
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
    batch: DiTLatentBatch,
    statistics: LatentStatistics,
    *,
    steps: int,
    seed: int,
    source_center: Optional[LatentFieldSet] = None,
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


__all__ = [
    "FIELD_NAMES", "FlowLoss", "FlowSample", "RectifiedFlowObjective",
    "apply_observation_clamp", "broadcast_sample_time", "compute_four_field_loss",
    "euler_sample", "four_field_loss", "generate_state_detail_latent",
    "interpolate", "masked_four_field_loss", "rectified_flow_interpolate",
    "rectified_flow_velocity", "sample_isotropic_noise", "velocity_target",
]
