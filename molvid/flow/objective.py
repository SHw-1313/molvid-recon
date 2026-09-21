"""Masked four-field rectified-flow mathematics and training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from ..latent.types import FrameLatentBatch, LatentBatch, LatentFields
from .sampling import apply_observation_clamp
from .source import FIELD_NAMES, _mask_fields, sample_isotropic_noise, sample_source


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


def four_field_loss(prediction: LatentFields, target: LatentFields, batch: LatentBatch) -> FlowLoss:
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


def _zero_observed(fields: LatentFields, batch: LatentBatch) -> LatentFields:
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    result = []
    for name in FIELD_NAMES:
        value = getattr(fields, name)
        result.append(value * (~observed).to(value.dtype).reshape(
            observed.shape + (1,) * (value.ndim - 2)
        ))
    return LatentFields(*result)


@dataclass(frozen=True)
class FlowSample:
    tau: Tensor
    noise: LatentFields
    source: LatentFields
    interpolated: LatentFields
    target: LatentFields


class RectifiedFlowObjective(nn.Module):
    def __init__(self, *, tau_min: float = 0.0, tau_max: float = 1.0) -> None:
        super().__init__()
        if not (0.0 <= tau_min < tau_max <= 1.0):
            raise ValueError("tau range must satisfy 0 <= tau_min < tau_max <= 1")
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)

    def sample(
        self,
        batch: LatentBatch,
        *,
        generator: Optional[torch.Generator] = None,
        source_center: Optional[LatentFields] = None,
        source_mode: str = "gaussian",
    ) -> FlowSample:
        tau = torch.rand(
            (batch.batch_size,), device=batch.state_h.device, dtype=batch.state_h.dtype, generator=generator
        )
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau
        noise = sample_isotropic_noise(batch.fields, generator=generator)
        source = sample_source(
            source_center,
            noise,
            source_mode=source_mode,
        )
        source = _mask_fields(source, batch)
        interpolated = LatentFields(
            rectified_flow_interpolate(batch.state_h, source.state_h, tau, sample_ids=batch.abid),
            rectified_flow_interpolate(batch.detail_h, source.detail_h, tau, sample_ids=batch.abid),
            rectified_flow_interpolate(batch.state_v, source.state_v, tau, sample_ids=batch.abid),
            rectified_flow_interpolate(batch.detail_v, source.detail_v, tau, sample_ids=batch.abid),
        )
        target = LatentFields(
            rectified_flow_velocity(batch.state_h, source.state_h),
            rectified_flow_velocity(batch.detail_h, source.detail_h),
            rectified_flow_velocity(batch.state_v, source.state_v),
            rectified_flow_velocity(batch.detail_v, source.detail_v),
        )
        interpolated = _mask_fields(interpolated, batch)
        target = _zero_observed(_mask_fields(target, batch), batch)
        interpolated = apply_observation_clamp(interpolated, batch.fields, batch)
        return FlowSample(tau, noise, source, interpolated, target)

    def loss(self, prediction: LatentFields, target: LatentFields, batch: LatentBatch) -> FlowLoss:
        return four_field_loss(prediction, target, batch)

    def forward(
        self, model: nn.Module, batch: LatentBatch, *, generator: Optional[torch.Generator] = None
    ) -> tuple[FlowLoss, FlowSample, LatentFields]:
        sample = self.sample(batch, generator=generator)
        prediction = model(batch.with_fields(sample.interpolated), sample.tau)
        if not isinstance(prediction, LatentFields):
            raise TypeError("the DiT must return a LatentFields")
        return self.loss(prediction, sample.target, batch), sample, prediction


def endpoint_from_velocity(
    interpolated: LatentFields,
    velocity: LatentFields,
    tau: Tensor,
    *,
    sample_ids: Tensor,
) -> LatentFields:
    """Estimate the data endpoint ``z_tau + (1-tau) u_theta``."""

    values = []
    for name in interpolated.names():
        current = getattr(interpolated, name)
        prediction = getattr(velocity, name)
        if prediction.shape != current.shape:
            raise ValueError("endpoint velocity and interpolated fields disagree")
        one_minus_tau = 1.0 - broadcast_sample_time(
            tau, current, sample_ids=sample_ids
        )
        values.append(current + one_minus_tau * prediction)
    return LatentFields(*values)


@dataclass(frozen=True)
class FrameFlowLoss:
    total: Tensor
    h: Tensor
    v: Tensor
    applicable_samples: Tensor


@dataclass(frozen=True)
class FrameFlowSample:
    flow_time: Tensor
    source: FrameLatentBatch
    noise: FrameLatentBatch
    interpolated: FrameLatentBatch
    target_velocity: FrameLatentBatch


def _sample_equal_frame_mse(prediction: Tensor, target: Tensor, batch: FrameLatentBatch) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.shape[:2] != batch.h.shape[:2]:
        raise ValueError("frame flow prediction/target axes disagree")
    mask = batch.atom_frame_mask()
    error = (prediction - target).float().square().flatten(start_dim=2).mean(dim=-1)
    sample = batch.abid.unsqueeze(0).expand(batch.frames, -1)
    sums = error.new_zeros((batch.batch_size,))
    counts = error.new_zeros((batch.batch_size,))
    sums.scatter_add_(0, sample[mask], error[mask])
    counts.scatter_add_(0, sample[mask], torch.ones_like(error[mask]))
    applicable = counts > 0
    if not bool(torch.any(applicable)):
        return prediction.sum() * 0.0, applicable
    return (sums[applicable] / counts[applicable]).mean(), applicable


def frame_endpoint_from_velocity(
    interpolated: FrameLatentBatch,
    velocity: FrameLatentBatch,
    flow_time: Tensor,
) -> FrameLatentBatch:
    if interpolated.h.shape != velocity.h.shape or interpolated.v.shape != velocity.v.shape:
        raise ValueError("frame endpoint inputs disagree")
    one_minus = 1.0 - broadcast_sample_time(flow_time, interpolated.h, sample_ids=interpolated.abid)
    one_minus_v = 1.0 - broadcast_sample_time(flow_time, interpolated.v, sample_ids=interpolated.abid)
    return interpolated.with_features(
        interpolated.h + one_minus * velocity.h,
        interpolated.v + one_minus_v * velocity.v,
    )


class FrameRectifiedFlowObjective(nn.Module):
    """Two-field per-frame RF objective with equal sample weighting."""

    def sample(
        self,
        target: FrameLatentBatch,
        source_center: FrameLatentBatch,
        *,
        generator: Optional[torch.Generator] = None,
        flow_time: Tensor | None = None,
    ) -> FrameFlowSample:
        from .source import sample_frame_source

        if target.h.shape != source_center.h.shape or target.v.shape != source_center.v.shape:
            raise ValueError("frame flow target and source center shapes differ")
        if flow_time is None:
            flow_time = torch.rand(
                (target.batch_size,), device=target.h.device, dtype=target.h.dtype, generator=generator
            )
        else:
            flow_time = torch.as_tensor(flow_time, device=target.h.device, dtype=target.h.dtype).reshape(-1)
            if flow_time.shape != (target.batch_size,) or not torch.isfinite(flow_time).all():
                raise ValueError("flow_time override must contain one finite value per sample")
            if torch.any(flow_time < 0) or torch.any(flow_time > 1):
                raise ValueError("flow_time override must lie in [0,1]")
        source, noise = sample_frame_source(source_center, generator=generator)
        h = rectified_flow_interpolate(target.h, source.h, flow_time, sample_ids=target.abid)
        v = rectified_flow_interpolate(target.v, source.v, flow_time, sample_ids=target.abid)
        interpolated = target.with_features(h, v)
        velocity = target.with_features(target.h - source.h, target.v - source.v)
        return FrameFlowSample(flow_time, source, noise, interpolated, velocity)

    def loss(self, prediction: FrameLatentBatch, target: FrameLatentBatch) -> FrameFlowLoss:
        h, applicable_h = _sample_equal_frame_mse(prediction.h, target.h, target)
        v, applicable_v = _sample_equal_frame_mse(prediction.v, target.v, target)
        if not torch.equal(applicable_h, applicable_v):
            raise RuntimeError("scalar/vector frame flow applicability differs")
        return FrameFlowLoss(total=0.5 * h + 0.5 * v, h=h, v=v, applicable_samples=applicable_h)
