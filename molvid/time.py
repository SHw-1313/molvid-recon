"""Physical-time features shared by history, flow fields and the decoder."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


TIME_SCALE_PS = 100.0
MOTION_TIME_SCALES_PS = (10.0, 100.0, 1000.0, 10000.0)


def _signed_log1p(value: Tensor) -> Tensor:
    return value.sign() * torch.log1p(value.abs())


def multiscale_signed_features(
    value_ps: Tensor,
    *,
    scales_ps: tuple[float, ...] = MOTION_TIME_SCALES_PS,
) -> Tensor:
    """Return linear and signed-log features without periodic aliasing."""

    value = torch.as_tensor(value_ps)
    if not scales_ps or any(not math.isfinite(float(scale)) or float(scale) <= 0 for scale in scales_ps):
        raise ValueError("physical time scales must be finite and positive")
    features: list[Tensor] = []
    for scale in scales_ps:
        normalized = value / float(scale)
        features.extend((normalized, _signed_log1p(normalized)))
    return torch.stack(features, dim=-1)


@dataclass(frozen=True)
class MotionTimeFeatures:
    query_horizon_ps: Tensor
    query_delta_ps: Tensor
    history_span_ps: Tensor
    observed_delta_ps: Tensor
    observed_delta_mask: Tensor
    encoded: Tensor


def motion_time_features(
    observed_time_ps: Tensor,
    observed_frame_mask: Tensor,
    query_time_ps: Tensor,
    query_frame_mask: Tensor,
) -> MotionTimeFeatures:
    """Build shift-invariant M-path time quantities and multiscale features."""

    observed = torch.as_tensor(observed_time_ps)
    observed_mask = torch.as_tensor(
        observed_frame_mask, device=observed.device, dtype=torch.bool
    )
    query = torch.as_tensor(query_time_ps, device=observed.device, dtype=observed.dtype)
    query_mask = torch.as_tensor(query_frame_mask, device=observed.device, dtype=torch.bool)
    if observed.ndim != 2 or observed_mask.shape != observed.shape:
        raise ValueError("observed time/mask must have shape [B,H]")
    if query.ndim != 2 or query_mask.shape != query.shape or query.shape[0] != observed.shape[0]:
        raise ValueError("query time/mask must have shape [B,Q] with the observed batch")
    valid_count = observed_mask.sum(dim=1).long()
    if bool(torch.any(valid_count < 1)):
        raise ValueError("motion time features require observed history")
    last_index = valid_count - 1
    last = observed.gather(1, last_index.unsqueeze(1)).squeeze(1)
    first = observed[:, 0]
    horizon = query - last.unsqueeze(1)
    previous = torch.cat((last.unsqueeze(1), query[:, :-1]), dim=1)
    query_delta = query - previous
    if bool(torch.any(query_mask & (horizon <= 0))) or bool(torch.any(query_mask & (query_delta <= 0))):
        raise ValueError("valid query times must follow the last observation with positive intervals")
    observed_delta = observed[:, 1:] - observed[:, :-1]
    observed_delta_mask = observed_mask[:, 1:] & observed_mask[:, :-1]
    if bool(torch.any(observed_delta_mask & (observed_delta <= 0))):
        raise ValueError("valid observed intervals must be positive")
    span = last - first
    valid = observed_delta_mask.to(dtype=observed.dtype)
    mean_observed_delta = (observed_delta * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    query_frames = query.shape[1]
    encoded = torch.cat((
        multiscale_signed_features(horizon),
        multiscale_signed_features(query_delta),
        multiscale_signed_features(span[:, None]).expand(-1, query_frames, -1),
        multiscale_signed_features(mean_observed_delta[:, None]).expand(-1, query_frames, -1),
        query_mask.to(dtype=query.dtype).unsqueeze(-1),
    ), dim=-1)
    encoded = encoded * query_mask.to(dtype=encoded.dtype).unsqueeze(-1)
    return MotionTimeFeatures(
        query_horizon_ps=horizon,
        query_delta_ps=query_delta,
        history_span_ps=span,
        observed_delta_ps=observed_delta,
        observed_delta_mask=observed_delta_mask,
        encoded=encoded,
    )


def physical_time_features(
    time_ps: Tensor,
    frame_mask: Tensor,
    *,
    reference_time_ps: Tensor | None = None,
    previous_time_ps: Tensor | None = None,
    scale_ps: float = TIME_SCALE_PS,
) -> Tensor:
    """Return fixed dimensionless features for a ``[B,T]`` physical clock.

    The features contain relative time, adjacent real ``dt``, signed log1p
    transforms, two fixed Fourier bands for each value, and validity.  Flow
    progress is intentionally absent.
    """

    time = torch.as_tensor(time_ps)
    mask = torch.as_tensor(frame_mask, device=time.device, dtype=torch.bool)
    if time.ndim != 2 or mask.shape != time.shape:
        raise ValueError("time_ps and frame_mask must have shape [B,T]")
    if not math.isfinite(float(scale_ps)) or float(scale_ps) <= 0:
        raise ValueError("physical time scale must be finite and positive")
    batch, frames = time.shape
    if reference_time_ps is None:
        reference = time[:, 0]
    else:
        reference = torch.as_tensor(reference_time_ps, device=time.device, dtype=time.dtype).reshape(batch)
    if previous_time_ps is None:
        previous = torch.cat((reference[:, None], time[:, :-1]), dim=1)
    else:
        first = torch.as_tensor(previous_time_ps, device=time.device, dtype=time.dtype).reshape(batch, 1)
        previous = torch.cat((first, time[:, :-1]), dim=1)
    relative = (time - reference[:, None]) / float(scale_ps)
    delta = (time - previous) / float(scale_ps)
    valid = mask.to(dtype=time.dtype)
    relative = relative * valid
    delta = delta * valid
    values = [relative, delta, _signed_log1p(relative), _signed_log1p(delta)]
    for frequency in (1.0, 2.0):
        values.extend(
            (
                torch.sin(math.pi * frequency * relative),
                torch.cos(math.pi * frequency * relative) * valid,
                torch.sin(math.pi * frequency * delta),
                torch.cos(math.pi * frequency * delta) * valid,
            )
        )
    values.append(valid)
    return torch.stack(values, dim=-1)


def pairwise_time_features(query_time_ps: Tensor, key_time_ps: Tensor, *, scale_ps: float = TIME_SCALE_PS) -> Tensor:
    """Return signed relative-time features for ``[B,Q]`` and ``[B,K]``."""

    query = torch.as_tensor(query_time_ps)
    key = torch.as_tensor(key_time_ps, device=query.device, dtype=query.dtype)
    if query.ndim != 2 or key.ndim != 2 or query.shape[0] != key.shape[0]:
        raise ValueError("pairwise physical times must have shapes [B,Q] and [B,K]")
    delta = (query[:, :, None] - key[:, None, :]) / float(scale_ps)
    return torch.stack(
        (delta, _signed_log1p(delta), torch.sin(math.pi * delta), torch.cos(math.pi * delta)),
        dim=-1,
    )


class PhysicalTimeEmbedding(nn.Module):
    """Module-owned projection of the shared fixed physical-time definition."""

    feature_width = 13

    def __init__(self, output_width: int, *, zero_output: bool = False) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(self.feature_width, int(output_width)),
            nn.SiLU(),
            nn.Linear(int(output_width), int(output_width)),
        )
        if zero_output:
            nn.init.zeros_(self.projection[-1].weight)
            nn.init.zeros_(self.projection[-1].bias)

    def forward(
        self,
        time_ps: Tensor,
        frame_mask: Tensor,
        *,
        reference_time_ps: Tensor | None = None,
        previous_time_ps: Tensor | None = None,
    ) -> Tensor:
        return self.projection(
            physical_time_features(
                time_ps,
                frame_mask,
                reference_time_ps=reference_time_ps,
                previous_time_ps=previous_time_ps,
            )
        )
