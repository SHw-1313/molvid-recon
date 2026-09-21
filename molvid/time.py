"""Physical-time features shared by history, flow fields and the decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


TIME_SCALE_PS = 100.0


def _signed_log1p(value: Tensor) -> Tensor:
    return value.sign() * torch.log1p(value.abs())


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
