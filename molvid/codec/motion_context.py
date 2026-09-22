"""Observed-only motion summaries used by the Frame DiT M branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..equivariant import AxisPreservingLinear
from ..latent.types import ObservedContext, QuerySpec
from ..time import motion_time_features, multiscale_signed_features


@dataclass(frozen=True)
class MotionContext:
    h: Tensor  # [N, scalar_width]
    v: Tensor  # [N, 3, vector_width]
    history_span_ps: Tensor  # [B]
    observed_delta_ps: Tensor  # [B,H-1]
    observed_delta_mask: Tensor  # [B,H-1]


class MotionContextEncoder(nn.Module):
    """Aggregate scalar/vector teacher differences without future inputs."""

    def __init__(self, channels: int, scalar_width: int, vector_width: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        # dt and history-span each contribute linear/log features at 4 scales.
        self.weight_score = nn.Sequential(
            nn.Linear(self.channels + 16, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, 1),
        )
        self.scalar_summary = nn.Sequential(
            nn.Linear(4 * self.channels, self.scalar_width),
            nn.SiLU(),
            nn.Linear(self.scalar_width, self.scalar_width),
        )
        self.vector_summary = AxisPreservingLinear(2 * self.channels, self.vector_width)

    def forward(self, context: ObservedContext) -> MotionContext:
        latent = context.latent
        if latent.frames < 2:
            raise ValueError("motion context requires at least two observed frames")
        interval_mask = latent.frame_mask[:, 1:] & latent.frame_mask[:, :-1]
        delta_ps = latent.time_ps[:, 1:] - latent.time_ps[:, :-1]
        if bool(torch.any(interval_mask & (delta_ps <= 0))):
            raise ValueError("observed motion intervals must have positive physical dt")
        valid_count = latent.frame_mask.sum(dim=1).long()
        last = valid_count - 1
        span = latent.time_ps.gather(1, last.unsqueeze(1)).squeeze(1) - latent.time_ps[:, 0]
        dh = latent.h[1:] - latent.h[:-1]
        dv = latent.v[1:] - latent.v[:-1]
        state = 0.5 * (latent.h[1:] + latent.h[:-1])
        dt_features = multiscale_signed_features(delta_ps).index_select(0, latent.abid).transpose(0, 1)
        span_features = multiscale_signed_features(span[:, None]).squeeze(1).index_select(0, latent.abid)
        span_features = span_features.unsqueeze(0).expand(latent.frames - 1, -1, -1)
        score = self.weight_score(torch.cat((state, dt_features, span_features), dim=-1)).squeeze(-1)
        atom_interval_mask = interval_mask.index_select(0, latent.abid).transpose(0, 1)
        score = score.masked_fill(~atom_interval_mask, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=0) * atom_interval_mask.to(dtype=score.dtype)
        weight = weight / weight.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        weighted_dh = (weight.unsqueeze(-1) * dh).sum(dim=0)
        weighted_abs_dh = (weight.unsqueeze(-1) * dh.abs()).sum(dim=0)
        vector_magnitude = dv.float().square().sum(dim=2).sqrt().to(dtype=dh.dtype)
        weighted_vector_magnitude = (weight.unsqueeze(-1) * vector_magnitude).sum(dim=0)
        atom_last = last.index_select(0, latent.abid)
        atom = torch.arange(latent.num_atoms, device=latent.h.device)
        relative_h = latent.h[atom_last, atom] - latent.h[0]
        relative_v = latent.v[atom_last, atom] - latent.v[0]
        scalar_input = torch.cat(
            (weighted_dh, weighted_abs_dh, weighted_vector_magnitude, relative_h), dim=-1
        )
        # Subtraction makes static histories exactly zero despite MLP biases.
        scalar = self.scalar_summary(scalar_input) - self.scalar_summary(torch.zeros_like(scalar_input))
        weighted_dv = (weight.unsqueeze(-1).unsqueeze(-1) * dv).sum(dim=0)
        vector = self.vector_summary(torch.cat((weighted_dv, relative_v), dim=-1))
        return MotionContext(
            h=scalar,
            v=vector,
            history_span_ps=span,
            observed_delta_ps=delta_ps,
            observed_delta_mask=interval_mask,
        )

    def contract(self) -> dict[str, Any]:
        return {
            "inputs": "observed teacher h/v and observed physical time only",
            "differences": ["adjacent", "relative_to_last"],
            "aggregation": "learned_valid_interval_weights",
            "static_history_output": "exact_zero",
            "vector_axis": "SO3_equivariant_no_xyz_mixing",
        }


class MotionContextInjector(nn.Module):
    """Inject motion content and explicit time into one DiT block."""

    time_feature_width = 33

    def __init__(self, scalar_width: int, vector_width: int) -> None:
        super().__init__()
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.time_projection = nn.Sequential(
            nn.Linear(self.time_feature_width, self.scalar_width),
            nn.SiLU(),
            nn.Linear(self.scalar_width, self.scalar_width),
        )
        self.scalar_f = nn.Sequential(
            nn.Linear(3 * self.scalar_width, 2 * self.scalar_width),
            nn.SiLU(),
            nn.Linear(2 * self.scalar_width, self.scalar_width),
        )
        self.vector_mix = AxisPreservingLinear(self.vector_width, self.vector_width)
        self.vector_content_gate = nn.Linear(3 * self.scalar_width, self.vector_width)
        self.scalar_residual_gate = nn.Parameter(torch.zeros(self.scalar_width))
        self.vector_residual_gate = nn.Parameter(torch.zeros(self.vector_width))

    def forward(
        self,
        motion: MotionContext,
        condition: Tensor,
        context: ObservedContext,
        query: QuerySpec,
    ) -> tuple[Tensor, Tensor]:
        time = motion_time_features(
            context.latent.time_ps,
            context.latent.frame_mask,
            query.time_ps,
            query.frame_mask,
        ).encoded
        if time.shape[-1] != self.time_feature_width:
            raise RuntimeError("motion time feature width changed")
        time = self.time_projection(time).index_select(0, context.latent.abid).transpose(0, 1)
        motion_h = motion.h.unsqueeze(0).expand(condition.shape[0], -1, -1)
        combined = torch.cat((motion_h, condition, time), dim=-1)
        zero_combined = torch.cat((torch.zeros_like(motion_h), condition, time), dim=-1)
        scalar = self.scalar_f(combined) - self.scalar_f(zero_combined)
        vector = self.vector_mix(motion.v).unsqueeze(0).expand(condition.shape[0], -1, -1, -1)
        vector = vector * torch.sigmoid(self.vector_content_gate(combined)).unsqueeze(2)
        mask = query.frame_mask.index_select(0, context.latent.abid).transpose(0, 1)
        return (
            torch.tanh(self.scalar_residual_gate) * scalar * mask.unsqueeze(-1),
            torch.tanh(self.vector_residual_gate).view(1, 1, 1, -1)
            * vector * mask.unsqueeze(-1).unsqueeze(-1),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "scalar_combination": "F(motion,state,time)-F(0,state,time)",
            "vector_combination": "invariant_gate_times_channel_mixed_motion_vector",
            "time_scales_ps": [10, 100, 1000, 10000],
            "time_quantities": [
                "query_horizon", "query_delta", "history_span", "mean_observed_delta"
            ],
            "outer_gate_initialization": "zero",
        }


__all__ = ["MotionContext", "MotionContextEncoder", "MotionContextInjector"]
