"""Causal, physical-time-conditioned temporal mixing for PVB features.

The spatial PVB backbone produces invariant scalar features ``h`` and
SE(3)-equivariant vector features ``v`` per frame.  This module only mixes the
time axis: attention weights are scalar, vector channels are transformed on
the final channel axis, and the Cartesian axis is never linearly mixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TemporalState:
    """A time-major temporal feature state and its physical clock."""

    h: Tensor
    v: Tensor
    frame_mask: Tensor
    time_ps: Tensor
    abid: Tensor

    @property
    def frames(self) -> int:
        return int(self.h.shape[0])

    @property
    def atoms(self) -> int:
        return int(self.h.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.frame_mask.shape[0])


def _normalise_clock(
    time_ps: Tensor,
    frame_mask: Optional[Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Validate a physical clock and return tensors on the feature device."""

    time = torch.as_tensor(time_ps, device=device, dtype=dtype)
    if time.ndim == 1:
        time = time.unsqueeze(0)
    if time.ndim != 2 or time.shape[1] < 1:
        raise ValueError("time_ps must have shape [B, T] or [T]")
    if not torch.isfinite(time).all():
        raise ValueError("time_ps contains NaN or Inf")

    if frame_mask is None:
        mask = torch.ones(time.shape, device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(frame_mask, device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != time.shape:
            raise ValueError("frame_mask must have the same shape as time_ps")

    for sample in range(time.shape[0]):
        valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError("every sample must contain at least one valid frame")
        values = time[sample].index_select(0, valid)
        if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
            raise ValueError("valid time_ps values must be strictly increasing")
    return time, mask


def _prepare_context(
    h: Tensor,
    time_ps: Tensor,
    frame_mask: Optional[Tensor],
    abid: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if h.ndim != 3:
        raise ValueError("h must have shape [T, N, C]")
    time, mask = _normalise_clock(
        time_ps,
        frame_mask,
        device=h.device,
        dtype=h.dtype,
    )
    frames, atoms = int(h.shape[0]), int(h.shape[1])
    if time.shape[1] != frames:
        raise ValueError("time_ps length must equal h.shape[0]")
    if abid is None:
        sample_id = torch.zeros(atoms, device=h.device, dtype=torch.long)
    else:
        sample_id = torch.as_tensor(abid, device=h.device, dtype=torch.long).flatten()
        if sample_id.numel() != atoms:
            raise ValueError("abid must have shape [N]")
        if torch.any(sample_id < 0) or torch.any(sample_id >= time.shape[0]):
            raise ValueError("abid contains an invalid sample id")
    return time, mask, sample_id


def _mask_features(
    h: Tensor,
    v: Tensor,
    frame_mask: Tensor,
    abid: Tensor,
) -> tuple[Tensor, Tensor]:
    valid = frame_mask.index_select(0, abid).transpose(0, 1)
    return h * valid.unsqueeze(-1), v * valid.unsqueeze(-1).unsqueeze(-1)


class ContinuousTimeBias(nn.Module):
    """RBF/MLP bias from physical relative time in picoseconds."""

    def __init__(
        self,
        num_heads: int,
        *,
        num_rbf: int = 16,
        time_scale_ps: float = 100.0,
        max_time_ps: float = 10000.0,
        mlp_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_heads < 1 or num_rbf < 1:
            raise ValueError("num_heads and num_rbf must be positive")
        if time_scale_ps <= 0 or max_time_ps <= 0:
            raise ValueError("time scales must be positive")
        self.time_scale_ps = float(time_scale_ps)
        max_log_time = torch.log1p(torch.tensor(max_time_ps / time_scale_ps))
        centers = torch.linspace(0.0, float(max_log_time), num_rbf)
        width = float(centers[1] - centers[0]) if num_rbf > 1 else 1.0
        self.register_buffer("centers", centers)
        self.register_buffer("width", torch.tensor(max(width, 1e-3)))
        hidden = int(mlp_hidden or max(16, num_rbf))
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_heads),
        )

    def forward(self, relative_time_ps: Tensor) -> Tensor:
        if torch.any(relative_time_ps < 0):
            raise ValueError("relative time must be non-negative")
        scaled = torch.log1p(relative_time_ps / self.time_scale_ps)
        centers = self.centers.to(device=scaled.device, dtype=scaled.dtype)
        width = self.width.to(device=scaled.device, dtype=scaled.dtype)
        rbf = torch.exp(-0.5 * ((scaled.unsqueeze(-1) - centers) / width) ** 2)
        return self.mlp(rbf)


class SO3ChannelNorm(nn.Module):
    """Normalize vector channels over xyz without mixing Cartesian axes."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, v: Tensor) -> Tensor:
        # Vector norms are accumulated in FP32; the surrounding projections may
        # still execute under BF16 autocast.
        v_fp32 = v.float()
        denom = v_fp32.square().mean(dim=2, keepdim=True).add(self.eps).sqrt()
        return v_fp32 / denom * self.scale.float().view(1, 1, 1, -1)


class ScalarVectorFFN(nn.Module):
    """Pointwise scalar/vector FFN with invariant vector norms and scalar gates."""

    def __init__(self, scalar_channels: int, vector_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.scalar_in = nn.Linear(scalar_channels + vector_channels, hidden_channels)
        self.scalar_out = nn.Linear(hidden_channels, scalar_channels)
        self.vector_in = nn.Linear(vector_channels, hidden_channels, bias=False)
        self.vector_out = nn.Linear(hidden_channels, vector_channels, bias=False)
        self.vector_gate = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, h: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        v_norm = v.float().square().sum(dim=2).add(1e-6).sqrt()
        scalar_hidden = F.silu(self.scalar_in(torch.cat([h, v_norm], dim=-1)))
        h_out = self.scalar_out(scalar_hidden)
        v_hidden = self.vector_in(v)
        gate = torch.sigmoid(self.vector_gate(scalar_hidden)).unsqueeze(2)
        v_out = self.vector_out(v_hidden * gate)
        return h_out, v_out


class CausalEquivariantTemporalAttention(nn.Module):
    """Local left-looking attention over the same atom across time."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        vector_head_dim: Optional[int] = None,
        window_size: Optional[int] = 8,
        time_num_rbf: int = 16,
        time_scale_ps: float = 100.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be positive or None")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or max(1, scalar_channels // num_heads))
        self.vector_head_dim = int(vector_head_dim or max(1, vector_channels // num_heads))
        self.window_size = window_size
        self.scalar_norm = nn.LayerNorm(scalar_channels)
        self.vector_norm = SO3ChannelNorm(vector_channels)
        scalar_width = self.num_heads * self.head_dim
        vector_width = self.num_heads * self.vector_head_dim
        self.q_proj = nn.Linear(scalar_channels, scalar_width, bias=False)
        self.k_proj = nn.Linear(scalar_channels, scalar_width, bias=False)
        self.scalar_value = nn.Linear(scalar_channels, scalar_width, bias=False)
        self.scalar_out = nn.Linear(scalar_width, scalar_channels, bias=False)
        self.vector_value = nn.Linear(vector_channels, vector_width, bias=False)
        self.vector_out = nn.Linear(vector_width, vector_channels, bias=False)
        self.time_bias = ContinuousTimeBias(
            self.num_heads,
            num_rbf=time_num_rbf,
            time_scale_ps=time_scale_ps,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        time_ps: Tensor,
        frame_mask: Optional[Tensor] = None,
        abid: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("v must have shape [T, N, 3, Cv]")
        time, mask, sample_id = _prepare_context(h, time_ps, frame_mask, abid)
        frames, atoms = int(h.shape[0]), int(h.shape[1])
        h_norm = self.scalar_norm(h)
        v_norm = self.vector_norm(v)
        q = self.q_proj(h_norm).view(frames, atoms, self.num_heads, self.head_dim)
        k = self.k_proj(h_norm).view(frames, atoms, self.num_heads, self.head_dim)
        scalar_value = self.scalar_value(h_norm).view(
            frames, atoms, self.num_heads, self.head_dim
        )
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        scalar_value = scalar_value.permute(1, 2, 0, 3)
        logits = torch.einsum("nhid,nhjd->nhij", q, k) / (self.head_dim**0.5)

        atom_time = time.index_select(0, sample_id)
        relative = (atom_time[:, :, None] - atom_time[:, None, :]).abs()
        logits = logits + self.time_bias(relative).permute(0, 3, 1, 2)

        positions = torch.arange(frames, device=h.device)
        allowed = positions[None, :] <= positions[:, None]
        if self.window_size is not None:
            allowed = allowed & (
                positions[:, None] - positions[None, :] < self.window_size
            )
        key_valid = mask.index_select(0, sample_id)
        allowed = allowed.unsqueeze(0).unsqueeze(0) & key_valid[:, None, None, :]
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        attention = self.dropout(torch.softmax(logits, dim=-1))

        scalar_out = torch.einsum("nhij,nhjd->nhid", attention, scalar_value)
        scalar_out = scalar_out.permute(2, 0, 1, 3).reshape(
            frames, atoms, self.num_heads * self.head_dim
        )
        scalar_out = self.scalar_out(scalar_out)

        vector_value = self.vector_value(v_norm).view(
            frames, atoms, 3, self.num_heads, self.vector_head_dim
        )
        vector_value = vector_value.permute(1, 3, 0, 2, 4)
        vector_out = torch.einsum("nhij,nhjtd->nhitd", attention, vector_value)
        vector_out = vector_out.permute(2, 0, 3, 1, 4).reshape(
            frames, atoms, 3, self.num_heads * self.vector_head_dim
        )
        vector_out = self.vector_out(vector_out)
        return _mask_features(scalar_out, vector_out, mask, sample_id)


class CausalEquivariantTemporalBlock(nn.Module):
    """Attention plus scalar/vector FFN with prefix-causal masking."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        ffn_channels: Optional[int] = None,
        num_heads: int = 4,
        window_size: Optional[int] = 8,
        time_num_rbf: int = 16,
        time_scale_ps: float = 100.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = CausalEquivariantTemporalAttention(
            scalar_channels,
            vector_channels,
            num_heads=num_heads,
            window_size=window_size,
            time_num_rbf=time_num_rbf,
            time_scale_ps=time_scale_ps,
            dropout=dropout,
        )
        self.ffn_h_norm = nn.LayerNorm(scalar_channels)
        self.ffn_v_norm = SO3ChannelNorm(vector_channels)
        self.ffn = ScalarVectorFFN(
            scalar_channels,
            vector_channels,
            int(ffn_channels or scalar_channels * 4),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        time_ps: Tensor,
        frame_mask: Optional[Tensor] = None,
        abid: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        time, mask, sample_id = _prepare_context(h, time_ps, frame_mask, abid)
        attn_h, attn_v = self.attention(h, v, time, mask, sample_id)
        h = h + self.dropout(attn_h)
        v = v + self.dropout(attn_v)
        h, v = _mask_features(h, v, mask, sample_id)
        ffn_h, ffn_v = self.ffn(
            self.ffn_h_norm(h),
            self.ffn_v_norm(v),
        )
        h = h + self.dropout(ffn_h)
        v = v + self.dropout(ffn_v)
        return _mask_features(h, v, mask, sample_id)


class CausalTemporalDownsample(nn.Module):
    """Causal stride-2 pooling with a right-edge physical timestamp.

    Each output token consumes one non-overlapping two-frame chunk.  Its clock
    is the latest valid timestamp in that chunk, i.e. the right edge of the
    causal receptive field.  Invalid/padded frames contribute zero weight.
    """

    def __init__(self, scalar_channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(scalar_channels, 1)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        time_ps: Tensor,
        frame_mask: Optional[Tensor] = None,
        abid: Optional[Tensor] = None,
    ) -> TemporalState:
        time, mask, sample_id = _prepare_context(h, time_ps, frame_mask, abid)
        frames, atoms = int(h.shape[0]), int(h.shape[1])
        output_frames = (frames + 1) // 2
        h_values: list[Tensor] = []
        v_values: list[Tensor] = []
        output_mask = []
        output_time = []
        atom_mask = mask.index_select(0, sample_id).transpose(0, 1)
        for output_index in range(output_frames):
            start = output_index * 2
            stop = min(frames, start + 2)
            local_h = h[start:stop]
            local_v = v[start:stop]
            local_valid = atom_mask[start:stop].unsqueeze(-1)
            weights = torch.sigmoid(self.gate(local_h)) * local_valid
            denominator = weights.sum(dim=0).clamp_min(1e-6)
            h_values.append((local_h * weights).sum(dim=0) / denominator)
            v_values.append((local_v * weights.unsqueeze(2)).sum(dim=0) / denominator.unsqueeze(1))
            sample_valid = mask[:, start:stop].any(dim=1)
            output_mask.append(sample_valid)
            local_time = time[:, start:stop].masked_fill(~mask[:, start:stop], float("-inf"))
            latest = local_time.max(dim=1).values
            output_time.append(torch.where(sample_valid, latest, torch.zeros_like(latest)))
        return TemporalState(
            h=torch.stack(h_values, dim=0),
            v=torch.stack(v_values, dim=0),
            frame_mask=torch.stack(output_mask, dim=1),
            time_ps=torch.stack(output_time, dim=1),
            abid=sample_id,
        )


class CausalTemporalEncoder(nn.Module):
    """Temporal encoder supporting token ratios 1, 2, and 4."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        ratio: int = 4,
        num_layers: int = 1,
        ffn_channels: Optional[int] = None,
        num_heads: int = 4,
        window_size: Optional[int] = 8,
        time_num_rbf: int = 16,
        time_scale_ps: float = 100.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if ratio not in (1, 2, 4):
            raise ValueError("ratio must be one of 1, 2, or 4")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        stages = 0 if ratio == 1 else (1 if ratio == 2 else 2)
        self.ratio = ratio
        self.stage_blocks = nn.ModuleList()
        for _ in range(stages + 1):
            self.stage_blocks.append(
                nn.ModuleList(
                    [
                        CausalEquivariantTemporalBlock(
                            scalar_channels,
                            vector_channels,
                            ffn_channels=ffn_channels,
                            num_heads=num_heads,
                            window_size=window_size,
                            time_num_rbf=time_num_rbf,
                            time_scale_ps=time_scale_ps,
                            dropout=dropout,
                        )
                        for _ in range(num_layers)
                    ]
                )
            )
        self.downsamplers = nn.ModuleList(
            [CausalTemporalDownsample(scalar_channels) for _ in range(stages)]
        )

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        time_ps: Tensor,
        frame_mask: Optional[Tensor] = None,
        abid: Optional[Tensor] = None,
    ) -> TemporalState:
        time, mask, sample_id = _prepare_context(h, time_ps, frame_mask, abid)
        state = TemporalState(h, v, mask, time, sample_id)
        for stage, blocks in enumerate(self.stage_blocks):
            for block in blocks:
                state.h, state.v = block(
                    state.h,
                    state.v,
                    state.time_ps,
                    state.frame_mask,
                    state.abid,
                )
            if stage < len(self.downsamplers):
                state = self.downsamplers[stage](
                    state.h,
                    state.v,
                    state.time_ps,
                    state.frame_mask,
                    state.abid,
                )
        return state


class CausalTemporalUpsample(nn.Module):
    """Causally query latent tokens at requested physical target times."""

    def __init__(
        self,
        scalar_channels: int,
        *,
        time_scale_ps: float = 100.0,
    ) -> None:
        super().__init__()
        if time_scale_ps <= 0:
            raise ValueError("time_scale_ps must be positive")
        self.time_scale_ps = float(time_scale_ps)
        self.query = nn.Sequential(
            nn.Linear(1, scalar_channels),
            nn.SiLU(),
            nn.Linear(scalar_channels, scalar_channels),
        )

    def forward(
        self,
        latent_h: Tensor,
        latent_v: Tensor,
        latent_time_ps: Tensor,
        latent_mask: Tensor,
        target_time_ps: Tensor,
        target_mask: Optional[Tensor] = None,
        abid: Optional[Tensor] = None,
    ) -> TemporalState:
        if latent_h.ndim != 3 or latent_v.ndim != 4:
            raise ValueError("latent features must be [L, N, C] and [L, N, 3, Cv]")
        if latent_v.shape[:2] != latent_h.shape[:2] or latent_v.shape[2] != 3:
            raise ValueError("latent vector features have an invalid shape")
        latent_time, latent_valid = _normalise_clock(
            latent_time_ps,
            latent_mask,
            device=latent_h.device,
            dtype=latent_h.dtype,
        )
        target_time, target_valid = _normalise_clock(
            target_time_ps,
            target_mask,
            device=latent_h.device,
            dtype=latent_h.dtype,
        )
        if latent_time.shape[0] != target_time.shape[0]:
            raise ValueError("latent and target clocks must have the same batch size")
        atoms = int(latent_h.shape[1])
        if abid is None:
            sample_id = torch.zeros(atoms, device=latent_h.device, dtype=torch.long)
        else:
            sample_id = torch.as_tensor(abid, device=latent_h.device, dtype=torch.long).flatten()
            if sample_id.numel() != atoms:
                raise ValueError("abid must have shape [N]")
            if torch.any(sample_id < 0) or torch.any(sample_id >= latent_time.shape[0]):
                raise ValueError("abid contains an invalid sample id")

        target_frames = int(target_time.shape[1])
        out_h = latent_h.new_zeros((target_frames, atoms, latent_h.shape[-1]))
        out_v = latent_v.new_zeros((target_frames, atoms, 3, latent_v.shape[-1]))
        for sample in range(latent_time.shape[0]):
            atom_indices = torch.nonzero(sample_id == sample, as_tuple=False).flatten()
            if atom_indices.numel() == 0:
                continue
            valid_indices = torch.nonzero(latent_valid[sample], as_tuple=False).flatten()
            clock = latent_time[sample].index_select(0, valid_indices)
            query_time = target_time[sample]
            right = torch.searchsorted(clock, query_time, right=True) - 1
            right = right.clamp(min=0, max=clock.numel() - 1)
            left = (right - 1).clamp(min=0)
            right_index = valid_indices.index_select(0, right)
            left_index = valid_indices.index_select(0, left)
            right_time = clock.index_select(0, right)
            left_time = clock.index_select(0, left)
            denominator = (right_time - left_time).clamp_min(1e-6)
            alpha = ((query_time - left_time) / denominator).clamp(0.0, 1.0)
            alpha = torch.where(right == left, torch.ones_like(alpha), alpha)
            h_right = latent_h.index_select(0, right_index).index_select(1, atom_indices)
            h_left = latent_h.index_select(0, left_index).index_select(1, atom_indices)
            v_right = latent_v.index_select(0, right_index).index_select(1, atom_indices)
            v_left = latent_v.index_select(0, left_index).index_select(1, atom_indices)
            alpha_h = alpha.view(-1, 1, 1)
            alpha_v = alpha.view(-1, 1, 1, 1)
            h_query = (1.0 - alpha_h) * h_left + alpha_h * h_right
            v_query = (1.0 - alpha_v) * v_left + alpha_v * v_right
            relative = (query_time - right_time).abs().div(self.time_scale_ps).log1p()
            h_query = h_query + self.query(relative.view(-1, 1)).unsqueeze(1)
            out_h[:, atom_indices] = h_query
            out_v[:, atom_indices] = v_query
        out_h, out_v = _mask_features(out_h, out_v, target_valid, sample_id)
        return TemporalState(out_h, out_v, target_valid, target_time, sample_id)


class CausalTemporalDecoder(nn.Module):
    """Target-time causal upsampling followed by temporal refinement blocks."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        num_layers: int = 1,
        ffn_channels: Optional[int] = None,
        num_heads: int = 4,
        window_size: Optional[int] = 8,
        time_num_rbf: int = 16,
        time_scale_ps: float = 100.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = CausalTemporalUpsample(
            scalar_channels,
            time_scale_ps=time_scale_ps,
        )
        self.refinement = nn.ModuleList(
            [
                CausalEquivariantTemporalBlock(
                    scalar_channels,
                    vector_channels,
                    ffn_channels=ffn_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    time_num_rbf=time_num_rbf,
                    time_scale_ps=time_scale_ps,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        latent: TemporalState,
        *,
        target_time_ps: Tensor,
        target_mask: Optional[Tensor] = None,
    ) -> TemporalState:
        state = self.upsample(
            latent.h,
            latent.v,
            latent.time_ps,
            latent.frame_mask,
            target_time_ps,
            target_mask,
            latent.abid,
        )
        for block in self.refinement:
            state.h, state.v = block(
                state.h,
                state.v,
                state.time_ps,
                state.frame_mask,
                state.abid,
            )
        return state


# Concise aliases for downstream codec code.
TemporalBlock = CausalEquivariantTemporalBlock
TemporalDownsample = CausalTemporalDownsample
TemporalUpsample = CausalTemporalUpsample
TemporalEncoder = CausalTemporalEncoder
TemporalDecoder = CausalTemporalDecoder
