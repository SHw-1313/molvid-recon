"""Scalar/vector attention, modulation and factorized DiT parameter blocks."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..equivariant import AxisPreservingLinear, SO3ChannelNorm
from ..codec.motion_context import MotionContextInjector
from ..latent.types import LatentBatch
from .backend import reference_block_forward
from .local_geometry import LocalGeometryMessage

class ScalarVectorAttention(nn.Module):
    """Multi-head attention with scalar q/k and shared scalar weights."""

    def __init__(self, scalar_width: int, vector_width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if scalar_width % heads or vector_width % heads:
            raise ValueError("scalar and vector widths must be divisible by heads")
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.heads = int(heads)
        self.scalar_head = scalar_width // heads
        self.vector_head = vector_width // heads
        self.q = nn.Linear(scalar_width, scalar_width)
        self.k = nn.Linear(scalar_width, scalar_width)
        self.scalar_value = nn.Linear(scalar_width, scalar_width)
        self.scalar_out = nn.Linear(scalar_width, scalar_width)
        self.vector_value = AxisPreservingLinear(vector_width, vector_width)
        self.vector_out = AxisPreservingLinear(vector_width, vector_width)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        key_mask: Tensor,
        *,
        query_mask: Tensor | None = None,
        pair_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("attention expects [B,L,Dh] and [B,L,3,Dv]")
        batch, length = h.shape[:2]
        q = self.q(h).reshape(batch, length, self.heads, self.scalar_head).transpose(1, 2)
        k = self.k(h).reshape(batch, length, self.heads, self.scalar_head).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.scalar_head)
        if pair_bias is not None:
            if pair_bias.shape != logits.shape:
                raise ValueError("attention pair_bias must have shape [B,heads,L,L]")
            logits = logits + pair_bias
        valid = key_mask.to(dtype=torch.bool).reshape(batch, 1, 1, length)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        scalar_value = self.scalar_value(h).reshape(
            batch, length, self.heads, self.scalar_head
        )
        scalar_mixed = torch.einsum("bhij,bjhd->bihd", weights, scalar_value).reshape(
            batch, length, self.scalar_width
        )
        vector_value = self.vector_value(v).reshape(
            batch, length, 3, self.heads, self.vector_head
        ).permute(0, 3, 1, 2, 4)
        vector_mixed = torch.einsum("bhij,bhjrc->bhirc", weights, vector_value)
        vector_mixed = vector_mixed.permute(0, 2, 3, 1, 4).reshape(
            batch, length, 3, self.vector_width
        )
        scalar_output = self.scalar_out(scalar_mixed)
        vector_output = self.vector_out(vector_mixed)
        if query_mask is not None:
            if query_mask.shape != h.shape[:2]:
                raise ValueError("attention query_mask must have shape [B,L]")
            query = query_mask.to(dtype=scalar_output.dtype).unsqueeze(-1)
            scalar_output = scalar_output * query
            vector_output = vector_output * query.unsqueeze(-1)
        return scalar_output, vector_output


class ScalarVectorCrossAttention(nn.Module):
    """Scalar q/k cross-attention with shared weights for vector values."""

    def __init__(self, scalar_width: int, vector_width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if scalar_width % heads or vector_width % heads:
            raise ValueError("scalar and vector widths must be divisible by heads")
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.heads = int(heads)
        self.scalar_head = scalar_width // heads
        self.vector_head = vector_width // heads
        self.q = nn.Linear(scalar_width, scalar_width)
        self.k = nn.Linear(scalar_width, scalar_width)
        self.scalar_value = nn.Linear(scalar_width, scalar_width)
        self.scalar_out = nn.Linear(scalar_width, scalar_width)
        self.vector_value = AxisPreservingLinear(vector_width, vector_width)
        self.vector_out = AxisPreservingLinear(vector_width, vector_width)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        query_h: Tensor,
        query_v: Tensor,
        memory_h: Tensor,
        memory_v: Tensor,
        *,
        key_mask: Tensor,
        query_mask: Tensor,
        pair_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if query_h.ndim != 3 or query_v.shape != (*query_h.shape[:2], 3, self.vector_width):
            raise ValueError("cross-attention query expects [B,Q,Dh] and [B,Q,3,Dv]")
        if memory_h.ndim != 3 or memory_v.shape != (*memory_h.shape[:2], 3, self.vector_width):
            raise ValueError("cross-attention memory expects [B,M,Dh] and [B,M,3,Dv]")
        batch, queries = query_h.shape[:2]
        memory = int(memory_h.shape[1])
        if key_mask.shape != (batch, memory) or query_mask.shape != (batch, queries):
            raise ValueError("cross-attention masks disagree with query/memory axes")
        q = self.q(query_h).reshape(batch, queries, self.heads, self.scalar_head).transpose(1, 2)
        k = self.k(memory_h).reshape(batch, memory, self.heads, self.scalar_head).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.scalar_head)
        if pair_bias is not None:
            if pair_bias.shape != logits.shape:
                raise ValueError("cross-attention pair_bias must have shape [B,heads,Q,M]")
            logits = logits + pair_bias
        logits = logits.masked_fill(~key_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        scalar_value = self.scalar_value(memory_h).reshape(batch, memory, self.heads, self.scalar_head).permute(0, 2, 1, 3)
        scalar = torch.matmul(weights, scalar_value).transpose(1, 2).reshape(batch, queries, self.scalar_width)
        vector_value = self.vector_value(memory_v).reshape(batch, memory, 3, self.heads, self.vector_head).permute(0, 3, 1, 2, 4)
        vector = torch.einsum("bhqm,bhmrc->bhqrc", weights, vector_value).permute(0, 2, 3, 1, 4).reshape(batch, queries, 3, self.vector_width)
        query = query_mask.to(dtype=scalar.dtype).unsqueeze(-1)
        return self.scalar_out(scalar) * query, self.vector_out(vector) * query.unsqueeze(-1)


class AdaLNZero(nn.Module):
    """Scalar shifts are allowed; vector modulation has scale/gate only."""

    def __init__(self, scalar_width: int, vector_width: int) -> None:
        super().__init__()
        self.scalar_norm = nn.LayerNorm(scalar_width, elementwise_affine=False)
        self.vector_norm = SO3ChannelNorm(vector_width)
        self.modulation = nn.Linear(scalar_width, 3 * scalar_width + 2 * vector_width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h: Tensor, v: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = self.modulation(condition)
        dh = h.shape[-1]
        dv = v.shape[-1]
        h_shift, h_scale, h_gate, v_scale, v_gate = torch.split(
            values, (dh, dh, dh, dv, dv), dim=-1
        )
        h_norm = self.scalar_norm(h) * (1.0 + h_scale) + h_shift
        v_norm = self.vector_norm(v) * (1.0 + v_scale).unsqueeze(-2)
        return h_norm, v_norm, h_gate, v_gate


class ScalarVectorFFN(nn.Module):
    def __init__(self, scalar_width: int, vector_width: int, multiplier: int) -> None:
        super().__init__()
        scalar_hidden = int(scalar_width * multiplier)
        vector_hidden = int(vector_width * multiplier)
        self.scalar_in = nn.Linear(scalar_width + vector_width, scalar_hidden)
        self.scalar_out = nn.Linear(scalar_hidden, scalar_width)
        self.vector_in = AxisPreservingLinear(vector_width, vector_hidden)
        self.vector_out = AxisPreservingLinear(vector_hidden, vector_width)
        self.vector_gate = nn.Linear(scalar_hidden, vector_hidden)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        *,
        scalar_vector: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        scalar_vector = v if scalar_vector is None else scalar_vector
        if scalar_vector.shape != v.shape:
            raise ValueError("scalar_vector must have the same shape as the FFN vector input")
        vector_norm = scalar_vector.float().square().sum(dim=-2).add(1.0e-6).sqrt()
        vector_norm = vector_norm.to(dtype=h.dtype)
        scalar_hidden = F.silu(self.scalar_in(torch.cat((h, vector_norm), dim=-1)))
        scalar = self.scalar_out(scalar_hidden)
        vector_hidden = self.vector_in(v)
        gate = torch.sigmoid(self.vector_gate(scalar_hidden)).unsqueeze(-2)
        vector = self.vector_out(vector_hidden * gate)
        return scalar, vector.to(dtype=v.dtype)


class FactorizedDiTBlock(nn.Module):
    def __init__(
        self,
        scalar_width: int,
        vector_width: int,
        heads: int,
        ffn_multiplier: int,
        dropout: float,
        ffn_norm_source: str = "post_adaln",
    ) -> None:
        super().__init__()
        if ffn_norm_source not in ("post_adaln", "pre_adaln"):
            raise ValueError("ffn_norm_source must be 'post_adaln' or 'pre_adaln'")
        self.spatial = ScalarVectorAttention(scalar_width, vector_width, heads, dropout)
        self.temporal = ScalarVectorAttention(scalar_width, vector_width, heads, dropout)
        self.ffn = ScalarVectorFFN(scalar_width, vector_width, ffn_multiplier)
        self.spatial_adaln = AdaLNZero(scalar_width, vector_width)
        self.temporal_adaln = AdaLNZero(scalar_width, vector_width)
        self.ffn_adaln = AdaLNZero(scalar_width, vector_width)
        self.ffn_norm_source = str(ffn_norm_source)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        condition: Tensor,
        batch: LatentBatch,
        block_groups: list[Tensor],
        block_samples: list[int],
    ) -> tuple[Tensor, Tensor]:
        return reference_block_forward(self, h, v, condition, batch, block_groups, block_samples)


class FrameDiTBlock(nn.Module):
    """History cross-attention, spatial groups, future time, then equivariant FFN."""

    def __init__(
        self,
        scalar_width: int,
        vector_width: int,
        heads: int,
        ffn_multiplier: int,
        dropout: float,
        *,
        local_geometry: bool = False,
        motion_context: bool = False,
    ) -> None:
        super().__init__()
        self.history = ScalarVectorCrossAttention(scalar_width, vector_width, heads, dropout)
        self.spatial = ScalarVectorAttention(scalar_width, vector_width, heads, dropout)
        self.temporal = ScalarVectorAttention(scalar_width, vector_width, heads, dropout)
        self.ffn = ScalarVectorFFN(scalar_width, vector_width, ffn_multiplier)
        self.history_adaln = AdaLNZero(scalar_width, vector_width)
        self.spatial_adaln = AdaLNZero(scalar_width, vector_width)
        self.temporal_adaln = AdaLNZero(scalar_width, vector_width)
        self.ffn_adaln = AdaLNZero(scalar_width, vector_width)
        self.history_time_bias = nn.Linear(4, heads, bias=False)
        self.temporal_time_bias = nn.Linear(4, heads, bias=False)
        self.local_geometry = (
            LocalGeometryMessage(scalar_width, vector_width)
            if bool(local_geometry) else None
        )
        self.motion_context = (
            MotionContextInjector(scalar_width, vector_width)
            if bool(motion_context) else None
        )
        if motion_context:
            self.history_time_decay_raw = nn.Parameter(torch.full((heads,), -4.0))
            self.temporal_time_decay_raw = nn.Parameter(torch.full((heads,), -4.0))
        else:
            self.register_parameter("history_time_decay_raw", None)
            self.register_parameter("temporal_time_decay_raw", None)
