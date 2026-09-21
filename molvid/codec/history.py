"""Trainable zero-preserving compression of observed frame features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..equivariant import AxisPreservingLinear
from ..latent.types import HistoryMemory, ObservedContext
from ..time import PhysicalTimeEmbedding, pairwise_time_features
from .haar import haar_lift
from .state_detail import StateDetailCodec


class TemporalDifferenceUpdate(nn.Module):
    """One same-atom temporal update driven only by feature differences."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("history channels must divide attention heads")
        self.channels, self.heads, self.head_width = int(channels), int(heads), int(channels // heads)
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.time_bias = nn.Linear(4, heads, bias=False)
        self.scalar_out = nn.Linear(channels, channels, bias=False)
        self.vector_out = AxisPreservingLinear(channels, channels)
        nn.init.zeros_(self.scalar_out.weight)
        nn.init.zeros_(self.vector_out.weight)

    def forward(self, h: Tensor, v: Tensor, time_ps: Tensor, frame_mask: Tensor, abid: Tensor) -> tuple[Tensor, Tensor]:
        frames, atoms, channels = h.shape
        scalar = self.norm(h).permute(1, 0, 2)
        q = self.q(scalar).reshape(atoms, frames, self.heads, self.head_width).permute(0, 2, 1, 3)
        k = self.k(scalar).reshape(atoms, frames, self.heads, self.head_width).permute(0, 2, 1, 3)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_width)
        pair = pairwise_time_features(time_ps, time_ps)
        bias = self.time_bias(pair).permute(0, 3, 1, 2).index_select(0, abid)
        logits = logits + bias
        valid = frame_mask.index_select(0, abid)
        logits = logits.masked_fill(~valid[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid[:, None, :, None].to(dtype=weights.dtype)
        scalar_heads = h.permute(1, 0, 2).reshape(atoms, frames, self.heads, self.head_width).permute(0, 2, 1, 3)
        scalar_mean = torch.matmul(weights, scalar_heads)
        scalar_difference = scalar_mean - scalar_heads * weights.sum(dim=-1, keepdim=True)
        scalar_difference = scalar_difference.permute(0, 2, 1, 3).reshape(atoms, frames, channels).permute(1, 0, 2)
        vector_heads = v.permute(1, 0, 2, 3).reshape(atoms, frames, 3, self.heads, self.head_width).permute(0, 3, 1, 2, 4)
        vector_mean = torch.einsum("ahij,ahjrc->ahirc", weights, vector_heads)
        vector_difference = vector_mean - vector_heads * weights.sum(dim=-1, keepdim=True).unsqueeze(-1)
        vector_difference = vector_difference.permute(0, 2, 3, 1, 4).reshape(atoms, frames, 3, channels).permute(1, 0, 2, 3)
        return h + self.scalar_out(scalar_difference), v + self.vector_out(vector_difference)


class HistoryEncoder(nn.Module):
    """Temporal update followed by complete R4 state/detail history groups."""

    def __init__(self, channels: int = 128, scalar_width: int = 256, vector_width: int = 128, heads: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.temporal = TemporalDifferenceUpdate(channels, heads=heads)
        self.state_detail = StateDetailCodec(channels, mode="ratio4_state_detail")
        # Frame Joint uses only the warm-started encoder/gate half.  Keeping
        # inverse-Haar decoder weights frozen makes their non-role explicit.
        for name in ("state_decode_h", "state_decode_v", "detail_decode_h", "detail_decode_v"):
            getattr(self.state_detail, name).requires_grad_(False)
        self.time_embedding = PhysicalTimeEmbedding(channels)
        self.detail_scalar_f = nn.Sequential(nn.Linear(3 * channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        nn.init.zeros_(self.detail_scalar_f[-1].weight)
        nn.init.zeros_(self.detail_scalar_f[-1].bias)
        self.detail_vector_norm = nn.Linear(channels, channels, bias=False)
        self.detail_vector_time = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.detail_vector_norm.weight)
        nn.init.zeros_(self.detail_vector_time.weight)
        self.scalar_memory = nn.Linear(2 * channels, scalar_width)
        self.vector_memory = AxisPreservingLinear(2 * channels, vector_width)

    def load_compatible_state_detail(self, source: nn.Module) -> tuple[str, ...]:
        if not isinstance(source, StateDetailCodec) or source.ratio != 4 or source.channels != self.channels:
            raise ValueError("history warm start requires a compatible R4 StateDetailCodec")
        target = self.state_detail.state_dict()
        state = source.state_dict()
        if list(target) != list(state):
            raise ValueError("history state/detail warm-start keys differ")
        self.state_detail.load_state_dict(state, strict=True)
        for name in ("state_decode_h", "state_decode_v", "detail_decode_h", "detail_decode_v"):
            getattr(self.state_detail, name).requires_grad_(False)
        return tuple(state)

    def set_trainable(self, enabled: bool) -> None:
        self.requires_grad_(bool(enabled))
        for name in ("state_decode_h", "state_decode_v", "detail_decode_h", "detail_decode_v"):
            getattr(self.state_detail, name).requires_grad_(False)

    def _full_group(self, h: Tensor, v: Tensor, time: Tensor, mask: Tensor, abid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lifted_h = haar_lift(h, 4, frame_mask=mask, abid=abid)
        lifted_v = haar_lift(v, 4, frame_mask=mask, abid=abid)
        assert lifted_h.detail is not None and lifted_v.detail is not None
        state_h = self.state_detail.state_encode_h(lifted_h.state)
        state_v = self.state_detail.state_encode_v(lifted_v.state)
        raw_h = lifted_h.detail.reshape(1, h.shape[1], 3 * self.channels)
        raw_v = lifted_v.detail.permute(0, 1, 3, 2, 4).reshape(1, h.shape[1], 3, 3 * self.channels)
        projected_h = self.state_detail.detail_encode_h(raw_h)
        projected_v = self.state_detail.detail_encode_v(raw_v)
        embedded = self.time_embedding(time, mask).mean(dim=1).index_select(0, abid).unsqueeze(0)
        context = torch.cat((projected_h, state_h, embedded), dim=-1)
        zero_context = torch.cat((torch.zeros_like(projected_h), state_h, embedded), dim=-1)
        detail_h = projected_h + self.detail_scalar_f(context) - self.detail_scalar_f(zero_context)
        vector_norm = projected_v.float().square().sum(dim=2).add(1.0e-6).sqrt().to(dtype=state_h.dtype)
        gate = torch.sigmoid(
            self.state_detail.detail_gate(state_h)
            + self.detail_vector_norm(vector_norm)
            + self.detail_vector_time(embedded)
        )
        detail_v = projected_v * gate.unsqueeze(2)
        return state_h, state_v, detail_h, detail_v

    def forward(self, context: ObservedContext) -> HistoryMemory:
        latent = context.latent
        h, v = self.temporal(latent.h, latent.v, latent.time_ps, latent.frame_mask, latent.abid)
        scalar_tokens: list[Tensor] = []
        vector_tokens: list[Tensor] = []
        detail_h_tokens: list[Tensor] = []
        detail_v_tokens: list[Tensor] = []
        token_masks: list[Tensor] = []
        group_times: list[Tensor] = []
        group_masks: list[Tensor] = []
        complete_prefix = 4 * (int(latent.frame_mask.sum(dim=1).min()) // 4)
        frame = 0
        while frame < latent.frames:
            # Mixed-length batches use the shortest complete prefix.  This is
            # one metadata decision per call; no partial four-frame group is
            # ever sent through Haar.
            full = frame + 4 <= complete_prefix
            if full:
                state_h, state_v, detail_h, detail_v = self._full_group(
                    h[frame:frame + 4], v[frame:frame + 4],
                    latent.time_ps[:, frame:frame + 4], latent.frame_mask[:, frame:frame + 4], latent.abid,
                )
                scalar_tokens.append(self.scalar_memory(torch.cat((state_h, detail_h), dim=-1)))
                vector_tokens.append(self.vector_memory(torch.cat((state_v, detail_v), dim=-1)))
                detail_h_tokens.append(detail_h)
                detail_v_tokens.append(detail_v)
                token_masks.append(latent.frame_mask[:, frame:frame + 4].all(dim=-1, keepdim=True))
                group_times.append(latent.time_ps[:, frame:frame + 4].unsqueeze(1))
                group_masks.append(latent.frame_mask[:, frame:frame + 4].unsqueeze(1))
                frame += 4
            else:
                state_h = self.state_detail.state_encode_h(h[frame:frame + 1])
                state_v = self.state_detail.state_encode_v(v[frame:frame + 1])
                detail_h = torch.zeros_like(state_h)
                detail_v = torch.zeros_like(state_v)
                scalar_tokens.append(self.scalar_memory(torch.cat((state_h, detail_h), dim=-1)))
                vector_tokens.append(self.vector_memory(torch.cat((state_v, detail_v), dim=-1)))
                detail_h_tokens.append(detail_h)
                detail_v_tokens.append(detail_v)
                token_masks.append(latent.frame_mask[:, frame:frame + 1])
                padded_time = latent.time_ps.new_zeros((latent.batch_size, 1, 4))
                padded_mask = latent.frame_mask.new_zeros((latent.batch_size, 1, 4))
                padded_time[:, 0, 0] = latent.time_ps[:, frame]
                padded_mask[:, 0, 0] = latent.frame_mask[:, frame]
                group_times.append(padded_time)
                group_masks.append(padded_mask)
                frame += 1
        return HistoryMemory(
            h=torch.cat(scalar_tokens, dim=0),
            v=torch.cat(vector_tokens, dim=0),
            token_mask=torch.cat(token_masks, dim=1),
            group_time_ps=torch.cat(group_times, dim=1),
            group_frame_mask=torch.cat(group_masks, dim=1),
            abid=latent.abid,
            detail_h=torch.cat(detail_h_tokens, dim=0),
            detail_v=torch.cat(detail_v_tokens, dim=0),
        )
