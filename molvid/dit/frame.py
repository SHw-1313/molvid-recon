"""History-conditioned per-frame scalar/vector rectified-flow field."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..codec.motion_context import MotionContext
from ..latent.adapter import FrameLatentAdapter
from ..latent.types import FrameLatentBatch, HistoryMemory, ObservedContext, QuerySpec
from ..time import PhysicalTimeEmbedding, pairwise_time_features
from .backend import FactorizedLayout, block_spatial_attention, build_factorized_layout
from .blocks import FrameDiTBlock


def _safe_ids(value: Tensor, size: int) -> Tensor:
    return value.to(dtype=torch.long).remainder(int(size))


class FrameDiT(nn.Module):
    """Joint non-causal flow field over every requested future frame."""

    def __init__(
        self,
        *,
        adapter: FrameLatentAdapter,
        scalar_width: int = 256,
        vector_width: int = 128,
        depth: int = 4,
        heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        max_atom_type: int = 128,
        max_block_type: int = 256,
        max_component_type: int = 128,
        geometry_enabled: bool = False,
        motion_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.geometry_enabled = bool(geometry_enabled)
        self.motion_enabled = bool(motion_enabled)
        if min(self.scalar_width, self.vector_width, self.depth, self.heads) < 1:
            raise ValueError("FrameDiT widths/depth/heads must be positive")
        if self.geometry_enabled and self.depth < 4:
            raise ValueError("the G branch requires depth >= 4 for blocks 2 and 4")
        if self.scalar_width % self.heads or self.vector_width % self.heads:
            raise ValueError("FrameDiT widths must divide heads")
        if float(dropout) != 0.0:
            raise ValueError("Frame Joint v1 fixes dropout at zero")
        self.query_time = PhysicalTimeEmbedding(self.scalar_width)
        self.flow_time = nn.Sequential(
            nn.Linear(3, self.scalar_width), nn.SiLU(), nn.Linear(self.scalar_width, self.scalar_width)
        )
        self.reference_scalar = nn.Linear(adapter.codec_width, self.scalar_width)
        self.reference_vector = nn.Linear(adapter.codec_width, self.vector_width, bias=False)
        self.atom_embedding = nn.Embedding(int(max_atom_type), self.scalar_width)
        self.block_embedding = nn.Embedding(int(max_block_type), self.scalar_width)
        self.component_embedding = nn.Embedding(int(max_component_type), self.scalar_width)
        self.blocks = nn.ModuleList(
            [
                FrameDiTBlock(
                    self.scalar_width,
                    self.vector_width,
                    self.heads,
                    int(ffn_multiplier),
                    0.0,
                    local_geometry=self.geometry_enabled and index in {1, 3},
                    motion_context=self.motion_enabled,
                )
                for index in range(self.depth)
            ]
        )
        # History cross-attention uses scalar queries and memory vector
        # values; its query-vector normalization is intentionally outside the
        # computation graph.  Do not leave that dead affine scale trainable.
        for block in self.blocks:
            block.history_adaln.vector_norm.scale.requires_grad_(False)
        self._layout_cache: FactorizedLayout | None = None

    def set_trainable(self, enabled: bool) -> None:
        self.requires_grad_(bool(enabled))
        for block in self.blocks:
            block.history_adaln.vector_norm.scale.requires_grad_(False)

    def _condition(
        self,
        noisy: FrameLatentBatch,
        context: ObservedContext,
        query: QuerySpec,
        flow_time: Tensor,
    ) -> Tensor:
        reference_time = context.latent.time_ps[:, 0]
        previous_time = context.latent.time_ps.gather(
            1, (context.latent.frame_mask.sum(dim=1).long() - 1).unsqueeze(1)
        ).squeeze(1)
        time = self.query_time(
            query.time_ps,
            query.frame_mask,
            reference_time_ps=reference_time,
            previous_time_ps=previous_time,
        ).index_select(0, noisy.abid).transpose(0, 1)
        s = torch.as_tensor(flow_time, device=noisy.h.device, dtype=noisy.h.dtype).reshape(noisy.batch_size)
        flow_features = torch.stack((s, torch.sin(torch.pi * s), torch.cos(torch.pi * s)), dim=-1)
        flow = self.flow_time(flow_features).index_select(0, noisy.abid).unsqueeze(0)
        reference_h = context.latent.h[0]
        identity = (
            self.atom_embedding(_safe_ids(noisy.atom_type, self.atom_embedding.num_embeddings))
            + self.block_embedding(_safe_ids(noisy.block_type, self.block_embedding.num_embeddings))
            + self.component_embedding(_safe_ids(noisy.component_id, self.component_embedding.num_embeddings))
        ).unsqueeze(0)
        return time + flow + identity + self.reference_scalar(reference_h).unsqueeze(0)

    def forward(
        self,
        noisy: FrameLatentBatch,
        *,
        context: ObservedContext,
        query: QuerySpec,
        history_memory: HistoryMemory,
        flow_time: Tensor,
        motion_context: MotionContext | None = None,
    ) -> FrameLatentBatch:
        if noisy.time_ps.shape != query.time_ps.shape or (
            noisy.frame_mask is not query.frame_mask
            and not torch.equal(noisy.frame_mask, query.frame_mask)
        ):
            raise ValueError("noisy future latent and QuerySpec disagree")
        if noisy.topology is not context.topology and noisy.topology.contract() != context.topology.contract():
            raise ValueError("observed and query topology differ")
        if self.motion_enabled != (motion_context is not None):
            raise ValueError("motion context presence must match the explicit M switch")
        h, v = self.adapter.project_inputs(noisy)
        condition = self._condition(noisy, context, query, flow_time)
        h = h + condition
        reference_v = self.reference_vector(context.latent.v[0]).unsqueeze(0)
        v = v + reference_v
        layout = self._layout_cache
        if layout is None or not layout.matches(noisy):
            layout = build_factorized_layout(noisy)
            self._layout_cache = layout
        query_atom_mask = noisy.frame_mask.index_select(0, noisy.abid)
        history_atom_mask = history_memory.token_mask.index_select(0, noisy.abid)
        history_time = history_memory.representative_time_ps()
        cross_features = pairwise_time_features(query.time_ps, history_time).index_select(0, noisy.abid)
        temporal_features = pairwise_time_features(query.time_ps, query.time_ps).index_select(0, noisy.abid)
        memory_h = history_memory.h.permute(1, 0, 2)
        memory_v = history_memory.v.permute(1, 0, 2, 3)
        for block in self.blocks:
            if block.motion_context is not None:
                assert motion_context is not None
                motion_h, motion_v = block.motion_context(
                    motion_context, condition, context, query
                )
                h = h + motion_h
                v = v + motion_v
            h_norm, v_norm, h_gate, v_gate = block.history_adaln(h, v, condition)
            cross_bias = block.history_time_bias(cross_features).permute(0, 3, 1, 2)
            if block.history_time_decay_raw is not None:
                cross_distance = (
                    (query.time_ps[:, :, None] - history_time[:, None, :]).abs()
                    / 100.0
                ).sqrt().index_select(0, noisy.abid)
                cross_bias = cross_bias - F.softplus(
                    block.history_time_decay_raw
                ).view(1, -1, 1, 1) * cross_distance.unsqueeze(1)
            cross_h, cross_v = block.history(
                h_norm.permute(1, 0, 2),
                v_norm.permute(1, 0, 2, 3),
                memory_h,
                memory_v,
                key_mask=history_atom_mask,
                query_mask=query_atom_mask,
                pair_bias=cross_bias,
            )
            h = h + torch.tanh(h_gate) * cross_h.permute(1, 0, 2)
            v = v + torch.tanh(v_gate).unsqueeze(-2) * cross_v.permute(1, 0, 2, 3)

            h_norm, v_norm, h_gate, v_gate = block.spatial_adaln(h, v, condition)
            spatial_h, spatial_v = block_spatial_attention(h_norm, v_norm, noisy, block.spatial, layout)
            h = h + torch.tanh(h_gate) * spatial_h
            v = v + torch.tanh(v_gate).unsqueeze(-2) * spatial_v
            if block.local_geometry is not None:
                local_h, local_v = block.local_geometry(h, v, noisy, context)
                h = h + local_h
                v = v + local_v

            h_norm, v_norm, h_gate, v_gate = block.temporal_adaln(h, v, condition)
            temporal_bias = block.temporal_time_bias(temporal_features).permute(0, 3, 1, 2)
            if block.temporal_time_decay_raw is not None:
                temporal_distance = (
                    (query.time_ps[:, :, None] - query.time_ps[:, None, :]).abs()
                    / 100.0
                ).sqrt().index_select(0, noisy.abid)
                temporal_bias = temporal_bias - F.softplus(
                    block.temporal_time_decay_raw
                ).view(1, -1, 1, 1) * temporal_distance.unsqueeze(1)
            temporal_h, temporal_v = block.temporal(
                h_norm.permute(1, 0, 2),
                v_norm.permute(1, 0, 2, 3),
                query_atom_mask,
                query_mask=query_atom_mask,
                pair_bias=temporal_bias,
            )
            h = h + torch.tanh(h_gate) * temporal_h.permute(1, 0, 2)
            v = v + torch.tanh(v_gate).unsqueeze(-2) * temporal_v.permute(1, 0, 2, 3)

            h_norm, v_norm, h_gate, v_gate = block.ffn_adaln(h, v, condition)
            ffn_h, ffn_v = block.ffn(h_norm, v_norm)
            h = h + torch.tanh(h_gate) * ffn_h
            v = v + torch.tanh(v_gate).unsqueeze(-2) * ffn_v
        atom_mask = noisy.atom_frame_mask()
        h = h * atom_mask.unsqueeze(-1)
        v = v * atom_mask.unsqueeze(-1).unsqueeze(-1)
        return self.adapter.project_outputs(noisy, h, v)

    def contract(self) -> dict[str, object]:
        base: dict[str, object] = {
            "schema_version": "molvid.frame_joint.dit.v1",
            "adapter": self.adapter.contract(),
            "scalar_width": self.scalar_width,
            "vector_width": self.vector_width,
            "depth": self.depth,
            "heads": self.heads,
            "attention_order": ["history_cross", "spatial_groups", "future_temporal", "equivariant_ffn"],
            "future_attention": "bidirectional_noncausal",
            "physical_time_bias": True,
            "future_fields": ["h", "v"],
        }
        if not self.geometry_enabled and not self.motion_enabled:
            return base
        base.update({
            "schema_version": "molvid.frame_joint.dit.v2",
            "geometry_enabled": self.geometry_enabled,
            "motion_enabled": self.motion_enabled,
            "local_geometry_blocks_one_based": [2, 4] if self.geometry_enabled else [],
            "local_geometry": (
                self.blocks[1].local_geometry.contract()
                if self.geometry_enabled and self.blocks[1].local_geometry is not None
                else None
            ),
            "motion_context": (
                self.blocks[0].motion_context.contract()
                if self.motion_enabled and self.blocks[0].motion_context is not None
                else None
            ),
            "time_decay": (
                "learned_residual_minus_softplus(lambda_head)*sqrt(abs_dt_over_100ps)"
                if self.motion_enabled else None
            ),
        })
        return base
