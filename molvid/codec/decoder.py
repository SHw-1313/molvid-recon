"""Joint temporal/local decoder from observed and generated frame features."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..dit.blocks import ScalarVectorAttention, ScalarVectorFFN
from ..equivariant import AxisPreservingLinear, SO3ChannelNorm
from ..geometry.coordinates import restore_origin
from ..latent.types import FrameLatentBatch, ObservedContext, QuerySpec
from ..time import PhysicalTimeEmbedding, pairwise_time_features
from .heads import EquivariantCoordinateHead


@dataclass(frozen=True)
class TrajectoryDecoderOutput:
    coordinates: Tensor
    centered_coordinates: Tensor
    h: Tensor
    v: Tensor
    frame_mask: Tensor


class CovalentMessage(nn.Module):
    """Equivariant local messages on the fixed, coordinate-independent bond graph."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.bond_embedding = nn.Embedding(2, 16)
        self.scalar_message = nn.Sequential(
            nn.Linear(2 * channels + 17, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, channels),
        )
        self.direction_gate = nn.Linear(2 * channels + 17, channels)
        self.vector_neighbor = AxisPreservingLinear(channels, channels)

    def forward(self, h: Tensor, v: Tensor, context: ObservedContext) -> tuple[Tensor, Tensor]:
        bonds = context.topology.covalent_bond_index
        if bonds.shape[1] == 0:
            return torch.zeros_like(h), torch.zeros_like(v)
        source = torch.cat((bonds[0], bonds[1]))
        destination = torch.cat((bonds[1], bonds[0]))
        bond_type = torch.cat(
            (context.topology.covalent_bond_type, context.topology.covalent_bond_type)
        ).clamp(0, 1)
        reference = context.reference_centered_coordinates().to(dtype=h.dtype)
        displacement = reference.index_select(0, source) - reference.index_select(0, destination)
        distance = torch.linalg.vector_norm(displacement.float(), dim=-1).to(dtype=h.dtype).clamp_min(1.0e-6)
        direction = displacement / distance.unsqueeze(-1)
        embedded_bond = self.bond_embedding(bond_type).to(dtype=h.dtype)
        edge_context = torch.cat(
            (
                h.index_select(1, source),
                h.index_select(1, destination),
                distance.view(1, -1, 1).expand(h.shape[0], -1, -1),
                embedded_bond.view(1, source.numel(), -1).expand(h.shape[0], -1, -1),
            ),
            dim=-1,
        )
        # Autocast may run the message MLPs in BF16 while the residual stream
        # remains FP32.  Accumulate graph messages in the residual dtype.
        scalar_message = self.scalar_message(edge_context).to(dtype=h.dtype)
        vector_message = self.vector_neighbor(v.index_select(1, source)).to(dtype=v.dtype)
        direction_message = (
            direction.view(1, -1, 3, 1)
            * self.direction_gate(edge_context).unsqueeze(2)
        ).to(dtype=v.dtype)
        vector_message = vector_message + direction_message
        scalar = torch.zeros_like(h)
        vector = torch.zeros_like(v)
        scalar.index_add_(1, destination, scalar_message)
        vector.index_add_(1, destination, vector_message)
        degree = h.new_zeros((h.shape[1],))
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=h.dtype))
        return scalar / degree.clamp_min(1).view(1, -1, 1), vector / degree.clamp_min(1).view(1, -1, 1, 1)


class TrajectoryDecoderBlock(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.scalar_norm = nn.LayerNorm(channels)
        self.vector_norm = SO3ChannelNorm(channels)
        self.temporal = ScalarVectorAttention(channels, channels, heads, 0.0)
        self.temporal_time_bias = nn.Linear(4, heads, bias=False)
        self.local = CovalentMessage(channels)
        self.ffn = ScalarVectorFFN(channels, channels, 2)
        self.temporal_scalar_gate = nn.Parameter(torch.zeros(channels))
        self.temporal_vector_gate = nn.Parameter(torch.zeros(channels))
        self.local_scalar_gate = nn.Parameter(torch.zeros(channels))
        self.local_vector_gate = nn.Parameter(torch.zeros(channels))
        self.ffn_scalar_gate = nn.Parameter(torch.zeros(channels))
        self.ffn_vector_gate = nn.Parameter(torch.zeros(channels))

    def forward(self, h: Tensor, v: Tensor, time_ps: Tensor, frame_mask: Tensor, context: ObservedContext) -> tuple[Tensor, Tensor]:
        abid = context.latent.abid
        atom_mask = frame_mask.index_select(0, abid)
        pair = pairwise_time_features(time_ps, time_ps).index_select(0, abid)
        pair_bias = self.temporal_time_bias(pair).permute(0, 3, 1, 2)
        temporal_h, temporal_v = self.temporal(
            self.scalar_norm(h).permute(1, 0, 2),
            self.vector_norm(v).permute(1, 0, 2, 3),
            atom_mask,
            query_mask=atom_mask,
            pair_bias=pair_bias,
        )
        h = h + torch.tanh(self.temporal_scalar_gate) * temporal_h.permute(1, 0, 2)
        v = v + torch.tanh(self.temporal_vector_gate).view(1, 1, 1, -1) * temporal_v.permute(1, 0, 2, 3)
        local_h, local_v = self.local(self.scalar_norm(h), self.vector_norm(v), context)
        h = h + torch.tanh(self.local_scalar_gate) * local_h
        v = v + torch.tanh(self.local_vector_gate).view(1, 1, 1, -1) * local_v
        ffn_h, ffn_v = self.ffn(self.scalar_norm(h), self.vector_norm(v))
        h = h + torch.tanh(self.ffn_scalar_gate) * ffn_h
        v = v + torch.tanh(self.ffn_vector_gate).view(1, 1, 1, -1) * ffn_v
        valid = atom_mask.transpose(0, 1)
        return h * valid.unsqueeze(-1), v * valid.unsqueeze(-1).unsqueeze(-1)


class TrajectoryDecoder(nn.Module):
    """Two-layer joint decoder over observed and generated physical frames."""

    def __init__(
        self,
        channels: int = 128,
        *,
        depth: int = 2,
        heads: int = 8,
        coordinate_head: EquivariantCoordinateHead | None = None,
    ) -> None:
        super().__init__()
        if int(depth) != 2:
            raise ValueError("Frame Joint v1 fixes decoder depth at two")
        if int(channels) % int(heads):
            raise ValueError("decoder channels must divide heads")
        self.channels = int(channels)
        self.time_embedding = PhysicalTimeEmbedding(channels, zero_output=True)
        self.blocks = nn.ModuleList([TrajectoryDecoderBlock(channels, heads) for _ in range(depth)])
        self.coordinate_head = coordinate_head or EquivariantCoordinateHead(channels)

    @classmethod
    def from_codec_head(cls, coordinate_head: nn.Module, *, channels: int = 128, heads: int = 8) -> "TrajectoryDecoder":
        if not isinstance(coordinate_head, EquivariantCoordinateHead):
            raise ValueError("decoder warm start requires EquivariantCoordinateHead")
        head = copy.deepcopy(coordinate_head)
        head.requires_grad_(True)
        return cls(channels, heads=heads, coordinate_head=head)

    def forward(
        self,
        context: ObservedContext,
        future: FrameLatentBatch | None = None,
        query: QuerySpec | None = None,
        *,
        return_all: bool = False,
    ) -> TrajectoryDecoderOutput:
        observed = context.latent
        if (future is None) != (query is None):
            raise ValueError("future latent and QuerySpec must be supplied together")
        if future is None:
            h, v = observed.h, observed.v
            time_ps, frame_mask = observed.time_ps, observed.frame_mask
            query_frames = 0
        else:
            assert query is not None
            if future.width != self.channels or future.topology.contract() != observed.topology.contract():
                raise ValueError("decoder future latent differs from observed feature contract")
            if future.time_ps.shape != query.time_ps.shape or not torch.equal(future.frame_mask, query.frame_mask):
                raise ValueError("decoder future latent and QuerySpec disagree")
            h = torch.cat((observed.h, future.h), dim=0)
            v = torch.cat((observed.v, future.v), dim=0)
            time_ps = torch.cat((observed.time_ps, query.time_ps), dim=1)
            frame_mask = torch.cat((observed.frame_mask, query.frame_mask), dim=1)
            query_frames = future.frames
        h = h + self.time_embedding(
            time_ps,
            frame_mask,
            reference_time_ps=observed.time_ps[:, 0],
        ).index_select(0, observed.abid).transpose(0, 1)
        for block in self.blocks:
            h, v = block(h, v, time_ps, frame_mask, context)
        centered = self.coordinate_head(h, v)
        coordinates = restore_origin(centered, context.sample_origin.to(dtype=centered.dtype), observed.abid)
        if query_frames and not return_all:
            coordinates = coordinates[-query_frames:]
            centered = centered[-query_frames:]
            frame_mask = frame_mask[:, -query_frames:]
        return TrajectoryDecoderOutput(
            coordinates=coordinates,
            centered_coordinates=centered,
            h=h,
            v=v,
            frame_mask=frame_mask,
        )
