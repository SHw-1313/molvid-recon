"""Joint multi-frame coordinate decoding for the PVB codec.

The temporal module produces invariant/vector features.  This file turns
those features into coordinates relative to an explicit frame-zero anchor and
optionally applies one latent-conditioned TorchMD spatial refinement call over
all requested frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from .temporal_codec import CausalTemporalDecoder, TemporalState
from .torchmd_et import TorchMD_VQ_ET


def _topology_field(topology: Any, name: str) -> Any:
    if isinstance(topology, Mapping):
        if name not in topology:
            raise ValueError(f"topology is missing required field {name!r}")
        return topology[name]
    if not hasattr(topology, name):
        raise ValueError(f"topology is missing required field {name!r}")
    return getattr(topology, name)


@dataclass
class CodecLatent:
    """Deterministic latent plus the structural information allowed to decode it."""

    z_h: Tensor
    z_v: Tensor
    latent_mask: Tensor
    latent_time_ps: Tensor
    x_anchor: Tensor
    topology: Any = None
    abid: Optional[Tensor] = None

    @classmethod
    def from_temporal_state(
        cls,
        state: TemporalState,
        x_anchor: Tensor,
        topology: Any = None,
    ) -> "CodecLatent":
        return cls(
            z_h=state.h,
            z_v=state.v,
            latent_mask=state.frame_mask,
            latent_time_ps=state.time_ps,
            x_anchor=x_anchor,
            topology=topology,
            abid=state.abid,
        )

    def as_temporal_state(self) -> TemporalState:
        abid = self.abid
        if abid is None:
            abid = torch.zeros(
                self.z_h.shape[1], device=self.z_h.device, dtype=torch.long
            )
        return TemporalState(
            h=self.z_h,
            v=self.z_v,
            frame_mask=self.latent_mask,
            time_ps=self.latent_time_ps,
            abid=abid,
        )


@dataclass
class CoordinateDecoderOutput:
    """Joint coordinate prediction and decoded temporal features."""

    x_coarse: Tensor
    x_hat: Tensor
    h: Tensor
    v: Tensor
    frame_mask: Tensor
    time_ps: Tensor

    @property
    def coordinates(self) -> Tensor:
        return self.x_hat


class _DisplacementHead(nn.Module):
    """Equivariant vector-to-coordinate head with invariant scalar gates."""

    def __init__(self, scalar_channels: int, vector_channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(scalar_channels, vector_channels)
        self.out = nn.Linear(vector_channels, 1, bias=False)

    def forward(self, h: Tensor, v: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(h)).unsqueeze(2)
        # Coordinates remain an FP32 geometry island under BF16 autocast.
        return self.out(v * gate).squeeze(-1).float()


def _prepare_refiner_graph(
    x_coarse: Tensor,
    x_anchor: Tensor,
    topology: Any,
) -> tuple[Tensor, ...]:
    """Expand one static topology into frame-isolated TorchMD graph inputs."""

    frames, atoms = int(x_coarse.shape[0]), int(x_coarse.shape[1])
    z = torch.as_tensor(_topology_field(topology, "z"), device=x_coarse.device, dtype=torch.long)
    b = torch.as_tensor(_topology_field(topology, "b"), device=x_coarse.device, dtype=torch.long)
    batch = torch.as_tensor(
        _topology_field(topology, "batch"), device=x_coarse.device, dtype=torch.long
    )
    edge_index = torch.as_tensor(
        _topology_field(topology, "edge_index"), device=x_coarse.device, dtype=torch.long
    )
    bond_type = torch.as_tensor(
        _topology_field(topology, "bond_type"), device=x_coarse.device, dtype=torch.long
    ).flatten()
    if z.numel() not in (atoms, frames * atoms):
        raise ValueError("topology z has an invalid atom count")
    if b.numel() not in (atoms, frames * atoms):
        raise ValueError("topology b has an invalid atom count")
    if batch.numel() not in (atoms, frames * atoms):
        raise ValueError("topology batch has an invalid atom count")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("topology edge_index must have shape [2, E]")
    if bond_type.numel() not in (edge_index.shape[1],):
        raise ValueError("topology bond_type must have one entry per edge")

    if z.numel() == atoms:
        z = z.unsqueeze(0).expand(frames, -1).reshape(-1)
    if b.numel() == atoms:
        b = b.unsqueeze(0).expand(frames, -1).reshape(-1)
    if batch.numel() == atoms:
        sample_count = int(batch.max().item()) + 1 if batch.numel() else 1
        frame_ids = torch.arange(frames, device=x_coarse.device).repeat_interleave(atoms)
        batch = batch.repeat(frames) + frame_ids * sample_count

    if edge_index.numel():
        if int(edge_index.max().item()) < atoms:
            offsets = torch.arange(frames, device=x_coarse.device, dtype=torch.long)
            offsets = offsets[:, None, None] * atoms
            edge_index = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            bond_type = bond_type.repeat(frames)
        elif int(edge_index.max().item()) >= frames * atoms:
            raise ValueError("topology edge_index contains an out-of-range atom")
        if torch.any(batch[edge_index[0]] != batch[edge_index[1]]):
            raise ValueError("topology contains a cross-frame or cross-sample edge")
    else:
        edge_index = edge_index.reshape(2, 0)

    x_flat = x_coarse.reshape(frames * atoms, 3)
    anchor_flat = x_anchor.unsqueeze(0).expand(frames, -1, -1).reshape(frames * atoms, 3)
    edge_vec_t = x_flat[edge_index[0]] - x_flat[edge_index[1]]
    edge_weight_t = torch.linalg.vector_norm(edge_vec_t, dim=-1)
    edge_vec_0 = anchor_flat[edge_index[0]] - anchor_flat[edge_index[1]]
    edge_weight_0 = torch.linalg.vector_norm(edge_vec_0, dim=-1)
    return (
        z,
        b,
        batch,
        edge_index,
        bond_type,
        edge_weight_0,
        edge_vec_0,
        edge_weight_t,
        edge_vec_t,
    )


class LatentConditionedSpatialRefiner(nn.Module):
    """One batched TorchMD call conditioned on decoded invariant features."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_rbf: int = 50,
        num_heads: int = 8,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_z: int = 128,
        max_b: int = 64,
        max_num_neighbors: int = 32,
        spatial_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if hidden_channels <= 1:
            raise ValueError("hidden_channels must exceed the one conditioning channel")
        self.condition = nn.Linear(scalar_channels, 1)
        self.spatial_encoder = spatial_encoder if spatial_encoder is not None else TorchMD_VQ_ET(
            hidden_channels=hidden_channels,
            extra_channels=1,
            num_layers=num_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_b=max_b,
            max_num_neighbors=max_num_neighbors,
            cross_attn=True,
        )
        self.displacement = nn.Linear(hidden_channels, 1, bias=False)
        self.calls = 0

    @staticmethod
    def _unpack(result: Any) -> tuple[Tensor, Tensor]:
        if isinstance(result, (tuple, list)):
            if len(result) < 2:
                raise ValueError("spatial refiner must return scalar and vector features")
            h, v = result[0], result[1]
        else:
            h = getattr(result, "h", getattr(result, "scalar", None))
            v = getattr(result, "v", getattr(result, "vector", None))
        if not isinstance(h, Tensor) or not isinstance(v, Tensor):
            raise ValueError("spatial refiner must return tensor scalar/vector features")
        return h, v

    def forward(
        self,
        x_coarse: Tensor,
        h: Tensor,
        v: Tensor,
        topology: Any,
        *,
        x_anchor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.calls += 1
        graph = _prepare_refiner_graph(x_coarse, x_anchor, topology)
        z, b, batch, edge_index, bond_type, weight_0, vec_0, weight_t, vec_t = graph
        frames, atoms = int(x_coarse.shape[0]), int(x_coarse.shape[1])
        condition = self.condition(h).reshape(frames * atoms, 1)
        result = self.spatial_encoder(
            z=z,
            b=b,
            pos=x_coarse.reshape(frames * atoms, 3),
            batch=batch,
            t=condition,
            edge_index=edge_index,
            edge_weight_0=weight_0,
            edge_vec_0=vec_0,
            edge_weight_t=weight_t,
            edge_vec_t=vec_t,
            bond_type=bond_type,
        )
        h_refined, v_refined = self._unpack(result)
        if h_refined.shape[0] != frames * atoms or v_refined.shape[:2] != (frames * atoms, 3):
            raise ValueError("spatial refiner returned invalid feature shapes")
        delta = self.displacement(v_refined).squeeze(-1)
        return (
            delta.reshape(frames, atoms, 3),
            h_refined.reshape(frames, atoms, -1),
            v_refined.reshape(frames, atoms, 3, -1),
        )


class JointMultiFrameDecoder(nn.Module):
    """Decode all requested frames jointly from one latent clip."""

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        *,
        temporal_decoder: Optional[CausalTemporalDecoder] = None,
        spatial_refiner: Optional[nn.Module] = None,
        temporal_layers: int = 1,
        num_heads: int = 4,
        window_size: Optional[int] = 8,
        time_scale_ps: float = 100.0,
    ) -> None:
        super().__init__()
        self.temporal_decoder = temporal_decoder if temporal_decoder is not None else CausalTemporalDecoder(
            scalar_channels,
            vector_channels,
            num_layers=temporal_layers,
            num_heads=num_heads,
            window_size=window_size,
            time_scale_ps=time_scale_ps,
        )
        self.displacement = _DisplacementHead(scalar_channels, vector_channels)
        self.spatial_refiner = spatial_refiner

    def forward(
        self,
        latent: CodecLatent,
        *,
        target_time_ps: Tensor,
        target_mask: Optional[Tensor] = None,
    ) -> CoordinateDecoderOutput:
        """Decode coordinates; the API intentionally has no target-coordinate input."""

        if latent.z_h.ndim != 3 or latent.z_v.ndim != 4:
            raise ValueError("latent features must be [L, N, C] and [L, N, 3, Cv]")
        if latent.z_v.shape[:2] != latent.z_h.shape[:2] or latent.z_v.shape[2] != 3:
            raise ValueError("latent vector features have an invalid shape")
        if latent.x_anchor.ndim != 2 or latent.x_anchor.shape[0] != latent.z_h.shape[1] or latent.x_anchor.shape[1] != 3:
            raise ValueError("x_anchor must have shape [N, 3]")
        state = latent.as_temporal_state()
        decoded = self.temporal_decoder(
            state,
            target_time_ps=target_time_ps,
            target_mask=target_mask,
        )
        delta = self.displacement(decoded.h, decoded.v).float()
        x_anchor = latent.x_anchor.to(device=delta.device, dtype=torch.float32)
        x_coarse = x_anchor.unsqueeze(0) + delta
        x_hat = x_coarse
        h_out, v_out = decoded.h, decoded.v
        if self.spatial_refiner is not None:
            if latent.topology is None:
                raise ValueError("a topology is required when spatial_refiner is enabled")
            refined = self.spatial_refiner(
                x_coarse,
                decoded.h,
                decoded.v,
                latent.topology,
                x_anchor=x_anchor,
            )
            if not isinstance(refined, (tuple, list)) or len(refined) < 3:
                raise ValueError("spatial_refiner must return delta, scalar, and vector features")
            x_hat = x_coarse + refined[0].float()
            h_out, v_out = refined[1], refined[2]

        valid = decoded.frame_mask.index_select(0, state.abid).transpose(0, 1)
        x_coarse = torch.where(valid.unsqueeze(-1), x_coarse, x_anchor.unsqueeze(0))
        x_hat = torch.where(valid.unsqueeze(-1), x_hat, x_anchor.unsqueeze(0))
        return CoordinateDecoderOutput(
            x_coarse=x_coarse,
            x_hat=x_hat,
            h=h_out,
            v=v_out,
            frame_mask=decoded.frame_mask,
            time_ps=decoded.time_ps,
        )


# Short aliases for downstream codec construction.
JointDecoder = JointMultiFrameDecoder
CoordinateDecoder = JointMultiFrameDecoder
