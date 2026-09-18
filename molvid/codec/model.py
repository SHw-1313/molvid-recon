"""Current TorchMD plus block-local state/detail trajectory codec."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import nn

from ..data.batch import ClipBatch
from ..geometry.coordinates import center_coordinates
from ..geometry.types import StaticTopologyMetadata
from ..spatial.encoder import FrameEncoder
from .heads import CoordinateVectorStem, EquivariantCoordinateHead
from .state_detail import MatchedPoolingCodec, StateDetailCodec
from .types import RATIO_FOR_MODE, STATE_DETAIL_MODES, MatchedPoolingLatent, StateDetailLatent


class TrajectoryCodec(nn.Module):
    """Encode a centered clip and decode coordinates from structured latents."""

    def __init__(
        self,
        hidden_channels: int = 128,
        spatial_layers: int = 2,
        temporal_codec_mode: str = "ratio4_state_detail",
        num_rbf: int = 50,
        num_heads: int = 8,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        bond_construction: Mapping[str, Any] | str | None = None,
        topology_cache_capacity: int = 4096,
        topology_device_cache_capacity: int = 4096,
        distance_bond_min: float = 0.5,
        distance_bond_max: float = 2.2,
        distance_bond_max_num_neighbors: int = 64,
        distance_bond_cache_capacity: int = 4096,
        spatial_dtype: torch.dtype = torch.float32,
        freeze_frame_encoder: bool = False,
        frame_encoder_source_hash: str | None = None,
        coordinate_stem: str = "centered_vector",
    ) -> None:
        super().__init__()
        mode = str(temporal_codec_mode).lower()
        if mode not in STATE_DETAIL_MODES:
            raise ValueError(f"unsupported state/detail codec mode: {mode!r}")
        if coordinate_stem not in {"centered_vector", "none"}:
            raise ValueError("coordinate_stem must be centered_vector or none")
        if not isinstance(spatial_dtype, torch.dtype):
            raise TypeError("spatial_dtype must be a torch.dtype")
        self.temporal_codec_mode = mode
        self.temporal_ratio = RATIO_FOR_MODE[mode]
        self.coordinate_stem_mode = coordinate_stem
        self.frame_encoder = FrameEncoder(
            hidden_channels=hidden_channels,
            num_layers=spatial_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            bond_construction=bond_construction,
            topology_cache_capacity=topology_cache_capacity,
            topology_device_cache_capacity=topology_device_cache_capacity,
            distance_bond_min=distance_bond_min,
            distance_bond_max=distance_bond_max,
            distance_bond_max_num_neighbors=distance_bond_max_num_neighbors,
            distance_bond_cache_capacity=distance_bond_cache_capacity,
            dtype=spatial_dtype,
        )
        self.freeze_frame_encoder = bool(freeze_frame_encoder)
        self.frame_encoder_source_hash = frame_encoder_source_hash
        if self.freeze_frame_encoder:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad_(False)
            self.frame_encoder.eval()
        self.coordinate_stem = (
            CoordinateVectorStem(hidden_channels)
            if coordinate_stem == "centered_vector"
            else None
        )
        self.temporal_codec = (
            MatchedPoolingCodec(hidden_channels)
            if mode == "ratio4_matched_pooling"
            else StateDetailCodec(hidden_channels, mode=mode)
        )
        # Construct after the temporal banks: this is the old effective RNG order.
        self.coordinate_head = EquivariantCoordinateHead(hidden_channels)

    def prepare_batch(self, batch: ClipBatch) -> None:
        self.frame_encoder.prepare_batch(batch)

    def prepare_distance_bonds(
        self, references: Mapping[str, Mapping[str, Any]], *, device: torch.device
    ) -> list[dict[str, Any]]:
        return self.frame_encoder.prepare_distance_bonds(references, device=device)

    def encode(self, batch: ClipBatch) -> StateDetailLatent | MatchedPoolingLatent:
        centered, origin = center_coordinates(
            batch.x,
            frame_mask=batch.frame_mask,
            abid=batch.abid,
            atom_mask=batch.loss_mask,
        )
        features = self.frame_encoder(replace(batch, x=centered))
        if self.coordinate_stem is not None:
            coordinate_vectors = self.coordinate_stem(centered).to(dtype=features.v.dtype)
            features = replace(features, v=features.v + coordinate_vectors)
        return self.temporal_codec.encode(
            features.h,
            features.v,
            time_ps=batch.time_ps,
            frame_mask=batch.frame_mask,
            abid=batch.abid,
            sample_origin=origin,
            topology=StaticTopologyMetadata.from_batch(batch),
        )

    def decode(self, latent: StateDetailLatent | MatchedPoolingLatent):
        return self.temporal_codec.decode(latent, coordinate_head=self.coordinate_head)

    def forward(self, batch: ClipBatch):
        return self.decode(self.encode(batch))

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self.freeze_frame_encoder:
            self.frame_encoder.eval()
        return result



def build_codec(config: Mapping[str, Any]) -> TrajectoryCodec:
    """Build the current codec from its explicit constructor mapping."""

    return TrajectoryCodec(**dict(config))
