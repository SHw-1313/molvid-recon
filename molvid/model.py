"""End-to-end Frame Joint v1 model assembled from shallow components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .codec.decoder import TrajectoryDecoder, TrajectoryDecoderOutput
from .codec.frame import FrozenFrameTeacher
from .codec.history import HistoryEncoder
from .codec.motion_context import MotionContext, MotionContextEncoder
from .dit.frame import FrameDiT
from .flow.objective import frame_endpoint_from_velocity
from .latent.adapter import FrameLatentAdapter
from .latent.statistics import FrameLatentStatistics
from .latent.types import FrameLatentBatch, HistoryMemory, ObservedContext, QuerySpec


@dataclass(frozen=True)
class FrameJointOutput:
    """One full-model pass, including only explicitly requested decoder branches."""

    velocity: FrameLatentBatch
    normalized_endpoint: FrameLatentBatch
    endpoint: FrameLatentBatch
    history_memory: HistoryMemory
    generated: TrajectoryDecoderOutput | None = None
    clean: TrajectoryDecoderOutput | None = None
    near: TrajectoryDecoderOutput | None = None


@dataclass(frozen=True)
class FrameJointCondition:
    """Observed-only condition reusable across flow evaluations."""

    history_memory: HistoryMemory
    motion_context: MotionContext | None


class FrameJointModel(nn.Module):
    """Frozen target teacher plus trainable history, flow field, and decoder.

    The teacher is deliberately part of the module tree so checkpoints and DDP
    expose one auditable model.  It is kept in eval mode and never participates
    in autograd.
    """

    def __init__(
        self,
        *,
        target_teacher: FrozenFrameTeacher,
        history_encoder: HistoryEncoder,
        dit: FrameDiT,
        decoder: TrajectoryDecoder,
        statistics: FrameLatentStatistics,
        motion_encoder: MotionContextEncoder | None = None,
    ) -> None:
        super().__init__()
        if statistics.width != history_encoder.channels:
            raise ValueError("frame statistics and history encoder widths differ")
        if statistics.width != dit.adapter.codec_width or statistics.width != decoder.channels:
            raise ValueError("frame statistics, DiT adapter, and decoder widths differ")
        self.target_teacher = target_teacher
        self.history_encoder = history_encoder
        self.dit = dit
        self.decoder = decoder
        self.motion_encoder = motion_encoder
        self.register_buffer("frame_h_mean", statistics.h_mean.detach().clone())
        self.register_buffer("frame_h_std", statistics.h_std.detach().clone())
        self.register_buffer("frame_v_rms", statistics.v_rms.detach().clone())
        self.statistics_provenance = dict(statistics.provenance)
        self.statistics_hash = statistics.hash
        self.target_teacher.requires_grad_(False)
        self.target_teacher.eval()

    @classmethod
    def from_codec(
        cls,
        codec: nn.Module,
        statistics: FrameLatentStatistics,
        *,
        scalar_width: int = 256,
        vector_width: int = 128,
        depth: int = 4,
        heads: int = 8,
        geometry_enabled: bool = False,
        motion_enabled: bool = False,
    ) -> "FrameJointModel":
        channels = int(statistics.width)
        if channels != 128:
            raise ValueError("Frame Joint v1 requires the 128-channel target codec")
        teacher = FrozenFrameTeacher.from_codec(codec)
        history = HistoryEncoder(
            channels=channels,
            scalar_width=scalar_width,
            vector_width=vector_width,
            heads=heads,
        )
        history.load_compatible_state_detail(codec.temporal_codec)
        adapter = FrameLatentAdapter(
            codec_width=channels,
            scalar_width=scalar_width,
            vector_width=vector_width,
        )
        dit = FrameDiT(
            adapter=adapter,
            scalar_width=scalar_width,
            vector_width=vector_width,
            depth=depth,
            heads=heads,
            geometry_enabled=geometry_enabled,
            motion_enabled=motion_enabled,
        )
        decoder = TrajectoryDecoder.from_codec_head(
            codec.coordinate_head,
            channels=channels,
            heads=heads,
        )
        return cls(
            target_teacher=teacher,
            history_encoder=history,
            dit=dit,
            decoder=decoder,
            statistics=statistics,
            motion_encoder=(
                MotionContextEncoder(channels, scalar_width, vector_width)
                if motion_enabled else None
            ),
        )

    @property
    def statistics(self) -> FrameLatentStatistics:
        value = FrameLatentStatistics(
            h_mean=self.frame_h_mean,
            h_std=self.frame_h_std,
            v_rms=self.frame_v_rms,
            provenance=self.statistics_provenance,
        )
        if value.hash != self.statistics_hash:
            raise RuntimeError("registered frame statistics no longer match their contract hash")
        return value

    def normalize(self, latent: FrameLatentBatch) -> FrameLatentBatch:
        if latent.width != self.frame_h_mean.numel():
            raise ValueError("frame statistics width differs from latent")
        h_mean = self.frame_h_mean.to(dtype=latent.h.dtype).reshape(1, 1, -1)
        h_std = self.frame_h_std.to(dtype=latent.h.dtype).reshape(1, 1, -1)
        v_rms = self.frame_v_rms.to(dtype=latent.v.dtype).reshape(1, 1, 1, -1)
        return latent.with_features(
            (latent.h - h_mean) / h_std,
            latent.v / v_rms,
            statistics_hash=self.statistics_hash,
        )

    def inverse(self, latent: FrameLatentBatch) -> FrameLatentBatch:
        if latent.width != self.frame_h_mean.numel():
            raise ValueError("frame statistics width differs from latent")
        h_mean = self.frame_h_mean.to(dtype=latent.h.dtype).reshape(1, 1, -1)
        h_std = self.frame_h_std.to(dtype=latent.h.dtype).reshape(1, 1, -1)
        v_rms = self.frame_v_rms.to(dtype=latent.v.dtype).reshape(1, 1, 1, -1)
        return latent.with_features(
            latent.h * h_std + h_mean,
            latent.v * v_rms,
            statistics_hash=self.statistics_hash,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_teacher.eval()
        return self

    def prepare_condition(self, context: ObservedContext) -> FrameJointCondition:
        """Compute observed-only condition once without changing gradients."""

        history = self.history_encoder(context)
        motion = self.motion_encoder(context) if self.motion_encoder is not None else None
        return FrameJointCondition(history_memory=history, motion_context=motion)

    def forward(
        self,
        noisy_future: FrameLatentBatch,
        *,
        context: ObservedContext,
        query: QuerySpec,
        flow_time: Tensor,
        clean_future: FrameLatentBatch | None = None,
        decode_generated: bool = False,
        decode_clean: bool = False,
        decode_near: bool = False,
        prepared_condition: FrameJointCondition | None = None,
    ) -> FrameJointOutput:
        condition = prepared_condition or self.prepare_condition(context)
        if self.dit.motion_enabled != (condition.motion_context is not None):
            raise ValueError("prepared condition disagrees with the model M switch")
        velocity = self.dit(
            noisy_future,
            context=context,
            query=query,
            history_memory=condition.history_memory,
            flow_time=flow_time,
            motion_context=condition.motion_context,
        )
        normalized_endpoint = frame_endpoint_from_velocity(noisy_future, velocity, flow_time)
        endpoint = self.inverse(normalized_endpoint)
        generated = self.decoder(context, endpoint, query) if decode_generated else None
        if (decode_clean or decode_near) and clean_future is None:
            raise ValueError("clean/near decoder branches require a clean future target")
        clean = self.decoder(context, clean_future, query) if decode_clean else None
        # Near-data consistency trains only the decoder.  Detaching the endpoint
        # makes that scope explicit without changing the generated branch.
        near = self.decoder(context, endpoint.detach(), query) if decode_near else None
        return FrameJointOutput(
            velocity=velocity,
            normalized_endpoint=normalized_endpoint,
            endpoint=endpoint,
            history_memory=condition.history_memory,
            generated=generated,
            clean=clean,
            near=near,
        )

    def contract(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "molvid.frame_joint.model.v1",
            "future_representation": ["h", "v"],
            "teacher_frozen": True,
            "history_ratio": 4,
            "history_tail": "uncompressed",
            "dit": self.dit.contract(),
            "decoder_depth": len(self.decoder.blocks),
            "statistics_hash": self.statistics_hash,
        }
        if self.dit.geometry_enabled or self.dit.motion_enabled:
            result["schema_version"] = "molvid.frame_joint.model.v2"
            result["geometry_enabled"] = self.dit.geometry_enabled
            result["motion_enabled"] = self.dit.motion_enabled
            result["motion_encoder"] = (
                self.motion_encoder.contract() if self.motion_encoder is not None else None
            )
        return result
