"""Evaluation orchestration; truth is never passed to the generator as a condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..data.batch import ClipBatch
from ..generation import rollout, sample_clip
from ..latent.adapter import StateDetailLatentAdapter
from ..latent.statistics import LatentStatistics
from ..training.batches import prepare_batch_then_to_device
from .geometry import _metrics, aligned_rmsf_metrics, dynamic_acf_metrics
from .latent import oracle_vs_generated


@dataclass(frozen=True)
class EvalConfig:
    history_frames: int = 8
    steps: int = 8
    seed: int = 0
    source_mode: str = "conditional"
    center_kind: str = "repeat_last_coordinate_encode"

    def __post_init__(self) -> None:
        if self.history_frames not in (4, 8):
            raise ValueError("generation evaluation supports H4 or H8")
        if self.steps not in (8, 16):
            raise ValueError("generation evaluation supports 8 or 16 Euler steps")
        if self.source_mode not in ("gaussian", "conditional"):
            raise ValueError("source_mode must be gaussian or conditional")


@dataclass(frozen=True)
class RolloutTrack:
    """Reference trajectory held by evaluation, never supplied to generation."""

    template: ClipBatch
    reference_coordinates: Tensor
    system: str
    replica: str

    def __post_init__(self) -> None:
        if self.reference_coordinates.ndim != 3 or self.reference_coordinates.shape[1:] != self.template.x.shape[1:]:
            raise ValueError("rollout reference must have shape [T,N,3]")
        if self.reference_coordinates.shape[0] < 16 or (self.reference_coordinates.shape[0] - 8) % 8:
            raise ValueError("rollout reference requires 8 plus whole 8-frame segments")


def trajectory_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    history_frames: int | None = None,
) -> dict[str, Any]:
    """Keep geometric and temporal forecast intervals explicit."""

    result = dict(_metrics(prediction, target, batch, history_frames=history_frames))
    if history_frames is None:
        result["rmsf"] = aligned_rmsf_metrics(prediction, target, batch)
        result["dynamic"] = dynamic_acf_metrics(prediction, target, batch)
        result["observation_boundary"] = {}
    else:
        temporal = result["temporal"]
        result["rmsf"] = {
            "observed": temporal["observed"]["rmsf"],
            "future": temporal["future"]["rmsf"],
            "full_diagnostic": temporal["full_diagnostic"]["rmsf"],
        }
        result["dynamic"] = {
            "observed": temporal["observed"]["dynamic"],
            "future": temporal["future"]["dynamic"],
            "full_diagnostic": temporal["full_diagnostic"]["dynamic"],
        }
        result["observation_boundary"] = result["boundary"]
    return result


@torch.no_grad()
def evaluate_codec(
    codec: nn.Module,
    batch: ClipBatch,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Measure the codec reconstruction floor independently of DiT sampling."""

    if codec.training:
        raise ValueError("evaluation requires an eval-mode codec")
    coordinate = prepare_batch_then_to_device(codec, batch, torch.device(device))
    decoded = codec.decode(codec.encode(coordinate)).x_hat.float()
    target = coordinate.x.to(dtype=decoded.dtype)
    return trajectory_metrics(decoded, target, coordinate)


@torch.no_grad()
def evaluate_generation(
    codec: nn.Module,
    model: nn.Module,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    batch: ClipBatch,
    *,
    config: EvalConfig,
    codec_hash: str,
    data_hash: str,
) -> dict[str, Any]:
    """Report codec-oracle and generated metrics separately on one truth clip."""

    if codec.training:
        raise ValueError("evaluation requires an eval-mode codec")
    device = next(model.parameters()).device
    coordinate = prepare_batch_then_to_device(codec, batch, device)
    reference = coordinate.x.float()
    oracle = codec.decode(codec.encode(coordinate)).x_hat.float()
    predicted, generation = sample_clip(
        codec, model, adapter, statistics,
        template=batch,
        prefix_coordinates=batch.x[:config.history_frames],
        history_frames=config.history_frames,
        steps=config.steps,
        seed=config.seed,
        codec_hash=codec_hash,
        data_hash=data_hash,
        source_mode=config.source_mode,
        center_kind=config.center_kind,
    )
    oracle_metrics = trajectory_metrics(
        oracle, reference, coordinate, history_frames=config.history_frames
    )
    generated_metrics = trajectory_metrics(
        predicted, reference, coordinate, history_frames=config.history_frames
    )
    return {
        **oracle_vs_generated(oracle_metrics, generated_metrics),
        "generation": generation,
    }


@torch.no_grad()
def evaluate_rollout(
    codec: nn.Module,
    model: nn.Module,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    track: RolloutTrack,
    *,
    seeds: tuple[int, ...],
    steps: int,
    codec_hash: str,
    data_hash: str,
    source_mode: str = "conditional",
    center_kind: str = "repeat_last_coordinate_encode",
) -> dict[str, Any]:
    """Evaluate generated-prefix continuation against held-out reference windows."""

    if len(seeds) != (track.reference_coordinates.shape[0] - 8) // 8:
        raise ValueError("seed count must match reference segment count")
    prediction, generation = rollout(
        codec, model, adapter, statistics,
        template=track.template,
        prefix_coordinates=track.reference_coordinates[:8],
        seeds=seeds, steps=steps, codec_hash=codec_hash, data_hash=data_hash,
        source_mode=source_mode, center_kind=center_kind,
    )
    reference = track.reference_coordinates.to(prediction)
    rows = []
    for segment in range(len(seeds)):
        start = segment * 8
        rows.append({
            "segment": segment,
            "global_frame_interval": [start, start + 16],
            "target_future_global_interval": [start + 8, start + 16],
            "metrics": trajectory_metrics(
                prediction[start:start + 16],
                reference[start:start + 16],
                track.template,
                history_frames=8,
            ),
            "generation": generation[segment],
        })
    return {
        "system": track.system,
        "replica": track.replica,
        "prefix_source": "own_previous_generated_eight",
        "rows": rows,
        "prediction": prediction,
    }
