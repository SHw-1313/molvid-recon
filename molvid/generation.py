"""Observed-prefix trajectory generation and autoregressive continuation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .data.batch import ClipBatch
from .flow.sampling import generate_state_detail_latent
from .flow.source import build_observed_center
from .latent.adapter import StateDetailLatentAdapter
from .latent.conditioning import build_observation_condition
from .latent.statistics import LatentStatistics
from .training.batches import prepare_batch_then_to_device


def _block_positions(x: Tensor, block_id: Tensor) -> Tensor:
    """Recompute canonical per-block positions from supplied coordinates."""

    result = torch.empty_like(x)
    for identifier in torch.unique(block_id):
        members = block_id == identifier
        selected = x[:, members]
        result[:, members] = selected.mean(dim=1, keepdim=True).expand_as(selected)
    return result


def _observed_scaffold(
    template: ClipBatch, prefix_coordinates: Tensor, history_frames: int
) -> ClipBatch:
    """Replace every coordinate, including future, without reading template.x."""

    history = int(history_frames)
    if history not in (4, 8) or int(template.frames) != 16:
        raise ValueError("generation requires a 16-frame template and H=4 or H=8")
    prefix = torch.as_tensor(prefix_coordinates, device=template.block_id.device)
    if prefix.shape != (history, template.atype.numel(), 3):
        raise ValueError("observed prefix must have shape [H,N,3]")
    if not bool(torch.isfinite(prefix).all()):
        raise ValueError("observed prefix contains non-finite coordinates")
    if not bool(torch.all(template.frame_mask[:, :history])):
        raise ValueError("every sample must contain the supplied observed history")
    x = torch.cat((prefix, prefix[-1:].expand(16 - history, -1, -1)), dim=0)
    return replace(template, x=x, bpos=_block_positions(x, template.block_id))


@torch.no_grad()
def sample_clip(
    codec: nn.Module,
    model: nn.Module,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    *,
    template: ClipBatch,
    prefix_coordinates: Tensor,
    history_frames: int,
    steps: int,
    seed: int,
    codec_hash: str,
    data_hash: str,
    source_mode: str = "conditional",
    center_kind: str = "repeat_last_coordinate_encode",
) -> tuple[Tensor, dict[str, Any]]:
    """Generate one 16-frame clip using only the supplied observed prefix.

    The template supplies topology, masks and physical time. Its coordinate
    arrays are never read, so evaluation truth cannot enter the condition.
    """

    if source_mode not in ("conditional", "gaussian"):
        raise ValueError("source_mode must be conditional or gaussian")
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        raise ValueError("DiT has no parameters")
    device = parameter.device
    codec_parameter = next(iter(codec.parameters()), None)
    if codec_parameter is None or codec_parameter.device != device:
        raise ValueError("codec and DiT must be on the same device")
    if getattr(model, "adapter", None) is not adapter:
        raise ValueError("DiT and generation must share the same adapter")
    if statistics.state_h_mean.device != device:
        raise ValueError("latent statistics must be on the model device")
    if codec.training:
        raise ValueError("generation requires an eval-mode codec")
    scaffold = _observed_scaffold(template, prefix_coordinates, history_frames)
    coordinate = prepare_batch_then_to_device(codec, scaffold, device)
    # The historical rollout recomputed block centers on the model device.
    # CPU and CUDA reductions differ by a few ULPs on multi-atom blocks.
    coordinate = replace(
        coordinate, bpos=_block_positions(coordinate.x, coordinate.block_id)
    )
    latent = codec.encode(coordinate)
    packed = adapter.from_codec_latent(
        latent, codec_hash=codec_hash, data_hash=data_hash,
        origin_from_latent=True, loss_mask=coordinate.loss_mask,
    )
    observation = build_observation_condition(
        packed,
        history_frames=history_frames,
        coordinates=coordinate.x,
        frame_mask=coordinate.frame_mask,
        loss_mask=coordinate.loss_mask,
    )
    observed = packed.with_observation(
        observation.latent_observation_mask, sample_origin=observation.sample_origin
    )
    if source_mode == "conditional":
        center, _ = build_observed_center(
            center_kind,
            codec_model=codec,
            coordinate_batch=coordinate,
            target_batch=observed,
            adapter=adapter,
            statistics=statistics,
            history_frames=history_frames,
            codec_hash=codec_hash,
            data_hash=data_hash,
        )
    else:
        center = None
    generated, metadata = generate_state_detail_latent(
        model, adapter, observed, statistics,
        steps=steps, seed=seed, source_center=center, source_mode=source_mode,
    )
    decoded = codec.decode(generated).x_hat.float()
    if decoded.shape != coordinate.x.shape or not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("decoded generated clip is invalid")
    # Decoder reconstruction is not an exact boundary condition. The observed
    # physical prefix is clamped before its generated future is propagated.
    prediction = torch.cat((coordinate.x[:history_frames].float(), decoded[history_frames:]), dim=0)
    metadata = {
        **metadata,
        "history_frames": int(history_frames),
        "coordinate_prefix_clamp_exact": bool(
            torch.equal(prediction[:history_frames], coordinate.x[:history_frames].float())
        ),
        "conditioning": "observed_prefix_only",
    }
    return prediction, metadata


@torch.no_grad()
def rollout(
    codec: nn.Module,
    model: nn.Module,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    *,
    template: ClipBatch,
    prefix_coordinates: Tensor,
    seeds: Sequence[int],
    steps: int,
    codec_hash: str,
    data_hash: str,
    source_mode: str = "conditional",
    center_kind: str = "repeat_last_coordinate_encode",
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Continue 8-frame segments using only the previous generated segment."""

    if not seeds:
        raise ValueError("rollout requires at least one segment seed")
    if tuple(prefix_coordinates.shape) != (8, template.atype.numel(), 3):
        raise ValueError("rollout requires exactly eight observed atom frames")
    pieces = [prefix_coordinates.to(next(model.parameters()).device)]
    metadata: list[dict[str, Any]] = []
    for segment, seed in enumerate(seeds):
        clip, info = sample_clip(
            codec, model, adapter, statistics, template=template,
            prefix_coordinates=pieces[-1], history_frames=8, steps=steps,
            seed=int(seed), codec_hash=codec_hash, data_hash=data_hash,
            source_mode=source_mode, center_kind=center_kind,
        )
        pieces.append(clip[8:])
        metadata.append({**info, "segment": segment})
    return torch.cat(pieces, dim=0), metadata
