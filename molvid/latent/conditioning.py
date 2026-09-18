"""Observed-prefix conditions and independent online history-corruption views."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor

from ..geometry.coordinates import compute_masked_centroid_origin
from .types import FRAME_COUNT, LatentBatch, LatentFields

def _validate_frame_mask(frame_mask: Tensor, *, batch_size: int, frames: int) -> Tensor:
    mask = torch.as_tensor(frame_mask, dtype=torch.bool)
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, frames):
        raise ValueError(f"frame_mask must have shape [{batch_size},{frames}]")
    for sample in range(batch_size):
        valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if valid.numel() and not torch.equal(valid, torch.arange(valid.numel(), device=mask.device)):
            raise ValueError("frame_mask must contain a valid prefix for each sample")
    return mask


def frame_prefix_observation_mask(frame_mask: Tensor, history_frames: int) -> Tensor:
    """Return the observed prefix for H=0,4,8 without touching future data."""

    mask = torch.as_tensor(frame_mask, dtype=torch.bool)
    if mask.ndim != 2 or mask.shape[1] != FRAME_COUNT:
        raise ValueError(f"frame_mask must have shape [B,{FRAME_COUNT}]")
    history = int(history_frames)
    if history not in (0, 4, 8):
        raise ValueError("v1 supports only H=0, H=4, or H=8")
    for sample in range(mask.shape[0]):
        valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
        if valid.numel() and not torch.equal(
            valid, torch.arange(valid.numel(), device=mask.device)
        ):
            raise ValueError("frame_mask must contain a valid prefix for each sample")
    if torch.any(mask.sum(dim=1) < history):
        raise ValueError("history length exceeds the valid frame count of a sample")
    prefix = torch.arange(FRAME_COUNT, device=mask.device).view(1, -1) < history
    return mask & prefix


@dataclass(frozen=True)
class ObservationCondition:
    frame_observation_mask: Tensor
    latent_observation_mask: Tensor
    sample_origin: Tensor
    history_frames: int

    def contract(self) -> dict[str, Any]:
        return {
            "history_frames": int(self.history_frames),
            "frame_observation_shape": list(self.frame_observation_mask.shape),
            "latent_observation_shape": list(self.latent_observation_mask.shape),
            "origin_shape": list(self.sample_origin.shape),
            "origin_source": "observed_frame0_centroid" if self.history_frames else "fixed_zero",
        }


def _observed_origins(
    coordinates: Optional[Tensor],
    batch: LatentBatch,
    frame_observation_mask: Tensor,
    history_frames: int,
    atom_mask: Optional[Tensor],
) -> Tensor:
    origin = batch.sample_origin.new_zeros((batch.batch_size, 3))
    if history_frames == 0:
        return origin
    if coordinates is None:
        raise ValueError("coordinates are required only to derive the observed frame-0 origin for H>0")
    x = torch.as_tensor(coordinates, device=batch.state_h.device, dtype=batch.state_h.dtype)
    if x.ndim != 3 or x.shape[0] != FRAME_COUNT or x.shape[1] != batch.num_atoms or x.shape[2] != 3:
        raise ValueError("clean observed coordinates must have shape [T,N,3]")
    if not bool(frame_observation_mask[:, 0].all()):
        raise ValueError("H>0 requires an observed frame-0 for every sample")
    if atom_mask is None:
        raise ValueError("H>0 observation origin requires the static codec loss_mask")
    return compute_masked_centroid_origin(
        x,
        frame_mask=frame_observation_mask,
        abid=batch.abid,
        atom_mask=atom_mask,
    )


def build_observation_condition(
    batch: LatentBatch,
    *,
    history_frames: int,
    coordinates: Optional[Tensor] = None,
    frame_mask: Optional[Tensor] = None,
    observed_frame_mask: Optional[Tensor] = None,
    loss_mask: Optional[Tensor] = None,
) -> ObservationCondition:
    """Build block-aligned observation state without retaining target coordinates."""

    frames = int(batch.frame_time_ps.shape[1])
    source_mask = batch.block_frame_mask.new_ones((batch.batch_size, frames))
    if frame_mask is not None and observed_frame_mask is not None:
        raise ValueError("pass frame_mask or observed_frame_mask, not both")
    if frame_mask is not None:
        source_mask = _validate_frame_mask(
            torch.as_tensor(frame_mask, device=source_mask.device, dtype=torch.bool),
            batch_size=batch.batch_size,
            frames=frames,
        )
    else:
        source_mask.zero_()
        flat = batch.block_frame_mask.reshape(batch.batch_size, -1)
        source_mask[:, : min(frames, flat.shape[1])] = flat[:, :frames]
    if observed_frame_mask is not None:
        observed_frames = torch.as_tensor(
            observed_frame_mask, device=source_mask.device, dtype=torch.bool
        )
        if observed_frames.shape != source_mask.shape:
            raise ValueError("observed_frame_mask must have shape [B,T]")
        if torch.any(observed_frames & ~source_mask):
            raise ValueError("observed_frame_mask cannot include invalid source frames")
    else:
        observed_frames = frame_prefix_observation_mask(source_mask, history_frames)
    valid = batch.block_frame_mask
    observed_padded = observed_frames.new_zeros((batch.batch_size, batch.tokens * batch.ratio))
    observed_padded[:, :frames] = observed_frames
    observed_blocks = observed_padded.reshape(batch.batch_size, batch.tokens, batch.ratio)
    partial = (observed_blocks & valid).any(dim=-1) & ((~observed_blocks) & valid).any(dim=-1)
    if torch.any(partial):
        locations = torch.nonzero(partial, as_tuple=False).tolist()
        raise ValueError(
            "partial observed codec block is not supported; choose H=0,4,8 on a shared "
            f"R2/R4 boundary (locations={locations[:4]})"
        )
    latent_observed = (observed_blocks | ~valid).all(dim=-1) & valid.any(dim=-1)
    static_loss_mask = batch.loss_mask if loss_mask is None else loss_mask
    if static_loss_mask is not None:
        static_loss_mask = torch.as_tensor(
            static_loss_mask, device=batch.state_h.device, dtype=torch.bool
        ).flatten()
        if static_loss_mask.numel() != batch.num_atoms:
            raise ValueError("loss_mask must have shape [N]")
    origin = _observed_origins(
        coordinates,
        batch,
        observed_frames,
        int(history_frames),
        static_loss_mask,
    )
    return ObservationCondition(
        frame_observation_mask=observed_frames.detach().clone(),
        latent_observation_mask=latent_observed.detach().clone(),
        sample_origin=origin.detach().clone(),
        history_frames=int(history_frames),
    )


HISTORY_CORRUPTION_SCHEMA = "pvb.dit.architecture_sequential.v1.history_corruption.v1"
DEFAULT_PROBABILITIES = (0.50, 0.25, 0.25)
DEFAULT_SIGMAS_ANGSTROM = (0.0, 0.02, 0.05)


def _validate_history(history_frames: int) -> int:
    history = int(history_frames)
    if history not in (4, 8):
        raise ValueError("Round 4 history corruption supports only H4/H8")
    return history


def _validate_distribution(
    probabilities: Sequence[float], sigmas_angstrom: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    probabilities = tuple(float(value) for value in probabilities)
    sigmas = tuple(float(value) for value in sigmas_angstrom)
    if len(probabilities) != 3 or len(sigmas) != 3:
        raise ValueError("Round 4 requires exactly clean/0.02/0.05 corruption choices")
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("history-corruption probabilities must be finite and nonnegative")
    if sigmas != DEFAULT_SIGMAS_ANGSTROM:
        raise ValueError("Round 4 corruption sigmas are fixed at 0/0.02/0.05 Angstrom")
    if abs(sum(probabilities) - 1.0) > 1.0e-8:
        raise ValueError("history-corruption probabilities must sum to one")
    return probabilities, sigmas


def sample_history_sigmas(
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype,
    probabilities: Sequence[float] = DEFAULT_PROBABILITIES,
    sigmas_angstrom: Sequence[float] = DEFAULT_SIGMAS_ANGSTROM,
) -> Tensor:
    """Choose one fixed corruption level independently for each packed clip."""

    probabilities, sigmas = _validate_distribution(probabilities, sigmas_angstrom)
    if int(batch_size) < 1:
        raise ValueError("history corruption requires at least one sample")
    draw = torch.rand((int(batch_size),), device=device, dtype=dtype, generator=generator)
    first = probabilities[0]
    second = first + probabilities[1]
    choices = torch.tensor(sigmas, device=device, dtype=dtype)
    indices = torch.where(
        draw < first,
        torch.zeros_like(draw, dtype=torch.long),
        torch.where(draw < second, torch.ones_like(draw, dtype=torch.long), torch.full_like(draw, 2, dtype=torch.long)),
    )
    return choices.index_select(0, indices)


def history_frame_mask(batch: Any, history_frames: int) -> Tensor:
    """Return the valid observed physical-frame mask without reading future data."""

    history = _validate_history(history_frames)
    frame_mask = torch.as_tensor(batch.frame_mask, device=batch.x.device, dtype=torch.bool)
    if frame_mask.ndim != 2 or frame_mask.shape[1] != int(batch.x.shape[0]):
        raise ValueError("coordinate batch frame_mask must have shape [B,T]")
    prefix = torch.arange(int(batch.x.shape[0]), device=batch.x.device).view(1, -1) < history
    return frame_mask & prefix


def _atom_frame_mask(frame_mask: Tensor, abid: Tensor) -> Tensor:
    return frame_mask.index_select(0, torch.as_tensor(abid, device=frame_mask.device, dtype=torch.long)).transpose(0, 1)


@dataclass(frozen=True)
class HistoryCorruptionViews:
    """Independent coordinate views and their exact origin relationship."""

    condition_view: Any
    target_view: Any
    observed_frames: Tensor
    sigma_per_sample_angstrom: Tensor
    epsilon: Tensor
    applied_noise: Tensor
    clean_origin: Tensor
    condition_origin: Tensor
    target_gauge_translation: Tensor
    cache_policy: str = "disabled_online_corruption"
    schema: str = HISTORY_CORRUPTION_SCHEMA

    def metadata(self) -> dict[str, Any]:
        sigmas = self.sigma_per_sample_angstrom.detach().float()
        return {
            "schema": self.schema,
            "cache_policy": self.cache_policy,
            "sigmas_angstrom": [float(value) for value in sigmas.detach().cpu().tolist()],
            "clean_samples": int((sigmas == 0).sum().item()),
            "sigma_0_02_samples": int((sigmas == 0.02).sum().item()),
            "sigma_0_05_samples": int((sigmas == 0.05).sum().item()),
            "target_gauge": "clean_centered_fields_with_condition_observed_frame0_origin",
        }


def make_history_corruption_views(
    batch: Any,
    *,
    history_frames: int,
    sigma_per_sample_angstrom: Tensor,
    epsilon: Tensor,
) -> HistoryCorruptionViews:
    """Make no-inplace noisy-condition and clean-target coordinate views.

    ``epsilon`` is supplied by the caller so its RNG is visibly independent of
    RF tau/epsilon sampling and so equivariance checks can rotate that exact
    draw.  It has unit Gaussian scale before multiplication by the selected
    per-clip sigma.
    """

    history = _validate_history(history_frames)
    x = torch.as_tensor(batch.x)
    if x.ndim != 3 or x.shape[-1] != 3 or int(x.shape[0]) != 16:
        raise ValueError("Round 4 requires coordinate x with shape [16,N,3]")
    epsilon = torch.as_tensor(epsilon, device=x.device, dtype=x.dtype)
    if epsilon.shape != x.shape or not bool(torch.isfinite(epsilon).all()):
        raise ValueError("history-corruption epsilon must be finite and match x")
    sigmas = torch.as_tensor(sigma_per_sample_angstrom, device=x.device, dtype=x.dtype).flatten()
    if sigmas.numel() != int(batch.batch_size):
        raise ValueError("one history corruption sigma is required per packed sample")
    allowed = (sigmas == 0.0) | (sigmas == 0.02) | (sigmas == 0.05)
    if not bool(torch.all(allowed)):
        raise ValueError("history corruption sigma must be one of 0, 0.02, 0.05 Angstrom")
    observed_frames = history_frame_mask(batch, history)
    atom_observed = _atom_frame_mask(observed_frames, batch.abid)
    atom_sigma = sigmas.index_select(0, torch.as_tensor(batch.abid, device=x.device, dtype=torch.long))
    applied_noise = epsilon * atom_sigma.view(1, -1, 1)
    applied_noise = applied_noise * atom_observed.unsqueeze(-1).to(dtype=x.dtype)
    # Both ``replace`` calls allocate independent views.  Future coordinates in
    # condition_view are clean but never inserted into the observation fields.
    condition_view = replace(batch, x=x.clone() + applied_noise)
    target_view = replace(batch, x=x.clone())
    clean_origin = compute_masked_centroid_origin(
        target_view.x,
        frame_mask=target_view.frame_mask,
        abid=target_view.abid,
        atom_mask=target_view.loss_mask,
    )
    condition_origin = compute_masked_centroid_origin(
        condition_view.x,
        frame_mask=condition_view.frame_mask,
        abid=condition_view.abid,
        atom_mask=condition_view.loss_mask,
    )
    return HistoryCorruptionViews(
        condition_view=condition_view,
        target_view=target_view,
        observed_frames=observed_frames,
        sigma_per_sample_angstrom=sigmas,
        epsilon=epsilon,
        applied_noise=applied_noise,
        clean_origin=clean_origin,
        condition_origin=condition_origin,
        target_gauge_translation=condition_origin - clean_origin,
    )


def _expand_mask(mask: Tensor, value: Tensor) -> Tensor:
    return mask.reshape(mask.shape + (1,) * (value.ndim - 2))


def _merge_observed_fields(
    clean: LatentFields,
    condition: LatentFields,
    batch: LatentBatch,
) -> LatentFields:
    if batch.observed_mask is None:
        raise ValueError("history corruption requires an observation mask")
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    values = []
    for name in clean.names():
        target = getattr(clean, name)
        supplied = getattr(condition, name)
        if supplied.shape != target.shape:
            raise ValueError("condition/target latent fields disagree")
        values.append(torch.where(_expand_mask(observed, target), supplied, target))
    return LatentFields(*values)


@dataclass(frozen=True)
class HistoryConditionedLatents:
    """Clean target and noisy observed latent representations for one batch."""

    target: LatentBatch
    observed: LatentBatch
    observation: ObservationCondition
    views: HistoryCorruptionViews


def combine_clean_target_with_condition(
    clean_target: LatentBatch,
    condition_latent: LatentBatch,
    *,
    views: HistoryCorruptionViews,
    history_frames: int,
) -> HistoryConditionedLatents:
    """Put noisy observed fields into a clean target in the condition gauge.

    The clean target fields are already translation-free codec quantities.  We
    therefore re-express them by assigning the observed condition origin rather
    than re-encoding a globally translated target.  ``target_view`` remains the
    unmodified clean physical coordinate reference for future geometry loss.
    """

    history = _validate_history(history_frames)
    if clean_target.batch_size != condition_latent.batch_size or clean_target.tokens != condition_latent.tokens:
        raise ValueError("condition and clean target latent batches disagree")
    for name in ("token_mask", "block_frame_mask", "abid", "loss_mask"):
        first = getattr(clean_target, name)
        second = getattr(condition_latent, name)
        if first is not None and second is not None and not torch.equal(first, second):
            raise ValueError(f"condition and clean target differ at {name}")
    observation = build_observation_condition(
        clean_target,
        history_frames=history,
        coordinates=views.condition_view.x,
        frame_mask=views.condition_view.frame_mask,
        loss_mask=views.condition_view.loss_mask,
    )
    if not torch.equal(observation.frame_observation_mask, views.observed_frames):
        raise RuntimeError("condition observation mask disagrees with corruption view")
    # This is the explicit gauge re-expression: clean centred fields decode in
    # the same frame-0 origin that produced the noisy observed fields.
    target = clean_target.with_observation(
        observation.latent_observation_mask,
        sample_origin=observation.sample_origin,
    )
    if not torch.equal(target.sample_origin, views.condition_origin):
        raise RuntimeError("target origin does not use the noisy condition origin")
    observed = target.with_fields(
        _merge_observed_fields(target.fields, condition_latent.fields, target)
    ).zero_invalid()
    if not torch.equal(observed.sample_origin, target.sample_origin):
        raise RuntimeError("observation clamp origin diverged from target gauge")
    return HistoryConditionedLatents(
        target=target,
        observed=observed,
        observation=observation,
        views=views,
    )
