"""Observed-history corruption for Round 4 of the sequential DiT study.

The helper deliberately separates three objects that are easy to conflate:

* ``condition_view`` owns the online-noisy observed coordinates;
* ``target_view`` is an independent, clean physical-coordinate view; and
* ``target`` carries clean centred latent fields in the *condition* origin
  gauge, while ``observed`` replaces only its observed latent tokens with the
  corresponding condition fields.

This keeps the RF target and the Round-2 future-bond target clean, but makes
the observation clamp and conditional repeat-last source use exactly the
history that was supplied to the DiT.  No cache is exposed by this module:
each call is an online corruption draw.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
from torch import Tensor

from .state_detail_codec_v2 import compute_masked_centroid_origin
from .state_detail_latent_adapter import (
    DiTLatentBatch,
    LatentFieldSet,
    ObservationCondition,
    build_observation_condition,
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
    if probabilities != DEFAULT_PROBABILITIES:
        raise ValueError("Round 4 corruption probabilities are fixed at 50/25/25 percent")
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
    clean: LatentFieldSet,
    condition: LatentFieldSet,
    batch: DiTLatentBatch,
) -> LatentFieldSet:
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
    return LatentFieldSet(*values)


@dataclass(frozen=True)
class HistoryConditionedLatents:
    """Clean target and noisy observed latent representations for one batch."""

    target: DiTLatentBatch
    observed: DiTLatentBatch
    observation: ObservationCondition
    views: HistoryCorruptionViews


def combine_clean_target_with_condition(
    clean_target: DiTLatentBatch,
    condition_latent: DiTLatentBatch,
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


__all__ = [
    "DEFAULT_PROBABILITIES",
    "DEFAULT_SIGMAS_ANGSTROM",
    "HISTORY_CORRUPTION_SCHEMA",
    "HistoryConditionedLatents",
    "HistoryCorruptionViews",
    "combine_clean_target_with_condition",
    "history_frame_mask",
    "make_history_corruption_views",
    "sample_history_sigmas",
]
