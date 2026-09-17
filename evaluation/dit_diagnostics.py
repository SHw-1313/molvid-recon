"""Per-sample diagnostics for the frozen state/detail DiT pilot.

This module is deliberately separate from ``evaluation.dit_evaluation``.  The
pilot evaluator reports one aggregate per batch; this module keeps sample and
system identity, makes horizon slicing explicit, and contains only
observation-derived controls.  It never changes the codec, RF objective, or
sampling implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from data.clip_dataset import ClipBatch
from evaluation.codec_evaluation import (
    _aligned_prediction,
    _frame_view,
    _metrics,
)
from module.latent_rectified_flow import (
    FIELD_NAMES,
    LatentFieldSet,
    apply_observation_clamp,
    four_field_loss,
    rectified_flow_interpolate,
    rectified_flow_velocity,
    sample_isotropic_noise,
)
from module.state_detail_latent_adapter import (
    DiTLatentBatch,
    LatentStatistics,
    StateDetailLatentAdapter,
)
from module.state_detail_codec_v2 import StateDetailLatent


DIAGNOSTICS_SCHEMA = "pvb.dit.state_detail.diagnostics.v1"
SAMPLE_ID_RE = re.compile(r"^(?P<system>.+)_(?P<replica>R[0-9]+)_w(?P<window>[0-9]+)$")
HISTORY_FRAMES = (0, 4, 8)
FORECAST_LENGTHS = (4, 8)
TAU_MIDPOINTS = (0.05, 0.25, 0.50, 0.75, 0.95)
PERTURBATION_SCALES = (0.0, 0.01, 0.05, 0.10)


def stable_seed(
    master_seed: int,
    sample_id: str,
    history_frames: int,
    draw_id: int,
    diagnostic_kind: str,
) -> int:
    """Derive a process/order-independent 31-bit seed from the diagnostic key."""

    payload = "\0".join(
        (
            str(int(master_seed)),
            str(sample_id),
            str(int(history_frames)),
            str(int(draw_id)),
            str(diagnostic_kind),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def parse_sample_id(sample_id: str) -> dict[str, Any]:
    match = SAMPLE_ID_RE.match(str(sample_id))
    if match is None:
        raise ValueError(f"unsupported validation sample id: {sample_id!r}")
    return {
        "sample_id": str(sample_id),
        "system": match.group("system"),
        "replica": match.group("replica"),
        "window": int(match.group("window")),
    }


def frame_intervals(history_frames: int, total_frames: int = 16) -> dict[str, Any]:
    history = int(history_frames)
    total = int(total_frames)
    if history not in HISTORY_FRAMES:
        raise ValueError("diagnostics support H=0, H=4, and H=8")
    if history > total:
        raise ValueError("history exceeds the coordinate sequence")
    return {
        "observed": [0, history],
        "future": [history, total],
        "boundary": None if history == 0 else [history - 1, history],
        "full_diagnostic": [0, total],
    }


def horizon_indices(
    history_frames: int, forecast_length: int, total_frames: int = 16
) -> tuple[int, ...]:
    intervals = frame_intervals(history_frames, total_frames)
    start = int(intervals["future"][0])
    stop = start + int(forecast_length)
    if forecast_length not in FORECAST_LENGTHS:
        raise ValueError("diagnostics support forecast lengths L=4 and L=8")
    if stop > int(intervals["future"][1]):
        raise ValueError("requested horizon is longer than the available future")
    return tuple(range(start, stop))


def _as_float(value: Any) -> float | None:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().item()
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def json_safe(value: Any) -> Any:
    """Convert report values at an explicit JSON/reporting boundary."""

    if isinstance(value, Tensor):
        if value.numel() == 1:
            scalar = value.detach().float().item()
            return scalar if math.isfinite(scalar) else None
        return value.detach().float().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return value


def _sample_atom_mask(batch: DiTLatentBatch) -> Tensor:
    return batch.state_atom_mask()


def _future_field_masks(batch: DiTLatentBatch) -> dict[str, Tensor]:
    if batch.observed_mask is None:
        raise ValueError("diagnostics require an observation mask")
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    return {
        name: value & ~observed for name, value in batch.field_masks().items()
    }


def _expand_mask(mask: Tensor, value: Tensor) -> Tensor:
    return mask.reshape(mask.shape + (1,) * (value.ndim - 2))


def observed_coordinate_baseline(
    batch: ClipBatch,
    history_frames: int,
    *,
    kind: str,
) -> tuple[Tensor | None, dict[str, Any]]:
    """Construct a coordinate-only control without reading future coordinates."""

    history = int(history_frames)
    if history not in HISTORY_FRAMES:
        raise ValueError("baseline supports H=0, H=4, and H=8")
    if kind not in ("coordinate_persistence", "coordinate_constant_velocity"):
        raise ValueError(f"unsupported coordinate baseline: {kind}")
    if history == 0:
        return None, {
            "applicable": False,
            "reason": "no observed frame exists for an observed-only coordinate baseline",
        }
    if batch.batch_size != 1:
        raise ValueError("coordinate baseline construction expects one sample")
    if history > batch.frames:
        raise ValueError("history exceeds batch frames")
    frame_mask = batch.frame_mask[0].to(dtype=torch.bool)
    if not bool(frame_mask[:history].all()):
        raise ValueError("observed coordinate baseline requires a valid observed prefix")

    coordinates = batch.x
    prediction = torch.zeros_like(coordinates)
    prediction[:history] = coordinates[:history]
    if kind == "coordinate_persistence":
        prediction[history:] = coordinates[history - 1].unsqueeze(0)
        return prediction, {
            "applicable": True,
            "source_interval": [0, history],
            "construction": "copy_last_observed_coordinate",
        }

    if history < 2:
        raise ValueError("constant-velocity baseline requires at least two observations")
    time_ps = batch.time_ps[0]
    delta = time_ps[history - 1] - time_ps[history - 2]
    if not bool(torch.isfinite(delta)) or float(delta) <= 0.0:
        raise ValueError("observed physical time interval must be positive")
    velocity = (coordinates[history - 1] - coordinates[history - 2]) / delta
    for frame in range(history, batch.frames):
        prediction[frame] = coordinates[history - 1] + velocity * (
            time_ps[frame] - time_ps[history - 1]
        )
    return prediction, {
        "applicable": True,
        "source_interval": [0, history],
        "construction": "two_observed_frames_with_physical_dt",
        "source_delta_time_ps": float(delta.detach().float().item()),
    }


def observed_coordinate_scaffold(batch: ClipBatch, history_frames: int) -> Tensor:
    """Return coordinates containing only the observed prefix and a fixed repeat future."""

    history = int(history_frames)
    if history not in HISTORY_FRAMES:
        raise ValueError("scaffold supports H=0, H=4, and H=8")
    if batch.batch_size != 1:
        raise ValueError("scaffold construction expects one sample")
    scaffold = torch.zeros_like(batch.x)
    if history:
        scaffold[:history] = batch.x[:history]
        scaffold[history:] = batch.x[history - 1].unsqueeze(0)
    return scaffold


def latent_block_state_persistence(
    batch: DiTLatentBatch,
    history_frames: int,
) -> tuple[LatentFieldSet, dict[str, Any]]:
    """Copy the last complete observed state token and use raw-zero detail in the future."""

    if batch.batch_size != 1:
        raise ValueError("latent persistence construction expects one sample")
    history = int(history_frames)
    if history not in HISTORY_FRAMES:
        raise ValueError("latent persistence supports H=0, H=4, and H=8")
    if batch.observed_mask is None:
        raise ValueError("latent persistence requires an observation mask")
    fields = batch.fields.clone()
    observed = batch.observed_mask[0]
    valid = batch.token_mask[0]
    observed_atom = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    future = batch.state_atom_mask() & ~observed_atom
    observed_tokens = torch.nonzero(observed & valid, as_tuple=False).flatten()
    if observed_tokens.numel() == 0:
        fields = LatentFieldSet.zeros_like(fields)
        return fields, {
            "applicable": False,
            "reason": "H=0 has no observed state token",
            "detail_space": "raw_codec_zero",
        }
    source = int(observed_tokens[-1].item())
    fields = LatentFieldSet(
        torch.where(
            _expand_mask(future, fields.state_h),
            fields.state_h[source : source + 1].expand_as(fields.state_h),
            fields.state_h,
        ),
        torch.where(
            _expand_mask(future, fields.detail_h),
            torch.zeros_like(fields.detail_h),
            fields.detail_h,
        ),
        torch.where(
            _expand_mask(future, fields.state_v),
            fields.state_v[source : source + 1].expand_as(fields.state_v),
            fields.state_v,
        ),
        torch.where(
            _expand_mask(future, fields.detail_v),
            torch.zeros_like(fields.detail_v),
            fields.detail_v,
        ),
    )
    return fields, {
        "applicable": True,
        "source_token": source,
        "source_interval": [0, history],
        "construction": "last_complete_observed_state_plus_raw_zero_detail",
        "detail_space": "raw_codec_zero",
    }


def standardized_raw_zero_detail(
    statistics: LatentStatistics,
    detail_h_like: Tensor,
    detail_v_like: Tensor,
) -> tuple[Tensor, Tensor]:
    """Represent raw codec detail zero in standardized scalar/vector fields."""

    detail_h_zero = -statistics._broadcast(
        statistics.detail_h_mean, detail_h_like
    ) / statistics._broadcast(statistics.detail_h_std, detail_h_like)
    detail_v_zero = torch.zeros_like(detail_v_like)
    return detail_h_zero, detail_v_zero


def _metric_for_frames(
    prediction: Tensor,
    target: Tensor,
    batch: ClipBatch,
    frames: Sequence[int],
) -> dict[str, Any]:
    selected = tuple(int(index) for index in frames)
    if not selected:
        return {"available": False, "frame_interval": None}
    index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
    view = _frame_view(batch, selected)
    result = _metrics(
        prediction.index_select(0, index),
        target.index_select(0, index),
        view,
        history_frames=0,
    )
    return {
        "available": True,
        "frame_interval": [selected[0], selected[-1] + 1],
        "metrics": result["future"],
        "temporal": result.get("temporal", {}).get("future", {}),
    }


def trajectory_metric_record(
    prediction: Tensor,
    target: Tensor,
    batch: ClipBatch,
    history_frames: int,
) -> dict[str, Any]:
    """Return full-future, L4/L8, observed, boundary and full diagnostic sections."""

    history = int(history_frames)
    if history not in HISTORY_FRAMES:
        raise ValueError("trajectory metrics support H=0, H=4, and H=8")
    result = _metrics(prediction, target, batch, history_frames=history)
    horizons: dict[str, Any] = {}
    for length in FORECAST_LENGTHS:
        try:
            horizons[f"L{length}"] = _metric_for_frames(
                prediction,
                target,
                batch,
                horizon_indices(history, length, int(prediction.shape[0])),
            )
        except ValueError as exc:
            horizons[f"L{length}"] = {
                "available": False,
                "reason": str(exc),
            }
    time_ps = torch.as_tensor(batch.time_ps[0], device=prediction.device)
    intervals = frame_intervals(history, int(prediction.shape[0]))
    time_intervals = {}
    for key, value in intervals.items():
        if value is None or int(value[0]) == int(value[1]):
            time_intervals[key] = None
        else:
            time_intervals[key] = [
                float(time_ps[int(value[0])].detach().float().item()),
                float(time_ps[int(value[1]) - 1].detach().float().item()),
            ]
    protocol = {
        "history_frames": history,
        "frame_intervals": intervals,
        "time_ps_intervals": time_intervals,
        "physical_time_units": "ps",
        "boundary_semantics": "transition H-1 -> H, not an isolated future frame",
        "align_mask": "per-frame Kabsch alignment",
        "loss_mask": "frame validity intersected with static loss_mask",
        "contact_cutoff_angstrom": 4.5,
        "contact_exclusion_rule": "exclude_covalent_bond_pairs_only",
        "dynamic_correlation_semantics": "generated-vs-target aligned velocity correlation",
    }
    return {
        "observed": result.get("observed", {}),
        "future": result.get("future", {}),
        "boundary": result.get("boundary", {}),
        "full_diagnostic": result.get("full_diagnostic", result.get("all_frames", {})),
        "temporal": result.get("temporal", {}),
        "horizons": horizons,
        "protocol": protocol,
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_path: Sequence[str] = ("metrics", "future"),
) -> dict[str, Any]:
    """Aggregate numeric metric rows sample-equally and system-equally."""

    def lookup(row: Mapping[str, Any]) -> Mapping[str, Any]:
        value: Any = row
        for key in metric_path:
            if not isinstance(value, Mapping):
                return {}
            value = value.get(key, {})
        return value if isinstance(value, Mapping) else {}

    def numeric_values(group: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
        values: dict[str, list[float]] = {}
        for row in group:
            for key, value in lookup(row).items():
                scalar = _as_float(value)
                if scalar is not None:
                    values.setdefault(str(key), []).append(scalar)
        return values

    def means(group: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        return {
            key: sum(values) / len(values)
            for key, values in numeric_values(group).items()
            if values
        }

    sample_mean = means(rows)
    by_system: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        system = str(row.get("system", "unknown"))
        by_system.setdefault(system, []).append(row)
    system_means = [means(group) for group in by_system.values()]
    keys = sorted({key for value in system_means for key in value})
    system_equal = {
        key: sum(float(value[key]) for value in system_means if key in value)
        / max(sum(1 for value in system_means if key in value), 1)
        for key in keys
    }
    return {
        "sample_equal": sample_mean,
        "system_equal": system_equal,
        "sample_count": len(rows),
        "system_count": len(by_system),
        "aggregation": "sample_equal_mean_then_system_equal_mean; contact_f1_is_macro",
        "metric_path": list(metric_path),
    }


def _pairwise_rmsd(first: Tensor, second: Tensor, mask: Tensor) -> float | None:
    valid = mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        return None
    count = int(valid.sum().item())
    return float(
        (first[valid] - second[valid]).square().sum().div(max(count, 1)).sqrt()
        .detach()
        .float()
        .item()
    )


def diversity_summary(
    samples: Tensor,
    batch: ClipBatch,
) -> dict[str, Any]:
    """Report raw and per-frame-aligned pairwise spread using xyz-sum RMSD."""

    if samples.ndim != 4 or samples.shape[-1] != 3:
        raise ValueError("samples must have shape [draw,T,N,3]")
    if samples.shape[1:] != batch.x.shape:
        raise ValueError("sample coordinates do not match the batch")
    atom_mask = (
        batch.frame_mask[0].to(device=samples.device, dtype=torch.bool).unsqueeze(-1)
        & batch.loss_mask.to(device=samples.device, dtype=torch.bool).unsqueeze(0)
    )
    raw_values: list[float] = []
    aligned_values: list[float] = []
    for first in range(samples.shape[0]):
        for second in range(first + 1, samples.shape[0]):
            raw = _pairwise_rmsd(samples[first], samples[second], atom_mask)
            if raw is not None:
                raw_values.append(raw)
            aligned, _ = _aligned_prediction(
                samples[second], samples[first], batch, frames=range(batch.frames)
            )
            aligned_value = _pairwise_rmsd(samples[first], aligned, atom_mask)
            if aligned_value is not None:
                aligned_values.append(aligned_value)
    return {
        "sample_count": int(samples.shape[0]),
        "pair_count": int(samples.shape[0] * (samples.shape[0] - 1) // 2),
        "pairwise_raw_rmsd": None if not raw_values else sum(raw_values) / len(raw_values),
        "pairwise_aligned_rmsd": None
        if not aligned_values
        else sum(aligned_values) / len(aligned_values),
        "rmsd_definition": "sqrt(sum_xyz_squared / valid_atom_count)",
        "primary_metric": "not_best_of_n",
    }


def _field_summary(value: Tensor, mask: Tensor) -> dict[str, Any]:
    expanded = _expand_mask(mask, value)
    selected = value.masked_select(expanded)
    if selected.numel() == 0:
        return {"valid_elements": 0, "mean": None, "std": None, "rms": None}
    selected = selected.float()
    return {
        "valid_elements": int(selected.numel()),
        "mean": float(selected.mean().item()),
        "std": float(selected.std(unbiased=False).item()),
        "rms": float(selected.square().mean().sqrt().item()),
    }


def latent_summary(
    fields: LatentFieldSet,
    masks: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Summarize scalar fields and vector norms without directional means."""

    result: dict[str, Any] = {}
    for name in FIELD_NAMES:
        value = getattr(fields, name)
        mask = masks[name]
        if value.ndim == 4:
            vector_norm = value.float().square().sum(dim=-2).sqrt()
            scalar = _field_summary(vector_norm, mask)
            # The field mask is token-shaped [K,N], while vector_norm is
            # channel-shaped [K,N,C].  Expand it at the reporting boundary so
            # quantiles are computed over valid vector channels, not by
            # accidentally aligning N with C.
            valid = vector_norm.masked_select(_expand_mask(mask, vector_norm)).float()
            if valid.numel():
                quantiles = torch.quantile(
                    valid, torch.tensor((0.0, 0.5, 0.9, 0.99, 1.0), device=valid.device)
                ).tolist()
                scalar["norm_quantiles"] = [float(item) for item in quantiles]
            scalar["vector_axis_mean_subtraction"] = False
        else:
            scalar = _field_summary(value, mask)
        result[name] = scalar
    state_rms = result.get("state_h", {}).get("rms")
    detail_rms = result.get("detail_h", {}).get("rms")
    result["scalar_detail_to_state_rms_ratio"] = (
        None
        if state_rms in (None, 0.0) or detail_rms is None
        else float(detail_rms) / float(state_rms)
    )
    return result


def latent_norm_comparison(
    reference: LatentFieldSet,
    generated: LatentFieldSet,
    masks: Mapping[str, Tensor],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in FIELD_NAMES:
        reference_value = getattr(reference, name)
        generated_value = getattr(generated, name)
        if reference_value.ndim == 4:
            reference_value = reference_value.float().square().sum(dim=-2).sqrt()
            generated_value = generated_value.float().square().sum(dim=-2).sqrt()
        else:
            reference_value = reference_value.float().abs()
            generated_value = generated_value.float().abs()
        mask = masks[name]
        reference_rms = _field_summary(reference_value, mask).get("rms")
        generated_rms = _field_summary(generated_value, mask).get("rms")
        result[name] = {
            "reference_rms": reference_rms,
            "generated_rms": generated_rms,
            "generated_to_reference_rms_ratio": None
            if reference_rms in (None, 0.0) or generated_rms is None
            else float(generated_rms) / float(reference_rms),
        }
    return result


def _zero_observed(fields: LatentFieldSet, batch: DiTLatentBatch) -> LatentFieldSet:
    if batch.observed_mask is None:
        raise ValueError("observation mask is required")
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    return LatentFieldSet(
        *(getattr(fields, name) * (~_expand_mask(observed, getattr(fields, name))).to(
            dtype=getattr(fields, name).dtype
        ) for name in FIELD_NAMES)
    )


def fixed_tau_flow_diagnostics(
    model: torch.nn.Module,
    batch: DiTLatentBatch,
    statistics: LatentStatistics,
    *,
    sample_id: str,
    history_frames: int,
    tau_values: Sequence[float] = TAU_MIDPOINTS,
    draw_ids: Sequence[int] = (0, 1),
    master_seed: int = 20260907,
    autocast_context=None,
) -> list[dict[str, Any]]:
    """Evaluate the production four-field RF loss at fixed tau/noise draws."""

    normalized = statistics.normalize(batch)
    rows: list[dict[str, Any]] = []
    context = autocast_context if autocast_context is not None else torch.no_grad
    for draw_id in draw_ids:
        seed = stable_seed(master_seed, sample_id, history_frames, draw_id, "rf_tau")
        try:
            generator = torch.Generator(device=normalized.state_h.device).manual_seed(seed)
        except (RuntimeError, TypeError):
            generator = torch.Generator().manual_seed(seed)
        noise = sample_isotropic_noise(normalized.fields, generator=generator)
        target = _zero_observed(
            LatentFieldSet(
                *(rectified_flow_velocity(getattr(normalized.fields, name), getattr(noise, name))
                  for name in FIELD_NAMES)
            ),
            normalized,
        )
        for tau_value in tau_values:
            tau = torch.full(
                (normalized.batch_size,),
                float(tau_value),
                device=normalized.state_h.device,
                dtype=normalized.state_h.dtype,
            )
            interpolated = LatentFieldSet(
                *(
                    rectified_flow_interpolate(
                        getattr(normalized.fields, name),
                        getattr(noise, name),
                        tau,
                        sample_ids=normalized.abid,
                    )
                    for name in FIELD_NAMES
                )
            )
            interpolated = apply_observation_clamp(
                interpolated, normalized.fields, normalized
            )
            with torch.no_grad():
                if autocast_context is None:
                    prediction = model(normalized.with_fields(interpolated), tau)
                else:
                    with autocast_context():
                        prediction = model(normalized.with_fields(interpolated), tau)
            loss = four_field_loss(prediction, target, normalized)
            rows.append(
                {
                    "sample_id": sample_id,
                    "history_frames": int(history_frames),
                    "draw_id": int(draw_id),
                    "seed": int(seed),
                    "tau": float(tau_value),
                    "fields": {
                        name: float(value.detach().float().item())
                        for name, value in loss.fields.items()
                    },
                    "total": float(loss.total.detach().float().item()),
                    "valid_elements": {
                        name: int(value) for name, value in loss.valid_elements.items()
                    },
                    "normalized_latent": True,
                }
            )
    return rows


def perturb_normalized_oracle(
    batch: DiTLatentBatch,
    statistics: LatentStatistics,
    *,
    scope: str,
    scale: float,
    seed: int,
) -> tuple[LatentFieldSet, dict[str, Any]]:
    """Perturb only valid unobserved normalized oracle fields, then inverse-normalize."""

    if scope not in ("state_only", "detail_only"):
        raise ValueError("perturbation scope must be state_only or detail_only")
    normalized = statistics.normalize(batch)
    try:
        generator = torch.Generator(device=normalized.state_h.device).manual_seed(int(seed))
    except (RuntimeError, TypeError):
        generator = torch.Generator().manual_seed(int(seed))
    noise = sample_isotropic_noise(normalized.fields, generator=generator)
    values = {name: getattr(normalized.fields, name).clone() for name in FIELD_NAMES}
    future_masks = _future_field_masks(normalized)
    actual: dict[str, float] = {}
    for name in FIELD_NAMES:
        active = name.startswith("state") if scope == "state_only" else name.startswith("detail")
        value = values[name]
        if active:
            delta = float(scale) * getattr(noise, name)
            value = value + delta * _expand_mask(future_masks[name], value)
            count = int(_expand_mask(future_masks[name], value).sum().item())
            actual[name] = (
                None
                if count == 0
                else float(
                    (delta * _expand_mask(future_masks[name], value))
                    .square()
                    .sum()
                    .div(count)
                    .sqrt()
                    .detach()
                    .float()
                    .item()
                )
            )
        else:
            actual[name] = 0.0
        values[name] = value
    fields = apply_observation_clamp(
        LatentFieldSet(*(values[name] for name in FIELD_NAMES)),
        normalized.fields,
        normalized,
    )
    raw = statistics.inverse_fields(fields)
    return raw, {
        "scope": scope,
        "scale": float(scale),
        "seed": int(seed),
        "actual_noise_rms": actual,
        "source": "normalized_oracle_future_only",
    }


def field_swap_fields(
    oracle_latent: StateDetailLatent,
    generated_latent: StateDetailLatent,
    batch: DiTLatentBatch,
    *,
    swap: str,
) -> LatentFieldSet:
    """Construct explicitly non-deployable oracle/generated field swaps."""

    if swap not in (
        "oracle_state_generated_detail",
        "generated_state_oracle_detail",
        "generated_state_raw_zero_detail",
    ):
        raise ValueError(f"unsupported field swap: {swap}")
    if batch.observed_mask is None:
        raise ValueError("field swaps require an observation mask")
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    future = _future_field_masks(batch)
    oracle = LatentFieldSet(
        oracle_latent.state_h,
        oracle_latent.detail_h,
        oracle_latent.state_v,
        oracle_latent.detail_v,
    )
    generated = LatentFieldSet(
        generated_latent.state_h,
        generated_latent.detail_h,
        generated_latent.state_v,
        generated_latent.detail_v,
    )
    values: dict[str, Tensor] = {}
    for name in FIELD_NAMES:
        if swap == "oracle_state_generated_detail":
            source = generated if name.startswith("detail") else oracle
        elif swap == "generated_state_oracle_detail":
            source = generated if name.startswith("state") else oracle
        else:
            source = generated if name.startswith("state") else None
        value = getattr(oracle, name).clone()
        if source is None:
            value = torch.zeros_like(value)
        else:
            value = torch.where(
                _expand_mask(future[name], value),
                getattr(source, name),
                value,
            )
        value = torch.where(_expand_mask(observed, value), getattr(oracle, name), value)
        values[name] = value
    return LatentFieldSet(*(values[name] for name in FIELD_NAMES))


@dataclass(frozen=True)
class DiagnosticRunContract:
    schema: str
    code_commit: str
    input_hash: str
    master_seed: int
    histories: tuple[int, ...]
    steps: tuple[int, ...]
    draws: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "code_commit": self.code_commit,
            "input_hash": self.input_hash,
            "master_seed": int(self.master_seed),
            "histories": list(self.histories),
            "sampling_steps": list(self.steps),
            "draw_ids": list(self.draws),
        }


__all__ = [
    "DIAGNOSTICS_SCHEMA",
    "DiagnosticRunContract",
    "FORECAST_LENGTHS",
    "HISTORY_FRAMES",
    "PERTURBATION_SCALES",
    "TAU_MIDPOINTS",
    "aggregate_rows",
    "diversity_summary",
    "field_swap_fields",
    "fixed_tau_flow_diagnostics",
    "frame_intervals",
    "horizon_indices",
    "json_safe",
    "latent_block_state_persistence",
    "latent_norm_comparison",
    "latent_summary",
    "observed_coordinate_baseline",
    "observed_coordinate_scaffold",
    "parse_sample_id",
    "perturb_normalized_oracle",
    "stable_seed",
    "standardized_raw_zero_detail",
    "trajectory_metric_record",
]
