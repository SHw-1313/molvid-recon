"""A-only diagnostics for reassessing the frozen R4 source checkpoints.

This module deliberately lives outside the production DiT, RF, codec, and
trainer paths.  It adds only evaluation helpers: fixed-epsilon Euler sampling
for 8/16/32 steps, honest legacy-row aggregation, and metrics whose
applicability is explicit for degenerate trajectories.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from data.clip_dataset import ClipBatch
from evaluation.codec_evaluation import (
    _align_mask,
    _align_trajectory_to_first,
    _aligned_prediction,
    _aligned_rmsd,
    _bond_rmse,
    _frame_atom_mask,
    _frame_view,
    _mask,
    _nonbond_pairs,
    _rmsd,
)
from evaluation.dit_diagnostics import parse_sample_id
from module.latent_flow_source import combine_source_fields
from module.latent_rectified_flow import (
    FIELD_NAMES,
    apply_observation_clamp,
    sample_isotropic_noise,
)
from module.state_detail_latent_adapter import (
    DiTLatentBatch,
    LatentFieldSet,
    LatentStatistics,
    StateDetailLatentAdapter,
    contract_hash,
)


REASSESSMENT_SCHEMA = "pvb.dit.state_detail.source_reassessment.v2"
FIXED_EULER_STEPS = (8, 16, 32)
HISTORIES = (4, 8)
DRAW_IDS = (0, 1, 2, 3)
ROLLOUT_DRAW_IDS = (0, 1)
RATIO = 4
CONTACT_CUTOFF_ANGSTROM = 4.5
RMSF_NEAR_ZERO_ANGSTROM = 1.0e-5
ACF_STD_EPSILON = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reassessment_seed(
    master_seed: int,
    sample_id: str,
    history_frames: int,
    draw_id: int,
) -> int:
    """Return an order-independent epsilon seed.

    The key intentionally excludes arm, Euler steps, batch position, and
    checkpoint step.  Those choices are made by the caller after the epsilon
    draw is fixed, so source arms and step-count diagnostics are paired.
    """

    payload = "\0".join(
        (
            "dit-source-reassessment-epsilon-v2",
            str(int(master_seed)),
            str(sample_id),
            str(int(history_frames)),
            str(int(draw_id)),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def seed_contract(master_seed: int) -> dict[str, Any]:
    return {
        "master_seed": int(master_seed),
        "key_fields": ["sample_id", "history_frames", "draw_id"],
        "excluded_fields": ["arm", "steps", "batch_position", "checkpoint_step"],
        "schema": "sha256(v2 namespace | master | sample | H | draw)[:8] mod 2^31",
    }


def tensor_bytes_hash(fields: LatentFieldSet) -> str:
    digest = hashlib.sha256()
    for name in FIELD_NAMES:
        value = getattr(fields, name).detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json_float(value: Any) -> float | None:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().item()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean_mapping(values: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, list[float]] = defaultdict(list)
    def visit(value: Mapping[str, Any], prefix: str = "") -> None:
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, Mapping):
                visit(item, name)
                continue
            scalar = _json_float(item)
            if scalar is not None:
                result[name].append(scalar)

    for value in values:
        visit(value)
    return {key: sum(items) / len(items) for key, items in sorted(result.items()) if items}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(dict(value))
    return rows


def _legacy_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    raw_sample_id = row.get("sample_id")
    is_batch = isinstance(raw_sample_id, list)
    sample_id: str | None = None if is_batch else str(raw_sample_id)
    parsed: dict[str, Any] = {}
    if sample_id is not None:
        try:
            parsed = parse_sample_id(sample_id)
        except ValueError:
            parsed = {}
    generation = row.get("generation", {})
    generation = generation if isinstance(generation, Mapping) else {}
    diagnostic = row.get("diagnostic_metrics", {})
    diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
    return {
        "schema": f"{REASSESSMENT_SCHEMA}.legacy_row.v1",
        "arm": str(arm),
        "row_type": "main_batch" if is_batch else "subset_clip",
        "sample_id": raw_sample_id,
        "system": parsed.get("system"),
        "replica": parsed.get("replica"),
        "window": parsed.get("window"),
        "history_frames": row.get("history_frames"),
        "steps": row.get("steps"),
        "draw": row.get("draw_id"),
        "seed": generation.get("seed"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "future": dict(diagnostic.get("future", {}))
        if isinstance(diagnostic.get("future", {}), Mapping)
        else {},
        "diagnostic_metrics": diagnostic,
    }


def _nested_subset_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average draws within clip, clips within system, then systems."""

    by_clip: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        system = row.get("system")
        sample_id = row.get("sample_id")
        if not system or not isinstance(sample_id, str):
            raise ValueError("subset aggregation requires one parsed clip row per sample")
        by_clip[(str(system), sample_id)].append(row)
    clip_rows: list[dict[str, Any]] = []
    for (system, sample_id), clip_group in sorted(by_clip.items()):
        clip_rows.append(
            {
                "system": system,
                "sample_id": sample_id,
                "draw_count": len(clip_group),
                "future": _mean_mapping([row.get("future", {}) for row in clip_group]),
            }
        )
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        by_system[str(row["system"])].append(row)
    system_rows = [
        {
            "system": system,
            "clip_count": len(group),
            "future": _mean_mapping([row["future"] for row in group]),
        }
        for system, group in sorted(by_system.items())
    ]
    return {
        "aggregation": "draw_equal_then_clip_equal_then_system_equal",
        "row_count": len(rows),
        "draw_count": len({row.get("draw") for row in rows}),
        "clip_count": len(clip_rows),
        "system_count": len(system_rows),
        "system_equal": _mean_mapping([row["future"] for row in system_rows]),
        "clip_rows": clip_rows,
        "system_rows": system_rows,
    }


def reaggregate_legacy_jsonl(
    old_root: str | Path,
    *,
    checkpoint_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Re-index the old 52 batch and 128 clip rows without fabricating systems."""

    old_root = Path(old_root)
    normalized: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, Any]] = {}
    for arm, checkpoint_value in checkpoint_paths.items():
        checkpoint_path = Path(checkpoint_value)
        jsonl_path = old_root / arm / "generation_metrics.jsonl"
        checkpoint_hash = sha256_file(checkpoint_path)
        raw_rows = _read_jsonl(jsonl_path)
        rows = [
            _legacy_row(
                row,
                arm=arm,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
            )
            for row in raw_rows
        ]
        normalized.extend(rows)
        counts = {
            row_type: sum(row["row_type"] == row_type for row in rows)
            for row_type in ("main_batch", "subset_clip")
        }
        groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row["row_type"]),
                int(row["history_frames"]),
                int(row["steps"]),
            )
            groups[key].append(row)
        summaries: list[dict[str, Any]] = []
        for (row_type, history, steps), group in sorted(groups.items()):
            draws = sorted({row.get("draw") for row in group})
            if row_type == "main_batch":
                summaries.append(
                    {
                        "row_type": row_type,
                        "history_frames": history,
                        "steps": steps,
                        "draws": draws,
                        "aggregation": "batch_row_equal; system_equal_not_available",
                        "row_count": len(group),
                        "system_count": None,
                        "clip_count": None,
                        "system_equal": _mean_mapping([row["future"] for row in group]),
                        "system_fields": None,
                    }
                )
            else:
                subset = _nested_subset_aggregate(group)
                subset.update(
                    {
                        "row_type": row_type,
                        "history_frames": history,
                        "steps": steps,
                        "draws": draws,
                    }
                )
                summaries.append(subset)
        by_arm[str(arm)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "jsonl": str(jsonl_path),
            "row_counts": counts,
            "total_rows": len(rows),
            "groups": summaries,
        }
    expected_per_arm = {"main_batch": 52, "subset_clip": 128}
    actual = {
        row_type: sum(row["row_type"] == row_type for row in normalized)
        for row_type in expected_per_arm
    }
    actual_per_arm = {
        arm: dict(value["row_counts"])
        for arm, value in sorted(by_arm.items())
    }
    return {
        "schema": f"{REASSESSMENT_SCHEMA}.legacy_reaggregation.v1",
        "old_root": str(old_root),
        "expected_row_counts": {
            row_type: int(count) * len(by_arm)
            for row_type, count in expected_per_arm.items()
        },
        "expected_row_counts_per_arm": expected_per_arm,
        "actual_row_counts": actual,
        "actual_row_counts_per_arm": actual_per_arm,
        "row_count_check": bool(
            by_arm
            and all(counts == expected_per_arm for counts in actual_per_arm.values())
        ),
        "row_count_check_scope": "each checkpoint arm independently; global counts are also reported",
        "normalized_rows": normalized,
        "arms": by_arm,
        "missing_from_legacy": [
            "system_equal_for_main_batch_rows",
            "train_split_fixed_clip_evaluation",
            "8_step_and_32_step_fixed_epsilon_evaluation",
            "true_contact_occupancy_across_time",
            "short_rollout",
        ],
        "test_payload_opened": False,
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p90": None, "p95": None, "max": None}
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


def _safe_pearson(first: Tensor, second: Tensor, *, label: str) -> dict[str, Any]:
    first = first.detach().float().reshape(-1)
    second = second.detach().float().reshape(-1)
    if first.numel() == 0 or second.numel() == 0:
        return {"value": None, "applicable": False, "reason": f"{label}: no valid samples"}
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= ACF_STD_EPSILON:
        return {
            "value": None,
            "applicable": False,
            "reason": f"{label}: constant or near-constant signal (centered norm <= {ACF_STD_EPSILON:g})",
        }
    return {
        "value": float((first * second).sum().div(denominator)),
        "applicable": True,
        "reason": None,
    }


def _batch_field(batch: Any, name: str) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name)
    return getattr(batch, name, None)


def _contact_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor,
    frames: Sequence[int],
    *,
    cutoff: float = CONTACT_CUTOFF_ANGSTROM,
) -> dict[str, Any]:
    """Compute contact F1 and time-averaged occupancy over all nonbond pairs."""

    if not frames:
        return {
            "contact_f1": None,
            "contact_occupancy_mae": None,
            "contact_occupancy_reason": "empty frame interval",
        }
    source, destination = _nonbond_pairs(batch, int(prediction.shape[1]), prediction.device)
    if source.numel() == 0:
        return {
            "contact_f1": None,
            "contact_occupancy_mae": None,
            "contact_occupancy_reason": "no valid nonbond pair in this system",
        }
    pred_sum = torch.zeros(source.shape, device=prediction.device, dtype=torch.float32)
    target_sum = torch.zeros_like(pred_sum)
    pair_counts = torch.zeros_like(pred_sum)
    frame_f1: list[float] = []
    frame_disagreement: list[float] = []
    frame_precision: list[float] = []
    frame_recall: list[float] = []
    for frame in frames:
        pair_valid = mask[int(frame), source] & mask[int(frame), destination]
        if not bool(pair_valid.any()):
            continue
        pred_distance = torch.linalg.vector_norm(
            prediction[int(frame), source] - prediction[int(frame), destination], dim=-1
        )
        target_distance = torch.linalg.vector_norm(
            target[int(frame), source] - target[int(frame), destination], dim=-1
        )
        pred_contact = pred_distance < float(cutoff)
        target_contact = target_distance < float(cutoff)
        tp = int((pred_contact & target_contact & pair_valid).sum())
        fp = int((pred_contact & ~target_contact & pair_valid).sum())
        fn = int((~pred_contact & target_contact & pair_valid).sum())
        target_count = int((target_contact & pair_valid).sum())
        predicted_count = tp + fp
        union = tp + fp + fn
        precision = 1.0 if predicted_count == 0 and target_count == 0 else (tp / predicted_count if predicted_count else 0.0)
        recall = 1.0 if target_count == 0 else tp / target_count
        f1 = 1.0 if union == 0 else (2.0 * tp / (2.0 * tp + fp + fn))
        frame_precision.append(precision)
        frame_recall.append(recall)
        frame_f1.append(f1)
        frame_disagreement.append(float(fp + fn) / max(int(pair_valid.sum()), 1))
        pred_sum[pair_valid] += pred_contact[pair_valid].float()
        target_sum[pair_valid] += target_contact[pair_valid].float()
        pair_counts[pair_valid] += 1.0
    valid_pairs = pair_counts > 0
    occupancy = None
    if bool(valid_pairs.any()):
        occupancy = float(
            ((pred_sum[valid_pairs] / pair_counts[valid_pairs])
             - (target_sum[valid_pairs] / pair_counts[valid_pairs])).abs().mean()
        )
    return {
        "contact_f1": sum(frame_f1) / len(frame_f1) if frame_f1 else None,
        "contact_precision": sum(frame_precision) / len(frame_precision) if frame_precision else None,
        "contact_recall": sum(frame_recall) / len(frame_recall) if frame_recall else None,
        "contact_frame_disagreement_legacy": sum(frame_disagreement) / len(frame_disagreement) if frame_disagreement else None,
        "contact_occupancy_mae": occupancy,
        "contact_occupancy_reason": "pairwise contact indicator averaged across selected time frames before absolute difference",
        "contact_pair_count": int(valid_pairs.sum()),
        "contact_cutoff_angstrom": float(cutoff),
        "contact_exclusion_rule": "exclude_covalent_bond_pairs_only; never mix molecules",
    }


def _system_drmsd(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor,
    frames: Sequence[int],
) -> float:
    """Compute dRMSD using only within-system atom pairs.

    ``ClipBatch`` stores several clips in one packed atom axis.  The legacy
    helper uses ``torch.pdist`` over that entire axis, which silently creates
    cross-system pairs.  This A-only implementation keeps each ``abid``
    separate and pools only the valid within-system pair errors.
    """

    abid_value = _batch_field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(prediction.shape[1], device=prediction.device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=prediction.device, dtype=torch.long).flatten()
    if abid.numel() != prediction.shape[1]:
        raise ValueError("abid must contain one system id per atom")
    squared: list[Tensor] = []
    for frame in frames:
        for sample in torch.unique(abid, sorted=True).tolist():
            nodes = torch.nonzero(mask[int(frame)] & (abid == int(sample)), as_tuple=False).flatten()
            if nodes.numel() < 2:
                continue
            pred_dist = torch.pdist(prediction[int(frame)].index_select(0, nodes))
            target_dist = torch.pdist(target[int(frame)].index_select(0, nodes))
            squared.append((pred_dist - target_dist).square())
    if not squared:
        return 0.0
    return float(torch.cat(squared).mean().sqrt())


def _geometry_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    frames: Sequence[int],
) -> dict[str, Any]:
    mask = _mask(batch, int(prediction.shape[0]), prediction.device)
    contact = _contact_metrics(prediction, target, batch, mask, frames)
    if not frames:
        return {
            "aligned_rmsd": None,
            "centroid_gauge_raw_rmsd": None,
            "drmsd": None,
            "bond_rmse": None,
            **contact,
        }
    return {
        "aligned_rmsd": _aligned_rmsd(prediction, target, batch, mask, frames),
        "centroid_gauge_raw_rmsd": _rmsd(prediction, target, mask, frames),
        "drmsd": _system_drmsd(prediction, target, batch, mask, frames),
        "bond_rmse": _bond_rmse(prediction, target, batch, mask, frames),
        **contact,
    }


def _rmsf_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    frames: Sequence[int],
) -> dict[str, Any]:
    if len(frames) < 2:
        return {
            "prediction": None,
            "target": None,
            "ratio": None,
            "atom_pearson": {"value": None, "applicable": False, "reason": "fewer than two frames"},
            "applicability_reason": "fewer than two frames",
        }
    selected = tuple(int(value) for value in frames)
    view = _frame_view(batch, selected)
    index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
    pred = prediction.index_select(0, index)
    truth = target.index_select(0, index)
    frame_valid = _frame_atom_mask(view, len(selected), prediction.device)
    mask = _mask(view, len(selected), prediction.device)
    alignment_mask = _align_mask(view, len(selected), prediction.device)
    pred = _align_trajectory_to_first(pred, view, frame_valid, alignment_mask)
    truth = _align_trajectory_to_first(truth, view, frame_valid, alignment_mask)
    stable = mask.all(dim=0)
    if not bool(stable.any()):
        return {
            "prediction": None,
            "target": None,
            "ratio": None,
            "atom_pearson": {"value": None, "applicable": False, "reason": "no atom valid for all selected frames"},
            "applicability_reason": "no atom valid for all selected frames",
        }
    pred_values = torch.linalg.vector_norm(pred[:, stable] - pred[:, stable].mean(dim=0), dim=-1).square().mean(dim=0).sqrt()
    truth_values = torch.linalg.vector_norm(truth[:, stable] - truth[:, stable].mean(dim=0), dim=-1).square().mean(dim=0).sqrt()
    pred_mean = float(pred_values.mean())
    truth_mean = float(truth_values.mean())
    pearson = _safe_pearson(pred_values, truth_values, label="atom RMSF")
    return {
        "prediction": pred_mean,
        "target": truth_mean,
        "ratio": None if truth_mean <= RMSF_NEAR_ZERO_ANGSTROM else pred_mean / truth_mean,
        "atom_pearson": pearson,
        "atom_count": int(pred_values.numel()),
        "prediction_stats_angstrom": _stats(pred_values.detach().cpu().tolist()),
        "target_stats_angstrom": _stats(truth_values.detach().cpu().tolist()),
        "prediction_atom_values_angstrom": [float(value) for value in pred_values.detach().cpu()],
        "target_atom_values_angstrom": [float(value) for value in truth_values.detach().cpu()],
        "near_zero_threshold_angstrom": RMSF_NEAR_ZERO_ANGSTROM,
        "applicability_reason": None,
    }


def _velocity_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    frames: Sequence[int],
) -> dict[str, Any]:
    if len(frames) < 3:
        return {
            "velocity_rmse_angstrom_per_ps": None,
            "velocity_prediction_target_pearson": {"value": None, "applicable": False, "reason": "fewer than three frames"},
            "velocity_lag1_acf": {
                "prediction": None,
                "target": None,
                "applicable": False,
                "reason": "fewer than three frames",
            },
        }
    selected = tuple(int(value) for value in frames)
    if any(second != first + 1 for first, second in zip(selected, selected[1:])):
        raise ValueError("velocity metrics require a contiguous frame interval")
    view = _frame_view(batch, selected)
    index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
    pred = prediction.index_select(0, index)
    truth = target.index_select(0, index)
    frame_valid = _frame_atom_mask(view, len(selected), prediction.device)
    mask = _mask(view, len(selected), prediction.device)
    alignment_mask = _align_mask(view, len(selected), prediction.device)
    pred = _align_trajectory_to_first(pred, view, frame_valid, alignment_mask)
    truth = _align_trajectory_to_first(truth, view, frame_valid, alignment_mask)
    dt = torch.as_tensor(view["delta_time_ps"], device=prediction.device, dtype=torch.float32)[0]
    pred_velocity = (pred[1:] - pred[:-1]) / dt.reshape(-1, 1, 1)
    truth_velocity = (truth[1:] - truth[:-1]) / dt.reshape(-1, 1, 1)
    transition_mask = mask[1:] & mask[:-1]
    values = transition_mask.unsqueeze(-1).expand_as(pred_velocity)
    if not bool(values.any()):
        reason = "no valid atom in velocity transitions"
        return {
            "velocity_rmse_angstrom_per_ps": None,
            "velocity_prediction_target_pearson": {"value": None, "applicable": False, "reason": reason},
            "velocity_lag1_acf": {"prediction": None, "target": None, "applicable": False, "reason": reason},
        }
    error = (pred_velocity - truth_velocity).masked_select(values)
    pred_flat = pred_velocity.masked_select(values)
    truth_flat = truth_velocity.masked_select(values)
    dynamic = _safe_pearson(pred_flat, truth_flat, label="velocity prediction/target")
    common = transition_mask.all(dim=0)
    if pred_velocity.shape[0] >= 2 and bool(common.any()):
        pred_acf = _safe_pearson(
            pred_velocity[:-1, common].reshape(-1),
            pred_velocity[1:, common].reshape(-1),
            label="prediction velocity lag-1 ACF",
        )
        truth_acf = _safe_pearson(
            truth_velocity[:-1, common].reshape(-1),
            truth_velocity[1:, common].reshape(-1),
            label="target velocity lag-1 ACF",
        )
        acf = {
            "prediction": pred_acf["value"],
            "target": truth_acf["value"],
            "applicable": bool(pred_acf["applicable"] and truth_acf["applicable"]),
            "prediction_reason": pred_acf["reason"],
            "target_reason": truth_acf["reason"],
            "lag": 1,
            "units": "dimensionless",
            "signal": "Kabsch-aligned frame-to-frame velocity in angstrom_per_ps",
        }
    else:
        acf = {
            "prediction": None,
            "target": None,
            "applicable": False,
            "reason": "fewer than two velocity samples with common valid atoms",
            "lag": 1,
            "units": "dimensionless",
        }
    return {
        "velocity_rmse_angstrom_per_ps": float(error.square().mean().sqrt()),
        "velocity_prediction_target_pearson": dynamic,
        "velocity_lag1_acf": acf,
        "dt_ps": [float(value) for value in dt.detach().cpu()],
        "velocity_count": int(pred_flat.numel()),
    }


def _displacement_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    frames: Sequence[int],
    *,
    ratio: int = RATIO,
) -> dict[str, Any]:
    if len(frames) < 2:
        return {"applicable": False, "reason": "fewer than two frames"}
    selected = tuple(int(value) for value in frames)
    if any(second != first + 1 for first, second in zip(selected, selected[1:])):
        raise ValueError("displacement metrics require a contiguous frame interval")
    view = _frame_view(batch, selected)
    index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
    pred = prediction.index_select(0, index)
    truth = target.index_select(0, index)
    frame_valid = _frame_atom_mask(view, len(selected), prediction.device)
    mask = _mask(view, len(selected), prediction.device)
    alignment_mask = _align_mask(view, len(selected), prediction.device)
    pred = _align_trajectory_to_first(pred, view, frame_valid, alignment_mask)
    truth = _align_trajectory_to_first(truth, view, frame_valid, alignment_mask)
    dt = torch.as_tensor(view["delta_time_ps"], device=prediction.device, dtype=torch.float32)[0]
    pred_delta = pred[1:] - pred[:-1]
    truth_delta = truth[1:] - truth[:-1]
    transition_mask = mask[1:] & mask[:-1]
    labels = ["block_boundary" if int(frame) % int(ratio) == 0 else "within_block" for frame in selected[1:]]
    result: dict[str, Any] = {
        "applicable": True,
        "alignment": "per-trajectory-Kabsch-to-own-first-selected-frame",
        "units": {"displacement": "angstrom", "velocity": "angstrom_per_ps"},
        "groups": {},
    }
    for group in ("all", "within_block", "block_boundary"):
        values_pred: list[float] = []
        values_truth: list[float] = []
        velocity_pred: list[float] = []
        velocity_truth: list[float] = []
        endpoints: list[int] = []
        for transition, label in enumerate(labels):
            if group != "all" and label != group:
                continue
            valid = transition_mask[transition]
            if not bool(valid.any()):
                continue
            pred_mag = torch.linalg.vector_norm(pred_delta[transition], dim=-1)[valid]
            truth_mag = torch.linalg.vector_norm(truth_delta[transition], dim=-1)[valid]
            values_pred.extend(float(value) for value in pred_mag.detach().cpu())
            values_truth.extend(float(value) for value in truth_mag.detach().cpu())
            velocity_pred.extend(float(value) for value in (pred_mag / dt[transition]).detach().cpu())
            velocity_truth.extend(float(value) for value in (truth_mag / dt[transition]).detach().cpu())
            endpoints.append(int(selected[transition + 1]))
        result["groups"][group] = {
            "frame_endpoints": endpoints,
            "prediction_displacement_angstrom": _stats(values_pred),
            "target_displacement_angstrom": _stats(values_truth),
            "prediction_velocity_angstrom_per_ps": _stats(velocity_pred),
            "target_velocity_angstrom_per_ps": _stats(velocity_truth),
        }
    return result


def _frequency_retention(prediction: Tensor, target: Tensor, batch: Any, frames: Sequence[int]) -> dict[str, Any]:
    if len(frames) < 3:
        return {"value": None, "applicable": False, "reason": "fewer than three frames"}
    selected = tuple(int(value) for value in frames)
    view = _frame_view(batch, selected)
    index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
    pred = prediction.index_select(0, index)
    truth = target.index_select(0, index)
    mask = _mask(view, len(selected), prediction.device)
    valid = mask.all(dim=0)
    if not bool(valid.any()):
        return {"value": None, "applicable": False, "reason": "no atom valid for all frequency frames"}
    pred_signal = pred[:, valid] - pred[:, valid].mean(dim=0, keepdim=True)
    truth_signal = truth[:, valid] - truth[:, valid].mean(dim=0, keepdim=True)
    pred_power = torch.fft.rfft(pred_signal, dim=0).abs().square()[1:].sum()
    truth_power = torch.fft.rfft(truth_signal, dim=0).abs().square()[1:].sum()
    if float(truth_power) <= 1.0e-12:
        return {
            "value": None,
            "applicable": False,
            "reason": "target nonzero-frequency power is zero",
            "definition": "total nonzero-frequency power ratio only; not spectral-shape agreement",
        }
    return {
        "value": float(pred_power / truth_power),
        "applicable": True,
        "definition": "total nonzero-frequency power ratio only; not spectral-shape agreement",
    }


def _boundary_metrics(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    if history <= 0 or history >= int(prediction.shape[0]):
        return {"applicable": False, "reason": "no observation-to-future boundary"}
    mask = _mask(batch, int(prediction.shape[0]), prediction.device)
    valid = mask[history - 1] & mask[history]
    if not bool(valid.any()):
        return {"applicable": False, "reason": "no valid atoms at observation boundary"}
    pred_step = prediction[history] - prediction[history - 1]
    truth_step = target[history] - target[history - 1]
    pred_mag = torch.linalg.vector_norm(pred_step, dim=-1)[valid]
    truth_mag = torch.linalg.vector_norm(truth_step, dim=-1)[valid]
    return {
        "applicable": True,
        "frame_interval": [history - 1, history],
        "prediction_displacement_angstrom": _stats(pred_mag.detach().cpu().tolist()),
        "target_displacement_angstrom": _stats(truth_mag.detach().cpu().tolist()),
        "vector_error_angstrom": float((pred_step[valid] - truth_step[valid]).square().sum(dim=-1).mean().sqrt()),
        "dt_ps": float(batch.delta_time_ps[0, history - 1]),
    }


def _torsion_metric(batch: Any) -> dict[str, Any]:
    value = _batch_field(batch, "torsion_index")
    if value is None or torch.as_tensor(value).numel() == 0:
        return {"value": None, "applicable": False, "reason": "torsion_index is not present"}
    return {"value": None, "applicable": False, "reason": "torsion_index support is not implemented in this reassessment layer"}


def _section_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    frames: Sequence[int],
) -> dict[str, Any]:
    geometry = _geometry_metrics(prediction, target, batch, frames)
    geometry["rmsf"] = _rmsf_metrics(prediction, target, batch, frames)
    geometry["velocity"] = _velocity_metrics(prediction, target, batch, frames)
    geometry["displacement"] = _displacement_metrics(prediction, target, batch, frames)
    geometry["frequency_retention"] = _frequency_retention(prediction, target, batch, frames)
    return geometry


def reassessment_trajectory_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    history_frames: int,
) -> dict[str, Any]:
    """Return per-system future metrics and frame/transition diagnostics."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must both have shape [T,N,3]")
    history = int(history_frames)
    if history not in HISTORIES:
        raise ValueError("reassessment supports H=4 and H=8")
    total = int(prediction.shape[0])
    future = list(range(history, total))
    observed = list(range(history))
    all_frames = list(range(total))
    horizons: dict[str, Any] = {}
    for length in (4, 8):
        indices = list(range(history, min(history + length, total)))
        horizons[f"L{length}"] = {
            "available": len(indices) == length,
            "frame_interval": [indices[0], indices[-1] + 1] if indices else None,
            "metrics": _section_metrics(prediction, target, batch, indices) if len(indices) == length else None,
        }
    frame_curve = []
    mask = _mask(batch, total, prediction.device)
    for frame in all_frames:
        one = _geometry_metrics(prediction, target, batch, [frame])
        frame_curve.append(
            {
                "frame": frame,
                "observed": frame < history,
                "future": frame >= history,
                "aligned_rmsd": one["aligned_rmsd"],
                "centroid_gauge_raw_rmsd": one["centroid_gauge_raw_rmsd"],
                "bond_rmse": one["bond_rmse"],
                "contact_f1": one.get("contact_f1"),
                "valid_atom_count": int(mask[frame].sum()),
            }
        )
    return {
        "future": _section_metrics(prediction, target, batch, future),
        "observed": _section_metrics(prediction, target, batch, observed) if observed else None,
        "all_frames": _section_metrics(prediction, target, batch, all_frames),
        "horizons": horizons,
        "boundary": _boundary_metrics(prediction, target, batch, history),
        "torsion": _torsion_metric(batch),
        "frame_curve": frame_curve,
        "protocol": {
            "history_frames": history,
            "future_interval": [history, total],
            "physical_time_units": "ps",
            "aligned_rmsd": "per-frame Kabsch using align_mask; RMSD over loss_mask",
            "dRMSD": "per-system within-system atom-pair distance error; no cross-molecule pairs",
            "rmsf_alignment": "each prediction/target trajectory aligned to its own first selected frame",
            "velocity_alignment": "per-trajectory Kabsch-aligned frame-to-frame velocity",
            "compression_block_ratio": RATIO,
            "compression_boundary_rule": "transition endpoint frame % ratio == 0",
            "contact_occupancy": "average contact indicator across selected frames before absolute difference",
            "frequency_retention": "total nonzero-frequency power ratio only; not spectral-shape agreement",
        },
    }


def future_diversity(
    samples: Tensor,
    batch: Any,
    history_frames: int,
) -> dict[str, Any]:
    """Compute pairwise future-only diversity without observed frames."""

    if samples.ndim != 4 or samples.shape[-1] != 3:
        raise ValueError("samples must have shape [draw,T,N,3]")
    future = tuple(range(int(history_frames), int(samples.shape[1])))
    mask = _mask(batch, int(samples.shape[1]), samples.device).index_select(
        0, torch.as_tensor(future, device=samples.device)
    )
    raw_values: list[float] = []
    aligned_values: list[float] = []
    for first in range(int(samples.shape[0])):
        for second in range(first + 1, int(samples.shape[0])):
            raw = samples[first].index_select(0, torch.as_tensor(future, device=samples.device)) - samples[second].index_select(0, torch.as_tensor(future, device=samples.device))
            valid = mask.unsqueeze(-1).expand_as(raw)
            if bool(valid.any()):
                raw_values.append(float(raw.masked_select(valid).square().mean().sqrt()))
            aligned, _ = _aligned_prediction(
                samples[second], samples[first], batch, frames=future
            )
            delta = samples[first].index_select(0, torch.as_tensor(future, device=samples.device)) - aligned.index_select(0, torch.as_tensor(future, device=samples.device))
            if bool(valid.any()):
                aligned_values.append(float(delta.masked_select(valid).square().mean().sqrt()))
    return {
        "draw_count": int(samples.shape[0]),
        "pair_count": int(samples.shape[0] * (samples.shape[0] - 1) // 2),
        "future_interval": [int(history_frames), int(samples.shape[1])],
        "pairwise_raw_rmsd": None if not raw_values else sum(raw_values) / len(raw_values),
        "pairwise_aligned_rmsd": None if not aligned_values else sum(aligned_values) / len(aligned_values),
        "rmsd_definition": "sqrt(mean squared xyz over future valid atoms)",
        "primary_metric": "not_best_of_n; observed frames excluded",
    }


def _mask_valid_fields(fields: LatentFieldSet, batch: DiTLatentBatch) -> LatentFieldSet:
    masks = batch.field_masks()
    return LatentFieldSet(
        fields.state_h * masks["state_h"].unsqueeze(-1).to(fields.state_h.dtype),
        fields.detail_h * masks["detail_h"].unsqueeze(-1).to(fields.detail_h.dtype),
        fields.state_v * masks["state_v"].unsqueeze(-1).unsqueeze(-1).to(fields.state_v.dtype),
        fields.detail_v * masks["detail_v"].unsqueeze(-1).unsqueeze(-1).to(fields.detail_v.dtype),
    )


def generate_fixed_noise_latent(
    model: torch.nn.Module,
    adapter: StateDetailLatentAdapter,
    batch: DiTLatentBatch,
    statistics: LatentStatistics,
    *,
    noise: LatentFieldSet,
    steps: int,
    source_center: LatentFieldSet | None,
    source_mode: str,
) -> tuple[Any, dict[str, Any]]:
    """Run the existing Euler equations with a caller-owned noise draw.

    The equations and clamp order match ``module.latent_rectified_flow``.  The
    only reassessment-specific extension is accepting 32 steps and separating
    noise creation from the arm/step loop.
    """

    steps = int(steps)
    if steps not in FIXED_EULER_STEPS:
        raise ValueError(f"reassessment supports Euler steps {FIXED_EULER_STEPS}")
    if batch.observed_mask is None:
        raise ValueError("fixed-noise sampling requires an observation mask")
    normalized = statistics.normalize(batch)
    if tuple(getattr(noise, name).shape for name in FIELD_NAMES) != tuple(
        getattr(normalized.fields, name).shape for name in FIELD_NAMES
    ):
        raise ValueError("fixed noise shape does not match the normalized latent batch")
    was_training = model.training
    model.eval()
    source = combine_source_fields(source_center, noise, source_mode=source_mode)
    current = apply_observation_clamp(
        _mask_valid_fields(source, normalized), normalized.fields, normalized
    )
    with torch.no_grad():
        for step in range(steps):
            tau = torch.full(
                (normalized.batch_size,),
                float(step) / float(steps),
                device=normalized.state_h.device,
                dtype=normalized.state_h.dtype,
            )
            prediction = model(normalized.with_fields(current), tau)
            current = LatentFieldSet(
                *(getattr(current, name) + getattr(prediction, name) / float(steps) for name in FIELD_NAMES)
            )
            current = apply_observation_clamp(current, normalized.fields, normalized)
    sampled = normalized.with_fields(current).zero_invalid()
    observed = normalized.observed_mask.index_select(0, normalized.abid).transpose(0, 1)
    errors: list[Tensor] = []
    for name in FIELD_NAMES:
        value = getattr(current, name)
        clean = getattr(normalized.fields, name)
        expanded = observed.reshape(observed.shape + (1,) * (value.ndim - 2))
        if bool(expanded.any()):
            errors.append((value - clean).abs().masked_select(expanded).max())
    clamp_error = float(torch.stack(errors).max()) if errors else 0.0
    raw = statistics.inverse_normalize(sampled)
    if was_training:
        model.train()
    metadata = {
        "solver": "euler",
        "steps": steps,
        "deterministic": True,
        "fixed_noise": True,
        "noise_sha256": tensor_bytes_hash(noise),
        "observed_clamp_exact": clamp_error == 0.0,
        "observed_clamp_max_abs": clamp_error,
        "stats_hash": statistics.hash,
        "adapter_hash": contract_hash(adapter.contract()),
        "source_mode": source_mode,
        "source_center": source_mode == "conditional",
    }
    return adapter.make_generated_latent(raw, raw.fields), metadata


def make_fixed_noise(
    normalized_batch: DiTLatentBatch,
    *,
    seed: int,
) -> tuple[LatentFieldSet, dict[str, Any]]:
    try:
        generator = torch.Generator(device=normalized_batch.state_h.device).manual_seed(int(seed))
    except (RuntimeError, TypeError):
        generator = torch.Generator().manual_seed(int(seed))
    noise = sample_isotropic_noise(normalized_batch.fields, generator=generator)
    return noise, {"seed": int(seed), "noise_sha256": tensor_bytes_hash(noise)}


def aggregate_generated_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate generated rows draw -> clip -> system, without mixing steps."""

    by_clip: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        system = row.get("system")
        sample_id = row.get("sample_id")
        if not system or not isinstance(sample_id, str):
            raise ValueError("generated rows require parsed system and sample_id")
        by_clip[(str(system), sample_id)].append(row)
    clip_rows = []
    for (system, sample_id), group in sorted(by_clip.items()):
        future = [
            row.get("metrics", {}).get("future", {})
            for row in group
            if isinstance(row.get("metrics", {}).get("future", {}), Mapping)
        ]
        clip_rows.append(
            {
                "system": system,
                "sample_id": sample_id,
                "draw_count": len(group),
                "future": _mean_mapping(future),
            }
        )
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        by_system[str(row["system"])].append(row)
    system_rows = [
        {
            "system": system,
            "clip_count": len(group),
            "future": _mean_mapping([row["future"] for row in group]),
        }
        for system, group in sorted(by_system.items())
    ]
    return {
        "aggregation": "draw_equal_then_clip_equal_then_system_equal",
        "row_count": len(rows),
        "draw_count": len({row.get("draw") for row in rows}),
        "clip_count": len(clip_rows),
        "system_count": len(system_rows),
        "system_equal": _mean_mapping([row["future"] for row in system_rows]),
        "clip_rows": clip_rows,
        "system_rows": system_rows,
    }


__all__ = [
    "ACF_STD_EPSILON",
    "DRAW_IDS",
    "FIXED_EULER_STEPS",
    "HISTORIES",
    "REASSESSMENT_SCHEMA",
    "aggregate_generated_rows",
    "future_diversity",
    "generate_fixed_noise_latent",
    "make_fixed_noise",
    "reassessment_seed",
    "reassessment_trajectory_metrics",
    "reaggregate_legacy_jsonl",
    "seed_contract",
    "sha256_file",
    "tensor_bytes_hash",
]
