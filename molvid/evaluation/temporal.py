"""Physical-time motion metrics with one observed-reference alignment.

The v3 metrics deliberately keep geometry alignment separate from supervision:
prediction and MD frames are each aligned to the same last-observed MD frame.
No generated frame is fitted to its hidden future target.  Internal-distance
and torsion increments are computed without coordinate alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


TEMPORAL_METRIC_SCHEMA = "molvid.frame_gm.temporal_metrics.v3"
ALIGNMENT_SCHEMA = "last_observed_fixed_reference.v1"


@dataclass(frozen=True)
class TemporalMetricResult:
    metrics: Mapping[str, Any]
    profiles: Mapping[str, Tensor]


def _field(batch: Any, name: str) -> Any:
    if isinstance(batch, Mapping):
        if name not in batch:
            raise ValueError(f"batch is missing {name!r}")
        return batch[name]
    if not hasattr(batch, name):
        raise ValueError(f"batch is missing {name!r}")
    return getattr(batch, name)


def _average_ranks(value: Tensor) -> Tensor:
    """Return deterministic average ranks, including exact ties."""

    flat = value.detach().to(device="cpu", dtype=torch.float64).flatten()
    order = torch.argsort(flat, stable=True)
    sorted_value = flat.index_select(0, order)
    ranks = torch.empty_like(sorted_value)
    start = 0
    while start < sorted_value.numel():
        stop = start + 1
        while stop < sorted_value.numel() and bool(sorted_value[stop] == sorted_value[start]):
            stop += 1
        ranks[start:stop] = 0.5 * float(start + stop - 1)
        start = stop
    result = torch.empty_like(ranks)
    result.index_copy_(0, order, ranks)
    return result.to(device=value.device, dtype=torch.float32)


def _correlation(first: Tensor, second: Tensor, *, method: str) -> dict[str, Any]:
    first = first.float().flatten()
    second = second.float().flatten()
    finite = torch.isfinite(first) & torch.isfinite(second)
    first, second = first[finite], second[finite]
    if first.numel() < 2:
        return {"available": False, "reason": "insufficient_values", "value": None, "count": int(first.numel())}
    if method == "spearman":
        first, second = _average_ranks(first), _average_ranks(second)
    elif method != "pearson":
        raise ValueError(f"unsupported correlation method {method!r}")
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 1.0e-12:
        return {"available": False, "reason": "zero_variance", "value": None, "count": int(first.numel())}
    value = float(((first * second).sum() / denominator).detach().cpu())
    return {"available": True, "reason": None, "value": value, "count": int(first.numel())}


def _kabsch_apply(coordinates: Tensor, source: Tensor, reference: Tensor) -> tuple[Tensor, str]:
    count = int(source.shape[0])
    if count == 0:
        return coordinates, "none"
    if torch.equal(source, reference):
        return coordinates, "identity"
    output_dtype = coordinates.dtype
    coordinates = coordinates.to(dtype=torch.float64)
    source = source.to(dtype=torch.float64)
    reference = reference.to(dtype=torch.float64)
    source_center = source.mean(dim=0)
    reference_center = reference.mean(dim=0)
    centered = coordinates - source_center
    if count < 3:
        return (centered + reference_center).to(dtype=output_dtype), "translation_only"
    covariance = (source - source_center).transpose(0, 1) @ (reference - reference_center)
    left, _singular, right_transpose = torch.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_transpose
    if torch.linalg.det(rotation) < 0:
        left = left.clone()
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return (centered @ rotation + reference_center).to(dtype=output_dtype), "kabsch"


def align_to_last_observed_reference(
    coordinates: Tensor,
    reference_coordinates: Tensor,
    batch: Any,
    *,
    history_frames: int,
) -> tuple[Tensor, dict[str, int]]:
    """Align every valid frame to the same last-observed reference frame."""

    if coordinates.shape != reference_coordinates.shape or coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates and reference_coordinates must have shape [T,N,3]")
    history = int(history_frames)
    if not 1 <= history < coordinates.shape[0]:
        raise ValueError("history_frames must leave at least one query slot")
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=coordinates.device, dtype=torch.bool)
    abid = torch.as_tensor(_field(batch, "abid"), device=coordinates.device, dtype=torch.long).flatten()
    align_mask = torch.as_tensor(_field(batch, "align_mask"), device=coordinates.device, dtype=torch.bool).flatten()
    if frame_mask.ndim != 2 or frame_mask.shape[1] != coordinates.shape[0]:
        raise ValueError("frame_mask must have shape [B,T]")
    if abid.numel() != coordinates.shape[1] or align_mask.numel() != coordinates.shape[1]:
        raise ValueError("packed atom metadata disagrees with coordinates")
    if abid.numel() and (int(abid.min()) < 0 or int(abid.max()) >= frame_mask.shape[0]):
        raise ValueError("abid contains a sample index outside frame_mask")
    reference_index = history - 1
    aligned = coordinates.clone()
    modes = {"identity": 0, "kabsch": 0, "translation_only": 0, "none": 0}
    for sample in range(int(frame_mask.shape[0])):
        if not bool(frame_mask[sample, reference_index]):
            raise ValueError("last observed reference frame is invalid")
        nodes = torch.nonzero(abid == sample, as_tuple=False).flatten()
        alignment_nodes = nodes[align_mask.index_select(0, nodes)]
        if alignment_nodes.numel() == 0:
            raise ValueError("each packed sample needs at least one alignment atom")
        reference = reference_coordinates[reference_index].index_select(0, alignment_nodes)
        for frame in torch.nonzero(frame_mask[sample], as_tuple=False).flatten().tolist():
            current = coordinates[frame].index_select(0, nodes)
            source = coordinates[frame].index_select(0, alignment_nodes)
            transformed, mode = _kabsch_apply(current, source, reference)
            aligned[frame, nodes] = transformed
            modes[mode] += 1
    return aligned, modes


def _future_sequence(frame_mask: Tensor, sample: int, history: int) -> list[int]:
    valid = torch.nonzero(frame_mask[sample], as_tuple=False).flatten().tolist()
    future = [int(frame) for frame in valid if int(frame) >= history]
    return [history - 1, *future]


def _rmsf_profile(coordinates: Tensor) -> Tensor:
    if torch.equal(coordinates, coordinates[:1].expand_as(coordinates)):
        return coordinates.new_zeros((coordinates.shape[1],))
    centered = coordinates.to(dtype=torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    return centered.square().sum(dim=-1).mean(dim=0).sqrt().to(dtype=coordinates.dtype)


def _profile_summary(prediction: Tensor, target: Tensor) -> dict[str, Any]:
    if prediction.numel() == 0:
        unavailable = {"available": False, "reason": "no_valid_values", "value": None, "count": 0}
        return {
            "available": False,
            "reason": "no_valid_values",
            "count": 0,
            "mae_A": None,
            "pearson": unavailable,
            "spearman": unavailable,
        }
    return {
        "available": True,
        "reason": None,
        "count": int(prediction.numel()),
        "mae_A": float((prediction - target).abs().mean().detach().cpu()),
        "pearson": _correlation(prediction, target, method="pearson"),
        "spearman": _correlation(prediction, target, method="spearman"),
    }


def _rmsf_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    history: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=prediction.device, dtype=torch.bool)
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long).flatten()
    loss_mask = torch.as_tensor(_field(batch, "loss_mask"), device=prediction.device, dtype=torch.bool).flatten()
    block_id = torch.as_tensor(_field(batch, "block_id"), device=prediction.device, dtype=torch.long).flatten()
    atom_prediction: list[Tensor] = []
    atom_target: list[Tensor] = []
    atom_sample: list[Tensor] = []
    atom_block: list[Tensor] = []
    residue_prediction: list[Tensor] = []
    residue_target: list[Tensor] = []
    residue_sample: list[Tensor] = []
    residue_block: list[Tensor] = []
    per_sample: list[dict[str, Any]] = []
    for sample in range(int(frame_mask.shape[0])):
        frames = _future_sequence(frame_mask, sample, history)[1:]
        nodes = torch.nonzero((abid == sample) & loss_mask, as_tuple=False).flatten()
        if not frames or nodes.numel() == 0:
            per_sample.append({"sample": sample, "available": False, "reason": "no_valid_future_atoms"})
            continue
        index = torch.as_tensor(frames, device=prediction.device, dtype=torch.long)
        pred_profile = _rmsf_profile(prediction.index_select(0, index).index_select(1, nodes))
        target_profile = _rmsf_profile(target.index_select(0, index).index_select(1, nodes))
        atom_prediction.append(pred_profile)
        atom_target.append(target_profile)
        atom_sample.append(torch.full_like(nodes, sample))
        atom_block.append(block_id.index_select(0, nodes))
        sample_residue_prediction: list[Tensor] = []
        sample_residue_target: list[Tensor] = []
        sample_residue_ids: list[int] = []
        local_blocks = block_id.index_select(0, nodes)
        for residue in torch.unique(local_blocks, sorted=True).tolist():
            member = local_blocks == int(residue)
            sample_residue_prediction.append(pred_profile[member].mean())
            sample_residue_target.append(target_profile[member].mean())
            sample_residue_ids.append(int(residue))
        pred_residue = torch.stack(sample_residue_prediction)
        target_residue = torch.stack(sample_residue_target)
        residue_prediction.append(pred_residue)
        residue_target.append(target_residue)
        residue_sample.append(torch.full((len(sample_residue_ids),), sample, device=prediction.device, dtype=torch.long))
        residue_block.append(torch.as_tensor(sample_residue_ids, device=prediction.device, dtype=torch.long))
        per_sample.append({
            "sample": sample,
            "available": True,
            "reason": None,
            "future_frames": len(frames),
            "atom_count": int(nodes.numel()),
            "residue_count": len(sample_residue_ids),
            "mean_prediction_A": float(pred_profile.mean().detach().cpu()),
            "mean_target_A": float(target_profile.mean().detach().cpu()),
            "mean_amplitude_ratio": (
                float((pred_profile.mean() / target_profile.mean()).detach().cpu())
                if float(target_profile.mean()) > 1.0e-12
                else None
            ),
        })
    device = prediction.device
    atom_pred = torch.cat(atom_prediction) if atom_prediction else torch.empty(0, device=device)
    atom_true = torch.cat(atom_target) if atom_target else torch.empty(0, device=device)
    residue_pred = torch.cat(residue_prediction) if residue_prediction else torch.empty(0, device=device)
    residue_true = torch.cat(residue_target) if residue_target else torch.empty(0, device=device)
    valid_samples = [item for item in per_sample if item.get("available")]
    system_prediction = (
        sum(float(item["mean_prediction_A"]) for item in valid_samples) / len(valid_samples)
        if valid_samples else None
    )
    system_target = (
        sum(float(item["mean_target_A"]) for item in valid_samples) / len(valid_samples)
        if valid_samples else None
    )
    return {
        "available": bool(valid_samples),
        "reason": None if valid_samples else "no_valid_future_atoms",
        "alignment": ALIGNMENT_SCHEMA,
        "window": "valid future query frames only",
        "atom_profile": _profile_summary(atom_pred, atom_true),
        "residue_profile": _profile_summary(residue_pred, residue_true),
        "system_mean_prediction_A": system_prediction,
        "system_mean_target_A": system_target,
        "mean_amplitude_ratio": (
            system_prediction / system_target
            if system_prediction is not None and system_target is not None and abs(system_target) > 1.0e-12
            else None
        ),
        "per_sample": per_sample,
    }, {
        "atom_prediction_A": atom_pred,
        "atom_target_A": atom_true,
        "atom_sample": torch.cat(atom_sample) if atom_sample else torch.empty(0, device=device, dtype=torch.long),
        "atom_block_id": torch.cat(atom_block) if atom_block else torch.empty(0, device=device, dtype=torch.long),
        "residue_prediction_A": residue_pred,
        "residue_target_A": residue_true,
        "residue_sample": torch.cat(residue_sample) if residue_sample else torch.empty(0, device=device, dtype=torch.long),
        "residue_block_id": torch.cat(residue_block) if residue_block else torch.empty(0, device=device, dtype=torch.long),
    }


def _increment_metrics(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=prediction.device, dtype=torch.bool)
    time_ps = torch.as_tensor(_field(batch, "time_ps"), device=prediction.device, dtype=torch.float32)
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long).flatten()
    loss_mask = torch.as_tensor(_field(batch, "loss_mask"), device=prediction.device, dtype=torch.bool).flatten()
    displacement_error: list[Tensor] = []
    velocity_error: list[Tensor] = []
    pred_velocity: list[Tensor] = []
    target_velocity: list[Tensor] = []
    intervals: list[float] = []
    for sample in range(int(frame_mask.shape[0])):
        sequence = _future_sequence(frame_mask, sample, history)
        nodes = torch.nonzero((abid == sample) & loss_mask, as_tuple=False).flatten()
        for previous, current in zip(sequence[:-1], sequence[1:]):
            dt = time_ps[sample, current] - time_ps[sample, previous]
            if not bool(torch.isfinite(dt)) or float(dt) <= 0:
                raise ValueError("valid query times must have positive physical intervals")
            pred_step = prediction[current].index_select(0, nodes) - prediction[previous].index_select(0, nodes)
            target_step = target[current].index_select(0, nodes) - target[previous].index_select(0, nodes)
            error = pred_step - target_step
            displacement_error.append(error.square().sum(dim=-1))
            velocity_error.append((error / dt).square().sum(dim=-1))
            pred_velocity.append((pred_step / dt).reshape(-1))
            target_velocity.append((target_step / dt).reshape(-1))
            intervals.append(float(dt.detach().cpu()))
    if not displacement_error:
        return {
            "available": False,
            "reason": "insufficient_valid_frames",
            "displacement_increment_rmse_A": None,
            "finite_difference_velocity_rmse_A_per_ps": None,
            "velocity_correlation": {"available": False, "reason": "insufficient_values", "value": None, "count": 0},
            "intervals_ps": [],
            "valid_atom_increments": 0,
        }
    displacement = torch.cat(displacement_error)
    velocity = torch.cat(velocity_error)
    return {
        "available": True,
        "reason": None,
        "displacement_increment_rmse_A": float(displacement.mean().sqrt().detach().cpu()),
        "finite_difference_velocity_rmse_A_per_ps": float(velocity.mean().sqrt().detach().cpu()),
        "velocity_correlation": _correlation(torch.cat(pred_velocity), torch.cat(target_velocity), method="pearson"),
        "intervals_ps": sorted(set(intervals)),
        "valid_atom_increments": int(displacement.numel()),
        "definition": "Euclidean frame-increment error; velocity divides each increment by its true dt_ps",
    }


def _unique_local_pairs(bond_index: Tensor, nodes: set[int]) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    adjacency = {node: set() for node in nodes}
    bonds: set[tuple[int, int]] = set()
    for first, second in bond_index.transpose(0, 1).tolist():
        first, second = int(first), int(second)
        if first not in nodes or second not in nodes or first == second:
            continue
        edge = (min(first, second), max(first, second))
        bonds.add(edge)
        adjacency[first].add(second)
        adjacency[second].add(first)
    pairs = set(bonds)
    for center in sorted(adjacency):
        neighbors = sorted(adjacency[center])
        for index, first in enumerate(neighbors):
            for second in neighbors[index + 1:]:
                pairs.add((min(first, second), max(first, second)))
    return sorted(pairs), adjacency


def _torsions(adjacency: Mapping[int, set[int]], *, maximum: int = 4096) -> list[tuple[int, int, int, int]]:
    values: set[tuple[int, int, int, int]] = set()
    for middle_left in sorted(adjacency):
        for middle_right in sorted(adjacency[middle_left]):
            if middle_left >= middle_right:
                continue
            for first in sorted(adjacency[middle_left] - {middle_right}):
                for last in sorted(adjacency[middle_right] - {middle_left}):
                    if first == last:
                        continue
                    value = (first, middle_left, middle_right, last)
                    reverse = tuple(reversed(value))
                    values.add(min(value, reverse))
                    if len(values) >= int(maximum):
                        return sorted(values)
    return sorted(values)


def _dihedral(coordinates: Tensor, indices: Tensor) -> Tensor:
    points = coordinates.index_select(0, indices.reshape(-1)).reshape(indices.shape[0], 4, 3)
    b0 = points[:, 1] - points[:, 0]
    b1 = points[:, 2] - points[:, 1]
    b2 = points[:, 3] - points[:, 2]
    b1_unit = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1.0e-8)
    first = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    second = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    x = (first * second).sum(dim=-1)
    y = (torch.cross(b1_unit, first, dim=-1) * second).sum(dim=-1)
    return torch.atan2(y, x)


def _circular_delta(current: Tensor, previous: Tensor) -> Tensor:
    return torch.remainder(current - previous + math.pi, 2.0 * math.pi) - math.pi


def _internal_metrics(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=prediction.device, dtype=torch.bool)
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long).flatten()
    loss_mask = torch.as_tensor(_field(batch, "loss_mask"), device=prediction.device, dtype=torch.bool).flatten()
    bond_index = torch.as_tensor(_field(batch, "bond_index"), device=prediction.device, dtype=torch.long)
    distance_prediction: list[Tensor] = []
    distance_target: list[Tensor] = []
    torsion_prediction: list[Tensor] = []
    torsion_target: list[Tensor] = []
    pair_count = torsion_count = 0
    for sample in range(int(frame_mask.shape[0])):
        sequence = _future_sequence(frame_mask, sample, history)
        nodes_tensor = torch.nonzero((abid == sample) & loss_mask, as_tuple=False).flatten()
        nodes = set(int(value) for value in nodes_tensor.tolist())
        pairs, adjacency = _unique_local_pairs(bond_index, nodes)
        torsions = _torsions(adjacency)
        pair_count += len(pairs)
        torsion_count += len(torsions)
        pair_tensor = torch.as_tensor(pairs, device=prediction.device, dtype=torch.long) if pairs else None
        torsion_tensor = torch.as_tensor(torsions, device=prediction.device, dtype=torch.long) if torsions else None
        for previous, current in zip(sequence[:-1], sequence[1:]):
            if pair_tensor is not None:
                pred_current = torch.linalg.vector_norm(
                    prediction[current].index_select(0, pair_tensor[:, 0]) - prediction[current].index_select(0, pair_tensor[:, 1]), dim=-1
                )
                pred_previous = torch.linalg.vector_norm(
                    prediction[previous].index_select(0, pair_tensor[:, 0]) - prediction[previous].index_select(0, pair_tensor[:, 1]), dim=-1
                )
                target_current = torch.linalg.vector_norm(
                    target[current].index_select(0, pair_tensor[:, 0]) - target[current].index_select(0, pair_tensor[:, 1]), dim=-1
                )
                target_previous = torch.linalg.vector_norm(
                    target[previous].index_select(0, pair_tensor[:, 0]) - target[previous].index_select(0, pair_tensor[:, 1]), dim=-1
                )
                distance_prediction.append(pred_current - pred_previous)
                distance_target.append(target_current - target_previous)
            if torsion_tensor is not None:
                torsion_prediction.append(_circular_delta(_dihedral(prediction[current], torsion_tensor), _dihedral(prediction[previous], torsion_tensor)))
                torsion_target.append(_circular_delta(_dihedral(target[current], torsion_tensor), _dihedral(target[previous], torsion_tensor)))

    def report(
        predicted: Sequence[Tensor],
        truth: Sequence[Tensor],
        missing_reason: str,
        *,
        mae_name: str,
    ) -> dict[str, Any]:
        if not predicted:
            return {"available": False, "reason": missing_reason, mae_name: None, "pearson": {"available": False, "reason": missing_reason, "value": None, "count": 0}}
        pred = torch.cat(list(predicted))
        true = torch.cat(list(truth))
        return {
            "available": True,
            "reason": None,
            mae_name: float((pred - true).abs().mean().detach().cpu()),
            "pearson": _correlation(pred, true, method="pearson"),
            "value_count": int(pred.numel()),
        }

    return {
        "pair_selection": "static covalent 1-hop plus 1-3 pairs; observed/topology only",
        "distance_increment": report(
            distance_prediction,
            distance_target,
            "missing_local_pairs",
            mae_name="mae_A",
        ),
        "distance_pair_count": pair_count,
        "torsion_selection": "static four-atom bond paths; maximum 4096 per sample",
        "torsion_increment_radian": report(
            torsion_prediction,
            torsion_target,
            "missing_torsion_paths",
            mae_name="mae_radian",
        ),
        "torsion_count": torsion_count,
    }


def _msd_metrics(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=prediction.device, dtype=torch.bool)
    time_ps = torch.as_tensor(_field(batch, "time_ps"), device=prediction.device, dtype=torch.float32)
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long).flatten()
    loss_mask = torch.as_tensor(_field(batch, "loss_mask"), device=prediction.device, dtype=torch.bool).flatten()
    curve: list[dict[str, Any]] = []
    errors: list[float] = []
    reference = history - 1
    for sample in range(int(frame_mask.shape[0])):
        nodes = torch.nonzero((abid == sample) & loss_mask, as_tuple=False).flatten()
        for frame in _future_sequence(frame_mask, sample, history)[1:]:
            pred_delta = prediction[frame].index_select(0, nodes) - prediction[reference].index_select(0, nodes)
            target_delta = target[frame].index_select(0, nodes) - target[reference].index_select(0, nodes)
            pred_msd = float(pred_delta.square().sum(dim=-1).mean().detach().cpu())
            target_msd = float(target_delta.square().sum(dim=-1).mean().detach().cpu())
            error = abs(pred_msd - target_msd)
            errors.append(error)
            curve.append({
                "sample": sample,
                "frame_index": int(frame),
                "horizon_ps": float((time_ps[sample, frame] - time_ps[sample, reference]).detach().cpu()),
                "prediction_A2": pred_msd,
                "target_A2": target_msd,
                "absolute_error_A2": error,
                "atom_count": int(nodes.numel()),
            })
    return {
        "available": bool(curve),
        "reason": None if curve else "no_valid_future_frames",
        "reference": "last observed frame after shared fixed-reference alignment",
        "curve": curve,
        "curve_mae_A2": sum(errors) / len(errors) if errors else None,
    }


def temporal_metrics_v3(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    history_frames: int,
) -> TemporalMetricResult:
    """Compute v3 motion metrics and reusable atom/residue RMSF profiles."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have shape [T,N,3]")
    history = int(history_frames)
    aligned_prediction, prediction_modes = align_to_last_observed_reference(
        prediction, target, batch, history_frames=history
    )
    aligned_target, target_modes = align_to_last_observed_reference(
        target, target, batch, history_frames=history
    )
    rmsf, profiles = _rmsf_metrics(aligned_prediction, aligned_target, batch, history)
    return TemporalMetricResult(
        metrics={
            "schema": TEMPORAL_METRIC_SCHEMA,
            "history_frames": history,
            "coordinate_unit": "angstrom",
            "physical_time_unit": "picosecond",
            "alignment": {
                "schema": ALIGNMENT_SCHEMA,
                "reference_frame_index": history - 1,
                "reference_coordinates": "MD last observed frame",
                "atom_set": "fixed declared align_mask per packed system",
                "prediction_fit_target_future": False,
                "prediction_modes": prediction_modes,
                "target_modes": target_modes,
            },
            "increments": _increment_metrics(aligned_prediction, aligned_target, batch, history),
            "rmsf": rmsf,
            "msd": _msd_metrics(aligned_prediction, aligned_target, batch, history),
            "internal": _internal_metrics(prediction, target, batch, history),
        },
        profiles=profiles,
    )


def system_mean_rmsf_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize already system-equal RMSF means without atom pseudoreplication."""

    prediction: list[float] = []
    target: list[float] = []
    for row in rows:
        first = row.get("system_mean_prediction_A")
        second = row.get("system_mean_target_A")
        if first is None or second is None:
            continue
        first, second = float(first), float(second)
        if math.isfinite(first) and math.isfinite(second):
            prediction.append(first)
            target.append(second)
    if not prediction:
        unavailable = {"available": False, "reason": "no_available_systems", "value": None, "count": 0}
        return {
            "available": False,
            "reason": "no_available_systems",
            "system_count": 0,
            "mae_A": None,
            "pearson": unavailable,
            "spearman": unavailable,
            "std_prediction_A": None,
            "std_target_A": None,
            "spread_ratio_prediction_over_target": None,
        }
    pred = torch.tensor(prediction, dtype=torch.float64)
    truth = torch.tensor(target, dtype=torch.float64)
    pred_std = float(pred.std(unbiased=False))
    target_std = float(truth.std(unbiased=False))
    return {
        "available": True,
        "reason": None,
        "system_count": len(prediction),
        "mae_A": float((pred - truth).abs().mean()),
        "pearson": _correlation(pred, truth, method="pearson"),
        "spearman": _correlation(pred, truth, method="spearman"),
        "std_prediction_A": pred_std,
        "std_target_A": target_std,
        "spread_ratio_prediction_over_target": pred_std / target_std if target_std > 1.0e-12 else None,
    }


__all__ = [
    "ALIGNMENT_SCHEMA",
    "TEMPORAL_METRIC_SCHEMA",
    "TemporalMetricResult",
    "align_to_last_observed_reference",
    "system_mean_rmsf_summary",
    "temporal_metrics_v3",
]
