"""Round-trip metrics and control evaluation for the multi-frame codec."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor

from data.clip_dataset import STATIC_TASK, ClipBatch
from trainer.codec_losses import acceleration_loss, velocity_loss

try:
    from torch_cluster import radius_graph
except (ImportError, OSError):
    radius_graph = None


EVALUATION_SCHEMA = "pvb.codec.eval.v2"
CONTACT_CUTOFF_ANGSTROM = 4.5
CONTACT_EXCLUSION_RULE = "exclude_covalent_bond_pairs_only"
ALIGNED_RMSD_NAME = "aligned_rmsd"
RAW_RMSD_NAME = "centroid_gauge_raw_rmsd"


def _field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _prediction(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Mapping) and "x_hat" in value:
        return value["x_hat"]
    if hasattr(value, "x_hat"):
        return value.x_hat
    raise TypeError("control predictor must return coordinates or expose x_hat")


def _mask(batch: Any, frames: int, device: torch.device) -> Tensor:
    frame_valid = _frame_atom_mask(batch, frames, device)
    loss_mask = torch.as_tensor(_field(batch, "loss_mask"), device=device, dtype=torch.bool)
    return frame_valid & loss_mask.unsqueeze(0)


def _frame_atom_mask(batch: Any, frames: int, device: torch.device) -> Tensor:
    """Return frame validity in the packed ``[T, N]`` atom space."""

    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=device, dtype=torch.bool)
    abid = torch.as_tensor(_field(batch, "abid"), device=device, dtype=torch.long).flatten()
    if frame_mask.ndim != 2 or frame_mask.shape[1] != frames:
        raise ValueError("frame_mask must have shape [B, T] matching coordinates")
    if abid.numel() == 0 or torch.any(abid < 0) or torch.any(abid >= frame_mask.shape[0]):
        raise ValueError("abid contains an invalid sample index")
    return frame_mask.index_select(0, abid).transpose(0, 1)


def _align_mask(batch: Any, frames: int, device: torch.device) -> Tensor:
    """Return the declared rigid-alignment atom mask in ``[T, N]`` space."""

    value = _field(batch, "align_mask")
    if value is None:
        raise ValueError("evaluation batch must declare align_mask")
    align_mask = torch.as_tensor(value, device=device, dtype=torch.bool).flatten()
    if align_mask.numel() == 0:
        raise ValueError("align_mask must contain at least one atom")
    return _frame_atom_mask(batch, frames, device) & align_mask.unsqueeze(0)


def _pairs(batch: Any, device: torch.device) -> Optional[Tensor]:
    value = _field(batch, "bond_index")
    if value is None:
        return None
    value = torch.as_tensor(value, device=device, dtype=torch.long)
    if value.numel() == 0:
        return None
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError("bond_index must have shape [2, E]")
    return value


def _safe_mean(value: Tensor) -> float:
    return float(value.mean().detach().cpu()) if value.numel() else 0.0


def _rmsd(prediction: Tensor, target: Tensor, mask: Tensor, frames: Sequence[int]) -> float:
    values = []
    for frame in frames:
        valid = mask[frame]
        if torch.any(valid):
            values.append((prediction[frame, valid] - target[frame, valid]).square().sum(dim=-1).mean())
    return math.sqrt(max(_safe_mean(torch.stack(values)) if values else 0.0, 0.0))


def _kabsch_align(source: Tensor, reference: Tensor) -> tuple[Tensor, str]:
    """Align one ``[N, 3]`` source frame to a reference frame.

    The fallback for fewer than three alignment atoms is deliberately
    translation-only.  With no alignment atom, the source is left untouched;
    callers record that case as an unaligned frame rather than inventing a
    rotation from insufficient evidence.
    """

    if source.ndim != 2 or reference.shape != source.shape or source.shape[-1] != 3:
        raise ValueError("Kabsch inputs must both have shape [N, 3]")
    count = int(source.shape[0])
    if count == 0:
        return source, "none"
    source_center = source.mean(dim=0)
    reference_center = reference.mean(dim=0)
    source_centered = source - source_center
    if count < 3:
        return source_centered + reference_center, "translation_only"
    covariance = source_centered.transpose(0, 1) @ (reference - reference_center)
    left, _singular, right_transpose = torch.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_transpose
    if torch.linalg.det(rotation) < 0:
        left = left.clone()
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return source_centered @ rotation + reference_center, "kabsch"


def _aligned_prediction(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    frames: Sequence[int] | None = None,
) -> tuple[Tensor, dict[str, int]]:
    """Kabsch-align prediction frame-by-frame using the declared ``align_mask``."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("coordinates must both have shape [T, N, 3]")
    frame_count = int(prediction.shape[0])
    align = _align_mask(batch, frame_count, prediction.device)
    selected = tuple(range(frame_count)) if frames is None else tuple(int(i) for i in frames)
    aligned = prediction.clone()
    modes = {"kabsch": 0, "translation_only": 0, "none": 0}
    for frame in selected:
        if frame < 0 or frame >= frame_count:
            raise ValueError(f"frame index {frame} is outside the coordinate sequence")
        valid = align[frame]
        source = prediction[frame, valid]
        reference = target[frame, valid]
        frame_aligned, mode = _kabsch_align(source, reference)
        if mode == "none":
            aligned[frame] = prediction[frame]
        else:
            # Reapply the transform estimated on alignment atoms to every atom
            # in this frame.  The transform is represented by the row-vector
            # map used in _kabsch_align.
            source_center = source.mean(dim=0)
            reference_center = reference.mean(dim=0)
            if mode == "translation_only":
                aligned[frame] = prediction[frame] - source_center + reference_center
            else:
                covariance = (source - source_center).transpose(0, 1) @ (
                    reference - reference_center
                )
                left, _singular, right_transpose = torch.linalg.svd(
                    covariance, full_matrices=False
                )
                rotation = left @ right_transpose
                if torch.linalg.det(rotation) < 0:
                    left = left.clone()
                    left[:, -1] *= -1
                    rotation = left @ right_transpose
                aligned[frame] = (prediction[frame] - source_center) @ rotation + reference_center
        modes[mode] += 1
    return aligned, modes


def _aligned_rmsd(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor,
    frames: Sequence[int],
) -> float:
    aligned, _modes = _aligned_prediction(prediction, target, batch, frames=frames)
    values = []
    for frame in frames:
        valid = mask[frame]
        if torch.any(valid):
            values.append((aligned[frame, valid] - target[frame, valid]).square().sum(dim=-1).mean())
    return math.sqrt(max(_safe_mean(torch.stack(values)) if values else 0.0, 0.0))


def _drmsd(prediction: Tensor, target: Tensor, mask: Tensor, frames: Sequence[int]) -> float:
    values = []
    for frame in frames:
        valid_indices = torch.nonzero(mask[frame], as_tuple=False).flatten()
        if valid_indices.numel() < 2:
            continue
        pred_dist = torch.pdist(prediction[frame].index_select(0, valid_indices))
        target_dist = torch.pdist(target[frame].index_select(0, valid_indices))
        values.append((pred_dist - target_dist).square().mean())
    return math.sqrt(max(_safe_mean(torch.stack(values)) if values else 0.0, 0.0))


def _bond_rmse(prediction: Tensor, target: Tensor, batch: Any, mask: Tensor, frames: Sequence[int]) -> float:
    pairs = _pairs(batch, prediction.device)
    if pairs is None:
        return 0.0
    values = []
    source, destination = pairs
    for frame in frames:
        pair_mask = mask[frame, source] & mask[frame, destination]
        if torch.any(pair_mask):
            pred_dist = torch.linalg.vector_norm(prediction[frame, source] - prediction[frame, destination], dim=-1)
            target_dist = torch.linalg.vector_norm(target[frame, source] - target[frame, destination], dim=-1)
            values.append((pred_dist[pair_mask] - target_dist[pair_mask]).square())
    return math.sqrt(max(_safe_mean(torch.cat(values)) if values else 0.0, 0.0))


def _nonbond_pairs(batch: Any, atoms: int, device: torch.device) -> tuple[Tensor, Tensor]:
    abid_value = _field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(atoms, device=device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=device, dtype=torch.long)
    if abid.numel() != atoms:
        raise ValueError("abid must have one entry per atom")
    source_parts: list[Tensor] = []
    destination_parts: list[Tensor] = []
    for sample in torch.unique(abid, sorted=True).tolist():
        nodes = torch.nonzero(abid == int(sample), as_tuple=False).flatten()
        if nodes.numel() < 2:
            continue
        local = torch.triu_indices(nodes.numel(), nodes.numel(), offset=1, device=device)
        source_parts.append(nodes.index_select(0, local[0]))
        destination_parts.append(nodes.index_select(0, local[1]))
    if not source_parts:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    source = torch.cat(source_parts)
    destination = torch.cat(destination_parts)
    pairs = _pairs(batch, device)
    if pairs is not None:
        bond_codes = torch.minimum(pairs[0], pairs[1]) * atoms + torch.maximum(pairs[0], pairs[1])
        codes = source * atoms + destination
        keep = ~torch.isin(codes, bond_codes)
        source, destination = source[keep], destination[keep]
    return source, destination


def _candidate_nonbond_pairs(
    batch: Any, positions: Sequence[Tensor], radius: float
) -> tuple[Tensor, Tensor]:
    atoms = int(positions[0].shape[0])
    if atoms <= 2048 or radius_graph is None:
        return _nonbond_pairs(batch, atoms, positions[0].device)
    abid_value = _field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(atoms, device=positions[0].device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=positions[0].device, dtype=torch.long)
    code_parts: list[Tensor] = []
    for position in positions:
        edges = radius_graph(
            position,
            r=float(radius),
            batch=abid,
            loop=False,
            max_num_neighbors=128,
        )
        if edges.numel():
            low = torch.minimum(edges[0], edges[1])
            high = torch.maximum(edges[0], edges[1])
            code_parts.append(torch.unique(low * atoms + high))
    if not code_parts:
        empty = torch.empty(0, device=positions[0].device, dtype=torch.long)
        return empty, empty
    codes = torch.unique(torch.cat(code_parts))
    source = torch.div(codes, atoms, rounding_mode="floor")
    destination = codes.remainder(atoms)
    pairs = _pairs(batch, positions[0].device)
    if pairs is not None:
        bond_codes = torch.minimum(pairs[0], pairs[1]) * atoms + torch.maximum(pairs[0], pairs[1])
        keep = ~torch.isin(codes, bond_codes)
        source, destination = source[keep], destination[keep]
    return source, destination


def _nonbond_count(batch: Any, valid_mask: Tensor) -> int:
    abid_value = _field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(valid_mask.shape[0], device=valid_mask.device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=valid_mask.device, dtype=torch.long)
    total = 0
    for sample in torch.unique(abid, sorted=True).tolist():
        count = int((valid_mask & (abid == int(sample))).sum().item())
        total += count * (count - 1) // 2
    pairs = _pairs(batch, valid_mask.device)
    if pairs is not None:
        same_sample = abid.index_select(0, pairs[0]) == abid.index_select(0, pairs[1])
        valid_bonds = same_sample & valid_mask.index_select(0, pairs[0]) & valid_mask.index_select(0, pairs[1])
        if torch.any(valid_bonds):
            atoms = int(valid_mask.shape[0])
            bond_codes = (
                torch.minimum(pairs[0], pairs[1]) * atoms
                + torch.maximum(pairs[0], pairs[1])
            )
            total -= int(torch.unique(bond_codes[valid_bonds]).numel())
    return max(total, 1)


def _safe_ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator <= 0:
        return float(empty_value)
    return float(numerator) / float(denominator)


def contact_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor,
    frames: Sequence[int],
    *,
    cutoff: float = CONTACT_CUTOFF_ANGSTROM,
) -> dict[str, float]:
    """Compare nonbond contact identities, retaining false positives/negatives.

    The candidate universe is every within-sample nonbond pair for small
    structures, or the union of the prediction/target radius candidates for
    larger structures.  Covalent pairs are excluded explicitly.  Counts are
    aggregated per frame with equal frame weight and are never reduced to an
    absolute difference of contact totals.
    """

    if cutoff <= 0 or not math.isfinite(float(cutoff)):
        raise ValueError("contact cutoff must be finite and positive")
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "target_contacts": 0,
        "noncontact_pairs": 0,
        "total_pairs": 0,
    }
    frame_values: list[dict[str, float]] = []
    for frame in frames:
        source, destination = _candidate_nonbond_pairs(
            batch, (prediction[frame], target[frame]), float(cutoff)
        )
        pair_mask = mask[frame, source] & mask[frame, destination]
        source = source[pair_mask]
        destination = destination[pair_mask]
        total_pairs = _nonbond_count(batch, mask[frame])
        if source.numel():
            pred_dist = torch.linalg.vector_norm(
                prediction[frame, source] - prediction[frame, destination], dim=-1
            )
            target_dist = torch.linalg.vector_norm(
                target[frame, source] - target[frame, destination], dim=-1
            )
            predicted_contacts = pred_dist < float(cutoff)
            target_contacts = target_dist < float(cutoff)
            tp = int((predicted_contacts & target_contacts).sum().item())
            fp = int((predicted_contacts & ~target_contacts).sum().item())
            fn = int((~predicted_contacts & target_contacts).sum().item())
            target_count = int(target_contacts.sum().item())
        else:
            tp = fp = fn = target_count = 0
        noncontact_count = max(int(total_pairs) - target_count, 0)
        predicted_count = tp + fp
        union = tp + fp + fn
        precision = _safe_ratio(tp, predicted_count, empty_value=1.0 if target_count == 0 else 0.0)
        recall = _safe_ratio(tp, target_count, empty_value=1.0)
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn, empty_value=1.0 if union == 0 else 0.0)
        jaccard = _safe_ratio(tp, union, empty_value=1.0)
        false_positive_rate = _safe_ratio(fp, noncontact_count, empty_value=0.0)
        false_negative_rate = _safe_ratio(fn, target_count, empty_value=0.0)
        occupancy_mae = _safe_ratio(fp + fn, int(total_pairs), empty_value=0.0)
        frame_values.append(
            {
                "contact_precision": precision,
                "contact_recall": recall,
                "contact_f1": f1,
                "contact_jaccard": jaccard,
                "contact_false_positive_rate": false_positive_rate,
                "contact_false_negative_rate": false_negative_rate,
                "contact_occupancy_mae": occupancy_mae,
            }
        )
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["target_contacts"] += target_count
        totals["noncontact_pairs"] += noncontact_count
        totals["total_pairs"] += int(total_pairs)
    if not frame_values:
        return {
            "contact_precision": 0.0,
            "contact_recall": 0.0,
            "contact_f1": 0.0,
            "contact_jaccard": 0.0,
            "contact_false_positive_rate": 0.0,
            "contact_false_negative_rate": 0.0,
            "contact_occupancy_mae": 0.0,
        }
    return {
        key: sum(value[key] for value in frame_values) / len(frame_values)
        for key in frame_values[0]
    }


def _contact_error(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor,
    frames: Sequence[int],
    cutoff: float = CONTACT_CUTOFF_ANGSTROM,
) -> float:
    """Legacy name for contact occupancy MAE, retained explicitly."""

    return contact_metrics(
        prediction, target, batch, mask, frames, cutoff=cutoff
    )["contact_occupancy_mae"]


def _clash_rate(prediction: Tensor, batch: Any, mask: Tensor, frames: Sequence[int], cutoff: float = 0.7) -> float:
    values = []
    for frame in frames:
        source, destination = _candidate_nonbond_pairs(batch, (prediction[frame],), cutoff)
        pair_mask = mask[frame, source] & mask[frame, destination]
        total_pairs = _nonbond_count(batch, mask[frame])
        if torch.any(pair_mask):
            distance = torch.linalg.vector_norm(prediction[frame, source] - prediction[frame, destination], dim=-1)
            values.append((distance[pair_mask] < cutoff).sum().float() / float(total_pairs))
        else:
            values.append(torch.zeros((), device=prediction.device))
    return _safe_mean(torch.stack(values)) if values else 0.0


def _wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _dihedral(points: Tensor) -> Tensor:
    b0 = points[..., 1, :] - points[..., 0, :]
    b1 = points[..., 2, :] - points[..., 1, :]
    b2 = points[..., 3, :] - points[..., 2, :]
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def _torsion_change(prediction: Tensor, target: Tensor, batch: Any, mask: Tensor, frames: Sequence[int]) -> float:
    torsions = _field(batch, "torsion_index")
    if torsions is None:
        return 0.0
    torsions = torch.as_tensor(torsions, device=prediction.device, dtype=torch.long)
    if torsions.numel() == 0:
        return 0.0
    if torsions.ndim != 2 or torsions.shape[0] != 4:
        raise ValueError("torsion_index must have shape [4, Q]")
    values = []
    for frame in frames:
        valid = mask[frame].index_select(0, torsions.reshape(-1)).reshape(4, -1).all(0)
        if torch.any(valid):
            pred_angle = _dihedral(prediction[frame, torsions].transpose(0, 1))[valid]
            target_angle = _dihedral(target[frame, torsions].transpose(0, 1))[valid]
            values.append(_wrap_angle(pred_angle - target_angle).abs())
    return _safe_mean(torch.cat(values)) if values else 0.0


def _frequency_retention(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if prediction.shape[0] < 3:
        return 1.0 if torch.allclose(prediction, target) else 0.0
    valid = mask.all(0)
    if not torch.any(valid):
        return 0.0
    pred_signal = prediction[:, valid] - prediction[:, valid].mean(0, keepdim=True)
    target_signal = target[:, valid] - target[:, valid].mean(0, keepdim=True)
    pred_power = torch.fft.rfft(pred_signal, dim=0).abs().square()[1:].sum()
    target_power = torch.fft.rfft(target_signal, dim=0).abs().square()[1:].sum()
    if target_power <= 1e-12:
        return 1.0 if pred_power <= 1e-12 else 0.0
    return float((pred_power / target_power).detach().cpu())


def _safe_correlation(first: Tensor, second: Tensor) -> float:
    if first.numel() == 0 or second.numel() == 0:
        return 0.0
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 1.0e-12:
        return 1.0 if torch.allclose(first, second) else 0.0
    return float(((first * second).sum() / denominator).detach().cpu())


def _time_deltas(batch: Any, sample: int, frames: int, device: torch.device) -> Tensor:
    value = _field(batch, "delta_time_ps")
    if value is None:
        return torch.ones(max(frames - 1, 0), device=device, dtype=torch.float32)
    deltas = torch.as_tensor(value, device=device, dtype=torch.float32)
    if deltas.ndim == 1:
        deltas = deltas.unsqueeze(0)
    if deltas.ndim != 2 or deltas.shape[1] != max(frames - 1, 0):
        raise ValueError("delta_time_ps must have shape [B, T-1] for dynamic metrics")
    return deltas[sample].clamp_min(1.0e-8)


def _align_trajectory_to_first(
    coordinates: Tensor,
    batch: Any,
    frame_valid: Tensor,
    alignment_mask: Tensor,
) -> Tensor:
    """Remove per-trajectory rigid motion using each stream's first frame."""

    aligned = coordinates.clone()
    abid = torch.as_tensor(_field(batch, "abid"), device=coordinates.device, dtype=torch.long)
    for sample in torch.unique(abid, sorted=True).tolist():
        nodes = torch.nonzero(abid == int(sample), as_tuple=False).flatten()
        valid_frames = torch.nonzero(frame_valid[:, nodes].any(dim=1), as_tuple=False).flatten()
        if valid_frames.numel() == 0:
            continue
        reference_index = int(valid_frames[0])
        reference = coordinates[reference_index, nodes]
        reference_alignment = alignment_mask[reference_index, nodes]
        for frame_index in valid_frames.tolist():
            current = coordinates[frame_index, nodes]
            common = alignment_mask[frame_index, nodes] & reference_alignment
            source_alignment = current[common]
            reference_points = reference[common]
            if source_alignment.numel() == 0:
                continue
            source_center = source_alignment.mean(dim=0)
            reference_center = reference_points.mean(dim=0)
            centered = current - source_center
            if int(source_alignment.shape[0]) < 3:
                aligned[frame_index, nodes] = centered + reference_center
                continue
            covariance = (source_alignment - source_center).transpose(0, 1) @ (
                reference_points - reference_center
            )
            left, _singular, right_transpose = torch.linalg.svd(
                covariance, full_matrices=False
            )
            rotation = left @ right_transpose
            if torch.linalg.det(rotation) < 0:
                left = left.clone()
                left[:, -1] *= -1
                rotation = left @ right_transpose
            aligned[frame_index, nodes] = centered @ rotation + reference_center
    return aligned


def _frame_view(batch: Any, frames: Sequence[int]) -> dict[str, Any]:
    """Build a shallow batch view whose temporal metadata matches the selected frames."""

    values = dict(batch) if isinstance(batch, Mapping) else dict(vars(batch))
    frame_mask = values.get("frame_mask")
    if frame_mask is None:
        raise ValueError("evaluation batch must declare frame_mask")
    frame_mask = torch.as_tensor(frame_mask)
    total_frames = int(frame_mask.shape[1])
    selected = tuple(int(index) for index in frames)
    if any(index < 0 or index >= total_frames for index in selected):
        raise ValueError("requested frame interval is outside the batch")
    index = torch.as_tensor(selected, device=frame_mask.device, dtype=torch.long)
    values["frame_mask"] = frame_mask.index_select(1, index)
    for name in ("time_ps", "frame_time_ps"):
        value = values.get(name)
        if value is not None:
            value = torch.as_tensor(value)
            if value.ndim >= 2 and value.shape[1] == total_frames:
                values[name] = value.index_select(1, index)
    for name in ("x", "bpos"):
        value = values.get(name)
        if value is not None:
            value = torch.as_tensor(value)
            if value.ndim >= 1 and value.shape[0] == total_frames:
                values[name] = value.index_select(0, index)
    delta = values.get("delta_time_ps")
    if delta is not None:
        delta = torch.as_tensor(delta)
        if len(selected) < 2:
            values["delta_time_ps"] = delta.new_empty((delta.shape[0], 0))
        elif "time_ps" in values:
            values["delta_time_ps"] = values["time_ps"][:, 1:] - values["time_ps"][:, :-1]
        else:
            previous = torch.as_tensor(
                frames[:-1], device=delta.device, dtype=torch.long
            )
            values["delta_time_ps"] = delta.index_select(1, previous)
    return values


def aligned_rmsf_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor | None = None,
    *,
    frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute RMSF after per-frame rigid-body handling, equal by sample."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("coordinates must both have shape [T, N, 3]")
    if frames is not None:
        selected = tuple(int(index) for index in frames)
        view = _frame_view(batch, selected)
        index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
        prediction = prediction.index_select(0, index)
        target = target.index_select(0, index)
        batch = view
        if mask is not None:
            mask = mask.index_select(0, index)
    if mask is None:
        mask = _mask(batch, prediction.shape[0], prediction.device)
    frame_valid = _frame_atom_mask(batch, prediction.shape[0], prediction.device)
    alignment_mask = _align_mask(batch, prediction.shape[0], prediction.device)
    aligned_prediction = _align_trajectory_to_first(
        prediction, batch, frame_valid, alignment_mask
    )
    aligned_target = _align_trajectory_to_first(
        target, batch, frame_valid, alignment_mask
    )
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long)
    prediction_samples: list[Tensor] = []
    target_samples: list[Tensor] = []
    prediction_atoms: list[Tensor] = []
    target_atoms: list[Tensor] = []
    for sample in torch.unique(abid, sorted=True).tolist():
        nodes = torch.nonzero(abid == int(sample), as_tuple=False).flatten()
        pred_values: list[Tensor] = []
        target_values: list[Tensor] = []
        for node in nodes.tolist():
            valid = mask[:, node]
            if not torch.any(valid):
                continue
            pred_atom = aligned_prediction[valid, node]
            target_atom = aligned_target[valid, node]
            pred_values.append(
                torch.linalg.vector_norm(pred_atom - pred_atom.mean(0), dim=-1)
                .square()
                .mean()
                .sqrt()
            )
            target_values.append(
                torch.linalg.vector_norm(target_atom - target_atom.mean(0), dim=-1)
                .square()
                .mean()
                .sqrt()
            )
        if pred_values:
            pred_stack = torch.stack(pred_values)
            target_stack = torch.stack(target_values)
            prediction_atoms.append(pred_stack)
            target_atoms.append(target_stack)
            prediction_samples.append(pred_stack.mean())
            target_samples.append(target_stack.mean())
    if not prediction_samples:
        return {
            "prediction": 0.0,
            "target": 0.0,
            "absolute_error": 0.0,
            "correlation": 0.0,
            "alignment": "per-frame-Kabsch-to-own-trajectory-frame0",
            "aggregation": "sample_equal_mean",
        }
    prediction_mean = torch.stack(prediction_samples).mean()
    target_mean = torch.stack(target_samples).mean()
    return {
        "prediction": float(prediction_mean.detach().cpu()),
        "target": float(target_mean.detach().cpu()),
        "absolute_error": float((prediction_mean - target_mean).abs().detach().cpu()),
        "correlation": _safe_correlation(torch.cat(prediction_atoms), torch.cat(target_atoms)),
        "alignment": "per-frame-Kabsch-to-own-trajectory-frame0",
        "aggregation": "sample_equal_mean",
    }


def dynamic_acf_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    mask: Tensor | None = None,
    *,
    frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure lag-1 ACF and prediction/target correlation on aligned velocity."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("coordinates must both have shape [T, N, 3]")
    if frames is not None:
        selected = tuple(int(index) for index in frames)
        view = _frame_view(batch, selected)
        index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
        prediction = prediction.index_select(0, index)
        target = target.index_select(0, index)
        batch = view
        if mask is not None:
            mask = mask.index_select(0, index)
    frames = int(prediction.shape[0])
    if mask is None:
        mask = _mask(batch, frames, prediction.device)
    frame_valid = _frame_atom_mask(batch, frames, prediction.device)
    alignment_mask = _align_mask(batch, frames, prediction.device)
    aligned_prediction = _align_trajectory_to_first(
        prediction, batch, frame_valid, alignment_mask
    )
    aligned_target = _align_trajectory_to_first(
        target, batch, frame_valid, alignment_mask
    )
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long)
    prediction_acf: list[float] = []
    target_acf: list[float] = []
    dynamic_correlations: list[float] = []
    for sample in torch.unique(abid, sorted=True).tolist():
        nodes = torch.nonzero(abid == int(sample), as_tuple=False).flatten()
        stable = mask[:, nodes].all(dim=0)
        if not torch.any(stable) or frames < 3:
            continue
        stable_nodes = nodes[stable]
        deltas = _time_deltas(batch, int(sample), frames, prediction.device)
        pred_velocity = (
            aligned_prediction[1:, stable_nodes]
            - aligned_prediction[:-1, stable_nodes]
        ) / deltas.view(-1, 1, 1)
        target_velocity = (
            aligned_target[1:, stable_nodes]
            - aligned_target[:-1, stable_nodes]
        ) / deltas.view(-1, 1, 1)
        pred_velocity = pred_velocity - pred_velocity.mean(dim=0, keepdim=True)
        target_velocity = target_velocity - target_velocity.mean(dim=0, keepdim=True)
        dynamic_correlations.append(
            _safe_correlation(pred_velocity.reshape(-1), target_velocity.reshape(-1))
        )
        if pred_velocity.shape[0] >= 2:
            prediction_acf.append(
                _safe_correlation(pred_velocity[:-1].reshape(-1), pred_velocity[1:].reshape(-1))
            )
            target_acf.append(
                _safe_correlation(target_velocity[:-1].reshape(-1), target_velocity[1:].reshape(-1))
            )
    if not dynamic_correlations:
        return {
            "prediction": 0.0,
            "target": 0.0,
            "absolute_error": 0.0,
            "dynamic_correlation": 0.0,
            "dynamic_correlation_absolute_error": 0.0,
            "signal": "per-trajectory-Kabsch-aligned-frame-to-frame-velocity",
            "lag": 1,
            "units": "angstrom_per_ps",
            "mean_removed": True,
            "aggregation": "sample_equal_mean",
        }
    pred_acf = sum(prediction_acf) / len(prediction_acf) if prediction_acf else 0.0
    target_acf = sum(target_acf) / len(target_acf) if target_acf else 0.0
    dynamic_correlation = sum(dynamic_correlations) / len(dynamic_correlations)
    return {
        "prediction": pred_acf,
        "target": target_acf,
        "absolute_error": abs(pred_acf - target_acf),
        "dynamic_correlation": dynamic_correlation,
        "dynamic_correlation_absolute_error": abs(1.0 - dynamic_correlation),
        "signal": "per-trajectory-Kabsch-aligned-frame-to-frame-velocity",
        "lag": 1,
        "units": "angstrom_per_ps",
        "mean_removed": True,
        "aggregation": "sample_equal_mean",
    }


def _empty_temporal_metrics() -> dict[str, Any]:
    return {
        "velocity_rmse": 0.0,
        "acceleration_rmse": 0.0,
        "frequency_retention": 0.0,
        "rmsf": {},
        "dynamic": {},
    }


def _temporal_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    frames: Sequence[int],
) -> dict[str, Any]:
    selected = tuple(int(index) for index in frames)
    if not selected:
        return _empty_temporal_metrics()
    view = _frame_view(batch, selected)
    index = torch.as_tensor(selected, device=prediction.device, dtype=torch.long)
    prediction_view = prediction.index_select(0, index)
    target_view = target.index_select(0, index)
    mask = _mask(view, len(selected), prediction.device)
    return {
        "velocity_rmse": math.sqrt(
            max(float(velocity_loss(prediction_view, target_view, view).detach().cpu()), 0.0)
        ),
        "acceleration_rmse": math.sqrt(
            max(float(acceleration_loss(prediction_view, target_view, view).detach().cpu()), 0.0)
        ),
        "frequency_retention": _frequency_retention(
            prediction_view, target_view, mask
        ),
        "rmsf": aligned_rmsf_metrics(
            prediction_view, target_view, view, mask
        ),
        "dynamic": dynamic_acf_metrics(
            prediction_view, target_view, view, mask
        ),
    }


def _boundary_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    history_frames: int,
) -> dict[str, Any]:
    boundary = int(history_frames)
    if boundary <= 0 or boundary >= prediction.shape[0]:
        return {
            "available": False,
            "history_frames": boundary,
            "frame_interval": None,
            "valid_elements": 0,
            "predicted_step_magnitude": None,
            "target_step_magnitude": None,
            "step_difference": None,
        }
    mask = _mask(batch, prediction.shape[0], prediction.device)
    valid = mask[boundary - 1] & mask[boundary]
    if not torch.any(valid):
        return {
            "available": False,
            "history_frames": boundary,
            "frame_interval": [boundary - 1, boundary],
            "valid_elements": 0,
            "predicted_step_magnitude": None,
            "target_step_magnitude": None,
            "step_difference": None,
        }
    predicted_step = prediction[boundary] - prediction[boundary - 1]
    target_step = target[boundary] - target[boundary - 1]
    return {
        "available": True,
        "history_frames": boundary,
        "frame_interval": [boundary - 1, boundary],
        "valid_elements": int(valid.sum().item()),
        "predicted_step_magnitude": float(
            predicted_step[valid].square().sum(dim=-1).mean().sqrt().detach().cpu()
        ),
        "target_step_magnitude": float(
            target_step[valid].square().sum(dim=-1).mean().sqrt().detach().cpu()
        ),
        "step_difference": float(
            (predicted_step[valid] - target_step[valid]).square().sum(dim=-1).mean().sqrt().detach().cpu()
        ),
    }


def _metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    history_frames: int | None = None,
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("control output and target must both have shape [T, N, 3]")
    mask = _mask(batch, prediction.shape[0], prediction.device)
    total_frames = int(prediction.shape[0])
    if history_frames is None:
        observed = None
        future = list(range(1, total_frames))
    else:
        history = int(history_frames)
        if history not in (0, 4, 8):
            raise ValueError("history_frames must be H=0, H=4, or H=8")
        if history > total_frames:
            raise ValueError("history_frames cannot exceed the coordinate sequence")
        observed = list(range(history))
        future = list(range(history, total_frames))
    all_frames = list(range(prediction.shape[0]))

    def one(frames: Sequence[int]) -> dict[str, float]:
        contacts = contact_metrics(prediction, target, batch, mask, frames)
        raw_rmsd = _rmsd(prediction, target, mask, frames)
        return {
            # ``rmsd`` is retained as an explicit legacy/raw alias.  New
            # reports must use aligned_rmsd or centroid_gauge_raw_rmsd.
            "rmsd": raw_rmsd,
            "aligned_rmsd": _aligned_rmsd(prediction, target, batch, mask, frames),
            "centroid_gauge_raw_rmsd": raw_rmsd,
            "drmsd": _drmsd(prediction, target, mask, frames),
            "bond_rmse": _bond_rmse(prediction, target, batch, mask, frames),
            "contact_error": contacts["contact_occupancy_mae"],
            **contacts,
            "clash_rate": _clash_rate(prediction, batch, mask, frames),
            "torsion_change": _torsion_change(prediction, target, batch, mask, frames),
        }

    future_metrics = one(future) if future else one([])
    result = {
        "frame0": one([0]),
        "future": future_metrics,
        "all_frames": one(all_frames),
    }
    protocol = {
        "aligned_rmsd": "per-frame Kabsch using align_mask; RMSD over loss_mask",
        "raw_rmsd": "centroid_gauge_raw_rmsd; direct coordinate difference",
        "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
        "contact_exclusion_rule": CONTACT_EXCLUSION_RULE,
        "frequency_retention_interpretation": "values above one indicate excessive predicted motion",
    }
    if history_frames is None:
        result.update(
            {
                "velocity_rmse": math.sqrt(
                    max(float(velocity_loss(prediction, target, batch).detach().cpu()), 0.0)
                ),
                "acceleration_rmse": math.sqrt(
                    max(float(acceleration_loss(prediction, target, batch).detach().cpu()), 0.0)
                ),
                "frequency_retention": _frequency_retention(prediction, target, mask),
            }
        )
        protocol["future_frame_interval"] = [1, total_frames]
    else:
        assert observed is not None
        observed_metrics = one(observed) if observed else one([])
        full_temporal = _temporal_metrics(prediction, target, batch, all_frames)
        observed_temporal = _temporal_metrics(prediction, target, batch, observed)
        future_temporal = _temporal_metrics(prediction, target, batch, future)
        result["observed"] = observed_metrics
        result["boundary"] = _boundary_metrics(prediction, target, batch, history_frames)
        result["temporal"] = {
            "observed": observed_temporal,
            "future": future_temporal,
            "full_diagnostic": full_temporal,
        }
        result["full_diagnostic"] = full_temporal
        result.update(
            {
                "velocity_rmse": future_temporal["velocity_rmse"],
                "acceleration_rmse": future_temporal["acceleration_rmse"],
                "frequency_retention": future_temporal["frequency_retention"],
            }
        )
        protocol.update(
            {
                "history_frames": int(history_frames),
                "observed_frame_interval": [0, int(history_frames)],
                "future_frame_interval": [int(history_frames), total_frames],
                "boundary_frame_interval": (
                    None
                    if int(history_frames) == 0
                    else [int(history_frames) - 1, int(history_frames)]
                ),
                "full_diagnostic_frame_interval": [0, total_frames],
                "temporal_metric_sections": {
                    "observed": "observed_frame_interval",
                    "future": "future_frame_interval",
                    "full_diagnostic": "full_diagnostic_frame_interval",
                },
            }
        )
    result["evaluation_protocol"] = protocol
    return result


@dataclass(frozen=True)
class EvaluationControl:
    name: str
    predictor: Callable[[Any], Any]
    ratio: int
    temporal: bool
    loss_evaluator: Optional[Callable[[Tensor, Any], Mapping[str, float]]] = None


def anchor_control(name: str = "ratio1_no_temporal") -> EvaluationControl:
    def predict(batch: Any) -> Tensor:
        x = torch.as_tensor(_field(batch, "x"))
        return x[0].unsqueeze(0).expand_as(x)
    return EvaluationControl(name=name, predictor=predict, ratio=1, temporal=False)


def model_control(
    name: str,
    model: Callable[[Any], Any],
    *,
    ratio: int,
    temporal: bool,
    loss_evaluator: Optional[Callable[[Tensor, Any], Mapping[str, float]]] = None,
) -> EvaluationControl:
    return EvaluationControl(
        name=name,
        predictor=model,
        ratio=int(ratio),
        temporal=bool(temporal),
        loss_evaluator=loss_evaluator,
    )


def _mean_records(records: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    if not records:
        return {"frame0": {}, "future": {}, "all_frames": {}}
    result: dict[str, Any] = {}
    for section in ("frame0", "future", "all_frames"):
        keys = records[0][section]
        result[section] = {
            key: sum(record[section][key] for record in records) / len(records)
            for key in keys
        }
    for key in ("velocity_rmse", "acceleration_rmse", "frequency_retention"):
        result[key] = sum(record[key] for record in records) / len(records)
    return result


def _mean_loss_records(records: list[Mapping[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = tuple(records[0])
    return {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in keys
    }


def _limited_batches(batches: Iterable[Any], max_batches: Optional[int]) -> Iterable[Any]:
    if max_batches is not None and int(max_batches) < 0:
        raise ValueError("max_batches must be non-negative")
    for index, batch in enumerate(batches):
        if max_batches is not None and index >= int(max_batches):
            break
        yield batch


def evaluate_controls(
    controls: Sequence[EvaluationControl],
    batches: Iterable[Any],
    *,
    max_batches: Optional[int] = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate controls without an unqualified cross-time-bucket mean."""

    output: dict[str, Any] = {"schema_version": EVALUATION_SCHEMA, "controls": {}}
    memory_device = torch.device(device) if device is not None else None
    for control in controls:
        per_bucket: dict[str, dict[str, Any]] = {}
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(memory_device)
        sample_count = 0
        for batch in _limited_batches(batches, max_batches):
            prediction = _prediction(control.predictor(batch))
            target = torch.as_tensor(_field(batch, "x"), device=prediction.device, dtype=prediction.dtype)
            bucket_ids = tuple(str(item) for item in _field(batch, "time_bucket_id"))
            if not bucket_ids or len(set(bucket_ids)) != 1:
                raise ValueError("evaluation requires homogeneous time buckets")
            bucket = bucket_ids[0]
            batch_metrics = _metrics(prediction, target, batch)
            delta = _field(batch, "delta_time_ps")
            frame_count = int(target.shape[0])
            if frame_count > 1 and torch.as_tensor(delta).numel():
                native_delta = float(torch.as_tensor(delta).mean())
                latent_interval = native_delta * control.ratio
            else:
                native_delta = None
                latent_interval = None
            spans = torch.as_tensor(_field(batch, "time_ps"))[:, -1] - torch.as_tensor(_field(batch, "time_ps"))[:, 0]
            atom_count = int(target.shape[1])
            bucket_state = per_bucket.setdefault(
                bucket,
                {
                    "records": [],
                    "loss_records": [],
                    "sample_count": 0,
                    "clip_spans": [],
                    "native_deltas": [],
                    "latent_intervals": [],
                    "latent_tokens": [],
                },
            )
            bucket_state["records"].append(batch_metrics)
            if control.loss_evaluator is not None:
                bucket_state["loss_records"].append(
                    dict(control.loss_evaluator(prediction, batch))
                )
            batch_size = int(getattr(batch, "batch_size", len(bucket_ids)))
            bucket_state["sample_count"] += batch_size
            bucket_state["clip_spans"].extend(float(value) for value in spans.tolist())
            if native_delta is not None:
                bucket_state["native_deltas"].append(native_delta)
                bucket_state["latent_intervals"].append(latent_interval)
            bucket_state["latent_tokens"].append(int(math.ceil(frame_count / control.ratio) * atom_count))
            sample_count += batch_size
        elapsed = time.perf_counter() - start
        by_bucket = {}
        for bucket, state in per_bucket.items():
            by_bucket[bucket] = {
                "sample_count": state["sample_count"],
                "native_delta_time_ps": (sum(state["native_deltas"]) / len(state["native_deltas"]) if state["native_deltas"] else None),
                "physical_clip_span_ps": (sum(state["clip_spans"]) / len(state["clip_spans"]) if state["clip_spans"] else 0.0),
                "latent_interval_ps": (sum(state["latent_intervals"]) / len(state["latent_intervals"]) if state["latent_intervals"] else None),
                "latent_tokens": (sum(state["latent_tokens"]) / len(state["latent_tokens"]) if state["latent_tokens"] else 0.0),
                "metrics": _mean_records(state["records"]),
            }
            if state["loss_records"]:
                by_bucket[bucket]["loss"] = _mean_loss_records(state["loss_records"])
        output["controls"][control.name] = {
            "ratio": control.ratio,
            "temporal": control.temporal,
            "sample_count": sample_count,
            "runtime": {
                "wall_time_s": elapsed,
                "samples_per_s": sample_count / elapsed if elapsed > 0 else 0.0,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(memory_device)) if torch.cuda.is_available() else 0,
            },
            "by_time_bucket": by_bucket,
        }
    return output


def report_markdown(report: Mapping[str, Any]) -> str:
    split = str(report.get("evaluation_data", {}).get("source_split", "valid"))
    split_label = {"train": "Train", "valid": "Validation", "validation": "Validation"}.get(split, split)
    lines = [
        f"# PVB codec round-trip evaluation ({split_label} split)",
        "",
        "Metrics are stratified by native time bucket; no cross-bucket mean is reported.",
        "Aligned RMSD uses per-frame Kabsch on align_mask and evaluates loss_mask; raw RMSD is explicitly centroid_gauge_raw_rmsd.",
        f"Contacts use a {CONTACT_CUTOFF_ANGSTROM:g} Å cutoff and exclude covalent bond pairs only.",
        "",
    ]
    for name, control in report.get("controls", {}).items():
        lines.extend([
            f"## {name}",
            "",
            f"| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | {split_label} total loss | Frame-0 aligned RMSD | Future aligned RMSD | Future raw RMSD | Future dRMSD | Contact F1 | Velocity RMSE | Acceleration RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for bucket, values in control.get("by_time_bucket", {}).items():
            metrics = values["metrics"]
            loss = values.get("loss", {})
            total_loss = loss.get("total")
            loss_text = f"{total_loss:.6g}" if total_loss is not None else "—"
            native_delta = values["native_delta_time_ps"]
            latent_interval = values["latent_interval_ps"]
            lines.append(
                f"| {bucket} | {native_delta if native_delta is not None else '—'} | "
                f"{values['physical_clip_span_ps']:.4g} | "
                f"{latent_interval if latent_interval is not None else '—'} | "
                f"{loss_text} | {metrics['frame0'].get('aligned_rmsd', 0.0):.6g} | "
                f"{metrics['future'].get('aligned_rmsd', 0.0):.6g} | "
                f"{metrics['future'].get('centroid_gauge_raw_rmsd', 0.0):.6g} | "
                f"{metrics['future'].get('drmsd', 0.0):.6g} | "
                f"{metrics['future'].get('contact_f1', 0.0):.6g} | "
                f"{metrics.get('velocity_rmse', 0.0):.6g} | "
                f"{metrics.get('acceleration_rmse', 0.0):.6g} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    import json
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(markdown_path).write_text(report_markdown(report), encoding="utf-8")


__all__ = [
    "EVALUATION_SCHEMA",
    "ALIGNED_RMSD_NAME",
    "RAW_RMSD_NAME",
    "CONTACT_CUTOFF_ANGSTROM",
    "CONTACT_EXCLUSION_RULE",
    "EvaluationControl",
    "aligned_rmsf_metrics",
    "anchor_control",
    "contact_metrics",
    "dynamic_acf_metrics",
    "evaluate_controls",
    "model_control",
    "report_markdown",
    "write_report",
]
