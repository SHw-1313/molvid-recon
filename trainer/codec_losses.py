"""Masked physical-unit losses for the multi-frame PVB codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor

from data.clip_dataset import STATIC_TASK, TRAJECTORY_TASK


_MISSING = object()


def _field(batch: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(batch, Mapping):
        if name in batch:
            return batch[name]
    elif hasattr(batch, name):
        return getattr(batch, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"batch is missing required field {name!r}")


def _bucket_ids(batch: Any) -> tuple[str, ...]:
    value = _field(batch, "time_bucket_id")
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _check_coordinates(prediction: Tensor, target: Tensor) -> None:
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [T, N, 3]")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target must have shape {tuple(prediction.shape)}, got {tuple(target.shape)}"
        )


def _frame_atom_mask(batch: Any, frames: int, device: torch.device, *, temporal: bool = False) -> Tensor:
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=device, dtype=torch.bool)
    abid = torch.as_tensor(_field(batch, "abid"), device=device, dtype=torch.long).flatten()
    atom_mask = torch.as_tensor(_field(batch, "loss_mask"), device=device, dtype=torch.bool).flatten()
    if frame_mask.ndim != 2 or frame_mask.shape[1] != frames:
        raise ValueError("frame_mask must have shape [B, T] matching the coordinates")
    if abid.numel() != atom_mask.numel():
        raise ValueError("abid and loss_mask must contain one entry per atom")
    if abid.numel() and (abid.min() < 0 or abid.max() >= frame_mask.shape[0]):
        raise ValueError("abid contains an invalid sample id")
    valid = frame_mask.index_select(0, abid).transpose(0, 1)
    valid = valid & atom_mask.unsqueeze(0)
    if temporal:
        task = torch.as_tensor(_field(batch, "task"), device=device, dtype=torch.long).flatten()
        atom_task = task.index_select(0, abid)
        valid = valid & atom_task.eq(TRAJECTORY_TASK).unsqueeze(0)
    return valid


def _pair_index(batch: Any, pair_index: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
    if pair_index is None:
        pair_index = _field(batch, "bond_index", default=None)
    if pair_index is None:
        return None
    pair_index = torch.as_tensor(pair_index, device=device, dtype=torch.long)
    if pair_index.numel() == 0:
        return None
    if pair_index.ndim != 2 or pair_index.shape[0] != 2:
        raise ValueError("pair indices must have shape [2, E]")
    atoms = int(torch.as_tensor(_field(batch, "loss_mask")).numel())
    if torch.any(pair_index < 0) or torch.any(pair_index >= atoms):
        raise ValueError("pair index contains an out-of-range atom")
    return pair_index


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[: mask.ndim] != mask.shape:
        raise ValueError("loss values and mask have incompatible shapes")
    if not torch.any(mask):
        return _zero(values)
    return values[mask].mean()


def masked_coordinate_loss(prediction: Tensor, target: Tensor, batch: Any) -> Tensor:
    """Mean squared coordinate error over valid frames and atoms."""

    _check_coordinates(prediction, target)
    mask = _frame_atom_mask(batch, prediction.shape[0], prediction.device)
    return _masked_mean((prediction - target).square().mean(dim=-1), mask)


def pair_distance_loss(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    pair_index: Optional[Tensor] = None,
) -> Tensor:
    """MSE of pair distances for local/contact or covalent pairs."""

    _check_coordinates(prediction, target)
    pairs = _pair_index(batch, pair_index, prediction.device)
    if pairs is None:
        return _zero(prediction)
    source, destination = pairs
    pred_distance = torch.linalg.vector_norm(
        prediction[:, source] - prediction[:, destination], dim=-1
    )
    target_distance = torch.linalg.vector_norm(
        target[:, source] - target[:, destination], dim=-1
    )
    atom_mask = _frame_atom_mask(batch, prediction.shape[0], prediction.device)
    pair_mask = atom_mask[:, source] & atom_mask[:, destination]
    return _masked_mean((pred_distance - target_distance).square(), pair_mask)


def local_contact_distance_loss(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    contact_index: Optional[Tensor] = None,
) -> Tensor:
    """Local/contact distance loss; bond pairs are the explicit fallback."""

    return pair_distance_loss(
        prediction, target, batch, pair_index=contact_index
    )


def bond_length_loss(prediction: Tensor, target: Tensor, batch: Any) -> Tensor:
    """MSE of covalent bond lengths from the packed bond index."""

    return pair_distance_loss(prediction, target, batch)


def _delta_time(batch: Any, frames: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if frames < 2:
        return torch.empty((0, 0), device=device, dtype=dtype)
    delta = torch.as_tensor(_field(batch, "delta_time_ps"), device=device, dtype=dtype)
    if delta.ndim != 2 or delta.shape[1] != frames - 1:
        raise ValueError("delta_time_ps must have shape [B, T-1]")
    if not torch.isfinite(delta).all() or torch.any(delta <= 0):
        raise ValueError("delta_time_ps must be finite and positive")
    return delta


def _trajectory_available(batch: Any, device: torch.device) -> bool:
    task = torch.as_tensor(_field(batch, "task"), device=device, dtype=torch.long)
    return bool(torch.any(task == TRAJECTORY_TASK))


def _velocity_and_mask(prediction: Tensor, batch: Any) -> tuple[Optional[Tensor], Optional[Tensor]]:
    frames = int(prediction.shape[0])
    if frames < 2 or not _trajectory_available(batch, prediction.device):
        return None, None
    delta = _delta_time(batch, frames, prediction.device, prediction.dtype)
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long)
    dt = delta.index_select(0, abid).transpose(0, 1)
    velocity = (prediction[1:] - prediction[:-1]) / dt.unsqueeze(-1)
    valid = _frame_atom_mask(batch, frames, prediction.device, temporal=True)
    return velocity, valid[:-1] & valid[1:]


def _acceleration_and_mask(prediction: Tensor, batch: Any) -> tuple[Optional[Tensor], Optional[Tensor]]:
    frames = int(prediction.shape[0])
    if frames < 3 or not _trajectory_available(batch, prediction.device):
        return None, None
    delta = _delta_time(batch, frames, prediction.device, prediction.dtype)
    abid = torch.as_tensor(_field(batch, "abid"), device=prediction.device, dtype=torch.long)
    dt = delta.index_select(0, abid).transpose(0, 1)
    velocity_prev = (prediction[1:-1] - prediction[:-2]) / dt[:-1].unsqueeze(-1)
    velocity_next = (prediction[2:] - prediction[1:-1]) / dt[1:].unsqueeze(-1)
    # Centered derivative on a nonuniform grid: divide the velocity change by
    # the average of the adjacent physical intervals.
    acceleration = 2.0 * (velocity_next - velocity_prev) / (
        dt[:-1] + dt[1:]
    ).unsqueeze(-1)
    valid = _frame_atom_mask(batch, frames, prediction.device, temporal=True)
    return acceleration, valid[:-2] & valid[1:-1] & valid[2:]


def velocity_loss(prediction: Tensor, target: Tensor, batch: Any) -> Tensor:
    """MSE of physical velocities using the explicit per-interval clock."""

    _check_coordinates(prediction, target)
    pred_velocity, mask = _velocity_and_mask(prediction, batch)
    target_velocity, target_mask = _velocity_and_mask(target, batch)
    if pred_velocity is None or target_velocity is None:
        return _zero(prediction)
    assert mask is not None and target_mask is not None
    return _masked_mean((pred_velocity - target_velocity).square().mean(dim=-1), mask & target_mask)


def acceleration_loss(prediction: Tensor, target: Tensor, batch: Any) -> Tensor:
    """MSE of centered nonuniform-grid accelerations in physical units."""

    _check_coordinates(prediction, target)
    pred_acceleration, mask = _acceleration_and_mask(prediction, batch)
    target_acceleration, target_mask = _acceleration_and_mask(target, batch)
    if pred_acceleration is None or target_acceleration is None:
        return _zero(prediction)
    assert mask is not None and target_mask is not None
    return _masked_mean(
        (pred_acceleration - target_acceleration).square().mean(dim=-1),
        mask & target_mask,
    )


@dataclass(frozen=True)
class CodecLossWeights:
    coordinate: float = 1.0
    local: float = 0.0
    bond: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CodecLossWeights":
        if value is None:
            return cls()
        result = cls(**{key: float(value[key]) for key in cls.__dataclass_fields__ if key in value})
        if any(weight < 0 or not torch.isfinite(torch.tensor(weight)) for weight in result.__dict__.values()):
            raise ValueError("loss weights must be finite and non-negative")
        return result

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


@dataclass
class BucketNormalization:
    bucket_id: str
    count: int
    coordinate_mean: Tensor
    coordinate_std: Tensor
    velocity_scale: float
    acceleration_scale: float
    min_count: int
    epsilon: float
    used_fallback: bool = False

    def __post_init__(self) -> None:
        self.bucket_id = str(self.bucket_id)
        self.count = int(self.count)
        self.min_count = int(self.min_count)
        self.epsilon = float(self.epsilon)
        self.coordinate_mean = torch.as_tensor(self.coordinate_mean, dtype=torch.float32).reshape(3).cpu()
        self.coordinate_std = torch.as_tensor(self.coordinate_std, dtype=torch.float32).reshape(3).clamp_min(self.epsilon).cpu()
        self.velocity_scale = max(float(self.velocity_scale), self.epsilon)
        self.acceleration_scale = max(float(self.acceleration_scale), self.epsilon)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket_id": self.bucket_id,
            "count": self.count,
            "coordinate_mean": self.coordinate_mean.tolist(),
            "coordinate_std": self.coordinate_std.tolist(),
            "velocity_scale": self.velocity_scale,
            "acceleration_scale": self.acceleration_scale,
            "min_count": self.min_count,
            "epsilon": self.epsilon,
            "used_fallback": bool(self.used_fallback),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BucketNormalization":
        required = {"bucket_id", "count", "coordinate_mean", "coordinate_std", "velocity_scale", "acceleration_scale", "min_count", "epsilon"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"normalization entry is missing fields: {sorted(missing)}")
        return cls(
            bucket_id=str(value["bucket_id"]),
            count=int(value["count"]),
            coordinate_mean=value["coordinate_mean"],
            coordinate_std=value["coordinate_std"],
            velocity_scale=float(value["velocity_scale"]),
            acceleration_scale=float(value["acceleration_scale"]),
            min_count=int(value["min_count"]),
            epsilon=float(value["epsilon"]),
            used_fallback=bool(value.get("used_fallback", False)),
        )


def _stats_temporal_values(target: Tensor, batch: Any, *, acceleration: bool) -> tuple[Optional[Tensor], Optional[Tensor]]:
    if acceleration:
        return _acceleration_and_mask(target, batch)
    return _velocity_and_mask(target, batch)


def fit_time_bucket_normalization(
    batches: Iterable[Any],
    *,
    min_count: int = 32,
    epsilon: float = 1e-6,
) -> dict[str, BucketNormalization]:
    """Fit train-split coordinate/motion scales with guarded fallbacks."""

    if int(min_count) < 1 or float(epsilon) <= 0:
        raise ValueError("min_count must be positive and epsilon must be positive")
    accum: dict[str, dict[str, Any]] = {}
    for batch in batches:
        target = torch.as_tensor(_field(batch, "x"), dtype=torch.float32)
        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError("batch x must have shape [T, N, 3]")
        device = target.device
        frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=device, dtype=torch.bool)
        abid = torch.as_tensor(_field(batch, "abid"), device=device, dtype=torch.long)
        sample_buckets = _bucket_ids(batch)
        if len(sample_buckets) != frame_mask.shape[0]:
            raise ValueError("time_bucket_id must have one entry per sample")
        atom_buckets = tuple(sample_buckets[int(index)] for index in abid.cpu().tolist())
        coordinate_valid = _frame_atom_mask(batch, target.shape[0], device)
        velocities, velocity_mask = _stats_temporal_values(target, batch, acceleration=False)
        accelerations, acceleration_mask = _stats_temporal_values(target, batch, acceleration=True)
        for bucket in dict.fromkeys(sample_buckets):
            state = accum.setdefault(
                bucket,
                {
                    "count": 0,
                    "sum": torch.zeros(3),
                    "sum_sq": torch.zeros(3),
                    "velocity_sq": 0.0,
                    "velocity_count": 0,
                    "acceleration_sq": 0.0,
                    "acceleration_count": 0,
                },
            )
            atom_bucket_mask = torch.tensor(
                [item == bucket for item in atom_buckets], device=device, dtype=torch.bool
            )
            coord_mask = coordinate_valid & atom_bucket_mask.unsqueeze(0)
            values = target[coord_mask].detach().cpu()
            if values.numel():
                state["count"] += int(values.shape[0])
                state["sum"] += values.sum(0)
                state["sum_sq"] += values.square().sum(0)
            if velocities is not None and velocity_mask is not None:
                motion_mask = velocity_mask & atom_bucket_mask.unsqueeze(0)
                values = velocities[motion_mask].detach().cpu()
                if values.numel():
                    state["velocity_sq"] += float(values.square().sum())
                    state["velocity_count"] += int(values.numel())
            if accelerations is not None and acceleration_mask is not None:
                motion_mask = acceleration_mask & atom_bucket_mask.unsqueeze(0)
                values = accelerations[motion_mask].detach().cpu()
                if values.numel():
                    state["acceleration_sq"] += float(values.square().sum())
                    state["acceleration_count"] += int(values.numel())

    result: dict[str, BucketNormalization] = {}
    for bucket, state in accum.items():
        count = int(state["count"])
        fallback = count < int(min_count)
        if fallback:
            mean = torch.zeros(3)
            std = torch.ones(3)
            velocity_scale = 1.0
            acceleration_scale = 1.0
        else:
            mean = state["sum"] / max(count, 1)
            variance = (state["sum_sq"] / max(count, 1) - mean.square()).clamp_min(0.0)
            std = variance.sqrt().clamp_min(float(epsilon))
            velocity_scale = (
                state["velocity_sq"] / max(state["velocity_count"], 1)
            ) ** 0.5 if state["velocity_count"] else 1.0
            acceleration_scale = (
                state["acceleration_sq"] / max(state["acceleration_count"], 1)
            ) ** 0.5 if state["acceleration_count"] else 1.0
        result[bucket] = BucketNormalization(
            bucket_id=bucket,
            count=count,
            coordinate_mean=mean,
            coordinate_std=std,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            min_count=int(min_count),
            epsilon=float(epsilon),
            used_fallback=fallback,
        )
    return result


def _normalization_for(
    normalization: Mapping[str, BucketNormalization | Mapping[str, Any]] | None,
    batch: Any,
) -> Optional[BucketNormalization]:
    if not normalization:
        return None
    buckets = _bucket_ids(batch)
    if not buckets or len(set(buckets)) != 1:
        raise ValueError("loss normalization requires one homogeneous time bucket")
    value = normalization.get(buckets[0])
    if value is None:
        return None
    return value if isinstance(value, BucketNormalization) else BucketNormalization.from_dict(value)


def compute_codec_losses(
    prediction: Any,
    batch: Any,
    *,
    target: Optional[Tensor] = None,
    weights: CodecLossWeights | Mapping[str, Any] | None = None,
    contact_index: Optional[Tensor] = None,
    normalization: Mapping[str, BucketNormalization | Mapping[str, Any]] | None = None,
) -> dict[str, Tensor]:
    """Compute raw metrics and normalized weighted losses for one clip batch."""

    if isinstance(prediction, Tensor):
        coordinates = prediction
    elif isinstance(prediction, Mapping) and "x_hat" in prediction:
        coordinates = prediction["x_hat"]
    elif hasattr(prediction, "x_hat"):
        coordinates = prediction.x_hat
    else:
        raise TypeError("prediction must be a [T,N,3] tensor or expose x_hat")
    # Geometry and all reductions are evaluated in FP32 even when the model
    # forward is enclosed by BF16 autocast.
    coordinates = coordinates.float()
    if target is None:
        target = torch.as_tensor(
            _field(batch, "x"), device=coordinates.device, dtype=torch.float32
        )
    else:
        target = torch.as_tensor(
            target, device=coordinates.device, dtype=torch.float32
        )
    _check_coordinates(coordinates, target)
    selected = weights if isinstance(weights, CodecLossWeights) else CodecLossWeights.from_mapping(weights)
    norm = _normalization_for(normalization, batch)
    coordinate = masked_coordinate_loss(coordinates, target, batch)
    local = local_contact_distance_loss(coordinates, target, batch, contact_index=contact_index)
    bond = bond_length_loss(coordinates, target, batch)
    velocity_raw = velocity_loss(coordinates, target, batch)
    acceleration_raw = acceleration_loss(coordinates, target, batch)
    velocity = velocity_raw / (norm.velocity_scale ** 2) if norm is not None else velocity_raw
    acceleration = acceleration_raw / (norm.acceleration_scale ** 2) if norm is not None else acceleration_raw
    total = (
        selected.coordinate * coordinate
        + selected.local * local
        + selected.bond * bond
        + selected.velocity * velocity
        + selected.acceleration * acceleration
    )
    return {
        "total": total,
        "coordinate": coordinate,
        "local": local,
        "bond": bond,
        "velocity": velocity,
        "acceleration": acceleration,
        "velocity_raw": velocity_raw,
        "acceleration_raw": acceleration_raw,
    }


__all__ = [
    "BucketNormalization",
    "CodecLossWeights",
    "acceleration_loss",
    "bond_length_loss",
    "compute_codec_losses",
    "fit_time_bucket_normalization",
    "local_contact_distance_loss",
    "masked_coordinate_loss",
    "pair_distance_loss",
    "velocity_loss",
]
