"""Packed clip contract with explicit physical time and atom identity."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..geometry.topology import stable_topology_id


SCHEMA_VERSION = "pvb.clip.v1"
STATIC_TIME_BUCKET_ID = "static"
STATIC_TASK = 0
TRAJECTORY_TASK = 1
TASK_NAMES = {STATIC_TASK: "static", TRAJECTORY_TASK: "trajectory"}


class ClipValidationError(ValueError):
    """Raised when a clip violates the v1 data contract."""


def canonical_time_fields(
    time_ps: Sequence[float] | np.ndarray,
    delta_time_ps: Sequence[float] | np.ndarray | None = None,
    *,
    atol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated physical timestamps and intervals.

    This is the only helper used to derive intervals.  Timestamps are local
    to a clip, but their units are always picoseconds.
    """

    time = np.asarray(time_ps, dtype=np.float64)
    if time.ndim != 1 or time.size == 0:
        raise ClipValidationError("time_ps must be a non-empty 1-D array")
    if not np.isfinite(time).all():
        raise ClipValidationError("time_ps contains NaN or Inf")
    if time.size > 1 and not np.all(np.diff(time) > 0):
        raise ClipValidationError("time_ps must be strictly increasing")

    derived = np.diff(time)
    if delta_time_ps is None:
        delta = derived
    else:
        delta = np.asarray(delta_time_ps, dtype=np.float64)
        if delta.ndim != 1 or delta.size != max(0, time.size - 1):
            raise ClipValidationError(
                "delta_time_ps must have length len(time_ps)-1"
            )
        if not np.isfinite(delta).all() or (delta <= 0).any():
            raise ClipValidationError("delta_time_ps must be finite and positive")
        if not np.allclose(delta, derived, rtol=1e-6, atol=atol):
            raise ClipValidationError(
                "delta_time_ps disagrees with adjacent time_ps values"
            )
    if delta.size and ((delta <= 0).any() or not np.isfinite(delta).all()):
        raise ClipValidationError("derived delta_time_ps must be finite and positive")
    return time.astype(np.float32), delta.astype(np.float32)


def _array(record: Mapping[str, Any], key: str, dtype: np.dtype) -> np.ndarray:
    if key not in record:
        raise ClipValidationError(f"missing required field: {key}")
    return np.asarray(record[key], dtype=dtype)


def _task_id(value: Any) -> int:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "static":
            return STATIC_TASK
        if lowered == "trajectory":
            return TRAJECTORY_TASK
    if isinstance(value, (int, np.integer)) and int(value) in TASK_NAMES:
        return int(value)
    raise ClipValidationError(f"unknown task: {value!r}")


def _bond_array(value: Any) -> np.ndarray:
    bonds = np.asarray(value, dtype=np.int64)
    if bonds.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    if bonds.ndim != 2 or bonds.shape[0] != 2:
        raise ClipValidationError("bond_index must have shape [2, E]")
    return bonds


def validate_clip_record(
    record: Mapping[str, Any],
    *,
    time_bucket_id: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one serialized clip.

    The returned mapping is safe to pass to :func:`collate_clip_records`.
    Pair-only records are rejected rather than being silently upgraded.
    """

    if not isinstance(record, Mapping):
        raise ClipValidationError("clip record must be a mapping")
    if "x0" in record or "x1" in record:
        raise ClipValidationError("pair records are not valid clip records")
    if record.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ClipValidationError(
            f"unsupported schema_version: {record.get('schema_version')!r}"
        )

    x = _array(record, "x", np.float32)
    bpos = _array(record, "bpos", np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ClipValidationError("x must have shape [T, N, 3]")
    if bpos.shape != x.shape:
        raise ClipValidationError("bpos must have the same shape as x")
    if not np.isfinite(x).all() or not np.isfinite(bpos).all():
        raise ClipValidationError("x and bpos must be finite")
    t_len, atom_count = x.shape[:2]

    atype = _array(record, "atype", np.int64)
    btype = _array(record, "btype", np.int64)
    block_id = _array(record, "block_id", np.int64)
    component_id = _array(record, "component_id", np.int64)
    atom_source_index = _array(record, "atom_source_index", np.int64)
    loss_mask = _array(record, "loss_mask", np.bool_)
    align_mask = _array(record, "align_mask", np.bool_)
    edge_mask = _array(record, "edge_mask", np.int64)
    one_d = {
        "atype": atype,
        "btype": btype,
        "block_id": block_id,
        "component_id": component_id,
        "atom_source_index": atom_source_index,
        "loss_mask": loss_mask,
        "align_mask": align_mask,
        "edge_mask": edge_mask,
    }
    for key, value in one_d.items():
        if value.ndim != 1 or value.size != atom_count:
            raise ClipValidationError(f"{key} must have shape [N]")
    if np.unique(atom_source_index).size != atom_count:
        raise ClipValidationError("atom_source_index must preserve unique atom order")
    if not align_mask.any():
        raise ClipValidationError("align_mask must contain at least one atom")
    if (loss_mask & ~np.isfinite(x).all(axis=(0, 2))).any():
        raise ClipValidationError("loss_mask selects non-finite coordinates")

    bonds = _bond_array(record.get("bond_index", []))
    if bonds.size and ((bonds < 0).any() or (bonds >= atom_count).any()):
        raise ClipValidationError("bond_index contains an out-of-range atom")

    time, delta = canonical_time_fields(
        record.get("time_ps", []), record.get("delta_time_ps")
    )
    if time.size != t_len:
        raise ClipValidationError("time_ps length must equal x.shape[0]")
    task = _task_id(record.get("task", "trajectory"))
    if task == STATIC_TASK and t_len != 1:
        raise ClipValidationError("static clips must have T=1")
    if task == TRAJECTORY_TASK and t_len < 2:
        raise ClipValidationError("trajectory clips must have T>=2")

    bucket = str(record.get("time_bucket_id", ""))
    if not bucket:
        raise ClipValidationError("time_bucket_id is required")
    if time_bucket_id is not None and bucket != time_bucket_id:
        raise ClipValidationError(
            f"mixed time buckets: expected {time_bucket_id!r}, got {bucket!r}"
        )
    identities = record.get("atom_identity")
    if not isinstance(identities, list) or len(identities) != atom_count:
        raise ClipValidationError("atom_identity must be a list with one entry per atom")
    if len(set(map(str, identities))) != atom_count:
        raise ClipValidationError("atom_identity entries must be unique")

    normalized = dict(record)
    normalized.update(
        {
            "schema_version": SCHEMA_VERSION,
            "x": x,
            "bpos": bpos,
            "atype": atype,
            "btype": btype,
            "block_id": block_id,
            "component_id": component_id,
            "atom_source_index": atom_source_index,
            "loss_mask": loss_mask,
            "align_mask": align_mask,
            "edge_mask": edge_mask,
            "bond_index": bonds,
            "time_ps": time,
            "delta_time_ps": delta,
            "task": task,
            "time_bucket_id": bucket,
        }
    )
    return normalized


@dataclass
class ClipBatch:
    """Packed time-major batch consumed by the future codec path."""

    x: torch.Tensor  # [T, N_total, 3]
    bpos: torch.Tensor  # [T, N_total, 3]
    atype: torch.Tensor  # [N_total]
    btype: torch.Tensor  # [N_total]
    block_id: torch.Tensor  # [N_total]
    component_id: torch.Tensor  # [N_total]
    atom_source_index: torch.Tensor  # [N_total]
    abid: torch.Tensor  # [N_total]
    atom_ptr: torch.Tensor  # [B+1]
    bond_index: torch.Tensor  # [2, E_total]
    edge_mask: torch.Tensor  # [N_total]
    loss_mask: torch.Tensor  # [N_total]
    align_mask: torch.Tensor  # [N_total]
    frame_mask: torch.Tensor  # [B, T]
    time_ps: torch.Tensor  # [B, T]
    delta_time_ps: torch.Tensor  # [B, T-1]
    time_bucket_id: tuple[str, ...]
    topology_id: tuple[str, ...]
    task: torch.Tensor  # [B]
    sample_id: tuple[str, ...]
    # Immutable host metadata populated by collate. It is deliberately a
    # tuple so ClipBatch.to(cuda) never has to read atom_ptr back from the
    # device in the graph hot path.
    atom_counts: tuple[int, ...] = ()
    atom_identity_sha256: tuple[str, ...] = ()
    host_task_ids: tuple[int, ...] = ()

    @property
    def batch_size(self) -> int:
        return int(self.atom_ptr.numel() - 1)

    @property
    def host_atom_counts(self) -> tuple[int, ...]:
        """Return prevalidated per-sample atom counts without CUDA sync."""

        if self.atom_counts:
            if len(self.atom_counts) != self.batch_size:
                raise ClipValidationError("atom_counts must match atom_ptr batch size")
            return tuple(int(value) for value in self.atom_counts)
        if self.atom_ptr.device.type != "cpu":
            raise RuntimeError(
                "CUDA ClipBatch is missing host atom_counts; collate the CPU batch "
                "before transfer"
            )
        values = tuple(
            int(self.atom_ptr[index + 1] - self.atom_ptr[index])
            for index in range(self.batch_size)
        )
        return values

    @property
    def host_atom_identity_hashes(self) -> tuple[str, ...]:
        """Return immutable per-sample atom-identity hashes from CPU collation."""

        if self.atom_identity_sha256:
            if len(self.atom_identity_sha256) != self.batch_size:
                raise ClipValidationError(
                    "atom_identity_sha256 must match atom_ptr batch size"
                )
            return tuple(str(value) for value in self.atom_identity_sha256)
        if self.atom_ptr.device.type != "cpu":
            raise RuntimeError(
                "CUDA ClipBatch is missing host atom_identity_sha256; "
                "collate the CPU batch before transfer"
            )
        return ()

    @property
    def frames(self) -> int:
        return int(self.x.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.x.shape[1])

    def pin_memory(self) -> "ClipBatch":
        """Pin every tensor field for a non-blocking CUDA transfer."""

        return replace(
            self,
            **{
                field.name: getattr(self, field.name).pin_memory()
                if isinstance(getattr(self, field.name), torch.Tensor)
                else getattr(self, field.name)
                for field in fields(self)
            },
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ClipBatch":
        """Move tensor fields while preserving typed clip metadata."""

        return replace(
            self,
            **{
                field.name: getattr(self, field.name).to(
                    device=device, non_blocking=non_blocking
                )
                if isinstance(getattr(self, field.name), torch.Tensor)
                else getattr(self, field.name)
                for field in fields(self)
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def collate_clip_records(
    records: Sequence[Mapping[str, Any]],
    *,
    validate: bool = True,
) -> ClipBatch:
    """Pack variable-atom records into the canonical ``[T, N_total, 3]`` form."""

    if not records:
        raise ClipValidationError("cannot collate an empty clip batch")
    normalized = [
        validate_clip_record(record) if validate else dict(record)
        for record in records
    ]
    first = normalized[0]
    task = int(first["task"])
    frames = int(first["x"].shape[0])
    bucket = str(first["time_bucket_id"])
    for item in normalized[1:]:
        if int(item["task"]) != task:
            raise ClipValidationError("mixed task types are not allowed in one batch")
        if int(item["x"].shape[0]) != frames:
            raise ClipValidationError("mixed temporal lengths are not allowed in one batch")
        if str(item["time_bucket_id"]) != bucket:
            raise ClipValidationError("mixed sampling-interval buckets are not allowed")

    atom_ptr = [0]
    abid_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    bpos_parts: list[np.ndarray] = []
    flat_fields = {
        key: []
        for key in (
            "atype",
            "btype",
            "block_id",
            "component_id",
            "atom_source_index",
            "edge_mask",
            "loss_mask",
            "align_mask",
        )
    }
    bond_parts: list[np.ndarray] = []
    times: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    sample_ids: list[str] = []
    topology_ids: list[str] = []
    identity_hashes: list[str] = []
    buckets: list[str] = []
    tasks: list[int] = []

    for sample_idx, item in enumerate(normalized):
        n_atoms = int(item["x"].shape[1])
        atom_ptr.append(atom_ptr[-1] + n_atoms)
        x_parts.append(item["x"])
        bpos_parts.append(item["bpos"])
        abid_parts.append(np.full(n_atoms, sample_idx, dtype=np.int64))
        for key in flat_fields:
            flat_fields[key].append(item[key])
        bonds = item["bond_index"].copy()
        if bonds.size:
            bonds += atom_ptr[-2]
        bond_parts.append(bonds)
        times.append(item["time_ps"])
        deltas.append(item["delta_time_ps"])
        sample_ids.append(str(item.get("sample_id", sample_idx)))
        topology_ids.append(stable_topology_id(item))
        identity_payload = json.dumps(
            [str(value) for value in item["atom_identity"]],
            separators=(",", ":"),
        ).encode()
        identity_hashes.append(hashlib.sha256(identity_payload).hexdigest())
        buckets.append(str(item["time_bucket_id"]))
        tasks.append(int(item["task"]))

    if bond_parts and any(b.size for b in bond_parts):
        packed_bonds = np.concatenate(
            [b for b in bond_parts if b.size], axis=1
        )
    else:
        packed_bonds = np.empty((2, 0), dtype=np.int64)

    return ClipBatch(
        x=torch.from_numpy(np.concatenate(x_parts, axis=1)),
        bpos=torch.from_numpy(np.concatenate(bpos_parts, axis=1)),
        atype=torch.from_numpy(np.concatenate(flat_fields["atype"])),
        btype=torch.from_numpy(np.concatenate(flat_fields["btype"])),
        block_id=torch.from_numpy(np.concatenate(flat_fields["block_id"])),
        component_id=torch.from_numpy(np.concatenate(flat_fields["component_id"])),
        atom_source_index=torch.from_numpy(
            np.concatenate(flat_fields["atom_source_index"])
        ),
        abid=torch.from_numpy(np.concatenate(abid_parts)),
        atom_ptr=torch.tensor(atom_ptr, dtype=torch.long),
        bond_index=torch.from_numpy(packed_bonds),
        edge_mask=torch.from_numpy(np.concatenate(flat_fields["edge_mask"])),
        loss_mask=torch.from_numpy(np.concatenate(flat_fields["loss_mask"])),
        align_mask=torch.from_numpy(np.concatenate(flat_fields["align_mask"])),
        frame_mask=torch.ones((len(normalized), frames), dtype=torch.bool),
        time_ps=torch.from_numpy(np.stack(times, axis=0)),
        delta_time_ps=torch.from_numpy(np.stack(deltas, axis=0)),
        time_bucket_id=tuple(buckets),
        topology_id=tuple(topology_ids),
        task=torch.tensor(tasks, dtype=torch.long),
        sample_id=tuple(sample_ids),
        atom_counts=tuple(int(value) for value in np.diff(np.asarray(atom_ptr))),
        atom_identity_sha256=tuple(identity_hashes),
        host_task_ids=tuple(tasks),
    )
