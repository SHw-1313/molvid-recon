"""Versioned packed clip contract and mmap storage for trajectory data.

The legacy PVB datasets expose independent ``x0``/``x1`` records.  This
module deliberately uses a separate schema: one item is an ordered clip with
one topology and an explicit physical clock.  The writer uses a versioned
NumPy binary mmap payload with JSON metadata.
"""

from __future__ import annotations

import hashlib
import io
import json
import mmap
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "pvb.clip.v1"
STORAGE_FORMAT = "npz-v1"
STATIC_TIME_BUCKET_ID = "static"
NPZ_ARRAY_KEYS = (
    "x",
    "bpos",
    "atype",
    "btype",
    "block_id",
    "component_id",
    "atom_source_index",
    "edge_mask",
    "loss_mask",
    "align_mask",
    "bond_index",
    "time_ps",
    "delta_time_ps",
)
STATIC_TASK = 0
TRAJECTORY_TASK = 1
TASK_NAMES = {STATIC_TASK: "static", TRAJECTORY_TASK: "trajectory"}


def stable_topology_id(record: Mapping[str, Any], *, require_stable: bool = False) -> str:
    """Derive the stable graph-topology identifier stored in a clip batch."""

    explicit = record.get("topology_id")
    if explicit not in (None, ""):
        return str(explicit)
    system = record.get("system_id")
    fingerprint = record.get("topology_fingerprint")
    if system not in (None, "") and fingerprint not in (None, ""):
        return f"{system}::{fingerprint}"
    if system not in (None, ""):
        return str(system)
    if fingerprint not in (None, ""):
        return str(fingerprint)
    if require_stable:
        raise ClipValidationError(
            "a stable topology_id/system_id/topology_fingerprint is required"
        )
    return str(record.get("sample_id", ""))


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


def _stable_block_ids(bpos: np.ndarray, btype: np.ndarray) -> np.ndarray:
    """Recover stable block ids from legacy block-center positions and types."""

    ids = np.empty(bpos.shape[0], dtype=np.int64)
    seen: dict[tuple[int, tuple[float, float, float]], int] = {}
    next_id = 0
    for index, (position, block_type) in enumerate(zip(bpos, btype)):
        key = (int(block_type), tuple(float(value) for value in position))
        if key not in seen:
            seen[key] = next_id
            next_id += 1
        ids[index] = seen[key]
    return ids


def legacy_static_record_to_clip(
    legacy_record: Mapping[str, Any],
    *,
    sample_id: str,
    source: str,
    split: str,
    legacy_id: str | None = None,
) -> dict[str, Any]:
    """Adapt one existing static ``x0``/``b0`` block record to ``T=1``.

    The old stores contain only the selected block order, so atom identities
    are explicit stable identifiers for that stored order.  No coordinate
    frame is replicated and no legacy record is rewritten.
    """

    if not isinstance(legacy_record, Mapping):
        raise ClipValidationError("legacy static record must be a mapping")
    x0 = _array(legacy_record, "x0", np.float32)
    b0 = _array(legacy_record, "b0", np.float32)
    if x0.ndim != 2 or x0.shape[-1] != 3:
        raise ClipValidationError("legacy x0 must have shape [N, 3]")
    if b0.shape != x0.shape:
        raise ClipValidationError("legacy b0 must have the same shape as x0")
    atom_count = int(x0.shape[0])
    atype = _array(legacy_record, "atype", np.int64)
    btype = _array(legacy_record, "btype", np.int64)
    if atype.ndim != 1 or atype.size != atom_count:
        raise ClipValidationError("legacy atype must have shape [N]")
    if btype.ndim != 1 or btype.size != atom_count:
        raise ClipValidationError("legacy btype must have shape [N]")

    edge_mask = np.asarray(
        legacy_record.get("edge_mask", np.zeros(atom_count, dtype=np.int64)),
        dtype=np.int64,
    )
    loss_mask = np.asarray(
        legacy_record.get("mask", np.ones(atom_count, dtype=np.bool_)),
        dtype=np.bool_,
    )
    if edge_mask.ndim != 1 or edge_mask.size != atom_count:
        raise ClipValidationError("legacy edge_mask must have shape [N]")
    if loss_mask.ndim != 1 or loss_mask.size != atom_count:
        raise ClipValidationError("legacy mask must have shape [N]")

    legacy_name = str(legacy_id if legacy_id is not None else sample_id)
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "source": str(source),
        "split": str(split),
        "task": "static",
        "coordinate_unit": "angstrom",
        "time_bucket_id": STATIC_TIME_BUCKET_ID,
        "time_ps": np.asarray([0.0], dtype=np.float32),
        "delta_time_ps": np.empty(0, dtype=np.float32),
        "x": x0[None, :, :],
        "bpos": b0[None, :, :],
        "atype": atype,
        "btype": btype,
        "block_id": _stable_block_ids(b0, btype),
        "component_id": edge_mask.copy(),
        "atom_source_index": np.arange(atom_count, dtype=np.int64),
        "atom_identity": [
            f"{source}:{split}:{legacy_name}:atom:{index}"
            for index in range(atom_count)
        ],
        "edge_mask": edge_mask,
        "loss_mask": loss_mask,
        # Static records are not aligned; all stored atoms are valid anchors.
        "align_mask": np.ones(atom_count, dtype=np.bool_),
        "bond_index": _bond_array(legacy_record.get("bond_index", [])),
        "legacy_id": legacy_name,
        "atom_identity_provenance": "legacy_block_order",
        "static_alignment": "none",
    }
    return validate_clip_record(record)


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def source_inventory_hash(entries: Iterable[Mapping[str, Any]]) -> str:
    """Hash a deterministic path/size/mtime inventory without reading raw data."""

    payload = "\n".join(
        json.dumps(_jsonable(dict(item)), sort_keys=True, separators=(",", ":"))
        for item in sorted(entries, key=lambda item: str(item["path"]))
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class ClipMMapWriter:
    """Append validated clip records to a versioned mmap directory.

    ``npz-v1`` stores numeric arrays as NumPy binary members and keeps only
    scalar/list metadata in one JSON member.  The v1 store has one decoding
    path; obsolete gzip/JSON payloads are not accepted.
    """

    def __init__(self, root: str | os.PathLike[str], *, overwrite: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_path = self.root / "data.bin"
        self.index_path = self.root / "index.txt"
        if not overwrite and (self.data_path.exists() or self.index_path.exists()):
            raise FileExistsError(
                f"refusing to overwrite existing clip store: {self.root}"
            )
        if overwrite:
            for path in (self.data_path, self.index_path, self.root / "stats.json"):
                if path.exists():
                    path.unlink()
        self._data = self.data_path.open("wb")
        self._index = self.index_path.open("w", encoding="utf-8")
        self.count = 0
        self.bytes = 0

    def append(self, record: Mapping[str, Any]) -> None:
        normalized = validate_clip_record(record)
        sample_id = str(normalized.get("sample_id", self.count))
        arrays = {
            key: np.asarray(normalized[key])
            for key in NPZ_ARRAY_KEYS
        }
        metadata = {
            key: value
            for key, value in normalized.items()
            if key not in NPZ_ARRAY_KEYS
        }
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            metadata=np.asarray(
                json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":"))
            ),
            **arrays,
        )
        compressed = buffer.getvalue()
        start = self.bytes
        self._data.write(compressed)
        self.bytes += len(compressed)
        frames, atoms = normalized["x"].shape[:2]
        self._index.write(
            f"{sample_id}\t{start}\t{self.bytes}\t{atoms}\t{frames}\t"
            f"{normalized['time_bucket_id']}\n"
        )
        self.count += 1

    def close(self) -> None:
        if self._data.closed:
            return
        self._data.flush()
        self._index.flush()
        self._data.close()
        self._index.close()
        stats = {
            "count": self.count,
            "compressed_bytes": self.bytes,
            "storage_format": STORAGE_FORMAT,
        }
        (self.root / "stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def __enter__(self) -> "ClipMMapWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ClipMMapDataset(torch.utils.data.Dataset):
    """Read the clip store without changing the legacy ``MMAPDataset``."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self._index: list[tuple[str, int, int]] = []
        self._clip_index_fields: list[tuple[str, int, int, int, int, str] | None] = []
        self._data_file = None
        self._mmap = None
        with (self.root / "index.txt").open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    raise ClipValidationError(f"malformed index line: {line!r}")
                sample_id, start, end = fields[:3]
                start_int, end_int = int(start), int(end)
                self._index.append((sample_id, start_int, end_int))
                if len(fields) >= 6:
                    try:
                        self._clip_index_fields.append(
                            (
                                sample_id,
                                start_int,
                                end_int,
                                int(fields[3]),
                                int(fields[4]),
                                str(fields[5]),
                            )
                        )
                    except ValueError:
                        self._clip_index_fields.append(None)
                else:
                    self._clip_index_fields.append(None)
        self._validate_store_metadata()
        self._open_handles()

    def _validate_store_metadata(self) -> None:
        data_path = self.root / "data.bin"
        if not data_path.is_file():
            raise ClipValidationError(f"clip store is missing data.bin: {data_path}")
        stats_path = self.root / "stats.json"
        if stats_path.is_file():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ClipValidationError(f"invalid clip store stats: {stats_path}") from exc
            if stats.get("storage_format") != STORAGE_FORMAT:
                raise ClipValidationError(
                    f"unsupported clip storage format: {stats.get('storage_format')!r}"
                )
            if int(stats.get("count", -1)) != len(self._index):
                raise ClipValidationError("clip store stats count disagrees with index")
            if "compressed_bytes" in stats and int(
                stats["compressed_bytes"]
            ) != int(data_path.stat().st_size):
                raise ClipValidationError("clip store stats byte count disagrees with data.bin")

    def _open_handles(self) -> None:
        if self._mmap is not None:
            return
        self._data_file = (self.root / "data.bin").open("rb")
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_data_file"] = None
        state["_mmap"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_handles()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._open_handles()
        sample_id, start, end = self._index[index]
        payload = self._mmap[start:end]
        if payload[:2] != b"PK":
            raise ClipValidationError(
                f"unsupported clip payload in {self.root}: expected npz-v1"
            )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            record = dict(metadata)
            for key in NPZ_ARRAY_KEYS:
                record[key] = np.array(archive[key], copy=True)
        record["sample_id"] = record.get("sample_id", sample_id)
        return record

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            if self._data_file is not None:
                self._data_file.close()
            self._mmap = None
            self._data_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass



    def clip_spec_table(self):
        """Return sampler metadata from ``index.txt`` without decoding clips."""

        from .clip_batching import ClipSpecTable

        if self._clip_index_fields and all(
            entry is not None for entry in self._clip_index_fields
        ):
            return ClipSpecTable.from_clip_index(
                [entry for entry in self._clip_index_fields if entry is not None]
            )
        # This fallback is only for hand-written/old fixtures without the
        # v1 atom/frame columns. New npz-v1 writers always populate them.
        return ClipSpecTable.from_specs(
            [
                self._record_to_clip_spec(self[index], index)
                for index in range(len(self))
            ]
        )

    @staticmethod
    def _record_to_clip_spec(record: Mapping[str, Any], index: int):
        from .clip_batching import ClipItemSpec

        normalized = validate_clip_record(record)
        time = np.asarray(normalized["time_ps"], dtype=np.float64)
        delta = np.asarray(normalized["delta_time_ps"], dtype=np.float64)
        native = normalized.get("sampled_delta_time_ps")
        if native is None and delta.size:
            native = float(delta[0])
        return ClipItemSpec(
            index=index,
            atoms=int(normalized["x"].shape[1]),
            frames=int(normalized["x"].shape[0]),
            task=normalized["task"],
            time_bucket_id=str(normalized["time_bucket_id"]),
            native_delta_time_ps=native,
            physical_clip_span_ps=float(time[-1] - time[0]),
            sample_id=str(normalized.get("sample_id", index)),
        )

    @staticmethod
    def collate_fn(records: Sequence[Mapping[str, Any]]) -> ClipBatch:
        return collate_clip_records(records)


class StaticClipDataset(torch.utils.data.Dataset):
    """Expose an existing legacy static block store as canonical T=1 clips."""

    def __init__(
        self,
        mmap_dir: str | os.PathLike[str],
        *,
        source: str,
        split: str,
    ) -> None:
        super().__init__()
        from .mmap_dataset import MMAPDataset

        self.root = Path(mmap_dir)
        self.source = str(source)
        self.split = str(split)
        self._legacy = MMAPDataset(str(self.root))

    def __len__(self) -> int:
        return len(self._legacy)

    def clip_spec_table(self):
        """Return static T=1 metadata from the legacy index properties."""

        from .clip_batching import ClipSpecTable

        atom_counts: list[int] = []
        # ANI1x and PCQM4Mv2 write exact atom counts as property zero. PDBBind
        # writes an adjacency budget there, so read its small static store to
        # obtain the actual packed atom count used by T*N.
        if self.source.lower() in {"ani1x", "pcqm4mv2"}:
            for properties in self._legacy._properties:
                try:
                    atom_counts.append(int(properties[0]))
                except (IndexError, TypeError, ValueError):
                    atom_counts.append(-1)
            if any(count < 1 for count in atom_counts):
                atom_counts = []
        if not atom_counts:
            atom_counts = [
                int(len(self._legacy[index]["atype"]))
                for index in range(len(self._legacy))
            ]
        return ClipSpecTable.from_static_index(atom_counts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        legacy_id = self._legacy._indexes[index][0]
        sample_id = f"static_{self.source}_{self.split}_{index:08d}"
        return legacy_static_record_to_clip(
            self._legacy[index],
            sample_id=sample_id,
            source=self.source,
            split=self.split,
            legacy_id=legacy_id,
        )

    @staticmethod
    def collate_fn(records: Sequence[Mapping[str, Any]]) -> ClipBatch:
        return collate_clip_records(records)
