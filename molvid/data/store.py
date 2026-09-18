"""Clip storage and legacy static-record read boundary."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from .batch import (
    ClipBatch,
    ClipValidationError,
    SCHEMA_VERSION,
    STATIC_TIME_BUCKET_ID,
    _array,
    _bond_array,
    collate_clip_records,
    validate_clip_record,
)

STORAGE_FORMAT = "npz-v1"
NPZ_ARRAY_KEYS = (
    "x", "bpos", "atype", "btype", "block_id", "component_id",
    "atom_source_index", "edge_mask", "loss_mask", "align_mask",
    "bond_index", "time_ps", "delta_time_ps",
)

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


def static_record_to_clip(
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

        from .sampling import ClipSpecTable

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
        from .sampling import ClipItemSpec

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

        self.root = Path(mmap_dir)
        self.source = str(source)
        self.split = str(split)
        self._legacy = BlockStore(str(self.root))

    def __len__(self) -> int:
        return len(self._legacy)

    def clip_spec_table(self):
        """Return static T=1 metadata from the legacy index properties."""

        from .sampling import ClipSpecTable

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
        return static_record_to_clip(
            self._legacy[index],
            sample_id=sample_id,
            source=self.source,
            split=self.split,
            legacy_id=legacy_id,
        )

    @staticmethod
    def collate_fn(records: Sequence[Mapping[str, Any]]) -> ClipBatch:
        return collate_clip_records(records)
def decompress(compressed_x):
    buf = io.BytesIO(compressed_x)
    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
        serialized_x = f.read().decode()
    x = json.loads(serialized_x)
    return x

class BlockStore(torch.utils.data.Dataset):

    def __init__(self, mmap_dir: str, specify_data: Optional[str] = None, specify_index: Optional[str] = None,
                 approx_length: int = 1, name: Optional[str] = None) -> None:
        super().__init__()

        self._indexes = []
        self._properties = []
        self._origin = []
        self._index_dict = {}
        _index_path = os.path.join(mmap_dir, 'index.txt') if specify_index is None else specify_index
        origin = None
        origin_index = 0
        with open(_index_path, 'r') as f:
            for line in f.readlines():
                messages = line.strip().split('\t')
                _id, start, end = messages[:3]
                if not origin:
                    # _id format: pdb_..._index, for example: '1f95_24042'
                    origin = ''.join(_id.split('_')[:-1])
                cur_origin = ''.join(_id.split('_')[:-1])
                _property = messages[3:]
                self._indexes.append((_id, int(start), int(end)))
                self._properties.append(_property)
                if cur_origin == origin:
                    self._origin.append(origin_index)
                else:
                    origin_index += 1
                    self._origin.append(origin_index)
                    origin = cur_origin
        for idx in range(len(self._indexes)):
            if self._origin[idx] not in self._index_dict:
                self._index_dict[self._origin[idx]] = []
            self._index_dict[self._origin[idx]].append(idx)
        _data_path = os.path.join(mmap_dir, 'data.bin') if specify_data is None else specify_data
        self._data_file = open(_data_path, 'rb')
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.approx_length = approx_length
        self.name = name or ""

    def __del__(self):
        self._mmap.close()
        self._data_file.close()

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        _, start, end = self._indexes[idx]
        data = decompress(self._mmap[start:end])

        if data.get("mask") is None:
            data["mask"] = [1] * len(data["atype"])
        if data.get("edge_mask") is None:
            data["edge_mask"] = [0] * len(data["atype"])

        return data
