"""Frozen split manifests and deterministic train/validation dataset plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from torch.utils.data import Subset

from ..runtime import canonical_hash, sha256_file
from .batch import SCHEMA_VERSION
from .preprocess import ClipPreprocessConfig
from .sampling import TaskAwareClipBatchSampler
from .store import ClipMMapDataset, STORAGE_FORMAT, source_inventory_hash


_SAMPLE_ID = re.compile(r"^(?P<system>.+)_(?P<replica>R[0-9]+)_w(?P<window>[0-9]+)$")


@dataclass(frozen=True)
class DatasetSplits:
    manifest_root: Path
    manifest: Mapping[str, Any]
    materialization: Mapping[str, Any]
    train: ClipMMapDataset
    valid: ClipMMapDataset
    data_hash: str
    train_index_hash: str
    valid_index_hash: str

    def close(self) -> None:
        self.train.close()
        self.valid.close()


@dataclass(frozen=True)
class ValidationPlan:
    subset: Subset
    batches: tuple[tuple[int, ...], ...]
    selected_dataset_indices: tuple[int, ...]
    selected_sample_ids: tuple[str, ...]
    systems: tuple[str, ...]
    windows: tuple[int, ...]
    schedule_hash: str


def check_split_overlap(source_splits: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject both within-split duplicates and cross-split identity overlap."""

    for field in ("selected_systems", "sample_ids"):
        seen: set[str] = set()
        for split in ("train", "valid", "test"):
            values = tuple(str(value) for value in source_splits[split][field])
            if len(values) != len(set(values)):
                raise ValueError(f"{split} contains duplicate {field}")
            overlap = seen.intersection(values)
            if overlap:
                raise ValueError(f"{field} overlap across splits: {sorted(overlap)[:3]}")
            seen.update(values)


def build_manifest(
    source_splits: Mapping[str, Mapping[str, Any]],
    *,
    source_root: str | Path,
    frames_per_clip: int,
    time_bucket_id: str,
    max_tokens: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze explicitly selected systems and sample IDs without reassigning splits."""

    check_split_overlap(source_splits)
    if int(frames_per_clip) < 1 or int(max_tokens) < 1:
        raise ValueError("frames_per_clip and max_tokens must be positive")
    frozen_splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        item = source_splits[split]
        systems = list(str(value) for value in item["selected_systems"])
        sample_ids = list(str(value) for value in item["sample_ids"])
        frozen_splits[split] = {
            "selected_systems": systems,
            "selected_system_count": len(systems),
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
        }
    manifest = {
        "schema_version": "molvid.clip.manifest.v1",
        "status": "FROZEN",
        "source_root": str(Path(source_root).resolve()),
        "frames_per_clip": int(frames_per_clip),
        "time_bucket_id": str(time_bucket_id),
        "max_tokens": int(max_tokens),
        "source_splits": frozen_splits,
        "test_sampling": {"opened": False},
        "metadata": dict(metadata or {}),
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    return manifest


def load_datasets(manifest_root: str | Path) -> DatasetSplits:
    """Open only train and validation payloads from a verified frozen manifest."""

    root = Path(manifest_root).resolve()
    if any(part.lower() == "test" for part in root.parts):
        raise RuntimeError("manifest root must not be a test split")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    materialization = json.loads((root / "materialization.json").read_text(encoding="utf-8"))
    manifest_content = dict(manifest)
    recorded_hash = manifest_content.pop("manifest_content_sha256", None)
    if recorded_hash != canonical_hash(manifest_content):
        raise RuntimeError("frozen manifest content hash is invalid")
    materialization_content = dict(materialization)
    materialization_hash = materialization_content.pop("materialization_sha256", None)
    if materialization_hash != canonical_hash(materialization_content):
        raise RuntimeError("materialization content hash is invalid")
    if manifest.get("status") != "FROZEN" or materialization.get("materialized") is not True:
        raise RuntimeError("manifest and clip materialization must be frozen and complete")
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("test split must remain sealed")
    splits = manifest["source_splits"]
    check_split_overlap(splits)
    for split in ("train", "valid", "test"):
        expected = tuple(str(value) for value in splits[split]["sample_ids"])
        if int(materialization["splits"][split].get("count", -1)) != len(expected):
            raise RuntimeError(f"{split} materialization count differs from manifest")
    stores: dict[str, ClipMMapDataset] = {}
    try:
        for split in ("train", "valid"):
            store_root = root / "clip_store" / split
            store = ClipMMapDataset(store_root)
            stores[split] = store
            expected = tuple(str(value) for value in splits[split]["sample_ids"])
            actual = tuple(str(row[0]) for row in store._index)
            if actual != expected:
                raise RuntimeError(f"{split} index differs from frozen manifest")
            index_hash = sha256_file(store_root / "index.txt")
            recorded_index_hash = materialization["splits"][split].get("index_sha256")
            if recorded_index_hash not in (None, index_hash):
                raise RuntimeError(f"{split} index SHA256 differs from materialization")
        train_index_hash = sha256_file(root / "clip_store/train/index.txt")
        valid_index_hash = sha256_file(root / "clip_store/valid/index.txt")
        data_contract = {
            "schema": (
                "pvb.dit.state_detail.t1_data_contract.v1"
                if manifest.get("schema_version") == "pvb.codec.state_detail.t1_manifest.v1"
                else "molvid.data.contract.v1"
            ),
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "materialization_sha256": materialization["materialization_sha256"],
            "train": materialization["splits"]["train"],
            "valid": materialization["splits"]["valid"],
            "train_index_sha256": train_index_hash,
            "valid_index_sha256": valid_index_hash,
            "test_opened": False,
        }
        return DatasetSplits(
            manifest_root=root,
            manifest=manifest,
            materialization=materialization,
            train=stores["train"],
            valid=stores["valid"],
            data_hash=canonical_hash(data_contract),
            train_index_hash=train_index_hash,
            valid_index_hash=valid_index_hash,
        )
    except Exception:
        for store in stores.values():
            store.close()
        raise


def make_validation_plan(
    dataset: ClipMMapDataset,
    *,
    systems: Sequence[str],
    windows: Sequence[int],
    replicas: Sequence[str],
    max_tokens: int,
    seed: int,
) -> ValidationPlan:
    """Select the explicit system/replica/window grid without opening test data."""

    expected_systems = tuple(sorted(str(value) for value in systems))
    expected_windows = tuple(int(value) for value in windows)
    expected_replicas = tuple(str(value) for value in replicas)
    selected: list[tuple[str, str, int, int]] = []
    for index, row in enumerate(dataset._index):
        sample_id = str(row[0])
        match = _SAMPLE_ID.match(sample_id)
        if match is None:
            raise RuntimeError(f"validation sample id has an unsupported form: {sample_id}")
        window = int(match.group("window"))
        if window in expected_windows:
            selected.append((match.group("system"), match.group("replica"), window, index))
    selected.sort()
    expected = {
        (system, replica, window)
        for system in expected_systems
        for replica in expected_replicas
        for window in expected_windows
    }
    actual = {(system, replica, window) for system, replica, window, _ in selected}
    if actual != expected or len(selected) != len(expected):
        raise RuntimeError("validation plan does not contain the requested grid")
    selected_indices = tuple(item[3] for item in selected)
    selected_ids = tuple(str(dataset._index[index][0]) for index in selected_indices)
    subset = Subset(dataset, list(selected_indices))
    sampler = TaskAwareClipBatchSampler(
        subset,
        max_tokens=max_tokens,
        seed=seed,
        shuffle=False,
        replacement=False,
        oversize_policy="error",
    )
    batches = tuple(tuple(int(index) for index in batch) for batch in sampler.global_batches)
    schedule_hash = canonical_hash(
        {
            "windows": list(expected_windows),
            "sample_ids": list(selected_ids),
            "batches": [list(batch) for batch in batches],
        }
    )
    return ValidationPlan(
        subset=subset,
        batches=batches,
        selected_dataset_indices=selected_indices,
        selected_sample_ids=selected_ids,
        systems=expected_systems,
        windows=expected_windows,
        schedule_hash=schedule_hash,
    )


def source_inventory(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    """Capture source path, size and modification time for preprocess evidence."""

    entries = []
    for path in sorted(set(paths)):
        stat = path.stat()
        entries.append({
            "path": str(path.relative_to(root)),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return entries


def write_split_manifest(
    output_path: str | Path,
    *,
    source: str,
    raw_root: str | Path,
    config: ClipPreprocessConfig,
    source_version: str,
    timestamp_provenance: str,
    topology_policy: str,
    split_policy: str,
    systems: Sequence[str],
    inventory: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    """Write the same physical-time and source-inventory contract as preprocessing."""

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "storage_format": STORAGE_FORMAT,
        "source": source,
        "raw_root": str(Path(raw_root).resolve()),
        "source_version": source_version,
        "source_inventory_sha256": source_inventory_hash(inventory),
        "coordinate_unit": "angstrom",
        "timestamp_provenance": timestamp_provenance,
        "native_delta_time_ps": (
            config.atlas_native_dt_ps if source == "atlas" else config.misato_native_dt_ps
        ),
        "clip_len": config.clip_len,
        "window_stride": config.window_stride,
        "source_stride": config.source_stride,
        "topology_policy": topology_policy,
        "split_policy": split_policy,
        "systems": list(systems),
        "inventory": list(inventory),
        "counts": dict(counts),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
