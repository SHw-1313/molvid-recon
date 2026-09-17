#!/usr/bin/env python3
"""Freeze and optionally materialize the state/detail codec v2 T1 split.

The source archive is already split at the system level.  This script never
reassigns a system across source splits: it validates the source inventory,
selects a deterministic disjoint subset from each source split, and writes all
62 native T=16 windows for the three replicas.  Training resampling is a
separate sampler concern; the manifest records its fixed 24-window-per-
trajectory cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data.clip_dataset import ClipMMapDataset, ClipMMapWriter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/manifest_20260904_token80000"
MAX_TOKENS = 80000
SOURCE_SPLITS = ("train", "valid", "test")
REPLICAS = ("R1", "R2", "R3")
WINDOWS = tuple(range(62))
SELECTION_SEED = 20260904
SELECTION_NAMESPACE = "pvb-state-detail-codec-v2-t1-system-selection-v1"
EXCLUDED_HISTORICAL_SYSTEMS = frozenset(
    {"atlas_5e3e_A", "atlas_1v7r_A", "atlas_2wlt_A"}
)
TARGET_COUNTS = {"train": 48, "valid": 8, "test": 8}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"invalid index row {path}:{line_number}: {line!r}")
        sample_id, start, end, atoms, frames, bucket = fields
        rows.append(
            {
                "sample_id": sample_id,
                "start": int(start),
                "end": int(end),
                "atoms": int(atoms),
                "frames": int(frames),
                "time_bucket_id": bucket,
            }
        )
    return rows


def split_sample_id(sample_id: str) -> tuple[str, str, int]:
    prefix, window_text = str(sample_id).rsplit("_w", 1)
    system, replica = prefix.rsplit("_", 1)
    if not system.startswith("atlas_"):
        raise ValueError(f"unexpected ATLAS sample id: {sample_id!r}")
    if replica not in REPLICAS or not window_text.isdigit():
        raise ValueError(f"unexpected ATLAS sample id: {sample_id!r}")
    return system, replica, int(window_text)


def inventory_by_system(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, dict[str, dict[int, Mapping[str, Any]]]]:
    inventory: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for row in rows:
        system, replica, window = split_sample_id(str(row["sample_id"]))
        if int(row["frames"]) != 16 or str(row["time_bucket_id"]) != "dt_100ps":
            raise ValueError(f"{split} has non-T=16/dt_100ps row: {row}")
        if int(row["atoms"]) < 1 or int(row["end"]) <= int(row["start"]):
            raise ValueError(f"{split} has invalid storage range: {row}")
        system_rows = inventory.setdefault(system, {}).setdefault(replica, {})
        if window in system_rows:
            raise ValueError(f"duplicate sample id in {split}: {row['sample_id']}")
        system_rows[window] = row
    for system, replicas in inventory.items():
        if set(replicas) != set(REPLICAS):
            raise ValueError(f"{split} system {system} does not have exactly R1/R2/R3")
        for replica, windows in replicas.items():
            if set(windows) != set(WINDOWS):
                missing = sorted(set(WINDOWS).difference(windows))
                extra = sorted(set(windows).difference(WINDOWS))
                raise ValueError(
                    f"{split} {system}_{replica} does not have windows 0..61; "
                    f"missing={missing[:4]} extra={extra[:4]}"
                )
    return inventory


def selection_key(split: str, system: str) -> str:
    value = f"{SELECTION_NAMESPACE}|seed={SELECTION_SEED}|split={split}|system={system}"
    return hashlib.sha256(value.encode()).hexdigest()


def select_systems(inventories: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for split in SOURCE_SPLITS:
        candidates = sorted(
            system
            for system in inventories[split]
            if system not in EXCLUDED_HISTORICAL_SYSTEMS
            and max(
                int(row["atoms"]) * int(row["frames"])
                for replicas in inventories[split][system].values()
                for row in replicas.values()
            ) <= MAX_TOKENS
        )
        ranked = sorted(candidates, key=lambda system: (selection_key(split, system), system))
        selected[split] = ranked[: TARGET_COUNTS[split]]
        if len(selected[split]) != TARGET_COUNTS[split]:
            raise RuntimeError(f"{split} has only {len(selected[split])} eligible systems")
    all_selected = [system for systems in selected.values() for system in systems]
    if len(set(all_selected)) != 64:
        raise RuntimeError("selected systems are not disjoint")
    return selected


def ordered_ids(systems: Iterable[str]) -> list[str]:
    return [
        f"{system}_{replica}_w{window:06d}"
        for system in sorted(systems)
        for replica in REPLICAS
        for window in WINDOWS
    ]


def build_manifest(source_root: Path, inventories: Mapping[str, Mapping[str, Any]], selected: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    source_manifest = source_root / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    split_ids = {split: ordered_ids(selected[split]) for split in SOURCE_SPLITS}
    all_ids = [sample_id for ids in split_ids.values() for sample_id in ids]
    if len(all_ids) != 64 * len(REPLICAS) * len(WINDOWS):
        raise RuntimeError("T1 selected sample count is not 64*3*62")
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("T1 selected sample IDs overlap")
    manifest = {
        "schema_version": "pvb.codec.state_detail.t1_manifest.v1",
        "status": "FROZEN",
        "selection_seed": SELECTION_SEED,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_algorithm": "rank sha256(namespace|seed|source_split|system), then system name",
        "excluded_historical_t0_systems": sorted(EXCLUDED_HISTORICAL_SYSTEMS),
        "source_root": str(source_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_manifest_split_policy": source_payload.get("split_policy"),
        "source_manifest_system_count": len(source_payload.get("systems", [])),
        "max_tokens": MAX_TOKENS,
        "eligibility_filter": "all 186 native T=16 clips for a system satisfy frames*atoms <= max_tokens",
        "eligible_system_counts": {
            split: sum(
                1
                for system, replicas in inventories[split].items()
                if system not in EXCLUDED_HISTORICAL_SYSTEMS
                and max(
                    int(row["atoms"]) * int(row["frames"])
                    for replica_rows in replicas.values()
                    for row in replica_rows.values()
                ) <= MAX_TOKENS
            )
            for split in SOURCE_SPLITS
        },
        "source_splits": {
            split: {
                "source_index": str(source_root / split / "index.txt"),
                "source_index_sha256": sha256_file(source_root / split / "index.txt"),
                "source_system_count": len(inventories[split]),
                "selected_systems": list(selected[split]),
                "selected_system_count": len(selected[split]),
                "selected_system_max_atoms": {
                    system: max(
                        int(row["atoms"])
                        for replica_rows in inventories[split][system].values()
                        for row in replica_rows.values()
                    )
                    for system in selected[split]
                },
                "sample_ids": split_ids[split],
                "sample_count": len(split_ids[split]),
            }
            for split in SOURCE_SPLITS
        },
        "systems": sorted(set(all_ids[i].rsplit("_R", 1)[0] for i in range(len(all_ids)))),
        "replicas": list(REPLICAS),
        "windows": [0, 61],
        "window_count_per_trajectory": len(WINDOWS),
        "frames_per_clip": 16,
        "time_bucket_id": "dt_100ps",
        "native_delta_time_ps": 100.0,
        "precision": "fp32",
        "lazy_loading": True,
        "train_sampling": {
            "clips_per_trajectory_per_epoch": 24,
            "sampling": "without replacement within epoch, deterministic resampling across epochs",
            "replacement": False,
            "epoch_seed": "seed + epoch + trajectory hash",
        },
        "validation_sampling": {
            "coverage": "all 62 windows for all three replicas of eight systems",
            "replacement": False,
        },
        "test_sampling": {
            "coverage": "all 62 windows for all three replicas of eight systems",
            "replacement": False,
            "opened": False,
        },
        "counts": {
            "systems_total": 64,
            "train_systems": TARGET_COUNTS["train"],
            "valid_systems": TARGET_COUNTS["valid"],
            "test_systems": TARGET_COUNTS["test"],
            "train_trajectories": TARGET_COUNTS["train"] * len(REPLICAS),
            "valid_trajectories": TARGET_COUNTS["valid"] * len(REPLICAS),
            "test_trajectories": TARGET_COUNTS["test"] * len(REPLICAS),
            "train_clips_materialized": len(split_ids["train"]),
            "valid_clips_materialized": len(split_ids["valid"]),
            "test_clips_materialized": len(split_ids["test"]),
            "train_clips_per_sampled_epoch": TARGET_COUNTS["train"] * len(REPLICAS) * 24,
            "valid_clips": len(split_ids["valid"]),
            "test_clips": len(split_ids["test"]),
        },
        "system_disjointness": {
            "train_valid": True,
            "train_test": True,
            "valid_test": True,
            "historical_t0_excluded": True,
        },
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    return manifest


def materialize_split(
    source_root: Path,
    output_root: Path,
    split: str,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    source = ClipMMapDataset(source_root / split)
    try:
        by_id = {str(row[0]): index for index, row in enumerate(source._index)}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise RuntimeError(f"{split} selected IDs missing from source: {missing[:8]}")
        destination = output_root / split
        with ClipMMapWriter(destination, overwrite=False) as writer:
            for count, sample_id in enumerate(sample_ids, 1):
                record = source[by_id[sample_id]]
                if str(record["sample_id"]) != sample_id:
                    raise RuntimeError(f"source record ID mismatch for {sample_id}")
                writer.append(record)
                if count % 256 == 0:
                    print(f"{split}: materialized {count}/{len(sample_ids)}", flush=True)
        return {
            "split": split,
            "root": str(destination),
            "count": len(sample_ids),
            "index_sha256": sha256_file(destination / "index.txt"),
            "data_sha256": sha256_file(destination / "data.bin"),
            "stats_sha256": sha256_file(destination / "stats.json"),
        }
    finally:
        source.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root if args.source_root.is_absolute() else ROOT / args.source_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        if not args.materialize:
            raise FileExistsError(f"refusing to overwrite frozen T1 manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = dict(manifest)
        recorded_hash = content.pop("manifest_content_sha256", None)
        if recorded_hash != canonical_hash(content):
            raise RuntimeError("existing frozen manifest content hash is invalid")
        if Path(manifest["source_root"]) != source_root:
            raise RuntimeError("existing manifest source root disagrees with requested source root")
    else:
        inventories = {
            split: inventory_by_system(read_index(source_root / split / "index.txt"), split)
            for split in SOURCE_SPLITS
        }
        selected = select_systems(inventories)
        manifest = build_manifest(source_root, inventories, selected)
        output_root.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    materialization: dict[str, Any] = {"materialized": False, "splits": {}}
    if args.materialize:
        for split in SOURCE_SPLITS:
            materialization["splits"][split] = materialize_split(
                source_root, output_root / "clip_store", split,
                manifest["source_splits"][split]["sample_ids"],
            )
        materialization["materialized"] = True
        materialization["materialization_sha256"] = canonical_hash(materialization)
        (output_root / "materialization.json").write_text(
            json.dumps(materialization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"manifest": str(manifest_path), "materialization": materialization}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
