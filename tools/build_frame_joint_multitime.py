"""Materialize Frame Joint v1 multi-time ATLAS train/valid stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from molvid.data.manifest import check_split_overlap
from molvid.data.preprocess import ClipPreprocessConfig, iter_atlas_records
from molvid.data.store import ClipMMapWriter
from molvid.runtime import canonical_hash, sha256_file


def _tag(sample_id: str, delta_ps: int) -> str:
    head, window = str(sample_id).rsplit("_w", 1)
    return f"{head}_dt_{int(delta_ps)}ps_w{window}"


def _systems(manifest: dict[str, Any], view: str) -> list[str]:
    values = sorted({str(item).rsplit("_R", 1)[0] for item in manifest["views"][view]["sample_ids"]})
    if any(not value.startswith("atlas_") for value in values):
        raise ValueError(f"unexpected ATLAS sample id in {view}")
    return [value.removeprefix("atlas_") for value in values]


def _build_raw_store(
    output_root: Path,
    raw_root: Path,
    systems: Iterable[str],
    *,
    split: str,
    source_stride: int,
    delta_ps: int,
) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    config = ClipPreprocessConfig(source_stride=source_stride)
    systems = list(systems)
    with ClipMMapWriter(output_root) as writer:
        for position, system in enumerate(systems, start=1):
            print(f"[{split}/dt_{delta_ps}ps] system {position}/{len(systems)} {system}", flush=True)
            for record in iter_atlas_records(raw_root, system, split=split, config=config):
                sample_id = _tag(str(record["sample_id"]), delta_ps)
                record["sample_id"] = sample_id
                writer.append(record)
                ids.append(sample_id)
    return ids


def _copy_index_segment(
    data_out: Any,
    index_out: Any,
    source_root: Path,
    *,
    delta_ps: int | None,
    offset: int,
) -> tuple[list[str], int]:
    ids: list[str] = []
    with (source_root / "data.bin").open("rb") as source_data:
        shutil.copyfileobj(source_data, data_out, length=8 * 1024 * 1024)
    with (source_root / "index.txt").open(encoding="utf-8") as source_index:
        for line in source_index:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"malformed source index line: {line!r}")
            sample_id = _tag(fields[0], delta_ps) if delta_ps is not None else fields[0]
            start = int(fields[1]) + offset
            end = int(fields[2]) + offset
            rewritten = [sample_id, str(start), str(end), *fields[3:]]
            index_out.write("\t".join(rewritten) + "\n")
            ids.append(sample_id)
    return ids, offset + int((source_root / "data.bin").stat().st_size)


def _concatenate_stores(output_root: Path, segments: list[tuple[Path, int | None]]) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "data.bin"
    index_path = output_root / "index.txt"
    ids: list[str] = []
    offset = 0
    with data_path.open("wb") as data_out, index_path.open("w", encoding="utf-8") as index_out:
        for source_root, delta_ps in segments:
            segment_ids, offset = _copy_index_segment(
                data_out, index_out, source_root, delta_ps=delta_ps, offset=offset
            )
            ids.extend(segment_ids)
    (output_root / "stats.json").write_text(json.dumps({
        "count": len(ids),
        "compressed_bytes": int(data_path.stat().st_size),
        "storage_format": "npz-v1",
    }, indent=2, sort_keys=True) + "\n")
    return ids


def _split_payload(systems: list[str], sample_ids: list[str]) -> dict[str, Any]:
    return {
        "selected_systems": [f"atlas_{item}" for item in systems],
        "selected_system_count": len(systems),
        "sample_ids": list(sample_ids),
        "sample_count": len(sample_ids),
    }


def _store_metadata(root: Path, sample_ids: list[str]) -> dict[str, Any]:
    return {
        "count": len(sample_ids),
        "data_bin_size_bytes": int((root / "data.bin").stat().st_size),
        "index_sha256": sha256_file(root / "index.txt"),
        "stats_sha256": sha256_file(root / "stats.json"),
        "relative_root": str(root.name),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-100-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads((args.source_manifest / "manifest.json").read_text())
    train_systems = _systems(source_manifest, "train192")
    valid_systems = _systems(source_manifest, "valid")
    test_systems = [str(value) for value in source_manifest.get("test_systems", [])]
    output_root = args.output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing multi-time root: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite existing staging root: {staging}")
    staging.mkdir(parents=True)
    try:
        train_200 = staging / "raw_dt_200ps";
        train_400 = staging / "raw_dt_400ps";
        valid_300 = staging / "raw_dt_300ps"
        _build_raw_store(train_200, args.raw_root, train_systems, split="train", source_stride=20, delta_ps=200)
        _build_raw_store(train_400, args.raw_root, train_systems, split="train", source_stride=40, delta_ps=400)
        _build_raw_store(valid_300, args.raw_root, valid_systems, split="valid", source_stride=30, delta_ps=300)

        train_root = output_root / "clip_store" / "train"
        valid_root = output_root / "clip_store" / "valid"
        train_ids = _concatenate_stores(
            train_root,
            [
                (args.source_100_root / "train192", 100),
                (train_200, None),
                (train_400, None),
            ],
        )
        valid_ids = _concatenate_stores(valid_root, [(valid_300, None)])

        source_splits = {
            "train": _split_payload(train_systems, train_ids),
            "valid": _split_payload(valid_systems, valid_ids),
            "test": _split_payload([item.removeprefix("atlas_") for item in test_systems], []),
        }
        check_split_overlap(source_splits)
        manifest = {
            "schema_version": "molvid.frame_joint.multitime.manifest.v1",
            "status": "FROZEN",
            "source_root": str(args.raw_root.resolve()),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(args.source_manifest / "manifest.json"),
            "frames_per_clip": 16,
            "time_bucket_id": "multi",
            "max_tokens": 40000,
            "source_splits": source_splits,
            "test_sampling": {"opened": False},
            "metadata": {
                "dataset": "ATLAS",
                "train_delta_time_ps": [100, 200, 400],
                "valid_delta_time_ps": [300],
                "source_native_delta_time_ps": 10,
                "source_strides": {"dt_100ps": 10, "dt_200ps": 20, "dt_400ps": 40, "dt_300ps": 30},
                "bucket_weights": {"dt_100ps": 1 / 3, "dt_200ps": 1 / 3, "dt_400ps": 1 / 3},
                "trajectory_epoch_clip_budget": 24,
                "test_opened": False,
            },
        }
        manifest["manifest_content_sha256"] = canonical_hash(manifest)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        materialization = {
            "schema_version": "molvid.frame_joint.multitime.materialization.v1",
            "materialized": True,
            "payload_copied": True,
            "test_opened": False,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "splits": {
                "train": _store_metadata(train_root, train_ids),
                "valid": _store_metadata(valid_root, valid_ids),
                "test": {"count": 0, "payload_copied": False},
            },
        }
        materialization["materialization_sha256"] = canonical_hash(materialization)
        (output_root / "materialization.json").write_text(json.dumps(materialization, indent=2, sort_keys=True) + "\n")
        report = {
            "output_root": str(output_root),
            "train_count": len(train_ids),
            "valid_count": len(valid_ids),
            "train_buckets": {str(bucket): sum(f"_dt_{bucket}ps_" in item for item in train_ids) for bucket in (100, 200, 400)},
            "valid_bucket": sum("_dt_300ps_" in item for item in valid_ids),
            "manifest_sha256": sha256_file(output_root / "manifest.json"),
            "materialization_sha256": sha256_file(output_root / "materialization.json"),
        }
        (output_root / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
