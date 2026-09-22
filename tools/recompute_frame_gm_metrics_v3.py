"""Recompute versioned motion metrics from saved coordinates without generation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.temporal import (
    TEMPORAL_METRIC_SCHEMA,
    system_mean_rmsf_summary,
    temporal_metrics_v3,
)
from molvid.runtime import atomic_write_json, configure_device, sha256_file


ROW_SCHEMA = "molvid.frame_gm.metric_row.v3"
SUMMARY_SCHEMA = "molvid.frame_gm.metric_summary.v3"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--paired-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, TensorLike):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


TensorLike = torch.Tensor


def _value(record: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = record
    for name in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


SCALARS: dict[str, tuple[str, ...]] = {
    "displacement_increment_rmse_A": ("increments", "displacement_increment_rmse_A"),
    "finite_difference_velocity_rmse_A_per_ps": ("increments", "finite_difference_velocity_rmse_A_per_ps"),
    "velocity_correlation": ("increments", "velocity_correlation", "value"),
    "atom_rmsf_mae_A": ("rmsf", "atom_profile", "mae_A"),
    "atom_rmsf_pearson": ("rmsf", "atom_profile", "pearson", "value"),
    "atom_rmsf_spearman": ("rmsf", "atom_profile", "spearman", "value"),
    "residue_rmsf_mae_A": ("rmsf", "residue_profile", "mae_A"),
    "residue_rmsf_pearson": ("rmsf", "residue_profile", "pearson", "value"),
    "residue_rmsf_spearman": ("rmsf", "residue_profile", "spearman", "value"),
    "system_mean_rmsf_prediction_A": ("rmsf", "system_mean_prediction_A"),
    "system_mean_rmsf_target_A": ("rmsf", "system_mean_target_A"),
    "rmsf_mean_amplitude_ratio": ("rmsf", "mean_amplitude_ratio"),
    "msd_curve_mae_A2": ("msd", "curve_mae_A2"),
    "internal_distance_increment_mae_A": ("internal", "distance_increment", "mae_A"),
    "internal_distance_increment_pearson": ("internal", "distance_increment", "pearson", "value"),
    "torsion_increment_mae_radian": ("internal", "torsion_increment_radian", "mae_radian"),
    "torsion_increment_pearson": ("internal", "torsion_increment_radian", "pearson", "value"),
}


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _flat_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = record["metrics"]
    return {name: _value(metrics, path) for name, path in SCALARS.items()}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_anchor: dict[tuple[Any, ...], list[dict[str, float | None]]] = defaultdict(list)
    for row in rows:
        meta = row["meta"]
        key = (
            row["path"], row["clock"], meta["lag_ps"], meta["history_frames"],
            meta["system"], meta["replica"], meta["anchor_id"],
        )
        by_anchor[key].append(_flat_metrics(row))
    anchor_rows: list[dict[str, Any]] = []
    for key, values in by_anchor.items():
        row = {
            "path": key[0], "clock": key[1], "lag_ps": key[2], "history_frames": key[3],
            "system": key[4], "replica": key[5], "anchor_id": key[6], "draw_count": len(values),
        }
        row.update({name: _mean(value[name] for value in values) for name in SCALARS})
        anchor_rows.append(row)
    by_replica: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        key = (row["path"], row["clock"], row["lag_ps"], row["history_frames"], row["system"], row["replica"])
        by_replica[key].append(row)
    replica_rows: list[dict[str, Any]] = []
    for key, values in by_replica.items():
        row = {
            "path": key[0], "clock": key[1], "lag_ps": key[2], "history_frames": key[3],
            "system": key[4], "replica": key[5], "anchor_count": len(values),
        }
        row.update({name: _mean(value[name] for value in values) for name in SCALARS})
        replica_rows.append(row)
    by_system: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in replica_rows:
        key = (row["path"], row["clock"], row["lag_ps"], row["history_frames"], row["system"])
        by_system[key].append(row)
    system_rows: list[dict[str, Any]] = []
    for key, values in by_system.items():
        row = {
            "path": key[0], "clock": key[1], "lag_ps": key[2], "history_frames": key[3],
            "system": key[4], "replica_count": len(values),
        }
        row.update({name: _mean(value[name] for value in values) for name in SCALARS})
        system_rows.append(row)
    by_bucket: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in system_rows:
        by_bucket[(row["path"], row["clock"], row["lag_ps"], row["history_frames"])].append(row)
    buckets: dict[str, Any] = {}
    for key, values in sorted(by_bucket.items()):
        name = f"{key[0]}|{key[1]}|{key[2]}ps|H{key[3]}"
        summary = {metric: _mean(row[metric] for row in values) for metric in SCALARS}
        summary["availability"] = {
            metric: sum(row[metric] is not None for row in values) for metric in SCALARS
        }
        summary["system_count"] = len(values)
        summary["systems"] = sorted(str(row["system"]) for row in values)
        summary["system_mean_rmsf_distribution"] = system_mean_rmsf_summary([
            {
                "system_mean_prediction_A": row["system_mean_rmsf_prediction_A"],
                "system_mean_target_A": row["system_mean_rmsf_target_A"],
            }
            for row in values
        ])
        buckets[name] = summary
    return sorted(system_rows, key=lambda row: (
        str(row["path"]), str(row["clock"]), int(row["lag_ps"]), int(row["history_frames"]), str(row["system"])
    )), buckets


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["path", "clock", "lag_ps", "history_frames", "system", "replica_count", *SCALARS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fields} for row in rows)


def main() -> int:
    args = _args()
    device = configure_device(args.device, deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("metrics v3 tensor computations require CUDA")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing existing output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    profiles_root = args.output / "profiles"
    profiles_root.mkdir(exist_ok=True)
    source_rows_path = args.source_run / "rows.jsonl"
    source_rows = _read_jsonl(source_rows_path)
    if args.limit is not None:
        source_rows = source_rows[: int(args.limit)]
    output_rows_path = args.output / "rows.jsonl"
    completed: set[str] = set()
    if args.resume and output_rows_path.is_file():
        completed = {str(row["row_key"]) for row in _read_jsonl(output_rows_path)}
    dataset = ClipMMapDataset(args.paired_store)
    by_id = {str(row[0]): index for index, row in enumerate(dataset._index)}
    written = 0
    try:
        for position, source_row in enumerate(source_rows):
            row_key = str(source_row["row_key"])
            if row_key in completed:
                continue
            meta = dict(source_row["meta"])
            sample_id = str(meta["sample_id"])
            if sample_id not in by_id:
                raise KeyError(f"paired store is missing {sample_id!r}")
            record = dataset[by_id[sample_id]]
            cpu_batch = collate_clip_records([record])
            batch = cpu_batch.to(device)
            coordinate_path = args.source_run / "coordinates" / str(source_row["coordinate_file"])
            with np.load(coordinate_path) as archive:
                prediction = torch.as_tensor(np.array(archive["prediction"], copy=True), device=device, dtype=torch.float32)
            target = batch.x.float()
            result = temporal_metrics_v3(
                prediction,
                target,
                batch,
                history_frames=int(meta["history_frames"]),
            )
            profile_name = hashlib.sha1(row_key.encode()).hexdigest() + ".npz"
            np.savez_compressed(
                profiles_root / profile_name,
                **{name: value.detach().cpu().numpy() for name, value in result.profiles.items()},
            )
            output_row = {
                "schema": ROW_SCHEMA,
                "metric_schema": TEMPORAL_METRIC_SCHEMA,
                "row_key": row_key,
                "path": source_row["path"],
                "clock": source_row["clock"],
                "seed": source_row.get("seed"),
                "meta": meta,
                "metrics": _jsonable(result.metrics),
                "profile_file": profile_name,
                "coordinate_source": {
                    "relative_path": str(Path("coordinates") / str(source_row["coordinate_file"])),
                    "sha256": sha256_file(coordinate_path),
                    "source_run": str(args.source_run.resolve()),
                },
                "legacy_summary_preserved_for_provenance_only": source_row.get("summary"),
            }
            with output_rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(output_row, sort_keys=True) + "\n")
            written += 1
            if (position + 1) % 50 == 0:
                print(json.dumps({"processed": position + 1, "total": len(source_rows), "written": written}), flush=True)
    finally:
        dataset.close()
    all_rows = _read_jsonl(output_rows_path)
    expected_keys = {str(row["row_key"]) for row in source_rows}
    selected_rows = [row for row in all_rows if str(row["row_key"]) in expected_keys]
    if len(selected_rows) != len(expected_keys):
        raise RuntimeError("metrics v3 rows are incomplete; resume before aggregating")
    per_system, buckets = _aggregate(selected_rows)
    _write_csv(args.output / "per_system.csv", per_system)
    metrics = {
        "schema": SUMMARY_SCHEMA,
        "metric_schema": TEMPORAL_METRIC_SCHEMA,
        "source_rows": len(source_rows),
        "source_rows_sha256": sha256_file(source_rows_path),
        "source_protocol_sha256": sha256_file(args.source_run / "protocol.json"),
        "paired_store_index_sha256": sha256_file(args.paired_store / "index.txt"),
        "aggregation": "draw/seed mean -> anchor mean -> replica mean -> system equal",
        "bucket_summary": buckets,
        "historical_rows_mutated": False,
        "test_opened": False,
    }
    atomic_write_json(args.output / "metrics.json", metrics)
    atomic_write_json(args.output / "run_manifest.json", {
        "schema": "molvid.frame_gm.metric_recompute_run.v1",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "source_run": str(args.source_run.resolve()),
        "paired_store": str(args.paired_store.resolve()),
        "output_rows_sha256": sha256_file(output_rows_path),
        "metrics_sha256": sha256_file(args.output / "metrics.json"),
        "per_system_sha256": sha256_file(args.output / "per_system.csv"),
        "row_count": len(selected_rows),
    })
    print(json.dumps({"complete": True, "rows": len(selected_rows), "written": written, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
