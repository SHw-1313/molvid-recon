"""Derive train-only fixed-history views from exact raw-indexed 100 ps clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from molvid.data.batch import canonical_time_fields, validate_clip_record
from molvid.data.store import ClipMMapDataset, ClipMMapWriter
from molvid.runtime import canonical_hash, sha256_file


LAGS_PS = (100, 200, 400)
OBSERVED_RELATIVE_PS = (-300, -200, -100, 0)
FUTURE_HORIZON_PS = 1200
TOTAL_SLOTS = 16
NATIVE_DT_PS = 10
SOURCE_SAMPLE_DT_PS = 100
SOURCE_WINDOW_STRIDE = 16
_SOURCE_ID = re.compile(
    r"^(?P<trajectory>atlas_.+_R[123])_dt_100ps_w(?P<window>[0-9]{6})$"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-store", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-data-hash", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-windows", type=int, nargs="+", default=[0, 20, 40, 61])
    parser.add_argument("--systems-limit", type=int)
    return parser.parse_args()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def derive_fixed_record(
    source: Mapping[str, Any],
    *,
    source_dataset_index: int,
    source_window: int,
    lag_ps: int,
) -> dict[str, Any]:
    """Select exact 100 ps source frames; padded slots remain explicitly invalid."""

    normalized = validate_clip_record(source)
    if normalized.get("split") != "train":
        raise ValueError("fixed-history training views require a train source record")
    if normalized["time_bucket_id"] != "dt_100ps" or normalized["x"].shape[0] != 16:
        raise ValueError("fixed-history training source must be a 16-frame 100 ps clip")
    lag = int(lag_ps)
    if lag not in LAGS_PS:
        raise ValueError(f"unsupported fixed-history train lag {lag}")
    stride = lag // SOURCE_SAMPLE_DT_PS
    source_positions = [0, 1, 2, 3] + list(range(3 + stride, 16, stride))
    expected_query = FUTURE_HORIZON_PS // lag
    if len(source_positions) != 4 + expected_query or source_positions[-1] != 15:
        raise RuntimeError("fixed-history source indexing does not end at 1200 ps")
    padding = TOTAL_SLOTS - len(source_positions)
    padded_positions = source_positions + [source_positions[-1]] * padding
    frame_mask = np.zeros(TOTAL_SLOTS, dtype=np.bool_)
    frame_mask[: len(source_positions)] = True
    slot_time = np.asarray(
        [*OBSERVED_RELATIVE_PS, *range(lag, (TOTAL_SLOTS - 4 + 1) * lag, lag)],
        dtype=np.float32,
    )
    time_ps, delta_time_ps = canonical_time_fields(slot_time)
    source_id = str(normalized["sample_id"])
    match = _SOURCE_ID.match(source_id)
    if match is None or int(match.group("window")) != int(source_window):
        raise ValueError(f"cannot parse source 100 ps sample id {source_id!r}")
    trajectory = match.group("trajectory")
    sample_id = f"{trajectory}_dt_{lag}ps_wf{int(source_window):06d}"
    raw_start = int(source_window) * SOURCE_WINDOW_STRIDE * (SOURCE_SAMPLE_DT_PS // NATIVE_DT_PS)
    raw_indices = [
        raw_start + position * (SOURCE_SAMPLE_DT_PS // NATIVE_DT_PS)
        for position in source_positions
    ]
    padded_raw_indices: list[int | None] = raw_indices + [None] * padding
    record = dict(normalized)
    record.update({
        "sample_id": sample_id,
        "time_bucket_id": f"fixed_history_dt_{lag}ps",
        "time_ps": time_ps,
        "delta_time_ps": delta_time_ps,
        "frame_mask": frame_mask,
        "x": np.asarray(normalized["x"])[padded_positions].copy(),
        "bpos": np.asarray(normalized["bpos"])[padded_positions].copy(),
        "source_stride": lag // NATIVE_DT_PS,
        "sampled_delta_time_ps": float(lag),
        "timestamp_provenance": "exact_subset_of_verified_raw_indexed_100ps_train_clip",
        "view_kind": "fixed_history_h4_horizon_1200ps",
        "lag_ps": lag,
        "history_frames": 4,
        "query_valid_frames": expected_query,
        "future_horizon_ps": FUTURE_HORIZON_PS,
        "observed_relative_time_ps": list(OBSERVED_RELATIVE_PS),
        "query_relative_time_ps": list(range(lag, FUTURE_HORIZON_PS + 1, lag)),
        "valid_relative_time_ps": [
            *OBSERVED_RELATIVE_PS,
            *range(lag, FUTURE_HORIZON_PS + 1, lag),
        ],
        "slot_relative_time_ps": slot_time.tolist(),
        "anchor_id": f"w{int(source_window):06d}:frame3",
        "anchor_source_position": 3,
        "anchor_raw_frame_index": raw_indices[3],
        "source_raw_frame_indices": padded_raw_indices,
        "source_absolute_time_ps": [
            None if index is None else float(index * NATIVE_DT_PS)
            for index in padded_raw_indices
        ],
        "source_dataset_index": int(source_dataset_index),
        "source_sample_id_100ps": source_id,
        "source_window_100ps": int(source_window),
        "source_coordinate_sha256": _array_hash(np.asarray(normalized["x"])),
        "derived_valid_coordinate_sha256": _array_hash(
            np.asarray(normalized["x"])[source_positions]
        ),
        "padding_coordinate_policy": "repeat_last_valid_but_frame_mask_false",
        "padding_time_policy": "arithmetic_continuation_but_frame_mask_false",
    })
    view_contract = {
        "schema": "molvid.frame_gm.fixed_history_train_record.v1",
        "sample_id": sample_id,
        "source_sample_id_100ps": source_id,
        "source_raw_frame_indices": padded_raw_indices,
        "valid_relative_time_ps": record["valid_relative_time_ps"],
        "frame_mask": frame_mask.tolist(),
        "source_coordinate_sha256": record["source_coordinate_sha256"],
        "derived_valid_coordinate_sha256": record["derived_valid_coordinate_sha256"],
    }
    record["view_contract_hash"] = canonical_hash(view_contract)
    return validate_clip_record(record)


def main() -> int:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing existing output {args.output_root}")
    windows = sorted(set(int(value) for value in args.anchor_windows))
    if not windows or any(value < 0 for value in windows):
        raise ValueError("anchor windows must be non-negative")
    source_manifest_sha = sha256_file(args.source_manifest)
    source = ClipMMapDataset(args.source_train_store)
    selected: list[tuple[int, int, str]] = []
    systems: list[str]
    try:
        candidates: list[tuple[int, int, str, str]] = []
        for index, (sample_id, _start, _end) in enumerate(source._index):
            match = _SOURCE_ID.match(str(sample_id))
            if match is None:
                continue
            window = int(match.group("window"))
            if window not in windows:
                continue
            trajectory = match.group("trajectory")
            system = trajectory.rsplit("_R", 1)[0]
            candidates.append((index, window, trajectory, system))
        systems = sorted({item[3] for item in candidates})
        if args.systems_limit is not None:
            systems = systems[: int(args.systems_limit)]
        selected = [
            (index, window, trajectory)
            for index, window, trajectory, system in candidates
            if system in systems
        ]
        expected_trajectories = {
            f"{system}_R{replica}" for system in systems for replica in (1, 2, 3)
        }
        actual_trajectories = {trajectory for _index, _window, trajectory in selected}
        if actual_trajectories != expected_trajectories:
            raise RuntimeError("fixed-history train trajectory grid is incomplete")
        counts = {
            trajectory: sum(item[2] == trajectory for item in selected)
            for trajectory in actual_trajectories
        }
        if set(counts.values()) != {len(windows)}:
            raise RuntimeError("fixed-history train anchor grid is incomplete")
        store_root = args.output_root / "clip_store" / "train"
        records = 0
        observed_hashes: dict[tuple[str, int], str] = {}
        with ClipMMapWriter(store_root) as writer:
            for source_index, window, trajectory in selected:
                source_record = source[source_index]
                views = [
                    derive_fixed_record(
                        source_record,
                        source_dataset_index=source_index,
                        source_window=window,
                        lag_ps=lag,
                    )
                    for lag in LAGS_PS
                ]
                reference = views[0]
                for record in views:
                    key = (trajectory, window)
                    digest = _array_hash(np.asarray(record["x"][:4]))
                    previous = observed_hashes.setdefault(key, digest)
                    if digest != previous or record["atom_identity"] != reference["atom_identity"]:
                        raise RuntimeError("matched train views changed observed coordinates or atom identity")
                    writer.append(record)
                    records += 1
    finally:
        source.close()
    manifest = {
        "schema": "molvid.frame_gm.fixed_history_train_views.v1",
        "split": "train",
        "source_data_hash": str(args.source_data_hash),
        "normalization_source_data_hash": str(args.source_data_hash),
        "source_train_store": str(args.source_train_store.resolve()),
        "source_train_index_sha256": sha256_file(args.source_train_store / "index.txt"),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": source_manifest_sha,
        "derivation": "exact frame subsets of verified 100 ps training clips; no interpolation",
        "raw_native_dt_ps": NATIVE_DT_PS,
        "source_sample_dt_ps": SOURCE_SAMPLE_DT_PS,
        "source_window_stride_sampled_frames": SOURCE_WINDOW_STRIDE,
        "view_kind": "fixed_history_h4_horizon_1200ps",
        "observed_relative_time_ps": list(OBSERVED_RELATIVE_PS),
        "query_lag_ps": list(LAGS_PS),
        "query_valid_frames": {
            str(lag): FUTURE_HORIZON_PS // lag for lag in LAGS_PS
        },
        "future_horizon_ps": FUTURE_HORIZON_PS,
        "total_slots": TOTAL_SLOTS,
        "padding_valid": False,
        "systems": systems,
        "system_count": len(systems),
        "trajectory_count": len({item[2] for item in selected}),
        "anchor_windows_100ps": windows,
        "source_record_count": len(selected),
        "record_count": records,
        "store_index_sha256": sha256_file(store_root / "index.txt"),
        "test_opened": False,
    }
    manifest["contract_hash"] = canonical_hash(manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "fixed_history_train_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_root),
        "records": records,
        "systems": len(systems),
        "trajectories": manifest["trajectory_count"],
        "index_sha256": manifest["store_index_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "contract_hash": manifest["contract_hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
