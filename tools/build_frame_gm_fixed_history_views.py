"""Build raw-indexed H4 fixed-history views with a 1200 ps future horizon."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from molvid.data.io import read_topology
from molvid.data.preprocess import make_clip_record, topology_metadata
from molvid.data.store import ClipMMapDataset, ClipMMapWriter
from molvid.runtime import canonical_hash, sha256_file


LAGS_PS = (100, 200, 300, 400)
OBSERVED_OFFSETS_PS = (-300, -200, -100, 0)
FUTURE_HORIZON_PS = 1200
TOTAL_SLOTS = 16
NATIVE_DT_PS = 10


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-valid-store", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-windows", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--systems-limit", type=int)
    return parser.parse_args()


def _selected(source: Path, anchors: list[int], limit: int | None) -> tuple[list[str], list[dict[str, Any]]]:
    dataset = ClipMMapDataset(source)
    selected: list[dict[str, Any]] = []
    try:
        parsed: list[tuple[str, str, int, int, str]] = []
        for index, (sample_id, _start, _stop) in enumerate(dataset._index):
            value = str(sample_id)
            if "_dt_300ps_" not in value:
                continue
            match = re.match(r"^(atlas_.+)_R([123])_dt_300ps_w(\d{6})$", value)
            if match is None:
                raise RuntimeError(f"cannot parse validation sample id {value!r}")
            window = int(match.group(3))
            if window in anchors:
                parsed.append((match.group(1), f"R{match.group(2)}", window, index, value))
        systems = sorted({item[0] for item in parsed})
        if limit is not None:
            systems = systems[: int(limit)]
        for system, replica, window, index, sample_id in parsed:
            if system not in systems:
                continue
            record = dataset[index]
            if (
                str(record["system_id"]) != system.removeprefix("atlas_")
                or str(record["replica"]) != replica
                or str(record.get("split")) != "valid"
            ):
                raise RuntimeError(f"source metadata disagrees for {sample_id}")
            selected.append({
                "system": system,
                "system_id": system.removeprefix("atlas_"),
                "replica": replica,
                "anchor_window_300": window,
                "source_index_300": index,
                "source_sample_id_300": sample_id,
            })
    finally:
        dataset.close()
    expected = {
        (system, replica, window)
        for system in systems
        for replica in ("R1", "R2", "R3")
        for window in anchors
    }
    actual = {(item["system"], item["replica"], item["anchor_window_300"]) for item in selected}
    if expected != actual:
        raise RuntimeError(f"fixed-history anchor selection mismatch: {sorted(expected - actual)[:5]}")
    return systems, selected


def _read_needed(xtc: Path, pdb: Path, needed: set[int]) -> tuple[dict[int, np.ndarray], dict[int, float], int]:
    import mdtraj as md

    coordinates: dict[int, np.ndarray] = {}
    times: dict[int, float] = {}
    raw_index = 0
    observed_delta = None
    previous = None
    for chunk in md.iterload(str(xtc), top=str(pdb), chunk=256):
        xyz = np.asarray(chunk.xyz, dtype=np.float32) * 10.0
        time_ps = np.asarray(chunk.time, dtype=np.float64)
        for frame, time_value in zip(xyz, time_ps):
            if previous is not None:
                delta = float(time_value - previous)
                observed_delta = delta if observed_delta is None else observed_delta
                if not np.isclose(delta, observed_delta, rtol=1.0e-5, atol=1.0e-3):
                    raise RuntimeError(f"irregular native clock in {xtc}")
            if raw_index in needed:
                coordinates[raw_index] = frame
                times[raw_index] = float(time_value)
            previous = float(time_value)
            raw_index += 1
    if observed_delta is None or not np.isclose(observed_delta, NATIVE_DT_PS, rtol=1.0e-4, atol=1.0e-3):
        raise RuntimeError(f"native dt mismatch for {xtc}: {observed_delta}")
    missing = sorted(needed - set(coordinates))
    if missing:
        raise RuntimeError(f"{xtc} is missing raw frames {missing[:5]}")
    return coordinates, times, raw_index


def main() -> int:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing existing output {args.output_root}")
    anchors = sorted(set(int(value) for value in args.anchor_windows))
    if not anchors or any(value < 0 for value in anchors):
        raise ValueError("anchor windows must be non-negative")
    systems, selected = _selected(args.source_valid_store, anchors, args.systems_limit)
    store_root = args.output_root / "clip_store" / "valid"
    store_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for system in systems:
        system_id = system.removeprefix("atlas_")
        pdb = args.raw_root / system_id / f"{system_id}.pdb"
        if not pdb.is_file():
            raise FileNotFoundError(pdb)
        metadata = topology_metadata(read_topology(pdb), complex_topology=False)
        topology_sha256 = sha256_file(pdb)
        for replica in ("R1", "R2", "R3"):
            xtc = args.raw_root / system_id / f"{system_id}_prod_{replica}_fit.xtc"
            if not xtc.is_file():
                raise FileNotFoundError(xtc)
            rows = [item for item in selected if item["system"] == system and item["replica"] == replica]
            needed: set[int] = set()
            for item in rows:
                anchor = (int(item["anchor_window_300"]) * 16 + 15) * 30
                item["anchor_raw_frame_index"] = anchor
                needed.update(anchor + offset // NATIVE_DT_PS for offset in OBSERVED_OFFSETS_PS)
                needed.update(anchor + horizon // NATIVE_DT_PS for horizon in range(100, FUTURE_HORIZON_PS + 1, 100))
            coordinates, absolute_times, raw_count = _read_needed(xtc, pdb, needed)
            trajectory_sha256 = sha256_file(xtc)
            source_inventory.append({
                "path": str(xtc.resolve()),
                "size": xtc.stat().st_size,
                "mtime_ns": xtc.stat().st_mtime_ns,
                "sha256": trajectory_sha256,
                "topology_path": str(pdb.resolve()),
                "topology_sha256": topology_sha256,
                "raw_frame_count": raw_count,
            })
            for item in rows:
                anchor = int(item["anchor_raw_frame_index"])
                observed_indices = [anchor + offset // NATIVE_DT_PS for offset in OBSERVED_OFFSETS_PS]
                for lag in LAGS_PS:
                    query_count = FUTURE_HORIZON_PS // lag
                    future_indices = [anchor + step * (lag // NATIVE_DT_PS) for step in range(1, query_count + 1)]
                    actual_indices = observed_indices + future_indices
                    padded_indices: list[int | None] = actual_indices + [None] * (TOTAL_SLOTS - len(actual_indices))
                    last_coordinate = coordinates[actual_indices[-1]]
                    frames = np.stack([
                        coordinates[index] if index is not None else last_coordinate
                        for index in padded_indices
                    ])
                    synthetic_times = np.asarray(
                        [absolute_times[anchor] + offset for offset in OBSERVED_OFFSETS_PS]
                        + [absolute_times[anchor] + step * lag for step in range(1, TOTAL_SLOTS - 4 + 1)],
                        dtype=np.float64,
                    )
                    sample_id = (
                        f"{system}_{replica}_a{int(item['anchor_window_300']):06d}_"
                        f"fixed_history_dt_{lag}ps_H4"
                    )
                    record = make_clip_record(
                        frames,
                        synthetic_times,
                        metadata,
                        sample_id=sample_id,
                        source="atlas",
                        system_id=system_id,
                        replica=replica,
                        split="valid",
                        source_stride=lag // NATIVE_DT_PS,
                        native_dt_ps=float(NATIVE_DT_PS),
                        timestamp_provenance="ATLAS_XTC_time_ps_with_masked_padding_clock",
                    )
                    frame_mask = np.zeros(TOTAL_SLOTS, dtype=np.bool_)
                    frame_mask[: len(actual_indices)] = True
                    valid_relative_times = [*OBSERVED_OFFSETS_PS, *range(lag, FUTURE_HORIZON_PS + 1, lag)]
                    source_absolute_times: list[float | None] = [
                        absolute_times[index] if index is not None else None
                        for index in padded_indices
                    ]
                    coordinate_sha256 = hashlib.sha256(
                        np.ascontiguousarray(frames[: len(actual_indices)]).view(np.uint8)
                    ).hexdigest()
                    view_contract = {
                        "schema": "molvid.frame_gm.fixed_history_record.v1",
                        "sample_id": sample_id,
                        "trajectory_sha256": trajectory_sha256,
                        "topology_sha256": topology_sha256,
                        "source_raw_frame_indices": padded_indices,
                        "source_absolute_time_ps": source_absolute_times,
                        "valid_relative_time_ps": valid_relative_times,
                        "frame_mask": frame_mask.tolist(),
                        "source_coordinate_sha256": coordinate_sha256,
                    }
                    record.update({
                        "frame_mask": frame_mask,
                        "time_bucket_id": f"fixed_history_dt_{lag}ps",
                        "view_kind": "fixed_history_h4_horizon_1200ps",
                        "lag_ps": lag,
                        "history_frames": 4,
                        "query_valid_frames": query_count,
                        "future_horizon_ps": FUTURE_HORIZON_PS,
                        "observed_relative_time_ps": list(OBSERVED_OFFSETS_PS),
                        "query_relative_time_ps": list(range(lag, FUTURE_HORIZON_PS + 1, lag)),
                        "valid_relative_time_ps": valid_relative_times,
                        "slot_relative_time_ps": [
                            *OBSERVED_OFFSETS_PS,
                            *range(lag, (TOTAL_SLOTS - 4 + 1) * lag, lag),
                        ],
                        "anchor_id": f"a{int(item['anchor_window_300']):06d}",
                        "anchor_window_300": int(item["anchor_window_300"]),
                        "anchor_raw_frame_index": anchor,
                        "anchor_time_ps": absolute_times[anchor],
                        "source_raw_frame_indices": padded_indices,
                        "source_absolute_time_ps": source_absolute_times,
                        "valid_source_absolute_time_ps": [absolute_times[index] for index in actual_indices],
                        "source_sample_id_300": item["source_sample_id_300"],
                        "padding_coordinate_policy": "repeat_last_valid_but_frame_mask_false",
                        "padding_time_policy": "arithmetic_continuation_but_frame_mask_false",
                        "raw_trajectory_path": str(xtc.resolve()),
                        "raw_trajectory_sha256": trajectory_sha256,
                        "topology_path": str(pdb.resolve()),
                        "topology_sha256": topology_sha256,
                        "raw_frame_count": raw_count,
                        "source_coordinate_sha256": coordinate_sha256,
                        "view_contract_hash": canonical_hash(view_contract),
                    })
                    records.append(record)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["system_id"]), str(record["replica"]), str(record["anchor_id"]))
        grouped.setdefault(key, []).append(record)
    for key, views in grouped.items():
        if sorted(int(record["lag_ps"]) for record in views) != list(LAGS_PS):
            raise RuntimeError(f"incomplete fixed-history lag group {key}")
        reference = views[0]
        for record in views[1:]:
            if record["atom_identity"] != reference["atom_identity"]:
                raise RuntimeError(f"atom identity changed across fixed-history views {key}")
            if not np.array_equal(record["x"][:4], reference["x"][:4]):
                raise RuntimeError(f"observed coordinates changed across fixed-history views {key}")
    with ClipMMapWriter(store_root) as writer:
        for record in records:
            writer.append(record)
    protocol = {
        "schema": "molvid.frame_gm.fixed_history_views.v1",
        "source_valid_store": str(args.source_valid_store.resolve()),
        "source_valid_index_sha256": sha256_file(args.source_valid_store / "index.txt"),
        "raw_root": str(args.raw_root.resolve()),
        "raw_native_dt_ps": NATIVE_DT_PS,
        "view_kind": "fixed_history_h4_horizon_1200ps",
        "observed_relative_time_ps": list(OBSERVED_OFFSETS_PS),
        "query_lag_ps": list(LAGS_PS),
        "query_valid_frames": {str(lag): FUTURE_HORIZON_PS // lag for lag in LAGS_PS},
        "future_horizon_ps": FUTURE_HORIZON_PS,
        "total_slots": TOTAL_SLOTS,
        "padding_valid": False,
        "systems": systems,
        "replicas": ["R1", "R2", "R3"],
        "anchor_windows_300": anchors,
        "record_count": len(records),
        "source_inventory": source_inventory,
        "selected_records": selected,
        "test_opened": False,
    }
    protocol["contract_hash"] = canonical_hash(protocol)
    manifest = args.output_root / "fixed_history_manifest.json"
    manifest.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_root),
        "records": len(records),
        "systems": len(systems),
        "index_sha256": sha256_file(store_root / "index.txt"),
        "manifest_sha256": sha256_file(manifest),
        "contract_hash": protocol["contract_hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
