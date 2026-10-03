"""Build persistent MDTraj XTC frame-offset sidecars for raw ATLAS I/O."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import struct
import time
from typing import Any

import numpy as np

from molvid.data.learned_memory_formal import RawMemoryDataset
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _build_one(task: tuple[str, str, int]) -> dict[str, Any]:
    xtc_value, sidecar_value, minimum_frames = task
    xtc = Path(xtc_value)
    sidecar = Path(sidecar_value)
    source = xtc.stat()
    if sidecar.is_file():
        with np.load(sidecar, allow_pickle=False) as cache:
            offsets = np.asarray(cache["offsets"], dtype=np.int64)
            valid = (
                int(cache["xtc_size"]) == int(source.st_size)
                and int(cache["xtc_mtime_ns"]) == int(source.st_mtime_ns)
                and offsets.size >= minimum_frames
            )
        if valid:
            return {
                "xtc": str(xtc), "xtc_size": int(source.st_size),
                "xtc_mtime_ns": int(source.st_mtime_ns), "frames": int(offsets.size),
                "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar),
                "reused": True,
            }
    started = time.perf_counter()
    # MDTraj's offset scanner performs one small seek/read per frame.  On the
    # cold ATLAS filesystem that turns a 100 ns trajectory into minutes of
    # random I/O.  Reading the XTC bytes sequentially and applying MDTraj's
    # published frame-size recurrence yields the identical byte offsets.
    payload = xtc.read_bytes()
    if len(payload) < 8:
        raise ValueError(f"truncated XTC header: {xtc}")
    magic, atoms = struct.unpack_from(">ii", payload, 0)
    if magic != 1995 or atoms < 1:
        raise ValueError(f"invalid XTC magic/atom count: {xtc}")
    if atoms <= 9:
        frame_bytes = 56 + 12 * atoms
        if len(payload) % frame_bytes:
            raise ValueError(f"short-atom XTC size is not frame aligned: {xtc}")
        offsets = np.arange(0, len(payload), frame_bytes, dtype=np.int64)
    else:
        values: list[int] = []
        position = 0
        while position + 92 <= len(payload):
            frame_magic, frame_atoms = struct.unpack_from(">ii", payload, position)
            if frame_magic != 1995 or frame_atoms != atoms:
                raise ValueError(f"XTC frame header differs at byte {position}: {xtc}")
            compressed_bytes = struct.unpack_from(">i", payload, position + 88)[0]
            if compressed_bytes < 0:
                raise ValueError(f"negative XTC compressed block size: {xtc}")
            padded_bytes = compressed_bytes + 3 - ((compressed_bytes + 3) % 4)
            values.append(position)
            position += 92 + padded_bytes
        if position != len(payload):
            raise ValueError(f"XTC frames do not end at file size: {xtc}")
        offsets = np.asarray(values, dtype=np.int64)
    del payload
    if offsets.size < minimum_frames or offsets.ndim != 1:
        raise ValueError(f"XTC offset scan is shorter than frozen evidence: {xtc}")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            offsets=offsets,
            xtc_size=np.asarray(source.st_size, dtype=np.int64),
            xtc_mtime_ns=np.asarray(source.st_mtime_ns, dtype=np.int64),
        )
    temporary.replace(sidecar)
    return {
        "xtc": str(xtc), "xtc_size": int(source.st_size),
        "xtc_mtime_ns": int(source.st_mtime_ns), "frames": int(offsets.size),
        "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar),
        "scan_seconds": time.perf_counter() - started, "reused": False,
    }


def main() -> int:
    args = _args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, Any] = {}
    for split in ("train", "valid"):
        dataset = RawMemoryDataset(
            manifest_root=args.manifest_root,
            raw_root=args.raw_root,
            split=split,
        )
        split_identities[split] = dataset.identity
        for item in dataset.items:
            key = str(item.xtc)
            task = tasks.setdefault(key, {
                "xtc": key,
                "sidecar": str(
                    args.output / item.system / f"{item.xtc.name}.offsets.npz"
                ),
                "minimum_frames": 0,
                "splits": set(),
            })
            task["minimum_frames"] = max(
                int(task["minimum_frames"]), item.maximum_evidenced_raw_index + 1
            )
            task["splits"].add(split)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    submitted = [
        (value["xtc"], value["sidecar"], int(value["minimum_frames"]))
        for value in tasks.values()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(_build_one, task): task for task in submitted}
        for completed, future in enumerate(as_completed(future_to_task), start=1):
            result = future.result()
            result["splits"] = sorted(tasks[result["xtc"]]["splits"])
            results.append(result)
            print(json.dumps({
                "completed": completed,
                "total": len(submitted),
                "xtc": result["xtc"],
                "seconds": result.get("scan_seconds", 0.0),
                "reused": result["reused"],
            }, sort_keys=True), flush=True)
    stable_results = [
        {key: value for key, value in result.items() if key not in {"scan_seconds", "reused"}}
        for result in sorted(results, key=lambda row: row["xtc"])
    ]
    cache_hash = canonical_hash(stable_results)
    manifest = {
        "schema": "molvid.raw_xtc_offset_cache.v1",
        "raw_root": str(args.raw_root.resolve()),
        "manifest_root": str(args.manifest_root.resolve()),
        "splits": ["train", "valid"],
        "trajectories": len(stable_results),
        "source_bytes": sum(int(row["xtc_size"]) for row in stable_results),
        "sidecar_bytes": sum(Path(row["sidecar"]).stat().st_size for row in stable_results),
        "cache_hash": cache_hash,
        "records": stable_results,
        "split_source_identities": split_identities,
        "workers": args.workers,
        "wall_seconds": time.perf_counter() - started,
        "gpu_hours": 0.0,
        "test_opened": False,
    }
    atomic_write_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "complete": True, "trajectories": len(stable_results),
        "cache_hash": cache_hash, "wall_seconds": manifest["wall_seconds"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
