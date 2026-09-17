#!/usr/bin/env python3
"""Convert current npz-v1 clips into the original PVB gzip-JSON pair store.

This is an external data adapter.  It does not import or modify PVB_origin.
Each T=16 clip contributes exactly T-1 adjacent pairs, preserving the stored
atom order, topology, coordinates, masks, and native physical interval.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import mmap
import time
from pathlib import Path
from typing import Iterator

import numpy as np


ARRAY_KEYS = (
    "x",
    "bpos",
    "atype",
    "btype",
    "edge_mask",
    "loss_mask",
    "bond_index",
    "time_ps",
    "delta_time_ps",
)


class ClipStore:
    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "index.txt"
        self.data_path = root / "data.bin"
        self.entries: list[tuple[str, int, int]] = []
        with self.index_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    raise ValueError(f"malformed clip index line: {line!r}")
                self.entries.append((fields[0], int(fields[1]), int(fields[2])))

    def __iter__(self) -> Iterator[dict]:
        data_file = self.data_path.open("rb")
        mapped = mmap.mmap(data_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for sample_id, start, end in self.entries:
                payload = mapped[start:end]
                if payload[:2] != b"PK":
                    raise ValueError(
                        f"{self.root}: expected npz-v1 payload for {sample_id}"
                    )
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata"].item()))
                    record = dict(metadata)
                    for key in ARRAY_KEYS:
                        record[key] = np.array(archive[key], copy=True)
                record["sample_id"] = str(record.get("sample_id", sample_id))
                yield record
        finally:
            mapped.close()
            data_file.close()


def _json_bytes(value: dict) -> bytes:
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return gzip.compress(raw, compresslevel=6, mtime=0)


def _pair_record(clip: dict, frame: int) -> dict:
    x = np.asarray(clip["x"], dtype=np.float32)
    bpos = np.asarray(clip["bpos"], dtype=np.float32)
    atype = np.asarray(clip["atype"], dtype=np.int64)
    btype = np.asarray(clip["btype"], dtype=np.int64)
    edge_mask = np.asarray(clip["edge_mask"], dtype=np.int64)
    loss_mask = np.asarray(clip["loss_mask"], dtype=np.bool_)
    bond_index = np.asarray(clip["bond_index"], dtype=np.int64)
    time_ps = np.asarray(clip["time_ps"], dtype=np.float32)
    delta_ps = np.asarray(clip["delta_time_ps"], dtype=np.float32)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[-1] != 3:
        raise ValueError("clip x must have shape [T>=2,N,3]")
    if bpos.shape != x.shape:
        raise ValueError("clip bpos shape does not match x")
    if frame < 0 or frame + 1 >= x.shape[0]:
        raise IndexError(frame)
    n_atoms = x.shape[1]
    if any(value.shape != (n_atoms,) for value in (atype, btype, edge_mask, loss_mask)):
        raise ValueError("clip atom fields do not match x")
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError("clip bond_index must have shape [2,E]")
    if time_ps.shape != (x.shape[0],) or delta_ps.shape != (x.shape[0] - 1,):
        raise ValueError("clip time fields do not match x")
    source = str(clip.get("source", "unknown"))
    return {
        "atype": atype.tolist(),
        "btype": btype.tolist(),
        "edge_mask": edge_mask.tolist(),
        "mask": loss_mask.tolist(),
        "x0": x[frame].tolist(),
        "b0": bpos[frame].tolist(),
        "x1": x[frame + 1].tolist(),
        "b1": bpos[frame + 1].tolist(),
        "bond_index": bond_index.tolist(),
        "env": 11 if source == "misato" else 3,
        "baseline_d_source": source,
        "baseline_d_clip_id": str(clip.get("sample_id", "")),
        "baseline_d_frame": int(frame),
        "baseline_d_time_ps": float(time_ps[frame]),
        "baseline_d_delta_time_ps": float(delta_ps[frame]),
        "baseline_d_time_bucket_id": str(clip.get("time_bucket_id", "")),
    }


def convert(input_root: Path, output_root: Path, *, overwrite: bool) -> dict:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output store: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "data.bin"
    index_path = output_root / "index.txt"
    stats_path = output_root / "stats.json"
    if overwrite:
        for path in (data_path, index_path, stats_path):
            if path.exists():
                path.unlink()

    started = time.time()
    clip_count = 0
    pair_count = 0
    bytes_written = 0
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    with data_path.open("wb") as data_file, index_path.open("w", encoding="utf-8") as index_file:
        for clip in ClipStore(input_root):
            clip_count += 1
            source = str(clip.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            bucket = str(clip.get("time_bucket_id", ""))
            for frame in range(int(np.asarray(clip["x"]).shape[0]) - 1):
                pair = _pair_record(clip, frame)
                compressed = _json_bytes(pair)
                start = bytes_written
                data_file.write(compressed)
                bytes_written += len(compressed)
                pair_id = f"{clip['sample_id']}_f{frame:02d}"
                # Original UniDataset.get_len reads property zero as its batch complexity.
                index_file.write(f"{pair_id}\t{start}\t{bytes_written}\t{len(pair['atype'])}\n")
                pair_count += 1
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    report = {
        "schema_version": "pvb.origin.pair-migration.v1",
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "input_storage_format": "npz-v1",
        "output_storage_format": "gzip-json-v1 (original PVB mmap)",
        "mapping": "each T=16 clip -> 15 adjacent (frame j, frame j+1) pairs; no interpolation, reordering, recentering, graph cut, or PDB input",
        "coordinate_unit": "angstrom",
        "clip_count": clip_count,
        "pair_count": pair_count,
        "bytes_written": bytes_written,
        "source_clip_counts": source_counts,
        "pair_counts_by_time_bucket": bucket_counts,
        "elapsed_seconds": time.time() - started,
    }
    stats_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = convert(args.input_root, args.output_root, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
