#!/usr/bin/env python
"""Convert prepared PVB clip records to the legacy PVB pair-store format.

This wrapper is intentionally kept in the isolated run directory.  It does
not modify PVB_origin or the prepared clip stores.  One legacy record is
written for each clip, using frame 0 as x0 and frame 1 as x1; the topology and
atom order are copied unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path


def _compress(data: dict) -> bytes:
    payload = json.dumps(data).encode("utf-8")
    return gzip.compress(payload, compresslevel=6)


def _write_store(source: str, input_root: Path, output_root: Path, split: str,
                 max_records: int | None) -> dict:
    # Import the current clip reader only; the training/evaluation code stays
    # in the unmodified reference checkout.
    from data.clip_dataset import ClipMMapDataset

    source_root = input_root / split
    target_root = output_root / source / split.replace("valid", "valid_block")
    target_root.mkdir(parents=True, exist_ok=True)
    data_path = target_root / "data.bin"
    index_path = target_root / "index.txt"
    if data_path.exists() or index_path.exists():
        raise FileExistsError(f"refusing to overwrite existing store: {target_root}")

    dataset = ClipMMapDataset(source_root)
    count = min(len(dataset), max_records) if max_records is not None else len(dataset)
    offset = 0
    atom_total = 0
    byte_total = 0
    sha = hashlib.sha256()
    started = time.time()
    with data_path.open("wb") as data_file, index_path.open("w", encoding="utf-8") as index_file:
        for index in range(count):
            item = dataset[index]
            x = item["x"]
            bpos = item["bpos"]
            if x.shape[0] < 2:
                raise ValueError(f"clip {item.get('sample_id', index)} has fewer than two frames")
            if x.shape[1] != len(item["atype"]):
                raise ValueError(f"atom count mismatch in {item.get('sample_id', index)}")
            delta = float(item["delta_time_ps"][0])
            record = {
                "atype": item["atype"].astype("int64").tolist(),
                "btype": item["btype"].astype("int64").tolist(),
                "x0": x[0].astype("float32").tolist(),
                "b0": bpos[0].astype("float32").tolist(),
                "x1": x[1].astype("float32").tolist(),
                "b1": bpos[1].astype("float32").tolist(),
                "edge_mask": item["edge_mask"].astype("int64").tolist(),
                "mask": item["loss_mask"].astype("bool").tolist(),
                "bond_index": item["bond_index"].astype("int64").tolist(),
                "source": str(item["source"]),
                "system_id": str(item["system_id"]),
                "replica": str(item["replica"]),
                "clip_id": str(item["sample_id"]),
                "time_bucket_id": str(item["time_bucket_id"]),
                "delta_time_ps": delta,
                "frame_start_ps": float(item["time_ps"][0]),
                "frame_end_ps": float(item["time_ps"][1]),
                "migration": "pvb.clip.v1.frame0_to_frame1.legacy_pair.v1",
            }
            compressed = _compress(record)
            written = data_file.write(compressed)
            clip_id = str(item["sample_id"])
            # The first property is the legacy DynamicBatchWrapper complexity
            # value.  N is conservative and matches the original n complexity.
            n_atoms = int(x.shape[1])
            index_file.write(f"{clip_id}\t{offset}\t{offset + written}\t{n_atoms}\n")
            offset += written
            byte_total += written
            atom_total += n_atoms
            sha.update(compressed)
            if (index + 1) % 1000 == 0:
                data_file.flush()
                index_file.flush()
                print(f"{source}/{split}: {index + 1}/{count}", flush=True)

    stats = {
        "source": source,
        "split": split,
        "input_root": str(source_root),
        "output_root": str(target_root),
        "input_records": len(dataset),
        "records_written": count,
        "total_atoms": atom_total,
        "compressed_bytes": byte_total,
        "data_sha256": sha.hexdigest(),
        "frame_mapping": {"x0": 0, "x1": 1},
        "time_policy": "native clip interval; no interpolation",
        "elapsed_s": time.time() - started,
    }
    (target_root / "migration_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--misato-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for source, root in (("atlas", args.atlas_root), ("misato", args.misato_root)):
        for split in ("train", "valid"):
            all_stats.append(_write_store(source, root, args.output_root, split, args.max_records))
    manifest = {
        "schema": "pvb.origin.legacy_pair_migration.v1",
        "sources": all_stats,
        "seed": 20260810,
        "pdb_used": False,
    }
    (args.output_root / "migration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
