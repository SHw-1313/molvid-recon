#!/usr/bin/env python3
"""Pack the exact selected T0 NPZ payloads into a portable lazy clip store.

The selected index files reference sparse offsets in a much larger immutable
source data.bin.  This utility copies each referenced byte range verbatim and
rewrites only the local offsets; it never decodes or regenerates a clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/atlas_selected_trajectories/clip_store"
DEFAULT_OUTPUT = ROOT / "outputs/state_detail_codec_v2/t0_data/clip_store"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_split(source_root: Path, output_root: Path, split: str) -> dict[str, Any]:
    source_index = source_root / split / "index.txt"
    source_data = (source_root / split / "data.bin").resolve()
    if not source_index.is_file() or not source_data.is_file():
        raise FileNotFoundError(f"source {split} store is incomplete: {source_root / split}")
    output_split = output_root / split
    output_split.mkdir(parents=True)
    output_data = output_split / "data.bin"
    lines: list[str] = []
    payloads: list[dict[str, Any]] = []
    with source_data.open("rb") as source_handle, output_data.open("wb") as output_handle:
        for line_number, line in enumerate(source_index.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 6:
                raise ValueError(f"invalid source index row {source_index}:{line_number}")
            sample_id, start_text, end_text = fields[:3]
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start:
                raise ValueError(f"invalid source byte range for {sample_id}: {start}:{end}")
            source_handle.seek(start)
            payload = source_handle.read(end - start)
            if len(payload) != end - start:
                raise ValueError(f"short source payload for {sample_id}")
            packed_start = output_handle.tell()
            output_handle.write(payload)
            packed_end = output_handle.tell()
            fields[1] = str(packed_start)
            fields[2] = str(packed_end)
            lines.append("\t".join(fields))
            payloads.append(
                {
                    "sample_id": sample_id,
                    "source_start": start,
                    "source_end": end,
                    "packed_start": packed_start,
                    "packed_end": packed_end,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    (output_split / "index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_path = source_root / split / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    stats["count"] = len(lines)
    stats["compressed_bytes"] = output_data.stat().st_size
    (output_split / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "source_index": str(source_index),
        "source_index_sha256": _sha256(source_index),
        "source_data": str(source_data),
        "source_data_size": source_data.stat().st_size,
        "packed_index": str(output_split / "index.txt"),
        "packed_index_sha256": _sha256(output_split / "index.txt"),
        "packed_data": str(output_data),
        "packed_data_size": output_data.stat().st_size,
        "payload_count": len(payloads),
        "payloads": payloads,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source_root = args.source_root if args.source_root.is_absolute() else ROOT / args.source_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite packed T0 store: {output_root}")
    output_root.mkdir(parents=True)
    for name in ("manifest.json",):
        shutil.copy2(source_root / name, output_root / name)
    provenance = {
        "schema_version": "pvb.codec.state_detail.t0_packed_store.v1",
        "source_root": str(source_root),
        "source_manifest_sha256": _sha256(source_root / "manifest.json"),
        "splits": {
            split: _pack_split(source_root, output_root, split)
            for split in ("train", "valid")
        },
    }
    (output_root / "payload_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "train_payload_count": provenance["splits"]["train"]["payload_count"],
                "valid_payload_count": provenance["splits"]["valid"]["payload_count"],
                "train_bytes": provenance["splits"]["train"]["packed_data_size"],
                "valid_bytes": provenance["splits"]["valid"]["packed_data_size"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
