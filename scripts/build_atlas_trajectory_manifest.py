#!/usr/bin/env python
"""Build a temporal 49-clip/13-clip ATLAS train/validation selection.

The generated index files retain the original byte offsets and point at the
source ``data.bin`` through a symlink.  No clip payload is copied or changed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SAMPLE_RE = re.compile(
    r"(?P<system>atlas_[^_]+_[^_]+)_R(?P<replica>[0-9]+)_w(?P<window>[0-9]+)$"
)
DEFAULT_SYSTEMS = ("atlas_5e3e_A", "atlas_1v7r_A", "atlas_2wlt_A")


def _read_index(root: Path) -> dict[str, tuple[str, int, str, int]]:
    records: dict[str, tuple[str, int, str, int]] = {}
    with (root / "index.txt").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"malformed index line {line_number + 1}: {line!r}")
            sample_id = fields[0]
            if sample_id in records:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            records[sample_id] = (line, line_number, str(root), int(fields[3]))
    return records


def _parse(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"cannot parse ATLAS sample_id: {sample_id}")
    return match.group("system"), int(match.group("replica")), int(match.group("window"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--train-clips", type=int, default=49)
    parser.add_argument("--valid-clips", type=int, default=13)
    args = parser.parse_args()

    source_roots = [Path(root) for root in args.source_root]
    all_records: dict[str, tuple[str, int, str, int]] = {}
    for source_root in source_roots:
        for sample_id, location in _read_index(source_root).items():
            if sample_id in all_records:
                raise ValueError(f"duplicate sample_id across source roots: {sample_id}")
            all_records[sample_id] = location

    required_windows = args.train_clips + args.valid_clips
    selected: dict[str, dict[int, list[tuple[int, str, tuple[str, int, str, int]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample_id, location in all_records.items():
        system, replica, window = _parse(sample_id)
        if system in args.systems and replica in args.replicas:
            selected[system][replica].append((window, sample_id, location))

    train_lines: list[str] = []
    valid_lines: list[str] = []
    train_ids: list[str] = []
    valid_ids: list[str] = []
    tracks: list[dict[str, Any]] = []
    source_roots_used: set[Path] = set()
    for system in args.systems:
        for replica in args.replicas:
            rows = sorted(selected[system][replica], key=lambda item: item[0])
            windows = [item[0] for item in rows]
            if windows != list(range(required_windows)):
                raise ValueError(
                    f"{system} R{replica}: expected windows 0..{required_windows - 1}, "
                    f"found {windows}"
                )
            train_rows = rows[: args.train_clips]
            valid_rows = rows[args.train_clips :]
            if len(valid_rows) != args.valid_clips:
                raise ValueError(f"{system} R{replica}: validation clip count mismatch")
            train_lines.extend(row[2][0] for row in train_rows)
            valid_lines.extend(row[2][0] for row in valid_rows)
            train_ids.extend(row[1] for row in train_rows)
            valid_ids.extend(row[1] for row in valid_rows)
            source_roots_used.update(Path(row[2][2]) for row in rows)
            tracks.append(
                {
                    "system": system,
                    "replica": replica,
                    "atom_count": train_rows[0][2][3],
                    "train_windows": [row[0] for row in train_rows],
                    "valid_windows": [row[0] for row in valid_rows],
                    "train_sample_ids": [row[1] for row in train_rows],
                    "valid_sample_ids": [row[1] for row in valid_rows],
                    "source_roots": sorted({row[2][2] for row in rows}),
                }
            )

    if len(source_roots_used) != 1:
        raise ValueError(
            "selected tracks span multiple source data.bin files; this lightweight "
            "index-only subset requires one source root: "
            + ", ".join(sorted(str(root) for root in source_roots_used))
        )
    source_root = next(iter(source_roots_used))
    output_root = args.output_root
    train_root = output_root / "train"
    valid_root = output_root / "valid"
    train_root.mkdir(parents=True, exist_ok=True)
    valid_root.mkdir(parents=True, exist_ok=True)
    for root, lines in ((train_root, train_lines), (valid_root, valid_lines)):
        (root / "index.txt").write_text("".join(lines), encoding="utf-8")
        data_link = root / "data.bin"
        if data_link.exists() or data_link.is_symlink():
            data_link.unlink()
        data_link.symlink_to(source_root / "data.bin")
        (root / "stats.json").write_text(
            json.dumps(
                {
                    "count": len(lines),
                    "storage_format": "npz-v1",
                    "source_root": str(source_root),
                    "selection": "9 trajectories; first 49 clips train, last 13 clips validation",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    selected_metrics = [item for item in metrics.get("systems", []) if item.get("system") in args.systems]
    manifest = {
        "selection": {
            "dataset": "ATLAS",
            "systems": list(args.systems),
            "replicas": list(args.replicas),
            "trajectory_count": len(args.systems) * len(args.replicas),
            "train_clips_per_trajectory": args.train_clips,
            "validation_clips_per_trajectory": args.valid_clips,
            "train_window_range": [0, args.train_clips - 1],
            "validation_window_range": [args.train_clips, required_windows - 1],
            "split_policy": "temporal split within each replica; no cross-trajectory mixing",
        },
        "source_root": str(source_root),
        "train_root": str(train_root),
        "valid_root": str(valid_root),
        "metrics": selected_metrics,
        "tracks": tracks,
        "train_sample_ids": train_ids,
        "valid_sample_ids": valid_ids,
        "leakage_check": {
            "train_count": len(train_ids),
            "validation_count": len(valid_ids),
            "intersection_count": len(set(train_ids) & set(valid_ids)),
            "train_trajectory_count": len({sample_id.rsplit("_w", 1)[0] for sample_id in train_ids}),
            "validation_trajectory_count": len({sample_id.rsplit("_w", 1)[0] for sample_id in valid_ids}),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["leakage_check"], sort_keys=True))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
