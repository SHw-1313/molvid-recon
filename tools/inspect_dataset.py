"""Inspect verified train/validation clip stores without opening the test split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from molvid.data.manifest import load_datasets
from molvid.data.store import ClipMMapDataset


_SAMPLE = re.compile(r"^(?P<system>.+)_(?P<replica>R[0-9]+)_w[0-9]+$")


def inspect_split(dataset: ClipMMapDataset, *, max_clips: int | None = None) -> dict[str, Any]:
    """Count real stored records; mark a limited scan as partial."""

    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be positive")
    available = len(dataset)
    scanned = min(available, max_clips) if max_clips is not None else available
    systems: set[str] = set()
    trajectories: set[str] = set()
    buckets: Counter[str] = Counter()
    atoms: list[int] = []
    frames: list[int] = []
    intervals: list[float] = []
    for index in range(scanned):
        record = dataset[index]
        sample_id = str(record["sample_id"])
        match = _SAMPLE.match(sample_id)
        if match is None:
            system = str(record.get("source", sample_id))
            trajectory = sample_id
        else:
            system = match.group("system")
            trajectory = f"{system}_{match.group('replica')}"
        systems.add(system)
        trajectories.add(trajectory)
        frame_count, atom_count = np.asarray(record["x"]).shape[:2]
        atoms.append(int(atom_count))
        frames.append(int(frame_count))
        buckets[str(record["time_bucket_id"])] += 1
        intervals.extend(float(value) for value in np.asarray(record["delta_time_ps"]))
    return {
        "clips_available": available,
        "clips_scanned": scanned,
        "partial": scanned != available,
        "systems_scanned": len(systems),
        "trajectories_scanned": len(trajectories),
        "by_time_bucket": dict(sorted(buckets.items())),
        "atoms_min": min(atoms, default=None),
        "atoms_max": max(atoms, default=None),
        "frames_min": min(frames, default=None),
        "frames_max": max(frames, default=None),
        "interval_ps_min": min(intervals, default=None),
        "interval_ps_max": max(intervals, default=None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "both"), default="both")
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args(argv)
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    datasets = load_datasets(args.manifest_root)
    try:
        selected = ("train", "valid") if args.split == "both" else (args.split,)
        result = {
            "schema_version": "molvid.dataset.inspection.v1",
            "data_hash": datasets.data_hash,
            "splits": {
                split: inspect_split(getattr(datasets, split), max_clips=args.max_clips)
                for split in selected
            },
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        datasets.close()


if __name__ == "__main__":
    raise SystemExit(main())
