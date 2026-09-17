#!/usr/bin/env python
"""Build half-size ordered ATLAS/MISATO clip stores for the codec path.

Example (the paths are intentionally supplied by the operator)::

    python scripts/preprocess_trajectory_clips.py atlas \
      --root /data1/repo/BioKinema/data_atlas \
      --output-root /data4/users/sihao/data/pvb_cross_dataset_20260810

The default ATLAS stride is 10 (100 ps sampled clips from the verified 10 ps
native trajectory); MISATO keeps its verified 80 ps native interval.  Neither
path interpolates coordinates or changes the native timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.clip_dataset import ClipMMapWriter
from data.trajectory_clips import (
    ClipPreprocessConfig,
    SourceDataError,
    _bucket_id,
    _inventory,
    atlas_split,
    iter_atlas_records,
    iter_misato_records,
    select_fraction,
    write_split_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("atlas", "misato"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument(
        "--source-stride",
        type=int,
        default=None,
        help="raw-frame stride; defaults to 10 for ATLAS and 1 for MISATO",
    )
    parser.add_argument("--atlas-native-dt-ps", type=float, default=10.0)
    parser.add_argument("--misato-native-dt-ps", type=float, default=80.0)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop on the first invalid source system instead of recording a skip",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace output stores created by an earlier run",
    )
    return parser


def _read_atlas_ids(root: Path) -> list[str]:
    path = root / "atlas_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing ATLAS system manifest: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise SourceDataError("ATLAS system manifest contains duplicate systems")
    return ids


def _read_misato_ids(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for filename, split in (
        ("train_MD.txt", "train"),
        ("val_MD.txt", "valid"),
        ("test_MD.txt", "test"),
    ):
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"missing MISATO split manifest: {path}")
        ids = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise SourceDataError(f"duplicate MISATO systems in {path}")
        result[split] = ids
    all_ids = [item for ids in result.values() for item in ids]
    if len(all_ids) != len(set(all_ids)):
        raise SourceDataError("MISATO train/val/test manifests overlap")
    return result


def _choose_systems(
    source: str,
    root: Path,
    *,
    fraction: float,
    seed: int,
    max_systems: int | None,
) -> dict[str, list[str]]:
    if source == "atlas":
        selected = [
            system
            for system in _read_atlas_ids(root)
            if select_fraction(system, fraction=fraction, seed=seed)
        ]
        result: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
        for system in selected:
            result[atlas_split(system, seed=seed)].append(system)
        if max_systems is not None:
            ordered = [system for split in ("train", "valid", "test") for system in result[split]]
            keep = set(ordered[:max_systems])
            result = {split: [system for system in systems if system in keep] for split, systems in result.items()}
        return result

    result = {}
    for split, systems in _read_misato_ids(root).items():
        selected = [
            system
            for system in systems
            if select_fraction(system, fraction=fraction, seed=seed)
        ]
        result[split] = selected
    if max_systems is not None:
        ordered = [system for split in ("train", "valid", "test") for system in result[split]]
        keep = set(ordered[:max_systems])
        result = {split: [system for system in systems if system in keep] for split, systems in result.items()}
    return result


def _inventory_paths(source: str, root: Path, systems_by_split: dict[str, list[str]]) -> list[Path]:
    paths: list[Path] = []
    if source == "atlas":
        for system in [item for values in systems_by_split.values() for item in values]:
            system_dir = root / system
            paths.extend(system_dir.glob("*"))
    else:
        paths.extend(
            root / name
            for name in ("MD.hdf5", "train_MD.txt", "val_MD.txt", "test_MD.txt")
        )
        for system in [item for values in systems_by_split.values() for item in values]:
            paths.append(root / "parameter_restart_files_MD" / system.lower() / f"{system}.pdb")
    return [path for path in paths if path.exists()]


def _process(
    args: argparse.Namespace,
    config: ClipPreprocessConfig,
    systems_by_split: dict[str, list[str]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    source = args.source
    root = args.root
    source_dt = config.atlas_native_dt_ps if source == "atlas" else config.misato_native_dt_ps
    bucket_root = args.output_root / "clips" / source / _bucket_id(source_dt * config.source_stride)
    counts: dict[str, int] = {}
    skipped: list[dict[str, Any]] = []
    hdf5 = None
    if source == "misato":
        hdf5 = h5py.File(root / "MD.hdf5", "r")
    try:
        for split in ("train", "valid", "test"):
            output_dir = bucket_root / split
            writer = ClipMMapWriter(output_dir, overwrite=args.overwrite)
            record_count = 0
            success_count = 0
            split_systems = systems_by_split[split]
            try:
                for system_index, system in enumerate(split_systems, start=1):
                    if system_index == 1 or system_index % 10 == 0:
                        print(
                            f"[{source}/{split}] system {system_index}/{len(split_systems)} "
                            f"{system}; records={record_count}",
                            flush=True,
                        )
                    try:
                        if source == "atlas":
                            records = iter_atlas_records(
                                root,
                                system,
                                split=split,
                                config=config,
                            )
                        else:
                            assert hdf5 is not None
                            records = iter_misato_records(
                                root,
                                system,
                                split=split,
                                hdf5=hdf5,
                                config=config,
                            )
                        system_records = 0
                        for record in records:
                            writer.append(record)
                            record_count += 1
                            system_records += 1
                        if system_records == 0:
                            raise SourceDataError("trajectory yielded no complete clip")
                        success_count += 1
                    except Exception as exc:  # recorded; strict mode makes this fatal
                        error = {
                            "source": source,
                            "split": split,
                            "system_id": system,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        skipped.append(error)
                        if args.strict:
                            raise
            finally:
                writer.close()
            counts[f"{split}_systems_ok"] = success_count
            counts[f"{split}_records"] = record_count
    finally:
        if hdf5 is not None:
            hdf5.close()
    counts["systems_selected"] = sum(len(items) for items in systems_by_split.values())
    counts["systems_skipped"] = len(skipped)
    counts["records_total"] = sum(value for key, value in counts.items() if key.endswith("_records"))
    return counts, skipped


def main() -> None:
    args = _parser().parse_args()
    source_stride = args.source_stride
    if source_stride is None:
        source_stride = 10 if args.source == "atlas" else 1
    config = ClipPreprocessConfig(
        clip_len=args.clip_len,
        window_stride=args.window_stride,
        source_stride=source_stride,
        atlas_native_dt_ps=args.atlas_native_dt_ps,
        misato_native_dt_ps=args.misato_native_dt_ps,
    )
    config.validate()
    if not args.root.exists():
        raise FileNotFoundError(args.root)
    systems_by_split = _choose_systems(
        args.source,
        args.root,
        fraction=args.fraction,
        seed=args.seed,
        max_systems=args.max_systems,
    )
    counts, skipped = _process(args, config, systems_by_split)
    inventory = _inventory(_inventory_paths(args.source, args.root, systems_by_split), args.root)
    source_dt = config.atlas_native_dt_ps if args.source == "atlas" else config.misato_native_dt_ps
    bucket_root = args.output_root / "clips" / args.source / _bucket_id(source_dt * config.source_stride)
    manifest = write_split_manifest(
        bucket_root / "manifest.json",
        source=args.source,
        raw_root=args.root,
        config=config,
        source_version=(
            "ATLAS archive README: 100 ns trajectories saved every 10 ps"
            if args.source == "atlas"
            else "MISATO Zenodo record 7711953; native dt supplied by agents/HANDOFF.md"
        ),
        timestamp_provenance=(
            "ATLAS_XTC_time_ps" if args.source == "atlas" else "MISATO_Handoff_verified_native_dt_ps"
        ),
        topology_policy=(
            "PDB topology, heavy supported protein atoms, bonds from topology"
            if args.source == "atlas"
            else "provided MISATO PDB topology, heavy supported protein+MOL atoms, bonds from topology"
        ),
        split_policy=(
            f"system-level deterministic hash split/filter; fraction={args.fraction}; seed={args.seed}"
            if args.source == "atlas"
            else f"official MISATO split preserved; deterministic system filter fraction={args.fraction}; seed={args.seed}"
        ),
        systems=[system for split in ("train", "valid", "test") for system in systems_by_split[split]],
        inventory=inventory,
        counts=counts,
    )
    manifest["skipped"] = skipped
    (bucket_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bucket_root": str(bucket_root), "counts": counts, "skipped": skipped}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
