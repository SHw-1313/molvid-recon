#!/usr/bin/env python
"""Measure motion statistics for candidate ATLAS trajectories.

The processed ATLAS clips are aligned independently at clip creation time.
Consequently, all statistics here are computed within each 16-frame clip;
transitions across clip boundaries are deliberately not treated as physical
frame-to-frame transitions.

The reported high-pass energy is the mean coordinate power in temporal FFT
bins k >= 2 (the DC and slowest non-zero bin are excluded), in Angstrom^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data.clip_dataset import ClipMMapDataset


SAMPLE_RE = re.compile(
    r"(?P<system>atlas_[^_]+_[^_]+)_R(?P<replica>[0-9]+)_w(?P<window>[0-9]+)$"
)
DEFAULT_SYSTEMS = ("atlas_5e3e_A", "atlas_1v7r_A", "atlas_2wlt_A")


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a row-vector rotation and translation mapping mobile to reference."""

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vh = np.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def _clip_metrics(record: dict[str, Any]) -> dict[str, float]:
    x = np.asarray(record["x"], dtype=np.float64)
    valid_mask = np.asarray(record["loss_mask"], dtype=bool)
    align_mask = np.asarray(record["align_mask"], dtype=bool) & valid_mask
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"unexpected coordinate shape: {x.shape}")
    if int(align_mask.sum()) < 3:
        raise ValueError("at least three alignment atoms are required")

    reference = x[0, align_mask]
    aligned = np.empty_like(x)
    aligned[0] = x[0]
    frame_rmsd: list[float] = []
    local_displacements: list[float] = []
    for frame_idx in range(1, x.shape[0]):
        rotation, translation = _kabsch(x[frame_idx, align_mask], reference)
        aligned_frame = x[frame_idx] @ rotation + translation
        aligned[frame_idx] = aligned_frame

        displacement = aligned_frame[valid_mask] - aligned[frame_idx - 1, valid_mask]
        local = np.sqrt(np.sum(displacement * displacement, axis=-1))
        frame_rmsd.append(float(np.sqrt(np.mean(displacement * displacement))))
        local_displacements.append(float(local.max()))

    centered = aligned[:, valid_mask] - aligned[:, valid_mask].mean(axis=0, keepdims=True)
    rmsf_per_atom = np.sqrt(np.mean(np.sum(centered * centered, axis=-1), axis=0))
    displacement_from_first = np.sqrt(
        np.sum((aligned[:, valid_mask] - aligned[0, valid_mask]) ** 2, axis=-1)
    )

    # np.fft.fft has bins [0, 1, ..., T/2, ..., T-1].  Remove DC and
    # the slowest non-zero pair (k=1 and k=T-1), then apply Parseval's
    # normalization to obtain mean coordinate power in Angstrom^2.
    temporal = aligned[:, valid_mask] - aligned[:, valid_mask].mean(axis=0, keepdims=True)
    spectrum = np.fft.fft(temporal, axis=0)
    high_pass_bins = np.ones(x.shape[0], dtype=bool)
    high_pass_bins[0] = False
    high_pass_bins[1] = False
    high_pass_bins[-1] = False
    high_pass_energy = np.abs(spectrum[high_pass_bins]) ** 2
    high_pass_energy = float(np.mean(np.sum(high_pass_energy, axis=0) / x.shape[0] ** 2))

    return {
        "median_frame_to_frame_aligned_rmsd": float(np.median(frame_rmsd)),
        "mean_frame_to_frame_aligned_rmsd": float(np.mean(frame_rmsd)),
        "chunk_high_pass_energy": high_pass_energy,
        "rmsf_mean": float(np.mean(rmsf_per_atom)),
        "mean_displacement_from_clip_start": float(np.mean(displacement_from_first)),
        "max_local_displacement": float(np.max(local_displacements)),
        "p95_local_displacement": float(np.percentile(local_displacements, 95)),
    }


def _load_index(roots: list[Path]) -> tuple[dict[str, tuple[ClipMMapDataset, int, str, int]], dict[str, Any]]:
    records: dict[str, tuple[ClipMMapDataset, int, str, int]] = {}
    datasets: dict[str, Any] = {}
    for root in roots:
        split = root.name
        dataset = ClipMMapDataset(str(root))
        datasets[split] = dataset
        index_path = root / "index.txt"
        with index_path.open() as handle:
            for index, line in enumerate(handle):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    raise ValueError(f"invalid index line in {index_path}: {line!r}")
                sample_id = fields[0]
                if sample_id in records:
                    raise ValueError(f"duplicate sample_id across roots: {sample_id}")
                records[sample_id] = (dataset, index, split, int(fields[3]))
    return records, datasets


def _parse_sample(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"cannot parse ATLAS sample_id: {sample_id}")
    return match.group("system"), int(match.group("replica")), int(match.group("window"))


def measure_system(
    system: str,
    records: dict[str, tuple[ClipMMapDataset, int, str, int]],
    replicas: tuple[int, ...] = (1, 2, 3),
    clip_count: int = 62,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectory_rows: list[dict[str, Any]] = []
    for replica in replicas:
        selected: list[tuple[int, str, tuple[ClipMMapDataset, int, str, int]]] = []
        for sample_id, location in records.items():
            parsed_system, parsed_replica, window = _parse_sample(sample_id)
            if parsed_system == system and parsed_replica == replica:
                selected.append((window, sample_id, location))
        selected.sort()
        if len(selected) != clip_count or [window for window, _, _ in selected] != list(range(clip_count)):
            raise ValueError(
                f"{system} R{replica}: expected windows 0..{clip_count - 1}, "
                f"found {[window for window, _, _ in selected]}"
            )

        per_clip = []
        for window, sample_id, (dataset, index, split, atom_count) in selected:
            per_clip.append(_clip_metrics(dataset[index]))
        row: dict[str, Any] = {
            "system": system,
            "replica": replica,
            "atom_count": selected[0][2][3],
            "clip_count": len(selected),
            "source_split": sorted({location[2] for _, _, location in selected}),
            "sample_ids": [sample_id for _, sample_id, _ in selected],
        }
        for key in per_clip[0]:
            values = np.asarray([item[key] for item in per_clip], dtype=np.float64)
            row[key] = float(np.max(values)) if key == "max_local_displacement" else float(np.mean(values))
            row[f"{key}_p50"] = float(np.percentile(values, 50))
            row[f"{key}_p95"] = float(np.percentile(values, 95))
        row["frame_transition_count"] = (len(selected) * 15)
        trajectory_rows.append(row)

    metric_keys = (
        "median_frame_to_frame_aligned_rmsd",
        "chunk_high_pass_energy",
        "rmsf_mean",
        "mean_displacement_from_clip_start",
        "max_local_displacement",
    )
    system_row: dict[str, Any] = {
        "system": system,
        "atom_count": trajectory_rows[0]["atom_count"],
        "replicas": list(replicas),
        "clip_count_per_replica": clip_count,
    }
    for key in metric_keys:
        values = [row[key] for row in trajectory_rows]
        system_row[key] = float(np.median(values))
        system_row[f"{key}_replica_min"] = float(np.min(values))
        system_row[f"{key}_replica_max"] = float(np.max(values))
    return trajectory_rows, system_row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = [{key: value for key, value in row.items() if not isinstance(value, (list, dict))} for row in rows]
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=[],
        help="ATLAS clip split root; repeat for train and valid",
    )
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [Path(item) for item in args.roots] or [
        Path("/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps/train"),
        Path("/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps/valid"),
    ]
    records, datasets = _load_index(roots)
    trajectory_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    for system in args.systems:
        trajectories, system_row = measure_system(system, records)
        trajectory_rows.extend(trajectories)
        system_rows.append(system_row)
        print(json.dumps(system_row, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": {
            "coordinate_unit": "angstrom",
            "clip_frames": 16,
            "frame_interval_ps": 100,
            "clip_boundary_policy": "exclude_cross_clip_transitions",
            "alignment_atoms": "record.align_mask intersect record.loss_mask",
            "high_pass": "FFT bins k>=2; exclude DC and k=1 pair; Parseval-normalized mean coordinate power",
            "rmsf": "per-atom RMS fluctuation within each aligned clip, averaged over atoms and clips",
            "local_displacement": "maximum valid-atom displacement over aligned adjacent frames within clips",
        },
        "roots": [str(root) for root in roots],
        "systems": system_rows,
        "trajectories": trajectory_rows,
        "dataset_lengths": {split: len(dataset) for split, dataset in datasets.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(args.output.with_suffix(".trajectories.csv"), trajectory_rows)
    _write_csv(args.output.with_suffix(".systems.csv"), system_rows)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
