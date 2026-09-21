"""Evaluate Frame Joint on one fixed valid window from each original trajectory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.runner import trajectory_metrics
from molvid.generation import sample_frame_joint
from molvid.runtime import atomic_write_json, configure_device
from molvid.training.batches import prepare_frame_joint_batch


FIELDS = (
    "aligned_rmsd",
    "bond_rmse",
    "clash_rate",
    "contact_f1",
    "contact_occupancy_mae",
    "rmsf_prediction",
    "rmsf_target",
    "rmsf_absolute_error",
    "rmsf_correlation",
    "velocity_rmse",
    "dynamic_correlation",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--valid-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows-per-trajectory", type=int, default=1)
    parser.add_argument("--expected-systems", type=int, default=None)
    parser.add_argument("--expected-trajectories", type=int, default=None)
    return parser.parse_args()


def _trajectory(sample_id: str) -> str:
    return str(sample_id).rsplit("_w", 1)[0]


def _system(trajectory: str) -> str:
    return re.sub(r"_R[0-9]+$", "", trajectory)


def _first_per_trajectory(dataset: ClipMMapDataset) -> list[int]:
    result: list[int] = []
    seen: set[str] = set()
    for index, (sample_id, _start, _end) in enumerate(dataset._index):
        trajectory = _trajectory(sample_id)
        if trajectory not in seen:
            seen.add(trajectory)
            result.append(index)
    return result


def _windows_per_trajectory(dataset: ClipMMapDataset, count: int) -> list[int]:
    if count < 1:
        raise ValueError("windows-per-trajectory must be positive")
    selected: dict[str, list[int]] = defaultdict(list)
    for index, (sample_id, _start, _end) in enumerate(dataset._index):
        trajectory = _trajectory(sample_id)
        if len(selected[trajectory]) < count:
            selected[trajectory].append(index)
    return [index for trajectory in sorted(selected) for index in selected[trajectory]]


def _concise(metrics: dict) -> dict[str, object]:
    future = metrics["future"]
    rmsf = metrics["rmsf"]["future"]
    dynamic = metrics["dynamic"]["future"]
    return {
        "aligned_rmsd": float(future["aligned_rmsd"]),
        "bond_rmse": float(future["bond_rmse"]),
        "clash_rate": float(future["clash_rate"]),
        "contact_f1": float(future["contact_f1"]),
        "contact_occupancy_mae": float(future["contact_occupancy_mae"]),
        "rmsf_prediction": (
            float(rmsf["prediction"]) if rmsf.get("prediction") is not None else None
        ),
        "rmsf_target": float(rmsf["target"]) if rmsf.get("target") is not None else None,
        "rmsf_absolute_error": (
            float(rmsf["absolute_error"])
            if rmsf.get("absolute_error") is not None
            else None
        ),
        "rmsf_correlation": (
            float(rmsf["correlation"])
            if rmsf.get("correlation_available", False)
            else None
        ),
        "velocity_rmse": float(metrics["velocity_rmse"]),
        "dynamic_correlation": (
            float(dynamic["dynamic_correlation"])
            if dynamic.get("available", False)
            else None
        ),
        "availability": {
            "rmsf": {"available": rmsf.get("available"), "reason": rmsf.get("reason")},
            "rmsf_correlation": {
                "available": rmsf.get("correlation_available"),
                "reason": rmsf.get("correlation_reason"),
            },
            "dynamic": {"available": dynamic.get("available"), "reason": dynamic.get("reason")},
            "dynamic_acf": {
                "available": dynamic.get("acf_available"),
                "reason": dynamic.get("acf_reason"),
            },
        },
    }


def _average(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    counts: dict[str, int] = {}
    for field in FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = sum(values) / len(values) if values else None
        counts[field] = len(values)
    result["available_counts"] = counts
    return result


def _system_equal(rows: list[dict]) -> dict:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row["system"]].append(row["summary"])
    system_rows = {system: _average(values) for system, values in sorted(groups.items())}
    return {
        "systems": system_rows,
        "mean": _average(list(system_rows.values())),
        "aggregation": "replica_mean_then_system_equal_mean",
    }


def main() -> int:
    args = _arguments()
    device = configure_device(args.device, deterministic=True)
    loaded = load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec,
        codec_sha256=args.codec_sha256,
        device=device,
    )
    model = loaded["model"]
    dataset = ClipMMapDataset(args.valid_store)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    total_seconds = 0.0
    total_future_frames = 0
    total_atom_frames = 0
    try:
        indices = (
            _first_per_trajectory(dataset)
            if args.windows_per_trajectory == 1
            else _windows_per_trajectory(dataset, args.windows_per_trajectory)
        )
        trajectories = [_trajectory(dataset._index[index][0]) for index in indices]
        if args.expected_trajectories is not None and len(set(trajectories)) != args.expected_trajectories:
            raise RuntimeError("evaluation store does not contain the expected trajectory count")
        if args.expected_systems is not None and len({_system(value) for value in trajectories}) != args.expected_systems:
            raise RuntimeError("evaluation store does not contain the expected system count")
        for index in indices:
            batch = collate_clip_records([dataset[index]])
            trajectory = _trajectory(batch.sample_id[0])
            system = _system(trajectory)
            for history in (4, 8):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                prediction, generation = sample_frame_joint(
                    model,
                    template=batch,
                    prefix_coordinates=batch.x[:history],
                    history_frames=history,
                    steps=args.steps,
                    seed=args.seed,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                prepared = prepare_frame_joint_batch(
                    model.target_teacher,
                    batch,
                    device=device,
                    normalizer=model,
                    history_frames=history,
                )
                oracle_future = model.decoder(
                    prepared.observed_context,
                    prepared.target_future,
                    prepared.query,
                ).coordinates.float()
                oracle = torch.cat((prepared.coordinate_batch.x[:history].float(), oracle_future), dim=0)
                target = prepared.coordinate_batch.x.float()
                generated_metrics = trajectory_metrics(
                    prediction, target, prepared.coordinate_batch, history_frames=history
                )
                oracle_metrics = trajectory_metrics(
                    oracle, target, prepared.coordinate_batch, history_frames=history
                )
                query_frames = batch.frames - history
                total_seconds += elapsed
                total_future_frames += query_frames
                total_atom_frames += query_frames * batch.atom_count
                key = f"{trajectory}_H{history}"
                arrays[f"{key}_generated"] = prediction.detach().cpu().numpy()
                arrays[f"{key}_target"] = target.detach().cpu().numpy()
                arrays[f"{key}_oracle"] = oracle.detach().cpu().numpy()
                rows.append({
                    "sample_id": batch.sample_id[0],
                    "trajectory": trajectory,
                    "system": system,
                    "history_frames": history,
                    "query_frames": query_frames,
                    "atoms": batch.atom_count,
                    "seconds": elapsed,
                    "summary": _concise(generated_metrics),
                    "oracle_summary": _concise(oracle_metrics),
                    "metrics": generated_metrics,
                    "oracle_metrics": oracle_metrics,
                    "generation": generation,
                })
        summaries = {}
        for history in (4, 8):
            selected = [row for row in rows if row["history_frames"] == history]
            generated = _system_equal(selected)
            oracle_rows = [
                {**row, "summary": row["oracle_summary"]} for row in selected
            ]
            summaries[f"H{history}"] = {
                "generated": generated,
                "clean_latent_decoder_oracle": _system_equal(oracle_rows),
            }
        output = {
            "schema_version": "molvid.frame_joint.tiny_eval.v2",
            "checkpoint_sha256": args.checkpoint_sha256,
            "codec_sha256": args.codec_sha256,
            "protocol": {
                "split": "original_tiny_valid",
                "systems": len({_system(value) for value in trajectories}),
                "trajectories": len(set(trajectories)),
                "windows_per_trajectory": args.windows_per_trajectory,
                "window_selection": "first_valid_windows_in_manifest_order",
                "noise_seeds": [args.seed],
                "histories": [4, 8],
                "euler_steps": args.steps,
                "best_of_n": False,
            },
            "throughput": {
                "total_generation_seconds": total_seconds,
                "future_frames_per_second": total_future_frames / total_seconds,
                "future_atom_frames_per_second": total_atom_frames / total_seconds,
            },
            "summary": summaries,
            "rows": rows,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output / "tiny_evaluation.json", output)
        np.savez_compressed(args.output / "tiny_coordinates.npz", **arrays)
        print(json.dumps({
            "output": str(args.output),
            "throughput": output["throughput"],
            "summary": summaries,
        }, sort_keys=True))
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
