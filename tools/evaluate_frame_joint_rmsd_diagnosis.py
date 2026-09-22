"""CUDA RMSD diagnosis for persistence, clean-latent oracle, and generated paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.geometry import bond_errors, coordinate_errors
from molvid.evaluation.runner import trajectory_metrics
from molvid.generation import sample_frame_joint
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch


SCALAR_FIELDS = (
    "aligned_rmsd",
    "drmsd",
    "bond_rmse",
    "contact_f1",
    "contact_occupancy_mae",
    "rmsf_prediction",
    "rmsf_target",
    "rmsf_absolute_error",
    "rmsf_correlation",
    "velocity_rmse",
    "dynamic_correlation",
)
PATHS = ("persistence", "clean_latent_decoder_oracle", "generated")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--valid-store", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append")
    return parser.parse_args()


def _trajectory(sample_id: str) -> str:
    return str(sample_id).rsplit("_w", 1)[0]


def _system(trajectory: str) -> str:
    return re.sub(r"_R[0-9]+$", "", trajectory)


def _replica(trajectory: str) -> str:
    match = re.search(r"(_R[0-9]+)$", trajectory)
    if match is None:
        raise ValueError(f"sample id has no replica suffix: {trajectory}")
    return match.group(1).removeprefix("_")


def _window(sample_id: str) -> int:
    match = re.search(r"_w([0-9]+)$", sample_id)
    if match is None:
        raise ValueError(f"sample id has no window suffix: {sample_id}")
    return int(match.group(1))


def _selected_ids(reference: Path, explicit: list[str] | None, limit: int | None) -> list[str]:
    if explicit:
        selected = list(dict.fromkeys(str(value) for value in explicit))
    else:
        payload = json.loads(reference.read_text(encoding="utf-8"))
        selected = list(dict.fromkeys(str(row["sample_id"]) for row in payload["rows"]))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no evaluation windows selected")
    return selected


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _summary(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    future = metrics["future"]
    rmsf = metrics["rmsf"]["future"]
    dynamic = metrics["dynamic"]["future"]
    return {
        "aligned_rmsd": _scalar(future["aligned_rmsd"]),
        "drmsd": _scalar(future["drmsd"]),
        "bond_rmse": _scalar(future["bond_rmse"]),
        "contact_f1": _scalar(future["contact_f1"]),
        "contact_occupancy_mae": _scalar(future["contact_occupancy_mae"]),
        "rmsf_prediction": _scalar(rmsf.get("prediction")),
        "rmsf_target": _scalar(rmsf.get("target")),
        "rmsf_absolute_error": _scalar(rmsf.get("absolute_error")),
        "rmsf_correlation": (
            _scalar(rmsf.get("correlation"))
            if rmsf.get("correlation_available", False)
            else None
        ),
        "velocity_rmse": _scalar(metrics.get("velocity_rmse")),
        "dynamic_correlation": (
            _scalar(dynamic.get("dynamic_correlation"))
            if dynamic.get("available", False)
            else None
        ),
    }


def _per_frame(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Any,
    history: int,
) -> list[dict[str, float | int | None]]:
    time_ps = torch.as_tensor(batch.time_ps, device=prediction.device, dtype=torch.float32)
    if time_ps.ndim != 2 or time_ps.shape[0] != 1:
        raise ValueError("diagnosis expects one packed sample per window")
    baseline = time_ps[0, history - 1]
    result = []
    for frame in range(history, int(target.shape[0])):
        coordinate = coordinate_errors(prediction, target, batch, [frame])
        bond = bond_errors(prediction, target, batch, [frame])
        result.append({
            "frame_index": frame,
            "time_ps": float(time_ps[0, frame].detach().cpu()),
            "distance_from_last_observation_ps": float((time_ps[0, frame] - baseline).detach().cpu()),
            **{key: float(value) for key, value in coordinate.items()},
            **{key: float(value) for key, value in bond.items()},
        })
    return result


def _average_dict(values: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    values = list(values)
    result: dict[str, float | None] = {}
    for field in SCALAR_FIELDS:
        numeric = [float(item[field]) for item in values if item.get(field) is not None]
        result[field] = sum(numeric) / len(numeric) if numeric else None
    result["available_counts"] = {
        field: sum(item.get(field) is not None for item in values)
        for field in SCALAR_FIELDS
    }
    return result


def _hierarchical_average(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_replica: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_replica[(str(entry["system"]), str(entry["replica"]))].append(entry["metrics"])
    replica_means = {
        key: _average_dict(values) for key, values in sorted(by_replica.items())
    }
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (system, _replica), value in replica_means.items():
        by_system[system].append(value)
    system_means = {
        system: _average_dict(values) for system, values in sorted(by_system.items())
    }
    return {
        "systems": system_means,
        "mean": _average_dict(list(system_means.values())),
        "aggregation": "window_mean_then_replica_mean_then_system_equal_mean",
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    per_frame_summaries: dict[str, Any] = {}
    for history in (4, 8):
        for path in PATHS:
            selected = [row for row in rows if row["history_frames"] == history and row["path"] == path]
            key = f"H{history}"
            summaries.setdefault(key, {})[path] = _hierarchical_average([
                {"system": row["system"], "replica": row["replica"], "metrics": row["summary"]}
                for row in selected
            ])
            by_frame: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                for value in row["per_frame"]:
                    frame_key = (int(value["frame_index"]), float(value["distance_from_last_observation_ps"]))
                    by_frame[frame_key].append({
                        "system": row["system"],
                        "replica": row["replica"],
                        "metrics": value,
                    })
            per_frame_summaries.setdefault(key, {})[path] = {
                "frames": [
                    {
                        "frame_index": frame,
                        "distance_from_last_observation_ps": distance,
                        "summary": _hierarchical_average(values),
                    }
                    for (frame, distance), values in sorted(by_frame.items())
                ]
            }
    return {"summary": summaries, "per_frame_summary": per_frame_summaries}


def main() -> int:
    args = _arguments()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    device = configure_device(args.device, deterministic=True)
    selected = _selected_ids(args.reference_evaluation, args.sample_id, args.limit)
    loaded = load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec,
        codec_sha256=args.codec_sha256,
        device=device,
    )
    model = loaded["model"]
    model.eval()
    dataset = ClipMMapDataset(args.valid_store)
    index_by_id = {str(row[0]): index for index, row in enumerate(dataset._index)}
    missing = [sample_id for sample_id in selected if sample_id not in index_by_id]
    if missing:
        raise RuntimeError(f"selected windows missing from valid store: {missing[:3]}")
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    try:
        for window_number, sample_id in enumerate(selected):
            index = index_by_id[sample_id]
            record = dataset[index]
            batch = collate_clip_records([record])
            trajectory = _trajectory(sample_id)
            system = _system(trajectory)
            replica = _replica(trajectory)
            for history in (4, 8):
                prepared = prepare_frame_joint_batch(
                    model.target_teacher,
                    batch,
                    device=device,
                    normalizer=model,
                    history_frames=history,
                )
                target = prepared.coordinate_batch.x.float()
                cb = prepared.coordinate_batch
                time_ps = torch.as_tensor(cb.time_ps, device=device, dtype=torch.float32)
                mask_info = {
                    "coordinate_unit": str(record.get("coordinate_unit", "unknown")),
                    "time_bucket_id": str(record.get("time_bucket_id", "unknown")),
                    "native_delta_time_ps": float(np.asarray(record["delta_time_ps"])[0]) if len(record["delta_time_ps"]) else None,
                    "loss_mask_atoms": int(cb.loss_mask.bool().sum().item()),
                    "align_mask_atoms": int(cb.align_mask.bool().sum().item()),
                    "align_and_loss_atoms": int((cb.loss_mask.bool() & cb.align_mask.bool()).sum().item()),
                    "frame_valid_count": int(cb.frame_mask.bool().sum().item()),
                    "total_atoms": int(cb.atom_count),
                    "total_frames": int(cb.frames),
                }
                q = int(batch.frames - history)
                persistence = torch.cat((target[:history], target[history - 1:history].expand(q, -1, -1)), dim=0)
                oracle_future = model.decoder(
                    prepared.observed_context,
                    prepared.target_future,
                    prepared.query,
                ).coordinates.float()
                oracle = torch.cat((target[:history], oracle_future), dim=0)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                generated, generation = sample_frame_joint(
                    model,
                    template=batch,
                    prefix_coordinates=batch.x[:history],
                    history_frames=history,
                    steps=args.steps,
                    seed=args.seed,
                )
                torch.cuda.synchronize(device)
                generation_seconds = time.perf_counter() - started
                predictions = {
                    "persistence": persistence,
                    "clean_latent_decoder_oracle": oracle,
                    "generated": generated,
                }
                for path in PATHS:
                    prediction = predictions[path]
                    metrics = trajectory_metrics(prediction, target, cb, history_frames=history)
                    row_key = f"{system}_{replica}_w{_window(sample_id):06d}_H{history}_{path}"
                    arrays[row_key] = prediction.detach().cpu().numpy()
                    rows.append({
                        "sample_id": sample_id,
                        "trajectory": trajectory,
                        "system": system,
                        "replica": replica,
                        "window": _window(sample_id),
                        "history_frames": history,
                        "query_frames": q,
                        "path": path,
                        "atoms": int(batch.atom_count),
                        "physical_time_ps": [float(value) for value in time_ps[0].detach().cpu()],
                        "mask_info": mask_info,
                        "summary": _summary(metrics),
                        "per_frame": _per_frame(prediction, target, cb, history),
                        "metrics": metrics,
                        "generation": generation if path == "generated" else None,
                        "generation_seconds": generation_seconds if path == "generated" else 0.0,
                    })
        aggregate = _summarize_rows(rows)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output / "coordinates.npz", **arrays)
        output = {
            "schema_version": "molvid.frame_joint.rmsd_diagnosis.v1",
            "checkpoint": {"path": str(args.checkpoint), "sha256": args.checkpoint_sha256},
            "codec": {"path": str(args.codec), "sha256": args.codec_sha256},
            "valid_store": {
                "path": str(args.valid_store),
                "index_sha256": sha256_file(args.valid_store / "index.txt"),
            },
            "reference_evaluation": {
                "path": str(args.reference_evaluation),
                "sha256": sha256_file(args.reference_evaluation),
            },
            "protocol": {
                "selected_windows": selected,
                "systems": len({row["system"] for row in rows}),
                "trajectories": len({row["trajectory"] for row in rows}),
                "histories": [4, 8],
                "euler_steps": args.steps,
                "seed": args.seed,
                "paths": list(PATHS),
                "units": "angstrom",
                "physical_time_axis": "distance_from_last_observation_frame_ps",
                "aggregation": "window_mean_then_replica_mean_then_system_equal_mean",
                "generated_conditioning": "observed_prefix_only; future coordinates never passed to generator",
                "oracle_conditioning": "true future latent encoded by frozen target encoder; diagnostic only",
                "persistence_conditioning": "last valid observed coordinate copied through all query frames",
            },
            "coordinate_arrays": "coordinates.npz",
            **aggregate,
            "rows": rows,
        }
        atomic_write_json(args.output / "diagnosis.json", output)
        print(json.dumps({"output": str(args.output), "rows": len(rows), **aggregate["summary"]}, sort_keys=True))
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
