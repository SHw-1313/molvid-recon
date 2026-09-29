"""Evaluate learned-memory arms on raw ATLAS valid trajectories.

The protocol fixes two anchors per physical trajectory, crosses history span,
history sampling interval and query lag independently, and uses the same
anchors for both arms.  Generated paths receive observed raw history, the two
geometry anchors and query times only.  Future coordinates are encoded only
for scoring and the clean-latent oracle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.learned_memory_formal import RawMemoryDataset
from molvid.evaluation.geometry import (
    angle_statistics,
    bond_errors,
    clash_statistics,
    contact_scores,
    coordinate_errors,
)
from molvid.evaluation.physical import sampled_clash_metrics
from molvid.evaluation.temporal import (
    TEMPORAL_METRIC_SCHEMA,
    system_mean_rmsf_summary,
    temporal_metrics_v3,
)
from molvid.flow.sampling import euler_sample_frames
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.runtime import atomic_write_json, canonical_hash, configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch
from tools.recompute_frame_gm_metrics_v3 import ROW_SCHEMA, SCALARS, _flat_metrics, _jsonable


SCALARS.update({
    "clashes_per_1000_atoms": ("physical", "clashes_per_1000_atoms"),
    "affected_atom_fraction": ("physical", "affected_atom_fraction"),
    "vdw_severity_A2_per_atom": ("physical", "vdw_severity_A2_per_atom"),
})


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--limit-prefixes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _selected_anchors(dataset: RawMemoryDataset) -> list[dict[str, Any]]:
    """Choose fixed one-third/two-third eligible anchors per trajectory."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in dataset.items:
        grouped[item.trajectory_id].append(item)
    selected: list[dict[str, Any]] = []
    for trajectory, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item.anchor_raw_index)
        positions = (len(ordered) // 3, (2 * len(ordered)) // 3)
        if positions[0] == positions[1]:
            raise ValueError(f"trajectory {trajectory} lacks two distinct validation anchors")
        for slot, position in enumerate(positions):
            item = ordered[position]
            selected.append({
                "dataset_index": item.index,
                "trajectory_id": trajectory,
                "system": item.system,
                "replica": item.replica,
                "anchor_slot": slot,
                "window_index": item.window_index,
                "anchor_raw_index": item.anchor_raw_index,
                "sample_id": item.sample_id,
            })
    if len(grouped) != 24 or len(selected) != 48:
        raise ValueError("validation selection must contain 8 systems x3 replicas x2 anchors")
    return selected


def _conditions(
    anchors: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    spans = (800,) if arm == "M-short" else (800, 3200, 12800)
    values: list[dict[str, Any]] = []
    for anchor in anchors:
        for span in spans:
            for observed_dt in (100, 200, 400):
                for query_lag in (100, 200, 300, 400):
                    values.append({
                        **anchor,
                        "history_span_ps": span,
                        "observed_dt_ps": observed_dt,
                        "query_lag_ps": query_lag,
                    })
    return values


def _scoring_batch(anchors, future, device: torch.device):
    time_ps = torch.cat((anchors.time_ps, future.time_ps), dim=1)
    frame_mask = torch.cat((anchors.frame_mask, future.frame_mask), dim=1)
    return replace(
        anchors,
        x=torch.cat((anchors.x, future.x), dim=0),
        bpos=torch.cat((anchors.bpos, future.bpos), dim=0),
        frame_mask=frame_mask,
        time_ps=time_ps,
        delta_time_ps=torch.diff(time_ps, dim=1),
        time_bucket_id=tuple(
            f"learned_memory_valid_{int(value)}ps" for value in future.time_ps[:, 0]
        ),
    ).to(device)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dimensions = ("history_span_ps", "observed_dt_ps", "query_lag_ps")
    by_anchor: dict[tuple[Any, ...], list[dict[str, float | None]]] = defaultdict(list)
    for source in rows:
        meta = source["meta"]
        key = (
            source["path"], *(meta[name] for name in dimensions),
            meta["system"], meta["replica"], meta["anchor_id"],
        )
        by_anchor[key].append(_flat_metrics(source))
    anchor_rows: list[dict[str, Any]] = []
    for key, values in by_anchor.items():
        row = {
            "path": key[0],
            **{name: key[index + 1] for index, name in enumerate(dimensions)},
            "system": key[4], "replica": key[5], "anchor_id": key[6],
            "draw_count": len(values),
        }
        row.update({name: _mean(value[name] for value in values) for name in SCALARS})
        anchor_rows.append(row)
    by_replica: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for source in anchor_rows:
        key = (
            source["path"], *(source[name] for name in dimensions),
            source["system"], source["replica"],
        )
        by_replica[key].append(source)
    replica_rows: list[dict[str, Any]] = []
    for key, values in by_replica.items():
        row = {
            "path": key[0],
            **{name: key[index + 1] for index, name in enumerate(dimensions)},
            "system": key[4], "replica": key[5], "anchor_count": len(values),
        }
        row.update({name: _mean(value[name] for value in values) for name in SCALARS})
        replica_rows.append(row)
    by_system: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for source in replica_rows:
        key = (
            source["path"], *(source[name] for name in dimensions), source["system"],
        )
        by_system[key].append(source)
    system_rows: list[dict[str, Any]] = []
    for key, values in by_system.items():
        row = {
            "family": "learned_memory",
            "path": key[0],
            **{name: key[index + 1] for index, name in enumerate(dimensions)},
            "system": key[4], "replica_count": len(values),
        }
        row.update({name: _mean(value[name] for value in values) for name in SCALARS})
        system_rows.append(row)
    by_bucket: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in system_rows:
        by_bucket[(row["path"], *(row[name] for name in dimensions))].append(row)
    buckets: dict[str, Any] = {}
    for key, values in sorted(by_bucket.items()):
        name = f"{key[0]}|span={key[1]}ps|dt={key[2]}ps|lag={key[3]}ps"
        summary = {metric: _mean(row[metric] for row in values) for metric in SCALARS}
        summary["availability"] = {
            metric: sum(row[metric] is not None for row in values) for metric in SCALARS
        }
        summary["system_count"] = len(values)
        summary["systems"] = sorted(str(row["system"]) for row in values)
        summary["system_mean_rmsf_distribution"] = system_mean_rmsf_summary([
            {
                "system_mean_prediction_A": row["system_mean_rmsf_prediction_A"],
                "system_mean_target_A": row["system_mean_rmsf_target_A"],
            }
            for row in values
        ])
        buckets[name] = summary
    return sorted(system_rows, key=lambda row: (
        str(row["path"]), int(row["history_span_ps"]), int(row["observed_dt_ps"]),
        int(row["query_lag_ps"]), str(row["system"]),
    )), buckets


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "family", "path", "history_span_ps", "observed_dt_ps", "query_lag_ps",
        "system", "replica_count", *SCALARS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fields} for row in rows)


def _prepare(model, device, raw_history, anchors, future):
    return prepare_frame_joint_batch(
        model.target_teacher,
        future,
        device=device,
        normalizer=model,
        history_frames=2,
        raw_history=raw_history,
        geometry_anchors=anchors,
    )


def main() -> int:
    args = _args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    arm = str(raw_config["arm"])
    if arm not in {"M-short", "M-long"}:
        raise ValueError("learned-memory evaluation requires M-short or M-long")
    if args.steps != 16 or tuple(args.seeds) != (0, 1, 2):
        raise ValueError("formal evaluation fixes Euler16 and generation seeds 0,1,2")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing existing output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    profiles_root = args.output / "per_residue_profiles"
    coordinates_root = args.output / "coordinates"
    profiles_root.mkdir(exist_ok=True)
    coordinates_root.mkdir(exist_ok=True)
    device = configure_device("cuda", deterministic=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("B evaluation requires exactly one visible CUDA device")
    started = time.perf_counter()
    codec = raw_config["codec"]
    loaded = load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=codec["checkpoint"],
        codec_sha256=codec["sha256"],
        device=device,
    )
    model = loaded["model"].eval()
    if model.history_backend != "learned_memory":
        raise ValueError("checkpoint does not contain learned memory")
    dataset = RawMemoryDataset(
        manifest_root=Path(raw_config["data"]["manifest_root"]),
        raw_root=Path(raw_config["data"]["raw_root"]),
        offset_root=Path(raw_config["data"]["xtc_offset_root"]),
        split="valid",
    )
    anchors = _selected_anchors(dataset)
    conditions = _conditions(anchors, arm)
    if args.limit_prefixes is not None:
        conditions = conditions[: int(args.limit_prefixes)]
    physical_config = PhysicalAuxiliaryConfig.resolve(
        raw_config["sampled_auxiliary"]["physical"]
    )
    protocol = {
        "schema": "molvid.learned_memory.evaluation.v1",
        "arm": arm,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "validation_data": dataset.identity,
        "anchor_selection": anchors,
        "anchor_selection_sha256": canonical_hash(anchors),
        "conditions_sha256": canonical_hash(conditions),
        "prefixes": len(conditions),
        "expected_rows": len(conditions) * 5,
        "history_spans_ps": [800] if arm == "M-short" else [800, 3200, 12800],
        "observed_dt_ps": [100, 200, 400],
        "query_lag_ps": [100, 200, 300, 400],
        "validation_only_lag_ps": [300],
        "teacher_anchor_times_ps": [-800, 0],
        "future_Q": 12,
        "euler_steps": 16,
        "seeds": [0, 1, 2],
        "best_of_N": False,
        "aggregation": "draw -> anchor -> replica -> system equal",
        "alignment": "prediction and MD each to the same last-observed reference",
        "test_opened": False,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("resume evaluation identity differs")
    atomic_write_json(protocol_path, protocol)
    rows_path = args.output / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    completed = {str(row["row_key"]) for row in rows}
    checks_path = args.output / "checks.json"
    if not checks_path.exists():
        check = conditions[0]
        check_raw, check_anchors, check_future = dataset.load_inputs(
            [int(check["dataset_index"])],
            history_span_ps=int(check["history_span_ps"]),
            observed_dt_ps=int(check["observed_dt_ps"]),
            query_lag_ps=int(check["query_lag_ps"]),
        )
        atom_phase = torch.arange(check_future.atom_count, dtype=check_future.x.dtype)
        displacement = torch.sin(atom_phase)[None, :, None] * torch.tensor(
            [7.0, -3.0, 2.0], dtype=check_future.x.dtype
        )[None, None]
        altered_future = replace(check_future, x=check_future.x + displacement)
        with torch.no_grad():
            first = _prepare(model, device, check_raw, check_anchors, check_future)
            altered = _prepare(model, device, check_raw, check_anchors, altered_future)
            first_condition = model.prepare_condition(first.observed_context)
            altered_condition = model.prepare_condition(altered.observed_context)
            first_latent, _ = euler_sample_frames(
                model, context=first.observed_context, query=first.query,
                steps=args.steps, seed=0, prepared_condition=first_condition,
            )
            altered_latent, _ = euler_sample_frames(
                model, context=altered.observed_context, query=altered.query,
                steps=args.steps, seed=0, prepared_condition=altered_condition,
            )
            first_prediction = model.decoder(
                first.observed_context, first_latent, first.query
            ).coordinates
            altered_prediction = model.decoder(
                altered.observed_context, altered_latent, altered.query
            ).coordinates
            generation_difference = float(
                (first_prediction - altered_prediction).abs().max().cpu()
            )
            target_difference = float(
                (first.target_future.h - altered.target_future.h).abs().max().cpu()
            )
            observed_equal = bool(
                torch.equal(first.observed_context.latent.h, altered.observed_context.latent.h)
                and torch.equal(first.observed_context.latent.v, altered.observed_context.latent.v)
                and torch.equal(first.query.time_ps, altered.query.time_ps)
                and torch.equal(first.query.frame_mask, altered.query.frame_mask)
            )
        leakage = {
            "schema": "molvid.learned_memory.future_leakage_check.v1",
            "prefix": check,
            "future_target_latent_max_abs_difference": target_difference,
            "observed_and_query_equal": observed_equal,
            "generated_max_abs_difference": generation_difference,
            "passed": bool(
                target_difference > 0.0 and observed_equal and generation_difference == 0.0
            ),
            "euler_steps": args.steps,
            "seed": 0,
        }
        if not leakage["passed"]:
            raise RuntimeError(f"future leakage check failed: {leakage}")
        atomic_write_json(checks_path, leakage)
    teacher_calls: list[int] = []
    memory_calls: list[int] = []
    teacher_hook = model.target_teacher.register_forward_pre_hook(
        lambda _module, values: teacher_calls.append(int(values[0].frames))
    )
    memory_hook = model.history_encoder.register_forward_pre_hook(
        lambda _module, _values: memory_calls.append(1)
    )
    peak = 0
    try:
        for position, condition in enumerate(conditions):
            prefix = (
                f"{condition['sample_id']}|slot={condition['anchor_slot']}|"
                f"span={condition['history_span_ps']}|dt={condition['observed_dt_ps']}|"
                f"lag={condition['query_lag_ps']}"
            )
            expected_keys = {
                f"{prefix}|{path}|seed={seed}"
                for path, seed in (
                    ("persistence", None), ("oracle", None),
                    ("generated", 0), ("generated", 1), ("generated", 2),
                )
            }
            if expected_keys.issubset(completed):
                continue
            io_started = time.perf_counter()
            raw_history, geometry_anchors, future = dataset.load_inputs(
                [int(condition["dataset_index"])],
                history_span_ps=int(condition["history_span_ps"]),
                observed_dt_ps=int(condition["observed_dt_ps"]),
                query_lag_ps=int(condition["query_lag_ps"]),
            )
            io_seconds = time.perf_counter() - io_started
            before_teacher = len(teacher_calls)
            with torch.no_grad():
                prepared = _prepare(
                    model, device, raw_history, geometry_anchors, future
                )
                scoring = _scoring_batch(geometry_anchors, future, device)
                target = scoring.x.float()
                future_frames = list(range(2, scoring.frames))
                before_memory = len(memory_calls)
                shared_condition = model.prepare_condition(prepared.observed_context)
                if len(memory_calls) - before_memory != 1:
                    raise RuntimeError("learned memory was not prepared exactly once per prefix")
                generated: dict[int, torch.Tensor] = {}
                for seed in args.seeds:
                    latent, _metadata = euler_sample_frames(
                        model,
                        context=prepared.observed_context,
                        query=prepared.query,
                        steps=args.steps,
                        seed=int(seed),
                        prepared_condition=shared_condition,
                    )
                    generated[int(seed)] = model.decoder(
                        prepared.observed_context, latent, prepared.query
                    ).coordinates.float()
                paths = [
                    ("persistence", None, prepared.observed_context.coordinates[-1:].expand_as(prepared.target_coordinates).float()),
                    ("oracle", None, model.decoder(
                        prepared.observed_context, prepared.target_future, prepared.query
                    ).coordinates.float()),
                    *(("generated", seed, generated[int(seed)]) for seed in args.seeds),
                ]
                if teacher_calls[before_teacher:] != [2, 12]:
                    raise RuntimeError(f"evaluation teacher calls changed: {teacher_calls[before_teacher:]}")
                representative = (
                    condition["replica"] == "R1"
                    and int(condition["anchor_slot"]) == 0
                    and int(condition["observed_dt_ps"]) == 100
                    and int(condition["query_lag_ps"]) == 400
                    and int(condition["history_span_ps"])
                    == max(protocol["history_spans_ps"])
                )
                for path, seed, future_prediction in paths:
                    row_key = f"{prefix}|{path}|seed={seed}"
                    if row_key in completed:
                        continue
                    prediction = torch.cat((
                        prepared.observed_context.coordinates.float(), future_prediction
                    ))
                    temporal = temporal_metrics_v3(
                        prediction, target, scoring, history_frames=2
                    )
                    geometry = {
                        **coordinate_errors(prediction, target, scoring, future_frames),
                        **bond_errors(prediction, target, scoring, future_frames),
                        **contact_scores(prediction, target, scoring, future_frames),
                        **clash_statistics(prediction, scoring, future_frames),
                        "angles": angle_statistics(prediction, target, scoring, future_frames),
                        "coordinate_unit": "angstrom",
                        "angle_unit": "radian",
                    }
                    physical = sampled_clash_metrics(future_prediction, prepared, physical_config)
                    artifact = hashlib.sha256(row_key.encode()).hexdigest() + ".npz"
                    profile_file = coordinate_file = None
                    if representative:
                        np.savez_compressed(
                            profiles_root / artifact,
                            **{name: value.detach().cpu().numpy() for name, value in temporal.profiles.items()},
                        )
                        np.savez_compressed(
                            coordinates_root / artifact,
                            prediction=prediction.detach().cpu().numpy(),
                            target=target.detach().cpu().numpy(),
                            physical_time_ps=scoring.time_ps.detach().cpu().numpy(),
                        )
                        profile_file = artifact
                        coordinate_file = artifact
                    row = {
                        "schema": ROW_SCHEMA,
                        "metric_schema": TEMPORAL_METRIC_SCHEMA,
                        "metric_family": {
                            "schema": "molvid.learned_memory.metric_family.v1",
                            "family": "learned_memory",
                            "main_rmsf_available": True,
                            "reason": "Q12 raw validation sequence",
                        },
                        "row_key": row_key,
                        "path": path,
                        "clock": "true",
                        "seed": seed,
                        "meta": {
                            "sample_id": condition["sample_id"],
                            "system": condition["system"],
                            "replica": condition["replica"],
                            "anchor_id": f"slot{condition['anchor_slot']}_w{condition['window_index']:06d}",
                            "anchor_slot": condition["anchor_slot"],
                            "anchor_raw_frame_index": condition["anchor_raw_index"],
                            "history_span_ps": condition["history_span_ps"],
                            "observed_dt_ps": condition["observed_dt_ps"],
                            "query_lag_ps": condition["query_lag_ps"],
                            "history_frames": int(raw_history.coordinates.shape[0]),
                            "future_frames": 12,
                        },
                        "raw_io_seconds": io_seconds,
                        "memory_preparations": 1,
                        "teacher_frame_calls": [2, 12],
                        "profile_file": profile_file,
                        "coordinate_file": coordinate_file,
                        "metrics": _jsonable({
                            **temporal.metrics, "geometry": geometry, "physical": physical,
                        }),
                    }
                    with rows_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    completed.add(row_key)
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
            if (position + 1) % 8 == 0:
                print(json.dumps({
                    "prefixes": position + 1,
                    "total_prefixes": len(conditions),
                    "rows": len(rows),
                }, sort_keys=True), flush=True)
        expected = len(conditions) * 5
        selected_keys = {
            f"{condition['sample_id']}|slot={condition['anchor_slot']}|"
            f"span={condition['history_span_ps']}|dt={condition['observed_dt_ps']}|"
            f"lag={condition['query_lag_ps']}|{path}|seed={seed}"
            for condition in conditions
            for path, seed in (
                ("persistence", None), ("oracle", None),
                ("generated", 0), ("generated", 1), ("generated", 2),
            )
        }
        selected_rows = [row for row in rows if row["row_key"] in selected_keys]
        if len(selected_rows) != expected:
            raise RuntimeError(f"evaluation rows incomplete: {len(selected_rows)} != {expected}")
        per_system, buckets = _aggregate(selected_rows)
        _write_csv(args.output / "per_system.csv", per_system)
        elapsed = time.perf_counter() - started
        summary = {
            "schema": "molvid.learned_memory.metric_summary.v1",
            "arm": arm,
            "checkpoint_sha256": args.checkpoint_sha256,
            "protocol_sha256": sha256_file(protocol_path),
            "row_count": len(selected_rows),
            "prefix_count": len(conditions),
            "bucket_summary": buckets,
            "aggregation": protocol["aggregation"],
            "gpu_hours_this_invocation": elapsed / 3600.0,
            "peak_allocated_bytes": peak,
            "teacher_observed_frames_per_prefix": 2,
            "teacher_target_frames_per_prefix": 12,
            "memory_preparations_per_prefix": 1,
            "representative_profile_files": len(list(profiles_root.glob("*.npz"))),
            "representative_coordinate_files": len(list(coordinates_root.glob("*.npz"))),
            "rows_sha256": sha256_file(rows_path),
            "per_system_sha256": sha256_file(args.output / "per_system.csv"),
            "test_opened": False,
        }
        atomic_write_json(args.output / "summary.json", summary)
        print(json.dumps({
            "complete": True, "arm": arm, "rows": len(selected_rows),
            "gpu_hours": summary["gpu_hours_this_invocation"],
        }, sort_keys=True), flush=True)
    finally:
        teacher_hook.remove()
        memory_hook.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
