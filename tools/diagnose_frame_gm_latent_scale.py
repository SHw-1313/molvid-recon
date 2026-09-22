"""Diagnose train-only Frame Joint latent motion, source noise and decoder sensitivity.

The clean future is intentionally encoded for diagnosis.  None of the rows in
this tool are generation-quality results, and no valid/test record is opened.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.flow.sampling import euler_sample_frames
from molvid.flow.source import sample_frame_source
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch


SCHEMA = "molvid.frame_gm.latent_scale_diagnosis.v1"
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
ID_PATTERN = re.compile(
    r"^(?P<system>atlas_.+)_R(?P<replica>[123])_dt_(?P<lag>100|200|400)ps_w(?P<window>\d+)$"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--train-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--systems-limit", type=int, default=8)
    parser.add_argument("--histories", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--lags", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260922)
    parser.add_argument("--sensitivity-records", type=int, default=6)
    parser.add_argument("--sensitivity-norm-per-atom-frame", type=float, default=1.0)
    return parser.parse_args()


def _parse(sample_id: str) -> dict[str, Any]:
    match = ID_PATTERN.match(str(sample_id))
    if match is None:
        raise ValueError(f"unexpected train sample id {sample_id!r}")
    return {
        "system": match.group("system"),
        "replica": f"R{match.group('replica')}",
        "lag_ps": int(match.group("lag")),
        "window": int(match.group("window")),
    }


def _select(dataset: ClipMMapDataset, *, lags: Sequence[int], systems_limit: int) -> list[dict[str, Any]]:
    wanted_lags = {int(value) for value in lags}
    parsed: list[tuple[str, int, str, int, dict[str, Any]]] = []
    for index, (sample_id, _start, _stop) in enumerate(dataset._index):
        meta = _parse(str(sample_id))
        if meta["lag_ps"] in wanted_lags:
            parsed.append((meta["system"], meta["lag_ps"], str(sample_id), index, meta))
    systems = sorted({value[0] for value in parsed})[: int(systems_limit)]
    selected: list[dict[str, Any]] = []
    for system in systems:
        for lag in sorted(wanted_lags):
            candidates = [value for value in parsed if value[0] == system and value[1] == lag]
            if not candidates:
                raise RuntimeError(f"train store lacks {system} at {lag} ps")
            # One deterministic physical clip per system/lag is sufficient for
            # this bounded scale diagnosis.  The rule does not inspect arrays.
            chosen = min(candidates, key=lambda value: value[2])
            selected.append({
                **chosen[4],
                "sample_id": chosen[2],
                "dataset_index": chosen[3],
            })
    return selected


def _values(value: Tensor, mask: Tensor) -> Tensor:
    if value.ndim == 3:
        norm = torch.linalg.vector_norm(value.float(), dim=-1)
    elif value.ndim == 4 and value.shape[2] == 3:
        norm = torch.linalg.vector_norm(value.float().flatten(start_dim=2), dim=-1)
    else:
        raise ValueError("latent value must have shape [Q,N,C] or [Q,N,3,C]")
    selected = norm[mask]
    if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
        raise RuntimeError("latent norm selection is empty or non-finite")
    return selected


def _distribution(value: Tensor, mask: Tensor) -> dict[str, Any]:
    selected = _values(value, mask)
    q = torch.quantile(
        selected,
        torch.tensor(QUANTILES, device=selected.device, dtype=selected.dtype),
    )
    return {
        "count": int(selected.numel()),
        "mean": float(selected.mean().detach().cpu()),
        "rms": float(selected.square().mean().sqrt().detach().cpu()),
        "quantiles": {
            f"q{int(round(level * 100)):02d}": float(item.detach().cpu())
            for level, item in zip(QUANTILES, q)
        },
    }


def _difference(first: Any, second: Any) -> tuple[Tensor, Tensor]:
    return first.h.float() - second.h.float(), first.v.float() - second.v.float()


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _coordinate_rmse(first: Tensor, second: Tensor, prepared: Any) -> float:
    mask = prepared.query.frame_mask.index_select(
        0, prepared.target_future.abid
    ).transpose(0, 1)
    mask = mask & prepared.coordinate_batch.loss_mask.unsqueeze(0)
    error = (first.float() - second.float()).square().sum(dim=-1)
    if not bool(mask.any()) or not bool(torch.isfinite(error[mask]).all()):
        raise RuntimeError("decoder sensitivity coordinate error is invalid")
    return float(error[mask].mean().sqrt().detach().cpu())


def _scaled_random_like(
    value: Tensor,
    mask: Tensor,
    *,
    generator: torch.Generator,
    norm_per_atom_frame: float,
) -> tuple[Tensor, float]:
    draw = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
    expanded = mask.reshape(mask.shape + (1,) * (value.ndim - 2))
    draw = draw * expanded.to(dtype=draw.dtype)
    desired = float(norm_per_atom_frame) * math.sqrt(int(mask.sum()))
    observed = torch.linalg.vector_norm(draw.float())
    if float(observed) <= 0:
        raise RuntimeError("random decoder perturbation has zero norm")
    result = draw * (desired / observed).to(dtype=draw.dtype)
    actual = float(torch.linalg.vector_norm(result.float()).detach().cpu()) / math.sqrt(int(mask.sum()))
    return result, actual


@torch.no_grad()
def _sensitivity(
    model: Any,
    prepared: Any,
    *,
    seed: int,
    norm_per_atom_frame: float,
) -> dict[str, Any]:
    target = prepared.normalized_target
    mask = target.atom_frame_mask()
    h_delta, h_actual = _scaled_random_like(
        target.h, mask, generator=_make_generator(target.h.device, seed),
        norm_per_atom_frame=norm_per_atom_frame,
    )
    v_delta, v_actual = _scaled_random_like(
        target.v, mask, generator=_make_generator(target.h.device, seed + 1),
        norm_per_atom_frame=norm_per_atom_frame,
    )
    baseline = model.decoder(
        prepared.observed_context, prepared.target_future, prepared.query
    ).coordinates
    h_future = model.inverse(target.with_features(target.h + h_delta, target.v))
    v_future = model.inverse(target.with_features(target.h, target.v + v_delta))
    h_coordinate = model.decoder(
        prepared.observed_context, h_future, prepared.query
    ).coordinates
    v_coordinate = model.decoder(
        prepared.observed_context, v_future, prepared.query
    ).coordinates
    h_rmse = _coordinate_rmse(h_coordinate, baseline, prepared)
    v_rmse = _coordinate_rmse(v_coordinate, baseline, prepared)
    return {
        "future_informed": True,
        "baseline": "clean future latent decoded with observed context",
        "perturbation_space": "normalized latent statistics space",
        "requested_l2_norm_per_valid_atom_frame": float(norm_per_atom_frame),
        "h_only": {"actual_l2_norm_per_valid_atom_frame": h_actual, "coordinate_rmse_A": h_rmse},
        "v_only": {"actual_l2_norm_per_valid_atom_frame": v_actual, "coordinate_rmse_A": v_rmse},
        "v_over_h_coordinate_sensitivity": v_rmse / h_rmse if h_rmse > 0 else None,
    }


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _flatten_distributions(row: Mapping[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for family in ("motion_delta", "unit_source_noise", "actual_generation_error"):
        for field in ("h", "v"):
            value = row[family][field]
            flat[f"{family}_{field}_mean"] = float(value["mean"])
            flat[f"{family}_{field}_rms"] = float(value["rms"])
            for quantile, item in value["quantiles"].items():
                flat[f"{family}_{field}_{quantile}"] = float(item)
    return flat


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_system: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["system"]), int(row["lag_ps"]), int(row["history_frames"]))].append(row)
    for (system, lag, history), values in sorted(groups.items()):
        flat_values = [_flatten_distributions(value) for value in values]
        result: dict[str, Any] = {
            "system": system,
            "lag_ps": lag,
            "history_frames": history,
            "record_count": len(values),
        }
        for name in flat_values[0]:
            result[name] = _mean(value[name] for value in flat_values)
        sensitivities = [value["decoder_sensitivity"] for value in values if value["decoder_sensitivity"] is not None]
        result["decoder_h_coordinate_rmse_A"] = _mean(
            value["h_only"]["coordinate_rmse_A"] for value in sensitivities
        ) if sensitivities else None
        result["decoder_v_coordinate_rmse_A"] = _mean(
            value["v_only"]["coordinate_rmse_A"] for value in sensitivities
        ) if sensitivities else None
        per_system.append(result)
    by_lag: dict[str, Any] = {}
    for lag in sorted({int(row["lag_ps"]) for row in per_system}):
        for history in sorted({int(row["history_frames"]) for row in per_system}):
            selected = [row for row in per_system if row["lag_ps"] == lag and row["history_frames"] == history]
            if not selected:
                continue
            numeric = [name for name, value in selected[0].items() if isinstance(value, (int, float)) and name not in {"lag_ps", "history_frames", "record_count"}]
            by_lag[f"{lag}ps|H{history}"] = {
                "system_count": len(selected),
                "system_equal_means": {
                    name: _mean(row[name] for row in selected if row[name] is not None)
                    for name in numeric
                },
            }
    return per_system, by_lag


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# P1 train-only latent scale diagnosis",
        "",
        "This diagnostic intentionally encodes clean train futures. It is future-informed evidence about latent scale and decoder sensitivity, not a generation-quality table.",
        "",
        "- Latent motion is `normalized_future - repeat(normalized_last_observed)`.",
        "- Unit source noise is N(0,1) per normalized coefficient with invalid frames masked.",
        "- Actual generation error compares an observed-only Euler sample with the clean normalized train future.",
        "- Decoder sensitivity uses equal total normalized L2 norm per valid atom-frame for h-only and v-only perturbations.",
        "- Aggregation is record mean within system/lag/history, then system-equal mean. No valid or test record was opened.",
        "",
        "Detailed grouped values are in `metrics.json`; system rows are in `per_system.csv`.",
        "",
        f"Analyzed {metrics['row_count']} rows from {metrics['system_count']} train systems.",
    ]
    return "\n".join(lines) + "\n"


@torch.no_grad()
def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output {args.output}")
    if args.systems_limit < 1 or args.sensitivity_records < 0 or args.steps < 1:
        raise ValueError("systems-limit/steps must be positive and sensitivity-records non-negative")
    device = configure_device(args.device, deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("latent scale diagnosis requires CUDA")
    loaded = load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec,
        codec_sha256=args.codec_sha256,
        device=device,
    )
    model = loaded["model"]
    model.eval()
    dataset = ClipMMapDataset(args.train_store)
    selected = _select(dataset, lags=args.lags, systems_limit=args.systems_limit)
    args.output.mkdir(parents=True)
    rows_path = args.output / "rows.jsonl"
    rows: list[dict[str, Any]] = []
    sensitivity_used = 0
    try:
        for record_position, item in enumerate(selected):
            record = dataset[int(item["dataset_index"])]
            if str(record.get("split")) != "train":
                raise RuntimeError(f"latent diagnosis selected non-train record {item['sample_id']}")
            batch = collate_clip_records([record])
            for history in sorted(set(int(value) for value in args.histories)):
                prepared = prepare_frame_joint_batch(
                    model.target_teacher,
                    batch,
                    device=device,
                    normalizer=model,
                    history_frames=history,
                )
                mask = prepared.normalized_target.atom_frame_mask()
                delta_h, delta_v = _difference(prepared.normalized_target, prepared.source_center)
                row_seed = int(args.seed + record_position * 1009 + history * 17)
                _source, noise = sample_frame_source(
                    prepared.source_center,
                    generator=_make_generator(device, row_seed),
                )
                generated, generation = euler_sample_frames(
                    model,
                    context=prepared.observed_context,
                    query=prepared.query,
                    steps=args.steps,
                    seed=row_seed,
                )
                generated_normalized = model.normalize(generated)
                error_h, error_v = _difference(generated_normalized, prepared.normalized_target)
                sensitivity = None
                if history == min(args.histories) and sensitivity_used < args.sensitivity_records:
                    sensitivity = _sensitivity(
                        model,
                        prepared,
                        seed=row_seed + 500_000,
                        norm_per_atom_frame=args.sensitivity_norm_per_atom_frame,
                    )
                    sensitivity_used += 1
                row = {
                    "schema": SCHEMA,
                    "sample_id": item["sample_id"],
                    "system": item["system"],
                    "replica": item["replica"],
                    "lag_ps": item["lag_ps"],
                    "window": item["window"],
                    "history_frames": history,
                    "query_valid_frames": int(prepared.query.frame_mask.sum()),
                    "atom_count": int(prepared.normalized_target.num_atoms),
                    "valid_atom_frames": int(mask.sum()),
                    "space": "normalized target statistics space",
                    "future_informed_diagnostic": True,
                    "motion_delta": {
                        "definition": "clean_future - repeated_last_observed_center",
                        "h": _distribution(delta_h, mask),
                        "v": _distribution(delta_v, mask),
                    },
                    "unit_source_noise": {
                        "definition": "N(0,1) per normalized coefficient; same seed as Euler source",
                        "h": _distribution(noise.h, mask),
                        "v": _distribution(noise.v, mask),
                    },
                    "actual_generation_error": {
                        "definition": "observed-only Euler sample - clean future",
                        "h": _distribution(error_h, mask),
                        "v": _distribution(error_v, mask),
                        "generation": generation,
                    },
                    "decoder_sensitivity": sensitivity,
                }
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"records": record_position + 1, "total": len(selected), "rows": len(rows)}), flush=True)
    finally:
        dataset.close()
    per_system, by_lag = _aggregate(rows)
    _write_csv(args.output / "per_system.csv", per_system)
    metrics = {
        "schema": SCHEMA,
        "row_count": len(rows),
        "system_count": len({row["system"] for row in rows}),
        "selected_train_records": selected,
        "grouped_by_lag_history": by_lag,
        "aggregation": "record mean within system/lag/history -> system equal",
        "future_informed_diagnostic": True,
        "generation_quality_claim": False,
        "valid_opened": False,
        "test_opened": False,
    }
    atomic_write_json(args.output / "metrics.json", metrics)
    (args.output / "report.md").write_text(_report(metrics), encoding="utf-8")
    atomic_write_json(args.output / "run_manifest.json", {
        "schema": "molvid.frame_gm.latent_scale_run.v1",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": args.checkpoint_sha256},
        "codec": {"path": str(args.codec.resolve()), "sha256": args.codec_sha256},
        "train_store": {
            "path": str(args.train_store.resolve()),
            "index_sha256": sha256_file(args.train_store / "index.txt"),
        },
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "statistics_hash": model.statistics_hash,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "histories": sorted(set(int(value) for value in args.histories)),
        "lags_ps": sorted(set(int(value) for value in args.lags)),
        "sensitivity_records": sensitivity_used,
        "rows_sha256": sha256_file(rows_path),
        "metrics_sha256": sha256_file(args.output / "metrics.json"),
        "per_system_sha256": sha256_file(args.output / "per_system.csv"),
        "test_opened": False,
    })
    print(json.dumps({"complete": True, "rows": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
