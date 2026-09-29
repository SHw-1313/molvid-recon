"""Evaluate clean reconstruction and static denoising on held-out MD structures.

The diagnostic uses one deterministic validation structure per system/replica,
keeps clean coordinates exclusively as targets, and recreates identical noise
from a record-local seed for every independently evaluated checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.static import trajectory_record_to_static
from molvid.data.store import ClipMMapDataset
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.runtime import atomic_write_json, canonical_hash, configure_device, sha256_file
from molvid.training.static_decoder import (
    StaticDecoderLossConfig,
    StaticPerturbationConfig,
    prepare_static_decoder_batch,
    static_clash_loss_function,
    static_decoder_loss,
)


METRIC_NAMES = (
    "input_coordinate_mse_A2",
    "output_coordinate_mse_A2",
    "output_coordinate_rmse_A",
    "output_from_input_coordinate_mse_A2",
    "bond_distance_mse_A2",
    "clash_raw_A2_per_atom",
    "unscaled_training_objective",
    "realized_rms_displacement_A",
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()


def _validate_sigmas(values: Sequence[float]) -> tuple[float, ...]:
    sigmas = tuple(float(value) for value in values)
    if not sigmas:
        raise ValueError("at least one static evaluation sigma is required")
    if any(not math.isfinite(value) or value < 0.0 for value in sigmas):
        raise ValueError("static evaluation sigmas must be finite and non-negative")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("static evaluation sigmas must be unique")
    return sigmas


def _select_source_records(
    manifest: Mapping[str, Any], *, anchor_window: int
) -> list[dict[str, Any]]:
    """Select exactly one declared validation clip for every system/replica."""

    if manifest.get("test_opened") is not False:
        raise ValueError("held-out static evaluation requires test_opened=false")
    systems = tuple(str(value) for value in manifest.get("systems", ()))
    replicas = tuple(str(value) for value in manifest.get("replicas", ()))
    records = manifest.get("selected_records")
    if not systems or not replicas or not isinstance(records, list):
        raise ValueError("evaluation manifest lacks systems, replicas, or selected_records")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for value in records:
        if not isinstance(value, Mapping) or int(value.get("anchor_window_300", -1)) != anchor_window:
            continue
        row = dict(value)
        key = (str(row.get("system")), str(row.get("replica")))
        if key in selected:
            raise ValueError(f"duplicate held-out static selection for {key}")
        if not row.get("source_sample_id_300"):
            raise ValueError(f"held-out static selection {key} lacks source_sample_id_300")
        selected[key] = row
    expected = {(system, replica) for system in systems for replica in replicas}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(f"held-out static selection differs: missing={missing}, extra={extra}")
    return [selected[key] for key in sorted(selected)]


def _noise_seed(base_seed: int, sample_id: str, frame_index: int, sigma: float) -> int:
    identity = f"{int(base_seed)}|{sample_id}|{int(frame_index)}|{float(sigma):.9g}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (2**63 - 1)


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty static diagnostic group")
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [float(row["metrics"][name]) for row in rows]
        if any(not math.isfinite(value) for value in values):
            raise FloatingPointError(f"non-finite static diagnostic metric {name}")
        result[name] = float(np.mean(values, dtype=np.float64))
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Average replica rows inside a system, then give systems equal weight."""

    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["sigma_per_axis_A"]), str(row["system"]))].append(row)
    per_system: list[dict[str, Any]] = []
    for (sigma, system), values in sorted(grouped.items()):
        per_system.append(
            {
                "sigma_per_axis_A": sigma,
                "system": system,
                "replica_count": len(values),
                "metrics": _mean_metrics(values),
            }
        )
    by_sigma: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_system:
        by_sigma[float(row["sigma_per_axis_A"])].append(row)
    summary: dict[str, Any] = {}
    for sigma, values in sorted(by_sigma.items()):
        metrics = _mean_metrics(values)
        input_mse = metrics["input_coordinate_mse_A2"]
        output_mse = metrics["output_coordinate_mse_A2"]
        summary[f"{sigma:.9g}"] = {
            "sigma_per_axis_A": sigma,
            "system_count": len(values),
            "metrics": metrics,
            "coordinate_mse_reduction_fraction": (
                None if input_mse == 0.0 else float((input_mse - output_mse) / input_mse)
            ),
        }
    return per_system, summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "recipe", "static-config", "store", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/frame_gm_p2_formal_GM_260923.yaml")
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--anchor-window", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    sigmas = _validate_sigmas(args.sigmas)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = configure_device("cuda", deterministic=True)
    model_config = load_config(args.model_config, schema="molvid.frame_joint.train.v1")
    loaded = load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=model_config["codec"]["checkpoint"],
        codec_sha256=model_config["codec"]["sha256"],
        device=device,
    )
    model = loaded["model"].eval()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    physical_config = PhysicalAuxiliaryConfig.resolve(recipe["physical"])
    static_config = json.loads(args.static_config.read_text(encoding="utf-8"))
    auxiliary = static_config.get("static_decoder_auxiliary")
    if not isinstance(auxiliary, Mapping) or not isinstance(auxiliary.get("loss"), Mapping):
        raise ValueError("static config lacks the resolved static decoder loss")
    loss_config = StaticDecoderLossConfig.resolve(auxiliary["loss"])
    clash_function = static_clash_loss_function(physical_config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selection = _select_source_records(manifest, anchor_window=args.anchor_window)
    selection_identity = [
        {
            "system": row["system"],
            "replica": row["replica"],
            "source_sample_id": row["source_sample_id_300"],
            "anchor_window_300": row["anchor_window_300"],
        }
        for row in selection
    ]
    store_index_sha256 = sha256_file(args.store / "index.txt")
    declared_store_hash = manifest.get("source_valid_index_sha256")
    if declared_store_hash and store_index_sha256 != declared_store_hash:
        raise ValueError("source validation store index differs from its manifest")
    protocol = {
        "schema": "molvid.static_decoder.heldout_evaluation.v1",
        "checkpoint_sha256": args.checkpoint_sha256,
        "training_implementation_commit": auxiliary.get("gamma_calibration", {}).get(
            "code_commit"
        ),
        "evaluator_code_commit": _git_commit(),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "model_config_sha256": sha256_file(args.model_config),
        "recipe_sha256": sha256_file(args.recipe),
        "static_config_sha256": sha256_file(args.static_config),
        "store_index_sha256": store_index_sha256,
        "manifest_sha256": sha256_file(args.manifest),
        "selection_sha256": canonical_hash(selection_identity),
        "selection": selection_identity,
        "selection_rule": "manifest anchor_window_300 match, one clip per system/replica; middle valid frame",
        "sigmas_per_axis_A": list(sigmas),
        "sigma_definition": "independent Gaussian standard deviation per Cartesian axis",
        "noise_seed": int(args.seed),
        "noise_identity": "sha256(base seed, source sample id, frame index, sigma); identical across arms",
        "loss": loss_config.contract(),
        "aggregation": "replica mean within system, then system-equal mean",
        "precision": "fp32",
        "split": "valid",
        "test_opened": False,
    }
    atomic_write_json(args.output / "protocol.json", protocol)
    dataset = ClipMMapDataset(args.store)
    rows: list[dict[str, Any]] = []
    try:
        by_id = {str(row[0]): index for index, row in enumerate(dataset._index)}
        for selected in selection:
            source_id = str(selected["source_sample_id_300"])
            if source_id not in by_id:
                raise ValueError(f"selected held-out source is absent from store: {source_id}")
            record = dataset[by_id[source_id]]
            valid_frames = np.flatnonzero(np.asarray(record["frame_mask"], dtype=np.bool_))
            frame_index = int(valid_frames[len(valid_frames) // 2])
            static_record = trajectory_record_to_static(
                record,
                frame_index=frame_index,
                sample_id=f"{source_id}::heldout_static_middle_{frame_index:06d}",
            )
            cpu_batch = collate_clip_records([static_record])
            for sigma in sigmas:
                perturbation = StaticPerturbationConfig(
                    sigma_per_axis_angstrom=((0.05,) if sigma == 0.0 else (sigma,)),
                    clean_probability=(1.0 if sigma == 0.0 else 0.0),
                )
                local_seed = _noise_seed(args.seed, source_id, frame_index, sigma)
                prepared = prepare_static_decoder_batch(
                    model.target_teacher,
                    cpu_batch,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(local_seed),
                    perturbation=perturbation,
                )
                with torch.no_grad():
                    output = model.decoder(prepared.observed_context)
                    losses = static_decoder_loss(
                        output,
                        prepared,
                        loss_config,
                        clash_loss_function=clash_function,
                    )
                    valid = prepared.observed_context.loss_mask
                    target = prepared.target_coordinates[0].float()
                    input_coordinates = prepared.observed_context.coordinates[0].float()
                    prediction = output.coordinates[0].float()
                    input_mse = (input_coordinates[valid] - target[valid]).square().mean()
                    output_mse = (prediction[valid] - target[valid]).square().mean()
                    output_from_input_mse = (
                        prediction[valid] - input_coordinates[valid]
                    ).square().mean()
                metrics = {
                    "input_coordinate_mse_A2": float(input_mse),
                    "output_coordinate_mse_A2": float(output_mse),
                    "output_coordinate_rmse_A": float(output_mse.sqrt()),
                    "output_from_input_coordinate_mse_A2": float(output_from_input_mse),
                    "bond_distance_mse_A2": float(losses.raw["bond_distance"]),
                    "clash_raw_A2_per_atom": float(losses.raw["clash"]),
                    "unscaled_training_objective": float(losses.unscaled_total),
                    "realized_rms_displacement_A": float(
                        prepared.realized_rms_displacement_angstrom[0]
                    ),
                }
                if any(not math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"non-finite static diagnostic row: {source_id}")
                rows.append(
                    {
                        "schema": "molvid.static_decoder.heldout_row.v1",
                        "row_key": f"{source_id}|frame={frame_index}|sigma={sigma:.9g}",
                        "system": str(selected["system"]),
                        "replica": str(selected["replica"]),
                        "source_sample_id": source_id,
                        "source_frame_index": frame_index,
                        "source_time_ps": float(record["time_ps"][frame_index]),
                        "sigma_per_axis_A": sigma,
                        "noise_seed": local_seed,
                        "supervised_atoms": int(prepared.input_batch.loss_mask.sum()),
                        "metrics": metrics,
                    }
                )
            print(json.dumps({"completed_records": len(rows) // len(sigmas), "rows": len(rows)}), flush=True)
    finally:
        dataset.close()
    expected_rows = len(selection) * len(sigmas)
    if len(rows) != expected_rows or len({row["row_key"] for row in rows}) != expected_rows:
        raise RuntimeError("held-out static evaluation row coverage differs")
    rows_path = args.output / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    per_system, aggregate = _aggregate(rows)
    atomic_write_json(args.output / "per_system.json", {"rows": per_system})
    atomic_write_json(
        args.output / "summary.json",
        {
            "schema": "molvid.static_decoder.heldout_summary.v1",
            "checkpoint_sha256": args.checkpoint_sha256,
            "protocol_sha256": sha256_file(args.output / "protocol.json"),
            "record_count": len(selection),
            "row_count": len(rows),
            "system_count": len({row["system"] for row in rows}),
            "replica_count": len({row["replica"] for row in rows}),
            "aggregate": aggregate,
            "gpu_hours_this_invocation": (time.time() - started) / 3600.0,
            "test_opened": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
