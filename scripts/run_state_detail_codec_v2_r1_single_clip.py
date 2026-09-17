#!/usr/bin/env python3
"""Run the predeclared true single-clip repaired R1 reconstruction gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.clip_dataset import ClipMMapDataset
from evaluation.codec_evaluation import _metrics
from trainer.codec_trainer import CodecTrainer, prepare_batch_then_to_device

from scripts.run_state_detail_codec_v2_t0 import (
    DEVICE,
    FRAME_ENCODER_CHECKPOINT,
    ROOT,
    STORE_ROOT,
    _ListDataset,
    _make_config,
    _make_loader,
    _make_model,
    _records_for_micro,
    _runtime,
    _sha256,
    _state_hash,
    _evaluate_loader,
)


SAMPLE_ID = "atlas_5e3e_A_R1_w000000"
STEPS = 1000
LOG_EVERY = 25
ALIGNED_RMSD_MAX = 1.5
RAW_RMSD_MAX = 1.5
DRMSD_MAX = 1.5
BOND_RMSE_MAX = 0.5
MIN_ALIGNED_IMPROVEMENT = 0.50
FINAL_WINDOW_POINTS = 5
FINAL_WINDOW_TOLERANCE = 0.10


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _new_module_gradients(model: torch.nn.Module) -> dict[str, Any]:
    modules = {
        "coordinate_vector_stem": getattr(model, "coordinate_vector_stem", None),
        "state_detail_codec": getattr(model, "state_detail_codec", None),
    }
    result: dict[str, Any] = {}
    for module_name, module in modules.items():
        if module is None:
            result[module_name] = {"present": False, "parameters": {}, "all_finite_nonzero": False}
            continue
        parameters: dict[str, float | None] = {}
        for name, parameter in module.named_parameters():
            parameters[name] = (
                None
                if parameter.grad is None
                else float(parameter.grad.detach().float().abs().sum().cpu())
            )
        nonzero = [value for value in parameters.values() if value is not None and value > 0]
        finite = all(value is None or math.isfinite(value) for value in parameters.values())
        result[module_name] = {
            "present": True,
            "parameters": parameters,
            "all_finite_nonzero": bool(finite and len(nonzero) == len(parameters)),
        }
    result["all_finite_nonzero"] = all(
        bool(item.get("all_finite_nonzero"))
        for name, item in result.items()
        if name != "all_finite_nonzero"
    )
    return result


def _all_frames_metrics(evaluation: Mapping[str, Any]) -> Mapping[str, float]:
    if "metrics" in evaluation:
        return evaluation["metrics"]["all_frames"]
    return evaluation["all_frames"]


def _origin_only_metrics(model: Any, batch: Any) -> dict[str, Any]:
    moved = prepare_batch_then_to_device(model, batch, DEVICE, non_blocking=False)
    with torch.no_grad():
        _centered, origin = model._centered_batch(moved)
        prediction = origin.index_select(0, moved.abid).unsqueeze(0).expand_as(moved.x)
        result = _metrics(prediction, moved.x, moved)
    del moved
    torch.cuda.empty_cache()
    return result


def run(output_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("the formal single-clip R1 gate requires CUDA; no CPU fallback is allowed")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ClipMMapDataset(STORE_ROOT / "train")
    try:
        by_id = {str(row[0]): index for index, row in enumerate(dataset._index)}
        if SAMPLE_ID not in by_id:
            raise RuntimeError(f"single-clip sample is missing: {SAMPLE_ID}")
        record = dataset[by_id[SAMPLE_ID]]
        if str(record["sample_id"]) != SAMPLE_ID:
            raise RuntimeError("single-clip sample ID changed during loading")
        records = [record]
        loader = _make_loader(records, seed=20260903, shuffle=False, replacement=True)
        if len(loader) != 1:
            raise RuntimeError(f"single-clip loader must contain one batch, got {len(loader)}")
        model = _make_model("ratio1_state_detail", freeze_frame_encoder=True)
        trainer = CodecTrainer(
            model,
            loader,
            config=_make_config(STEPS, train_batches=1, schedule_steps=STEPS),
            device=DEVICE,
            non_blocking_transfer=False,
        )
        trainer.fit_normalization(loader)
        dataset_view = _ListDataset(records)
        initial = _evaluate_loader(
            trainer, loader, dataset_view, epoch=0, ratio=1, detailed=True
        )
        origin_only = _origin_only_metrics(model, next(iter(loader)))
        before_frame_hash = _state_hash(model.frame_encoder)
        batch = next(iter(loader))
        rows: list[dict[str, Any]] = []
        initial_record = {
            "step": 0,
            "optimizer_metrics": {},
            "evaluation": initial,
        }
        rows.append(initial_record)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        start_time = time.perf_counter()
        for step in range(1, STEPS + 1):
            optimizer_metrics = trainer.optimizer_step(batch)
            if step % LOG_EVERY == 0 or step == STEPS:
                evaluation = _evaluate_loader(
                    trainer, loader, dataset_view, epoch=0, ratio=1, detailed=True
                )
                rows.append(
                    {
                        "step": step,
                        "optimizer_metrics": optimizer_metrics,
                        "evaluation": evaluation,
                    }
                )
        torch.cuda.synchronize(DEVICE)
        elapsed = time.perf_counter() - start_time
        final = rows[-1]["evaluation"]
        final_metrics = _all_frames_metrics(final)
        baseline_metrics = _all_frames_metrics(origin_only)
        final_window = [
            float(_all_frames_metrics(row["evaluation"])["aligned_rmsd"])
            for row in rows[-FINAL_WINDOW_POINTS:]
        ]
        final_window_median = statistics.median(final_window)
        convergence = bool(
            math.isfinite(final_window_median)
            and abs(final_window[-1] - final_window_median) <= FINAL_WINDOW_TOLERANCE
            and final_window[-1] <= final_window[0] + FINAL_WINDOW_TOLERANCE
        )
        after_frame_hash = _state_hash(model.frame_encoder)
        gradient_summary = _new_module_gradients(model)
        finite_curve = _finite_numbers(rows)
        improvement = 1.0 - float(final_metrics["aligned_rmsd"]) / max(
            float(baseline_metrics["aligned_rmsd"]), 1.0e-8
        )
        gate_checks = {
            "finite_curve": finite_curve,
            "aligned_rmsd": float(final_metrics["aligned_rmsd"]) <= ALIGNED_RMSD_MAX,
            "centroid_gauge_raw_rmsd": float(final_metrics["centroid_gauge_raw_rmsd"]) <= RAW_RMSD_MAX,
            "drmsd": float(final_metrics["drmsd"]) <= DRMSD_MAX,
            "bond_rmse": float(final_metrics["bond_rmse"]) <= BOND_RMSE_MAX,
            "origin_only_aligned_improvement": improvement >= MIN_ALIGNED_IMPROVEMENT,
            "final_window_converged": convergence,
            "frame_encoder_unchanged": before_frame_hash == after_frame_hash,
            "new_module_gradients": bool(gradient_summary["all_finite_nonzero"]),
        }
        checkpoint = output_dir / "codec_step_001000.pt"
        trainer.save_checkpoint(checkpoint)
        resume_model = _make_model("ratio1_state_detail", freeze_frame_encoder=True)
        resume_loader = _make_loader(records, seed=20260903, shuffle=False, replacement=True)
        resume_trainer = CodecTrainer(
            resume_model,
            resume_loader,
            config=_make_config(STEPS, train_batches=1, schedule_steps=STEPS),
            device=DEVICE,
            non_blocking_transfer=False,
        )
        resume_trainer.load_checkpoint(checkpoint)
        loaded_step = resume_trainer.step
        resume_trainer.run(max_steps=STEPS + 1)
        resume = {
            "status": "passed" if resume_trainer.step == STEPS + 1 else "failed",
            "loaded_step": loaded_step,
            "resumed_step": resume_trainer.step,
        }
        gate_checks["checkpoint_resume"] = resume["status"] == "passed"
        curve_path = output_dir / "train_metrics.jsonl"
        curve_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        result = {
            "status": "passed" if all(gate_checks.values()) else "failed",
            "gate_name": "r1_true_single_clip_no_anchor_v1",
            "sample_id": SAMPLE_ID,
            "sample_count": 1,
            "frames": int(record["x"].shape[0]),
            "steps": STEPS,
            "log_every": LOG_EVERY,
            "seed": 20260903,
            "spatial_backbone": "torchmd_et",
            "coordinate_stem": "centered_vector",
            "precision": "fp32",
            "frozen_frame_encoder": True,
            "frame_encoder_checkpoint": str(FRAME_ENCODER_CHECKPOINT),
            "frame_encoder_checkpoint_sha256": _sha256(FRAME_ENCODER_CHECKPOINT),
            "frame_encoder_state_hash_before": before_frame_hash,
            "frame_encoder_state_hash_after": after_frame_hash,
            "frame_encoder_unchanged": before_frame_hash == after_frame_hash,
            "origin_only_baseline": origin_only,
            "initial": initial,
            "final": final,
            "final_metrics_all_frames": final_metrics,
            "thresholds_frozen_before_run": {
                "aligned_rmsd_max": ALIGNED_RMSD_MAX,
                "centroid_gauge_raw_rmsd_max": RAW_RMSD_MAX,
                "drmsd_max": DRMSD_MAX,
                "bond_rmse_max": BOND_RMSE_MAX,
                "minimum_origin_only_aligned_improvement": MIN_ALIGNED_IMPROVEMENT,
                "final_window_tolerance": FINAL_WINDOW_TOLERANCE,
            },
            "gate_checks": gate_checks,
            "origin_only_aligned_improvement": improvement,
            "final_window_aligned_rmsd": final_window,
            "final_window_median": final_window_median,
            "new_module_gradients": gradient_summary,
            "codec_only_parameter_count": sum(
                parameter.numel()
                for module in (model.coordinate_vector_stem, model.state_detail_codec)
                if module is not None
                for parameter in module.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "runtime": {
                "training_elapsed_s": elapsed,
                "steps_per_s": STEPS / max(elapsed, 1.0e-8),
                "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
                "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
            },
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_resume": resume,
            "curve_path": str(curve_path),
        }
        _write_json(output_dir / "result.json", result)
        if result["status"] != "passed":
            raise RuntimeError(f"formal R1 single-clip gate failed: {gate_checks}")
        del resume_trainer, resume_model, resume_loader, trainer, model, loader, batch
        torch.cuda.empty_cache()
        gc.collect()
        return result
    finally:
        dataset.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/state_detail_codec_v2/repair/r1_single_clip"
        / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S")),
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R1 gate directory: {output_dir}")
    result = run(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
