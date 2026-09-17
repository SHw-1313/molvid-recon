#!/usr/bin/env python3
"""Bounded cached-feature diagnostic for the repaired state/detail R1 decoder.

This is a diagnostic only.  It uses exactly one real training clip, caches
the frozen TorchMD features and centered targets, and optimizes a detached
copy of the existing pointwise codec path.  It never writes a manifest and is
not a T0 or T1 experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.codec_evaluation import _mask, _metrics
from module.state_detail_codec_v2 import StateDetailCodecV2
from trainer.codec_trainer import prepare_batch_then_to_device

from scripts.run_state_detail_codec_v2_t0 import (
    DEVICE,
    FRAME_ENCODER_CHECKPOINT,
    ROOT,
    STORE_ROOT,
    _make_model,
    _sha256,
)


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/state_detail_codec_v2/repair/r1_cached_feature_diagnostic"
SAMPLE_ID = "atlas_5e3e_A_R1_w000000"
SEED = 20260903
STEPS = 500
LOG_EVERY = 10


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _find_record(dataset: ClipMMapDataset) -> Mapping[str, Any]:
    matches = [index for index, row in enumerate(dataset._index) if str(row[0]) == SAMPLE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one diagnostic sample {SAMPLE_ID!r}, found {len(matches)}")
    record = dataset[matches[0]]
    if str(record["sample_id"]) != SAMPLE_ID:
        raise RuntimeError("diagnostic sample ID changed during loading")
    if int(record["x"].shape[0]) != 16 or np.asarray(record["x"]).dtype != np.float32:
        raise RuntimeError("diagnostic requires one FP32 T=16 trajectory clip")
    return record


def _coordinate_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return prediction.sum() * 0.0
    return (prediction[mask] - target[mask]).square().mean()


def _gradient_summary(module: torch.nn.Module) -> dict[str, Any]:
    values: dict[str, float | None] = {}
    for name, parameter in module.named_parameters():
        values[name] = (
            None
            if parameter.grad is None
            else float(parameter.grad.detach().float().abs().sum().cpu())
        )
    nonzero = [value for value in values.values() if value is not None and value > 0]
    finite = all(value is None or math.isfinite(value) for value in values.values())
    return {
        "parameters": values,
        "all_finite_nonzero": bool(finite and len(nonzero) == len(values)),
    }


def _record_metrics(
    codec: StateDetailCodecV2,
    h: torch.Tensor,
    v: torch.Tensor,
    batch: Any,
    target: torch.Tensor,
    origin: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        output = codec(
            h,
            v,
            time_ps=batch.time_ps,
            frame_mask=batch.frame_mask,
            abid=batch.abid,
            sample_origin=origin,
        )
    metrics = _metrics(output.x_hat, target, batch)
    return {
        "loss": float(_coordinate_loss(output.x_hat, target, _mask(batch, target.shape[0], target.device)).cpu()),
        "aligned_rmsd": metrics["all_frames"]["aligned_rmsd"],
        "centroid_gauge_raw_rmsd": metrics["all_frames"]["centroid_gauge_raw_rmsd"],
        "drmsd": metrics["all_frames"]["drmsd"],
        "bond_rmse": metrics["all_frames"]["bond_rmse"],
    }


def run(output_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R1 cached-feature diagnostic requires CUDA; no CPU fallback is allowed")
    _seed()
    dataset = ClipMMapDataset(STORE_ROOT / "train")
    try:
        record = _find_record(dataset)
        batch = collate_clip_records([record])
        model = _make_model(
            "ratio1_state_detail",
            freeze_frame_encoder=True,
            coordinate_stem="none",
        ).to(DEVICE)
        moved = prepare_batch_then_to_device(model, batch, DEVICE, non_blocking=False)
        model.eval()
        with torch.no_grad():
            centered_batch, origin = model._centered_batch(moved)
            encoded = model.frame_encoder(centered_batch)
            target = moved.x.clone()
            centered_target = centered_batch.x.clone()
            current_output = model(moved).x_hat
        cache_path = output_dir / "cached_features.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "pvb.codec.state_detail.r1_cached_features.v1",
                "sample_id": SAMPLE_ID,
                "h": encoded.h.cpu(),
                "v": encoded.v.cpu(),
                "centered_target": centered_target.cpu(),
                "target": target.cpu(),
                "origin": origin.cpu(),
                "frame_mask": moved.frame_mask.cpu(),
                "time_ps": moved.time_ps.cpu(),
                "feature_shapes": {
                    "h": list(encoded.h.shape),
                    "v": list(encoded.v.shape),
                    "centered_target": list(centered_target.shape),
                },
            },
            cache_path,
        )
        mask = _mask(moved, target.shape[0], target.device)
        origin_prediction = origin.index_select(0, moved.abid).unsqueeze(0).expand_as(target)
        current_metrics = _metrics(current_output, target, moved)
        origin_metrics = _metrics(origin_prediction, target, moved)

        # A linear equivariant oracle answers whether the cached vector
        # channels already contain centered coordinate information.  It is
        # fitted only inside this diagnostic and is never part of production.
        vector_rows = encoded.v[mask].reshape(-1, encoded.v.shape[-1])
        target_rows = centered_target[mask].reshape(-1, 1)
        oracle_weight = torch.linalg.lstsq(vector_rows, target_rows).solution
        oracle_centered = torch.einsum("tnkc,cq->tnkq", encoded.v, oracle_weight).squeeze(-1)
        oracle_prediction = oracle_centered + origin.index_select(0, moved.abid).unsqueeze(0)
        oracle_metrics = _metrics(oracle_prediction, target, moved)

        # Optimize a detached copy of the pre-stem pointwise codec on the
        # cached features.  This provides train RMSD/dRMSD/bond curves and
        # separates optimization/head limitations from feature information.
        diagnostic_codec = StateDetailCodecV2(128, mode="ratio1_state_detail").to(DEVICE)
        diagnostic_codec.train()
        optimizer = torch.optim.Adam(diagnostic_codec.parameters(), lr=1.0e-3)
        h = encoded.h.detach()
        v = encoded.v.detach()
        target = target.detach()
        rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        rows.append({"step": 0, **_record_metrics(diagnostic_codec, h, v, moved, target, origin)})
        for step in range(1, STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            output = diagnostic_codec(
                h,
                v,
                time_ps=moved.time_ps,
                frame_mask=moved.frame_mask,
                abid=moved.abid,
                sample_origin=origin,
            )
            loss = _coordinate_loss(output.x_hat, target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"cached pointwise diagnostic produced non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            if step % LOG_EVERY == 0 or step == STEPS:
                rows.append({"step": step, **_record_metrics(diagnostic_codec, h, v, moved, target, origin)})
        torch.cuda.synchronize(DEVICE)
        elapsed = time.perf_counter() - start_time
        final = rows[-1]
        result = {
            "status": "passed",
            "diagnostic_only": True,
            "sample_id": SAMPLE_ID,
            "sample_count": 1,
            "frames": int(batch.frames),
            "spatial_backbone": "torchmd_et",
            "frozen_frame_encoder": True,
            "frame_encoder_checkpoint": str(FRAME_ENCODER_CHECKPOINT),
            "frame_encoder_checkpoint_sha256": _sha256(FRAME_ENCODER_CHECKPOINT),
            "coordinate_stem_in_diagnostic": "none",
            "cache_path": str(cache_path),
            "cache_sha256": _sha256(cache_path),
            "cache_shapes": {
                "h": list(encoded.h.shape),
                "v": list(encoded.v.shape),
                "centered_target": list(centered_target.shape),
            },
            "origin_only_baseline": origin_metrics["all_frames"],
            "current_pointwise_initial": current_metrics["all_frames"],
            "linear_vector_oracle": oracle_metrics["all_frames"],
            "optimized_cached_pointwise_final": final,
            "optimized_cached_pointwise_gradient_summary": _gradient_summary(diagnostic_codec),
            "train_curve_metric_names": [
                "loss",
                "aligned_rmsd",
                "centroid_gauge_raw_rmsd",
                "drmsd",
                "bond_rmse",
            ],
            "steps": STEPS,
            "log_every": LOG_EVERY,
            "elapsed_s": elapsed,
            "interpretation": {
                "feature_information_test": "linear equivariant least-squares fit from cached frozen v to centered target",
                "head_test": "500-step detached StateDetailCodecV2 ratio1 pointwise path on cached h/v",
                "production_target_access": False,
            },
        }
        _write_json(output_dir / "diagnostic.json", result)
        (output_dir / "train_metrics.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        dataset.close()
        for name in ("model", "moved", "centered_batch", "encoded", "diagnostic_codec"):
            if name in locals():
                del locals()[name]
        torch.cuda.empty_cache()
        gc.collect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S")))
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory: {output_dir}")
    result = run(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
