#!/usr/bin/env python3
"""Run the bounded post-R1 smoke for the remaining state/detail controls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.clip_dataset import ClipMMapDataset
from module.state_detail_codec_v2 import StateDetailCodecV2
from trainer.codec_trainer import CodecTrainer, prepare_batch_then_to_device

from scripts.run_state_detail_codec_v2_t0 import (
    DEVICE,
    ROOT,
    STORE_ROOT,
    _ListDataset,
    _evaluate_loader,
    _make_config,
    _make_loader,
    _make_model,
    _sha256,
    _state_hash,
)


SAMPLE_ID = "atlas_5e3e_A_R1_w000000"
STEPS = 200
LOG_EVERY = 50
MODES = (
    "ratio2_state_detail",
    "ratio4_state_detail",
    "ratio4_matched_pooling",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gradients(model: torch.nn.Module) -> dict[str, Any]:
    modules = {
        "coordinate_vector_stem": getattr(model, "coordinate_vector_stem", None),
        "state_detail_codec": getattr(model, "state_detail_codec", None),
    }
    result: dict[str, Any] = {}
    for module_name, module in modules.items():
        parameters: dict[str, float | None] = {}
        if module is not None:
            for name, parameter in module.named_parameters():
                parameters[name] = (
                    None
                    if parameter.grad is None
                    else float(parameter.grad.detach().float().abs().sum().cpu())
                )
        nonzero = [value for value in parameters.values() if value is not None and value > 0]
        finite = all(value is None or math.isfinite(value) for value in parameters.values())
        result[module_name] = {
            "parameters": parameters,
            "all_finite_nonzero": bool(parameters and finite and len(nonzero) == len(parameters)),
        }
    result["all_finite_nonzero"] = all(
        item["all_finite_nonzero"] for item in result.values()
    )
    return result


def _zero_preserving_smoke(mode: str, device: torch.device) -> dict[str, Any]:
    if mode == "ratio4_matched_pooling":
        return {
            "applicable": False,
            "reason": "matched pooling has no state/detail semantics",
        }
    torch.manual_seed(20260903)
    h = torch.randn(16, 3, 8, device=device)
    v = torch.randn(16, 3, 3, 8, device=device)
    h[:] = h[0]
    v[:] = v[0]
    codec = StateDetailCodecV2(8, mode=mode).to(device)
    output = codec(
        h,
        v,
        time_ps=torch.arange(16, device=device, dtype=torch.float32).view(1, 16),
        frame_mask=torch.ones(1, 16, device=device, dtype=torch.bool),
        abid=torch.zeros(3, device=device, dtype=torch.long),
        sample_origin=torch.zeros(1, 3, device=device),
    )
    detail_zero = (
        output.latent.detail_h is None
        or torch.equal(output.latent.detail_h, torch.zeros_like(output.latent.detail_h))
    )
    motion_zero = torch.equal(output.x_hat, output.x_hat[0:1].expand_as(output.x_hat))
    return {
        "applicable": True,
        "detail_zero": bool(detail_zero),
        "coordinate_motion_zero": bool(motion_zero),
    }


def _compact_eval(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    result = {
        "loss": evaluation["loss"],
        "metrics": {
            "all_frames": metrics["all_frames"],
            "future": metrics["future"],
            "frequency_retention": metrics["frequency_retention"],
        },
        "detail_summary": evaluation["detail_summary"],
        "counterfactual": evaluation["counterfactual"],
    }
    return result


def run(output_dir: Path, modes: Sequence[str]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ratio smoke requires CUDA; no CPU fallback is allowed")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ClipMMapDataset(STORE_ROOT / "train")
    results: dict[str, Any] = {}
    try:
        by_id = {str(row[0]): index for index, row in enumerate(dataset._index)}
        if SAMPLE_ID not in by_id:
            raise RuntimeError(f"ratio smoke sample is missing: {SAMPLE_ID}")
        record = dataset[by_id[SAMPLE_ID]]
        records = [record]
        for mode in modes:
            mode_dir = output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=False)
            loader = _make_loader(records, seed=20260903, shuffle=False, replacement=True)
            model = _make_model(mode, freeze_frame_encoder=True)
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
                trainer, loader, dataset_view, epoch=0, ratio={
                    "ratio2_state_detail": 2,
                    "ratio4_state_detail": 4,
                    "ratio4_matched_pooling": 4,
                }[mode], detailed=True
            )
            before_hash = _state_hash(model.frame_encoder)
            batch = next(iter(loader))
            torch.cuda.reset_peak_memory_stats(DEVICE)
            start = time.perf_counter()
            curve: list[dict[str, Any]] = [{"step": 0, "evaluation": _compact_eval(initial)}]
            for step in range(1, STEPS + 1):
                optimizer_metrics = trainer.optimizer_step(batch)
                if step % LOG_EVERY == 0 or step == STEPS:
                    evaluation = _evaluate_loader(
                        trainer, loader, dataset_view, epoch=0, ratio={
                            "ratio2_state_detail": 2,
                            "ratio4_state_detail": 4,
                            "ratio4_matched_pooling": 4,
                        }[mode], detailed=True
                    )
                    curve.append(
                        {
                            "step": step,
                            "optimizer_metrics": optimizer_metrics,
                            "evaluation": _compact_eval(evaluation),
                        }
                    )
            torch.cuda.synchronize(DEVICE)
            elapsed = time.perf_counter() - start
            final = curve[-1]["evaluation"]
            final_eval = _evaluate_loader(
                trainer, loader, dataset_view, epoch=0, ratio={
                    "ratio2_state_detail": 2,
                    "ratio4_state_detail": 4,
                    "ratio4_matched_pooling": 4,
                }[mode], detailed=True
            )
            after_hash = _state_hash(model.frame_encoder)
            checkpoint = mode_dir / f"codec_step_{STEPS:08d}.pt"
            trainer.save_checkpoint(checkpoint)
            resume_model = _make_model(mode, freeze_frame_encoder=True)
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
            zero_smoke = _zero_preserving_smoke(mode, DEVICE)
            full = final_eval["counterfactual"]["full"]
            detail_zero = final_eval["counterfactual"]["detail_zero"]
            detail_delta = None
            if detail_zero is not None:
                detail_delta = {
                    "future_aligned_rmsd": detail_zero["future"]["aligned_rmsd"] - full["future"]["aligned_rmsd"],
                    "future_drmsd": detail_zero["future"]["drmsd"] - full["future"]["drmsd"],
                }
            gradients = _gradients(model)
            curve_path = mode_dir / "train_metrics.jsonl"
            curve_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in curve) + "\n",
                encoding="utf-8",
            )
            result = {
                "mode": mode,
                "status": "passed" if (
                    before_hash == after_hash
                    and gradients["all_finite_nonzero"]
                    and zero_smoke.get("detail_zero", True)
                    and zero_smoke.get("coordinate_motion_zero", True)
                    and resume_trainer.step == STEPS + 1
                ) else "failed",
                "sample_id": SAMPLE_ID,
                "sample_count": 1,
                "steps": STEPS,
                "log_every": LOG_EVERY,
                "spatial_backbone": "torchmd_et",
                "coordinate_stem": "centered_vector",
                "frame_encoder_state_hash_before": before_hash,
                "frame_encoder_state_hash_after": after_hash,
                "frame_encoder_unchanged": before_hash == after_hash,
                "new_module_gradients": gradients,
                "zero_preserving_smoke": zero_smoke,
                "initial": _compact_eval(initial),
                "final": final,
                "final_counterfactual": {
                    "full": full,
                    "detail_zero": detail_zero,
                    "detail_zero_minus_full": detail_delta,
                },
                "runtime": {
                    "training_elapsed_s": elapsed,
                    "steps_per_s": STEPS / max(elapsed, 1.0e-8),
                    "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
                    "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
                },
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_resume": {
                    "status": "passed" if resume_trainer.step == STEPS + 1 else "failed",
                    "loaded_step": loaded_step,
                    "resumed_step": resume_trainer.step,
                },
                "curve_path": str(curve_path),
                "model_contract": model.model_contract(),
            }
            _write_json(mode_dir / "result.json", result)
            results[mode] = result
            del resume_trainer, resume_model, resume_loader, trainer, model, loader, batch
            torch.cuda.empty_cache()
            gc.collect()
            if result["status"] != "passed":
                raise RuntimeError(f"ratio smoke failed for {mode}")
        overall = {
            "status": "passed",
            "sample_id": SAMPLE_ID,
            "steps": STEPS,
            "backbones": results,
            "shared_decoder_contract": {
                mode: {
                    "coordinate_head": results[mode]["model_contract"]["architecture"]["decoder"].get(
                        "coordinate_head", "shared_framewise_equivariant"
                    ),
                    "coordinate_stem": results[mode]["model_contract"]["architecture"]["decoder"].get(
                        "coordinate_stem"
                    ),
                }
                for mode in modes
            },
            "t1_started": False,
        }
        _write_json(output_dir / "summary.json", overall)
        return overall
    finally:
        dataset.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/state_detail_codec_v2/repair/ratio_smoke" / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S")))
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ratio smoke directory: {output_dir}")
    modes = MODES if args.mode == "all" else (args.mode,)
    result = run(output_dir, modes)
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
