#!/usr/bin/env python
"""Run one bounded, validation-only T0 shape smoke for one R2/R4 candidate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.dit_evaluation import _coordinates, evaluate_oracle_vs_generated
from module.latent_rectified_flow import generate_state_detail_latent
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    StateDetailLatentAdapter,
    LatentStatistics,
    build_observation_condition,
    contract_hash,
)
from trainer.codec_trainer import PVBCodecModel
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("ratio2_state_detail", "ratio4_state_detail"), required=True)
    parser.add_argument(
        "--data-root",
        default="outputs/state_detail_codec_v2/t0_data/clip_store/valid",
        help="bounded T0 clip store only; paths containing test are rejected",
    )
    parser.add_argument("--output-root", default="outputs/dit_state_detail_probe_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--history-frames", type=int, default=4)
    return parser


def _require_safe_inputs(data_root: Path, output_root: Path, max_steps: int) -> None:
    if "test" in str(data_root).lower():
        raise RuntimeError("the DiT smoke refuses any path containing the test split")
    if str(output_root).startswith("outputs/state_detail_codec_v2"):
        raise RuntimeError("the DiT smoke refuses the active T1 output root")
    if not 2 <= int(max_steps) <= 100:
        raise ValueError("smoke max_steps must be between 2 and 100, including resume")


def _finite_gradients(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("adapter.scalar_out.state_h"):
            group = "output_head_state_h"
        elif name.startswith("adapter.scalar_out.detail_h"):
            group = "output_head_detail_h"
        elif name.startswith("adapter.vector_out.state_v"):
            group = "output_head_state_v"
        elif name.startswith("adapter.vector_out.detail_v"):
            group = "output_head_detail_v"
        elif name.startswith("adapter"):
            group = "adapter"
        elif ".spatial." in name:
            group = "spatial_attention"
        elif ".temporal." in name:
            group = "temporal_attention"
        elif ".spatial_adaln." in name:
            group = "spatial_adaln"
        elif ".temporal_adaln." in name:
            group = "temporal_adaln"
        elif ".ffn_adaln." in name:
            group = "ffn_adaln"
        elif ".ffn.scalar_" in name:
            group = "scalar_ffn"
        elif ".ffn.vector_" in name:
            group = "vector_ffn"
        elif ".ffn." in name:
            group = "vector_ffn"
        else:
            group = name.split(".", 1)[0]
        groups.setdefault(group, []).append(parameter.grad.detach())
    result = {}
    for name, values in groups.items():
        result[name] = {
            "finite": all(bool(torch.isfinite(value).all()) for value in values),
            "nonzero": any(bool(torch.any(value != 0)) for value in values),
            "parameters": len(values),
        }
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_clip(data_root: Path, records: int):
    dataset = ClipMMapDataset(data_root)
    if len(dataset) < int(records):
        raise ValueError(f"T0 store has only {len(dataset)} records, requested {records}")
    selected = [dataset[index] for index in range(int(records))]
    batches = collate_clip_records(selected)
    return dataset, batches, selected


def run_smoke(
    *,
    candidate: str,
    data_root: str | Path,
    output_root: str | Path,
    device: str,
    max_steps: int,
    records: int,
    seed: int,
    history_frames: int,
) -> dict[str, Any]:
    ratio = 2 if candidate.startswith("ratio2") else 4
    data_root = Path(data_root)
    output_root = Path(output_root)
    _require_safe_inputs(data_root, output_root, max_steps)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    smoke_start = time.perf_counter()
    target_device = torch.device(device)
    dataset, clip_batch_cpu, selected = _load_clip(data_root, records)
    clip_batch = clip_batch_cpu.to(target_device)

    codec = PVBCodecModel(
        hidden_channels=128,
        spatial_layers=1,
        spatial_backbone="torchmd_et",
        temporal_layers=1,
        temporal_ratio=ratio,
        temporal_codec_mode=candidate,
        use_spatial_refiner=False,
        freeze_frame_encoder=False,
        coordinate_stem="centered_vector",
    ).to(target_device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.prepare_batch(clip_batch_cpu)
    codec_hash_before = module_state_hash(codec)
    frame_hash_before = module_state_hash(codec.frame_encoder)
    with torch.no_grad():
        oracle_latent = codec.encode(clip_batch)

    adapter = StateDetailLatentAdapter(
        codec_width=128, scalar_width=256, vector_width=128, ratio=ratio
    ).to(target_device)
    latent_batch = adapter.pack(
        oracle_latent,
        codec_hash=codec_hash_before,
        data_hash="t0_valid_shape_smoke",
        loss_mask=clip_batch.loss_mask,
    )
    condition = build_observation_condition(
        latent_batch,
        history_frames=history_frames,
        coordinates=clip_batch.x,
        frame_mask=clip_batch.frame_mask,
        loss_mask=clip_batch.loss_mask,
    )
    latent_batch = latent_batch.with_observation(
        condition.latent_observation_mask,
        sample_origin=condition.sample_origin,
    )
    stats = LatentStatistics.fit(
        [latent_batch],
        ratio=ratio,
        provenance={
            "source_split": "T0_valid",
            "data_root": str(data_root),
            "statistics_status": "synthetic_or_explicitly_labeled_smoke_only",
            "codec_hash": codec_hash_before,
        },
    )
    config = DiTTrainConfig(
        ratio=ratio,
        mode=candidate,
        max_steps=max_steps,
        seed=seed,
        amp=target_device.type == "cuda",
        output_root=str(output_root),
        data_hash="t0_valid_shape_smoke",
        codec_hash=codec_hash_before,
        stats_hash=stats.hash,
    )
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=256,
        vector_width=128,
        depth=4,
        heads=8,
        ffn_multiplier=4,
        dropout=0.0,
    ).to(target_device)
    trainer = DiTTrainer(
        model,
        adapter,
        config=config,
        statistics=stats,
        codec=codec,
        frame_encoder=codec.frame_encoder,
    )
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    train_step_seconds: list[float] = []

    def timed_train_step() -> tuple[dict[str, Any], float]:
        _synchronize(target_device)
        start = time.perf_counter()
        log = trainer.train_step(latent_batch)
        _synchronize(target_device)
        return log, max(time.perf_counter() - start, 1.0e-9)

    initial_log, elapsed_train = timed_train_step()
    train_step_seconds.append(elapsed_train)
    logs = [initial_log]
    while trainer.step < max_steps - 1 and time.perf_counter() - smoke_start < 15 * 60:
        log, elapsed_train = timed_train_step()
        logs.append(log)
        train_step_seconds.append(elapsed_train)
    if trainer.step > 100:
        raise RuntimeError("smoke exceeded the authorized optimizer-step bound")
    checkpoint = output_root / candidate / "smoke_resume.pt"
    trainer.save_checkpoint(checkpoint)
    recovered_stats = DiTTrainer.load_statistics_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    if recovered_stats.hash != stats.hash:
        raise RuntimeError("fresh checkpoint statistics recovery changed the statistics hash")
    statistics_fields = (
        "state_h_mean",
        "state_h_std",
        "detail_h_mean",
        "detail_h_std",
        "state_v_rms",
        "detail_v_rms",
    )
    if not all(
        torch.equal(
            getattr(recovered_stats, name), getattr(stats, name).detach().to(device="cpu")
        )
        for name in statistics_fields
    ):
        raise RuntimeError("fresh checkpoint statistics recovery changed statistic tensors")
    fresh_adapter = StateDetailLatentAdapter(
        codec_width=128, scalar_width=256, vector_width=128, ratio=ratio
    ).to(target_device)
    fresh_model = MolecularDiT(
        adapter=fresh_adapter,
        scalar_width=256,
        vector_width=128,
        depth=4,
        heads=8,
        ffn_multiplier=4,
        dropout=0.0,
    ).to(target_device)
    fresh_trainer = DiTTrainer(
        fresh_model,
        fresh_adapter,
        config=replace(config),
        statistics=None,
        codec=codec,
        frame_encoder=codec.frame_encoder,
    )
    fresh_payload = fresh_trainer.load_checkpoint(
        checkpoint, map_location=target_device
    )
    if fresh_trainer.statistics is None or fresh_trainer.statistics.hash != stats.hash:
        raise RuntimeError("fresh trainer did not reconstruct checkpoint statistics")
    fresh_recovery = {
        "step": fresh_trainer.step,
        "statistics_hash": fresh_trainer.statistics.hash,
        "statistics_tensor_count": len(statistics_fields),
        "checkpoint_schema": fresh_payload["schema"],
    }
    del fresh_payload, fresh_trainer, fresh_model, fresh_adapter
    resume_step = trainer.step
    trainer.load_checkpoint(checkpoint, map_location=target_device)
    log, elapsed_train = timed_train_step()
    logs.append(log)
    train_step_seconds.append(elapsed_train)
    resumed_step = trainer.step
    if resumed_step > 100:
        raise RuntimeError("checkpoint resume exceeded the authorized optimizer-step bound")
    train_activation_dtypes = dict(trainer.last_activation_dtypes)
    model.eval()
    _synchronize(target_device)
    sampling_start = time.perf_counter()
    with torch.no_grad(), trainer.autocast_context():
        generated8, meta8 = generate_state_detail_latent(
            model, adapter, latent_batch, stats, steps=8, seed=seed
        )
        generated16, meta16 = generate_state_detail_latent(
            model, adapter, latent_batch, stats, steps=16, seed=seed
        )
    _synchronize(target_device)
    sampling_seconds = max(time.perf_counter() - sampling_start, 1e-9)
    sampling_model_evaluations = 8 + 16
    with torch.no_grad():
        decoded8 = codec.decode(generated8)
        decoded16 = codec.decode(generated16)
    if not bool(torch.isfinite(_coordinates(decoded8)).all()) or not bool(torch.isfinite(_coordinates(decoded16)).all()):
        raise FloatingPointError("8-step or 16-step generated decode is non-finite")
    evaluation8 = evaluate_oracle_vs_generated(
        codec,
        oracle_latent=oracle_latent,
        generated_latent=generated8,
        batch=clip_batch,
        latent_flow_loss={
            name: logs[-1][f"{name}_loss"]
            for name in ("state_h", "detail_h", "state_v", "detail_v")
        },
        trunk_runtime={"sampling_model_evaluations": 8},
        history_frames=history_frames,
    )
    evaluation16 = evaluate_oracle_vs_generated(
        codec,
        oracle_latent=oracle_latent,
        generated_latent=generated16,
        batch=clip_batch,
        latent_flow_loss={
            name: logs[-1][f"{name}_loss"]
            for name in ("state_h", "detail_h", "state_v", "detail_v")
        },
        trunk_runtime={"sampling_model_evaluations": 16},
        history_frames=history_frames,
    )
    codec_hash_after = module_state_hash(codec)
    frame_hash_after = module_state_hash(codec.frame_encoder)
    elapsed = time.perf_counter() - smoke_start
    if elapsed > 15 * 60:
        raise RuntimeError("smoke exceeded the authorized 15-minute wall-time bound")
    result = {
        "schema_version": "pvb.dit.state_detail.smoke.v1",
        "candidate": candidate,
        "ratio": ratio,
        "status": "PASS",
        "execution_evidence_only": True,
        "scientific_claim": False,
        "data": {
            "root": str(data_root),
            "split": "T0_valid",
            "sample_ids": [str(item.get("sample_id", "")) for item in selected],
            "topology_ids": [str(value) for value in clip_batch_cpu.topology_id],
            "provenance": "existing T0 clip store; no T1 test access",
        },
        "contracts": {
            "codec_hash": codec_hash_before,
            "codec_hash_after": codec_hash_after,
            "frame_encoder_hash": frame_hash_before,
            "frame_encoder_hash_after": frame_hash_after,
            "statistics_hash": stats.hash,
            "recovered_statistics_hash": recovered_stats.hash,
            "statistics_tensor_count": len(statistics_fields),
            "adapter_hash": contract_hash(adapter.contract()),
            "model_hash": model.model_hash,
            "codec_unchanged": codec_hash_before == codec_hash_after,
            "frame_encoder_unchanged": frame_hash_before == frame_hash_after,
        },
        "losses": {
            "initial": initial_log,
            "final": logs[-1],
        },
        "gradients": _finite_gradients(model),
        "resume": {
            "checkpoint": str(checkpoint),
            "before": resume_step,
            "after": resumed_step,
            "fresh_process_recovery": fresh_recovery,
        },
        "generation": {
            "steps8": meta8,
            "steps16": meta16,
            "finite_decode": True,
            "raw_detail_h": None,
            "raw_detail_v": None,
            "observed_clamp_exact": bool(
                meta8["observed_clamp_exact"] and meta16["observed_clamp_exact"]
            ),
        },
        "evaluation": {
            "execution_evidence_only": True,
            "history_frames": int(history_frames),
            "steps8": evaluation8.as_dict(),
            "steps16": evaluation16.as_dict(),
        },
        "runtime": {
            "requested_device": str(device),
            "actual_device": str(target_device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_device_index": (
                int(torch.cuda.current_device()) if target_device.type == "cuda" else None
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None
            ),
            "amp_enabled": trainer.amp_enabled,
            "autocast_dtype": (
                str(torch.bfloat16) if trainer.amp_enabled else "disabled"
            ),
            "train_activation_dtypes": train_activation_dtypes,
            "wall_time_s": elapsed,
            "train_steps": len(train_step_seconds),
            "train_step_seconds_mean": sum(train_step_seconds) / len(train_step_seconds),
            "train_steps_per_s": len(train_step_seconds) / max(sum(train_step_seconds), 1e-9),
            "sampling_seconds": sampling_seconds,
            "sampling_model_evaluations": sampling_model_evaluations,
            "sampling_samples_per_s": latent_batch.batch_size / sampling_seconds,
            "trunk_tokens_per_s": (
                latent_batch.tokens * latent_batch.num_atoms * sampling_model_evaluations
            ) / sampling_seconds,
            "end_to_end_samples_per_s": latent_batch.batch_size / max(elapsed, 1e-9),
            "parameters": model.parameter_count,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)) if target_device.type == "cuda" else 0,
        },
        "warnings": [
            "Randomly initialized T0 shape smoke; not a trained codec or scientific DiT pilot.",
            "The smoke does not rank R2 versus R4.",
        ],
    }
    output_path = output_root / candidate / "smoke_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset.close()
    return result


def main() -> None:
    args = _parser().parse_args()
    result = run_smoke(
        candidate=args.candidate,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
        max_steps=args.max_steps,
        records=args.records,
        seed=args.seed,
        history_frames=args.history_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
