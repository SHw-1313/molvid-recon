"""Profile one actual-source sampled update on the largest real train record."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from molvid.checkpoints import load_codec_artifact
from molvid.cli.train_frame_joint import (
    _load_statistics,
    _open_data,
    _validate_frame_joint_config,
)
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.model import FrameJointModel
from molvid.runtime import atomic_write_json, configure_device, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import (
    FrameJointTrainer,
    JointLossConfig,
    SampledAuxiliaryConfig,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main() -> int:
    args = _arguments()
    raw = load_config(args.config, schema="molvid.frame_joint.train.v1")
    _validate_frame_joint_config(raw)
    sampled = raw.get("sampled_auxiliary", {})
    if not sampled.get("enabled", False):
        raise ValueError("P3 sampled profile requires sampled_auxiliary.enabled=true")
    if raw["data"].get("train_systems"):
        raise ValueError("largest-sample profile requires the full 192-system data config")
    training = raw["training"]
    device = configure_device(
        str(training.get("device", "cuda")),
        deterministic=bool(training.get("deterministic", False)),
    )
    seed = int(training["seed"])
    seed_all(seed)
    train, valid, data_hash, normalization_hash, relationship, close = _open_data(
        raw["data"]
    )
    del valid
    try:
        specs = get_clip_specs(train)
        largest = max(specs, key=lambda item: (item.effective_tokens, item.sample_id))
        record = train[largest.index]
        artifact = load_codec_artifact(
            raw["codec"]["checkpoint"],
            expected_sha256=str(raw["codec"]["sha256"]),
            device=device,
        )
        statistics = _load_statistics(
            raw["statistics"],
            device=device,
            normalization_data_hash=normalization_hash,
            codec_sha256=artifact.report.source_sha256,
        )
        model_config = raw["model"]
        model = FrameJointModel.from_codec(
            artifact.model,
            statistics,
            scalar_width=int(model_config["scalar_width"]),
            vector_width=int(model_config["vector_width"]),
            depth=int(model_config["depth"]),
            heads=int(model_config["heads"]),
            geometry_enabled=bool(model_config.get("geometry_enabled", False)),
            motion_enabled=bool(model_config.get("motion_enabled", False)),
        ).to(device)
        parent = torch.load(args.parent, map_location="cpu", weights_only=False)
        scales = {
            "residue_rmsf_A": 1.0,
            "displacement_squared_A2": 1.0,
            "internal_distance_increment_A": 1.0,
            "internal_distance_increment_product_A2": 1.0,
        }
        sampled_config = SampledAuxiliaryConfig.resolve({
            "enabled": True,
            "cadence": 8,
            "draws": 2,
            "euler_steps": 4,
            "activation_checkpoint": bool(sampled.get("activation_checkpoint", True)),
            "energy_weight": 1.0e-3,
            "observed_bond_weight": 1.0e-3,
            "feature_scales": scales,
        })
        rates = training["stages"][0]["learning_rates"]
        trainer = FrameJointTrainer(
            model,
            loss_config=JointLossConfig.resolve(parent["contracts"]["loss"]),
            learning_rates=rates,
            weight_decay=float(training.get("weight_decay", 0.01)),
            grad_clip=float(training.get("grad_clip", 1.0)),
            stage="continuation",
            amp=str(training.get("precision", "fp32")) == "bf16",
            data_hash=data_hash,
            teacher_artifact_sha256=artifact.report.source_sha256,
            schedule_contract={"profile_only": True},
            sampled_config=sampled_config,
        )
        warm = trainer.load_warm_start(
            args.parent, expected_sha256=args.parent_sha256
        )
        batch = collate_clip_records([record])
        history = (
            4 if str(record["time_bucket_id"]).startswith("fixed_history_dt_") else 8
        )
        prepared = prepare_frame_joint_batch(
            model.target_teacher,
            batch,
            device=device,
            normalizer=model,
            history_frames=history,
        )
        main_generator = torch.Generator(device=device).manual_seed(seed + 1000)
        sampled_generator = torch.Generator(device=device).manual_seed(seed + 2000)
        trainer.step = trainer.successful_updates = 7
        teacher_before = module_state_hash(model.target_teacher)
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        metrics = trainer.train_step(
            prepared,
            generator=main_generator,
            sampled_generator=sampled_generator,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients = {
            "history_encoder": _gradient_norm(model.history_encoder),
            "dit": _gradient_norm(model.dit),
            "decoder": _gradient_norm(model.decoder),
            "target_teacher": _gradient_norm(model.target_teacher),
        }
        teacher_after = module_state_hash(model.target_teacher)
        result = {
            "schema": "molvid.frame_gm.p3_largest_sample_profile.v1",
            "config": str(args.config.resolve()),
            "parent": str(args.parent.resolve()),
            "parent_sha256": args.parent_sha256,
            "sample_dataset_index": int(largest.index),
            "sample_id": largest.sample_id,
            "atoms": int(largest.atoms),
            "frames": int(largest.frames),
            "effective_atom_frames": int(largest.effective_tokens),
            "history_frames": history,
            "sampled_draws": sampled_config.draws,
            "sampled_euler_steps": sampled_config.euler_steps,
            "activation_checkpoint": sampled_config.activation_checkpoint,
            "step_seconds": elapsed,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "component_gradient_norms_after_clip": gradients,
            "teacher_state_hash_before": teacher_before,
            "teacher_state_hash_after": teacher_after,
            "teacher_state_unchanged": teacher_before == teacher_after,
            "warm_start_loaded_tensors": len(warm.loaded),
            "warm_start_initialized_tensors": len(warm.initialized),
            "data_hash": data_hash,
            "normalization_source_data_hash": normalization_hash,
            "data_relationship": relationship,
            "metrics": metrics,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "test_opened": False,
        }
        if (
            metrics["sampled_branch_active"] is not True
            or any(gradients[name] <= 0.0 for name in ("history_encoder", "dit", "decoder"))
            or gradients["target_teacher"] != 0.0
            or not result["teacher_state_unchanged"]
        ):
            raise RuntimeError(result)
        atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
    finally:
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
