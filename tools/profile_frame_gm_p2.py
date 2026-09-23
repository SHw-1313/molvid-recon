"""Profile one P2 arm on the largest real 192-system training record."""

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
from molvid.training.joint import FrameJointTrainer, JointLossConfig


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _arm_from_config(path: Path) -> str:
    stem = path.stem
    for prefix in ("frame_gm_p2_formal_", "frame_gm_p2_pilot_"):
        if stem.startswith(prefix) and stem.endswith("_260923"):
            arm = stem.removeprefix(prefix).removesuffix("_260923")
            if arm in {"B0", "G", "M", "GM"}:
                return arm
    raise ValueError("profile config filename does not identify a P2 arm")


def main() -> int:
    args = _args()
    raw = load_config(args.config, schema="molvid.frame_joint.train.v1")
    _validate_frame_joint_config(raw)
    data_config = dict(raw["data"])
    if data_config.get("train_systems"):
        raise ValueError("largest-sample profile requires the full-data formal arm config")
    arm = _arm_from_config(args.config)
    device = configure_device("cuda", deterministic=False)
    seed = int(raw["training"]["seed"])
    seed_all(seed)
    train, valid, data_hash, normalization_hash, relationship, close = _open_data(data_config)
    del valid
    try:
        specs = get_clip_specs(train)
        largest = max(specs, key=lambda item: (item.effective_tokens, item.sample_id))
        record = train[largest.index]
        if str(record["sample_id"]) != largest.sample_id:
            raise RuntimeError("largest-record specification disagrees with the dataset")
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
        loss_config = JointLossConfig.resolve(parent["contracts"]["loss"])
        rates = raw["training"]["stages"][0]["learning_rates"]
        trainer = FrameJointTrainer(
            model,
            loss_config=loss_config,
            learning_rates=rates,
            weight_decay=float(raw["training"].get("weight_decay", 0.01)),
            grad_clip=float(raw["training"].get("grad_clip", 1.0)),
            stage="continuation",
            amp=str(raw["training"].get("precision", "fp32")) == "bf16",
            data_hash=data_hash,
            teacher_artifact_sha256=artifact.report.source_sha256,
            schedule_contract={"profile_only": True},
        )
        warm = trainer.load_warm_start(args.parent, expected_sha256=args.parent_sha256)
        batch = collate_clip_records([record])
        history = 4 if str(record["time_bucket_id"]).startswith("fixed_history_dt_") else 8
        prepared = prepare_frame_joint_batch(
            model.target_teacher,
            batch,
            device=device,
            normalizer=model,
            history_frames=history,
        )
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        metrics = trainer.train_step(prepared, generator=generator)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        result = {
            "schema": "molvid.frame_gm.p2_largest_sample_profile.v1",
            "arm": arm,
            "config": str(args.config.resolve()),
            "parent": str(args.parent.resolve()),
            "parent_sha256": args.parent_sha256,
            "sample_dataset_index": int(largest.index),
            "sample_id": largest.sample_id,
            "atoms": int(largest.atoms),
            "frames": int(largest.frames),
            "effective_atom_frames": int(largest.effective_tokens),
            "valid_frames": int(batch.frame_mask.sum()),
            "history_frames": history,
            "step_seconds": seconds,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "incremental_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device) - baseline_allocated
            ),
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
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
        atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
    finally:
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
