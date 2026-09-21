"""Check paired bond continuations fork exactly from one parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import TrajectoryCappedBatchSampler
from molvid.data.store import ClipMMapDataset
from molvid.runtime import atomic_write_json, canonical_hash, configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.joint import FrameJointTrainer, JointLossConfig


RATES = {"history_encoder": 2.5e-5, "dit": 5.0e-5, "decoder": 1.25e-5}
SCHEDULE = {
    "schema_version": "molvid.frame_joint.schedule.v1",
    "stages": [
        {"name": "continuation", "updates": 5, "learning_rates": RATES},
    ],
    "history_order": [4, 8],
    "learning_rate_schedule": "linear_warmup_5pct_cosine_to_0.1.v1",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--train-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _arm(
    args: argparse.Namespace,
    device: torch.device,
    loss: Mapping[str, Any],
) -> tuple[FrameJointTrainer, torch.Generator, Mapping[str, Any]]:
    loaded = load_frame_joint_inference(
        args.parent,
        expected_sha256=args.parent_sha256,
        codec_path=args.codec,
        codec_sha256=args.codec_sha256,
        device=device,
    )
    parent_contracts = loaded["payload"]["contracts"]
    optimization = parent_contracts["optimization"]
    trainer = FrameJointTrainer(
        loaded["model"],
        loss_config=JointLossConfig.resolve(loss),
        learning_rates=RATES,
        weight_decay=float(optimization["weight_decay"]),
        grad_clip=float(optimization["grad_clip"]),
        stage="continuation",
        amp=bool(optimization["amp_bfloat16"]),
        data_hash=str(parent_contracts["data_hash"]),
        teacher_artifact_sha256=str(parent_contracts["teacher_artifact_sha256"]),
        schedule_contract=SCHEDULE,
    )
    generator = torch.Generator(device=device)
    parent = trainer.load_continuation_parent(
        args.parent,
        generator=generator,
        expected_sha256=args.parent_sha256,
    )
    trainer.set_learning_rates(RATES)
    return trainer, generator, parent


def _tensor_tree_max_abs(left: Any, right: Any) -> float:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return float("inf")
        if left.dtype.is_floating_point:
            return float((left.detach().cpu() - right.detach().cpu()).abs().max())
        return 0.0 if torch.equal(left.detach().cpu(), right.detach().cpu()) else float("inf")
    if isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right):
        return max((_tensor_tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
        return max((_tensor_tree_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0 if left == right else float("inf")


def main() -> int:
    args = _arguments()
    device = configure_device(args.device, deterministic=True)
    preview = torch.load(args.parent, map_location="cpu", weights_only=False)
    keep_loss = dict(preview["contracts"]["loss"])
    release_loss = dict(keep_loss)
    release_loss.update({
        "bond_enabled": False,
        "generated_bond": 0.0,
        "clean_bond": 0.0,
        "near_bond": 0.0,
    })
    keep, keep_generator, keep_parent = _arm(args, device, keep_loss)
    release, release_generator, release_parent = _arm(args, device, release_loss)
    model_error = _tensor_tree_max_abs(
        keep.base_model.state_dict(), release.base_model.state_dict()
    )
    optimizer_error = _tensor_tree_max_abs(
        keep.optimizer.state_dict(), release.optimizer.state_dict()
    )
    generator_start_equal = torch.equal(
        keep_generator.get_state().cpu(), release_generator.get_state().cpu()
    )
    cursor_equal = keep_parent["cursor"] == release_parent["cursor"]

    dataset = ClipMMapDataset(args.train_store)
    try:
        sampler = TrajectoryCappedBatchSampler(
            dataset,
            max_tokens=40_000,
            clips_per_trajectory=1,
            seed=20260920,
            shuffle=True,
            num_replicas=1,
            rank=0,
            drop_last=False,
        )
        sampler.set_epoch(0)
        batch = collate_clip_records([dataset[0]])
        keep_batch = prepare_frame_joint_batch(
            keep.base_model.target_teacher,
            batch,
            device=device,
            normalizer=keep.base_model,
            history_frames=8,
        )
        release_batch = prepare_frame_joint_batch(
            release.base_model.target_teacher,
            batch,
            device=device,
            normalizer=release.base_model,
            history_frames=8,
        )
        keep_metrics = keep.train_step(keep_batch, generator=keep_generator)
        release_metrics = release.train_step(release_batch, generator=release_generator)
    finally:
        dataset.close()
    generator_end_equal = torch.equal(
        keep_generator.get_state().cpu(), release_generator.get_state().cpu()
    )
    child_checkpoint = args.output / "continuation_child_resume_check.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    keep.save_checkpoint(
        child_checkpoint,
        cursor={
            "epoch": 0,
            "batch_index": 1,
            "sampler_state": sampler.state_dict(),
            "global_batch_schedule_hash": canonical_hash(sampler.global_batches),
        },
        generator=keep_generator,
    )
    child_sha256 = sha256_file(child_checkpoint)
    child_model = keep.base_model.state_dict()
    child_optimizer = keep.optimizer.state_dict()
    resumed, resumed_generator, _ = _arm(args, device, keep_loss)
    resumed.load_checkpoint(
        child_checkpoint,
        generator=resumed_generator,
        expected_sha256=child_sha256,
    )
    child_resume_model_error = _tensor_tree_max_abs(
        child_model, resumed.base_model.state_dict()
    )
    child_resume_optimizer_error = _tensor_tree_max_abs(
        child_optimizer, resumed.optimizer.state_dict()
    )
    child_resume_generator_equal = torch.equal(
        keep_generator.get_state().cpu(), resumed_generator.get_state().cpu()
    )
    child_resume_parent_equal = resumed.continuation_parent == keep.continuation_parent
    resumed.schedule_contract = {
        **SCHEDULE,
        "history_order": [8, 4],
    }
    try:
        resumed.load_checkpoint(
            child_checkpoint,
            generator=resumed_generator,
            expected_sha256=child_sha256,
        )
    except ValueError as error:
        schedule_mismatch_rejected = "contracts differ" in str(error)
    else:
        schedule_mismatch_rejected = False
    result = {
        "schema_version": "molvid.frame_joint.continuation_check.v1",
        "parent_sha256": args.parent_sha256,
        "parent_step": int(keep_parent["step"]),
        "sample_id": batch.sample_id[0],
        "history_frames": 8,
        "starting_model_max_abs": model_error,
        "starting_optimizer_max_abs": optimizer_error,
        "cursor_equal": cursor_equal,
        "generator_start_equal": generator_start_equal,
        "generator_end_equal": generator_end_equal,
        "child_checkpoint_sha256": child_sha256,
        "child_resume_model_max_abs": child_resume_model_error,
        "child_resume_optimizer_max_abs": child_resume_optimizer_error,
        "child_resume_generator_equal": child_resume_generator_equal,
        "child_resume_parent_equal": child_resume_parent_equal,
        "schedule_mismatch_rejected": schedule_mismatch_rejected,
        "flow_time_keep": keep_metrics["flow_time_mean"],
        "flow_time_release": release_metrics["flow_time_mean"],
        "keep_weighted_bonds": {
            name: keep_metrics[f"weighted_{name}"]
            for name in ("generated_bond", "clean_bond", "near_bond")
        },
        "release_weighted_bonds": {
            name: release_metrics[f"weighted_{name}"]
            for name in ("generated_bond", "clean_bond", "near_bond")
        },
    }
    result["passed"] = bool(
        model_error == 0
        and optimizer_error == 0
        and cursor_equal
        and generator_start_equal
        and generator_end_equal
        and child_resume_model_error == 0
        and child_resume_optimizer_error == 0
        and child_resume_generator_equal
        and child_resume_parent_equal
        and schedule_mismatch_rejected
        and keep_metrics["flow_time_mean"] == release_metrics["flow_time_mean"]
        and all(value == 0 for value in result["release_weighted_bonds"].values())
    )
    atomic_write_json(args.output / "continuation_check.json", result)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError("paired continuation check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
