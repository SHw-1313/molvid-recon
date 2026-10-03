"""Bounded CUDA profile of selected frozen formal batches without formal exposure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from molvid.checkpoints import load_codec_artifact
from molvid.data.learned_memory_formal import RawMemoryDataset
from molvid.data.sampling import TrajectoryCappedBatchSampler
from molvid.model import FrameJointModel
from molvid.runtime import atomic_write_json, configure_device, seed_all, sha256_file
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig
from molvid.training.learned_memory_formal import (
    _load_statistics,
    _scheduled_rates,
    _validate_config,
    _verify_release,
)
from molvid.training.learned_memory_schedule import (
    learned_memory_execution_manifest,
    learned_memory_execution_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, action="append", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.batch_index) <= 4 or len(set(args.batch_index)) != len(args.batch_index):
        raise ValueError("profile requires one to four distinct batch indices")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.code_commit:
        raise ValueError("executing code commit differs from fixed profile candidate")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "molvid", "tools", "tests"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("profile CUDA UUID differs from the explicit assignment")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(raw)
    _verify_release(raw)
    if raw["arm"] != "M-short" or raw["training"]["max_updates"] is not None:
        raise ValueError("profile must use the frozen formal M-short config")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result: dict[str, object] = {
        "schema": "molvid.learned_memory.formal_batch_profile.v1",
        "status": "running",
        "formal": False,
        "formal_exposure_counted": False,
        "code_commit": actual_commit,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "gpu_uuid": args.gpu_uuid,
        "requested_batch_indices": args.batch_index,
    }
    atomic_write_json(args.output / "profile.json", result)
    try:
        training = raw["training"]
        device = configure_device("cuda", deterministic=bool(training["deterministic"]))
        if torch.cuda.device_count() != 1:
            raise RuntimeError("bounded profile requires exactly one visible CUDA device")
        seed = int(training["seed"])
        seed_all(seed)
        dataset = RawMemoryDataset(
            manifest_root=Path(raw["data"]["manifest_root"]),
            raw_root=Path(raw["data"]["raw_root"]),
            offset_root=Path(raw["data"]["xtc_offset_root"]),
            split="train",
        )
        if dataset.identity["data_hash"] != raw["data"]["data_hash"]:
            raise ValueError("profile raw dataset identity differs")
        sampler = TrajectoryCappedBatchSampler(
            dataset,
            max_tokens=int(training["max_atom_frames_per_gpu"]),
            clips_per_trajectory=int(training["clips_per_trajectory"]),
            seed=seed,
            shuffle=True,
            num_replicas=1,
            rank=0,
            drop_last=False,
            oversize_policy="partition",
        )
        sampler.set_epoch(0)
        batches = list(iter(sampler))
        execution = learned_memory_execution_schedule(
            batches,
            sampler.specs,
            arm=raw["arm"],
            seed=seed,
            epoch=0,
            cadence=int(raw["sampled_auxiliary"]["cadence"]),
        )
        manifest = learned_memory_execution_manifest(execution)
        for index in args.batch_index:
            if not 0 <= index < len(execution) or not execution[index].auxiliary:
                raise ValueError(f"batch {index} is not a scheduled formal auxiliary batch")

        artifact = load_codec_artifact(
            raw["codec"]["checkpoint"],
            expected_sha256=raw["codec"]["sha256"],
            device=device,
        )
        codec = artifact.model.eval()
        codec.requires_grad_(False)
        statistics = _load_statistics(
            {**raw["statistics"], "codec_sha256": raw["codec"]["sha256"]},
            device,
        )
        model = FrameJointModel.from_codec(
            codec,
            statistics,
            **{key: raw["model"][key] for key in (
                "scalar_width", "vector_width", "depth", "heads",
                "geometry_enabled", "motion_enabled", "history_backend", "analysis_mode",
            )},
        ).to(device)
        rates = {name: float(value) for name, value in training["learning_rates"].items()}
        trainer = FrameJointTrainer(
            model,
            loss_config=JointLossConfig.resolve(raw["loss"]),
            learning_rates=rates,
            weight_decay=float(training["weight_decay"]),
            grad_clip=float(training["grad_clip"]),
            stage="continuation",
            amp=True,
            data_hash=dataset.identity["data_hash"],
            teacher_artifact_sha256=raw["codec"]["sha256"],
            sampled_config=SampledAuxiliaryConfig.resolve(raw["sampled_auxiliary"]),
        )
        warm = trainer.load_warm_start(
            raw["parent"]["path"], expected_sha256=raw["parent"]["sha256"]
        )
        teacher_hash = module_state_hash(model.target_teacher)
        teacher_calls: list[int] = []
        memory_calls: list[int] = []
        teacher_hook = model.target_teacher.register_forward_pre_hook(
            lambda _module, values: teacher_calls.append(int(values[0].frames))
        )
        memory_hook = model.history_encoder.register_forward_pre_hook(
            lambda _module, _values: memory_calls.append(1)
        )
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        sampled_generator = torch.Generator(device=device).manual_seed(seed + 2000)
        rows = []
        for profile_position, batch_index in enumerate(args.batch_index):
            entry = execution[batch_index]
            indices = batches[batch_index]
            raw_history, anchors, future = dataset.load_inputs(
                indices,
                history_span_ps=entry.history_span_ps,
                observed_dt_ps=entry.observed_dt_ps,
                query_lag_ps=entry.query_lag_ps,
            )
            before_teacher = len(teacher_calls)
            before_memory = len(memory_calls)
            torch.cuda.reset_peak_memory_stats()
            free_before_prepare, total_memory = torch.cuda.mem_get_info()
            prepare_started = time.perf_counter()
            prepared = prepare_frame_joint_batch(
                model.target_teacher,
                future,
                device=device,
                normalizer=model,
                history_frames=2,
                raw_history=raw_history,
                geometry_anchors=anchors,
            )
            torch.cuda.synchronize()
            prepare_seconds = time.perf_counter() - prepare_started
            prepare_peak = int(torch.cuda.max_memory_allocated())
            trainer.set_learning_rates(_scheduled_rates(rates, batch_index, len(execution)))
            torch.cuda.reset_peak_memory_stats()
            update_started = time.perf_counter()
            metrics = trainer.train_step(
                prepared,
                generator=generator,
                sampled_generator=sampled_generator,
                auxiliary_active=True,
                diagnose_gradients=False,
            )
            torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            calls = teacher_calls[before_teacher:]
            preparations = sum(memory_calls[before_memory:])
            if calls != [2, 12] or preparations != 1:
                raise RuntimeError("teacher/memory call contract changed during profile")
            missing = [
                name for name, parameter in model.history_encoder.named_parameters()
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            ]
            if missing:
                raise RuntimeError(f"profile learned-memory gradients missing/nonfinite: {missing}")
            free_after, _ = torch.cuda.mem_get_info()
            rows.append({
                "profile_position": profile_position,
                "batch_index": batch_index,
                "sample_ids": [dataset[index].sample_id for index in indices],
                "atoms": int(anchors.atom_count),
                "teacher_atom_frames": int(14 * anchors.atom_count),
                "raw_valid_atom_frames": int(
                    raw_history.frame_mask.index_select(0, raw_history.topology.abid).sum()
                ),
                "history_span_ps": entry.history_span_ps,
                "observed_dt_ps": entry.observed_dt_ps,
                "query_lag_ps": entry.query_lag_ps,
                "teacher_frame_calls": calls,
                "memory_preparations": preparations,
                "prepare_seconds": prepare_seconds,
                "prepare_peak_allocated_bytes": prepare_peak,
                "update_seconds": update_seconds,
                "update_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "device_free_before_prepare_bytes": int(free_before_prepare),
                "device_free_after_update_bytes": int(free_after),
                "device_total_bytes": int(total_memory),
                "loss": metrics["loss"],
                "grad_norm": metrics["grad_norm"],
                "sampled_streamed_backward": metrics.get("sampled_streamed_backward"),
                "condition_backward": metrics.get("sampled_diagnostics", {}).get("condition_backward"),
                "history_gradients_finite": True,
            })
            del prepared, raw_history, anchors, future
        teacher_hook.remove()
        memory_hook.remove()
        if module_state_hash(model.target_teacher) != teacher_hash:
            raise RuntimeError("frozen teacher changed during profile")
        result.update({
            "status": "completed",
            "execution_schedule_sha256": manifest["sha256"],
            "paired_schedule_sha256": manifest["paired_schedule_sha256"],
            "warm_start_report": warm.as_dict(),
            "rows": rows,
            "peak_allocated_bytes": max(row["update_peak_allocated_bytes"] for row in rows),
            "elapsed_seconds": time.perf_counter() - started,
        })
    except BaseException as error:
        result.update({
            "status": "failed",
            "failure": {"type": type(error).__name__, "message": str(error)},
            "elapsed_seconds": time.perf_counter() - started,
        })
        atomic_write_json(args.output / "profile.json", result)
        raise
    atomic_write_json(args.output / "profile.json", result)


if __name__ == "__main__":
    main()
