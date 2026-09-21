"""Fit statistics and train the staged Frame Joint v1 model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..checkpoints import capture_rng_state, frame_target_provenance, load_codec_artifact
from ..codec.frame import FrozenFrameTeacher
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.manifest import load_datasets
from ..data.sampling import TaskAwareClipBatchSampler, TrajectoryCappedBatchSampler
from ..data.store import ClipMMapDataset
from ..latent.statistics import FrameLatentStatistics
from ..model import FrameJointModel
from ..runtime import atomic_write_json, canonical_hash, configure_device, seed_all, sha256_file
from ..training.batches import prepare_batch_then_to_device, prepare_frame_joint_batch
from ..training.joint import (
    FrameJointTrainer,
    JointLossConfig,
    STAGES,
    calibrate_generated_bond_weight,
)


SCHEMA = "molvid.frame_joint.train.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-statistics", action="store_true")
    parser.add_argument("--fit-statistics-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--continuation-parent", type=Path)
    parser.add_argument("--continuation-parent-sha256")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _distributed_device(training: Mapping[str, Any]) -> tuple[torch.device, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Frame Joint DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = configure_device(
            f"cuda:{local_rank}", deterministic=bool(training.get("deterministic", False))
        )
    else:
        device = configure_device(
            str(training.get("device", "cuda")),
            deterministic=bool(training.get("deterministic", False)),
        )
    return device, rank, local_rank, world


def _direct_data_hash(train: ClipMMapDataset, valid: ClipMMapDataset) -> str:
    return canonical_hash({
        "schema": "molvid.frame_joint.direct_clip_stores.v1",
        "train_index_sha256": sha256_file(train.root / "index.txt"),
        "valid_index_sha256": sha256_file(valid.root / "index.txt"),
        "train_count": len(train),
        "valid_count": len(valid),
        "test_opened": False,
    })


def _open_data(data: Mapping[str, Any]):
    manifest_root = data.get("manifest_root")
    if manifest_root:
        splits = load_datasets(manifest_root, train_view=str(data.get("train_view", "train")))
        return splits.train, splits.valid, splits.data_hash, splits.close
    train = ClipMMapDataset(data["train_store"])
    valid = ClipMMapDataset(data["valid_store"])
    return train, valid, _direct_data_hash(train, valid), lambda: (train.close(), valid.close())


def _statistics_stream(
    teacher: FrozenFrameTeacher,
    dataset: ClipMMapDataset,
    *,
    device: torch.device,
    max_tokens: int,
):
    sampler = TaskAwareClipBatchSampler(
        dataset,
        max_tokens=max_tokens,
        seed=0,
        shuffle=False,
        replacement=False,
        oversize_policy="partition",
    )
    for indices in sampler.global_batches:
        batch = collate_clip_records([dataset[index] for index in indices])
        coordinate = prepare_batch_then_to_device(teacher, batch, device)
        latent, _ = teacher(coordinate)
        yield latent


def _fit_statistics(
    artifact: Any,
    dataset: ClipMMapDataset,
    *,
    device: torch.device,
    path: Path,
    data_hash: str,
    max_tokens: int,
) -> FrameLatentStatistics:
    teacher = FrozenFrameTeacher.from_codec(artifact.model).to(device).eval()
    statistics = FrameLatentStatistics.fit(
        _statistics_stream(teacher, dataset, device=device, max_tokens=max_tokens),
        provenance={
            "scope": "complete_training_store",
            "training_clip_count": len(dataset),
            "data_hash": data_hash,
            "codec_checkpoint_sha256": artifact.report.source_sha256,
            "vector_mean_subtraction": False,
            "vector_xyz_shared_scale": True,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statistics.state_dict(), path)
    atomic_write_json(path.with_suffix(".json"), {
        "statistics_hash": statistics.hash,
        "file_sha256": sha256_file(path),
        "provenance": dict(statistics.provenance),
    })
    return statistics.to(device)


def _load_statistics(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    data_hash: str,
    codec_sha256: str,
) -> FrameLatentStatistics:
    path = Path(config["path"])
    expected_file = str(config.get("sha256", ""))
    if expected_file and expected_file != sha256_file(path):
        raise ValueError("frame statistics file SHA-256 differs")
    statistics = FrameLatentStatistics.from_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    expected_contract = str(config.get("statistics_hash", ""))
    if expected_contract and expected_contract != statistics.hash:
        raise ValueError("frame statistics contract hash differs")
    if statistics.provenance.get("data_hash") != data_hash:
        raise ValueError("frame statistics were not fit on this training data")
    if statistics.provenance.get("codec_checkpoint_sha256") != codec_sha256:
        raise ValueError("frame statistics target encoder differs")
    return statistics.to(device)


def _stages(training: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = training.get("stages")
    if not isinstance(values, list) or not values:
        raise ValueError("training.stages must be a nonempty list")
    result: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, Mapping) or str(item.get("name")) not in STAGES:
            raise ValueError("each stage requires a supported name")
        updates = int(item.get("updates", 0))
        rates = item.get("learning_rates")
        if updates < 1 or not isinstance(rates, Mapping):
            raise ValueError("each stage requires positive updates and learning_rates")
        resolved_rates = {
            name: float(rates[name]) for name in ("history_encoder", "dit", "decoder")
        }
        if any(not math.isfinite(value) or value < 0 for value in resolved_rates.values()):
            raise ValueError("stage learning rates must be finite and non-negative")
        result.append({"name": str(item["name"]), "updates": updates, "learning_rates": resolved_rates})
    return tuple(result)


def _stage_at(stages: Sequence[Mapping[str, Any]], step: int) -> tuple[Mapping[str, Any], int]:
    offset = 0
    for stage in stages:
        end = offset + int(stage["updates"])
        if step < end:
            return stage, step - offset
        offset = end
    return stages[-1], int(stages[-1]["updates"]) - 1


def _scheduled_rates(stage: Mapping[str, Any], position: int) -> dict[str, float]:
    updates = int(stage["updates"])
    warmup = max(1, math.ceil(0.05 * updates))
    if position < warmup:
        factor = float(position + 1) / warmup
    else:
        progress = float(position - warmup) / max(1, updates - warmup - 1)
        factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return {name: float(value) * factor for name, value in stage["learning_rates"].items()}


def _sampler(
    dataset: ClipMMapDataset,
    training: Mapping[str, Any],
    *,
    seed: int,
    rank: int,
    world: int,
) -> TrajectoryCappedBatchSampler:
    return TrajectoryCappedBatchSampler(
        dataset,
        max_tokens=int(training["max_atom_frames_per_gpu"]),
        clips_per_trajectory=int(training["clips_per_trajectory"]),
        seed=seed,
        shuffle=True,
        num_replicas=world,
        rank=rank,
        drop_last=world > 1,
        time_bucket_weights=training.get("time_bucket_weights"),
    )


def _calibration_stream(
    model: FrameJointModel,
    dataset: ClipMMapDataset,
    training: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
    histories: Sequence[int],
):
    sampler = _sampler(dataset, training, seed=seed + 7000, rank=0, world=1)
    yielded = 0
    epoch = 0
    while yielded < 16:
        sampler.set_epoch(epoch)
        for indices in sampler:
            cpu_batch = collate_clip_records([dataset[index] for index in indices])
            history = int(histories[yielded % len(histories)])
            yield prepare_frame_joint_batch(
                model.target_teacher,
                cpu_batch,
                device=device,
                normalizer=model,
                history_frames=history,
            )
            yielded += 1
            if yielded == 16:
                return
        epoch += 1


def _trajectory(sample_id: str) -> str:
    value = str(sample_id).rsplit("_w", 1)[0]
    return re.sub(r"_dt_[0-9]+(?:\.[0-9]+)?ps$", "", value)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
    ).strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and args.continuation_parent:
        raise ValueError("ordinary resume and continuation parent are mutually exclusive")
    if args.continuation_parent and not args.continuation_parent_sha256:
        raise ValueError("continuation parent requires its SHA-256")
    if args.fit_statistics_only:
        args.fit_statistics = True
    raw = load_config(args.config, schema=SCHEMA)
    for section in ("data", "codec", "statistics", "model", "loss", "training"):
        if not isinstance(raw.get(section), Mapping):
            raise ValueError(f"configuration requires {section!r}")
    training = dict(raw["training"])
    if args.output_root is not None:
        training["output_root"] = str(args.output_root)
    device, rank, local_rank, world = _distributed_device(training)
    if args.fit_statistics and world != 1:
        raise ValueError("fit statistics in one CUDA process before launching DDP")
    seed = int(training["seed"])
    seed_all(seed + rank)
    train, valid, data_hash, close_data = _open_data(raw["data"])
    del valid  # Validation is opened for identity only; this command never opens test.
    try:
        codec_config = raw["codec"]
        artifact = load_codec_artifact(
            codec_config["checkpoint"],
            expected_sha256=str(codec_config["sha256"]),
            device=device,
        )
        codec = artifact.model.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        stats_path = Path(raw["statistics"]["path"])
        if args.fit_statistics:
            statistics = _fit_statistics(
                artifact,
                train,
                device=device,
                path=stats_path,
                data_hash=data_hash,
                max_tokens=int(raw["statistics"].get("max_atom_frames_per_gpu", 80000)),
            )
            print(json.dumps({
                "statistics": str(stats_path),
                "statistics_hash": statistics.hash,
                "sha256": sha256_file(stats_path),
            }, sort_keys=True))
            if args.fit_statistics_only:
                return 0
        else:
            statistics = _load_statistics(
                raw["statistics"],
                device=device,
                data_hash=data_hash,
                codec_sha256=artifact.report.source_sha256,
            )

        seed_all(seed + rank)
        model_config = raw["model"]
        model = FrameJointModel.from_codec(
            codec,
            statistics,
            scalar_width=int(model_config.get("scalar_width", 256)),
            vector_width=int(model_config.get("vector_width", 128)),
            depth=int(model_config.get("depth", 4)),
            heads=int(model_config.get("heads", 8)),
        ).to(device)
        stages = _stages(training)
        histories = tuple(int(value) for value in training.get("history_order", (4, 8)))
        if not histories or any(value < 1 for value in histories):
            raise ValueError("history_order must contain positive H values")
        schedule_contract = {
            "schema_version": "molvid.frame_joint.schedule.v1",
            "stages": [
                {
                    "name": str(stage["name"]),
                    "updates": int(stage["updates"]),
                    "learning_rates": dict(stage["learning_rates"]),
                }
                for stage in stages
            ],
            "history_order": list(histories),
            "learning_rate_schedule": "linear_warmup_5pct_cosine_to_0.1.v1",
        }
        total_updates = sum(int(stage["updates"]) for stage in stages)
        if args.max_steps is not None:
            total_updates = min(total_updates, int(args.max_steps))
        initial_stage = stages[0]
        resume_payload = None
        continuation_preview = None
        if args.resume:
            resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            initial_stage = {**stages[0], "name": str(resume_payload["contracts"]["stage"])}
        elif args.continuation_parent:
            continuation_preview = torch.load(
                args.continuation_parent, map_location="cpu", weights_only=False
            )
            if any(stage["name"] not in {"continuation", "frozen_decoder"} for stage in stages):
                raise ValueError(
                    "a continuation config may contain only continuation or frozen_decoder stages"
                )
            initial_stage = stages[0]
        wrapped: torch.nn.Module = model
        if world > 1:
            wrapped = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                # The frozen teacher owns rank-local topology caches and the
                # trainable model has no running-stat buffers.
                broadcast_buffers=False,
                # Stages deliberately freeze whole components after DDP is
                # constructed, then enable them again at later boundaries.
                find_unused_parameters=True,
            )
        loss_mapping = dict(raw["loss"])
        inherit_parent_loss = bool(loss_mapping.pop("inherit_parent", False))
        if resume_payload is not None:
            loss_mapping = dict(resume_payload["contracts"]["loss"])
        elif inherit_parent_loss:
            if continuation_preview is None:
                raise ValueError("loss.inherit_parent requires --continuation-parent")
            loss_mapping = dict(continuation_preview["contracts"]["loss"])
        calibration_requested = loss_mapping.get("generated_bond") in (None, "calibrate")
        if resume_payload is None and calibration_requested:
            loss_mapping["generated_bond"] = 0.0
        loss_config = JointLossConfig.resolve(loss_mapping)
        base_rates = {
            name: max(float(stage["learning_rates"][name]) for stage in stages)
            for name in ("history_encoder", "dit", "decoder")
        }
        trainer = FrameJointTrainer(
            wrapped,
            loss_config=loss_config,
            learning_rates=base_rates,
            weight_decay=float(training.get("weight_decay", 0.01)),
            grad_clip=float(training.get("grad_clip", 1.0)),
            stage=str(initial_stage["name"]),
            amp=str(training.get("precision", "fp32")) == "bf16",
            data_hash=data_hash,
            teacher_artifact_sha256=artifact.report.source_sha256,
            schedule_contract=schedule_contract,
            continuation_parent=(
                resume_payload["contracts"].get("continuation_parent")
                if resume_payload is not None
                else None
            ),
        )
        generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
        sampler = _sampler(train, training, seed=seed, rank=rank, world=world)
        epoch = batch_index = 0
        sampler.set_epoch(epoch)
        if args.resume:
            loaded = trainer.load_checkpoint(
                args.resume,
                generator=generator,
                expected_sha256=args.resume_sha256,
                rank=rank,
            )
            rank_states = loaded["extra_state"].get("rank_states", {})
            rank_cursor = rank_states.get(str(rank), {}).get("cursor") if isinstance(rank_states, Mapping) else None
            cursor = rank_cursor if isinstance(rank_cursor, Mapping) else loaded["cursor"]
            epoch, batch_index = int(cursor["epoch"]), int(cursor["batch_index"])
            sampler.set_epoch(epoch)
            sampler.validate_state_dict(cursor["sampler_state"])
            if canonical_hash(sampler.global_batches) != cursor["global_batch_schedule_hash"]:
                raise ValueError("resume batch schedule differs")
        elif args.continuation_parent:
            loaded = trainer.load_continuation_parent(
                args.continuation_parent,
                generator=generator,
                expected_sha256=args.continuation_parent_sha256,
                rank=rank,
            )
            rank_states = loaded["extra_state"].get("rank_states", {})
            rank_cursor = rank_states.get(str(rank), {}).get("cursor") if isinstance(rank_states, Mapping) else None
            cursor = rank_cursor if isinstance(rank_cursor, Mapping) else loaded["cursor"]
            epoch, batch_index = int(cursor["epoch"]), int(cursor["batch_index"])
            sampler.set_epoch(epoch)
            sampler.validate_state_dict(cursor["sampler_state"])
            if canonical_hash(sampler.global_batches) != cursor["global_batch_schedule_hash"]:
                raise ValueError("continuation parent batch schedule differs")
        output_root = Path(training["output_root"])
        resolved_config = {
            **raw,
            "resolved": {
                "data_hash": data_hash,
                "statistics_hash": statistics.hash,
                "statistics_sha256": sha256_file(stats_path),
                "codec_artifact_step": artifact.step,
                "world_size": world,
                "total_updates": total_updates,
                "output_root": str(output_root),
                "continuation_parent": trainer.continuation_parent,
                "loss_contract": trainer.loss_config.contract(),
            },
        }
        if rank == 0:
            output_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output_root / "resolved_config.json", resolved_config)
            atomic_write_json(
                output_root / "target_encoder_provenance.json",
                frame_target_provenance(
                    artifact,
                    source_path=codec_config["checkpoint"],
                    code_commit=_git_commit(),
                ),
            )
        exposures: Counter[str] = Counter()
        if args.dry_run:
            total_updates = min(total_updates, trainer.step + 1)
        last: dict[str, Any] = {}
        calibration_done = not calibration_requested or not loss_config.bond_enabled or loss_config.generated_bond > 0
        while trainer.step < total_updates:
            batches = list(iter(sampler))
            if batch_index >= len(batches):
                epoch += 1
                batch_index = 0
                sampler.set_epoch(epoch)
                batches = list(iter(sampler))
            indices = batches[batch_index]
            cpu_batch = collate_clip_records([train[index] for index in indices])
            history = histories[trainer.step % len(histories)]
            if history >= cpu_batch.frames:
                raise ValueError("sampled history must leave future query frames")
            stage, stage_position = _stage_at(stages, trainer.step)
            if trainer.stage != stage["name"]:
                trainer.configure_stage(str(stage["name"]))
            if stage["name"] in {"joint", "continuation"} and not calibration_done:
                coefficient = torch.zeros((), device=device)
                diagnostics: dict[str, Any] = {}
                if rank == 0:
                    coefficient_value, diagnostics = calibrate_generated_bond_weight(
                        model,
                        _calibration_stream(
                            model,
                            train,
                            training,
                            device=device,
                            seed=seed,
                            histories=histories,
                        ),
                        generator=torch.Generator(device=device).manual_seed(seed + 8000),
                        target_ratio=float(raw["loss"].get("calibration_target_ratio", 0.1)),
                    )
                    coefficient.fill_(coefficient_value)
                if world > 1:
                    dist.broadcast(coefficient, src=0)
                resolved_loss = dict(raw["loss"])
                resolved_loss.pop("calibration_target_ratio", None)
                resolved_loss["generated_bond"] = float(coefficient)
                trainer.loss_config = JointLossConfig.resolve(resolved_loss)
                calibration_done = True
                if rank == 0:
                    atomic_write_json(output_root / "bond_calibration.json", diagnostics)
                    resolved_config["resolved"]["loss_contract"] = trainer.loss_config.contract()
                    atomic_write_json(output_root / "resolved_config.json", resolved_config)
            trainer.set_learning_rates(_scheduled_rates(stage, stage_position))
            prepared = prepare_frame_joint_batch(
                model.target_teacher,
                cpu_batch,
                device=device,
                normalizer=model,
                history_frames=history,
            )
            last = trainer.train_step(prepared, generator=generator)
            last.update({
                "epoch": epoch,
                "batch_index": batch_index,
                "history_frames": history,
                "sample_ids": list(cpu_batch.sample_id),
                "atom_frames": cpu_batch.atom_count * cpu_batch.frames,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in trainer.optimizer.param_groups
                },
            })
            for sample_id in cpu_batch.sample_id:
                exposures[_trajectory(sample_id)] += 1
            batch_index += 1
            if rank == 0:
                with (output_root / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(last, sort_keys=True) + "\n")
            save_now = args.checkpoint_every and trainer.step % args.checkpoint_every == 0
            if save_now or trainer.step == total_updates:
                cursor = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "sampler_state": sampler.state_dict(),
                    "global_batch_schedule_hash": canonical_hash(sampler.global_batches),
                }
                local_state = {
                    "cursor": cursor,
                    "rng_state": capture_rng_state(),
                    "training_generator_state": generator.get_state().detach().cpu(),
                }
                states: list[Any] = [None for _ in range(world)]
                if world > 1:
                    dist.all_gather_object(states, local_state)
                else:
                    states[0] = local_state
                if rank == 0:
                    trainer.save_checkpoint(
                        output_root / f"frame_joint_step_{trainer.step:08d}.pt",
                        cursor=cursor,
                        generator=generator,
                        rank_states={str(index): state for index, state in enumerate(states)},
                    )
                if world > 1:
                    dist.barrier()
        if rank == 0:
            atomic_write_json(output_root / "exposure.json", {
                "successful_updates": trainer.successful_updates,
                "local_trajectory_clip_exposure": dict(sorted(exposures.items())),
                "clips_per_trajectory_per_epoch": int(training["clips_per_trajectory"]),
                "world_size": world,
            })
            print(json.dumps({"step": trainer.step, "metrics": last, "output_root": str(output_root)}, sort_keys=True))
        return 0
    finally:
        close_data()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
