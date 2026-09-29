"""Single-GPU formal training loop for paired learned-memory arms."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import torch

from ..checkpoints import capture_rng_state, frame_target_provenance, load_codec_artifact
from ..data.learned_memory_formal import RawMemoryDataset
from ..data.sampling import TrajectoryCappedBatchSampler
from ..latent.statistics import FrameLatentStatistics
from ..model import FrameJointModel
from ..runtime import atomic_write_json, canonical_hash, configure_device, seed_all, sha256_file
from .batches import prepare_frame_joint_batch
from .joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig
from .dit import module_state_hash
from .learned_memory_schedule import (
    SCHEMA as EXECUTION_SCHEMA,
    learned_memory_execution_manifest,
    learned_memory_execution_schedule,
)


SCHEMA = "molvid.learned_memory.train.v1"


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).strip()


def _trajectory(sample_id: str) -> str:
    return str(sample_id).rsplit("_w", 1)[0]


def _scheduled_rates(base: Mapping[str, float], position: int, updates: int) -> dict[str, float]:
    warmup = max(1, math.ceil(0.05 * updates))
    if position < warmup:
        factor = float(position + 1) / warmup
    else:
        progress = float(position - warmup) / max(1, updates - warmup - 1)
        factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return {name: float(value) * factor for name, value in base.items()}


def _validate_config(raw: Mapping[str, Any]) -> None:
    required = {
        "schema", "arm", "data", "codec", "statistics", "model", "parent",
        "loss", "sampled_auxiliary", "training", "release", "budget",
    }
    if set(raw) != required or raw.get("schema") != SCHEMA:
        raise ValueError(f"learned-memory config keys/schema differ: {sorted(set(raw) ^ required)}")
    if raw["arm"] not in {"M-short", "M-long"}:
        raise ValueError("formal arm must be M-short or M-long")
    training = raw["training"]
    allowed_training = {
        "seed", "device", "deterministic", "precision", "max_atom_frames_per_gpu",
        "clips_per_trajectory", "effective_epochs", "max_updates", "learning_rates",
        "weight_decay", "grad_clip", "gradient_diagnostics_every", "output_root",
    }
    if set(training) != allowed_training:
        raise ValueError(f"unknown learned-memory training keys: {sorted(set(training) ^ allowed_training)}")
    if int(training["effective_epochs"]) not in {1, 2}:
        raise ValueError("B formal exposure is frozen to one or two effective epochs")
    if int(training["clips_per_trajectory"]) != 24:
        raise ValueError("B formal exposure requires24 clips per trajectory per epoch")
    if str(training["precision"]) != "bf16" or str(training["device"]) != "cuda":
        raise ValueError("released B recipe requires CUDA BF16")
    model = raw["model"]
    expected_model = {
        "history_backend": "learned_memory", "analysis_mode": "physical",
        "depth": 4, "heads": 8, "scalar_width": 256, "vector_width": 128,
        "geometry_enabled": True, "motion_enabled": False,
    }
    if model != expected_model:
        raise ValueError("B formal model contract differs from the reviewed architecture")


def _verify_release(raw: Mapping[str, Any]) -> dict[str, Any]:
    release = raw["release"]
    recipe_path = Path(release["recipe_release"])
    capacity_path = Path(release["capacity_release"])
    if sha256_file(recipe_path) != release["recipe_release_sha256"]:
        raise ValueError("recipe_release identity differs")
    if sha256_file(capacity_path) != release["capacity_release_sha256"]:
        raise ValueError("capacity_release identity differs")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    if recipe.get("selected_arm") != "A1" or capacity.get("selected_arm") != "A1":
        raise ValueError("B formal requires the evaluated A1 release")
    if recipe["A1_checkpoint"]["sha256"] != raw["parent"]["sha256"]:
        raise ValueError("configured parent is not the released A1 checkpoint")
    if recipe["configuration"]["sha256"] != release["A1_configuration_sha256"]:
        raise ValueError("configured A1 recipe identity differs")
    return {"recipe": recipe, "capacity": capacity}


def _load_statistics(raw: Mapping[str, Any], device: torch.device) -> FrameLatentStatistics:
    path = Path(raw["path"])
    if sha256_file(path) != raw["sha256"]:
        raise ValueError("statistics file identity differs")
    statistics = FrameLatentStatistics.from_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    if statistics.hash != raw["statistics_hash"]:
        raise ValueError("statistics semantic hash differs")
    if statistics.provenance.get("data_hash") != raw["normalization_data_hash"]:
        raise ValueError("statistics normalization data differs")
    if statistics.provenance.get("codec_checkpoint_sha256") != raw["codec_sha256"]:
        raise ValueError("statistics teacher identity differs")
    return statistics.to(device)


def _exposure_state(
    trajectories: Counter[str], conditions: Counter[str], *,
    successful_updates: int, raw_atom_frames: int, teacher_atom_frames: int,
) -> dict[str, Any]:
    return {
        "schema": "molvid.learned_memory.exposure.v1",
        "successful_updates": int(successful_updates),
        "trajectory_clip_exposure": dict(sorted(trajectories.items())),
        "condition_clip_exposure": dict(sorted(conditions.items())),
        "raw_valid_atom_frames": int(raw_atom_frames),
        "teacher_atom_frames": int(teacher_atom_frames),
    }


def run_learned_memory_training(
    config_path: Path,
    *,
    resume: Path | None = None,
    resume_sha256: str | None = None,
    checkpoint_every: int = 1000,
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _validate_config(raw)
    releases = _verify_release(raw)
    training = raw["training"]
    device = configure_device(
        str(training["device"]), deterministic=bool(training["deterministic"])
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("B formal runner requires exactly one visible CUDA device")
    seed = int(training["seed"])
    seed_all(seed)
    dataset = RawMemoryDataset(
        manifest_root=Path(raw["data"]["manifest_root"]),
        raw_root=Path(raw["data"]["raw_root"]),
        offset_root=Path(raw["data"]["xtc_offset_root"]),
        split="train",
    )
    if (
        dataset.identity["xtc_offset_cache"]["manifest_sha256"]
        != raw["data"]["xtc_offset_manifest_sha256"]
    ):
        raise ValueError("frozen XTC offset manifest identity differs")
    expected_data_hash = raw["data"]["data_hash"]
    if expected_data_hash and expected_data_hash != dataset.identity["data_hash"]:
        raise ValueError("frozen raw-memory data hash differs")
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
    epoch_counts = []
    for epoch_value in range(int(training["effective_epochs"])):
        sampler.set_epoch(epoch_value)
        epoch_counts.append(len(sampler))
    total_updates = sum(epoch_counts)
    if training["max_updates"] is not None:
        total_updates = min(total_updates, int(training["max_updates"]))
    if total_updates < 1:
        raise ValueError("learned-memory training requires positive updates")

    codec_cfg = raw["codec"]
    artifact = load_codec_artifact(
        codec_cfg["checkpoint"],
        expected_sha256=codec_cfg["sha256"],
        device=device,
    )
    codec = artifact.model.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    statistics_cfg = {**raw["statistics"], "codec_sha256": codec_cfg["sha256"]}
    statistics = _load_statistics(statistics_cfg, device)
    model_cfg = raw["model"]
    seed_all(seed)
    model = FrameJointModel.from_codec(
        codec,
        statistics,
        scalar_width=model_cfg["scalar_width"],
        vector_width=model_cfg["vector_width"],
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        geometry_enabled=model_cfg["geometry_enabled"],
        motion_enabled=model_cfg["motion_enabled"],
        history_backend=model_cfg["history_backend"],
        analysis_mode=model_cfg["analysis_mode"],
    ).to(device)
    initial_history_hash = module_state_hash(model.history_encoder)
    rates = {name: float(value) for name, value in training["learning_rates"].items()}
    schedule_contract = {
        "schema": "molvid.learned_memory.schedule.v1",
        "execution_schema": EXECUTION_SCHEMA,
        "arm": raw["arm"],
        "seed": seed,
        "effective_epochs": int(training["effective_epochs"]),
        "updates_per_epoch": epoch_counts,
        "total_updates": total_updates,
        "clips_per_trajectory": int(training["clips_per_trajectory"]),
        "conditions": {
            "history_spans_ps": [800] if raw["arm"] == "M-short" else [800, 3200, 12800],
            "observed_dt_ps": [100, 200, 400],
            "query_lag_ps": [100, 200, 400],
            "teacher_anchor_times_ps": [-800, 0],
            "future_Q": 12,
        },
        "learning_rate_schedule": "linear_warmup_5pct_cosine_to_0.1.v1",
    }
    trainer = FrameJointTrainer(
        model,
        loss_config=JointLossConfig.resolve(raw["loss"]),
        learning_rates=rates,
        weight_decay=float(training["weight_decay"]),
        grad_clip=float(training["grad_clip"]),
        stage="continuation",
        amp=True,
        data_hash=dataset.identity["data_hash"],
        teacher_artifact_sha256=codec_cfg["sha256"],
        schedule_contract=schedule_contract,
        sampled_config=SampledAuxiliaryConfig.resolve(raw["sampled_auxiliary"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    sampled_generator = torch.Generator(device=device).manual_seed(seed + 2000)
    output = Path(training["output_root"])
    output.mkdir(parents=True, exist_ok=resume is not None)
    epoch = batch_index = 0
    trajectories: Counter[str] = Counter()
    conditions: Counter[str] = Counter()
    raw_atom_frames = teacher_atom_frames = 0
    resume_cursor: Mapping[str, Any] | None = None
    if resume is None:
        parent = raw["parent"]
        report = trainer.load_warm_start(
            parent["path"], expected_sha256=parent["sha256"]
        )
        atomic_write_json(output / "warm_start_report.json", report.as_dict())
    else:
        loaded = trainer.load_checkpoint(
            resume,
            generator=generator,
            sampled_generator=sampled_generator,
            expected_sha256=resume_sha256,
            rank=0,
        )
        cursor = loaded["cursor"]
        resume_cursor = cursor
        epoch, batch_index = int(cursor["epoch"]), int(cursor["batch_index"])
        exposure = loaded["extra_state"]["rank_states"]["0"]["exposure_state"]
        trajectories.update(exposure["trajectory_clip_exposure"])
        conditions.update(exposure["condition_clip_exposure"])
        raw_atom_frames = int(exposure["raw_valid_atom_frames"])
        teacher_atom_frames = int(exposure["teacher_atom_frames"])
    optimizer_ids = [id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]]
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(trainable_ids):
        raise RuntimeError("optimizer does not own every trainable parameter exactly once")
    teacher_hash = module_state_hash(model.target_teacher)
    teacher_calls: list[int] = []
    memory_calls: list[int] = []
    teacher_hook = model.target_teacher.register_forward_pre_hook(
        lambda _module, args: teacher_calls.append(int(args[0].frames))
    )
    memory_hook = model.history_encoder.register_forward_pre_hook(
        lambda _module, _args: memory_calls.append(1)
    )
    config_sha = sha256_file(config_path)
    atomic_write_json(output / "resolved_config.json", {
        **raw,
        "resolved": {
            "code_commit": _commit(),
            "config_sha256": config_sha,
            "data": dataset.identity,
            "total_updates": total_updates,
            "updates_per_epoch": epoch_counts,
            "model_contract": model.contract(),
            "loss_contract": trainer.loss_config.contract(),
            "sampled_auxiliary_contract": trainer.sampled_config.contract(),
            "initial_history_hash": initial_history_hash,
            "optimizer_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "recipe_release_status": releases["recipe"]["status"],
        },
    })
    atomic_write_json(output / "data_card.json", dataset.identity)
    atomic_write_json(output / "target_encoder_provenance.json", frame_target_provenance(
        artifact,
        source_path=codec_cfg["checkpoint"],
        code_commit=_commit(),
    ))
    peak = 0
    last: dict[str, Any] = {}

    def build_execution(current_epoch: int):
        sampler.set_epoch(current_epoch)
        batches = list(iter(sampler))
        entries = learned_memory_execution_schedule(
            batches,
            sampler.specs,
            arm=raw["arm"],
            seed=seed,
            epoch=current_epoch,
            cadence=trainer.sampled_config.cadence,
        )
        manifest = learned_memory_execution_manifest(entries)
        atomic_write_json(output / f"execution_epoch_{current_epoch:04d}.json", manifest)
        return batches, entries, manifest

    batches, execution, execution_manifest = build_execution(epoch)
    if resume_cursor is not None:
        if (
            resume_cursor.get("arm") != raw["arm"]
            or resume_cursor.get("execution_schedule_hash") != execution_manifest["sha256"]
            or resume_cursor.get("paired_schedule_hash")
            != execution_manifest["paired_schedule_sha256"]
            or not 0 <= batch_index <= len(batches)
        ):
            raise ValueError("resume cursor differs from the frozen execution schedule")
    completed = False
    failure: dict[str, str] | None = None
    try:
        while trainer.step < total_updates:
            if batch_index >= len(batches):
                epoch += 1
                batch_index = 0
                batches, execution, execution_manifest = build_execution(epoch)
            indices = batches[batch_index]
            entry = execution[batch_index]
            if tuple(indices) != entry.indices:
                raise RuntimeError("paired learned-memory schedule cursor disagrees")
            io_started = time.perf_counter()
            raw_history, anchors, future = dataset.load_inputs(
                indices,
                history_span_ps=entry.history_span_ps,
                observed_dt_ps=entry.observed_dt_ps,
                query_lag_ps=entry.query_lag_ps,
            )
            io_seconds = time.perf_counter() - io_started
            before_teacher = len(teacher_calls)
            before_memory = len(memory_calls)
            prepared = prepare_frame_joint_batch(
                model.target_teacher,
                future,
                device=device,
                normalizer=model,
                history_frames=2,
                raw_history=raw_history,
                geometry_anchors=anchors,
            )
            trainer.set_learning_rates(_scheduled_rates(rates, trainer.step, total_updates))
            diagnose = (
                trainer.step == 0
                or (int(training["gradient_diagnostics_every"]) > 0
                    and (trainer.step + 1) % int(training["gradient_diagnostics_every"]) == 0)
            )
            torch.cuda.reset_peak_memory_stats()
            update_started = time.perf_counter()
            last = trainer.train_step(
                prepared,
                generator=generator,
                sampled_generator=sampled_generator,
                auxiliary_active=entry.auxiliary,
                diagnose_gradients=diagnose,
            )
            torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            calls = teacher_calls[before_teacher:]
            if calls != [2, 12]:
                raise RuntimeError(f"teacher frame calls changed: {calls}")
            if sum(memory_calls[before_memory:]) != 1:
                raise RuntimeError("learned memory was not prepared exactly once per prefix")
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
            if diagnose:
                missing = [
                    name for name, parameter in model.history_encoder.named_parameters()
                    if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                ]
                if missing:
                    raise RuntimeError(f"learned-memory gradients missing/nonfinite: {missing}")
            raw_valid = int(
                raw_history.frame_mask.index_select(0, raw_history.topology.abid).sum()
            )
            teacher_valid = 14 * anchors.atom_count
            raw_atom_frames += raw_valid
            teacher_atom_frames += teacher_valid
            condition = (
                f"span_{entry.history_span_ps}ps|history_dt_{entry.observed_dt_ps}ps|"
                f"query_{entry.query_lag_ps}ps"
            )
            conditions[condition] += anchors.batch_size
            for index in indices:
                trajectories[_trajectory(dataset[index].sample_id)] += 1
            last.update({
                "epoch": epoch,
                "batch_index": batch_index,
                "sample_ids": [dataset[index].sample_id for index in indices],
                "history_span_ps": entry.history_span_ps,
                "observed_dt_ps": entry.observed_dt_ps,
                "query_lag_ps": entry.query_lag_ps,
                "scheduled_auxiliary": entry.auxiliary,
                "execution_schedule_hash": execution_manifest["sha256"],
                "paired_schedule_hash": execution_manifest["paired_schedule_sha256"],
                "raw_io_seconds": io_seconds,
                "update_seconds": update_seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "teacher_frame_calls": calls,
                "memory_preparations": 1,
                "raw_valid_atom_frames": raw_valid,
                "teacher_atom_frames": teacher_valid,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in trainer.optimizer.param_groups
                },
            })
            with (output / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last, sort_keys=True) + "\n")
            batch_index += 1
            save_now = checkpoint_every and trainer.step % checkpoint_every == 0
            if save_now or trainer.step == total_updates:
                cursor = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "sampler_state": sampler.state_dict(),
                    "execution_schedule_hash": execution_manifest["sha256"],
                    "paired_schedule_hash": execution_manifest["paired_schedule_sha256"],
                    "arm": raw["arm"],
                }
                exposure = _exposure_state(
                    trajectories,
                    conditions,
                    successful_updates=trainer.successful_updates,
                    raw_atom_frames=raw_atom_frames,
                    teacher_atom_frames=teacher_atom_frames,
                )
                local_state = {
                    "cursor": cursor,
                    "rng_state": capture_rng_state(),
                    "training_generator_state": generator.get_state().detach().cpu(),
                    "sampled_generator_state": sampled_generator.get_state().detach().cpu(),
                    "exposure_state": exposure,
                }
                trainer.save_checkpoint(
                    output / f"frame_joint_step_{trainer.step:08d}.pt",
                    cursor=cursor,
                    generator=generator,
                    sampled_generator=sampled_generator,
                    rank_states={"0": local_state},
                )
        if module_state_hash(model.target_teacher) != teacher_hash:
            raise RuntimeError("frozen teacher parameters changed")
        elapsed = time.perf_counter() - invocation_started
        result = {
            "schema": "molvid.learned_memory.train_manifest.v1",
            "status": "completed",
            "arm": raw["arm"],
            "successful_updates": trainer.successful_updates,
            "effective_epochs": int(training["effective_epochs"]),
            "training_seconds_this_invocation": elapsed,
            "gpu_hours_this_invocation": elapsed / 3600.0,
            "peak_allocated_bytes": peak,
            "teacher_observed_frames_per_prefix": 2,
            "teacher_target_frames_per_prefix": 12,
            "memory_preparations_per_prefix": 1,
            "parent_sha256": raw["parent"]["sha256"],
            "config_sha256": config_sha,
            "code_commit": _commit(),
            "data_hash": dataset.identity["data_hash"],
            "checkpoint": str(output / f"frame_joint_step_{trainer.step:08d}.pt"),
            "checkpoint_sha256": sha256_file(output / f"frame_joint_step_{trainer.step:08d}.pt"),
            "raw_valid_atom_frames": raw_atom_frames,
            "teacher_atom_frames": teacher_atom_frames,
            "test_opened": False,
        }
        atomic_write_json(output / "train_manifest.json", result)
        atomic_write_json(output / "exposure.json", _exposure_state(
            trajectories,
            conditions,
            successful_updates=trainer.successful_updates,
            raw_atom_frames=raw_atom_frames,
            teacher_atom_frames=teacher_atom_frames,
        ))
        completed = True
        return result
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        teacher_hook.remove()
        memory_hook.remove()
        invocation_elapsed = time.perf_counter() - invocation_started
        atomic_write_json(output / "cost_ledger.json", {
            "schema": "molvid.learned_memory.cost_ledger.v1",
            "status": "completed" if completed else "failed_or_interrupted",
            "arm": raw["arm"],
            "successful_updates": trainer.successful_updates,
            "invocation_seconds": invocation_elapsed,
            "gpu_hours": invocation_elapsed / 3600.0,
            "peak_allocated_bytes": max(peak, int(torch.cuda.max_memory_allocated())),
            "failure": failure,
            "budget_bucket": "pilot" if training["max_updates"] is not None else "formal",
        })
