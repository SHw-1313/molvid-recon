#!/usr/bin/env python3
"""Run one control of the authorized state/detail codec v2 T1 protocol.

The profile and full-run modes are intentionally separate from the historical
T0 runner.  This file owns only lazy selected-system loading, deterministic
per-trajectory capped resampling, the four existing controls, and validation
evaluation.  Test data are never opened by this command; the selection tool
opens test only after an explicit frozen validation rule exists.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from data.clip_batching import (
    ClipSpecTable,
    TaskAwareClipBatchSampler,
    get_clip_specs,
    make_clip_dataloader,
)
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from scripts.run_state_detail_codec_v2_t0 import (
    FRAME_ENCODER_CHECKPOINT,
    MAX_TOKENS,
    MODES,
    RATIOS,
    _evaluate_loader,
    _gradient_summary,
    _make_config,
    _make_model,
    _runtime,
    _runtime_profile,
    _sha256,
    _state_hash,
    _write_json,
    _scheduled_ids,
)
from trainer.codec_trainer import CodecTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/manifest_20260904_token80000"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/state_detail_codec_v2/t1"
SEED = 20260903
EPOCHS = 30
TRAIN_CLIPS_PER_TRAJECTORY = 24
PROFILE_STEPS = 200
PROFILE_LOG_EVERY = 20
TRAIN_LOG_EVERY = 25
MODES = tuple(MODES)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _load_manifest(manifest_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_root / "manifest.json"
    materialization_path = manifest_root / "materialization.json"
    if not manifest_path.is_file() or not materialization_path.is_file():
        raise FileNotFoundError(
            f"T1 manifest/materialization missing under {manifest_root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = dict(manifest)
    recorded = content.pop("manifest_content_sha256", None)
    if recorded != _canonical_hash(content):
        raise RuntimeError("T1 manifest content hash is invalid")
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("T1 manifest is not frozen")
    if manifest.get("counts", {}).get("systems_total") != 64:
        raise RuntimeError("T1 manifest does not describe exactly 64 systems")
    if int(manifest.get("max_tokens", -1)) != MAX_TOKENS:
        raise RuntimeError("T1 manifest does not preserve max_tokens=80000")
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if not materialization.get("materialized"):
        raise RuntimeError("T1 clip store has not been materialized")
    for split in ("train", "valid", "test"):
        expected = manifest["source_splits"][split]["sample_ids"]
        record = materialization["splits"].get(split, {})
        store = manifest_root / "clip_store" / split
        if int(record.get("count", -1)) != len(expected):
            raise RuntimeError(f"{split} materialization count disagrees with manifest")
        if not store.is_dir() or not (store / "data.bin").is_file():
            raise FileNotFoundError(f"materialized {split} store is missing: {store}")
    return manifest, materialization


def _dataset_ids(dataset: ClipMMapDataset) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in dataset._index)


def _trajectory_id(sample_id: str) -> str:
    return str(sample_id).rsplit("_w", 1)[0]


def _spec_signature(specs: ClipSpecTable) -> str:
    values = [
        (
            int(item.index),
            int(item.atoms),
            int(item.frames),
            int(item.task),
            str(item.time_bucket_id),
            str(item.sample_id),
        )
        for item in specs
    ]
    return _canonical_hash(values)


class TrajectoryCappedBatchSampler(Sampler[list[int]]):
    """Sample exactly K distinct windows per trajectory in every epoch.

    The selected windows are resampled deterministically from all 62 native
    windows at each epoch.  The inner task-aware sampler then packs them under
    the unchanged 80,000 T*N token budget with replacement disabled.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        max_tokens: int,
        clips_per_trajectory: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.specs = get_clip_specs(dataset)
        self.max_tokens = int(max_tokens)
        self.clips_per_trajectory = int(clips_per_trajectory)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._inner: TaskAwareClipBatchSampler | None = None
        self._selected_positions: tuple[int, ...] = ()
        self._trajectory_positions: dict[str, tuple[int, ...]] = {}
        for position, item in enumerate(self.specs):
            trajectory = _trajectory_id(item.sample_id)
            self._trajectory_positions.setdefault(trajectory, []).append(position)
        self._trajectory_positions = {
            key: tuple(value) for key, value in sorted(self._trajectory_positions.items())
        }
        if not self._trajectory_positions:
            raise ValueError("T1 training dataset contains no trajectories")
        for trajectory, positions in self._trajectory_positions.items():
            windows = {int(self.specs[pos].sample_id.rsplit("_w", 1)[1]) for pos in positions}
            if len(positions) != 62 or windows != set(range(62)):
                raise ValueError(
                    f"trajectory {trajectory} does not contain exactly windows 0..61"
                )
        if self.clips_per_trajectory < 1 or self.clips_per_trajectory > 62:
            raise ValueError("clips_per_trajectory must be in [1, 62]")
        self._source_signature = _spec_signature(self.specs)

    def _trajectory_rng(self, trajectory: str, epoch: int) -> np.random.Generator:
        token = f"{self.seed}|{epoch}|{trajectory}".encode()
        digest = hashlib.sha256(token).digest()
        value = int.from_bytes(digest[:8], "little", signed=False)
        return np.random.default_rng(value)

    def _make_inner(self, epoch: int) -> TaskAwareClipBatchSampler:
        selected: list[int] = []
        for trajectory, positions in self._trajectory_positions.items():
            rng = self._trajectory_rng(trajectory, epoch)
            choice = rng.choice(
                np.asarray(positions, dtype=np.int64),
                size=self.clips_per_trajectory,
                replace=False,
            )
            chosen = [int(value) for value in choice.tolist()]
            if self.shuffle:
                rng.shuffle(chosen)
            selected.extend(chosen)
        selected_specs = [self.specs[position] for position in selected]
        inner = TaskAwareClipBatchSampler(
            ClipSpecTable.from_specs(selected_specs),
            max_tokens=self.max_tokens,
            seed=self.seed + int(epoch),
            shuffle=self.shuffle,
            replacement=False,
            oversize_policy="error",
        )
        self._selected_positions = tuple(selected)
        return inner

    def _current(self) -> TaskAwareClipBatchSampler:
        if self._inner is None:
            self._inner = self._make_inner(self.epoch)
        return self._inner

    @property
    def global_batches(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(batch) for batch in self._current().global_batches)

    @property
    def selected_sample_ids(self) -> tuple[str, ...]:
        self._current()
        return tuple(self.specs[position].sample_id for position in self._selected_positions)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._inner = None
        self._selected_positions = ()

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(batch) for batch in self._current().global_batches])

    def __len__(self) -> int:
        return len(self._current())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "pvb.codec.t1.trajectory_capped_sampler.v1",
            "epoch": int(self.epoch),
            "seed": int(self.seed),
            "max_tokens": int(self.max_tokens),
            "clips_per_trajectory": int(self.clips_per_trajectory),
            "shuffle": bool(self.shuffle),
            "replacement": False,
            "trajectory_count": len(self._trajectory_positions),
            "source_spec_signature": self._source_signature,
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for key in (
            "schema_version",
            "seed",
            "max_tokens",
            "clips_per_trajectory",
            "shuffle",
            "replacement",
            "trajectory_count",
            "source_spec_signature",
        ):
            if state.get(key) != expected[key]:
                raise ValueError(f"T1 sampler {key} differs from checkpoint")
        epoch = state.get("epoch")
        if not isinstance(epoch, (int, np.integer)) or int(epoch) < 0:
            raise ValueError("T1 sampler epoch must be a non-negative integer")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        self.set_epoch(int(state["epoch"]))


def _make_datasets(manifest_root: Path) -> tuple[ClipMMapDataset, ClipMMapDataset]:
    manifest, _materialization = _load_manifest(manifest_root)
    datasets = tuple(
        ClipMMapDataset(manifest_root / "clip_store" / split)
        for split in ("train", "valid")
    )
    for split, dataset in zip(("train", "valid"), datasets):
        expected = tuple(manifest["source_splits"][split]["sample_ids"])
        actual = _dataset_ids(dataset)
        if actual != expected:
            raise RuntimeError(f"{split} materialized index does not match frozen manifest")
    return datasets


def _make_loaders(
    manifest_root: Path,
    *,
    seed: int,
) -> tuple[ClipMMapDataset, ClipMMapDataset, Any, Any, list[int]]:
    train_dataset, valid_dataset = _make_datasets(manifest_root)
    train_sampler = TrajectoryCappedBatchSampler(
        train_dataset,
        max_tokens=MAX_TOKENS,
        clips_per_trajectory=TRAIN_CLIPS_PER_TRAJECTORY,
        seed=seed,
        shuffle=True,
    )
    train_loader = make_clip_dataloader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
    )
    valid_sampler = TaskAwareClipBatchSampler(
        valid_dataset,
        max_tokens=MAX_TOKENS,
        seed=seed + 1,
        shuffle=False,
        replacement=False,
        oversize_policy="error",
    )
    valid_loader = make_clip_dataloader(
        valid_dataset,
        sampler=valid_sampler,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
    )
    epoch_batches: list[int] = []
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        selected = train_sampler.selected_sample_ids
        if len(selected) != 144 * TRAIN_CLIPS_PER_TRAJECTORY:
            raise RuntimeError(f"T1 epoch {epoch} selected an unexpected clip count")
        if len(set(selected)) != len(selected):
            raise RuntimeError(f"T1 epoch {epoch} selected duplicate clips")
        counts: dict[str, int] = {}
        for sample_id in selected:
            counts[_trajectory_id(sample_id)] = counts.get(_trajectory_id(sample_id), 0) + 1
        if set(counts.values()) != {TRAIN_CLIPS_PER_TRAJECTORY} or len(counts) != 144:
            raise RuntimeError(f"T1 epoch {epoch} violates per-trajectory cap: {counts}")
        epoch_batches.append(len(train_sampler))
    if any(count < 1 for count in epoch_batches):
        raise RuntimeError("T1 train sampler produced an empty epoch")
    valid_ids = _scheduled_ids(valid_loader, valid_dataset, 0)
    if valid_ids != list(_dataset_ids(valid_dataset)):
        raise RuntimeError("T1 validation sampler is not an exact no-replacement pass")
    train_sampler.set_epoch(0)
    return train_dataset, valid_dataset, train_loader, valid_loader, epoch_batches


def _protocol(
    manifest_root: Path,
    manifest: Mapping[str, Any],
    materialization: Mapping[str, Any],
    *,
    phase: str,
    mode: str,
    seed: int,
    total_steps: int,
    epoch_batches: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": "pvb.codec.state_detail.t1_protocol.v1",
        "phase": phase,
        "mode": mode,
        "seed": int(seed),
        "manifest_root": str(manifest_root),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "materialization_sha256": materialization.get("materialization_sha256"),
        "system_counts": {
            split: manifest["source_splits"][split]["selected_system_count"]
            for split in ("train", "valid", "test")
        },
        "trajectory_counts": manifest["counts"],
        "train_windows_available": list(manifest["windows"]),
        "train_clips_per_trajectory_per_epoch": TRAIN_CLIPS_PER_TRAJECTORY,
        "train_sampling": manifest["train_sampling"],
        "validation_coverage": manifest["validation_sampling"],
        "test_coverage": manifest["test_sampling"],
        "frames_per_clip": 16,
        "time_bucket_id": "dt_100ps",
        "native_delta_time_ps": 100.0,
        "lazy_loading": True,
        "replacement": False,
        "precision": "fp32",
        "max_tokens": MAX_TOKENS,
        "epochs": EPOCHS,
        "epoch_batch_counts": list(map(int, epoch_batches)),
        "total_steps": int(total_steps),
        "spatial_backbone": "torchmd_et",
        "common_frame_encoder_checkpoint": str(FRAME_ENCODER_CHECKPOINT),
        "common_frame_encoder_checkpoint_sha256": _sha256(FRAME_ENCODER_CHECKPOINT),
        "freeze_frame_encoder": True,
        "spatial_refiner": False,
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "coordinate_stem": "centered_vector",
        "coordinate_decoder": "shared_framewise_equivariant",
        "loss_schedule": "0%-10% coord/local/bond; 10%-30% adds velocity; 30%-100% adds acceleration",
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-6, warmup_steps=20, grad_clip=1.0)",
        "validation_selection_rule": {
            "primary": "minimum validation future aligned_rmsd across completed epochs",
            "tie_break_1": "minimum validation future drmsd",
            "tie_break_2": "minimum validation future bond_rmse",
            "tie_break_3": "lexicographic mode name",
            "test_opened_before_rule_frozen": False,
        },
        "top_two_seed_policy": "after one-seed validation ranking, rerun only the two highest-ranked controls with seeds 20260904 and 20260905",
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _run_steps(
    trainer: CodecTrainer,
    *,
    target_steps: int,
    log_path: Path,
    log_every: int,
) -> tuple[list[dict[str, Any]], int, float]:
    if target_steps < trainer.step:
        raise ValueError("target_steps cannot be below trainer.step")
    iterator = trainer._train_iterator
    if iterator is None:
        iterator = trainer._new_train_iterator()
        trainer._train_iterator = iterator
    rows: list[dict[str, Any]] = []
    tokens = 0
    start = time.perf_counter()
    while trainer.step < target_steps:
        if iterator is None:
            trainer._train_iterator = trainer._new_train_iterator()
            iterator = trainer._train_iterator
        try:
            batch = next(iterator)
        except StopIteration:
            trainer.epoch += 1
            trainer.batch_in_epoch = 0
            trainer._train_iterator = trainer._new_train_iterator()
            iterator = trainer._train_iterator
            batch = next(iterator)
        tokens += int(batch.x.shape[0] * batch.x.shape[1])
        metrics = trainer.optimizer_step(batch)
        trainer.batch_in_epoch += 1
        finished_epoch = trainer.batch_in_epoch >= len(trainer.train_loader)
        logged_epoch = trainer.epoch
        if finished_epoch:
            trainer.epoch += 1
            trainer.batch_in_epoch = 0
            trainer._train_iterator = None
            iterator = trainer._train_iterator
        if trainer.step % int(log_every) == 0 or trainer.step == target_steps:
            rows.append(
                {
                    "schema_version": "pvb.codec.train.t1.v1",
                    "split": "train",
                    "step": int(trainer.step),
                    "epoch": int(logged_epoch),
                    "batch_in_epoch": int(trainer.batch_in_epoch),
                    "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                    "tokens": int(tokens),
                    "metrics": metrics,
                }
            )
    torch.cuda.synchronize(trainer.device)
    _write_jsonl(log_path, rows)
    return rows, tokens, time.perf_counter() - start


def _metric_or_nan(record: Mapping[str, Any], *keys: str) -> float:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return float("nan")
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _profile(
    *,
    manifest_root: Path,
    output_dir: Path,
    mode: str,
    seed: int,
    profile_steps: int,
) -> dict[str, Any]:
    manifest, materialization = _load_manifest(manifest_root)
    _seed(seed)
    train_dataset, valid_dataset, train_loader, valid_loader, epoch_batches = _make_loaders(
        manifest_root, seed=seed
    )
    full_steps = sum(epoch_batches)
    config = _make_config(
        profile_steps,
        train_batches=epoch_batches[0],
        schedule_steps=profile_steps,
    )
    model = _make_model(mode, freeze_frame_encoder=True)
    trainer = CodecTrainer(
        model,
        train_loader,
        valid_loader,
        config=config,
        device="cuda:0",
        non_blocking_transfer=False,
    )
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    protocol = _protocol(
        manifest_root,
        manifest,
        materialization,
        phase="profile",
        mode=mode,
        seed=seed,
        total_steps=full_steps,
        epoch_batches=epoch_batches,
    )
    protocol.update({"profile_steps": int(profile_steps), "profile_schedule_steps": int(profile_steps)})
    _write_json(mode_dir / "protocol.json", protocol)
    trainer.fit_normalization(train_loader)
    first_batch = next(iter(train_loader))
    runtime_profile = _runtime_profile(model, first_batch)
    before_hash = _state_hash(model.frame_encoder)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    train_rows, token_count, wall = _run_steps(
        trainer,
        target_steps=profile_steps,
        log_path=mode_dir / "train_metrics.jsonl",
        log_every=PROFILE_LOG_EVERY,
    )
    final_checkpoint = trainer.save_checkpoint(mode_dir / f"codec_step_{trainer.step:08d}.pt")
    resume_model = _make_model(mode, freeze_frame_encoder=True)
    _resume_train_dataset, _resume_valid_dataset, resume_loader, _resume_valid_loader, _resume_epoch_batches = _make_loaders(
        manifest_root, seed=seed
    )
    resume_config = _make_config(
        profile_steps + 1,
        train_batches=epoch_batches[0],
        schedule_steps=profile_steps,
    )
    resume_trainer = CodecTrainer(
        resume_model,
        resume_loader,
        config=resume_config,
        device="cuda:0",
        non_blocking_transfer=False,
    )
    resume_trainer.load_checkpoint(final_checkpoint)
    loaded_step = resume_trainer.step
    resume_trainer.run(max_steps=loaded_step + 1, log_every=1)
    after_hash = _state_hash(model.frame_encoder)
    step_seconds = wall / max(profile_steps, 1)
    result = {
        "schema_version": "pvb.codec.state_detail.t1_profile.v1",
        "status": "passed",
        "mode": mode,
        "seed": int(seed),
        "profile_steps": int(profile_steps),
        "full_epoch_count": EPOCHS,
        "full_steps_estimate": int(full_steps),
        "epoch_batch_counts": list(map(int, epoch_batches)),
        "train_tokens": int(token_count),
        "profile_wall_s": float(wall),
        "profile_steps_per_s": float(profile_steps / max(wall, 1.0e-8)),
        "profile_tokens_per_s": float(token_count / max(wall, 1.0e-8)),
        "estimated_full_wall_s": float(step_seconds * full_steps),
        "estimated_full_hours": float(step_seconds * full_steps / 3600.0),
        "runtime_profile": runtime_profile,
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(trainer.device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(trainer.device)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "model_contract": model.model_contract(),
        "frame_encoder_source_hash": _sha256(FRAME_ENCODER_CHECKPOINT),
        "frame_encoder_state_hash_before": before_hash,
        "frame_encoder_state_hash_after": after_hash,
        "frame_encoder_unchanged": before_hash == after_hash,
        "new_module_gradients": _gradient_summary(model),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": _sha256(final_checkpoint),
        "resume": {
            "status": "passed" if resume_trainer.step == loaded_step + 1 else "failed",
            "loaded_step": int(loaded_step),
            "resumed_step": int(resume_trainer.step),
        },
        "finite_execution": all(
            math.isfinite(float(row["metrics"]["total"])) for row in train_rows
        ),
    }
    if not result["finite_execution"] or not result["frame_encoder_unchanged"]:
        result["status"] = "failed"
    if result["resume"]["status"] != "passed":
        result["status"] = "failed"
    _write_json(mode_dir / "result.json", result)
    del resume_trainer, resume_model, trainer, model
    train_dataset.close()
    valid_dataset.close()
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _full_train(
    *,
    manifest_root: Path,
    output_dir: Path,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    manifest, materialization = _load_manifest(manifest_root)
    _seed(seed)
    train_dataset, valid_dataset, train_loader, valid_loader, epoch_batches = _make_loaders(
        manifest_root, seed=seed
    )
    total_steps = sum(epoch_batches)
    config = _make_config(
        total_steps,
        train_batches=epoch_batches[0],
        schedule_steps=total_steps,
    )
    model = _make_model(mode, freeze_frame_encoder=True)
    trainer = CodecTrainer(
        model,
        train_loader,
        valid_loader,
        config=config,
        device="cuda:0",
        non_blocking_transfer=False,
    )
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    protocol = _protocol(
        manifest_root,
        manifest,
        materialization,
        phase="full",
        mode=mode,
        seed=seed,
        total_steps=total_steps,
        epoch_batches=epoch_batches,
    )
    _write_json(mode_dir / "protocol.json", protocol)
    trainer.fit_normalization(train_loader)
    first_batch = next(iter(train_loader))
    runtime_profile = _runtime_profile(model, first_batch)
    before_hash = _state_hash(model.frame_encoder)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    train_start = time.perf_counter()
    train_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_checkpoint: Path | None = None
    train_tokens_total = 0
    for epoch in range(EPOCHS):
        sampler = train_loader.batch_sampler
        sampler.set_epoch(epoch)
        selected_ids = list(sampler.selected_sample_ids)
        if len(selected_ids) != 144 * TRAIN_CLIPS_PER_TRAJECTORY:
            raise RuntimeError(f"epoch {epoch} selected unexpected T1 clip count")
        expected_batches = int(epoch_batches[epoch])
        if len(train_loader) != expected_batches:
            raise RuntimeError(
                f"epoch {epoch} batch schedule changed: {len(train_loader)} != {expected_batches}"
            )
        seen_ids: list[str] = []
        epoch_tokens = 0
        epoch_start = time.perf_counter()
        last_metrics: dict[str, float] = {}
        sampler_iterator = iter(train_loader)
        for batch_index in range(expected_batches):
            batch = next(sampler_iterator)
            seen_ids.extend(str(value) for value in batch.sample_id)
            batch_tokens = int(batch.x.shape[0] * batch.x.shape[1])
            epoch_tokens += batch_tokens
            train_tokens_total += batch_tokens
            last_metrics = trainer.optimizer_step(batch)
            trainer.batch_in_epoch = batch_index + 1
            if trainer.step % TRAIN_LOG_EVERY == 0 or trainer.step == total_steps:
                train_rows.append(
                    {
                        "schema_version": "pvb.codec.train.t1.v1",
                        "split": "train",
                        "step": int(trainer.step),
                        "epoch": int(epoch + 1),
                        "batch_in_epoch": int(batch_index + 1),
                        "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                        "tokens": int(batch_tokens),
                        "metrics": last_metrics,
                    }
                )
        if sorted(seen_ids) != sorted(selected_ids) or len(set(seen_ids)) != len(seen_ids):
            raise RuntimeError(
                f"epoch {epoch} selected-sample coverage failed: seen={len(seen_ids)}"
            )
        trainer.epoch = epoch + 1
        trainer.batch_in_epoch = 0
        trainer._train_iterator = None
        evaluation_start = time.perf_counter()
        validation = _evaluate_loader(
            trainer,
            valid_loader,
            valid_dataset,
            epoch=epoch + 1,
            ratio=RATIOS[mode],
            detailed=(epoch + 1 == EPOCHS),
        )
        score = _metric_or_nan(validation, "metrics", "future", "aligned_rmsd")
        if not math.isfinite(score):
            raise RuntimeError(f"epoch {epoch + 1} validation aligned RMSD is not finite")
        if score < best_score:
            best_score = score
            best_checkpoint = mode_dir / "codec_best.pt"
            trainer.save_checkpoint(best_checkpoint)
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_record = {
            "epoch": epoch + 1,
            "step": int(trainer.step),
            "train_sample_count": len(seen_ids),
            "train_unique_sample_count": len(set(seen_ids)),
            "train_exact_sample_coverage": True,
            "train_tokens": int(epoch_tokens),
            "epoch_elapsed_s": float(epoch_elapsed),
            "train_samples_per_s": float(len(seen_ids) / max(epoch_elapsed, 1.0e-8)),
            "train_tokens_per_s": float(epoch_tokens / max(epoch_elapsed, 1.0e-8)),
            "last_train_metrics": last_metrics,
            "validation_evaluation_elapsed_s": float(time.perf_counter() - evaluation_start),
            "validation_evaluation": validation,
        }
        epoch_rows.append(epoch_record)
        _write_json(mode_dir / "evaluation_latest.json", epoch_record)
        _write_jsonl(mode_dir / "train_metrics.jsonl", train_rows)
        _write_jsonl(
            mode_dir / "validation_metrics.jsonl",
            [
                {
                    "epoch": row["epoch"],
                    "step": row["step"],
                    "loss": row["validation_evaluation"]["loss"],
                    "metrics": row["validation_evaluation"]["metrics"],
                }
                for row in epoch_rows
            ],
        )
    torch.cuda.synchronize(trainer.device)
    training_elapsed = time.perf_counter() - train_start
    final_checkpoint = trainer.save_checkpoint(mode_dir / f"codec_step_{trainer.step:08d}.pt")
    resume_model = _make_model(mode, freeze_frame_encoder=True)
    _resume_train_dataset, _resume_valid_dataset, resume_loader, _resume_valid_loader, _resume_epoch_batches = _make_loaders(
        manifest_root, seed=seed
    )
    resume_config = _make_config(
        total_steps + 1,
        train_batches=epoch_batches[0],
        schedule_steps=total_steps,
    )
    resume_trainer = CodecTrainer(
        resume_model,
        resume_loader,
        config=resume_config,
        device="cuda:0",
        non_blocking_transfer=False,
    )
    resume_trainer.load_checkpoint(final_checkpoint)
    loaded_step = resume_trainer.step
    resume_trainer.run(max_steps=loaded_step + 1, log_every=1)
    final_validation = epoch_rows[-1]["validation_evaluation"]
    after_hash = _state_hash(model.frame_encoder)
    result = {
        "schema_version": "pvb.codec.state_detail.t1_result.v1",
        "status": "passed",
        "mode": mode,
        "seed": int(seed),
        "epochs": EPOCHS,
        "epoch_batch_counts": list(map(int, epoch_batches)),
        "total_steps": int(total_steps),
        "train_tokens": int(train_tokens_total),
        "training_elapsed_s": float(training_elapsed),
        "train_samples_per_s": float(sum(row["train_sample_count"] for row in epoch_rows) / max(training_elapsed, 1.0e-8)),
        "train_tokens_per_s": float(train_tokens_total / max(training_elapsed, 1.0e-8)),
        "runtime_profile": runtime_profile,
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(trainer.device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(trainer.device)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "model_contract": model.model_contract(),
        "frame_encoder_source_hash": _sha256(FRAME_ENCODER_CHECKPOINT),
        "frame_encoder_state_hash_before": before_hash,
        "frame_encoder_state_hash_after": after_hash,
        "frame_encoder_unchanged": before_hash == after_hash,
        "new_module_gradients": _gradient_summary(model),
        "epoch_metrics": epoch_rows,
        "final_validation": final_validation,
        "best_validation_future_aligned_rmsd": float(best_score),
        "best_epoch": min(
            row["epoch"]
            for row in epoch_rows
            if math.isclose(
                _metric_or_nan(row["validation_evaluation"], "metrics", "future", "aligned_rmsd"),
                best_score,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": _sha256(final_checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "best_checkpoint_sha256": _sha256(best_checkpoint) if best_checkpoint is not None else None,
        "resume": {
            "status": "passed" if resume_trainer.step == loaded_step + 1 else "failed",
            "loaded_step": int(loaded_step),
            "resumed_step": int(resume_trainer.step),
        },
        "curves": {
            "train_metrics": str(mode_dir / "train_metrics.jsonl"),
            "validation_metrics": str(mode_dir / "validation_metrics.jsonl"),
        },
    }
    if not result["frame_encoder_unchanged"] or result["resume"]["status"] != "passed":
        result["status"] = "failed"
    _write_json(mode_dir / "result.json", result)
    del resume_trainer, resume_model, trainer, model
    train_dataset.close()
    valid_dataset.close()
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("profile", "full"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--profile-steps", type=int, default=PROFILE_STEPS)
    args = parser.parse_args(argv)
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else ROOT / args.manifest_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not torch.cuda.is_available():
        raise RuntimeError("T1 requires CUDA; CPU fallback is forbidden")
    _runtime(torch.device("cuda:0"))
    if args.phase == "profile":
        result = _profile(
            manifest_root=manifest_root,
            output_dir=output_dir,
            mode=args.mode,
            seed=args.seed,
            profile_steps=args.profile_steps,
        )
    else:
        result = _full_train(
            manifest_root=manifest_root,
            output_dir=output_dir,
            mode=args.mode,
            seed=args.seed,
        )
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
