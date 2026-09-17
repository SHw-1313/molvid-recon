#!/usr/bin/env python
"""Run the matched R2/R4 real-T1 state/detail DiT pilot.

The runner is intentionally narrow: it opens only the frozen train and
validation clip stores, loads only the two approved T1 codec checkpoints,
fits train-only latent statistics, and trains the shared DiT with the fixed
H=4/H=8 schedule. It never constructs a test dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_batching import TaskAwareClipBatchSampler
from data.clip_dataset import ClipMMapDataset, ClipBatch, collate_clip_records
from evaluation.dit_evaluation import evaluate_oracle_vs_generated
from module.latent_rectified_flow import generate_state_detail_latent
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    build_observation_condition,
    contract_hash,
)
from scripts.run_state_detail_codec_v2_t0 import _state_hash
from scripts.run_state_detail_codec_v2_t1 import TrajectoryCappedBatchSampler
from trainer.codec_trainer import PVBCodecModel
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


PILOT_SCHEMA = "pvb.dit.state_detail.t1_pilot.v1"
DEFAULT_MANIFEST_ROOT = Path(
    "/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/"
    "manifest_20260904_token80000"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/dit_state_detail_pilot_v1")
MAX_TOKENS = 80000
TRAIN_CLIPS_PER_TRAJECTORY = 24
STATS_CLIPS_PER_TRAJECTORY = 62
HISTORY_SCHEDULE = (4, 8)
VALIDATION_WINDOWS = (0, 30, 61)
SAMPLE_ID_RE = re.compile(r"^(?P<system>.+)_(?P<replica>R[0-9]+)_w(?P<window>[0-9]+)$")


@dataclass(frozen=True)
class FrozenCodec:
    model: PVBCodecModel
    candidate: str
    ratio: int
    result_path: Path
    checkpoint_path: Path
    result_sha256: str
    checkpoint_sha256: str
    model_contract_hash: str
    codec_state_hash: str
    frame_encoder_state_hash: str
    frame_encoder_source_hash: str


@dataclass(frozen=True)
class PilotData:
    manifest_root: Path
    manifest: Mapping[str, Any]
    materialization: Mapping[str, Any]
    train: ClipMMapDataset
    valid: ClipMMapDataset
    data_hash: str
    train_index_hash: str
    valid_index_hash: str


@dataclass(frozen=True)
class ValidationPlan:
    subset: Subset
    batches: tuple[tuple[int, ...], ...]
    selected_dataset_indices: tuple[int, ...]
    selected_sample_ids: tuple[str, ...]
    systems: tuple[str, ...]
    windows: tuple[int, ...]
    schedule_hash: str


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(value), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    return generator.manual_seed(int(seed))


def _reject_test_path(path: Path, label: str) -> None:
    lowered = {part.lower() for part in path.parts}
    if "test" in lowered or "test" in str(path).lower().split("/"):
        raise RuntimeError(f"{label} must not reference a test split: {path}")


def _validate_manifest(manifest_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_root = manifest_root.resolve()
    if "test" in str(manifest_root).lower():
        raise RuntimeError("manifest root must not be a test path")
    manifest_path = manifest_root / "manifest.json"
    materialization_path = manifest_root / "materialization.json"
    if not manifest_path.is_file() or not materialization_path.is_file():
        raise FileNotFoundError(f"frozen manifest/materialization missing under {manifest_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = dict(manifest)
    recorded = content.pop("manifest_content_sha256", None)
    if recorded != _canonical_hash(content):
        raise RuntimeError("frozen manifest content hash is invalid")
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("frozen manifest status is not FROZEN")
    counts = manifest.get("counts", {})
    expected_counts = {
        "systems_total": 64,
        "train_systems": 48,
        "valid_systems": 8,
        "test_systems": 8,
        "train_clips_materialized": 8928,
        "valid_clips_materialized": 1488,
        "test_clips_materialized": 1488,
    }
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise RuntimeError(f"manifest count {key} disagrees with {expected}")
    if int(manifest.get("frames_per_clip", -1)) != 16:
        raise RuntimeError("pilot requires T=16 clips")
    if int(manifest.get("max_tokens", -1)) != MAX_TOKENS:
        raise RuntimeError("pilot requires the frozen max_tokens=80000 protocol")
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("test_sampling.opened is not false")
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if materialization.get("materialized") is not True:
        raise RuntimeError("frozen clip materialization is not complete")
    for split in ("train", "valid"):
        record = materialization.get("splits", {}).get(split, {})
        expected = manifest["source_splits"][split]["sample_ids"]
        if int(record.get("count", -1)) != len(expected):
            raise RuntimeError(f"{split} materialization count disagrees with manifest")
        root = manifest_root / "clip_store" / split
        if not root.is_dir() or not (root / "data.bin").is_file():
            raise FileNotFoundError(f"materialized {split} store is missing: {root}")
    return manifest, materialization


def _index_ids(dataset: ClipMMapDataset) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in dataset._index)


def _load_data(manifest_root: Path) -> PilotData:
    manifest, materialization = _validate_manifest(manifest_root)
    train_root = manifest_root / "clip_store" / "train"
    valid_root = manifest_root / "clip_store" / "valid"
    _reject_test_path(train_root, "train store")
    _reject_test_path(valid_root, "validation store")
    train = ClipMMapDataset(train_root)
    valid = ClipMMapDataset(valid_root)
    expected_train = tuple(str(value) for value in manifest["source_splits"]["train"]["sample_ids"])
    expected_valid = tuple(str(value) for value in manifest["source_splits"]["valid"]["sample_ids"])
    actual_train = _index_ids(train)
    actual_valid = _index_ids(valid)
    if actual_train != expected_train:
        train.close()
        valid.close()
        raise RuntimeError("train materialized index differs from the frozen manifest")
    if actual_valid != expected_valid:
        train.close()
        valid.close()
        raise RuntimeError("validation materialized index differs from the frozen manifest")
    train_index_hash = _sha256(train_root / "index.txt")
    valid_index_hash = _sha256(valid_root / "index.txt")
    data_contract = {
        "schema": "pvb.dit.state_detail.t1_data_contract.v1",
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "materialization_sha256": materialization["materialization_sha256"],
        "train": materialization["splits"]["train"],
        "valid": materialization["splits"]["valid"],
        "train_index_sha256": train_index_hash,
        "valid_index_sha256": valid_index_hash,
        "test_opened": False,
    }
    return PilotData(
        manifest_root=manifest_root,
        manifest=manifest,
        materialization=materialization,
        train=train,
        valid=valid,
        data_hash=contract_hash(data_contract),
        train_index_hash=train_index_hash,
        valid_index_hash=valid_index_hash,
    )


def _load_approved_codec(
    *,
    candidate: str,
    result_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> FrozenCodec:
    ratio = 2 if candidate == "ratio2_state_detail" else 4
    result_path = result_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("approved T1 result or checkpoint is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise RuntimeError(f"approved result status is not passed: {result.get('status')!r}")
    if result.get("mode") != candidate:
        raise RuntimeError("approved result mode does not match candidate")
    recorded_checkpoint_hash = result.get("best_checkpoint_sha256")
    actual_checkpoint_hash = _sha256(checkpoint_path)
    if recorded_checkpoint_hash != actual_checkpoint_hash:
        raise RuntimeError("approved checkpoint SHA256 disagrees with result.json")
    if Path(str(result.get("best_checkpoint", ""))).name != checkpoint_path.name:
        raise RuntimeError("checkpoint is not the recorded best checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("approved codec checkpoint is not a mapping")
    if payload.get("schema_version") != "pvb.codec.checkpoint.v2":
        raise RuntimeError("approved codec checkpoint schema is not v2")
    model_contract = payload.get("model_contract")
    model_state = payload.get("model_state")
    if not isinstance(model_contract, Mapping) or not isinstance(model_state, Mapping):
        raise RuntimeError("approved checkpoint lacks model contract or model state")
    constructor = model_contract.get("constructor", {})
    if constructor.get("temporal_codec_mode") != candidate:
        raise RuntimeError("checkpoint temporal codec mode does not match candidate")
    if int(constructor.get("temporal_ratio", -1)) != ratio:
        raise RuntimeError("checkpoint temporal ratio does not match candidate")
    if constructor.get("spatial_backbone") != "torchmd_et":
        raise RuntimeError("pilot requires the frozen torchmd_et frame encoder")
    if constructor.get("coordinate_stem") != "centered_vector":
        raise RuntimeError("pilot requires the repaired centered_vector codec contract")
    model = PVBCodecModel.from_model_contract(model_contract)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    codec_state_hash = _state_hash(model)
    frame_hash = _state_hash(model.frame_encoder)
    expected_frame_hash = result.get("frame_encoder_state_hash_after")
    if expected_frame_hash and frame_hash != expected_frame_hash:
        raise RuntimeError("loaded frame-encoder hash differs from approved result")
    source_hash = str(result.get("frame_encoder_source_hash", ""))
    if not source_hash:
        raise RuntimeError("approved result lacks frame-encoder source hash")
    return FrozenCodec(
        model=model,
        candidate=candidate,
        ratio=ratio,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        result_sha256=_sha256(result_path),
        checkpoint_sha256=actual_checkpoint_hash,
        model_contract_hash=contract_hash(model_contract),
        codec_state_hash=codec_state_hash,
        frame_encoder_state_hash=frame_hash,
        frame_encoder_source_hash=source_hash,
    )


def _make_sampler(
    dataset: ClipMMapDataset,
    *,
    seed: int,
    clips_per_trajectory: int,
) -> TrajectoryCappedBatchSampler:
    sampler = TrajectoryCappedBatchSampler(
        dataset,
        max_tokens=MAX_TOKENS,
        clips_per_trajectory=clips_per_trajectory,
        seed=seed,
        shuffle=False,
    )
    sampler.set_epoch(0)
    return sampler


def _batch_schedule_hash(
    sampler: TrajectoryCappedBatchSampler,
    *,
    epoch: int,
) -> str:
    sampler.set_epoch(epoch)
    return _current_batch_schedule_hash(sampler)


def _current_batch_schedule_hash(sampler: TrajectoryCappedBatchSampler) -> str:
    """Hash the materialized schedule that the next batch will consume."""

    return _canonical_hash(
        {
            "sampler": sampler.state_dict(),
            "selected_sample_ids": list(sampler.selected_sample_ids),
            "batches": [list(batch) for batch in sampler.global_batches],
        }
    )


def _make_validation_plan(dataset: ClipMMapDataset) -> ValidationPlan:
    selected: list[tuple[str, str, int, int]] = []
    for index, row in enumerate(dataset._index):
        sample_id = str(row[0])
        match = SAMPLE_ID_RE.match(sample_id)
        if match is None:
            raise RuntimeError(f"validation sample id has an unsupported form: {sample_id}")
        window = int(match.group("window"))
        if window in VALIDATION_WINDOWS:
            selected.append(
                (
                    match.group("system"),
                    match.group("replica"),
                    window,
                    index,
                )
            )
    selected.sort()
    systems = tuple(sorted({item[0] for item in selected}))
    if len(systems) != 8:
        raise RuntimeError(f"validation plan must contain eight systems, got {systems}")
    expected = {
        (system, replica, window)
        for system in systems
        for replica in ("R1", "R2", "R3")
        for window in VALIDATION_WINDOWS
    }
    actual = {(system, replica, window) for system, replica, window, _ in selected}
    if actual != expected:
        raise RuntimeError("validation plan does not contain all replicas and predefined windows")
    selected_indices = tuple(item[3] for item in selected)
    selected_ids = tuple(str(dataset._index[index][0]) for index in selected_indices)
    subset = Subset(dataset, list(selected_indices))
    sampler = TaskAwareClipBatchSampler(
        subset,
        max_tokens=MAX_TOKENS,
        seed=20260907,
        shuffle=False,
        replacement=False,
        oversize_policy="error",
    )
    batches = tuple(tuple(int(index) for index in batch) for batch in sampler.global_batches)
    schedule_hash = _canonical_hash(
        {
            "windows": list(VALIDATION_WINDOWS),
            "sample_ids": list(selected_ids),
            "batches": [list(batch) for batch in batches],
        }
    )
    return ValidationPlan(
        subset=subset,
        batches=batches,
        selected_dataset_indices=selected_indices,
        selected_sample_ids=selected_ids,
        systems=systems,
        windows=VALIDATION_WINDOWS,
        schedule_hash=schedule_hash,
    )


def _encode_batch(
    dataset: Any,
    indices: Sequence[int],
    *,
    codec: FrozenCodec,
    adapter: StateDetailLatentAdapter,
    data_hash: str,
    device: torch.device,
) -> tuple[Any, ClipBatch, ClipBatch]:
    records = [dataset[int(index)] for index in indices]
    batch_cpu = collate_clip_records(records)
    codec.model.prepare_batch(batch_cpu)
    batch = batch_cpu.to(device, non_blocking=True)
    with torch.no_grad():
        latent = codec.model.encode(batch)
    latent_batch = adapter.pack(
        latent,
        codec_hash=codec.codec_state_hash,
        data_hash=data_hash,
        loss_mask=batch.loss_mask,
    )
    return latent, batch_cpu, batch


def _observed_batch(
    latent_batch: Any,
    batch: ClipBatch,
    *,
    adapter: StateDetailLatentAdapter,
    history_frames: int,
) -> Any:
    condition = build_observation_condition(
        latent_batch,
        history_frames=history_frames,
        coordinates=batch.x,
        frame_mask=batch.frame_mask,
        loss_mask=batch.loss_mask,
    )
    return latent_batch.with_observation(
        condition.latent_observation_mask,
        sample_origin=condition.sample_origin,
    )


def _fit_statistics(
    *,
    data: PilotData,
    codec: FrozenCodec,
    adapter: StateDetailLatentAdapter,
    ratio: int,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[LatentStatistics, str]:
    statistics_path = output_dir / "statistics.pt"
    sampler = _make_sampler(
        data.train,
        seed=seed,
        clips_per_trajectory=STATS_CLIPS_PER_TRAJECTORY,
    )
    schedule_hash = _batch_schedule_hash(sampler, epoch=0)
    provenance = {
        "schema": "pvb.dit.state_detail.t1_train_statistics.v1",
        "source_split": "train",
        "manifest_content_sha256": data.manifest["manifest_content_sha256"],
        "materialization_sha256": data.materialization["materialization_sha256"],
        "data_hash": data.data_hash,
        "codec_candidate": codec.candidate,
        "codec_checkpoint_sha256": codec.checkpoint_sha256,
        "codec_state_hash": codec.codec_state_hash,
        "adapter_contract_hash": contract_hash(adapter.contract()),
        "stats_schedule_hash": schedule_hash,
        "train_clips_per_trajectory": STATS_CLIPS_PER_TRAJECTORY,
        "statistics_status": "production_t1_train_only",
    }
    if statistics_path.is_file():
        state = torch.load(statistics_path, map_location="cpu", weights_only=False)
        statistics = LatentStatistics.from_state_dict(state)
        if statistics.ratio != ratio or statistics.provenance != provenance:
            raise RuntimeError("existing statistics artifact provenance differs from this pilot run")
        return statistics, schedule_hash
    sampler.set_epoch(0)
    def encoded_batches() -> Iterable[Any]:
        for batch_number, indices in enumerate(sampler.global_batches, 1):
            latent, _batch_cpu, batch = _encode_batch(
                data.train,
                indices,
                codec=codec,
                adapter=adapter,
                data_hash=data.data_hash,
                device=device,
            )
            yield adapter.pack(
                latent,
                codec_hash=codec.codec_state_hash,
                data_hash=data.data_hash,
                loss_mask=batch.loss_mask,
            )
            if batch_number % 25 == 0:
                print(json.dumps({
                    "event": "statistics_progress",
                    "candidate": codec.candidate,
                    "batches": batch_number,
                    "schedule_hash": schedule_hash,
                }, sort_keys=True), flush=True)
    statistics = LatentStatistics.fit(
        encoded_batches(),
        ratio=ratio,
        provenance=provenance,
    ).to(device="cpu")
    _atomic_torch_save(statistics_path, statistics.state_dict())
    restored = LatentStatistics.from_state_dict(
        torch.load(statistics_path, map_location="cpu", weights_only=False)
    )
    if restored.hash != statistics.hash:
        raise RuntimeError("statistics artifact did not round-trip")
    return restored, schedule_hash


def _build_trainer(
    *,
    codec: FrozenCodec,
    adapter: StateDetailLatentAdapter,
    statistics: LatentStatistics,
    candidate: str,
    data_hash: str,
    target_steps: int,
    seed: int,
    output_root: Path,
    device: torch.device,
    execution_backend: str = "reference",
) -> DiTTrainer:
    config = DiTTrainConfig(
        ratio=codec.ratio,
        mode=candidate,
        max_steps=target_steps,
        seed=seed,
        amp=device.type == "cuda",
        output_root=str(output_root),
        data_hash=data_hash,
        codec_hash=codec.codec_state_hash,
        stats_hash=statistics.hash,
        observation_mixture=HISTORY_SCHEDULE,
        metadata={
            "phase": "t1_pilot",
            "pilot_schema": PILOT_SCHEMA,
            "candidate": candidate,
            "seed": seed,
            "statistics_hash": statistics.hash,
        },
    )
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=256,
        vector_width=128,
        depth=4,
        heads=8,
        ffn_multiplier=4,
        dropout=0.0,
        execution_backend=execution_backend,
    ).to(device)
    return DiTTrainer(
        model,
        adapter,
        config=config,
        statistics=statistics.to(device=device),
        codec=codec.model,
        frame_encoder=codec.model.frame_encoder,
    )


def _checkpoint_payload(
    trainer: DiTTrainer,
    *,
    candidate: str,
    data: PilotData,
    codec: FrozenCodec,
    statistics: LatentStatistics,
    train_schedule_hash: str,
    validation: ValidationPlan,
    seed: int,
    cursor: Mapping[str, int],
    target_steps: int,
    validation_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    payload["pilot"] = {
        "schema": PILOT_SCHEMA,
        "candidate": candidate,
        "seed": int(seed),
        "target_steps": int(target_steps),
        "data_hash": data.data_hash,
        "manifest_content_sha256": data.manifest["manifest_content_sha256"],
        "materialization_sha256": data.materialization["materialization_sha256"],
        "codec_checkpoint_sha256": codec.checkpoint_sha256,
        "codec_state_hash": codec.codec_state_hash,
        "frame_encoder_state_hash": codec.frame_encoder_state_hash,
        "statistics_hash": statistics.hash,
        "train_schedule_hash": train_schedule_hash,
        "validation_schedule_hash": validation.schedule_hash,
        "validation_windows": list(validation.windows),
        "history_schedule": list(HISTORY_SCHEDULE),
        "cursor": {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
        "validation_history": [dict(item) for item in validation_history],
        "test_opened": False,
    }
    return payload


def _restore_resume(
    trainer: DiTTrainer,
    checkpoint_path: Path,
    *,
    candidate: str,
    data: PilotData,
    codec: FrozenCodec,
    statistics: LatentStatistics,
    train_schedule_hash: str,
    validation: ValidationPlan,
    seed: int,
    target_steps: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    payload = trainer.load_checkpoint(checkpoint_path, map_location=trainer.device)
    pilot = payload.get("pilot")
    if not isinstance(pilot, Mapping) or pilot.get("schema") != PILOT_SCHEMA:
        raise RuntimeError("resume checkpoint lacks the pilot contract")
    expected = {
        "candidate": candidate,
        "seed": int(seed),
        "target_steps": int(target_steps),
        "data_hash": data.data_hash,
        "manifest_content_sha256": data.manifest["manifest_content_sha256"],
        "materialization_sha256": data.materialization["materialization_sha256"],
        "codec_checkpoint_sha256": codec.checkpoint_sha256,
        "codec_state_hash": codec.codec_state_hash,
        "frame_encoder_state_hash": codec.frame_encoder_state_hash,
        "statistics_hash": statistics.hash,
        "train_schedule_hash": train_schedule_hash,
        "validation_schedule_hash": validation.schedule_hash,
        "validation_windows": list(validation.windows),
        "history_schedule": list(HISTORY_SCHEDULE),
        "test_opened": False,
    }
    for key, value in expected.items():
        if pilot.get(key) != value:
            raise RuntimeError(f"resume checkpoint pilot contract differs at {key}")
    cursor = pilot.get("cursor")
    if not isinstance(cursor, Mapping):
        raise RuntimeError("resume checkpoint lacks a schedule cursor")
    if int(cursor.get("epoch", -1)) < 0 or int(cursor.get("batch_index", -1)) < 0:
        raise RuntimeError("resume checkpoint has an invalid schedule cursor")
    history = pilot.get("validation_history", [])
    if not isinstance(history, list):
        raise RuntimeError("resume checkpoint validation history is invalid")
    return {
        "epoch": int(cursor["epoch"]),
        "batch_index": int(cursor["batch_index"]),
    }, [dict(item) for item in history]


def _next_train_batch(
    dataset: ClipMMapDataset,
    sampler: TrajectoryCappedBatchSampler,
    cursor: dict[str, int],
) -> tuple[tuple[int, ...], str]:
    epoch = int(cursor["epoch"])
    sampler.set_epoch(epoch)
    batches = tuple(tuple(int(index) for index in batch) for batch in sampler.global_batches)
    if not batches:
        raise RuntimeError("training sampler produced no batches")
    if int(cursor["batch_index"]) >= len(batches):
        cursor["epoch"] = epoch + 1
        cursor["batch_index"] = 0
        sampler.set_epoch(epoch + 1)
        batches = tuple(tuple(int(index) for index in batch) for batch in sampler.global_batches)
    indices = batches[int(cursor["batch_index"])]
    cursor["batch_index"] += 1
    return indices, _current_batch_schedule_hash(sampler)


def _validation_rf_loss(
    trainer: DiTTrainer,
    plan: ValidationPlan,
    *,
    data_hash: str,
    codec: FrozenCodec,
    adapter: StateDetailLatentAdapter,
    device: torch.device,
    seed: int,
    step: int,
) -> dict[str, Any]:
    trainer.model.eval()
    field_numerators = {name: 0.0 for name in ("state_h", "detail_h", "state_v", "detail_v")}
    field_counts = {name: 0 for name in field_numerators}
    batch_count = 0
    sample_count = 0
    for history_offset, history_frames in enumerate(HISTORY_SCHEDULE):
        for batch_offset, local_indices in enumerate(plan.batches):
            latent, _batch_cpu, batch = _encode_batch(
                plan.subset,
                local_indices,
                codec=codec,
                adapter=adapter,
                data_hash=data_hash,
                device=device,
            )
            latent_batch = adapter.pack(
                latent,
                codec_hash=codec.codec_state_hash,
                data_hash=data_hash,
                loss_mask=batch.loss_mask,
            )
            observed_batch = _observed_batch(
                latent_batch,
                batch,
                adapter=adapter,
                history_frames=history_frames,
            )
            normalized = trainer._normalise_batch(observed_batch)
            generator = _make_generator(
                device,
                seed + 1000003 + int(step) * 17 + history_offset * 101 + batch_offset,
            )
            with torch.no_grad(), trainer.autocast_context():
                sample = trainer.flow.sample(normalized, generator=generator)
                prediction = trainer.model(
                    normalized.with_fields(sample.interpolated),
                    sample.tau,
                )
                loss = trainer.flow.loss(prediction, sample.target, normalized)
            for name, value in loss.fields.items():
                count = int(loss.valid_elements[name])
                field_counts[name] += count
                field_numerators[name] += float(value.detach().float().cpu()) * count
            batch_count += 1
            sample_count += int(batch.batch_size)
    field_losses = {
        name: field_numerators[name] / max(field_counts[name], 1)
        for name in field_numerators
    }
    return {
        "step": int(step),
        "history_frames": list(HISTORY_SCHEDULE),
        "total": sum(field_losses.values()) / 4.0,
        "fields": field_losses,
        "valid_elements": field_counts,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "protocol": {
            "observed_interval": "[0,H)",
            "future_interval": "[H,16)",
            "history_schedule": list(HISTORY_SCHEDULE),
        },
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return None if not values else sum(values) / len(values)


@torch.no_grad()
def _generated_validation(
    trainer: DiTTrainer,
    plan: ValidationPlan,
    *,
    data_hash: str,
    codec: FrozenCodec,
    adapter: StateDetailLatentAdapter,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    trainer.model.eval()
    output: dict[str, Any] = {
        "validation_windows": list(plan.windows),
        "systems": list(plan.systems),
        "sample_count": len(plan.selected_sample_ids),
        "schedule_hash": plan.schedule_hash,
        "history": {},
    }
    for history_frames in HISTORY_SCHEDULE:
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for batch_offset, local_indices in enumerate(plan.batches):
            oracle_latent, _batch_cpu, batch = _encode_batch(
                plan.subset,
                local_indices,
                codec=codec,
                adapter=adapter,
                data_hash=data_hash,
                device=device,
            )
            latent_batch = adapter.pack(
                oracle_latent,
                codec_hash=codec.codec_state_hash,
                data_hash=data_hash,
                loss_mask=batch.loss_mask,
            )
            observed_batch = _observed_batch(
                latent_batch,
                batch,
                adapter=adapter,
                history_frames=history_frames,
            )
            with trainer.autocast_context():
                generated_latent, generation_meta = generate_state_detail_latent(
                    trainer.model,
                    adapter,
                    observed_batch,
                    trainer.statistics,
                    steps=16,
                    seed=seed + 2000003 + history_frames * 1000 + batch_offset,
                )
            result = evaluate_oracle_vs_generated(
                codec.model,
                oracle_latent=oracle_latent,
                generated_latent=generated_latent,
                batch=batch,
                latent_flow_loss={},
                trunk_runtime={"sampling_steps": 16},
                history_frames=history_frames,
            )
            rows.append({
                "sample_ids": list(batch.sample_id),
                "generation": generation_meta,
                "evaluation": result.as_dict(),
            })
        output["history"][str(history_frames)] = {
            "future_aligned_rmsd": _mean_metric(
                rows,
                ("evaluation", "generated_result", "future", "aligned_rmsd"),
            ),
            "future_drmsd": _mean_metric(
                rows,
                ("evaluation", "generated_result", "future", "drmsd"),
            ),
            "generation_gap_future_aligned_rmsd": _mean_metric(
                rows,
                ("evaluation", "generation_gap", "future", "aligned_rmsd"),
            ),
            "generation_gap_future_drmsd": _mean_metric(
                rows,
                ("evaluation", "generation_gap", "future", "drmsd"),
            ),
            "rows": rows,
            "elapsed_s": time.perf_counter() - started,
            "frame_intervals": {
                "observed": [0, history_frames],
                "future": [history_frames, 16],
                "boundary": None if history_frames == 0 else [history_frames - 1, history_frames],
                "full_diagnostic": [0, 16],
            },
        }
    return output


def _resume_check(
    *,
    checkpoint_path: Path,
    trainer: DiTTrainer,
    data: PilotData,
    codec: FrozenCodec,
    statistics: LatentStatistics,
    validation: ValidationPlan,
    train_sampler: TrajectoryCappedBatchSampler,
    train_schedule_hash: str,
    candidate: str,
    seed: int,
    target_steps: int,
    output_root: Path,
    device: torch.device,
    execution_backend: str = "reference",
) -> dict[str, Any]:
    fresh_adapter = StateDetailLatentAdapter(
        codec_width=128,
        scalar_width=256,
        vector_width=128,
        ratio=codec.ratio,
    ).to(device)
    fresh_trainer = _build_trainer(
        codec=codec,
        adapter=fresh_adapter,
        statistics=statistics,
        candidate=candidate,
        data_hash=data.data_hash,
        target_steps=target_steps,
        seed=seed,
        output_root=output_root,
        device=device,
        execution_backend=execution_backend,
    )
    resume_cursor, _history = _restore_resume(
        fresh_trainer,
        checkpoint_path,
        candidate=candidate,
        data=data,
        codec=codec,
        statistics=statistics.to(device=device),
        train_schedule_hash=train_schedule_hash,
        validation=validation,
        seed=seed,
        target_steps=target_steps,
    )
    loaded_step = int(fresh_trainer.step)
    if loaded_step != int(trainer.step):
        raise RuntimeError("fresh resume trainer did not restore the current pilot step")
    indices, schedule_hash = _next_train_batch(data.train, train_sampler, resume_cursor)
    if schedule_hash != train_schedule_hash and int(resume_cursor["epoch"]) == 0:
        raise RuntimeError("resume schedule hash changed")
    history_frames = HISTORY_SCHEDULE[loaded_step % len(HISTORY_SCHEDULE)]
    latent, _batch_cpu, batch = _encode_batch(
        data.train,
        indices,
        codec=codec,
        adapter=fresh_adapter,
        data_hash=data.data_hash,
        device=device,
    )
    latent_batch = fresh_adapter.pack(
        latent,
        codec_hash=codec.codec_state_hash,
        data_hash=data.data_hash,
        loss_mask=batch.loss_mask,
    )
    observed_batch = _observed_batch(
        latent_batch,
        batch,
        adapter=fresh_adapter,
        history_frames=history_frames,
    )
    _synchronize(device)
    log = fresh_trainer.train_step(observed_batch)
    _synchronize(device)
    if int(fresh_trainer.step) != loaded_step + 1:
        raise RuntimeError("fresh resume trainer did not complete one additional step")
    if not math.isfinite(float(log["loss"])):
        raise FloatingPointError("resume-check loss is not finite")
    if fresh_trainer.frozen_state_hashes() != trainer.frozen_state_hashes():
        raise RuntimeError("frozen hashes changed during resume check")
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_step": loaded_step,
        "resumed_step": int(fresh_trainer.step),
        "resume_loss": float(log["loss"]),
        "history_frames": history_frames,
        "schedule_cursor_after_step": resume_cursor,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("ratio2_state_detail", "ratio4_state_detail"), required=True)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--codec-result", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-backend", choices=("reference", "factorized_v2"), default="reference")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--evaluate-generated", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("the real-T1 pilot requires an actual CUDA device")
    if int(args.steps) < 1 or int(args.steps) > 5000:
        raise ValueError("pilot steps must be in [1,5000]")
    if args.profile and int(args.steps) != 200:
        raise ValueError("a pilot profile must use exactly 200 steps")
    if not args.profile and int(args.steps) < 1000:
        raise ValueError("a non-profile pilot run requires at least 1000 steps")
    if int(args.validation_interval) < 1:
        raise ValueError("validation interval must be positive")
    output_text = str(args.output_root)
    if "state_detail_codec_v2/t1" in output_text or "/test" in output_text.lower():
        raise RuntimeError("pilot output cannot be under the active T1 root or a test path")


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    device = torch.device(args.device)
    manifest_root = args.manifest_root.resolve()
    output_root = args.output_root
    run_name = args.run_name or (
        f"profile_{int(args.steps)}_seed{int(args.seed)}"
        if args.profile
        else f"pilot_{int(args.steps)}_seed{int(args.seed)}"
    )
    run_dir = output_root / args.candidate / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(manifest_root)
    codec = _load_approved_codec(
        candidate=args.candidate,
        result_path=args.codec_result,
        checkpoint_path=args.codec_checkpoint,
        device=device,
    )
    adapter = StateDetailLatentAdapter(
        codec_width=128,
        scalar_width=256,
        vector_width=128,
        ratio=codec.ratio,
    ).to(device)
    train_sampler = _make_sampler(
        data.train,
        seed=int(args.seed),
        clips_per_trajectory=TRAIN_CLIPS_PER_TRAJECTORY,
    )
    # Hash the same epoch-0 ordering that _next_train_batch will consume.
    # TrajectoryCappedBatchSampler materializes its shuffled batches lazily in
    # set_epoch(), so hashing immediately after construction can describe a
    # different ordering and falsely reject the first training batch.
    train_sampler.set_epoch(0)
    train_schedule_hash = _batch_schedule_hash(train_sampler, epoch=0)
    statistics_dir = output_root / args.candidate / "shared"
    statistics, statistics_schedule_hash = _fit_statistics(
        data=data,
        codec=codec,
        adapter=adapter,
        ratio=codec.ratio,
        output_dir=statistics_dir,
        seed=int(args.seed),
        device=device,
    )
    validation = _make_validation_plan(data.valid)
    trainer = _build_trainer(
        codec=codec,
        adapter=adapter,
        statistics=statistics,
        candidate=args.candidate,
        data_hash=data.data_hash,
        target_steps=int(args.steps),
        seed=int(args.seed),
        output_root=output_root,
        device=device,
        execution_backend=args.execution_backend,
    )
    cursor = {"epoch": 0, "batch_index": 0}
    validation_history: list[dict[str, Any]] = []
    resume_path = args.resume
    if resume_path is not None:
        cursor, validation_history = _restore_resume(
            trainer,
            resume_path.resolve(),
            candidate=args.candidate,
            data=data,
            codec=codec,
            statistics=statistics.to(device=device),
            train_schedule_hash=train_schedule_hash,
            validation=validation,
            seed=int(args.seed),
            target_steps=int(args.steps),
        )
    started = time.perf_counter()
    step_seconds: list[float] = []
    optimizer_step_seconds: list[float] = []
    if not validation_history:
        validation_history.append(
            _validation_rf_loss(
                trainer,
                validation,
                data_hash=data.data_hash,
                codec=codec,
                adapter=adapter,
                device=device,
                seed=int(args.seed),
                step=trainer.step,
            )
        )
    while trainer.step < int(args.steps):
        indices, current_schedule_hash = _next_train_batch(data.train, train_sampler, cursor)
        if current_schedule_hash != train_schedule_hash and int(cursor["epoch"]) == 0:
            raise RuntimeError("training schedule hash changed")
        history_frames = HISTORY_SCHEDULE[(trainer.step) % len(HISTORY_SCHEDULE)]
        iteration_started = time.perf_counter()
        latent, _batch_cpu, batch = _encode_batch(
            data.train,
            indices,
            codec=codec,
            adapter=adapter,
            data_hash=data.data_hash,
            device=device,
        )
        latent_batch = adapter.pack(
            latent,
            codec_hash=codec.codec_state_hash,
            data_hash=data.data_hash,
            loss_mask=batch.loss_mask,
        )
        observed_batch = _observed_batch(
            latent_batch,
            batch,
            adapter=adapter,
            history_frames=history_frames,
        )
        _synchronize(device)
        step_started = time.perf_counter()
        log = trainer.train_step(observed_batch)
        _synchronize(device)
        optimizer_elapsed = max(time.perf_counter() - step_started, 1.0e-9)
        elapsed = max(time.perf_counter() - iteration_started, 1.0e-9)
        step_seconds.append(elapsed)
        optimizer_step_seconds.append(optimizer_elapsed)
        if trainer.step % int(args.validation_interval) == 0 or trainer.step == int(args.steps):
            validation_row = _validation_rf_loss(
                trainer,
                validation,
                data_hash=data.data_hash,
                codec=codec,
                adapter=adapter,
                device=device,
                seed=int(args.seed),
                step=trainer.step,
            )
            validation_history.append(validation_row)
            payload = _checkpoint_payload(
                trainer,
                candidate=args.candidate,
                data=data,
                codec=codec,
                statistics=statistics,
                train_schedule_hash=train_schedule_hash,
                validation=validation,
                seed=int(args.seed),
                cursor=cursor,
                target_steps=int(args.steps),
                validation_history=validation_history,
            )
            _atomic_torch_save(run_dir / "latest.pt", payload)
            best = min(
                validation_history,
                key=lambda row: float(row.get("total", float("inf"))),
            )
            if validation_row is best:
                _atomic_torch_save(run_dir / "best_validation.pt", payload)
            _atomic_json(run_dir / "latest_metrics.json", {
                "schema": PILOT_SCHEMA,
                "candidate": args.candidate,
                "validation_history": validation_history,
                "last_train": log,
            })
            print(json.dumps({
                "event": "pilot_validation",
                "candidate": args.candidate,
                "step": trainer.step,
                "validation_total": validation_row["total"],
                "train_loss": log["loss"],
            }, sort_keys=True), flush=True)
        if not math.isfinite(float(log["loss"])):
            raise FloatingPointError("pilot loss is not finite")
    _synchronize(device)
    elapsed = time.perf_counter() - started
    latest_path = run_dir / "latest.pt"
    if not latest_path.is_file():
        payload = _checkpoint_payload(
            trainer,
            candidate=args.candidate,
            data=data,
            codec=codec,
            statistics=statistics,
            train_schedule_hash=train_schedule_hash,
            validation=validation,
            seed=int(args.seed),
            cursor=cursor,
            target_steps=int(args.steps),
            validation_history=validation_history,
        )
        _atomic_torch_save(latest_path, payload)
    resume_check = _resume_check(
        checkpoint_path=latest_path,
        trainer=trainer,
        data=data,
        codec=codec,
        statistics=statistics,
        validation=validation,
        train_sampler=train_sampler,
        train_schedule_hash=train_schedule_hash,
        candidate=args.candidate,
        seed=int(args.seed),
        target_steps=int(args.steps),
        output_root=output_root,
        device=device,
        execution_backend=args.execution_backend,
    )
    generated = None
    if args.evaluate_generated:
        best_path = run_dir / "best_validation.pt"
        if not best_path.is_file():
            raise RuntimeError("generated evaluation requires a selected best_validation.pt")
        trainer.load_checkpoint(best_path, map_location=device)
        generated = _generated_validation(
            trainer,
            validation,
            data_hash=data.data_hash,
            codec=codec,
            adapter=adapter,
            device=device,
            seed=int(args.seed),
        )
    codec_after = _state_hash(codec.model)
    frame_after = _state_hash(codec.model.frame_encoder)
    if codec_after != codec.codec_state_hash or frame_after != codec.frame_encoder_state_hash:
        raise RuntimeError("frozen codec/frame-encoder hash changed during pilot")
    summary = {
        "schema": PILOT_SCHEMA,
        "status": "PASS",
        "profile": bool(args.profile),
        "execution_backend": args.execution_backend,
        "candidate": args.candidate,
        "ratio": codec.ratio,
        "seed": int(args.seed),
        "target_steps": int(args.steps),
        "completed_steps": int(trainer.step),
        "manifest_root": str(manifest_root),
        "data_hash": data.data_hash,
        "manifest_content_sha256": data.manifest["manifest_content_sha256"],
        "materialization_sha256": data.materialization["materialization_sha256"],
        "train_index_sha256": data.train_index_hash,
        "valid_index_sha256": data.valid_index_hash,
        "test_opened": False,
        "codec": {
            "result_path": str(codec.result_path),
            "result_sha256": codec.result_sha256,
            "checkpoint_path": str(codec.checkpoint_path),
            "checkpoint_sha256": codec.checkpoint_sha256,
            "model_contract_hash": codec.model_contract_hash,
            "codec_state_hash": codec.codec_state_hash,
            "frame_encoder_state_hash": codec.frame_encoder_state_hash,
            "frame_encoder_source_hash": codec.frame_encoder_source_hash,
        },
        "statistics": {
            "path": str(statistics_dir / "statistics.pt"),
            "hash": statistics.hash,
            "schedule_hash": statistics_schedule_hash,
            "provenance": dict(statistics.provenance),
        },
        "training_schedule": {
            "max_tokens": MAX_TOKENS,
            "clips_per_trajectory": TRAIN_CLIPS_PER_TRAJECTORY,
            "seed": int(args.seed),
            "history_schedule": list(HISTORY_SCHEDULE),
            "schedule_hash": train_schedule_hash,
            "cursor": cursor,
        },
        "validation": {
            "systems": list(validation.systems),
            "windows": list(validation.windows),
            "sample_count": len(validation.selected_sample_ids),
            "schedule_hash": validation.schedule_hash,
            "history": validation_history,
        },
        "runtime": {
            "requested_device": str(args.device),
            "actual_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_device_index": int(torch.cuda.current_device()),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "autocast_dtype": str(torch.bfloat16),
            "train_step_seconds_mean": (
                sum(step_seconds) / len(step_seconds) if step_seconds else None
            ),
            "optimizer_step_seconds_mean": (
                sum(optimizer_step_seconds) / len(optimizer_step_seconds)
                if optimizer_step_seconds else None
            ),
            "train_steps_per_s": (
                len(step_seconds) / sum(step_seconds) if step_seconds else None
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "wall_time_s": elapsed,
        },
        "generated_validation": generated,
        "resume_check": resume_check,
        "warnings": [
            "Real T1 pilot execution evidence only; no test split access.",
            "R2/R4 are not ranked by smoke/profile losses.",
        ],
    }
    _atomic_json(run_dir / "pilot_summary.json", summary)
    data.train.close()
    data.valid.close()
    return summary


def main() -> None:
    args = _parser().parse_args()
    result = run_pilot(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
