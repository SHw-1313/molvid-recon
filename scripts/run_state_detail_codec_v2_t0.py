#!/usr/bin/env python3
"""Run the bounded zero-preserving state/detail codec v2 T0 protocol.

This script owns only the four deterministic temporal-codec controls in the v2
phase.  It never materializes the later T1 manifest and refuses a non-CUDA run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.codec_evaluation import (
    ALIGNED_RMSD_NAME,
    CONTACT_CUTOFF_ANGSTROM,
    CONTACT_EXCLUSION_RULE,
    RAW_RMSD_NAME,
    _aligned_rmsd,
    _clash_rate,
    _drmsd,
    _frequency_retention,
    _mask,
    _mean_records,
    _metrics,
    _rmsd,
    aligned_rmsf_metrics,
    contact_metrics,
    dynamic_acf_metrics,
)
from trainer.codec_losses import CodecLossWeights, compute_codec_losses
from trainer.codec_trainer import (
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
    prepare_batch_then_to_device,
)


ROOT = Path(__file__).resolve().parents[1]
STORE_ROOT = ROOT / "outputs/atlas_selected_trajectories/clip_store"
FRAME_ENCODER_CHECKPOINT = (
    ROOT / "outputs/state_detail_codec_v2/preflight/"
    "torchmd_frame_encoder_step_00004979.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/state_detail_codec_v2/t0"
DEVICE = torch.device("cuda:0")
SEED = 20260903
MAX_TOKENS = 80000
EPOCHS = 30
SAFETY_CAP = 6000
MICRO_STEPS = 500
MICRO_LOG_EVERY = 10
SYSTEMS = ("atlas_5e3e_A", "atlas_1v7r_A", "atlas_2wlt_A")
REPLICAS = ("R1", "R2", "R3")
TRAIN_WINDOWS = tuple(range(49))
HOLDOUT_WINDOWS = tuple(range(49, 62))
MODES = (
    "ratio1_state_detail",
    "ratio2_state_detail",
    "ratio4_state_detail",
    "ratio4_matched_pooling",
)
RATIOS = {
    "ratio1_state_detail": 1,
    "ratio2_state_detail": 2,
    "ratio4_state_detail": 4,
    "ratio4_matched_pooling": 4,
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_index(root: Path) -> list[dict[str, Any]]:
    path = root / "index.txt"
    if not path.is_file():
        raise FileNotFoundError(f"clip index is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"invalid clip index row {path}:{line_number}")
        sample_id, start, end, atoms, frames, bucket = fields
        rows.append(
            {
                "sample_id": sample_id,
                "start": int(start),
                "end": int(end),
                "atoms": int(atoms),
                "frames": int(frames),
                "time_bucket_id": bucket,
            }
        )
    return rows


def _split_sample_id(sample_id: str) -> tuple[str, str, int]:
    prefix, window_text = str(sample_id).rsplit("_w", 1)
    system, replica = prefix.rsplit("_", 1)
    if system not in SYSTEMS or replica not in REPLICAS or not window_text.isdigit():
        raise ValueError(f"invalid T0 sample id: {sample_id!r}")
    return system, replica, int(window_text)


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, windows: Sequence[int], split_name: str
) -> list[str]:
    expected = {
        f"{system}_{replica}_w{window:06d}"
        for system in SYSTEMS
        for replica in REPLICAS
        for window in windows
    }
    actual = [str(row["sample_id"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise RuntimeError(
            f"{split_name} violates the exact T0 manifest: count={len(actual)} "
            f"expected={len(expected)} missing={missing[:5]} extra={extra[:5]}"
        )
    for row in rows:
        _split_sample_id(str(row["sample_id"]))
        if int(row["frames"]) != 16 or str(row["time_bucket_id"]) != "dt_100ps":
            raise RuntimeError(
                f"{split_name} sample {row['sample_id']} is not T=16/dt_100ps"
            )
        if int(row["atoms"]) < 1 or int(row["end"]) <= int(row["start"]):
            raise RuntimeError(f"invalid {split_name} storage range: {row}")
    return actual


def build_manifest_contract() -> dict[str, Any]:
    """Independently verify the immutable 441/117 clip contract."""

    train_root = STORE_ROOT / "train"
    valid_root = STORE_ROOT / "valid"
    train_rows = _read_index(train_root)
    valid_rows = _read_index(valid_root)
    train_ids = _validate_rows(train_rows, windows=TRAIN_WINDOWS, split_name="train")
    valid_ids = _validate_rows(
        valid_rows, windows=HOLDOUT_WINDOWS, split_name="late holdout"
    )
    overlap = sorted(set(train_ids).intersection(valid_ids))
    if overlap:
        raise RuntimeError(f"train/late-holdout overlap: {overlap[:5]}")
    source_manifest = STORE_ROOT / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest is missing: {source_manifest}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    leakage = source.get("leakage_check", {})
    if leakage.get("intersection_count") != 0:
        raise RuntimeError("source manifest reports train/holdout leakage")
    selection = source.get("selection", {})
    if tuple(selection.get("systems", ())) != SYSTEMS:
        raise RuntimeError("source manifest system selection differs from T0")
    if tuple(selection.get("replicas", ())) != (1, 2, 3):
        raise RuntimeError("source manifest replica selection differs from T0")
    if selection.get("train_window_range") != [0, 48] or selection.get(
        "validation_window_range"
    ) != [49, 61]:
        raise RuntimeError("source manifest window ranges differ from T0")
    source_train_ids = set(source.get("train_sample_ids", ()))
    source_valid_ids = set(source.get("valid_sample_ids", ()))
    if source_train_ids != set(train_ids) or source_valid_ids != set(valid_ids):
        raise RuntimeError("clip-store indexes disagree with source manifest sample IDs")
    source_roots = sorted(
        {
            str(root)
            for track in source.get("tracks", ())
            for root in track.get("source_roots", ())
        }
    )
    payload = {
        "schema_version": "pvb.codec.state_detail.t0_manifest.v2",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_root": source.get("source_root"),
        "source_roots": source_roots,
        "source_roots_exist_at_audit": {
            root: Path(root).exists() for root in source_roots
        },
        "train_index": str(train_root / "index.txt"),
        "train_index_sha256": _sha256(train_root / "index.txt"),
        "valid_index": str(valid_root / "index.txt"),
        "valid_index_sha256": _sha256(valid_root / "index.txt"),
        "systems": list(SYSTEMS),
        "replicas": list(REPLICAS),
        "train_windows": [0, 48],
        "late_holdout_windows": [49, 61],
        "frames_per_clip": 16,
        "time_bucket_id": "dt_100ps",
        "train_count": len(train_ids),
        "late_holdout_count": len(valid_ids),
        "train_sample_ids": train_ids,
        "late_holdout_sample_ids": valid_ids,
        "intersection_count": len(overlap),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["built_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _runtime(device: torch.device) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "T0_HARD_FAIL: CUDA is unavailable; no CPU fallback is permitted"
        )
    try:
        import torch_cluster
        from torch_cluster import radius_graph
    except Exception as exc:
        raise RuntimeError(
            "T0_HARD_FAIL: CUDA torch_cluster.radius_graph is unavailable"
        ) from exc
    if not callable(radius_graph):
        raise RuntimeError("T0_HARD_FAIL: torch_cluster.radius_graph is not callable")
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "physical_device_mapping": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "gpu": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_cluster": getattr(torch_cluster, "__version__", "unknown"),
    }


def _seed(value: int = SEED) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _make_loader(
    source: Any,
    *,
    seed: int,
    shuffle: bool,
    replacement: bool,
) -> Any:
    return make_clip_dataloader(
        source,
        max_tokens=MAX_TOKENS,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=seed,
        shuffle=shuffle,
        replacement=replacement,
    )


def _dataset_ids(dataset: ClipMMapDataset) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in dataset._index)


def _scheduled_ids(loader: Any, dataset: ClipMMapDataset, epoch: int) -> list[str]:
    sampler = loader.batch_sampler
    sampler.set_epoch(int(epoch))
    flattened = [int(index) for batch in sampler.global_batches for index in batch]
    expected = list(range(len(dataset)))
    if sorted(flattened) != expected:
        raise RuntimeError(
            f"epoch {epoch} is not an exact no-replacement pass: "
            f"count={len(flattened)} expected={len(expected)}"
        )
    ids = _dataset_ids(dataset)
    return [ids[index] for index in flattened]


def _records_for_micro(dataset: ClipMMapDataset) -> list[dict[str, Any]]:
    selected_ids = tuple(
        f"{system}_R1_w000000" for system in SYSTEMS
    )
    by_id = {str(row[0]): index for index, row in enumerate(dataset._index)}
    missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"micro samples are missing: {missing}")
    records = [dataset[by_id[sample_id]] for sample_id in selected_ids]
    if tuple(str(record["sample_id"]) for record in records) != selected_ids:
        raise RuntimeError("micro sample order changed")
    batch = collate_clip_records(records)
    if batch.frames != 16 or batch.x.dtype != torch.float32:
        raise RuntimeError("micro protocol requires FP32 T=16 clips")
    if int(batch.x.shape[0] * batch.x.shape[1]) > MAX_TOKENS:
        raise RuntimeError("micro batch exceeds max_tokens")
    return records


def _make_model(
    mode: str,
    *,
    freeze_frame_encoder: bool = True,
    coordinate_stem: str = "centered_vector",
) -> PVBCodecModel:
    if mode not in MODES:
        raise ValueError(f"unsupported T0 mode: {mode}")
    checkpoint_hash = _sha256(FRAME_ENCODER_CHECKPOINT)
    return PVBCodecModel(
        hidden_channels=128,
        spatial_layers=2,
        spatial_backbone="torchmd_et",
        temporal_layers=1,
        temporal_ratio=None,
        temporal_codec_mode=mode,
        num_rbf=50,
        num_heads=8,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_neighbors=32,
        neighbor_backend="cuda_radius",
        bond_construction={"mode": "topology"},
        spatial_execution={"mode": "full"},
        frame_encoder_checkpoint=FRAME_ENCODER_CHECKPOINT,
        freeze_frame_encoder=freeze_frame_encoder,
        frame_encoder_source_hash=checkpoint_hash,
        coordinate_stem=coordinate_stem,
    )


def _make_config(
    max_steps: int, *, train_batches: int, schedule_steps: int | None = None
) -> CodecTrainConfig:
    if train_batches < 1:
        raise ValueError("train loader must contain at least one batch")
    total_steps = int(schedule_steps) if schedule_steps is not None else EPOCHS * int(train_batches)
    ten_percent = int(math.ceil(0.10 * total_steps))
    thirty_percent = int(math.ceil(0.30 * total_steps))
    return CodecTrainConfig(
        lr=1.0e-4,
        weight_decay=1.0e-6,
        max_steps=int(max_steps),
        grad_clip=1.0,
        warmup_steps=20,
        device="cuda",
        precision="fp32",
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 0.5, 1.0),),
        loss_schedule=(
            (
                0,
                CodecLossWeights(coordinate=1.0, local=0.1, bond=0.1),
            ),
            (
                ten_percent,
                CodecLossWeights(
                    coordinate=1.0,
                    local=0.1,
                    bond=0.1,
                    velocity=0.1,
                ),
            ),
            (
                thirty_percent,
                CodecLossWeights(
                    coordinate=1.0,
                    local=0.1,
                    bond=0.1,
                    velocity=0.1,
                    acceleration=0.05,
                ),
            ),
        ),
        normalization_min_count=32,
        normalization_epsilon=1.0e-6,
    )


def _single_proxy(batch: Any, sample_index: int, start: int, stop: int) -> dict[str, Any]:
    bonds = torch.as_tensor(batch.bond_index, device=batch.x.device, dtype=torch.long)
    if bonds.numel():
        keep = (batch.abid[bonds[0]] == sample_index) & (
            batch.abid[bonds[1]] == sample_index
        )
        bonds = bonds[:, keep] - int(start)
    else:
        bonds = torch.empty((2, 0), device=batch.x.device, dtype=torch.long)
    return {
        "x": batch.x[:, start:stop],
        "frame_mask": batch.frame_mask[sample_index : sample_index + 1],
        "abid": torch.zeros(stop - start, device=batch.x.device, dtype=torch.long),
        "loss_mask": batch.loss_mask[start:stop],
        "align_mask": batch.align_mask[start:stop],
        "bond_index": bonds,
        "time_ps": batch.time_ps[sample_index : sample_index + 1],
        "delta_time_ps": batch.delta_time_ps[sample_index : sample_index + 1],
        "time_bucket_id": (str(batch.time_bucket_id[sample_index]),),
        "task": batch.task[sample_index : sample_index + 1],
    }


def _boundary_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, ratio: int
) -> dict[str, float]:
    indices = [index for index in range(1, prediction.shape[0]) if index % ratio == 0]
    pred_jumps: list[torch.Tensor] = []
    target_jumps: list[torch.Tensor] = []
    for index in indices:
        valid = mask[index] & mask[index - 1]
        if torch.any(valid):
            pred_jumps.append(
                torch.linalg.vector_norm(
                    prediction[index, valid] - prediction[index - 1, valid], dim=-1
                ).mean()
            )
            target_jumps.append(
                torch.linalg.vector_norm(
                    target[index, valid] - target[index - 1, valid], dim=-1
                ).mean()
            )
    if not pred_jumps:
        return {"prediction": 0.0, "target": 0.0, "absolute_error": 0.0}
    pred_mean = torch.stack(pred_jumps).mean()
    target_mean = torch.stack(target_jumps).mean()
    return {
        "prediction": float(pred_mean.detach().cpu()),
        "target": float(target_mean.detach().cpu()),
        "absolute_error": float((pred_mean - target_mean).abs().detach().cpu()),
    }


def _extended_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Any,
    *,
    ratio: int,
) -> dict[str, Any]:
    mask = _mask(batch, prediction.shape[0], prediction.device)
    per_frame: dict[str, dict[str, float]] = {}
    for frame in range(prediction.shape[0]):
        contacts = contact_metrics(prediction, target, batch, mask, [frame])
        raw_rmsd = _rmsd(prediction, target, mask, [frame])
        per_frame[f"frame_{frame:02d}"] = {
            "legacy_raw_rmsd": raw_rmsd,
            "aligned_rmsd": _aligned_rmsd(prediction, target, batch, mask, [frame]),
            "centroid_gauge_raw_rmsd": raw_rmsd,
            "drmsd": _drmsd(prediction, target, mask, [frame]),
            "contact_error": contacts["contact_occupancy_mae"],
            **contacts,
            "clash_rate": _clash_rate(prediction, batch, mask, [frame]),
        }
    offsets: dict[str, dict[str, float]] = {}
    for offset in range(ratio):
        frames = [frame for frame in range(prediction.shape[0]) if frame % ratio == offset]
        raw_rmsd = _rmsd(prediction, target, mask, frames)
        offsets[f"offset_{offset}"] = {
            "legacy_raw_rmsd": raw_rmsd,
            "aligned_rmsd": _aligned_rmsd(prediction, target, batch, mask, frames),
            "centroid_gauge_raw_rmsd": raw_rmsd,
            "drmsd": _drmsd(prediction, target, mask, frames),
        }
    return {
        "per_frame": per_frame,
        "block_offsets": offsets,
        "rmsf": aligned_rmsf_metrics(prediction, target, batch, mask),
        "lag1_acf": dynamic_acf_metrics(prediction, target, batch, mask),
        "frequency_retention": _frequency_retention(prediction, target, mask),
        "block_boundary_jump": _boundary_metrics(prediction, target, mask, ratio),
    }


def _nested_mean(values: Sequence[Any]) -> Any:
    if not values:
        return {}
    first = values[0]
    if first is None:
        return None
    if isinstance(first, (str, bool)):
        if any(value != first for value in values[1:]):
            raise ValueError("cannot average inconsistent non-numeric evaluation metadata")
        return first
    if isinstance(first, Mapping):
        keys = tuple(first)
        return {key: _nested_mean([value[key] for value in values]) for key in keys}
    return sum(float(value) for value in values) / len(values)


def _detail_summary(output: Any) -> dict[str, Any]:
    latent = output.latent
    diagnostics = output.diagnostics or {}
    result = {
        key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in diagnostics.items()
    }
    if hasattr(latent, "detail_h") and latent.detail_h is not None:
        result["detail_bank_zero_fraction"] = float(
            (latent.detail_h == 0).float().mean().detach().cpu()
        )
        result["detail_valid_blocks"] = int(latent.detail_valid.sum().item())
        result["detail_total_blocks"] = int(latent.detail_valid.numel())
    else:
        result["detail_bank_zero_fraction"] = None
        result["detail_valid_blocks"] = 0
        result["detail_total_blocks"] = 0
    return result


def _evaluate_loader(
    trainer: CodecTrainer,
    loader: Any,
    dataset: ClipMMapDataset,
    *,
    epoch: int,
    ratio: int,
    detailed: bool = False,
) -> dict[str, Any]:
    expected_ids = _scheduled_ids(loader, dataset, epoch)
    expected_set = set(expected_ids)
    seen: list[str] = []
    standard_records: list[dict[str, Any]] = []
    extended_records: list[dict[str, Any]] = []
    per_system_standard: dict[str, list[dict[str, Any]]] = {}
    per_system_extended: dict[str, list[dict[str, Any]]] = {}
    loss_sums: dict[str, float] = {}
    counterfactual_full: list[dict[str, Any]] = []
    counterfactual_zero: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    trainer.model.eval()
    sample_count = 0
    batch_count = 0
    for batch in loader:
        batch_ids = [str(value) for value in batch.sample_id]
        seen.extend(batch_ids)
        moved = prepare_batch_then_to_device(
            trainer.model, batch, trainer.device, non_blocking=False
        )
        with torch.no_grad():
            output = trainer.model(moved)
            losses = compute_codec_losses(
                output,
                moved,
                weights=trainer.config.weights_at(trainer.step),
                normalization=trainer.normalization_stats,
            )
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * len(batch_ids)
        detail_records.append(_detail_summary(output))
        full_batch_metrics = _metrics(output.x_hat, moved.x, moved)
        counterfactual_full.append(full_batch_metrics)
        if hasattr(output.latent, "detail_h") and output.latent.detail_h is not None:
            with torch.no_grad():
                detail_zero = trainer.model.state_detail_codec.decode_detail_zero(
                    output.latent
                )
            counterfactual_zero.append(_metrics(detail_zero.x_hat, moved.x, moved))
        counts = batch.host_atom_counts
        start = 0
        for sample_index, sample_id in enumerate(batch_ids):
            stop = start + int(counts[sample_index])
            proxy = _single_proxy(moved, sample_index, start, stop)
            target = moved.x[:, start:stop]
            prediction = output.x_hat[:, start:stop]
            standard = _metrics(prediction, target, proxy)
            extended = (
                _extended_metrics(prediction, target, proxy, ratio=ratio)
                if detailed
                else {}
            )
            standard_records.append(standard)
            if detailed:
                extended_records.append(extended)
            system = sample_id.rsplit("_R", 1)[0]
            per_system_standard.setdefault(system, []).append(standard)
            if detailed:
                per_system_extended.setdefault(system, []).append(extended)
            start = stop
        sample_count += len(batch_ids)
        batch_count += 1
        del output, moved
    if (
        len(seen) != len(expected_ids)
        or set(seen) != expected_set
        or len(set(seen)) != len(seen)
    ):
        raise RuntimeError(
            f"epoch {epoch} evaluation is not exact: seen={len(seen)} "
            f"expected={len(expected_ids)} unique={len(set(seen))}"
        )
    if sample_count != len(expected_ids):
        raise RuntimeError("evaluation sample count disagrees with its sampler contract")
    denominator = float(max(sample_count, 1))
    return {
        "epoch": int(epoch),
        "sample_count": sample_count,
        "batch_count": batch_count,
        "exact_coverage": True,
        "loss": {key: value / denominator for key, value in loss_sums.items()},
        "metrics": _mean_records(standard_records),
        "extended_metrics": _nested_mean(extended_records) if detailed else {},
        "per_system": {
            system: {
                "sample_count": len(per_system_standard[system]),
                "metrics": _mean_records(per_system_standard[system]),
                "extended_metrics": _nested_mean(per_system_extended[system])
                if detailed
                else {},
            }
            for system in sorted(per_system_standard)
        },
        "counterfactual": {
            "detail_zero_available": bool(counterfactual_zero),
            "full": _mean_records(counterfactual_full),
            "detail_zero": _mean_records(counterfactual_zero)
            if counterfactual_zero
            else None,
        },
        "detail_summary": _nested_mean(detail_records),
    }


def _runtime_profile(model: PVBCodecModel, batch: Any) -> dict[str, Any]:
    model.eval()
    moved = prepare_batch_then_to_device(model, batch, DEVICE, non_blocking=False)
    with torch.no_grad():
        model(moved)
    torch.cuda.synchronize(DEVICE)

    def measure(function: Any, iterations: int = 3) -> float:
        function()
        torch.cuda.synchronize(DEVICE)
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize(DEVICE)
        return (time.perf_counter() - start) / iterations

    def spatial_only() -> None:
        centered, _origin = model._centered_batch(moved)
        with torch.no_grad():
            model.frame_encoder(centered)

    def end_to_end() -> None:
        with torch.no_grad():
            model(moved)

    centered_batch, origin = model._centered_batch(moved)
    with torch.no_grad():
        encoded = model.frame_encoder(centered_batch)

    def temporal_only() -> None:
        with torch.no_grad():
            latent = model.state_detail_codec.encode(
                encoded.h,
                encoded.v,
                time_ps=moved.time_ps,
                frame_mask=moved.frame_mask,
                abid=moved.abid,
                sample_origin=origin,
                topology=None,
            )
            model.state_detail_codec.decode(latent)

    spatial_seconds = measure(spatial_only)
    temporal_seconds = measure(temporal_only)
    end_to_end_seconds = measure(end_to_end)
    token_count = int(batch.x.shape[0] * batch.x.shape[1])
    sample_count = int(batch.batch_size)
    del encoded, centered_batch, moved
    return {
        "iterations": 3,
        "spatial_seconds": spatial_seconds,
        "temporal_only_seconds": temporal_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "temporal_fraction_of_end_to_end": temporal_seconds / max(end_to_end_seconds, 1.0e-8),
        "end_to_end_samples_per_s": sample_count / max(end_to_end_seconds, 1.0e-8),
        "end_to_end_tokens_per_s": token_count / max(end_to_end_seconds, 1.0e-8),
        "token_count": token_count,
        "sample_count": sample_count,
    }


def _gradient_summary(model: PVBCodecModel) -> dict[str, Any]:
    codec = model.state_detail_codec
    modules: dict[str, torch.nn.Module] = {}
    coordinate_stem = getattr(model, "coordinate_vector_stem", None)
    if coordinate_stem is not None:
        modules["coordinate_vector_stem"] = coordinate_stem
    if codec is not None:
        modules["state_detail_codec"] = codec
    if not modules:
        return {"parameter_count": 0, "all_finite_nonzero": False, "parameters": {}}
    parameters: dict[str, float | None] = {}
    module_parameter_counts: dict[str, int] = {}
    for module_name, module in modules.items():
        module_parameter_counts[module_name] = sum(
            parameter.numel() for parameter in module.parameters()
        )
        for name, parameter in module.named_parameters():
            parameter_name = f"{module_name}.{name}"
            if parameter.grad is None:
                parameters[parameter_name] = None
            else:
                parameters[parameter_name] = float(
                    parameter.grad.detach().float().abs().sum().cpu()
                )
    nonzero = [value for value in parameters.values() if value is not None and value > 0]
    finite = all(value is None or math.isfinite(value) for value in parameters.values())
    return {
        "parameter_count": sum(module_parameter_counts.values()),
        "module_parameter_counts": module_parameter_counts,
        "parameters": parameters,
        "all_finite_nonzero": bool(finite and len(nonzero) == len(parameters)),
    }


def _run_unfreeze_smoke(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Prove one end-to-end unfreezing backward without adding a T0 control."""

    _seed()
    loader = _make_loader(
        records, seed=SEED, shuffle=False, replacement=True
    )
    if len(loader) != 1:
        raise RuntimeError("unfreeze smoke must use one fixed batch")
    model = _make_model("ratio1_state_detail", freeze_frame_encoder=False)
    trainer = CodecTrainer(
        model,
        loader,
        # Keep a valid three-phase schedule even though the smoke itself takes
        # only one optimizer step.  The schedule axis is deliberately separate
        # from the one-step execution budget so the config validator still sees
        # strictly increasing phase boundaries.
        config=_make_config(1, train_batches=1, schedule_steps=10),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    batch = next(iter(loader))
    trainer.fit_normalization(loader)
    before = _state_hash(model.frame_encoder)
    metrics = trainer.optimizer_step(batch)
    spatial_gradients = {
        name: float(parameter.grad.detach().float().abs().sum().cpu())
        for name, parameter in model.frame_encoder.named_parameters()
        if parameter.grad is not None
    }
    if not spatial_gradients or not any(value > 0 for value in spatial_gradients.values()):
        raise RuntimeError("unfrozen frame encoder received no nonzero gradient")
    after = _state_hash(model.frame_encoder)
    result = {
        "status": "passed",
        "steps": trainer.step,
        "metrics": metrics,
        "frame_encoder_frozen": False,
        "frame_encoder_state_changed": before != after,
        "nonzero_spatial_gradient_count": sum(value > 0 for value in spatial_gradients.values()),
        "spatial_gradient_norm_sum": sum(spatial_gradients.values()),
    }
    _write_json(output_dir / "unfreeze_smoke.json", result)
    del trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _run_micro(mode: str, records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    _seed()
    loader = _make_loader(records, seed=SEED, shuffle=False, replacement=True)
    if len(loader) != 1:
        raise RuntimeError(f"{mode} micro protocol expected one batch, got {len(loader)}")
    model = _make_model(mode, freeze_frame_encoder=True)
    trainer = CodecTrainer(
        model,
        loader,
        config=_make_config(
            MICRO_STEPS, train_batches=1, schedule_steps=MICRO_STEPS
        ),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    trainer.fit_normalization(loader)
    batch = next(iter(loader))
    before_frame_hash = _state_hash(model.frame_encoder)
    trainer.model.eval()
    initial = _evaluate_loader(
        trainer,
        loader,
        _ListDataset(records),
        epoch=0,
        ratio=RATIOS[mode],
        detailed=True,
    )
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    log_path = output_dir / mode / "train_metrics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.run(max_steps=MICRO_STEPS, log_path=log_path, log_every=MICRO_LOG_EVERY)
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - start
    final = _evaluate_loader(
        trainer,
        loader,
        _ListDataset(records),
        epoch=1,
        ratio=RATIOS[mode],
        detailed=True,
    )
    after_frame_hash = _state_hash(model.frame_encoder)
    initial_total = float(initial["loss"]["total"])
    final_total = float(final["loss"]["total"])
    reduction = (initial_total - final_total) / max(abs(initial_total), 1.0e-8)
    checkpoint = output_dir / mode / f"codec_step_{trainer.step:08d}.pt"
    trainer.save_checkpoint(checkpoint)
    resume_model = _make_model(mode, freeze_frame_encoder=True)
    resume_loader = _make_loader(records, seed=SEED, shuffle=False, replacement=True)
    resume_trainer = CodecTrainer(
        resume_model,
        resume_loader,
        config=_make_config(
            MICRO_STEPS, train_batches=1, schedule_steps=MICRO_STEPS
        ),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    resume_trainer.load_checkpoint(checkpoint)
    loaded_step = resume_trainer.step
    resume_trainer.run(max_steps=MICRO_STEPS + 1)
    if resume_trainer.step != MICRO_STEPS + 1:
        raise RuntimeError(f"{mode} micro checkpoint resume failed")
    grad_summary = _gradient_summary(model)
    runtime = {
        "training_elapsed_s": elapsed,
        "steps_per_s": MICRO_STEPS / max(elapsed, 1.0e-8),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
    }
    result = {
        "mode": mode,
        "ratio": RATIOS[mode],
        "steps": trainer.step,
        "train_batch_count": len(loader),
        "sample_ids": [str(record["sample_id"]) for record in records],
        "initial": initial,
        "final": final,
        "initial_total": initial_total,
        "final_total": final_total,
        "total_reduction": reduction,
        "micro_overfit_quality_pass": bool(
            math.isfinite(final_total) and reduction >= 0.30
        ),
        "frame_encoder_source_hash": _sha256(FRAME_ENCODER_CHECKPOINT),
        "frame_encoder_state_hash_before": before_frame_hash,
        "frame_encoder_state_hash_after": after_frame_hash,
        "frame_encoder_unchanged": before_frame_hash == after_frame_hash,
        "new_module_gradients": grad_summary,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "resume": {
            "status": "passed",
            "loaded_step": loaded_step,
            "resumed_step": resume_trainer.step,
        },
        "runtime": runtime,
        "model_contract": model.model_contract(),
    }
    _write_json(output_dir / mode / "result.json", result)
    del resume_trainer, resume_model, resume_loader, trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


class _ListDataset:
    """Minimal exact dataset view used only by the fixed three-clip micro run."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)
        self._index = [(str(record["sample_id"]),) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)


def _train_mode(
    mode: str,
    *,
    train_dataset: ClipMMapDataset,
    valid_dataset: ClipMMapDataset,
    run_dir: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    mode_dir = run_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    train_loader = _make_loader(
        train_dataset, seed=SEED, shuffle=True, replacement=False
    )
    valid_loader = _make_loader(
        valid_dataset, seed=SEED + 1, shuffle=False, replacement=False
    )
    train_ids = _dataset_ids(train_dataset)
    valid_ids = _dataset_ids(valid_dataset)
    epoch_batch_counts: list[int] = []
    for epoch in range(EPOCHS):
        scheduled = _scheduled_ids(train_loader, train_dataset, epoch)
        if set(scheduled) != set(train_ids) or len(scheduled) != len(train_ids):
            raise RuntimeError(
                f"train loader epoch {epoch} does not cover its exact index"
            )
        epoch_batch_counts.append(len(train_loader))
    if set(_scheduled_ids(valid_loader, valid_dataset, 0)) != set(valid_ids):
        raise RuntimeError("holdout loader does not cover its exact index")
    total_steps = sum(epoch_batch_counts)
    if total_steps > SAFETY_CAP:
        raise RuntimeError(
            f"{mode} requires {total_steps} steps, exceeding safety cap {SAFETY_CAP}"
        )
    _seed()
    model = _make_model(mode, freeze_frame_encoder=True)
    trainer = CodecTrainer(
        model,
        train_loader,
        valid_loader,
        config=_make_config(
            total_steps,
            train_batches=epoch_batch_counts[0],
            schedule_steps=total_steps,
        ),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    _write_json(
        mode_dir / "protocol.json",
        dict(
            protocol,
            mode=mode,
            ratio=RATIOS[mode],
            batches_per_epoch_schedule=epoch_batch_counts,
            total_steps=total_steps,
        ),
    )
    trainer.fit_normalization(train_loader)
    first_batch = next(iter(train_loader))
    common_encoder_hash = _state_hash(model.frame_encoder)
    runtime_profile = _runtime_profile(model, first_batch)
    with torch.no_grad():
        initial = _evaluate_loader(
            trainer,
            valid_loader,
            valid_dataset,
            epoch=0,
            ratio=RATIOS[mode],
            detailed=False,
        )
    step_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_checkpoint: Path | None = None
    train_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for epoch in range(EPOCHS):
        expected = _scheduled_ids(train_loader, train_dataset, epoch)
        if len(train_loader) != epoch_batch_counts[epoch]:
            raise RuntimeError(
                f"{mode} epoch {epoch} batch schedule changed after it was frozen"
            )
        sampler = train_loader.batch_sampler
        sampler.set_epoch(epoch)
        seen: list[str] = []
        train_tokens = 0
        last_metrics: dict[str, float] = {}
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            seen.extend(str(value) for value in batch.sample_id)
            train_tokens += int(batch.x.shape[0] * batch.x.shape[1])
            last_metrics = trainer.optimizer_step(batch)
            trainer.batch_in_epoch = batch_index + 1
            if trainer.step % MICRO_LOG_EVERY == 0 or trainer.step == total_steps:
                step_rows.append(
                    {
                        "schema_version": "pvb.codec.train.v2",
                        "mode": mode,
                        "step": int(trainer.step),
                        "epoch": int(epoch + 1),
                        "batch_in_epoch": int(batch_index + 1),
                        "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                        "metrics": last_metrics,
                    }
                )
        if sorted(seen) != sorted(expected) or len(set(seen)) != len(seen):
            raise RuntimeError(
                f"{mode} epoch {epoch} coverage failed: seen={len(seen)} "
                f"expected={len(expected)} unique={len(set(seen))}"
            )
        trainer.epoch = epoch + 1
        trainer.batch_in_epoch = 0
        trainer._train_iterator = None
        eval_start = time.perf_counter()
        evaluation = _evaluate_loader(
            trainer,
            valid_loader,
            valid_dataset,
            epoch=epoch + 1,
            ratio=RATIOS[mode],
            detailed=(epoch + 1 == EPOCHS),
        )
        score = float(evaluation["metrics"]["future"]["drmsd"])
        if score < best_score:
            best_score = score
            best_checkpoint = mode_dir / "codec_best.pt"
            trainer.save_checkpoint(best_checkpoint)
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "step": int(trainer.step),
                "train_sample_count": len(seen),
                "train_unique_sample_count": len(set(seen)),
                "train_exact_coverage": True,
                "train_tokens": train_tokens,
                "epoch_elapsed_s": epoch_elapsed,
                "train_samples_per_s": len(seen) / max(epoch_elapsed, 1.0e-8),
                "train_tokens_per_s": train_tokens / max(epoch_elapsed, 1.0e-8),
                "last_train_metrics": last_metrics,
                "holdout_evaluation_elapsed_s": time.perf_counter() - eval_start,
                "holdout_evaluation": evaluation,
            }
        )
        _write_json(mode_dir / "evaluation_latest.json", epoch_rows[-1])
        mode_dir.joinpath("train_metrics.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in step_rows) + "\n",
            encoding="utf-8",
        )
    torch.cuda.synchronize(DEVICE)
    training_elapsed = time.perf_counter() - train_start
    final_train_start = time.perf_counter()
    final_train = _evaluate_loader(
        trainer,
        train_loader,
        train_dataset,
        epoch=EPOCHS,
        ratio=RATIOS[mode],
        detailed=False,
    )
    final_train_evaluation_elapsed = time.perf_counter() - final_train_start
    final_holdout = epoch_rows[-1]["holdout_evaluation"]
    final_checkpoint = mode_dir / f"codec_step_{trainer.step:08d}.pt"
    trainer.save_checkpoint(final_checkpoint)
    resume_model = _make_model(mode, freeze_frame_encoder=True)
    resume_loader = _make_loader(
        train_dataset, seed=SEED, shuffle=True, replacement=False
    )
    resume_trainer = CodecTrainer(
        resume_model,
        resume_loader,
        config=_make_config(
            total_steps,
            train_batches=len(resume_loader),
            schedule_steps=total_steps,
        ),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    resume_trainer.load_checkpoint(final_checkpoint)
    loaded_step = resume_trainer.step
    if loaded_step != total_steps:
        raise RuntimeError(
            f"{mode} checkpoint step {loaded_step} disagrees with frozen total_steps {total_steps}"
        )
    resume_trainer.run(max_steps=loaded_step + 1)
    if resume_trainer.step != loaded_step + 1:
        raise RuntimeError(f"{mode} checkpoint resume failed")
    resume_record = {
        "status": "passed",
        "loaded_step": loaded_step,
        "resumed_step": resume_trainer.step,
    }
    _write_json(mode_dir / "checkpoint_resume.json", resume_record)
    final_encoder_hash = _state_hash(model.frame_encoder)
    runtime = {
        **runtime_profile,
        "training_elapsed_s": training_elapsed,
        "final_train_evaluation_elapsed_s": final_train_evaluation_elapsed,
        "total_steps": total_steps,
        "batches_per_epoch": epoch_batch_counts[0],
        "batches_per_epoch_schedule": epoch_batch_counts,
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
        "train_tokens_total": sum(int(row["train_tokens"]) for row in epoch_rows),
        "train_tokens_per_s": sum(int(row["train_tokens"]) for row in epoch_rows)
        / max(training_elapsed, 1.0e-8),
    }
    _write_json(mode_dir / "runtime.json", runtime)
    _write_json(mode_dir / "evaluation.json", {
        "schema_version": "pvb.codec.state_detail.evaluation.v2",
        "mode": mode,
        "initial_holdout": initial,
        "epochs": epoch_rows,
        "final_train": final_train,
        "final_holdout": final_holdout,
    })
    result = {
        "mode": mode,
        "ratio": RATIOS[mode],
        "temporal_layers": 1,
        "spatial_backbone": "torchmd_et",
        "precision": "fp32",
        "max_tokens": MAX_TOKENS,
        "epochs": EPOCHS,
        "steps": int(trainer.step),
        "safety_cap_steps": SAFETY_CAP,
        "batches_per_epoch": epoch_batch_counts[0],
        "batches_per_epoch_schedule": epoch_batch_counts,
        "total_steps": total_steps,
        "train_sample_count": len(train_dataset),
        "holdout_sample_count": len(valid_dataset),
        "train_replacement": False,
        "holdout_replacement": False,
        "exact_epoch_coverage": all(row["train_exact_coverage"] for row in epoch_rows),
        "holdout_exact_evaluation": all(
            row["holdout_evaluation"]["exact_coverage"] for row in epoch_rows
        ),
        "common_frame_encoder_checkpoint": str(FRAME_ENCODER_CHECKPOINT),
        "common_frame_encoder_checkpoint_sha256": _sha256(FRAME_ENCODER_CHECKPOINT),
        "common_frame_encoder_state_hash": common_encoder_hash,
        "final_frame_encoder_state_hash": final_encoder_hash,
        "frame_encoder_frozen_and_unchanged": common_encoder_hash == final_encoder_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "initial_holdout": initial,
        "final_train": final_train,
        "final_holdout": final_holdout,
        "epoch_metrics": epoch_rows,
        "runtime": runtime,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": _sha256(final_checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_checkpoint_sha256": (
            _sha256(best_checkpoint) if best_checkpoint is not None else None
        ),
        "checkpoint_resume": resume_record,
        "paths": {
            "protocol": str(mode_dir / "protocol.json"),
            "runtime": str(mode_dir / "runtime.json"),
            "train_metrics": str(mode_dir / "train_metrics.jsonl"),
            "evaluation": str(mode_dir / "evaluation.json"),
            "checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        },
        "model_contract": model.model_contract(),
    }
    _write_json(mode_dir / "result.json", result)
    del resume_trainer, resume_model, resume_loader, trainer, model, train_loader, valid_loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _plot_reports(run_dir: Path, results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    colors = {
        mode: color
        for mode, color in zip(MODES, ("#4c78a8", "#f58518", "#54a24b", "#e45756"))
    }

    def save(figure: Any, stem: str, rect: tuple[float, float, float, float] | None = None) -> None:
        if rect is None:
            figure.tight_layout()
        else:
            figure.tight_layout(rect=rect)
        for suffix in ("png", "pdf"):
            path = run_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=160 if suffix == "png" else None)
            outputs.append(str(path))
        plt.close(figure)

    def loss_series(result: Mapping[str, Any]) -> tuple[list[int], list[float], list[int], list[float]]:
        train_rows = [
            json.loads(line)
            for line in Path(result["paths"]["train_metrics"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        holdout_steps = [0]
        holdout_loss = [float(result["initial_holdout"]["loss"]["total"])]
        cumulative_step = 0
        schedule = result.get("batches_per_epoch_schedule", [])
        for epoch_index, row in enumerate(result["epoch_metrics"]):
            if epoch_index < len(schedule):
                cumulative_step += int(schedule[epoch_index])
            else:
                cumulative_step = int(row["step"])
            holdout_steps.append(cumulative_step)
            holdout_loss.append(float(row["holdout_evaluation"]["loss"]["total"]))
        return (
            [int(row["step"]) for row in train_rows],
            [float(row["metrics"]["total"]) for row in train_rows],
            holdout_steps,
            holdout_loss,
        )

    figure, axis = plt.subplots(figsize=(14, 6))
    performance_text: list[str] = []
    for mode, result in results.items():
        train_steps, train_loss, holdout_steps, holdout_loss = loss_series(result)
        axis.plot(
            train_steps,
            train_loss,
            label=mode,
            color=colors[mode],
            linestyle="-",
        )
        axis.plot(
            holdout_steps,
            holdout_loss,
            label=f"{mode} holdout/test",
            color=colors[mode],
            linestyle="--",
            marker="o",
            markersize=2.5,
        )
        runtime = result["runtime"]
        performance_text.append(
            f"{mode}: train {runtime['training_elapsed_s'] / 60.0:.1f} min | "
            f"{runtime['train_tokens_per_s'] / 1000.0:.1f}k tok/s | "
            f"e2e {runtime['end_to_end_seconds']:.3f} s/batch | "
            f"peak {runtime['peak_allocated_memory_bytes'] / 2**30:.1f} GiB | "
            f"params {result['parameter_count']:,}/{result['trainable_parameter_count']:,} trainable"
        )
    axis.set_title("T0 total loss: train (solid) vs late holdout/test (dashed)")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("total loss (log scale)")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8, ncol=2)
    figure.text(0.5, 0.01, "\n".join(performance_text), ha="center", va="bottom", fontsize=8, family="monospace")
    save(figure, "loss_curves", rect=(0.0, 0.14, 1.0, 0.96))

    figure, axes = plt.subplots(2, 4, figsize=(20, 10))
    metric_specs = (
        ("Future aligned RMSD", lambda value: value["metrics"]["future"]["aligned_rmsd"]),
        ("Future centroid-gauge raw RMSD", lambda value: value["metrics"]["future"]["centroid_gauge_raw_rmsd"]),
        ("Future dRMSD", lambda value: value["metrics"]["future"]["drmsd"]),
        ("Future bond RMSE", lambda value: value["metrics"]["future"]["bond_rmse"]),
        ("Contact F1 (final detailed eval)", lambda value: value["extended_metrics"]["per_frame"]["frame_01"]["contact_f1"] if value.get("extended_metrics") else float("nan")),
        ("Velocity RMSE", lambda value: value["metrics"]["velocity_rmse"]),
        ("Acceleration RMSE", lambda value: value["metrics"]["acceleration_rmse"]),
        ("Frequency power ratio", lambda value: value["metrics"]["frequency_retention"]),
    )
    for axis, (title, value_fn) in zip(axes.ravel(), metric_specs):
        for mode, result in results.items():
            values = [value_fn(row["holdout_evaluation"]) for row in result["epoch_metrics"]]
            if all(not math.isfinite(float(value)) for value in values):
                continue
            axis.plot(
                [row["epoch"] for row in result["epoch_metrics"]],
                values,
                color=colors[mode],
                marker="o",
                markersize=2,
                label=mode,
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Å")
    axes[0, 1].set_ylabel("Å")
    axes[0, 2].set_ylabel("Å")
    axes[0, 3].set_ylabel("Å")
    axes[1, 0].set_ylabel("F1")
    axes[1, 1].set_ylabel("Å/ps")
    axes[1, 2].set_ylabel("Å/ps²")
    axes[1, 3].set_ylabel("predicted / target power")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("T0 holdout quality curves", fontsize=15)
    figure.text(0.5, 0.01, "\n".join(performance_text), ha="center", va="bottom", fontsize=8, family="monospace")
    save(figure, "evaluation_curves", rect=(0.0, 0.10, 1.0, 0.94))

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for mode, result in results.items():
        final_extended = result["final_holdout"]["extended_metrics"]
        offsets = final_extended["block_offsets"]
        labels = list(offsets)
        axes[0, 0].plot(
            labels,
            [offsets[label]["aligned_rmsd"] for label in labels],
            marker="o",
            color=colors[mode],
            label=mode,
        )
        axes[0, 1].plot(
            labels,
            [offsets[label]["drmsd"] for label in labels],
            marker="o",
            color=colors[mode],
            label=mode,
        )
        boundary = final_extended["block_boundary_jump"]
        axes[1, 0].bar(
            mode,
            boundary["absolute_error"],
            color=colors[mode],
        )
        final_metrics = result["final_holdout"]["metrics"]["future"]
        axes[0, 2].bar(
            mode,
            final_metrics["contact_f1"],
            color=colors[mode],
        )
        detail = result["final_holdout"]["detail_summary"]
        axes[1, 1].bar(
            mode,
            detail.get("encoded_detail_h_norm", 0.0) or 0.0,
            color=colors[mode],
        )
        axes[1, 2].bar(
            mode,
            final_extended["lag1_acf"]["dynamic_correlation"],
            color=colors[mode],
        )
    axes[0, 0].set_title("Final per-block-offset aligned RMSD")
    axes[0, 1].set_title("Final per-block-offset dRMSD")
    axes[0, 2].set_title("Final future contact F1")
    axes[1, 0].set_title("Block-boundary jump absolute error")
    axes[1, 1].set_title("Encoded detail norm")
    axes[1, 2].set_title("Final dynamic correlation")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[0, 0].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("T0 temporal structure and detail utilization", fontsize=15)
    save(figure, "block_detail_metrics")

    labels = list(results)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    specifications = (
        ("Training wall time", "minutes", [results[name]["runtime"]["training_elapsed_s"] / 60.0 for name in labels]),
        ("Train throughput", "k tokens/s", [results[name]["runtime"]["train_tokens_per_s"] / 1000.0 for name in labels]),
        ("End-to-end batch", "seconds", [results[name]["runtime"]["end_to_end_seconds"] for name in labels]),
        ("Peak allocated memory", "GiB", [results[name]["runtime"]["peak_allocated_memory_bytes"] / 2**30 for name in labels]),
    )
    for axis, (title, unit, values) in zip(axes, specifications):
        bars = axis.bar(labels, values, color=[colors[name] for name in labels])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("T0 runtime and memory performance", fontsize=15)
    save(figure, "performance_summary")

    for mode, result in results.items():
        figure, axis = plt.subplots(figsize=(9, 5))
        train_steps, train_loss, holdout_steps, holdout_loss = loss_series(result)
        axis.plot(
            train_steps,
            train_loss,
            color=colors[mode],
            linestyle="-",
            label="train",
        )
        axis.plot(
            holdout_steps,
            holdout_loss,
            color=colors[mode],
            linestyle="--",
            marker="o",
            markersize=3,
            label="late holdout/test",
        )
        axis.set_title(f"{mode}: train vs late holdout/test total loss")
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("total loss (log scale)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
        save(figure, f"loss_curve_{mode}")
    return outputs


def _write_reports(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    micro: Mapping[str, Any],
    unfreeze: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mode, result in results.items():
        final = result["final_holdout"]
        rows.append(
            {
                "mode": mode,
                "ratio": result["ratio"],
                "active_feature_volume_per_atom": result["model_contract"]["architecture"]["temporal"].get("active_feature_volume_per_atom"),
                "pooling_semantics": result["model_contract"]["architecture"]["temporal"].get("pooling_semantics"),
                "parameter_count": result["parameter_count"],
                "trainable_parameter_count": result["trainable_parameter_count"],
                "final_future_aligned_rmsd": final["metrics"]["future"]["aligned_rmsd"],
                "final_future_centroid_gauge_raw_rmsd": final["metrics"]["future"]["centroid_gauge_raw_rmsd"],
                "final_future_drmsd": final["metrics"]["future"]["drmsd"],
                "final_future_bond_rmse": final["metrics"]["future"]["bond_rmse"],
                "final_future_contact_precision": final["metrics"]["future"]["contact_precision"],
                "final_future_contact_recall": final["metrics"]["future"]["contact_recall"],
                "final_future_contact_f1": final["metrics"]["future"]["contact_f1"],
                "final_future_contact_jaccard": final["metrics"]["future"]["contact_jaccard"],
                "final_velocity_rmse": final["metrics"]["velocity_rmse"],
                "final_acceleration_rmse": final["metrics"]["acceleration_rmse"],
                "final_dynamic_correlation": final["extended_metrics"]["lag1_acf"]["dynamic_correlation"],
                "final_dynamic_acf_prediction": final["extended_metrics"]["lag1_acf"]["prediction"],
                "final_dynamic_acf_target": final["extended_metrics"]["lag1_acf"]["target"],
                "final_rmsf_correlation": final["extended_metrics"]["rmsf"]["correlation"],
                "final_frequency_power_ratio": final["metrics"]["frequency_retention"],
                "final_boundary_jump_prediction": final["extended_metrics"]["block_boundary_jump"]["prediction"],
                "final_boundary_jump_target": final["extended_metrics"]["block_boundary_jump"]["target"],
                "final_boundary_jump_absolute_error": final["extended_metrics"]["block_boundary_jump"]["absolute_error"],
                "final_detail_norm": final["detail_summary"].get("encoded_detail_h_norm"),
                "training_elapsed_s": result["runtime"]["training_elapsed_s"],
                "train_tokens_per_s": result["runtime"]["train_tokens_per_s"],
                "end_to_end_seconds": result["runtime"]["end_to_end_seconds"],
                "peak_allocated_memory_bytes": result["runtime"]["peak_allocated_memory_bytes"],
                "peak_reserved_memory_bytes": result["runtime"]["peak_reserved_memory_bytes"],
                "checkpoint": result["checkpoint"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "best_checkpoint": result["best_checkpoint"],
                "best_checkpoint_sha256": result["best_checkpoint_sha256"],
            }
        )
    plots = _plot_reports(run_dir, results)
    report = {
        "schema_version": "pvb.codec.state_detail.t0_report.v2",
        "status": "WAITING_FOR_OPERATOR_REVIEW",
        "protocol": protocol,
        "manifest": manifest,
        "micro_overfit": micro,
        "unfreeze_smoke": unfreeze,
        "backbones": results,
        "aggregate_rows": rows,
        "plots": plots,
        "limitations": [
            "one seed",
            "three selected systems and nine trajectories are a development comparison, not architecture selection",
            "no T1 manifest or later architecture phase was started",
        ],
    }
    with (run_dir / "aggregate_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(run_dir / "aggregate_comparison.json", report)
    lines = [
        "# State/detail codec v2 T0 comparison",
        "",
        "Status: **WAITING_FOR_OPERATOR_REVIEW**.",
        "",
        "The four controls use the same frozen TorchMD frame encoder, exact 441/117 lazy split, FP32 losses, and one seed.",
        "",
        "| Mode | Ratio | Active elements/atom | Params | Trainable | Aligned RMSD | Centroid-gauge raw RMSD | dRMSD | Bond RMSE | Contact F1 | Velocity RMSE | Accel RMSE | Dynamic corr | RMSF corr | Frequency power ratio | Detail norm | Train min | Tok/s | E2E s/batch | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['ratio']} | {row['active_feature_volume_per_atom']} | "
            f"{row['parameter_count']:,} | {row['trainable_parameter_count']:,} | "
            f"{row['final_future_aligned_rmsd']:.6g} | "
            f"{row['final_future_centroid_gauge_raw_rmsd']:.6g} | "
            f"{row['final_future_drmsd']:.6g} | {row['final_future_bond_rmse']:.6g} | "
            f"{row['final_future_contact_f1']:.6g} | {row['final_velocity_rmse']:.6g} | "
            f"{row['final_acceleration_rmse']:.6g} | {row['final_dynamic_correlation']:.6g} | "
            f"{row['final_rmsf_correlation']:.6g} | {row['final_frequency_power_ratio']:.6g} | "
            f"{row['final_detail_norm'] if row['final_detail_norm'] is not None else 'n/a'} | "
            f"{row['training_elapsed_s'] / 60.0:.2f} | {row['train_tokens_per_s']:.1f} | "
            f"{row['end_to_end_seconds']:.3f} | {row['peak_allocated_memory_bytes'] / 2**30:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed protocol",
            "",
            f"- Manifest: {manifest['train_count']} train / {manifest['late_holdout_count']} late holdout, intersection={manifest['intersection_count']}, T={manifest['frames_per_clip']}, {manifest['time_bucket_id']}.",
            f"- Loader: lazy mmap, replacement=false, exact epoch coverage, FP32, max_tokens={MAX_TOKENS}, 30 complete epochs, safety cap={SAFETY_CAP} steps.",
            f"- Seed={SEED}; spatial_backbone=torchmd_et; common frozen frame state={FRAME_ENCODER_CHECKPOINT}.",
            "- Coordinate reconstruction: shared centered-vector stem (centered_vector) and shared framewise equivariant decoder; no per-atom x0 anchor.",
            f"- Evaluator: {ALIGNED_RMSD_NAME}=per-frame Kabsch on align_mask scored on loss_mask; {RAW_RMSD_NAME}=direct centroid-gauge coordinate RMSD; contacts use cutoff={CONTACT_CUTOFF_ANGSTROM:g} Å with rule={CONTACT_EXCLUSION_RULE}; dynamic correlation uses mean-removed Kabsch-aligned frame-to-frame velocity.",
            "- Loss plots use a logarithmic y-axis; solid lines are train total loss and same-color dashed lines are late-holdout/test total loss aligned to optimizer step.",
            "- ratio4_matched_pooling is linear two-bank pooling and is latent-volume-matched, not parameter-matched; its trainable parameter count is reported separately.",
            "- Loss schedule is resolved after the loader length is frozen: 0%-10% coordinate/local/bond; 10%-30% adds velocity; 30%-100% adds acceleration.",
            "",
            "## Evidence paths",
            "",
            f"- Aggregate JSON: `{run_dir / 'aggregate_comparison.json'}`",
            f"- Aggregate CSV: `{run_dir / 'aggregate_comparison.csv'}`",
            f"- Plots: `{', '.join(plots)}`",
            f"- Micro: `{run_dir / 'micro'}`",
            f"- Unfreeze smoke: `{run_dir / 'unfreeze_smoke.json'}`",
            "",
            "## Stop condition",
            "",
            "No T1 manifest, T1 training, static/dynamic large-data run, DiT, observation adapter, forecasting, rollout, or later architecture phase was started.",
        ]
    )
    (run_dir / "aggregate_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    global STORE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--store-root", type=Path, default=STORE_ROOT)
    parser.add_argument("--skip-micro", action="store_true")
    parser.add_argument("--skip-unfreeze-smoke", action="store_true")
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument(
        "--plot-only-run",
        type=Path,
        default=None,
        help="regenerate plots from an existing completed run without training",
    )
    args = parser.parse_args(argv)
    if args.plot_only_run is not None:
        plot_run = args.plot_only_run if args.plot_only_run.is_absolute() else ROOT / args.plot_only_run
        if not plot_run.is_dir():
            raise FileNotFoundError(f"existing plot run is missing: {plot_run}")
        results = {
            mode: json.loads((plot_run / mode / "result.json").read_text(encoding="utf-8"))
            for mode in MODES
        }
        plots = _plot_reports(plot_run, results)
        print(json.dumps({"run_dir": str(plot_run), "plots": plots}, indent=2, sort_keys=True))
        return 0
    STORE_ROOT = args.store_root if args.store_root.is_absolute() else ROOT / args.store_root
    selected = MODES if args.mode == "all" else (args.mode,)
    runtime = _runtime(DEVICE)
    if not FRAME_ENCODER_CHECKPOINT.is_file():
        raise FileNotFoundError(f"shared frame checkpoint is missing: {FRAME_ENCODER_CHECKPOINT}")
    manifest = build_manifest_contract()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    run_dir = output_root / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing T0 run: {run_dir}")
    run_dir.mkdir(parents=True)
    protocol = {
        "schema_version": "pvb.codec.state_detail.t0_protocol.v2",
        "runtime": runtime,
        "seed": SEED,
        "modes": list(selected),
        "systems": list(SYSTEMS),
        "replicas": list(REPLICAS),
        "train_windows": [0, 48],
        "late_holdout_windows": [49, 61],
        "train_count": 441,
        "late_holdout_count": 117,
        "frames_per_clip": 16,
        "time_bucket_id": "dt_100ps",
        "lazy_loading": True,
        "replacement": False,
        "exact_epoch_coverage": True,
        "precision": "fp32",
        "max_tokens": MAX_TOKENS,
        "epochs": EPOCHS,
        "safety_cap_steps": SAFETY_CAP,
        "spatial_backbone": "torchmd_et",
        "common_frame_encoder_checkpoint": str(FRAME_ENCODER_CHECKPOINT),
        "common_frame_encoder_checkpoint_sha256": _sha256(FRAME_ENCODER_CHECKPOINT),
        "freeze_frame_encoder": True,
        "spatial_refiner": False,
        "temporal_layers": 1,
        "block_local": True,
        "cross_block_temporal_attention": False,
        "coordinate_stem": "centered_vector",
        "coordinate_decoder": "shared_framewise_equivariant",
        "matched_pooling_semantics": "linear_two_bank_pooling",
        "evaluator": {
            "aligned_rmsd": "per-frame Kabsch using align_mask; RMSD over loss_mask",
            "raw_rmsd": "centroid_gauge_raw_rmsd; direct coordinate difference",
            "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
            "contact_exclusion_rule": CONTACT_EXCLUSION_RULE,
            "dynamic_signal": "per-trajectory-Kabsch-aligned-frame-to-frame-velocity",
            "dynamic_units": "angstrom_per_ps",
            "dynamic_mean_removed": True,
            "rmsf_alignment": "per-frame-Kabsch-to-own-trajectory-frame0",
            "rmsf_aggregation": "sample_equal_mean",
            "frequency_metric": "predicted_over_target_nonzero_temporal_power",
            "frequency_values_above_one": "excessive_predicted_motion",
        },
        "loss_schedule_resolved_after_loader_length": True,
    }
    _write_json(run_dir / "protocol.json", protocol)
    _write_json(run_dir / "manifest_contract.json", manifest)
    train_dataset = ClipMMapDataset(STORE_ROOT / "train")
    valid_dataset = ClipMMapDataset(STORE_ROOT / "valid")
    if len(train_dataset) != 441 or len(valid_dataset) != 117:
        raise RuntimeError("dataset lengths do not match exact T0 counts")
    micro_dir = run_dir / "micro"
    micro_dir.mkdir()
    micro: dict[str, Any]
    unfreeze: dict[str, Any]
    try:
        records = _records_for_micro(train_dataset)
        if args.micro_only:
            micro_backbones: dict[str, Any] = {}
            for mode in selected:
                print(f"[t0-micro] starting {mode}", flush=True)
                micro_backbones[mode] = _run_micro(mode, records, micro_dir)
                print(f"[t0-micro] finished {mode}", flush=True)
            micro = {"status": "passed", "backbones": micro_backbones}
            _write_json(run_dir / "micro_summary.json", micro)
            print(json.dumps({"run_dir": str(run_dir), "micro": micro}, indent=2, sort_keys=True))
            return 0
        if args.skip_micro:
            micro = {"status": "skipped_by_operator_argument"}
        else:
            micro_backbones: dict[str, Any] = {}
            for mode in selected:
                print(f"[t0-micro] starting {mode}", flush=True)
                micro_backbones[mode] = _run_micro(mode, records, micro_dir)
                print(f"[t0-micro] finished {mode}", flush=True)
            micro = {"status": "passed", "backbones": micro_backbones}
        _write_json(run_dir / "micro_summary.json", micro)
        if args.skip_unfreeze_smoke:
            unfreeze = {"status": "skipped_by_operator_argument"}
        else:
            unfreeze = _run_unfreeze_smoke(records, run_dir)
        results: dict[str, Mapping[str, Any]] = {}
        for mode in selected:
            print(f"[t0] starting {mode}", flush=True)
            results[mode] = _train_mode(
                mode,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                run_dir=run_dir,
                protocol=protocol,
            )
            print(f"[t0] finished {mode}", flush=True)
        report = _write_reports(run_dir, protocol, manifest, micro, unfreeze, results)
    finally:
        train_dataset.close()
        valid_dataset.close()
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "manifest_train_count": manifest["train_count"],
                "manifest_late_holdout_count": manifest["late_holdout_count"],
                "modes": list(results),
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
