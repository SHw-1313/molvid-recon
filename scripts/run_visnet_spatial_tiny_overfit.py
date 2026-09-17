#!/usr/bin/env python3
"""Run and report the exact 441/117 ViSNet spatial tiny-overfit protocol."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.codec_evaluation import _mean_records, _metrics
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
DEFAULT_OUTPUT = ROOT / "outputs/visnet_spatial_v1/tiny_overfit"
DEVICE = torch.device("cuda:0")
SEED = 20260901
MAX_TOKENS = 80000
EPOCHS = 30
SAFETY_CAP = 6000
BACKBONES = ("torchmd_et", "visnet_radius", "visnet_bonded")
SYSTEMS = ("atlas_5e3e_A", "atlas_1v7r_A", "atlas_2wlt_A")
REPLICAS = ("R1", "R2", "R3")
TRAIN_WINDOWS = tuple(range(49))
HOLDOUT_WINDOWS = tuple(range(49, 62))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(root: Path) -> list[dict[str, Any]]:
    path = root / "index.txt"
    if not path.is_file():
        raise FileNotFoundError(f"clip index is missing: {path}")
    rows = []
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
    if replica not in REPLICAS or not window_text.isdigit():
        raise ValueError(f"invalid selected clip sample_id: {sample_id!r}")
    return system, replica, int(window_text)


def _validate_index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    windows: Sequence[int],
    split_name: str,
) -> list[str]:
    expected = {
        f"{system}_{replica}_w{window:06d}"
        for system in SYSTEMS
        for replica in REPLICAS
        for window in windows
    }
    actual = [str(row["sample_id"]) for row in rows]
    if set(actual) != expected or len(actual) != len(expected):
        missing = sorted(expected.difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise RuntimeError(
            f"{split_name} index violates the exact manifest contract: "
            f"count={len(actual)} expected={len(expected)} missing={missing[:5]} extra={extra[:5]}"
        )
    for row in rows:
        system, _replica, window = _split_sample_id(str(row["sample_id"]))
        if system not in SYSTEMS or window not in windows:
            raise RuntimeError(f"unexpected {split_name} sample: {row['sample_id']}")
        if int(row["frames"]) != 16 or str(row["time_bucket_id"]) != "dt_100ps":
            raise RuntimeError(
                f"{split_name} sample {row['sample_id']} is not T=16/dt_100ps"
            )
        if int(row["atoms"]) < 1 or int(row["end"]) <= int(row["start"]):
            raise RuntimeError(f"invalid {split_name} storage range: {row}")
    return actual


def build_manifest_contract() -> dict[str, Any]:
    """Build a small, hashed manifest from lazy-store index metadata."""

    train_rows = _read_index(STORE_ROOT / "train")
    valid_rows = _read_index(STORE_ROOT / "valid")
    train_ids = _validate_index_rows(
        train_rows, windows=TRAIN_WINDOWS, split_name="train"
    )
    valid_ids = _validate_index_rows(
        valid_rows, windows=HOLDOUT_WINDOWS, split_name="late holdout"
    )
    overlap = sorted(set(train_ids).intersection(valid_ids))
    if overlap:
        raise RuntimeError(f"train/holdout sample overlap: {overlap[:5]}")
    source_manifest = STORE_ROOT / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest is missing: {source_manifest}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    leakage = source.get("leakage_check", {})
    if leakage.get("intersection_count") != 0:
        raise RuntimeError("source manifest reports train/holdout leakage")
    payload: dict[str, Any] = {
        "schema_version": "pvb.codec.visnet_spatial.tiny_manifest.v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "train_index": str(STORE_ROOT / "train/index.txt"),
        "train_index_sha256": _sha256(STORE_ROOT / "train/index.txt"),
        "valid_index": str(STORE_ROOT / "valid/index.txt"),
        "valid_index_sha256": _sha256(STORE_ROOT / "valid/index.txt"),
        "systems": list(SYSTEMS),
        "replicas": list(REPLICAS),
        "train_windows": [min(TRAIN_WINDOWS), max(TRAIN_WINDOWS)],
        "late_holdout_windows": [min(HOLDOUT_WINDOWS), max(HOLDOUT_WINDOWS)],
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


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TINY_OVERFIT_HARD_FAIL: CUDA is required; no CPU fallback is permitted"
        )
    props = torch.cuda.get_device_properties(DEVICE)
    return {
        "device": str(DEVICE),
        "gpu": props.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "total_memory_bytes": int(props.total_memory),
    }


def make_model(spatial_backbone: str) -> PVBCodecModel:
    return PVBCodecModel(
        hidden_channels=128,
        spatial_layers=2,
        spatial_backbone=spatial_backbone,
        temporal_layers=1,
        temporal_ratio=1,
        num_rbf=50,
        num_heads=8,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_neighbors=32,
        neighbor_backend="cuda_radius",
        bond_construction={"mode": "topology"},
        spatial_execution={"mode": "full"},
    )


def make_config(max_steps: int) -> CodecTrainConfig:
    return CodecTrainConfig(
        lr=1.0e-4,
        weight_decay=1.0e-6,
        max_steps=max_steps,
        grad_clip=1.0,
        warmup_steps=20,
        device="cuda",
        precision="fp32",
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 0.5, 1.0),),
        loss_schedule=(
            (0, CodecLossWeights(coordinate=1.0)),
            (
                100,
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


def _dataset_ids(dataset: ClipMMapDataset) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in dataset._index)


def _scheduled_ids(loader: Any, dataset: ClipMMapDataset, epoch: int) -> list[str]:
    sampler = loader.batch_sampler
    sampler.set_epoch(int(epoch))
    schedule = sampler.global_batches
    indices = [int(index) for batch in schedule for index in batch]
    expected_indices = list(range(len(dataset)))
    if sorted(indices) != expected_indices:
        raise RuntimeError(
            f"epoch {epoch} schedule is not an exact no-replacement pass: "
            f"count={len(indices)} expected={len(expected_indices)}"
        )
    ids = _dataset_ids(dataset)
    return [ids[index] for index in indices]


def make_loaders() -> tuple[ClipMMapDataset, ClipMMapDataset, Any, Any]:
    train_dataset = ClipMMapDataset(STORE_ROOT / "train")
    valid_dataset = ClipMMapDataset(STORE_ROOT / "valid")
    train_loader = make_clip_dataloader(
        train_dataset,
        max_tokens=MAX_TOKENS,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=SEED,
        shuffle=True,
        replacement=False,
    )
    valid_loader = make_clip_dataloader(
        valid_dataset,
        max_tokens=MAX_TOKENS,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=SEED + 1,
        shuffle=False,
        replacement=False,
    )
    return train_dataset, valid_dataset, train_loader, valid_loader


def _single_metric_batch(batch: Any, sample_index: int, start: int, stop: int) -> dict[str, Any]:
    bonds = torch.as_tensor(batch.bond_index, device=batch.x.device, dtype=torch.long)
    if bonds.numel():
        keep = (batch.abid[bonds[0]] == sample_index) & (batch.abid[bonds[1]] == sample_index)
        bonds = bonds[:, keep] - int(start)
    else:
        bonds = torch.empty((2, 0), dtype=torch.long, device=batch.x.device)
    fields: dict[str, Any] = {
        "x": batch.x[:, start:stop],
        "frame_mask": batch.frame_mask[sample_index : sample_index + 1],
        "abid": torch.zeros(stop - start, dtype=torch.long, device=batch.x.device),
        "loss_mask": batch.loss_mask[start:stop],
        "bond_index": bonds,
        "time_ps": batch.time_ps[sample_index : sample_index + 1],
        "delta_time_ps": batch.delta_time_ps[sample_index : sample_index + 1],
        "task": batch.task[sample_index : sample_index + 1],
    }
    torsions = getattr(batch, "torsion_index", None)
    if torsions is not None:
        torsions = torch.as_tensor(torsions, device=batch.x.device, dtype=torch.long)
        if torsions.numel():
            keep = (batch.abid[torsions] == sample_index).all(dim=0)
            fields["torsion_index"] = torsions[:, keep] - int(start)
        else:
            fields["torsion_index"] = torch.empty(
                (4, 0), dtype=torch.long, device=batch.x.device
            )
    return fields


def _evaluate_loader(
    trainer: CodecTrainer,
    loader: Any,
    dataset: ClipMMapDataset,
    *,
    epoch: int,
) -> dict[str, Any]:
    expected_ids = _scheduled_ids(loader, dataset, epoch)
    expected_set = set(expected_ids)
    seen: list[str] = []
    metric_records: list[dict[str, Any]] = []
    per_system: dict[str, list[dict[str, Any]]] = {}
    loss_sums: dict[str, float] = {}
    sample_count = 0
    batch_count = 0
    trainer.model.eval()
    for batch in loader:
        batch_ids = [str(value) for value in batch.sample_id]
        seen.extend(batch_ids)
        moved = prepare_batch_then_to_device(
            trainer.model, batch, DEVICE, non_blocking=False
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
        atom_ptr = moved.atom_ptr
        for sample_index, sample_id in enumerate(batch_ids):
            start = int(atom_ptr[sample_index])
            stop = int(atom_ptr[sample_index + 1])
            proxy = _single_metric_batch(moved, sample_index, start, stop)
            target = moved.x[:, start:stop]
            prediction = output.x_hat[:, start:stop]
            metrics = _metrics(prediction, target, proxy)
            metric_records.append(metrics)
            system = sample_id.rsplit("_R", 1)[0]
            per_system.setdefault(system, []).append(metrics)
        sample_count += len(batch_ids)
        batch_count += 1
        del output, moved
    if len(seen) != len(expected_ids) or set(seen) != expected_set or len(set(seen)) != len(seen):
        raise RuntimeError(
            f"epoch {epoch} evaluation is not exact: seen={len(seen)} "
            f"expected={len(expected_ids)} unique={len(set(seen))}"
        )
    if sample_count != len(expected_ids):
        raise RuntimeError("evaluation sample count disagrees with the sampler contract")
    denominator = float(max(sample_count, 1))
    mean_loss = {key: value / denominator for key, value in loss_sums.items()}
    return {
        "epoch": int(epoch),
        "sample_count": sample_count,
        "batch_count": batch_count,
        "metrics": _mean_records(metric_records),
        "per_system": {
            system: {
                "sample_count": len(records),
                "metrics": _mean_records(records),
            }
            for system, records in sorted(per_system.items())
        },
        "loss": mean_loss,
        "exact_coverage": True,
    }


def _graph_diagnostics(model: PVBCodecModel, batch: Any) -> dict[str, Any]:
    moved = prepare_batch_then_to_device(model, batch, DEVICE, non_blocking=False)
    with torch.no_grad():
        encoded = model.frame_encoder(moved)
    graph = encoded.graph
    edge_count = int(graph.edge_index.shape[1])
    bond_count = int(graph.bond_index.shape[1])
    return {
        "backbone": encoded.spatial_backbone,
        "graph_mode": encoded.graph_mode,
        "backend_used": encoded.backend_used,
        "nodes": int(graph.num_nodes),
        "distance_edges": int(graph.distance_edge_index.shape[1]),
        "union_edges": edge_count,
        "replicated_covalent_edges": bond_count,
        "binary_covalent_edge_features": int(graph.bond_type.sum().item()),
        "topology_bond_coverage": (
            None
            if encoded.graph_mode == "native_radius"
            else int(graph.bond_type.sum().item()) / max(edge_count, 1)
        ),
    }


def _write_train_log(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_epoch_log(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _write_train_log(path, records)


def _quality_report(initial_train: Mapping[str, Any], final_train: Mapping[str, Any], initial_valid: Mapping[str, Any], final_valid: Mapping[str, Any]) -> dict[str, Any]:
    initial_train_total = float(initial_train["loss"]["total"])
    final_train_total = float(final_train["loss"]["total"])
    train_reduction = (initial_train_total - final_train_total) / max(abs(initial_train_total), 1.0e-8)
    initial_drmsd = float(initial_valid["metrics"]["future"]["drmsd"])
    final_drmsd = float(final_valid["metrics"]["future"]["drmsd"])
    drmsd_improvement = (initial_drmsd - final_drmsd) / max(abs(initial_drmsd), 1.0e-8)
    dynamic_metric_names = (
        "rmsd",
        "drmsd",
        "bond_rmse",
        "contact_error",
        "clash_rate",
        "torsion_change",
        "velocity_rmse",
        "acceleration_rmse",
    )
    regressions = []
    for name in dynamic_metric_names:
        if name in {"velocity_rmse", "acceleration_rmse"}:
            before = float(initial_valid["metrics"][name])
            after = float(final_valid["metrics"][name])
        else:
            before = float(initial_valid["metrics"]["future"][name])
            after = float(final_valid["metrics"]["future"][name])
        if not math.isfinite(after) or after > max(10.0 * max(before, 1.0e-8), before + 1.0e-3):
            regressions.append({"metric": name, "initial": before, "final": after})
    return {
        "initial_train_total": initial_train_total,
        "final_train_total": final_train_total,
        "train_total_reduction": train_reduction,
        "initial_holdout_future_drmsd": initial_drmsd,
        "final_holdout_future_drmsd": final_drmsd,
        "holdout_future_drmsd_improvement": drmsd_improvement,
        "catastrophic_regressions": regressions,
        "train_quality_pass": bool(train_reduction >= 0.60),
        "future_drmsd_quality_pass": bool(drmsd_improvement >= 0.10),
        "dynamic_geometric_stability_pass": not regressions,
        "tiny_overfit_quality_pass": bool(
            train_reduction >= 0.60 and drmsd_improvement >= 0.10 and not regressions
        ),
    }


def _resume_check(spatial_backbone: str, checkpoint: Path, train_dataset: ClipMMapDataset) -> dict[str, Any]:
    loader = make_clip_dataloader(
        train_dataset,
        max_tokens=MAX_TOKENS,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=SEED,
        shuffle=True,
        replacement=False,
    )
    model = make_model(spatial_backbone)
    trainer = CodecTrainer(
        model,
        loader,
        config=make_config(SAFETY_CAP),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    trainer.load_checkpoint(checkpoint)
    loaded_step = int(trainer.step)
    trainer.run(max_steps=loaded_step + 1)
    resumed_step = int(trainer.step)
    if resumed_step != loaded_step + 1:
        raise RuntimeError(f"{spatial_backbone} resume did not complete one update")
    del trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "status": "passed",
        "checkpoint_step": loaded_step,
        "resumed_step": resumed_step,
    }


def run_backbone(
    spatial_backbone: str,
    run_dir: Path,
    *,
    train_dataset: ClipMMapDataset,
    valid_dataset: ClipMMapDataset,
    train_loader: Any,
    valid_loader: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    backend_dir = run_dir / spatial_backbone
    backend_dir.mkdir(parents=True, exist_ok=False)
    _write_json(backend_dir / "protocol.json", dict(protocol, spatial_backbone=spatial_backbone))
    model = make_model(spatial_backbone)
    train_config = make_config(SAFETY_CAP)
    trainer = CodecTrainer(
        model,
        train_loader,
        valid_loader,
        config=train_config,
        device=DEVICE,
        non_blocking_transfer=False,
    )
    train_ids = _dataset_ids(train_dataset)
    valid_ids = _dataset_ids(valid_dataset)
    train_schedule_ids = _scheduled_ids(train_loader, train_dataset, 0)
    valid_schedule_ids = _scheduled_ids(valid_loader, valid_dataset, 0)
    if train_schedule_ids != list(train_ids) and set(train_schedule_ids) != set(train_ids):
        raise RuntimeError("train schedule does not cover the lazy train index")
    if set(valid_schedule_ids) != set(valid_ids):
        raise RuntimeError("holdout schedule does not cover the lazy valid index")
    first_batch = next(iter(train_loader))
    if first_batch.x.dtype != torch.float32:
        raise RuntimeError("tiny-overfit protocol requires FP32 clip coordinates")

    normalization_start = time.perf_counter()
    trainer.fit_normalization(train_loader)
    torch.cuda.synchronize(DEVICE)
    normalization_elapsed = time.perf_counter() - normalization_start
    diagnostics = _graph_diagnostics(model, first_batch)
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    initial_train_start = time.perf_counter()
    initial_train = _evaluate_loader(
        trainer, train_loader, train_dataset, epoch=0
    )
    torch.cuda.synchronize(DEVICE)
    initial_train_elapsed = time.perf_counter() - initial_train_start
    initial_valid_start = time.perf_counter()
    initial_valid = _evaluate_loader(
        trainer, valid_loader, valid_dataset, epoch=0
    )
    torch.cuda.synchronize(DEVICE)
    initial_valid_elapsed = time.perf_counter() - initial_valid_start

    batches_per_epoch = len(train_loader)
    total_steps = EPOCHS * batches_per_epoch
    if total_steps > SAFETY_CAP:
        raise RuntimeError(
            f"30 exact epochs require {total_steps} steps, exceeding safety cap {SAFETY_CAP}"
        )
    step_log: list[dict[str, Any]] = []
    epoch_log: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        epoch_tokens = 0
        expected_ids = _scheduled_ids(train_loader, train_dataset, epoch)
        seen_ids: list[str] = []
        last_metrics: dict[str, float] = {}
        sampler = train_loader.batch_sampler
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            seen_ids.extend(str(value) for value in batch.sample_id)
            epoch_tokens += int(batch.x.shape[0] * batch.x.shape[1])
            last_metrics = trainer.optimizer_step(batch)
            trainer.batch_in_epoch = batch_index + 1
            if trainer.step % 10 == 0:
                step_log.append(
                    {
                        "schema_version": "pvb.codec.train.v1",
                        "split": "train",
                        "step": int(trainer.step),
                        "epoch": int(epoch),
                        "batch_in_epoch": int(batch_index + 1),
                        "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                        "metrics": last_metrics,
                    }
                )
        if sorted(seen_ids) != sorted(expected_ids) or len(set(seen_ids)) != len(seen_ids):
            raise RuntimeError(
                f"train epoch {epoch} is not exact: seen={len(seen_ids)} "
                f"expected={len(expected_ids)} unique={len(set(seen_ids))}"
            )
        trainer.epoch = epoch + 1
        trainer.batch_in_epoch = 0
        trainer._train_iterator = None
        evaluation_start = time.perf_counter()
        evaluation = _evaluate_loader(
            trainer, valid_loader, valid_dataset, epoch=epoch + 1
        )
        torch.cuda.synchronize(DEVICE)
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_log.append(
            {
                "epoch": epoch + 1,
                "step": int(trainer.step),
                "train_sample_count": len(seen_ids),
                "train_unique_sample_count": len(set(seen_ids)),
                "train_exact_coverage": True,
                "train_tokens": int(epoch_tokens),
                "epoch_elapsed_s": epoch_elapsed,
                "train_tokens_per_s": epoch_tokens / max(epoch_elapsed, 1.0e-8),
                "last_train_metrics": last_metrics,
                "holdout_evaluation": evaluation,
                "holdout_evaluation_elapsed_s": time.perf_counter() - evaluation_start,
            }
        )
        _write_epoch_log(backend_dir / "epoch_metrics.jsonl", epoch_log)
        _write_train_log(backend_dir / "train_metrics.jsonl", step_log)
    torch.cuda.synchronize(DEVICE)
    training_elapsed = time.perf_counter() - train_start

    final_train_start = time.perf_counter()
    final_train = _evaluate_loader(
        trainer, train_loader, train_dataset, epoch=EPOCHS
    )
    torch.cuda.synchronize(DEVICE)
    final_train_elapsed = time.perf_counter() - final_train_start
    final_valid = epoch_log[-1]["holdout_evaluation"]
    quality = _quality_report(initial_train, final_train, initial_valid, final_valid)
    total_train_tokens = sum(int(row["train_tokens"]) for row in epoch_log)
    train_tokens_per_s = total_train_tokens / max(training_elapsed, 1.0e-8)
    checkpoint = backend_dir / f"codec_step_{trainer.step:08d}.pt"
    checkpoint_start = time.perf_counter()
    trainer.save_checkpoint(checkpoint)
    torch.cuda.synchronize(DEVICE)
    checkpoint_elapsed = time.perf_counter() - checkpoint_start
    resume_start = time.perf_counter()
    resume = _resume_check(spatial_backbone, checkpoint, train_dataset)
    resume_elapsed = time.perf_counter() - resume_start
    result = {
        "spatial_backbone": spatial_backbone,
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "precision": "fp32",
        "max_tokens": MAX_TOKENS,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-6,
        "warmup_steps": 20,
        "grad_clip": 1.0,
        "epochs": EPOCHS,
        "safety_cap_steps": SAFETY_CAP,
        "steps": int(trainer.step),
        "batches_per_epoch": batches_per_epoch,
        "train_sample_count": len(train_dataset),
        "holdout_sample_count": len(valid_dataset),
        "train_replacement": False,
        "holdout_replacement": False,
        "train_exact_coverage_all_epochs": True,
        "holdout_evaluated_all_epochs": True,
        "graph_diagnostics": diagnostics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_holdout": initial_valid,
        "final_holdout": final_valid,
        "epoch_metrics": epoch_log,
        "quality": quality,
        "resume": resume,
        "normalization_elapsed_s": normalization_elapsed,
        "initial_train_evaluation_elapsed_s": initial_train_elapsed,
        "initial_holdout_evaluation_elapsed_s": initial_valid_elapsed,
        "training_elapsed_s": training_elapsed,
        "final_train_evaluation_elapsed_s": final_train_elapsed,
        "checkpoint_elapsed_s": checkpoint_elapsed,
        "resume_elapsed_s": resume_elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
        "total_train_tokens": total_train_tokens,
        "train_tokens_per_s": train_tokens_per_s,
        "checkpoint": str(checkpoint),
        "train_metrics": str(backend_dir / "train_metrics.jsonl"),
        "epoch_metrics_path": str(backend_dir / "epoch_metrics.jsonl"),
        "model_contract": model.model_contract(),
    }
    _write_json(backend_dir / "result.json", result)
    del trainer, model, train_loader, valid_loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _load_micro_summary(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {"status": "not_supplied"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _write_plots(run_dir: Path, results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prepared: dict[str, dict[str, Any]] = {}
    metric_specs = (
        ("Future dRMSD", lambda metrics: metrics["future"]["drmsd"]),
        ("Future RMSD", lambda metrics: metrics["future"]["rmsd"]),
        ("Future bond RMSE", lambda metrics: metrics["future"]["bond_rmse"]),
        ("Future contact error", lambda metrics: metrics["future"]["contact_error"]),
        ("Velocity RMSE", lambda metrics: metrics["velocity_rmse"]),
        ("Acceleration RMSE", lambda metrics: metrics["acceleration_rmse"]),
    )
    for backbone, result in results.items():
        train_rows = [
            json.loads(line)
            for line in Path(result["train_metrics"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        epoch_rows = list(result["epoch_metrics"])
        prepared[backbone] = {
            "train_rows": train_rows,
            "epoch_rows": epoch_rows,
            "epochs": [int(row["epoch"]) for row in epoch_rows],
            "performance": {
                "parameter_count": int(result["parameter_count"]),
                "training_minutes": float(result["training_elapsed_s"]) / 60.0,
                "train_tokens_per_s": float(result["train_tokens_per_s"]),
                "holdout_eval_seconds_per_epoch": sum(
                    float(row["holdout_evaluation_elapsed_s"]) for row in epoch_rows
                )
                / max(len(epoch_rows), 1),
                "peak_allocated_gib": float(result["peak_allocated_memory_bytes"]) / 2**30,
                "peak_reserved_gib": float(result["peak_reserved_memory_bytes"]) / 2**30,
            },
            "metric_values": {
                title: [
                    float(value_fn(row["holdout_evaluation"]["metrics"]))
                    for row in epoch_rows
                ]
                for title, value_fn in metric_specs
            },
        }

    def save_figure(
        figure: Any,
        stem: str,
        *,
        rect: tuple[float, float, float, float] | None = None,
    ) -> list[str]:
        if rect is None:
            figure.tight_layout()
        else:
            figure.tight_layout(rect=rect)
        paths = [run_dir / f"{stem}.png", run_dir / f"{stem}.pdf"]
        for path in paths:
            figure.savefig(path)
        plt.close(figure)
        return [str(path) for path in paths]

    outputs: list[str] = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for backbone, item in prepared.items():
        result = results[backbone]
        train_rows = item["train_rows"]
        axes[0].plot(
            [row["step"] for row in train_rows],
            [row["metrics"]["total"] for row in train_rows],
            label=backbone,
        )
        epoch_rows = item["epoch_rows"]
        axes[1].plot(
            [0] + [row["epoch"] for row in epoch_rows],
            [result["initial_holdout"]["loss"]["total"]]
            + [row["holdout_evaluation"]["loss"]["total"] for row in epoch_rows],
            label=backbone,
        )
    axes[0].set_title("Training total loss")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Late-holdout total loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[0].legend()
    axes[1].legend()
    outputs.extend(save_figure(figure, "loss_curves"))

    performance_lines = []
    for backbone, item in prepared.items():
        performance = item["performance"]
        quality = results[backbone]["quality"]["tiny_overfit_quality_pass"]
        performance_lines.append(
            f"{backbone}: train {performance['training_minutes']:.1f} min | "
            f"{performance['train_tokens_per_s'] / 1000.0:.1f}k tokens/s | "
            f"eval {performance['holdout_eval_seconds_per_epoch']:.1f} s/epoch | "
            f"alloc/reserved {performance['peak_allocated_gib']:.1f}/{performance['peak_reserved_gib']:.1f} GiB | "
            f"params {performance['parameter_count']:,} | tiny {'pass' if quality else 'fail'}"
        )

    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for axis, (title, _value_fn) in zip(axes, metric_specs):
        for backbone, item in prepared.items():
            axis.plot(
                item["epochs"],
                item["metric_values"][title],
                marker="o",
                linewidth=1.6,
                markersize=3,
                label=backbone,
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("dRMSD")
    axes[1].set_ylabel("RMSD")
    axes[2].set_ylabel("RMSE")
    axes[3].set_ylabel("error")
    axes[4].set_ylabel("RMSE")
    axes[5].set_ylabel("RMSE")
    axes[0].legend(loc="best")
    figure.suptitle("Late-holdout evaluation metrics by epoch", fontsize=15)
    figure.text(
        0.5,
        0.01,
        chr(10).join(performance_lines),
        ha="center",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    outputs.extend(save_figure(figure, "eval_metrics", rect=(0.0, 0.12, 1.0, 0.94)))

    for backbone, item in prepared.items():
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()
        for axis, (title, _value_fn) in zip(axes, metric_specs):
            axis.plot(
                item["epochs"],
                item["metric_values"][title],
                color="#1f77b4",
                marker="o",
                linewidth=1.8,
                markersize=3,
            )
            axis.set_title(title)
            axis.set_xlabel("epoch")
            axis.grid(alpha=0.25)
        performance = item["performance"]
        quality = results[backbone]["quality"]["tiny_overfit_quality_pass"]
        figure.suptitle(
            f"{backbone} late-holdout evaluation" + chr(10) +
            f"train {performance['training_minutes']:.1f} min | "
            f"{performance['train_tokens_per_s'] / 1000.0:.1f}k tokens/s | "
            f"eval {performance['holdout_eval_seconds_per_epoch']:.1f} s/epoch | "
            f"alloc/reserved {performance['peak_allocated_gib']:.1f}/{performance['peak_reserved_gib']:.1f} GiB | "
            f"params {performance['parameter_count']:,} | tiny {'PASS' if quality else 'FAIL'}",
            fontsize=12,
        )
        outputs.extend(
            save_figure(
                figure,
                f"eval_metrics_{backbone}",
                rect=(0.0, 0.0, 1.0, 0.90),
            )
        )

    labels = list(prepared)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    bar_specs = (
        (
            "Training wall time",
            "min",
            [prepared[name]["performance"]["training_minutes"] for name in labels],
        ),
        (
            "Training throughput",
            "k tokens/s",
            [prepared[name]["performance"]["train_tokens_per_s"] / 1000.0 for name in labels],
        ),
        (
            "Holdout eval time",
            "s / epoch",
            [prepared[name]["performance"]["holdout_eval_seconds_per_epoch"] for name in labels],
        ),
    )
    for axis, (title, unit, values) in zip(axes[:3], bar_specs):
        bars = axis.bar(labels, values, color=("#4c78a8", "#f58518", "#54a24b"))
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis = axes[3]
    width = 0.36
    allocated = [prepared[name]["performance"]["peak_allocated_gib"] for name in labels]
    reserved = [prepared[name]["performance"]["peak_reserved_gib"] for name in labels]
    bars_alloc = axis.bar(x - width / 2.0, allocated, width, label="allocated")
    bars_reserved = axis.bar(x + width / 2.0, reserved, width, label="reserved")
    axis.set_title("Peak GPU memory")
    axis.set_ylabel("GiB")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    for bars in (bars_alloc, bars_reserved):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Training and evaluation runtime performance", fontsize=15)
    outputs.extend(
        save_figure(
            figure,
            "performance_summary",
            rect=(0.0, 0.0, 1.0, 0.92),
        )
    )
    return outputs


def _write_reports(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    micro_summary: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = {
        name: result
        for name, result in results.items()
        if implementation.get("status") == "passed"
        and result["quality"]["tiny_overfit_quality_pass"]
        and micro_summary.get("backbones", {}).get(name, {}).get(
            "micro_overfit_quality_pass", False
        )
    }
    best_name = (
        min(
            candidates,
            key=lambda name: (
                float(candidates[name]["quality"]["final_holdout_future_drmsd"]),
                float(candidates[name]["training_elapsed_s"]),
            ),
        )
        if candidates
        else None
    )
    radius = candidates.get("visnet_radius")
    bonded = candidates.get("visnet_bonded")
    radius_within_ten_percent = False
    bonded_comparison: dict[str, Any] = {"eligible": bonded is not None}
    if radius is not None and best_name is not None:
        radius_drmsd = float(radius["quality"]["final_holdout_future_drmsd"])
        best_drmsd = float(candidates[best_name]["quality"]["final_holdout_future_drmsd"])
        radius_within_ten_percent = radius_drmsd <= 1.10 * max(best_drmsd, 1.0e-8)
    if radius is not None and bonded is not None:
        radius_geometry = radius["final_holdout"]["metrics"]["future"]
        bonded_geometry = bonded["final_holdout"]["metrics"]["future"]
        radius_drmsd = float(radius_geometry["drmsd"])
        bonded_drmsd = float(bonded_geometry["drmsd"])
        radius_rmsd = float(radius_geometry["rmsd"])
        bonded_rmsd = float(bonded_geometry["rmsd"])
        drmsd_improvement = (radius_drmsd - bonded_drmsd) / max(abs(radius_drmsd), 1.0e-8)
        rmsd_improvement = (radius_rmsd - bonded_rmsd) / max(abs(radius_rmsd), 1.0e-8)
        step_time_ratio = float(bonded["training_elapsed_s"]) / max(
            float(radius["training_elapsed_s"]), 1.0e-8
        )
        bonded_comparison.update(
            {
                "future_drmsd_improvement": drmsd_improvement,
                "future_rmsd_improvement": rmsd_improvement,
                "training_time_ratio_bonded_to_radius": step_time_ratio,
                "meaningful_coherent_improvement": bool(
                    drmsd_improvement >= 0.05 and rmsd_improvement >= 0.05
                ),
                "within_1_5x_radius_time": bool(step_time_ratio <= 1.5),
            }
        )
    if radius is not None and radius_within_ten_percent:
        recommendation = "visnet_radius"
        recommendation_reason = (
            "visnet_radius is within 10% of the best eligible backend on late-holdout "
            "future dRMSD and has the simplest graph/data requirements"
        )
    elif (
        bonded is not None
        and bonded_comparison.get("meaningful_coherent_improvement", False)
        and bonded_comparison.get("within_1_5x_radius_time", False)
    ):
        recommendation = "visnet_bonded"
        recommendation_reason = (
            "visnet_bonded improves both future dRMSD and future RMSD by at least 5% "
            "over visnet_radius while staying within 1.5x its training time"
        )
    elif not (radius is not None or bonded is not None) and "torchmd_et" in candidates:
        recommendation = "torchmd_et"
        recommendation_reason = (
            "both ViSNet variants failed implementation or tiny-overfit quality gates"
        )
    elif best_name is not None:
        recommendation = best_name
        recommendation_reason = (
            "fallback to the best eligible final late-holdout future dRMSD after the "
            "phase recommendation rules did not select a preferred ViSNet variant"
        )
    else:
        recommendation = None
        recommendation_reason = (
            "no backend passed both the implementation and tiny-overfit quality gates"
        )
    rows = []
    for name, result in results.items():
        quality = result["quality"]
        rows.append(
            {
                "spatial_backbone": name,
                "parameter_count": int(result["parameter_count"]),
                "implementation_pass": implementation.get("status") == "passed",
                "micro_overfit_pass": bool(
                    micro_summary.get("backbones", {}).get(name, {}).get(
                        "micro_overfit_quality_pass", False
                    )
                ),
                "tiny_overfit_quality_pass": quality["tiny_overfit_quality_pass"],
                "train_total_reduction": quality["train_total_reduction"],
                "holdout_future_drmsd_improvement": quality["holdout_future_drmsd_improvement"],
                "final_holdout_future_drmsd": quality["final_holdout_future_drmsd"],
                "training_elapsed_s": result["training_elapsed_s"],
                "train_tokens_per_s": result["train_tokens_per_s"],
                "peak_allocated_memory_bytes": result["peak_allocated_memory_bytes"],
                "peak_reserved_memory_bytes": result["peak_reserved_memory_bytes"],
                "graph_mode": result["graph_diagnostics"]["graph_mode"],
                "topology_bond_coverage": result["graph_diagnostics"]["topology_bond_coverage"],
                "checkpoint": result["checkpoint"],
            }
        )
    with (run_dir / "aggregate_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots = _write_plots(run_dir, results)
    report = {
        "schema_version": "pvb.codec.visnet_spatial.tiny_report.v1",
        "implementation": implementation,
        "micro_overfit": micro_summary,
        "manifest": manifest,
        "protocol": protocol,
        "backbones": results,
        "aggregate_rows": rows,
        "development_backend_recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "recommendation_diagnostics": {
            "eligible_backbones": sorted(candidates),
            "best_eligible_by_future_drmsd": best_name,
            "radius_within_10_percent_of_best": radius_within_ten_percent,
            "bonded_vs_radius": bonded_comparison,
        },
        "plots": plots,
        "limitations": [
            "no protein isolation",
            "one seed",
            "natural rather than parameter-matched models",
            "this is a development protocol, not a final scientific benchmark",
        ],
    }
    _write_json(run_dir / "aggregate_comparison.json", report)
    lines = [
        "# ViSNet spatial backbone v1 tiny-overfit report",
        "",
        "The report separates implementation correctness, the three-clip micro-overfit, and the final tiny-overfit quality gate.",
        "",
        f"Implementation status: **{implementation.get('status', 'not supplied')}**.",
        f"Development-backend recommendation: **{recommendation or 'none'}** ({recommendation_reason}).",
        "",
        "| Backbone | Parameters | Implementation | Micro 500-step | Tiny quality | Train loss reduction | Holdout future dRMSD improvement | Final future dRMSD | Train seconds | Train tokens/s | Peak alloc (GiB) | Peak reserved (GiB) | Graph mode |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['spatial_backbone']} | {row['parameter_count']} | {row['implementation_pass']} | "
            f"{row['micro_overfit_pass']} | {row['tiny_overfit_quality_pass']} | "
            f"{row['train_total_reduction']:.3%} | "
            f"{row['holdout_future_drmsd_improvement']:.3%} | "
            f"{row['final_holdout_future_drmsd']:.6g} | "
            f"{row['training_elapsed_s']:.1f} | "
            f"{row['train_tokens_per_s']:.1f} | "
            f"{row['peak_allocated_memory_bytes'] / 2**30:.2f} | "
            f"{row['peak_reserved_memory_bytes'] / 2**30:.2f} | {row['graph_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Required protocol facts",
            "",
            f"- Manifest: {manifest['train_count']} train and {manifest['late_holdout_count']} late-holdout clips; {manifest['intersection_count']} overlap; T={manifest['frames_per_clip']}; {manifest['time_bucket_id']}.",
            f"- Loader: lazy mmap, `replacement=false`, exact no-replacement epoch coverage, FP32, `max_tokens={MAX_TOKENS}`, {EPOCHS} epochs, safety cap {SAFETY_CAP} steps.",
            "- Every epoch records all train sample IDs exactly once and evaluates all 117 holdout clips, with per-system metrics.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{run_dir / 'aggregate_comparison.json'}`",
            f"- CSV: `{run_dir / 'aggregate_comparison.csv'}`",
            f"- Plots: `{', '.join(str(path) for path in plots)}`",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}." for item in report["limitations"])
    (run_dir / "aggregate_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial-backbone",
        choices=("all", *BACKBONES),
        default="all",
        help="backend to run; default runs all three serially",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--micro-summary", type=Path, default=None)
    parser.add_argument("--implementation-evidence", type=Path, default=None)
    args = parser.parse_args(argv)
    selected = BACKBONES if args.spatial_backbone == "all" else (args.spatial_backbone,)
    runtime = require_cuda()
    manifest = build_manifest_contract()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    run_dir = output_root / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing tiny-overfit run: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "manifest_contract.json", manifest)
    protocol = {
        "schema_version": "pvb.codec.visnet_spatial.tiny_overfit.v1",
        "runtime": runtime,
        "seed": SEED,
        "spatial_backbones": list(selected),
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
        "lr": 1.0e-4,
        "weight_decay": 1.0e-6,
        "warmup_steps": 20,
        "grad_clip": 1.0,
        "epochs": EPOCHS,
        "safety_cap_steps": SAFETY_CAP,
        "temporal_layers": 1,
        "temporal_ratio": 1,
    }
    _write_json(run_dir / "protocol.json", protocol)
    train_dataset, valid_dataset, train_loader, valid_loader = make_loaders()
    try:
        if len(train_dataset) != 441 or len(valid_dataset) != 117:
            raise RuntimeError("lazy dataset lengths do not match the exact manifest")
        micro_summary = _load_micro_summary(args.micro_summary)
        implementation = _load_micro_summary(args.implementation_evidence)
        results: dict[str, Mapping[str, Any]] = {}
        for backbone in selected:
            print(f"[tiny-overfit] starting {backbone}", flush=True)
            results[backbone] = run_backbone(
                backbone,
                run_dir,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                train_loader=train_loader,
                valid_loader=valid_loader,
                protocol=protocol,
            )
            print(
                f"[tiny-overfit] finished {backbone}: "
                f"quality={results[backbone]['quality']['tiny_overfit_quality_pass']}",
                flush=True,
            )
        report = _write_reports(
            run_dir, protocol, manifest, results, micro_summary, implementation
        )
    finally:
        train_dataset.close()
        valid_dataset.close()
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "manifest_sha256": manifest["built_manifest_sha256"],
                "backbones": list(results),
                "recommendation": report["development_backend_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(result["quality"]["tiny_overfit_quality_pass"] for result in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
