#!/usr/bin/env python3
"""Run the fixed three-clip ViSNet spatial-backbone micro-overfit."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import collate_clip_records
from trainer.codec_trainer import (
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
    prepare_batch_then_to_device,
)
from trainer.codec_losses import CodecLossWeights


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "outputs/atlas_selected_trajectories/clip_store/train"
BASE_OUTPUT = ROOT / "outputs/visnet_spatial_v1/micro_overfit"
DEVICE = torch.device("cuda:0")
SEED = 20260901
STEPS = 500
LOG_EVERY = 10
MAX_TOKENS = 80000
BACKBONES = ("torchmd_et", "visnet_radius", "visnet_bonded")
SAMPLE_IDS = (
    "atlas_5e3e_A_R1_w000000",
    "atlas_1v7r_A_R1_w000000",
    "atlas_2wlt_A_R1_w000000",
)


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MICRO_OVERFIT_HARD_FAIL: torch.cuda.is_available() is False; "
            "real-data overfitting requires CUDA and has no CPU fallback"
        )
    try:
        import torch_cluster
        from torch_cluster import radius_graph
    except Exception as exc:
        raise RuntimeError(
            "MICRO_OVERFIT_HARD_FAIL: CUDA torch_cluster.radius_graph is unavailable"
        ) from exc
    if not callable(radius_graph):
        raise RuntimeError(
            "MICRO_OVERFIT_HARD_FAIL: torch_cluster.radius_graph is not callable"
        )
    props = torch.cuda.get_device_properties(DEVICE)
    return {
        "device": str(DEVICE),
        "gpu": props.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_cluster": getattr(torch_cluster, "__version__", "unknown"),
        "total_memory_bytes": int(props.total_memory),
    }


def select_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not TRAIN_ROOT.is_dir():
        raise FileNotFoundError(f"selected train clip store is missing: {TRAIN_ROOT}")
    from data.clip_dataset import ClipMMapDataset

    dataset = ClipMMapDataset(TRAIN_ROOT)
    by_sample_id = {str(item[0]): index for index, item in enumerate(dataset._index)}
    missing = [sample_id for sample_id in SAMPLE_IDS if sample_id not in by_sample_id]
    if missing:
        raise RuntimeError(f"selected micro-overfit samples are missing: {missing}")
    records = [dataset[by_sample_id[sample_id]] for sample_id in SAMPLE_IDS]
    dataset.close()
    actual = tuple(str(record["sample_id"]) for record in records)
    if actual != SAMPLE_IDS:
        raise RuntimeError(f"selected sample order changed: {actual!r}")
    batch = collate_clip_records(records)
    atoms_per_sample = [
        int(batch.atom_ptr[index + 1] - batch.atom_ptr[index])
        for index in range(batch.batch_size)
    ]
    tokens = int(batch.frames * sum(atoms_per_sample))
    if tokens > MAX_TOKENS:
        raise RuntimeError(
            f"fixed overfit batch is oversized: {tokens} > max_tokens={MAX_TOKENS}"
        )
    return records, {
        "sample_ids": list(actual),
        "systems": [str(record["sample_id"]).rsplit("_R", 1)[0] for record in records],
        "frames": int(batch.frames),
        "atoms_per_sample": atoms_per_sample,
        "packed_atoms": int(batch.atom_count),
        "effective_tokens": tokens,
        "time_bucket_ids": list(batch.time_bucket_id),
        "source_split": "train",
        "validation_samples_used": False,
    }


def make_model(spatial_backbone: str) -> PVBCodecModel:
    selected = str(spatial_backbone).lower()
    if selected not in BACKBONES:
        raise ValueError(f"unsupported spatial backbone: {spatial_backbone!r}")
    return PVBCodecModel(
        hidden_channels=128,
        spatial_layers=2,
        spatial_backbone=selected,
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


def make_config(steps: int) -> CodecTrainConfig:
    return CodecTrainConfig(
        lr=1.0e-4,
        weight_decay=1.0e-6,
        max_steps=steps,
        grad_clip=1.0,
        warmup_steps=20,
        device="cuda",
        precision="fp32",
        bucket_specs=(
            TimeBucketSpec("dt_100ps", 100.0, 0.5, 1.0),
        ),
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


def make_loader(records: list[dict[str, Any]]) -> Any:
    loader = make_clip_dataloader(
        records,
        max_tokens=MAX_TOKENS,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=SEED,
        shuffle=False,
        replacement=True,
    )
    if len(loader) != 1:
        raise RuntimeError(
            "the three selected systems must form one repeated overfit batch; "
            f"got {len(loader)}"
        )
    return loader


def evaluate_records(trainer: CodecTrainer, records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = trainer.evaluate_batch(collate_clip_records(records))
    per_system = {}
    for record in records:
        metrics = trainer.evaluate_batch(collate_clip_records([record]))
        per_system[str(record["sample_id"])] = metrics
    return {"aggregate": aggregate, "per_system": per_system}


def graph_diagnostics(model: PVBCodecModel, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Record the graph actually consumed by one representative forward."""

    cpu_batch = collate_clip_records(records)
    moved = prepare_batch_then_to_device(model, cpu_batch, DEVICE)
    with torch.no_grad():
        encoded = model.frame_encoder(moved)
    graph = encoded.graph
    edge_count = int(graph.edge_index.shape[1])
    bond_count = int(graph.bond_index.shape[1])
    bond_feature_count = int(graph.bond_type.sum().item())
    result = {
        "backbone": encoded.spatial_backbone,
        "graph_mode": encoded.graph_mode,
        "backend_used": encoded.backend_used,
        "nodes": int(graph.num_nodes),
        "distance_edges": int(graph.distance_edge_index.shape[1]),
        "union_edges": edge_count,
        "replicated_covalent_edges": bond_count,
        "binary_covalent_edge_features": bond_feature_count,
        "topology_bond_coverage": (
            None if encoded.graph_mode == "native_radius" else bond_feature_count / max(edge_count, 1)
        ),
    }
    del encoded, moved, cpu_batch
    torch.cuda.synchronize(DEVICE)
    return result


def parse_log(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_steps = list(range(LOG_EVERY, STEPS + 1, LOG_EVERY))
    actual_steps = [int(row["step"]) for row in rows]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"micro-overfit log is incomplete: expected {expected_steps[-3:]}, "
            f"got {actual_steps[-3:]}"
        )
    totals = [float(row["metrics"]["total"]) for row in rows]
    if not all(math.isfinite(value) for value in totals):
        raise FloatingPointError(f"non-finite total loss in {path}")
    return {
        "records": len(rows),
        "first_logged_step": actual_steps[0],
        "last_logged_step": actual_steps[-1],
        "first_logged_total": totals[0],
        "last_logged_total": totals[-1],
        "minimum_logged_total": min(totals),
        "last_logged_epoch": int(rows[-1]["epoch"]),
    }


def verify_resume(
    spatial_backbone: str,
    records: list[dict[str, Any]],
    checkpoint: Path,
) -> dict[str, Any]:
    loader = make_loader(records)
    model = make_model(spatial_backbone)
    trainer = CodecTrainer(
        model,
        loader,
        config=make_config(STEPS + 1),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    trainer.load_checkpoint(checkpoint)
    loaded_step = int(trainer.step)
    if loaded_step != STEPS:
        raise RuntimeError(
            f"{spatial_backbone} checkpoint loaded at step {loaded_step}, expected {STEPS}"
        )
    trainer.run(max_steps=STEPS + 1)
    resumed_step = int(trainer.step)
    if resumed_step != STEPS + 1:
        raise RuntimeError(
            f"{spatial_backbone} checkpoint resume stopped at {resumed_step}"
        )
    del trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "status": "passed",
        "checkpoint_step": loaded_step,
        "resumed_step": resumed_step,
        "extended_target": STEPS + 1,
    }


def run_backbone(
    spatial_backbone: str,
    records: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    backbone_dir = run_dir / spatial_backbone
    backbone_dir.mkdir(parents=True, exist_ok=False)
    loader = make_loader(records)
    model = make_model(spatial_backbone)
    trainer = CodecTrainer(
        model,
        loader,
        config=make_config(STEPS),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    normalization_start = time.perf_counter()
    trainer.fit_normalization(loader)
    torch.cuda.synchronize(DEVICE)
    normalization_elapsed = time.perf_counter() - normalization_start

    diagnostics = graph_diagnostics(model, records)
    initial_start = time.perf_counter()
    initial = evaluate_records(trainer, records)
    torch.cuda.synchronize(DEVICE)
    initial_elapsed = time.perf_counter() - initial_start

    log_path = backbone_dir / "train_metrics.jsonl"
    torch.cuda.reset_peak_memory_stats(DEVICE)
    training_start = time.perf_counter()
    trainer.run(max_steps=STEPS, log_path=log_path, log_every=LOG_EVERY)
    torch.cuda.synchronize(DEVICE)
    training_elapsed = time.perf_counter() - training_start

    final_start = time.perf_counter()
    final = evaluate_records(trainer, records)
    torch.cuda.synchronize(DEVICE)
    final_elapsed = time.perf_counter() - final_start

    checkpoint = backbone_dir / f"codec_step_{trainer.step:08d}.pt"
    trainer.save_checkpoint(checkpoint)
    torch.cuda.synchronize(DEVICE)
    resume = verify_resume(spatial_backbone, records, checkpoint)
    log_summary = parse_log(log_path)
    initial_total = float(initial["aggregate"]["total"])
    final_total = float(final["aggregate"]["total"])
    reduction = (initial_total - final_total) / max(abs(initial_total), 1.0e-8)
    if not math.isfinite(initial_total) or not math.isfinite(final_total):
        raise FloatingPointError(f"{spatial_backbone} has a non-finite evaluation loss")
    result = {
        "spatial_backbone": spatial_backbone,
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "steps": int(trainer.step),
        "batch_count": len(loader),
        "batch_sample_ids": list(SAMPLE_IDS),
        "initial_metrics": initial,
        "final_metrics": final,
        "loss_log": log_summary,
        "aggregate_initial_total": initial_total,
        "aggregate_final_total": final_total,
        "aggregate_relative_reduction": reduction,
        "finite_losses_and_gradients": True,
        "micro_overfit_quality_pass": bool(reduction > 0.30),
        "graph_diagnostics": diagnostics,
        "resume": resume,
        "normalization_elapsed_s": normalization_elapsed,
        "initial_evaluation_elapsed_s": initial_elapsed,
        "training_elapsed_s": training_elapsed,
        "final_evaluation_elapsed_s": final_elapsed,
        "total_elapsed_s": normalization_elapsed + initial_elapsed + training_elapsed + final_elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "checkpoint": str(checkpoint),
        "log": str(log_path),
        "precision": "fp32",
        "bond_construction_mode": "topology",
        "model_contract": model.model_contract(),
    }
    (backbone_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial-backbone",
        choices=("all", *BACKBONES),
        default="all",
        help="backend to run; default runs all three backbones serially",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_OUTPUT,
        help="directory under which a timestamped run directory is created",
    )
    args = parser.parse_args(argv)
    selected_backbones = BACKBONES if args.spatial_backbone == "all" else (args.spatial_backbone,)
    runtime = require_cuda()
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    records, data_contract = select_records()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    run_dir = output_root / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing micro-overfit run: {run_dir}")
    run_dir.mkdir(parents=True)
    protocol = {
        "schema_version": "pvb.codec.visnet_spatial.micro_overfit.v1",
        "runtime": runtime,
        "seed": SEED,
        "steps": STEPS,
        "log_every": LOG_EVERY,
        "max_tokens": MAX_TOKENS,
        "data_root": str(TRAIN_ROOT),
        "data_contract": data_contract,
        "selection_policy": "one fixed R1/w000000 train clip from each selected system",
        "spatial_backbones": list(selected_backbones),
        "model_policy": "fresh full-width models with temporal_layers=1 and temporal_ratio=1",
        "bond_construction_mode": "topology for external backbones; ignored by visnet_radius",
        "precision": "fp32",
        "cuda_only": True,
        "quality_gate": "aggregate total loss is <70% of initialization after 500 steps",
        "resume_gate": "load step 500 checkpoint and complete one additional step",
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "status": "running",
        "protocol": protocol,
        "backbones": {},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for spatial_backbone in selected_backbones:
        print(f"[micro-overfit] starting {spatial_backbone}", flush=True)
        result = run_backbone(spatial_backbone, records, run_dir)
        summary["backbones"][spatial_backbone] = result
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[micro-overfit] finished {spatial_backbone}: "
            f"quality={result['micro_overfit_quality_pass']}",
            flush=True,
        )
    quality_failures = [
        name
        for name, result in summary["backbones"].items()
        if not result["micro_overfit_quality_pass"]
    ]
    summary["status"] = "completed" if not quality_failures else "quality_gate_failed"
    summary["micro_overfit_quality_failures"] = quality_failures
    summary["elapsed_s"] = time.perf_counter() - started
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not quality_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
