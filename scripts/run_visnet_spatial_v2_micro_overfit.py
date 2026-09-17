#!/usr/bin/env python3
"""Run the frozen 500-step v2 ViSNet geometry comparison."""

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
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from trainer.codec_losses import CodecLossWeights
from trainer.codec_trainer import (
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "outputs/atlas_selected_trajectories/clip_store/train"
BASE_OUTPUT = ROOT / "outputs/visnet_spatial_v2/micro_overfit"
DEVICE = torch.device("cuda:0")
SEED = 20260902
STEPS = 500
LOG_EVERY = 10
MAX_TOKENS = 80000
SAMPLE_IDS = (
    "atlas_5e3e_A_R1_w000000",
    "atlas_1v7r_A_R1_w000000",
    "atlas_2wlt_A_R1_w000000",
)
BACKBONE_CONFIGS = {
    "torchmd_et": {"lmax": 1, "vertex_type": None, "v2": False},
    "visnet_bonded": {"lmax": 1, "vertex_type": None, "v2": False},
    "visnet_v2_bonded_lmax1": {
        "backbone": "visnet_v2_bonded",
        "lmax": 1,
        "vertex_type": "edge",
        "v2": True,
    },
    "visnet_v2_radius_lmax1": {
        "backbone": "visnet_v2_radius",
        "lmax": 1,
        "vertex_type": "edge",
        "v2": True,
        "logical_batch_accumulation": True,
    },
    "visnet_v2_bonded_lmax2": {
        "backbone": "visnet_v2_bonded",
        "lmax": 2,
        "vertex_type": "edge",
        "v2": True,
        "logical_batch_accumulation": True,
    },
}


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "V2_MICRO_HARD_FAIL: CUDA is unavailable; this gate has no CPU fallback"
        )
    try:
        import torch_cluster
        from torch_cluster import radius_graph
    except Exception as exc:
        raise RuntimeError(
            "V2_MICRO_HARD_FAIL: CUDA torch_cluster.radius_graph is unavailable"
        ) from exc
    if not callable(radius_graph):
        raise RuntimeError("V2_MICRO_HARD_FAIL: radius_graph is not callable")
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
    dataset = ClipMMapDataset(TRAIN_ROOT)
    by_sample_id = {str(item[0]): index for index, item in enumerate(dataset._index)}
    missing = [sample_id for sample_id in SAMPLE_IDS if sample_id not in by_sample_id]
    if missing:
        raise RuntimeError(f"selected micro-overfit samples are missing: {missing}")
    records = [dataset[by_sample_id[sample_id]] for sample_id in SAMPLE_IDS]
    dataset.close()
    batch = collate_clip_records(records)
    tokens = int(batch.frames * batch.atom_count)
    if tokens > MAX_TOKENS:
        raise RuntimeError(f"fixed micro batch is oversized: {tokens} > {MAX_TOKENS}")
    return records, {
        "sample_ids": list(SAMPLE_IDS),
        "frames": int(batch.frames),
        "packed_atoms": int(batch.atom_count),
        "effective_tokens": tokens,
        "time_bucket_ids": list(batch.time_bucket_id),
        "source_split": "train",
        "validation_samples_used": False,
    }


def make_model(label: str) -> PVBCodecModel:
    settings = BACKBONE_CONFIGS[label]
    backbone = str(settings.get("backbone", label))
    kwargs: dict[str, Any] = {
        "hidden_channels": 128,
        "spatial_layers": 2,
        "spatial_backbone": backbone,
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "num_rbf": 50,
        "num_heads": 8,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "max_num_neighbors": 32,
        "neighbor_backend": "cuda_radius",
        "bond_construction": {"mode": "topology"},
        "spatial_execution": {"mode": "full"},
    }
    if settings["v2"]:
        kwargs.update(
            {
                "lmax": int(settings["lmax"]),
                "vertex_type": str(settings["vertex_type"]),
                "rbf_type": "expnorm",
                "vecnorm_type": "max_min",
                "trainable_vecnorm": False,
            }
        )
    return PVBCodecModel(**kwargs)


def make_config(steps: int) -> CodecTrainConfig:
    return CodecTrainConfig(
        lr=1.0e-4,
        weight_decay=1.0e-6,
        max_steps=int(steps),
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
        raise RuntimeError(f"expected one repeated micro batch, got {len(loader)}")
    return loader


def evaluate_records(trainer: CodecTrainer, records: list[dict[str, Any]]) -> dict[str, Any]:
    per_system: dict[str, dict[str, float]] = {}
    for record in records:
        per_system[str(record["sample_id"])] = trainer.evaluate_batch(
            collate_clip_records([record])
        )
        torch.cuda.empty_cache()
    keys = tuple(per_system[next(iter(per_system))])
    aggregate = {
        key: float(sum(metrics[key] for metrics in per_system.values()) / len(per_system))
        for key in keys
    }
    return {"aggregate": aggregate, "per_system": per_system}


def gradient_diagnostics(
    trainer: CodecTrainer, records: list[dict[str, Any]]
) -> dict[str, Any]:
    trainer.optimizer.zero_grad(set_to_none=True)
    loss_values: list[float] = []
    for record in records:
        batch = collate_clip_records([record])
        losses, _, moved_batch = trainer._loss_for_batch(batch)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("micro-overfit gradient diagnostic is non-finite")
        (losses["total"] / len(records)).backward()
        loss_values.append(float(losses["total"].detach().cpu()))
        del losses, moved_batch, batch
        torch.cuda.empty_cache()
    selected: dict[str, float] = {}
    for name, parameter in trainer.model.named_parameters():
        if not any(
            token in name
            for token in (
                "distance_expansion",
                "neighbor_embedding",
                "edge_embedding",
                "vis_mp_layers",
                "vec_out_norm",
            )
        ):
            continue
        if parameter.grad is not None:
            selected[name] = float(parameter.grad.detach().abs().sum().cpu())
    trainer.optimizer.zero_grad(set_to_none=True)
    finite = all(math.isfinite(value) for value in selected.values())
    nonzero = any(value > 0.0 for value in selected.values())
    return {
        "finite": finite,
        "nonzero": nonzero,
        "parameter_count_with_grad": len(selected),
        "nonzero_parameters": sorted(name for name, value in selected.items() if value > 0.0),
        "loss_used": float(sum(loss_values) / len(loss_values)),
    }


def graph_diagnostics(model: PVBCodecModel, records: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_clip_records(records)
    model.prepare_batch(batch)
    from trainer.codec_trainer import prepare_batch_then_to_device

    moved = prepare_batch_then_to_device(model, batch, DEVICE)
    with torch.no_grad():
        encoded = model.frame_encoder(moved)
    graph = encoded.graph
    result = {
        "backbone": encoded.spatial_backbone,
        "graph_mode": encoded.graph_mode,
        "backend_used": encoded.backend_used,
        "nodes": int(graph.num_nodes),
        "distance_edges": int(graph.distance_edge_index.shape[1]),
        "union_edges": int(graph.edge_index.shape[1]),
        "replicated_covalent_edges": int(graph.bond_index.shape[1]),
        "binary_covalent_edge_features": int(graph.bond_type.sum().item()),
    }
    del encoded, moved
    torch.cuda.synchronize(DEVICE)
    return result


def logical_optimizer_step(
    trainer: CodecTrainer, records: list[dict[str, Any]]
) -> dict[str, float]:
    """Apply one update for the fixed three-clip logical batch without co-resident graphs."""

    trainer.model.train()
    next_step = trainer.step + 1
    if trainer.config.warmup_steps:
        scale = min(1.0, next_step / float(trainer.config.warmup_steps))
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer.config.lr * scale
    trainer.optimizer.zero_grad(set_to_none=True)
    per_record_metrics: list[dict[str, float]] = []
    for record in records:
        batch = collate_clip_records([record])
        losses, _, moved_batch = trainer._loss_for_batch(batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("micro-overfit loss is NaN or Inf before backward")
        (total / len(records)).backward()
        per_record_metrics.append(trainer._metrics(losses, moved_batch))
        del losses, moved_batch, batch
        torch.cuda.empty_cache()
    if trainer.config.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.grad_clip)
    for parameter in trainer.model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("micro-overfit gradient is NaN or Inf")
    trainer.optimizer.step()
    trainer.step = next_step
    trainer.epoch += 1
    trainer.batch_in_epoch = 0
    trainer._train_iterator = None
    keys = tuple(per_record_metrics[0])
    return {
        key: float(
            sum(metrics[key] for metrics in per_record_metrics)
            / len(per_record_metrics)
        )
        for key in keys
    }


def run_logical_batch(
    trainer: CodecTrainer,
    records: list[dict[str, Any]],
    *,
    max_steps: int,
    log_path: Path,
    log_every: int,
) -> dict[str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last: dict[str, float] = {}
    with log_path.open("a", encoding="utf-8") as handle:
        while trainer.step < int(max_steps):
            last = logical_optimizer_step(trainer, records)
            if trainer.step % int(log_every) == 0 or trainer.step == int(max_steps):
                handle.write(
                    json.dumps(
                        {
                            "schema_version": "pvb.codec.train.v1",
                            "split": "train",
                            "step": int(trainer.step),
                            "epoch": int(trainer.epoch),
                            "batch_in_epoch": 0,
                            "learning_rate": float(
                                trainer.optimizer.param_groups[0]["lr"]
                            ),
                            "metrics": last,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
    return last


def verify_resume(label: str, records: list[dict[str, Any]], checkpoint: Path) -> dict[str, Any]:
    loader = make_loader(records)
    trainer = CodecTrainer(
        make_model(label),
        loader,
        config=make_config(STEPS + 1),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    trainer.load_checkpoint(checkpoint)
    if trainer.step != STEPS:
        raise RuntimeError(f"{label} loaded at {trainer.step}, expected {STEPS}")
    if BACKBONE_CONFIGS[label].get("logical_batch_accumulation", False):
        run_logical_batch(
            trainer,
            records,
            max_steps=STEPS + 1,
            log_path=checkpoint.parent / "resume_metrics.jsonl",
            log_every=STEPS + 1,
        )
    else:
        trainer.run(max_steps=STEPS + 1)
    if trainer.step != STEPS + 1:
        raise RuntimeError(f"{label} resume stopped at {trainer.step}")
    del trainer, loader
    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "passed", "checkpoint_step": STEPS, "resumed_step": STEPS + 1}


def run_one(label: str, records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    output = run_dir / label
    output.mkdir(parents=True, exist_ok=False)
    loader = make_loader(records)
    trainer = CodecTrainer(
        make_model(label),
        loader,
        config=make_config(STEPS),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    trainer.fit_normalization(loader)
    logical_accumulation = bool(
        BACKBONE_CONFIGS[label].get("logical_batch_accumulation", False)
    )
    diagnostic_records = records[:1] if logical_accumulation else records
    diagnostics = graph_diagnostics(trainer.model, diagnostic_records)
    initial = evaluate_records(trainer, records)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    log = output / "train_metrics.jsonl"
    if logical_accumulation:
        run_logical_batch(
            trainer,
            records,
            max_steps=STEPS,
            log_path=log,
            log_every=LOG_EVERY,
        )
    else:
        trainer.run(max_steps=STEPS, log_path=log, log_every=LOG_EVERY)
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - started
    final = evaluate_records(trainer, records)
    gradients = gradient_diagnostics(trainer, records)
    checkpoint = output / "codec_step_00000500.pt"
    trainer.save_checkpoint(checkpoint)
    resume = verify_resume(label, records, checkpoint)
    first = float(initial["aggregate"]["total"])
    last = float(final["aggregate"]["total"])
    reduction = (first - last) / max(abs(first), 1.0e-8)
    if not math.isfinite(first) or not math.isfinite(last):
        raise FloatingPointError(f"{label} has a non-finite loss")
    v2_geometry_gate = bool(BACKBONE_CONFIGS[label]["v2"])
    geometry_gate_pass = (
        not v2_geometry_gate
        or (
            gradients["finite"]
            and gradients["nonzero"]
        )
    )
    result = {
        "spatial_backbone": str(BACKBONE_CONFIGS[label].get("backbone", label)),
        "label": label,
        "lmax": int(BACKBONE_CONFIGS[label]["lmax"]),
        "vertex_type": BACKBONE_CONFIGS[label]["vertex_type"],
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "steps": STEPS,
        "logical_batch_accumulation": logical_accumulation,
        "logical_batch_size": len(records),
        "initial_metrics": initial,
        "final_metrics": final,
        "aggregate_initial_total": first,
        "aggregate_final_total": last,
        "aggregate_relative_reduction": reduction,
        "finite_losses_and_gradients": gradients["finite"],
        "geometry_gradient_diagnostics": gradients,
        "geometry_gradient_gate": (
            "required" if v2_geometry_gate else "not_applicable_legacy_backbone"
        ),
        "micro_overfit_quality_pass": bool(
            reduction >= 0.30 and geometry_gate_pass
        ),
        "graph_diagnostics": diagnostics,
        "runtime": {
            "training_elapsed_s": elapsed,
            "steps_per_second": STEPS / max(elapsed, 1.0e-9),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
        },
        "resume": resume,
        "checkpoint": str(checkpoint),
        "log": str(log),
        "precision": "fp32",
        "model_contract": trainer.model.model_contract(),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del trainer, loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("all", *BACKBONE_CONFIGS), default="all")
    parser.add_argument("--output-root", type=Path, default=BASE_OUTPUT)
    args = parser.parse_args(argv)
    runtime = require_cuda()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    records, data_contract = select_records()
    selected = tuple(BACKBONE_CONFIGS) if args.label == "all" else (args.label,)
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    run_dir = root / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    protocol = {
        "schema_version": "pvb.codec.visnet_spatial.micro_overfit.v2",
        "runtime": runtime,
        "seed": SEED,
        "steps": STEPS,
        "log_every": LOG_EVERY,
        "max_tokens": MAX_TOKENS,
        "data_root": str(TRAIN_ROOT),
        "data_contract": data_contract,
        "sample_ids": list(SAMPLE_IDS),
        "backbones": list(selected),
        "configs": {key: BACKBONE_CONFIGS[key] for key in selected},
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "precision": "fp32",
        "replacement": True,
        "checkpoint_resume": "step 500 to step 501",
        "quality_gate": "finite plus nonzero geometry gradients and at least 30 percent total-loss reduction",
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {"status": "running", "protocol": protocol, "backbones": {}}
    for label in selected:
        print(f"[v2-micro] starting {label}", flush=True)
        summary["backbones"][label] = run_one(label, records, run_dir)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[v2-micro] finished {label}: "
            f"loss={summary['backbones'][label]['aggregate_final_total']:.6g} "
            f"pass={summary['backbones'][label]['micro_overfit_quality_pass']}",
            flush=True,
        )
    failures = [
        label
        for label, result in summary["backbones"].items()
        if not result["micro_overfit_quality_pass"]
    ]
    summary["status"] = "completed" if not failures else "quality_gate_failed"
    summary["quality_failures"] = failures
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
