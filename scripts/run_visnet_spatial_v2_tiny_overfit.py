#!/usr/bin/env python3
"""Run the selected reference-faithful v2 backbone on the immutable tiny protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import run_visnet_spatial_tiny_overfit as v1_runner
from trainer.codec_trainer import PVBCodecModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/visnet_spatial_v2/tiny_overfit"
V2_BACKBONE = "visnet_v2_bonded"
V2_LMAX = 1
V2_VERTEX_TYPE = "edge"
V1_RUN_NAME = "run_20260901T192557"
_ORIGINAL_OPTIMIZER_STEP = v1_runner.CodecTrainer.optimizer_step
_ORIGINAL_GRAPH_DIAGNOSTICS = v1_runner._graph_diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _locate_v1_run() -> Path:
    candidates = (
        ROOT / "outputs/visnet_spatial_v1/tiny_overfit" / V1_RUN_NAME,
        ROOT.parent / "PVB/outputs/visnet_spatial_v1/tiny_overfit" / V1_RUN_NAME,
    )
    for candidate in candidates:
        if (candidate / "manifest_contract.json").is_file():
            return candidate
    raise FileNotFoundError(
        "immutable v1 tiny run is unavailable; refusing incomparable baseline reuse"
    )


def _assert_manifest_comparable(
    manifest: Mapping[str, Any], v1_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "schema_version",
        "source_manifest_sha256",
        "train_index_sha256",
        "valid_index_sha256",
        "systems",
        "replicas",
        "train_windows",
        "late_holdout_windows",
        "frames_per_clip",
        "time_bucket_id",
        "train_count",
        "late_holdout_count",
        "train_sample_ids",
        "late_holdout_sample_ids",
        "intersection_count",
    )
    for field in fields:
        if manifest.get(field) != v1_manifest.get(field):
            raise RuntimeError(
                f"v2 tiny manifest differs from immutable v1 field {field!r}"
            )
    if manifest["train_count"] != 441 or manifest["late_holdout_count"] != 117:
        raise RuntimeError("v2 tiny manifest is not exactly 441/117")
    if manifest["intersection_count"] != 0:
        raise RuntimeError("v2 tiny manifest has train/holdout overlap")
    return {
        "semantic_fields_match": True,
        "source_manifest_sha256_match": (
            manifest["source_manifest_sha256"]
            == v1_manifest["source_manifest_sha256"]
        ),
        "train_index_sha256_match": (
            manifest["train_index_sha256"] == v1_manifest["train_index_sha256"]
        ),
        "valid_index_sha256_match": (
            manifest["valid_index_sha256"] == v1_manifest["valid_index_sha256"]
        ),
        "built_manifest_sha256_match": (
            manifest.get("built_manifest_sha256")
            == v1_manifest.get("built_manifest_sha256")
        ),
        "built_manifest_hash_note": (
            "The v1 builder includes absolute source paths in its canonical payload; "
            "the worker path differs, so source/index hashes and all semantic rows "
            "are the comparability evidence."
        ),
    }


def _assert_protocol_comparable(
    protocol: Mapping[str, Any], v1_protocol: Mapping[str, Any]
) -> None:
    fields = (
        "seed",
        "systems",
        "replicas",
        "train_windows",
        "late_holdout_windows",
        "train_count",
        "late_holdout_count",
        "frames_per_clip",
        "time_bucket_id",
        "lazy_loading",
        "replacement",
        "exact_epoch_coverage",
        "precision",
        "max_tokens",
        "lr",
        "weight_decay",
        "warmup_steps",
        "grad_clip",
        "epochs",
        "safety_cap_steps",
        "temporal_layers",
        "temporal_ratio",
    )
    for field in fields:
        if protocol.get(field) != v1_protocol.get(field):
            raise RuntimeError(
                f"v2 tiny protocol differs from immutable v1 field {field!r}"
            )


def make_model(_spatial_backbone: str) -> PVBCodecModel:
    return PVBCodecModel(
        hidden_channels=128,
        spatial_layers=2,
        spatial_backbone=V2_BACKBONE,
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
        lmax=V2_LMAX,
        vertex_type=V2_VERTEX_TYPE,
        rbf_type="expnorm",
        trainable_rbf=False,
        vecnorm_type="max_min",
        trainable_vecnorm=False,
    )


def _single_packed_batch(batch: Any, sample_index: int) -> Any:
    """Return one clip without retaining the other clips' CUDA graph state."""

    start = int(batch.atom_ptr[sample_index])
    stop = int(batch.atom_ptr[sample_index + 1])
    bonds = torch.as_tensor(batch.bond_index, dtype=torch.long)
    if bonds.numel():
        keep = (batch.abid[bonds[0]] == sample_index) & (
            batch.abid[bonds[1]] == sample_index
        )
        bonds = bonds[:, keep] - start
    else:
        bonds = torch.empty((2, 0), dtype=torch.long)
    return replace(
        batch,
        x=batch.x[:, start:stop],
        bpos=batch.bpos[:, start:stop],
        atype=batch.atype[start:stop],
        btype=batch.btype[start:stop],
        block_id=batch.block_id[start:stop],
        component_id=batch.component_id[start:stop],
        atom_source_index=batch.atom_source_index[start:stop],
        abid=torch.zeros(stop - start, dtype=torch.long),
        atom_ptr=torch.tensor([0, stop - start], dtype=torch.long),
        bond_index=bonds,
        edge_mask=batch.edge_mask[start:stop],
        loss_mask=batch.loss_mask[start:stop],
        align_mask=batch.align_mask[start:stop],
        frame_mask=batch.frame_mask[sample_index : sample_index + 1],
        time_ps=batch.time_ps[sample_index : sample_index + 1],
        delta_time_ps=batch.delta_time_ps[sample_index : sample_index + 1],
        time_bucket_id=(str(batch.time_bucket_id[sample_index]),),
        topology_id=(str(batch.topology_id[sample_index]),),
        task=batch.task[sample_index : sample_index + 1],
        sample_id=(str(batch.sample_id[sample_index]),),
        atom_counts=(stop - start,),
        atom_identity_sha256=(str(batch.atom_identity_sha256[sample_index]),),
        host_task_ids=(int(batch.host_task_ids[sample_index]),),
    )


def _single_packed_batches(batch: Any) -> list[Any]:
    return [_single_packed_batch(batch, index) for index in range(batch.batch_size)]


def _logical_optimizer_step(trainer: Any, batch: Any) -> dict[str, float]:
    """Apply one loader-batch update while evaluating each clip separately."""

    if batch.batch_size == 1:
        return _ORIGINAL_OPTIMIZER_STEP(trainer, batch)
    trainer.model.train()
    next_step = trainer.step + 1
    if trainer.config.warmup_steps:
        scale = min(1.0, next_step / float(trainer.config.warmup_steps))
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer.config.lr * scale
    trainer.optimizer.zero_grad(set_to_none=True)
    clip_batches = _single_packed_batches(batch)
    per_clip_metrics: list[dict[str, float]] = []
    for clip_batch in clip_batches:
        losses, _, moved_batch = trainer._loss_for_batch(clip_batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("v2 tiny loss is NaN or Inf before backward")
        (total / len(clip_batches)).backward()
        per_clip_metrics.append(trainer._metrics(losses, moved_batch))
        del losses, moved_batch, clip_batch
        torch.cuda.empty_cache()
    if trainer.config.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.grad_clip)
    for parameter in trainer.model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("v2 tiny gradient is NaN or Inf")
    trainer.optimizer.step()
    trainer.step = next_step
    keys = tuple(per_clip_metrics[0])
    return {
        key: float(
            sum(metrics[key] for metrics in per_clip_metrics) / len(per_clip_metrics)
        )
        for key in keys
    }


def _safe_evaluate_loader(
    trainer: Any,
    loader: Any,
    dataset: Any,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Evaluate packed loader batches clip-wise to bound CUDA graph residency."""

    expected_ids = v1_runner._scheduled_ids(loader, dataset, epoch)
    expected_set = set(expected_ids)
    seen: list[str] = []
    metric_records: list[dict[str, Any]] = []
    per_system: dict[str, list[dict[str, Any]]] = {}
    loss_sums: dict[str, float] = {}
    sample_count = 0
    batch_count = 0
    trainer.model.eval()
    for packed_batch in loader:
        batch_ids = [str(value) for value in packed_batch.sample_id]
        seen.extend(batch_ids)
        for sample_index, clip_batch in enumerate(_single_packed_batches(packed_batch)):
            moved = v1_runner.prepare_batch_then_to_device(
                trainer.model, clip_batch, v1_runner.DEVICE, non_blocking=False
            )
            with torch.no_grad():
                output = trainer.model(moved)
                losses = v1_runner.compute_codec_losses(
                    output,
                    moved,
                    weights=trainer.config.weights_at(trainer.step),
                    normalization=trainer.normalization_stats,
                )
            for key, value in losses.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(
                    value.detach().cpu()
                )
            atom_count = int(moved.x.shape[1])
            metrics = v1_runner._metrics(
                output.x_hat,
                moved.x,
                v1_runner._single_metric_batch(moved, 0, 0, atom_count),
            )
            metric_records.append(metrics)
            system = batch_ids[sample_index].rsplit("_R", 1)[0]
            per_system.setdefault(system, []).append(metrics)
            sample_count += 1
            del output, losses, moved, clip_batch
            torch.cuda.empty_cache()
        batch_count += 1
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
        raise RuntimeError("evaluation sample count disagrees with the sampler contract")
    denominator = float(max(sample_count, 1))
    mean_loss = {key: value / denominator for key, value in loss_sums.items()}
    return {
        "epoch": int(epoch),
        "sample_count": sample_count,
        "batch_count": batch_count,
        "metrics": v1_runner._mean_records(metric_records),
        "per_system": {
            system: {
                "sample_count": len(records),
                "metrics": v1_runner._mean_records(records),
            }
            for system, records in sorted(per_system.items())
        },
        "loss": mean_loss,
        "exact_coverage": True,
    }


def _safe_graph_diagnostics(model: Any, batch: Any) -> dict[str, Any]:
    return _ORIGINAL_GRAPH_DIAGNOSTICS(model, _single_packed_batches(batch)[0])


def _result_row(
    name: str, result: Mapping[str, Any], *, implementation: str
) -> dict[str, Any]:
    final_holdout = result["final_holdout"]
    metrics = final_holdout["metrics"]
    future = metrics["future"]
    return {
        "spatial_backbone": name,
        "implementation": implementation,
        "tiny_quality_pass": bool(result["quality"]["tiny_overfit_quality_pass"]),
        "final_train_total": float(result["quality"]["final_train_total"]),
        "final_holdout_total": float(final_holdout["loss"]["total"]),
        "future_drmsd": float(future["drmsd"]),
        "future_rmsd": float(future["rmsd"]),
        "future_bond_rmse": float(future["bond_rmse"]),
        "future_contact_error": float(future["contact_error"]),
        "velocity_rmse": float(metrics["velocity_rmse"]),
        "acceleration_rmse": float(metrics["acceleration_rmse"]),
        "frequency_retention": float(metrics["frequency_retention"]),
        "frame0_drmsd": float(metrics["frame0"]["drmsd"]),
        "frame0_rmsd": float(metrics["frame0"]["rmsd"]),
        "parameter_count": int(result["parameter_count"]),
        "training_elapsed_s": float(result["training_elapsed_s"]),
        "train_tokens_per_s": float(result["train_tokens_per_s"]),
        "holdout_eval_s_per_epoch": sum(
            float(item["holdout_evaluation_elapsed_s"])
            for item in result["epoch_metrics"]
        ) / max(len(result["epoch_metrics"]), 1),
        "peak_allocated_memory_bytes": int(result["peak_allocated_memory_bytes"]),
        "peak_reserved_memory_bytes": int(result["peak_reserved_memory_bytes"]),
        "graph_mode": result["graph_diagnostics"]["graph_mode"],
        "backend_used": result["graph_diagnostics"]["backend_used"],
        "checkpoint": str(result["checkpoint"]),
    }


def _metric_value(row: Mapping[str, Any], key: str) -> float:
    metrics = row["holdout_evaluation"]["metrics"]
    if key == "future_drmsd":
        return float(metrics["future"]["drmsd"])
    if key == "future_rmsd":
        return float(metrics["future"]["rmsd"])
    if key == "future_bond_rmse":
        return float(metrics["future"]["bond_rmse"])
    if key == "future_contact_error":
        return float(metrics["future"]["contact_error"])
    if key == "velocity_rmse":
        return float(metrics["velocity_rmse"])
    if key == "acceleration_rmse":
        return float(metrics["acceleration_rmse"])
    raise KeyError(key)


def _write_comparison_plots(
    run_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    v1_run: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = (
        ("Future dRMSD", "future_drmsd"),
        ("Future RMSD", "future_rmsd"),
        ("Future bond RMSE", "future_bond_rmse"),
        ("Future contact error", "future_contact_error"),
        ("Velocity RMSE", "velocity_rmse"),
        ("Acceleration RMSE", "acceleration_rmse"),
    )
    epochs: dict[str, list[int]] = {}
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for name, result in results.items():
        epochs[name] = [int(item["epoch"]) for item in result["epoch_metrics"]]
        rows[name] = list(result["epoch_metrics"])
    plot_paths: list[str] = []

    def save(
        figure: Any,
        stem: str,
        rect: tuple[float, float, float, float] | None = None,
    ) -> None:
        if rect is None:
            figure.tight_layout()
        else:
            figure.tight_layout(rect=rect)
        for suffix in ("png", "pdf"):
            path = run_dir / f"{stem}.{suffix}"
            figure.savefig(path)
            plot_paths.append(str(path))
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, result in results.items():
        train_path = v1_run / name / "train_metrics.jsonl"
        if name == V2_BACKBONE:
            train_path = Path(result["train_metrics"])
        if train_path.is_file():
            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            axes[0].plot(
                [int(item["step"]) for item in train_rows],
                [float(item["metrics"]["total"]) for item in train_rows],
                label=name,
            )
        initial_total = float(result["initial_holdout"]["loss"]["total"])
        axes[1].plot(
            [0] + epochs[name],
            [initial_total]
            + [
                float(item["holdout_evaluation"]["loss"]["total"])
                for item in rows[name]
            ],
            label=name,
        )
    axes[0].set_title("Training total loss")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Late-holdout total loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[0].legend()
    axes[1].legend()
    save(figure, "loss_curves_comparison")

    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for axis, (title, metric_key) in zip(axes, metric_specs):
        for name, item_rows in rows.items():
            axis.plot(
                epochs[name],
                [_metric_value(item, metric_key) for item in item_rows],
                marker="o",
                linewidth=1.6,
                markersize=3,
                label=name,
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    for axis, ylabel in zip(
        axes,
        ("dRMSD", "RMSD", "RMSE", "error", "RMSE", "RMSE"),
    ):
        axis.set_ylabel(ylabel)
    axes[0].legend(loc="best")
    figure.suptitle(
        "Late-holdout evaluation comparison (absolute metrics)",
        fontsize=15,
    )
    runtime_lines = [
        (
            f"{name}: train {float(results[name]['training_elapsed_s']) / 60.0:.1f} min, "
            f"{float(results[name]['train_tokens_per_s']) / 1000.0:.1f}k tokens/s, "
            f"eval {sum(float(item['holdout_evaluation_elapsed_s']) for item in results[name]['epoch_metrics']) / max(len(results[name]['epoch_metrics']), 1):.1f} s/epoch, "
            f"peak {float(results[name]['peak_allocated_memory_bytes']) / 2**30:.1f} GiB alloc / "
            f"{float(results[name]['peak_reserved_memory_bytes']) / 2**30:.1f} GiB reserved"
        )
        for name in results
    ]
    figure.text(
        0.01,
        0.01,
        "Runtime annotations — " + "\n".join(runtime_lines),
        ha="left",
        va="bottom",
        fontsize=7.5,
        family="monospace",
    )
    save(figure, "eval_metrics_comparison", rect=(0.0, 0.15, 1.0, 1.0))

    labels = list(results)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 4, figsize=(21, 6))
    bar_specs = (
        (
            "Training wall time",
            "min",
            [
                float(results[name]["training_elapsed_s"]) / 60.0
                for name in labels
            ],
        ),
        (
            "Training throughput",
            "k tokens/s",
            [
                float(results[name]["train_tokens_per_s"]) / 1000.0
                for name in labels
            ],
        ),
        (
            "Holdout eval time",
            "s / epoch",
            [
                sum(
                    float(item["holdout_evaluation_elapsed_s"])
                    for item in results[name]["epoch_metrics"]
                )
                / max(len(results[name]["epoch_metrics"]), 1)
                for name in labels
            ],
        ),
    )
    colors = ("#4c78a8", "#f58518", "#54a24e", "#b279a2")
    for axis, (title, unit, values) in zip(axes[:3], bar_specs):
        bars = axis.bar(labels, values, color=colors[: len(labels)])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis = axes[3]
    width = 0.36
    allocated = [
        float(results[name]["peak_allocated_memory_bytes"]) / 2**30
        for name in labels
    ]
    reserved = [
        float(results[name]["peak_reserved_memory_bytes"]) / 2**30
        for name in labels
    ]
    bars_alloc = axis.bar(x - width / 2.0, allocated, width, label="allocated")
    bars_reserved = axis.bar(x + width / 2.0, reserved, width, label="reserved")
    axis.set_title("Peak GPU memory")
    axis.set_ylabel("GiB")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=22)
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
    figure.suptitle(
        "Tiny comparison runtime and memory (metrics are absolute)",
        fontsize=15,
    )
    save(figure, "performance_summary_comparison")
    return plot_paths


def _classify(rows: Mapping[str, Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    v2 = rows[V2_BACKBONE]
    torchmd = rows["torchmd_et"]
    v1_bonded = rows["visnet_bonded"]
    v1_drmsd = float(v1_bonded["future_drmsd"])
    v2_drmsd = float(v2["future_drmsd"])
    torchmd_drmsd = float(torchmd["future_drmsd"])
    v1_rmsd = float(v1_bonded["future_rmsd"])
    v2_rmsd = float(v2["future_rmsd"])
    torchmd_rmsd = float(torchmd["future_rmsd"])
    gap = max(v1_drmsd - torchmd_drmsd, 1.0e-8)
    recovery = (v1_drmsd - v2_drmsd) / gap
    diagnostics = {
        "future_drmsd_gap_recovery_vs_v1_bonded_to_torchmd": recovery,
        "v2_beats_v1_bonded_future_drmsd": v2_drmsd < v1_drmsd,
        "v2_beats_v1_bonded_future_rmsd": v2_rmsd < v1_rmsd,
        "v2_within_10_percent_of_torchmd_future_drmsd": v2_drmsd <= 1.10 * torchmd_drmsd,
        "v2_within_10_percent_of_torchmd_future_rmsd": v2_rmsd <= 1.10 * torchmd_rmsd,
    }
    if not bool(v2["tiny_quality_pass"]):
        return "IMPLEMENTED", "v2 parity passed but tiny quality gate failed", diagnostics
    if v2_drmsd < torchmd_drmsd and v2_rmsd < torchmd_rmsd:
        return (
            "V2_PREFERRED",
            "v2 beats TorchMD and v1 bonded on both future geometry metrics",
            diagnostics,
        )
    if (
        diagnostics["v2_within_10_percent_of_torchmd_future_drmsd"]
        and diagnostics["v2_within_10_percent_of_torchmd_future_rmsd"]
    ):
        return (
            "V2_COMPETITIVE",
            "v2 is within 10% of TorchMD on both future geometry metrics",
            diagnostics,
        )
    if (
        diagnostics["v2_beats_v1_bonded_future_drmsd"]
        and diagnostics["v2_beats_v1_bonded_future_rmsd"]
    ):
        return (
            "V2_PARTIAL_RECOVERY",
            "v2 improves both future geometry metrics over v1 bonded but does not reach TorchMD",
            diagnostics,
        )
    return (
        "V2_NO_GAIN",
        "v2 parity passed but does not improve both future geometry metrics over v1 bonded",
        diagnostics,
    )


def _write_report(
    run_dir: Path,
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    v1_run: Path,
    results: Mapping[str, Mapping[str, Any]],
    v2_result: Mapping[str, Any],
    v1_artifact_hashes: Mapping[str, str],
    manifest_comparability: Mapping[str, Any],
) -> dict[str, Any]:
    all_results = {**results, V2_BACKBONE: v2_result}
    rows = {
        name: _result_row(
            name,
            result,
            implementation="new_v2" if name == V2_BACKBONE else "reused_immutable_v1",
        )
        for name, result in all_results.items()
    }
    classification, reason, diagnostics = _classify(rows)
    plot_paths = _write_comparison_plots(run_dir, all_results, v1_run)
    with (run_dir / "aggregate_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    report: dict[str, Any] = {
        "schema_version": "pvb.codec.visnet_spatial.tiny_report.v2",
        "outcome_classification": classification,
        "classification_reason": reason,
        "classification_diagnostics": diagnostics,
        "implementation_pass": True,
        "micro_overfit_selection": {
            "selected_backbone": V2_BACKBONE,
            "selected_lmax": V2_LMAX,
            "selected_vertex_type": V2_VERTEX_TYPE,
            "selection_rule": "absolute final aggregate loss first, then runtime/memory",
            "lmax1_final_aggregate_loss": 0.33027148246765137,
            "lmax2_final_aggregate_loss": 0.33591245611508685,
        },
        "manifest": dict(manifest),
        "manifest_comparability": dict(manifest_comparability),
        "protocol": dict(protocol),
        "immutable_v1_run": str(v1_run),
        "immutable_v1_artifact_sha256": dict(v1_artifact_hashes),
        "backbones": {name: dict(result) for name, result in all_results.items()},
        "aggregate_rows": list(rows.values()),
        "plots": plot_paths,
        "limitations": [
            "one seed",
            "natural rather than parameter-matched models",
            "tiny development protocol is not a final scientific benchmark",
            "v2 radius is reported separately as a bounded performance comparison",
            "multi-seed confirmation is required before a production-backend decision",
        ],
    }
    (run_dir / "aggregate_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# ViSNet spatial backbone v2 tiny comparison",
        "",
        f"Outcome classification: **{classification}** — {reason}.",
        "",
        "| Backbone | Implementation | Tiny quality | Final holdout total | Future dRMSD | Future RMSD | Future bond RMSE | Velocity RMSE | Frequency | Train min | k tokens/s | Holdout eval s/epoch | Peak alloc GiB | Peak reserved GiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.values():
        lines.append(
            f"| {row['spatial_backbone']} | {row['implementation']} | "
            f"{row['tiny_quality_pass']} | {row['final_holdout_total']:.6g} | "
            f"{row['future_drmsd']:.6g} | {row['future_rmsd']:.6g} | "
            f"{row['future_bond_rmse']:.6g} | {row['velocity_rmse']:.6g} | "
            f"{row['frequency_retention']:.6g} | "
            f"{row['training_elapsed_s'] / 60.0:.1f} | "
            f"{row['train_tokens_per_s'] / 1000.0:.1f} | "
            f"{row['holdout_eval_s_per_epoch']:.1f} | "
            f"{row['peak_allocated_memory_bytes'] / 2**30:.2f} | "
            f"{row['peak_reserved_memory_bytes'] / 2**30:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Locked protocol",
            "",
            "- 441 train clips and 117 late-holdout clips; train/holdout intersection is zero.",
            "- Lazy loading, replacement=false, exact no-replacement epoch coverage, FP32, max_tokens=80000, 30 epochs, and a 6000-step safety cap.",
            "- temporal_layers=1, temporal_ratio=1, unchanged decoder/loss/optimizer/evaluator.",
            "- v1 TorchMD and v1 bonded rows are immutable stored results; v2 has a separate result/checkpoint directory.",
            "",
            "## Gap recovery",
            "",
            f"- {json.dumps(diagnostics, sort_keys=True)}",
            "",
            "## Artifacts",
            "",
            f"- JSON: {run_dir / 'aggregate_comparison.json'}",
            f"- CSV: {run_dir / 'aggregate_comparison.csv'}",
            f"- Plots: {', '.join(plot_paths)}",
        ]
    )
    (run_dir / "aggregate_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    v1_run = _locate_v1_run()
    v1_manifest = _load_json(v1_run / "manifest_contract.json")
    v1_protocol = _load_json(v1_run / "protocol.json")
    manifest = v1_runner.build_manifest_contract()
    manifest_comparability = _assert_manifest_comparable(manifest, v1_manifest)
    protocol = {
        "schema_version": "pvb.codec.visnet_spatial.tiny_overfit.v2",
        "runtime": v1_runner.require_cuda(),
        "seed": v1_protocol["seed"],
        "spatial_backbones": [V2_BACKBONE],
        "lmax": V2_LMAX,
        "vertex_type": V2_VERTEX_TYPE,
        "rbf_type": "expnorm",
        "vecnorm_type": "max_min",
        "trainable_vecnorm": False,
        **{
            field: v1_protocol[field]
            for field in (
                "systems",
                "replicas",
                "train_windows",
                "late_holdout_windows",
                "train_count",
                "late_holdout_count",
                "frames_per_clip",
                "time_bucket_id",
                "lazy_loading",
                "replacement",
                "exact_epoch_coverage",
                "precision",
                "max_tokens",
                "lr",
                "weight_decay",
                "warmup_steps",
                "grad_clip",
                "epochs",
                "safety_cap_steps",
                "temporal_layers",
                "temporal_ratio",
            )
        },
        "source_v1_protocol_sha256": _sha256(v1_run / "protocol.json"),
        "source_v1_manifest_contract_sha256": _sha256(
            v1_run / "manifest_contract.json"
        ),
    }
    _assert_protocol_comparable(protocol, v1_protocol)
    run_dir = DEFAULT_OUTPUT / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "manifest_contract.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    v1_artifact_hashes = {
        name: _sha256(v1_run / name / "result.json")
        for name in ("torchmd_et", "visnet_radius", "visnet_bonded")
    }
    v1_artifact_hashes["manifest_contract.json"] = _sha256(
        v1_run / "manifest_contract.json"
    )
    v1_artifact_hashes["protocol.json"] = _sha256(v1_run / "protocol.json")

    # Reuse the frozen v1 loader/training/evaluation implementation in-process.
    v1_runner.make_model = make_model
    train_dataset, valid_dataset, train_loader, valid_loader = v1_runner.make_loaders()
    original_evaluate_loader = v1_runner._evaluate_loader
    original_graph_diagnostics = v1_runner._graph_diagnostics
    original_optimizer_step = v1_runner.CodecTrainer.optimizer_step
    v1_runner._evaluate_loader = _safe_evaluate_loader
    v1_runner._graph_diagnostics = _safe_graph_diagnostics
    v1_runner.CodecTrainer.optimizer_step = _logical_optimizer_step
    try:
        if len(train_dataset) != 441 or len(valid_dataset) != 117:
            raise RuntimeError("v2 tiny lazy dataset length is not 441/117")
        print(f"[v2-tiny] starting {V2_BACKBONE}", flush=True)
        v2_result = v1_runner.run_backbone(
            V2_BACKBONE,
            run_dir,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_loader=train_loader,
            valid_loader=valid_loader,
            protocol=protocol,
        )
    finally:
        v1_runner._evaluate_loader = original_evaluate_loader
        v1_runner._graph_diagnostics = original_graph_diagnostics
        v1_runner.CodecTrainer.optimizer_step = original_optimizer_step
        train_dataset.close()
        valid_dataset.close()

    baseline_results = {
        name: _load_json(v1_run / name / "result.json")
        for name in ("torchmd_et", "visnet_radius", "visnet_bonded")
    }
    report = _write_report(
        run_dir,
        protocol=protocol,
        manifest=manifest,
        v1_run=v1_run,
        results=baseline_results,
        v2_result=v2_result,
        v1_artifact_hashes=v1_artifact_hashes,
        manifest_comparability=manifest_comparability,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "manifest_sha256": manifest["built_manifest_sha256"],
                "selected_backbone": V2_BACKBONE,
                "classification": report["outcome_classification"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
