#!/usr/bin/env python3
"""Aggregate the four PVB_origin baseline runs and make comparable plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS: dict[str, dict[str, str]] = {
    "A": {
        "label": "A static pretrain / direct",
        "short_label": "A static-direct",
        "initialization": "static",
        "mode": "direct",
        "metrics": "static_direct/baseline_a_metrics.json",
        "train_log": "",
        "valid": "",
    },
    "B": {
        "label": "B dynamic pretrain / direct",
        "short_label": "B dynamic-direct",
        "initialization": "dynamic",
        "mode": "direct",
        "metrics": "dynamic_direct/baseline_b_metrics.json",
        "train_log": "",
        "valid": "",
    },
    "C": {
        "label": "C static pretrain / retrain",
        "short_label": "C static-retrain",
        "initialization": "static",
        "mode": "retrain",
        "metrics": "static_retrain/eval_raw/baseline_b_metrics.json",
        "train_log": "static_retrain/train_pilot500.log",
        "valid": "static_retrain/valid_eval/baseline_d_valid_metrics.json",
    },
    "D": {
        "label": "D dynamic pretrain / retrain",
        "short_label": "D dynamic-retrain",
        "initialization": "dynamic",
        "mode": "retrain",
        "metrics": "dynamic_retrain/eval_raw/baseline_b_metrics.json",
        "train_log": "dynamic_retrain/baseline_d_train_gpu4_main.log",
        "valid": "dynamic_retrain/valid_eval/baseline_d_valid_metrics.json",
    },
}

BUCKETS = ("dt_100ps", "dt_80ps")
EVAL_METRICS = (
    ("one_step_rmsd", "one-step RMSD", "Å"),
    ("one_step_drmsd", "one-step dRMSD", "Å"),
    ("one_step_bond_rmse", "one-step bond RMSE", "Å"),
    ("one_step_contact_error", "one-step contact error", "fraction"),
    ("future_rmsd", "future RMSD", "Å"),
    ("future_drmsd", "future dRMSD", "Å"),
    ("future_bond_rmse", "future bond RMSE", "Å"),
    ("future_contact_error", "future contact error", "fraction"),
    ("future_clash_rate", "future clash rate", "fraction"),
    ("future_velocity_rmse", "future velocity RMSE", "Å/ps"),
    ("future_acceleration_rmse", "future acceleration RMSE", "Å/ps²"),
    ("frequency_retention", "frequency retention", "ratio"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _direct_one_step(bucket: dict[str, Any]) -> dict[str, Any]:
    return bucket.get("direct_one_step", bucket.get("one_step", {}))


def _metric_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, spec in CONDITIONS.items():
        report_path = root / spec["metrics"]
        report = _read_json(report_path)
        for bucket in BUCKETS:
            bucket_report = report["by_time_bucket"][bucket]
            one = _direct_one_step(bucket_report)
            rollout = bucket_report["rollout16"]
            future = rollout["future"]
            values = {
                "one_step_rmsd": one["rmsd"],
                "one_step_drmsd": one["drmsd"],
                "one_step_bond_rmse": one["bond_rmse"],
                "one_step_contact_error": one["contact_error"],
                "future_rmsd": future["rmsd"],
                "future_drmsd": future["drmsd"],
                "future_bond_rmse": future["bond_rmse"],
                "future_contact_error": future["contact_error"],
                "future_clash_rate": future["clash_rate"],
                "future_velocity_rmse": rollout["velocity_rmse"],
                "future_acceleration_rmse": rollout["acceleration_rmse"],
                "frequency_retention": rollout["frequency_retention"],
            }
            for metric, _, _ in EVAL_METRICS:
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": spec["label"],
                        "initialization": spec["initialization"],
                        "mode": spec["mode"],
                        "bucket": bucket,
                        "native_delta_time_ps": bucket_report["native_delta_time_ps"],
                        "sample_count": bucket_report["sample_count"],
                        "metric": metric,
                        "value": values[metric],
                        "report": str(report_path),
                    }
                )
    return rows


def _parse_train_log(path: Path, condition: str) -> list[dict[str, Any]]:
    if not path:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    latest: dict[int, float] = {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(
        r"(?P<step>\d{1,5})/1000[^\r\n]*?loss=(?P<loss>" + number + r")"
    )
    for line in text.replace("\r", "\n").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        step = int(match.group("step"))
        if 1 <= step <= 1000:
            latest[step] = float(match.group("loss"))
    return [
        {"condition": condition, "step": step, "loss": latest[step], "source": str(path)}
        for step in sorted(latest)
    ]


def _valid_summary(root: Path, spec: dict[str, str]) -> dict[str, Any] | None:
    if not spec["valid"]:
        return None
    path = root / spec["valid"]
    report = _read_json(path)
    return {
        "overall_loss_mean": report["overall_loss_mean"],
        "per_source": report["per_source"],
        "max_batches_total": report["max_batches_total"],
        "batches_per_source": report["batches_per_source"],
        "pair_bound": report["pair_bound"],
        "source_report": str(path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_loss(out: Path, loss_rows: list[dict[str, Any]], valid: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"C": "#1769aa", "D": "#d95f02"}
    for condition in ("C", "D"):
        rows = [row for row in loss_rows if row["condition"] == condition]
        if not rows:
            continue
        steps = np.asarray([row["step"] for row in rows], dtype=float)
        values = np.asarray([row["loss"] for row in rows], dtype=float)
        ax.plot(steps, values, color=colors[condition], alpha=0.18, linewidth=0.7)
        window = min(25, len(values))
        if window > 1:
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :],
                smooth,
                color=colors[condition],
                linewidth=2.0,
                label=f"{condition}: {CONDITIONS[condition]['label']}",
            )
        else:
            ax.plot(steps, values, color=colors[condition], linewidth=2.0, label=condition)
        if condition in valid:
            ax.axhline(
                valid[condition]["overall_loss_mean"],
                color=colors[condition],
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=f"{condition} valid mean",
            )
    ax.text(
        0.99,
        0.98,
        "A/B direct inference: no training loss",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.set_title("Unmodified PVB_origin retraining loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("legacy PVB training loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=180)
    fig.savefig(out / "loss_curves.pdf")
    plt.close(fig)


def _group_values(rows: list[dict[str, Any]], metric: str, bucket: str) -> np.ndarray:
    values = []
    for condition in CONDITIONS:
        match = next(
            row["value"]
            for row in rows
            if row["condition"] == condition and row["bucket"] == bucket and row["metric"] == metric
        )
        values.append(float(match))
    return np.asarray(values, dtype=float)


def _plot_metric_grid(
    out: Path,
    rows: list[dict[str, Any]],
    metrics: list[str],
    name: str,
    title: str,
) -> None:
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), squeeze=False)
    x = np.arange(len(CONDITIONS), dtype=float)
    width = 0.36
    colors = {"dt_100ps": "#4c78a8", "dt_80ps": "#f58518"}
    for index, metric in enumerate(metrics):
        ax = axes.flat[index]
        for offset, bucket in zip((-width / 2, width / 2), BUCKETS):
            values = _group_values(rows, metric, bucket)
            ax.bar(x + offset, values, width, label=bucket, color=colors[bucket])
        label = next(label for key, label, _ in EVAL_METRICS if key == metric)
        unit = next(unit for key, _, unit in EVAL_METRICS if key == metric)
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.set_xticks(x, list(CONDITIONS))
        ax.grid(axis="y", alpha=0.25)
        if "contact" in metric or "acceleration" in metric or "velocity" in metric:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for axis in axes.flat[len(metrics) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _write_markdown(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Unmodified PVB_origin four-condition baseline",
        "",
        "All four conditions use seed 20260810, the same 32-batch/48-clip validation selection, T=16 sequential rollout, and 10 SDE steps. PDB is excluded.",
        "",
        "The structure/motion values below are PVB-style clip metrics computed by the same output-local evaluator used for A/B. They are not the original eval_prot.py TICA/MSM numbers because the current versioned clip stores do not carry the original gzip-JSON trajectory paths.",
        "",
        "## Validation loss",
        "",
        "| condition | initialization | mode | valid loss mean |",
        "|---|---|---|---:|",
    ]
    for condition in CONDITIONS:
        item = summary["conditions"][condition]
        value = item.get("valid", {}).get("overall_loss_mean") if item.get("valid") else None
        rendered = "—" if value is None else f"{value:.6f}"
        lines.append(f"| {condition} | {item['initialization']} | {item['mode']} | {rendered} |")
    lines.extend(
        [
            "",
            "## Future-frame comparison",
            "",
            "| condition | bucket | RMSD | dRMSD | bond RMSE | contact error | velocity RMSE | acceleration RMSE | frequency retention |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        for bucket in BUCKETS:
            row = summary["metrics"][condition][bucket]
            lines.append(
                f"| {condition} | {bucket} | {row['future_rmsd']:.6f} | {row['future_drmsd']:.6f} | {row['future_bond_rmse']:.6f} | {row['future_contact_error']:.3e} | {row['future_velocity_rmse']:.6f} | {row['future_acceleration_rmse']:.3e} | {row['frequency_retention']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- loss_curves.png / loss_curves.pdf: retraining loss with validation-mean markers; direct conditions correctly have no training curve.",
            "- eval_reconstruction.png / eval_reconstruction.pdf: one-step and future reconstruction metrics by native time bucket.",
            "- eval_dynamics.png / eval_dynamics.pdf: motion, contact/clash, and frequency metrics by native time bucket.",
            "- eval_metrics.csv, training_loss.csv, and summary.json: machine-readable records.",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    metric_rows = _metric_rows(root)
    loss_rows: list[dict[str, Any]] = []
    valid: dict[str, Any] = {}
    metrics_by_condition: dict[str, dict[str, dict[str, float]]] = {}
    condition_summary: dict[str, Any] = {}
    for condition, spec in CONDITIONS.items():
        log_path = root / spec["train_log"] if spec["train_log"] else None
        if log_path is not None:
            loss_rows.extend(_parse_train_log(log_path, condition))
        valid_report = _valid_summary(root, spec)
        if valid_report is not None:
            valid[condition] = valid_report
        metrics_by_condition[condition] = {}
        for bucket in BUCKETS:
            metrics_by_condition[condition][bucket] = {
                row["metric"]: float(row["value"])
                for row in metric_rows
                if row["condition"] == condition and row["bucket"] == bucket
            }
        raw_report = _read_json(root / spec["metrics"])
        condition_summary[condition] = {
            "label": spec["label"],
            "short_label": spec["short_label"],
            "initialization": spec["initialization"],
            "mode": spec["mode"],
            "checkpoint": raw_report.get("checkpoint"),
            "checkpoint_sha256": raw_report.get("checkpoint_sha256"),
            "metrics_report": str(root / spec["metrics"]),
            "train_log": str(log_path) if log_path is not None else None,
            "valid": valid_report,
        }

    eval_fields = [
        "condition",
        "condition_label",
        "initialization",
        "mode",
        "bucket",
        "native_delta_time_ps",
        "sample_count",
        "metric",
        "value",
        "report",
    ]
    _write_csv(out / "eval_metrics.csv", metric_rows, eval_fields)
    _write_csv(out / "training_loss.csv", loss_rows, ["condition", "step", "loss", "source"])
    _write_csv(
        out / "validation_loss.csv",
        [
            {
                "condition": condition,
                "overall_loss_mean": report["overall_loss_mean"],
                "atlas_loss_mean": report["per_source"]["atlas_dt_100ps"]["loss_mean"],
                "misato_loss_mean": report["per_source"]["misato_dt_80ps"]["loss_mean"],
                "source": report["source_report"],
            }
            for condition, report in valid.items()
        ],
        ["condition", "overall_loss_mean", "atlas_loss_mean", "misato_loss_mean", "source"],
    )

    summary = {
        "schema_version": "pvb.origin.baselines.aggregate.v1",
        "reference_repo": "/data4/users/sihao/workspace/PVB_origin",
        "reference_commit": "c08e5e3cd49d45c6d748387e78224843bd356f50",
        "pdb_included": False,
        "policy": {
            "seed": 20260810,
            "max_tokens": 80000,
            "max_batches": 32,
            "selected_clips": 48,
            "selected_by_bucket": {"dt_100ps": 20, "dt_80ps": 28},
            "rollout_frames": 16,
            "sde_steps": 10,
            "train_pair_bound": 5000,
            "train_batches_per_source": 500,
            "train_max_epoch": 1,
        },
        "conditions": condition_summary,
        "metrics": metrics_by_condition,
        "training_loss_records": len(loss_rows),
        "validation": valid,
        "artifacts": {
            "eval_metrics_csv": str(out / "eval_metrics.csv"),
            "training_loss_csv": str(out / "training_loss.csv"),
            "validation_loss_csv": str(out / "validation_loss.csv"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_loss(out, loss_rows, valid)
    _plot_metric_grid(
        out,
        metric_rows,
        ["one_step_rmsd", "one_step_drmsd", "one_step_bond_rmse", "one_step_contact_error", "future_rmsd", "future_drmsd"],
        "eval_reconstruction",
        "PVB-style reconstruction metrics by condition and native time bucket",
    )
    _plot_metric_grid(
        out,
        metric_rows,
        ["future_bond_rmse", "future_contact_error", "future_clash_rate", "future_velocity_rmse", "future_acceleration_rmse", "frequency_retention"],
        "eval_dynamics",
        "PVB-style dynamics metrics by condition and native time bucket",
    )
    _write_markdown(out, summary)
    print(json.dumps({"output": str(out), "loss_records": len(loss_rows), "metric_rows": len(metric_rows), "valid_conditions": sorted(valid)}, sort_keys=True))


if __name__ == "__main__":
    main()
