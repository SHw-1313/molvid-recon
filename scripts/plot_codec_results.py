#!/usr/bin/env python3
"""Generate codec training-loss and native-time evaluation plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTROL_ORDER = (
    "ratio1_no_temporal",
    "ratio1_temporal",
    "ratio4_temporal",
)
CONTROL_LABELS = {
    "ratio1_no_temporal": "ratio-1 / no temporal",
    "ratio1_temporal": "ratio-1 / temporal",
    "ratio4_temporal": "ratio-4 / temporal",
}
CONTROL_COLORS = {
    "ratio1_no_temporal": "#1f77b4",
    "ratio1_temporal": "#ff7f0e",
    "ratio4_temporal": "#2ca02c",
}
BUCKET_ORDER = ("dt_80ps", "dt_100ps", "dt_1ns", "static")
BUCKET_COLORS = {
    "dt_80ps": "#4c78a8",
    "dt_100ps": "#f58518",
    "dt_1ns": "#54a24b",
    "static": "#b279a2",
}


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _read_training_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "pvb.codec.train.v1":
                raise ValueError(f"{path}:{line_number}: unsupported training log schema")
            if not isinstance(record.get("metrics"), Mapping):
                raise ValueError(f"{path}:{line_number}: metrics must be a mapping")
            records.append(record)
    records.sort(key=lambda item: int(item["step"]))
    if not records:
        raise ValueError(f"training log is empty: {path}")
    return records


def _smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or values.size <= 1:
        return values, np.arange(values.size)
    width = min(int(window), int(values.size))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode="valid"), np.arange(width - 1, values.size)


def plot_loss_curves(
    logs: Mapping[str, Path],
    output_dir: Path,
    *,
    smooth_window: int,
) -> None:
    panels = (
        ("total", "Total loss", ("total",)),
        ("spatial", "Spatial losses", ("coordinate", "local", "bond")),
        ("temporal", "Normalized temporal losses", ("velocity", "acceleration")),
        ("raw_temporal", "Raw physical temporal losses", ("velocity_raw", "acceleration_raw")),
    )
    line_styles = {
        "total": "-",
        "coordinate": "-",
        "local": "--",
        "bond": ":",
        "velocity": "-",
        "acceleration": "--",
        "velocity_raw": "-",
        "acceleration_raw": "--",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.ravel()
    for axis, (_, title, metric_names) in zip(axes, panels):
        for control in CONTROL_ORDER:
            path = logs.get(control)
            if path is None:
                continue
            records = _read_training_log(path)
            steps = np.asarray([int(item["step"]) for item in records], dtype=np.int64)
            for metric_name in metric_names:
                values = np.asarray(
                    [float(item["metrics"].get(metric_name, np.nan)) for item in records],
                    dtype=np.float64,
                )
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                raw_steps = steps[finite]
                raw_values = values[finite]
                axis.plot(
                    raw_steps,
                    raw_values,
                    color=CONTROL_COLORS[control],
                    alpha=0.14,
                    linewidth=0.8,
                )
                smooth_values, smooth_indices = _smooth(raw_values, smooth_window)
                axis.plot(
                    raw_steps[smooth_indices],
                    smooth_values,
                    color=CONTROL_COLORS[control],
                    linestyle=line_styles[metric_name],
                    linewidth=2.0,
                    label=f"{CONTROL_LABELS[control]} / {metric_name}",
                )
        axis.set_title(title)
        axis.set_ylabel("loss")
        axis.set_yscale("symlog", linthresh=1.0e-6)
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[2].set_xlabel("optimizer step")
    axes[3].set_xlabel("optimizer step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("PVB codec training losses (raw traces and moving average)", y=1.03)
    fig.tight_layout()
    _save_figure(fig, output_dir, "loss_curves")


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    if name == "frame0_rmsd":
        return float(metrics["frame0"]["rmsd"])
    if name == "future_rmsd":
        return float(metrics["future"]["rmsd"])
    if name == "future_drmsd":
        return float(metrics["future"]["drmsd"])
    if name == "future_bond_rmse":
        return float(metrics["future"]["bond_rmse"])
    if name == "future_contact_error":
        return float(metrics["future"]["contact_error"])
    if name == "future_clash_rate":
        return float(metrics["future"]["clash_rate"])
    if name == "velocity_rmse":
        return float(metrics["velocity_rmse"])
    if name == "acceleration_rmse":
        return float(metrics["acceleration_rmse"])
    if name == "frequency_retention":
        return float(metrics["frequency_retention"])
    raise KeyError(name)


def _evaluation_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    controls = payload.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("evaluation JSON has no controls mapping")
    rows: list[dict[str, Any]] = []
    for control in CONTROL_ORDER:
        result = controls.get(control)
        if not isinstance(result, Mapping):
            continue
        by_bucket = result.get("by_time_bucket")
        if not isinstance(by_bucket, Mapping):
            continue
        for bucket in BUCKET_ORDER:
            entry = by_bucket.get(bucket)
            if not isinstance(entry, Mapping):
                continue
            metrics = entry.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            loss = entry.get("loss", {})
            if not isinstance(loss, Mapping):
                loss = {}
            row: dict[str, Any] = {
                "control": control,
                "bucket": bucket,
                "sample_count": int(entry.get("sample_count", 0)),
                "native_delta_time_ps": float(entry.get("native_delta_time_ps", np.nan)),
                "physical_clip_span_ps": float(entry.get("physical_clip_span_ps", np.nan)),
                "latent_interval_ps": float(entry.get("latent_interval_ps", np.nan)),
            }
            for metric_name in (
                "frame0_rmsd",
                "future_rmsd",
                "future_drmsd",
                "future_bond_rmse",
                "future_contact_error",
                "future_clash_rate",
                "velocity_rmse",
                "acceleration_rmse",
                "frequency_retention",
            ):
                row[metric_name] = _metric_value(metrics, metric_name)
            for loss_name in (
                "total",
                "coordinate",
                "local",
                "bond",
                "velocity",
                "acceleration",
                "velocity_raw",
                "acceleration_raw",
            ):
                row[f"validation_{loss_name}_loss"] = float(loss.get(loss_name, np.nan))
            rows.append(row)
    return rows


def _plot_metric_grid(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    stem: str,
    title: str,
    specs: tuple[tuple[str, str, bool], ...],
) -> None:
    present_controls = [control for control in CONTROL_ORDER if any(row["control"] == control for row in rows)]
    present_buckets = [
        bucket for bucket in BUCKET_ORDER if any(row["bucket"] == bucket for row in rows)
    ]
    if not present_controls or not present_buckets:
        raise ValueError("evaluation report has no plottable control/bucket rows")
    lookup = {(row["control"], row["bucket"]): row for row in rows}
    columns = 2
    figure, axes = plt.subplots(
        (len(specs) + columns - 1) // columns,
        columns,
        figsize=(15, 4.3 * ((len(specs) + columns - 1) // columns)),
    )
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(present_controls), dtype=np.float64)
    width = 0.72 / len(present_buckets)
    for axis, (metric_name, ylabel, log_scale) in zip(axes, specs):
        for bucket_index, bucket in enumerate(present_buckets):
            values = np.asarray(
                [
                    lookup[(control, bucket)].get(metric_name, np.nan)
                    for control in present_controls
                ],
                dtype=np.float64,
            )
            offset = (bucket_index - (len(present_buckets) - 1) / 2.0) * width
            plotted = values.copy()
            if log_scale:
                plotted[plotted <= 0] = np.nan
            axis.bar(
                x + offset,
                plotted,
                width=width,
                label=bucket,
                color=BUCKET_COLORS.get(bucket, "#777777"),
                alpha=0.9,
            )
        axis.set_title(ylabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels([CONTROL_LABELS[item] for item in present_controls], rotation=15, ha="right")
        if log_scale:
            axis.set_yscale("log")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    for axis in axes[len(specs):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=len(present_buckets))
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    _save_figure(figure, output_dir, stem)


def _write_csv(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with (output_dir / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_train_logs(entries: list[str], eval_json: Path) -> dict[str, Path]:
    if entries:
        paths: dict[str, Path] = {}
        for entry in entries:
            if "=" not in entry:
                raise ValueError("--train-log must use CONTROL=PATH")
            control, raw_path = entry.split("=", 1)
            if control not in CONTROL_ORDER:
                raise ValueError(f"unknown control in --train-log: {control}")
            paths[control] = Path(raw_path)
        return paths
    root = eval_json.parent
    return {
        control: root / control / "train_metrics.jsonl"
        for control in CONTROL_ORDER
        if (root / control / "train_metrics.jsonl").is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, default=Path("outputs/g5/codec_eval.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/g5/plots"))
    parser.add_argument("--train-log", action="append", default=[], help="CONTROL=PATH; repeat per control")
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be positive")
    with args.eval_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = _parse_train_logs(args.train_log, args.eval_json)
    if logs:
        plot_loss_curves(logs, args.output_dir, smooth_window=args.smooth_window)
    split = str(payload.get("evaluation_data", {}).get("source_split", "valid"))
    split_label = {"train": "Train", "valid": "Validation", "validation": "Validation"}.get(split, split)
    rows = _evaluation_rows(payload)
    _write_csv(rows, args.output_dir)
    _plot_metric_grid(
        rows,
        args.output_dir,
        stem="eval_loss",
        title=f"{split_label} loss by native time bucket",
        specs=(
            ("validation_total_loss", f"{split_label} total loss", False),
            ("validation_coordinate_loss", "Coordinate loss", False),
            ("validation_local_loss", "Local loss", False),
            ("validation_bond_loss", "Bond loss", False),
            ("validation_velocity_loss", "Normalized velocity loss", False),
            ("validation_acceleration_loss", "Normalized acceleration loss", False),
        ),
    )
    _plot_metric_grid(
        rows,
        args.output_dir,
        stem="eval_reconstruction",
        title="Codec reconstruction and structure metrics by native time bucket",
        specs=(
            ("frame0_rmsd", "Frame-0 RMSD (A)", False),
            ("future_rmsd", "Future RMSD (A)", False),
            ("future_drmsd", "Future dRMSD (A)", False),
            ("future_bond_rmse", "Future bond RMSE (A)", False),
        ),
    )
    _plot_metric_grid(
        rows,
        args.output_dir,
        stem="eval_dynamics",
        title="Codec dynamics and PVB-style physical metrics by native time bucket",
        specs=(
            ("velocity_rmse", "Velocity RMSE", False),
            ("acceleration_rmse", "Acceleration RMSE", False),
            ("future_contact_error", "Future contact error", True),
            ("future_clash_rate", "Future clash rate", True),
            ("frequency_retention", "FFT frequency retention", False),
        ),
    )
    print(f"wrote plots and CSV to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
