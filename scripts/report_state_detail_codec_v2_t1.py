#!/usr/bin/env python3
"""Build immutable T1 comparison tables and readable loss/evaluation plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "ratio1_state_detail",
    "ratio2_state_detail",
    "ratio4_state_detail",
    "ratio4_matched_pooling",
)
DISPLAY = {
    "ratio1_state_detail": "R1-SD",
    "ratio2_state_detail": "R2-SD",
    "ratio4_state_detail": "R4-SD",
    "ratio4_matched_pooling": "R4-Matched",
}
COLORS = {
    "ratio1_state_detail": "#386cb0",
    "ratio2_state_detail": "#f0027f",
    "ratio4_state_detail": "#1b9e77",
    "ratio4_matched_pooling": "#e66101",
}
DEFAULT_PROFILE_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/profiles_20260904"
DEFAULT_FULL_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/full_20260904_seed20260903"
DEFAULT_SELECTION_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/selection_20260904"
DEFAULT_SEEDS_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/additional_seeds_20260904"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/report_20260904"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_results(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = root / mode / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[mode] = json.loads(path.read_text(encoding="utf-8"))
    return result


def metric(result: Mapping[str, Any], *keys: str) -> float:
    value: Any = result
    for key in keys:
        if not isinstance(value, Mapping):
            return float("nan")
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def final_row(mode: str, result: Mapping[str, Any]) -> dict[str, Any]:
    final = result["final_validation"]
    future = final["metrics"]["future"]
    contract = result.get("model_contract", {})
    temporal = contract.get("architecture", {}).get("temporal", {}) if isinstance(contract, Mapping) else {}
    active_volume = temporal.get(
        "active_feature_volume_per_atom",
        {
            "ratio1_state_detail": 2048,
            "ratio2_state_detail": 2048,
            "ratio4_state_detail": 1024,
            "ratio4_matched_pooling": 1024,
        }[mode],
    )
    return {
        "control": DISPLAY[mode],
        "mode": mode,
        "seed": result["seed"],
        "ratio": 1 if mode == "ratio1_state_detail" else 2 if mode == "ratio2_state_detail" else 4,
        "active_feature_volume_per_atom": active_volume,
        "parameter_count": result["parameter_count"],
        "trainable_parameter_count": result["trainable_parameter_count"],
        "best_epoch": result["best_epoch"],
        "best_validation_aligned_rmsd": result["best_validation_future_aligned_rmsd"],
        "final_validation_aligned_rmsd": future["aligned_rmsd"],
        "final_validation_raw_rmsd": future["centroid_gauge_raw_rmsd"],
        "final_validation_drmsd": future["drmsd"],
        "final_validation_bond_rmse": future["bond_rmse"],
        "final_validation_contact_f1": future["contact_f1"],
        "final_validation_velocity_rmse": final["metrics"]["velocity_rmse"],
        "final_validation_acceleration_rmse": final["metrics"]["acceleration_rmse"],
        "final_validation_dynamic_correlation": final["extended_metrics"]["lag1_acf"]["dynamic_correlation"],
        "final_validation_rmsf_correlation": final["extended_metrics"]["rmsf"]["correlation"],
        "final_validation_frequency_power_ratio": future["frequency_power_ratio"] if "frequency_power_ratio" in future else future.get("frequency_retention"),
        "training_elapsed_s": result["training_elapsed_s"],
        "train_tokens_per_s": result["train_tokens_per_s"],
        "runtime_end_to_end_s_per_batch": result["runtime_profile"]["end_to_end_seconds"],
        "peak_allocated_memory_bytes": result["peak_allocated_memory_bytes"],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "best_checkpoint": result["best_checkpoint"],
        "best_checkpoint_sha256": result["best_checkpoint_sha256"],
    }


def _plot_loss_and_quality(results: Mapping[str, Mapping[str, Any]], output_root: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    plots: list[str] = []
    figure, axis = plt.subplots(figsize=(13, 7))
    annotations: list[str] = []
    for mode in MODES:
        result = results[mode]
        color = COLORS[mode]
        train_rows = result.get("epoch_metrics", [])
        train_points = []
        valid_points = []
        for row in train_rows:
            step = int(row["step"])
            value = metric(row, "last_train_metrics", "total")
            if math.isfinite(value) and value > 0:
                train_points.append((step, value))
            valid_value = metric(row, "validation_evaluation", "loss", "total")
            if math.isfinite(valid_value) and valid_value > 0:
                valid_points.append((step, valid_value))
        if train_points:
            axis.plot(
                [point[0] for point in train_points],
                [point[1] for point in train_points],
                color=color,
                linewidth=1.7,
                label=f"{DISPLAY[mode]} train",
            )
        if valid_points:
            axis.plot(
                [point[0] for point in valid_points],
                [point[1] for point in valid_points],
                color=color,
                linestyle="--",
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                label=f"{DISPLAY[mode]} validation",
            )
        annotations.append(
            f"{DISPLAY[mode]}: {result['train_tokens_per_s']/1000:.1f}k tok/s | "
            f"{result['training_elapsed_s']/3600:.2f} h | "
            f"{result['peak_allocated_memory_bytes']/2**30:.2f} GiB"
        )
    axis.set_yscale("log")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("total loss (log scale)")
    axis.set_title("T1 total loss: train (solid) vs validation (dashed)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=9, ncol=2)
    figure.text(0.5, 0.01, "\n".join(annotations), ha="center", va="bottom", family="monospace", fontsize=8)
    path = output_root / "loss_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(output_root / "loss_curves.pdf", bbox_inches="tight")
    plt.close(figure)
    plots.extend([str(path), str(output_root / "loss_curves.pdf")])

    specs = (
        ("Validation aligned RMSD", ("metrics", "future", "aligned_rmsd"), "Å"),
        ("Validation dRMSD", ("metrics", "future", "drmsd"), "Å"),
        ("Validation bond RMSE", ("metrics", "future", "bond_rmse"), "Å"),
        ("Validation dynamic correlation", ("extended_metrics", "lag1_acf", "dynamic_correlation"), "corr"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (title, keys, ylabel) in zip(axes.ravel(), specs):
        for mode in MODES:
            rows = results[mode].get("epoch_metrics", [])
            values = [metric(row["validation_evaluation"], *keys) for row in rows]
            axis.plot(
                [row["epoch"] for row in rows],
                values,
                marker="o",
                markersize=2.5,
                color=COLORS[mode],
                label=DISPLAY[mode],
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("T1 validation quality curves", fontsize=15)
    path = output_root / "validation_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(output_root / "validation_curves.pdf", bbox_inches="tight")
    plt.close(figure)
    plots.extend([str(path), str(output_root / "validation_curves.pdf")])

    rows = [final_row(mode, results[mode]) for mode in MODES]
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    chart_specs = (
        ("Training wall time", "h", [row["training_elapsed_s"] / 3600.0 for row in rows]),
        ("Train throughput", "k tok/s", [row["train_tokens_per_s"] / 1000.0 for row in rows]),
        ("End-to-end batch", "s", [row["runtime_end_to_end_s_per_batch"] for row in rows]),
        ("Peak allocated memory", "GiB", [row["peak_allocated_memory_bytes"] / 2**30 for row in rows]),
    )
    labels = [row["control"] for row in rows]
    for axis, (title, unit, values) in zip(axes, chart_specs):
        bars = axis.bar(labels, values, color=[COLORS[mode] for mode in MODES])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("T1 runtime and memory", fontsize=15)
    path = output_root / "performance_summary.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    figure.savefig(output_root / "performance_summary.pdf", bbox_inches="tight")
    plt.close(figure)
    plots.extend([str(path), str(output_root / "performance_summary.pdf")])
    return plots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--seeds-root", type=Path, default=DEFAULT_SEEDS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    profile_root = args.profile_root if args.profile_root.is_absolute() else ROOT / args.profile_root
    full_root = args.full_root if args.full_root.is_absolute() else ROOT / args.full_root
    selection_root = args.selection_root if args.selection_root.is_absolute() else ROOT / args.selection_root
    seeds_root = args.seeds_root if args.seeds_root.is_absolute() else ROOT / args.seeds_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite T1 report root: {output_root}")
    profiles = load_results(profile_root)
    results = load_results(full_root)
    selection_path = selection_root / "selection_rule.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    test_path = selection_root / "test_evaluation.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
    additional: dict[str, dict[str, Any]] = {}
    if seeds_root.is_dir():
        for path in sorted(seeds_root.glob("*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            additional[f"{result['mode']}_seed{result['seed']}"] = result
    rows = [final_row(mode, results[mode]) for mode in MODES]
    report = {
        "schema_version": "pvb.codec.state_detail.t1_report.v1",
        "status": "WAITING_FOR_OPERATOR_REVIEW",
        "profiles": profiles,
        "one_seed": results,
        "one_seed_rows": rows,
        "validation_selection": selection,
        "selected_test": test,
        "additional_seed_results": additional,
        "limitations": [
            "one frozen system-level 48/8/8 split",
            "one seed for the four primary controls",
            "additional seeds are limited to the two validation-ranked controls",
            "test was opened only after the validation selection rule was frozen",
            "no T1 scientific claim is made beyond this bounded split",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "t1_comparison.json", report)
    with (output_root / "t1_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots = _plot_loss_and_quality(results, output_root / "plots")
    report["plots"] = plots
    write_json(output_root / "t1_comparison.json", report)
    lines = [
        "# State/detail codec v2 T1 comparison",
        "",
        "Status: **WAITING_FOR_OPERATOR_REVIEW**.",
        "",
        "The four controls use the frozen 64-system 48/8/8 split, common frozen `torchmd_et` features, FP32, and one seed.",
        "",
        "| Control | Best epoch | Best valid aligned RMSD | Final valid aligned RMSD | Final valid dRMSD | Bond RMSE | Contact F1 | Train h | k tok/s | Peak GiB | Params / trainable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['control']} | {row['best_epoch']} | {row['best_validation_aligned_rmsd']:.7g} | "
            f"{row['final_validation_aligned_rmsd']:.7g} | {row['final_validation_drmsd']:.7g} | "
            f"{row['final_validation_bond_rmse']:.7g} | {row['final_validation_contact_f1']:.7g} | "
            f"{row['training_elapsed_s']/3600:.2f} | {row['train_tokens_per_s']/1000:.2f} | "
            f"{row['peak_allocated_memory_bytes']/2**30:.2f} | {row['parameter_count']:,} / {row['trainable_parameter_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Frozen protocol",
            "",
            "- Train: 48 systems × R1/R2/R3 × 24 sampled windows per trajectory per epoch; epoch-local sampling is without replacement and resampled deterministically across epochs.",
            "- Validation: 8 systems × R1/R2/R3 × all 62 windows; exact no-replacement coverage.",
            "- Test: 8 systems × R1/R2/R3 × all 62 windows; opened only after `selection_rule.json` was frozen from validation metrics.",
            "- Model: `torchmd_et`, shared frame checkpoint, frozen frame encoder, centered-vector stem, no spatial refiner, temporal layers=1, temporal ratio=1.",
            "- Loss: unchanged staged coordinate/local/bond/velocity/acceleration schedule; optimizer and max token budget unchanged.",
            "- R4-Matched is latent-volume-matched but parameter-count qualified; it is not parameter matched to R4-SD.",
            "",
            "## Selection and test",
            "",
            f"- Validation selection packet: `{selection_path}`.",
            f"- Selected control: `{selection['selected_mode'] if selection else 'pending'}`.",
            f"- Test packet: `{test_path if test else 'not opened'}`.",
            "- The test metric is not used to choose the ratio.",
            "",
            "## Plots",
            "",
            "- `plots/loss_curves.png` and `.pdf` use a logarithmic y-axis; train is solid and validation is same-color dashed.",
            "- Runtime annotations on the loss plot show training wall time, throughput, and peak allocated memory per control.",
            "- `plots/validation_curves.png` and `plots/performance_summary.png` provide validation and runtime views.",
            "",
            "## Stop",
            "",
            "T1 is complete for this bounded request. No later architecture phase, T2/T1 expansion, or full-data work was started.",
        ]
    )
    (output_root / "t1_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_paths = [
        output_root / "t1_comparison.json",
        output_root / "t1_comparison.csv",
        output_root / "t1_comparison.md",
        *[Path(path) for path in plots],
    ]
    write_json(
        output_root / "artifact_hashes.json",
        {str(path.relative_to(output_root)): sha256_file(path) for path in artifact_paths},
    )
    print(json.dumps({"report_root": str(output_root), "plots": plots, "rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
