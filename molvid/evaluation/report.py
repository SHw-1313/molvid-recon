"""Review-safe aggregation and reports for codec and DiT evaluation."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .latent import oracle_vs_generated


def _future(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = row.get("metrics", {})
    future = metrics.get("future", {}) if isinstance(metrics, Mapping) else {}
    return future if isinstance(future, Mapping) else {}


def _means(values: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    columns: dict[str, list[float]] = defaultdict(list)
    for row in values:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)):
                columns[str(key)].append(float(value))
    return {key: sum(items) / len(items) for key, items in columns.items()}


def aggregate_by_system(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average draws within clip, clips within system, then systems equally."""

    by_clip: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("system") or not row.get("sample_id"):
            raise ValueError("evaluation rows require system and sample_id")
        by_clip[(str(row["system"]), str(row["sample_id"]))].append(row)
    clip_rows = [
        {
            "system": system,
            "sample_id": sample_id,
            "draw_count": len(group),
            "future": _means([_future(row) for row in group]),
        }
        for (system, sample_id), group in sorted(by_clip.items())
    ]
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        by_system[str(row["system"])].append(row)
    system_rows = [
        {
            "system": system,
            "clip_count": len(group),
            "future": _means([row["future"] for row in group]),
        }
        for system, group in sorted(by_system.items())
    ]
    return {
        "aggregation": "draw_equal_then_clip_equal_then_system_equal",
        "row_count": len(rows),
        "draw_count": len({row.get("draw") for row in rows}),
        "clip_count": len(clip_rows),
        "system_count": len(system_rows),
        "system_equal": _means([row["future"] for row in system_rows]),
        "clip_rows": clip_rows,
        "system_rows": system_rows,
    }


def aggregate_by_time_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep physical-time buckets separate; never report an unqualified mean."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = row.get("time_bucket_id")
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("evaluation row lacks time_bucket_id")
        groups[bucket].append(row)
    return {bucket: aggregate_by_system(group) for bucket, group in sorted(groups.items())}


def compare_runs(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Retain both arms alongside their numeric difference."""

    return oracle_vs_generated(reference, candidate)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Molvid evaluation", "", "Codec-oracle and generated results remain separate.", ""]
    for key, label in (("codec_oracle", "Codec oracle"), ("generated_result", "Generated")):
        section = report.get(key, {})
        future = section.get("future", {}) if isinstance(section, Mapping) else {}
        if isinstance(future, Mapping):
            lines.append(f"- {label} future aligned RMSD: {future.get('aligned_rmsd', 'n/a')}")
            lines.append(f"- {label} future dRMSD: {future.get('drmsd', 'n/a')}")
    return "\n".join(lines) + "\n"


def write_report(
    report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path
) -> tuple[Path, Path]:
    """Write a machine-readable report and its compact review rendering."""

    json_file, markdown_file = Path(json_path), Path(markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_file.write_text(_markdown(report), encoding="utf-8")
    return json_file, markdown_file


def plot_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Plot future aligned RMSD per time bucket without cross-bucket pooling."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = report.get("by_time_bucket", {})
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("plot report requires nonempty by_time_bucket metrics")
    labels = sorted(buckets)
    values = []
    for label in labels:
        summary = buckets[label]
        value = summary.get("system_equal", {}).get("aligned_rmsd")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"time bucket {label!r} lacks finite aligned RMSD")
        values.append(float(value))
    figure, axis = plt.subplots(figsize=(max(4, len(labels) * 1.2), 3))
    axis.bar(labels, values)
    axis.set_ylabel("Future aligned RMSD (Å)")
    axis.set_xlabel("Physical-time bucket")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
