"""Build a read-only, source-linked digest of historical experiment results.

Run only inside enter-container / torch-ito. This script never opens clip payloads
or checkpoints and never alters outputs/. It refuses to overwrite an archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs"
DESTINATION = ROOT / "results_archive"
INCLUDED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".md", ".yaml", ".png", ".svg", ".pdf"}
EXCLUDED_PARTS = {"clip_store", "neibu_clip_store", "pair_stores", "test"}
PLOT_SUFFIXES = {".png", ".svg", ".pdf"}
HISTORY_NAMES = {
    "train_metrics.jsonl", "epoch_metrics.jsonl", "train_history.jsonl",
    "validation_history.jsonl", "topology_five_epochs.jsonl",
    "distance_only_five_epochs.jsonl", "training_loss.csv", "validation_loss.csv",
    "resume_metrics.jsonl",
}
DOCUMENT_NAMES = {
    "protocol.json", "summary.json", "train_summary.json", "decision.json",
    "result.json", "pilot_summary.json", "aggregate_comparison.json",
    "selection.json", "status.json", "recheck_summary.json",
}
METRIC_NAMES = {
    "loss", "total", "train_loss", "valid_loss", "validation_loss", "rf_loss",
    "aligned_rmsd", "rmsd", "bond_rmse", "contact_f1", "contact_error",
    "clash_rate", "drmsd", "velocity_rmse", "acceleration_rmse",
    "rmsf_error", "mean_aligned_rmsd",
}
FIELDS = {
    "data": ("data_hash", "manifest_root", "data_root", "system_count", "train_systems", "valid_systems", "frames_per_clip", "clip_count", "time_bucket_id"),
    "model": ("model_type", "mode", "ratio", "backbone", "spatial_backbone", "execution_backend", "ffn_norm_source", "scalar_width", "vector_width", "depth", "heads", "source_mode"),
    "training": ("epochs", "steps", "target_steps", "completed_steps", "successful_updates", "successful_updates_per_arm", "lr", "learning_rate", "seed", "precision", "max_tokens"),
    "result": ("status", "selected_arm", "primary_metric", "aligned_rmsd", "bond_rmse", "contact_f1", "loss", "total"),
}
FAMILY_NAMES = {"pvb_origin_baselines": "origin_baselines"}


def _source_files() -> list[Path]:
    found = []
    for directory, subdirs, filenames in os.walk(SOURCE, followlinks=False):
        subdirs[:] = sorted(
            name for name in subdirs
            if name not in EXCLUDED_PARTS and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_symlink() or path.suffix.lower() not in INCLUDED_SUFFIXES:
                continue
            if not path.is_file():
                continue
            found.append(path)
    return sorted(found)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _archive_name(path: Path) -> str:
    parts = path.relative_to(SOURCE).parts
    family = FAMILY_NAMES.get(parts[0], parts[0])
    stem = "__".join(_safe(part) for part in parts[1:-1] + (path.stem,))
    return "raw/" + family + "/" + stem + path.suffix.lower()


def _date(path: Path) -> str:
    for part in path.relative_to(SOURCE).parts:
        if part.startswith("seed"):
            continue
        match = re.search(r"(?:^|_)(20\d{6})(?:T(\d{6}))?", part)
        if match:
            stamp = match.group(1)
            day = stamp[:4] + "-" + stamp[4:6] + "-" + stamp[6:]
            if match.group(2):
                clock = match.group(2)
                day += " " + clock[:2] + ":" + clock[2:4] + ":" + clock[4:]
            return day
    return "未记录"


def _run_key(path: Path) -> str:
    parts = path.relative_to(SOURCE).parts
    family = FAMILY_NAMES.get(parts[0], parts[0])
    directories = parts[1:-1]
    if not directories:
        return family + "/overview"
    if family == "dit_architecture_sequential_v1":
        stage = next((part for part in directories if part.startswith("round") or part == "baseline"), "overview")
        return family + "/" + directories[0] + "/" + stage
    if family == "dit_state_detail_pilot_v1":
        ratio = next((part for part in directories if part.startswith("ratio")), "overview")
        return family + "/" + directories[0] + "/" + ratio
    if family == "dit_source_ab_v1":
        arm = next((part for part in directories if part in {"gaussian", "conditional"}), "overview")
        return family + "/" + directories[0] + "/" + arm
    for index, part in enumerate(directories):
        if part.startswith("run_") or re.match(r"20\d{6}[_T]", part):
            return family + "/" + "/".join(directories[: index + 1])
    return family + "/" + directories[0]


def _read_json(path: Path) -> Mapping | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _scalars(value: object, prefix: tuple[str, ...] = ()):
    if not isinstance(value, Mapping) or len(prefix) > 6:
        return
    for key, item in value.items():
        path = prefix + (str(key),)
        if isinstance(item, Mapping):
            yield from _scalars(item, path)
        elif isinstance(item, (str, int, float, bool)) and len(str(item)) <= 110:
            yield path, item


def _short(value: object, limit: int = 60) -> str:
    result = str(value).replace("|", "/").replace("\n", " ")
    return result if len(result) <= limit else result[: limit - 1] + "…"


def _field_summary(documents: list[tuple[Path, Mapping]], category: str, maximum: int = 4) -> str:
    names = FIELDS[category]
    ordered = sorted(documents, key=lambda item: (
        0 if (item[0].name in {"protocol.json", "train_summary.json"}) == (category != "result") else 1,
        str(item[0]),
    ))
    found = {}
    for _, document in ordered:
        for path, value in _scalars(document):
            name = path[-1]
            if name in names and name not in found:
                found[name] = _short(value)
    return "; ".join(name + "=" + found[name] for name in names if name in found)[:400] or "未记录"


def _history(path: Path) -> dict[str, list[tuple[float, float]]]:
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                try:
                    label = row.get("condition") or path.stem
                    values[label].append((float(row["step"]), float(row["loss"])))
                except (KeyError, TypeError, ValueError):
                    continue
        return values
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, Mapping):
                continue
            position = row.get("epoch", row.get("step"))
            try:
                x = float(position)
            except (TypeError, ValueError):
                continue
            candidates = {
                "train": row.get("loss", row.get("total")),
                "validation": row.get("total") if path.name == "validation_history.jsonl" else None,
            }
            metrics = row.get("metrics")
            if isinstance(metrics, Mapping):
                candidates["train"] = metrics.get("total", candidates["train"])
            last = row.get("last_train_metrics")
            if isinstance(last, Mapping):
                inner = last.get("metrics", last)
                if isinstance(inner, Mapping):
                    candidates["train"] = inner.get("total", candidates["train"])
            holdout = row.get("holdout_evaluation")
            if isinstance(holdout, Mapping):
                holdout_loss = holdout.get("loss")
                if isinstance(holdout_loss, Mapping):
                    candidates["holdout"] = holdout_loss.get("total")
            if path.name == "validation_history.jsonl":
                candidates.pop("train", None)
            for label, value in candidates.items():
                try:
                    y = float(value)
                except (TypeError, ValueError):
                    continue
                if y == y and abs(y) != float("inf"):
                    values[label].append((x, y))
    return values


def _plot_history(path: Path, destination: Path) -> bool:
    series = _history(path)
    if not any(series.values()):
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, points in sorted(series.items()):
        if not points:
            continue
        stride = max(1, len(points) // 3000)
        drawn = points[::stride]
        if drawn[-1] != points[-1]:
            drawn.append(points[-1])
        axis.plot([x for x, _ in drawn], [y for _, y in drawn], label=label, linewidth=1.2)
    axis.set_xlabel("epoch" if "epoch" in path.name or "five_epochs" in path.name else "step")
    axis.set_ylabel("loss")
    axis.set_title(" / ".join(path.relative_to(SOURCE).parts[-3:]), fontsize=9)
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=135)
    plt.close(figure)
    return True


def _plot_pilot_validation(path: Path, destination: Path) -> bool:
    document = _read_json(path)
    validation = document.get("validation") if document is not None else None
    history = validation.get("history") if isinstance(validation, Mapping) else None
    if not isinstance(history, list):
        return False
    points = [
        (float(row["step"]), float(row["total"]))
        for row in history
        if isinstance(row, Mapping) and isinstance(row.get("step"), (int, float))
        and isinstance(row.get("total"), (int, float))
    ]
    if not points:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([x for x, _ in points], [y for _, y in points], linewidth=1.5)
    axis.set(xlabel="step", ylabel="validation RF loss", title=" / ".join(path.relative_to(SOURCE).parts[-3:]))
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=135)
    plt.close(figure)
    return True


def _metric_source(path: Path) -> bool:
    if path.name in {"evaluation.json", "evaluation_latest.json", "latest_metrics.json"}:
        return True
    if path.name.endswith("_metrics.json") and path.suffix == ".json":
        return True
    return path.name == "summary.json" and any(part.startswith("evaluation_") for part in path.parts)


def _write_archive(root: Path) -> tuple[int, int, int]:
    files = _source_files()
    if not files:
        raise RuntimeError("no historical result files found")
    manifest_rows = []
    used_names = set()
    raw_path = root / "raw_results.tar.gz"
    with tarfile.open(raw_path, "w:gz") as bundle:
        for path in files:
            member = _archive_name(path)
            if member in used_names:
                raise ValueError("archive name collision: " + member)
            used_names.add(member)
            bundle.add(path, arcname=member, recursive=False)
            manifest_rows.append({
                "source": str(path.relative_to(ROOT)), "archive_member": member,
                "sha256": _digest(path), "bytes": path.stat().st_size,
                "run": _run_key(path), "path_date": _date(path),
            })
    with (root / "raw_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    figures = root / "figures"
    figures.mkdir()
    figure_links: dict[str, list[str]] = defaultdict(list)
    generated = 0
    for path in files:
        name = _archive_name(path).removeprefix("raw/").replace("/", "__")
        family = FAMILY_NAMES.get(path.relative_to(SOURCE).parts[0], path.relative_to(SOURCE).parts[0])
        if path.suffix.lower() in PLOT_SUFFIXES:
            shutil.copy2(path, figures / name)
            figure_links[family].append(name)
        elif path.name in HISTORY_NAMES:
            chart = Path(name).with_suffix(".loss.png").name
            if _plot_history(path, figures / chart):
                generated += 1
                figure_links[family].append(chart)
        elif path.name == "pilot_summary.json":
            chart = Path(name).with_suffix(".validation.loss.png").name
            if _plot_pilot_validation(path, figures / chart):
                generated += 1
                figure_links[family].append(chart)

    groups: dict[str, list[tuple[Path, Mapping]]] = defaultdict(list)
    metric_rows = []
    for path in files:
        if path.suffix != ".json":
            continue
        if path.name not in DOCUMENT_NAMES and not _metric_source(path):
            continue
        document = _read_json(path)
        if document is None:
            continue
        if path.name in DOCUMENT_NAMES:
            groups[_run_key(path)].append((path, document))
        if _metric_source(path):
            scope = "final" if "evaluation_final" in path.parts else "quick" if "evaluation_quick" in path.parts else "unspecified"
            for key_path, value in _scalars(document):
                if key_path[-1] in METRIC_NAMES and isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_rows.append({
                        "run": _run_key(path), "scope": scope,
                        "source": str(path.relative_to(ROOT)),
                        "metric_path": ".".join(key_path), "value": value,
                    })
    with (root / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "scope", "source", "metric_path", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)

    family_groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted({_run_key(path) for path in files}):
        family_groups[key.split("/", 1)[0]].append(key)
    report_dir = root / "runs"
    report_dir.mkdir()
    index = [
        "# 实验结果整理版", "",
        "先看 [关键实验结论](KEY_RESULTS.md)，再按实验组进入具体运行。关键表按",
        "生成时间—数据设置—模型设置—训练设置—结果组织。时间取自路径中的运行标识，",
        "不是 Git 检出后的文件 mtime；未记录的时间不推断。所有数值来自列出的原始结果文件，",
        "未重训、未打开 test clip、未改动 outputs/。不同运行或 quick/final 评估不混作同一结论。", "",
        "逐 epoch/step loss 见 figures/ 中的曲线；最终或单次 checkpoint 指标仅保留在",
        "[evaluation_metrics.csv](evaluation_metrics.csv)，按 scope 区分 final/quick/未注明，不为其造图。", "",
        "原始 JSON/JSONL/CSV/已有图片被语义化命名后装入 [raw_results.tar.gz](raw_results.tar.gz)。",
        "[raw_manifest.csv](raw_manifest.csv) 记录原路径、新名称、SHA-256、大小与运行标识。",
        "训练数据、checkpoint、日志和 test split 不在包内；原始 outputs/ 原封不动保留。", "",
        "| 实验组 | 生成时间（路径） | 运行数 | 图数 |",
        "| --- | --- | ---: | ---: |",
    ]
    for family, keys in sorted(family_groups.items()):
        family_docs = [item for key in keys for item in groups[key]]
        dates = sorted({date for path, _ in family_docs if (date := _date(path)) != "未记录"})
        date_text = dates[0] if len(dates) == 1 else (dates[0] + "～" + dates[-1] if dates else "未记录")
        page = report_dir / (family + ".md")
        index.append("| [" + family + "](runs/" + family + ".md) | " + date_text
                     + " | " + str(len(keys)) + " | " + str(len(figure_links[family])) + " |")
        lines = ["# " + family, "", "生成时间由运行路径标识推断；下表按运行而非按原始文件逐行展开。", "",
                 "| 运行 | 时间 | 数据设置 | 模型设置 | 训练设置 | 结果 | 主要原始记录 |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for key in keys:
            docs = groups[key]
            dates = sorted({date for path, _ in docs if (date := _date(path)) != "未记录"})
            source_names = ", ".join(str(path.relative_to(SOURCE)) for path, _ in docs[:3])
            lines.append("| " + key.split("/", 1)[1] + " | " + (dates[0] if dates else "未记录") + " | " + " | ".join(
                _field_summary(docs, category) for category in FIELDS
            ) + " | " + _short(source_names, 160) + " |")
        lines += ["", "## 图", ""]
        if figure_links[family]:
            for name in sorted(figure_links[family]):
                lines.append("- [" + name + "](../figures/" + name + ")")
        else:
            lines.append("此组未发现可画的训练 loss 序列或原有图；单次评估保留数据。")
        lines += ["", "[原始文件清单](../raw_manifest.csv) · [评估指标数据](../evaluation_metrics.csv)", ""]
        page.write_text("\n".join(lines), encoding="utf-8")
    index += ["", "归档共 " + str(len(files)) + " 份结构化文件/原有图，新增 " + str(generated) + " 张训练曲线，",
              "提取 " + str(len(metric_rows)) + " 个评估标量。若某个设置或结果显示“未记录”，请查该运行的原始记录；不要把缺失值视为零。", ""]
    (root / "README.md").write_text("\n".join(index), encoding="utf-8")

    with tarfile.open(raw_path, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != len(manifest_rows):
            raise RuntimeError("raw archive member count changed")
        for member, row in zip(members, manifest_rows):
            if member.name != row["archive_member"]:
                raise RuntimeError("raw archive member ordering changed")
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError("raw archive member is unreadable")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise RuntimeError("raw archive hash mismatch: " + member.name)
    return len(files), generated, len(metric_rows)


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError("refusing to overwrite existing results_archive")
    with tempfile.TemporaryDirectory(prefix="results_archive_", dir=ROOT) as temporary:
        temporary_root = Path(temporary)
        counts = _write_archive(temporary_root)
        temporary_root.rename(DESTINATION)
        print(f"Archived {counts[0]} raw files, {counts[1]} new loss plots, {counts[2]} evaluation metrics: {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
