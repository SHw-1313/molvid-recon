"""Prepare disjoint direct-store shards and combine frame-statistics fits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any

import torch
import yaml

from molvid.latent.statistics import FrameLatentStatistics
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file


def _index_rows(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _link_or_replace(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def prepare(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    train_index = source_root / "clip_store" / "train" / "index.txt"
    valid_root = source_root / "clip_store" / "valid"
    rows = _index_rows(train_index)
    if len(rows) < args.shards:
        raise ValueError("training store is smaller than the requested shard count")
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    for shard in range(args.shards):
        root = output / f"shard{shard}"
        train = root / "train"
        train.mkdir(parents=True)
        _link_or_replace(train / "data.bin", source_root / "clip_store" / "train" / "data.bin")
        selected = rows[shard::args.shards]
        (train / "index.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
        (train / "stats.json").write_text(json.dumps({
            "count": len(selected),
            "compressed_bytes": int((train / "data.bin").stat().st_size),
            "storage_format": "npz-v1",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        valid_link = root / "valid"
        _link_or_replace(valid_link, valid_root)
        (root / "counts.json").write_text(json.dumps({
            "shard": shard,
            "count": len(selected),
            "atom_frames": sum(int(row.split("\t")[3]) * int(row.split("\t")[4]) for row in selected),
            "index_sha256": sha256_file(train / "index.txt"),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_root": str(output),
        "shards": args.shards,
        "train_count": len(rows),
        "source_index_sha256": sha256_file(train_index),
    }, sort_keys=True))


def write_configs(args: argparse.Namespace) -> None:
    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    output = Path(args.output_root).resolve()
    for shard in range(args.shards):
        config: dict[str, Any] = copy.deepcopy(base)
        root = output / f"shard{shard}"
        config["data"] = {
            "train_store": str(root / "train"),
            "valid_store": str(root / "valid"),
        }
        config["statistics"]["path"] = str(root / "frame_statistics.pt")
        config["statistics"]["sha256"] = ""
        config["statistics"]["statistics_hash"] = ""
        config["training"]["output_root"] = str(root / "fit_output")
        path = output / f"config_shard{shard}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"configs": [str(output / f"config_shard{i}.yaml") for i in range(args.shards)]}))


def combine(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    states = []
    total = 0.0
    for shard in range(args.shards):
        root = output / f"shard{shard}"
        state = torch.load(root / "frame_statistics.pt", map_location="cpu", weights_only=False)
        stats = FrameLatentStatistics.from_state_dict(state)
        count = float(json.loads((root / "counts.json").read_text())["atom_frames"])
        states.append((stats, count))
        total += count
    if not states or total <= 0:
        raise ValueError("no shard statistics found")
    first = states[0][0]
    if any(stats.width != first.width for stats, _ in states):
        raise ValueError("shard statistics widths differ")
    h_sum = torch.zeros(first.width, dtype=torch.float64)
    h_square = torch.zeros_like(h_sum)
    v_square = torch.zeros_like(h_sum)
    for stats, count in states:
        h_mean = stats.h_mean.to(dtype=torch.float64)
        h_std = stats.h_std.to(dtype=torch.float64)
        v_rms = stats.v_rms.to(dtype=torch.float64)
        h_sum += h_mean * count
        h_square += (h_std.square() + h_mean.square()) * count
        v_square += v_rms.square() * (3.0 * count)
    mean = h_sum / total
    variance = (h_square / total - mean.square()).clamp_min(0.0)
    stats = FrameLatentStatistics(
        h_mean=mean.to(dtype=torch.float32),
        h_std=variance.sqrt().clamp_min(1.0e-6).to(dtype=torch.float32),
        v_rms=(v_square / (3.0 * total)).sqrt().clamp_min(1.0e-6).to(dtype=torch.float32),
        provenance={
            "scope": "complete_training_store",
            "training_clip_count": int(args.train_count),
            "data_hash": str(args.data_hash),
            "codec_checkpoint_sha256": str(args.codec_sha256),
            "vector_mean_subtraction": False,
            "vector_xyz_shared_scale": True,
            "sharded_fit": {"shards": int(args.shards), "atom_frame_count": int(total)},
        },
    )
    path = Path(args.stats_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats.state_dict(), path)
    atomic_write_json(path.with_suffix(".json"), {
        "statistics_hash": stats.hash,
        "file_sha256": sha256_file(path),
        "provenance": dict(stats.provenance),
    })
    print(json.dumps({
        "statistics": str(path),
        "statistics_hash": stats.hash,
        "sha256": sha256_file(path),
        "atom_frame_count": int(total),
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.set_defaults(func=prepare)
    p = sub.add_parser("configs")
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.set_defaults(func=write_configs)
    p = sub.add_parser("combine")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--stats-path", type=Path, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--train-count", type=int, required=True)
    p.add_argument("--data-hash", required=True)
    p.add_argument("--codec-sha256", required=True)
    p.set_defaults(func=combine)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
