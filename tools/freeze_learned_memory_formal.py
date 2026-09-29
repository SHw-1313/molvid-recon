"""Freeze paired M-short/M-long configs from the evaluated A1 release."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from molvid.data.learned_memory_formal import RawMemoryDataset
from molvid.data.sampling import TrajectoryCappedBatchSampler
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file
from molvid.training.learned_memory_schedule import (
    learned_memory_execution_manifest,
    learned_memory_execution_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-release", type=Path, required=True)
    parser.add_argument("--capacity-release", type=Path, required=True)
    parser.add_argument("--a1-config", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-updates", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(args.recipe_release.read_text(encoding="utf-8"))
    capacity = json.loads(args.capacity_release.read_text(encoding="utf-8"))
    a1 = json.loads(args.a1_config.read_text(encoding="utf-8"))
    if recipe.get("selected_arm") != "A1" or capacity.get("selected_arm") != "A1":
        raise ValueError("release does not select evaluated A1")
    if sha256_file(args.a1_config) != recipe["configuration"]["sha256"]:
        raise ValueError("A1 config identity differs from release")
    if sha256_file(args.codec) != recipe["teacher_sha256"]:
        raise ValueError("teacher identity differs from release")
    if sha256_file(args.statistics) != recipe["statistics_sha256"]:
        raise ValueError("statistics identity differs from release")
    parent = Path(recipe["A1_checkpoint"]["path"])
    if sha256_file(parent) != recipe["A1_checkpoint"]["sha256"]:
        raise ValueError("A1 checkpoint identity differs from release")
    dataset = RawMemoryDataset(
        manifest_root=args.manifest_root,
        raw_root=args.raw_root,
        split="train",
    )
    training = {
        "seed": int(a1["training"]["seed"]),
        "device": "cuda",
        "deterministic": True,
        "precision": "bf16",
        "max_atom_frames_per_gpu": int(a1["training"]["max_atom_frames_per_gpu"]),
        "clips_per_trajectory": int(a1["training"]["clips_per_trajectory"]),
        "effective_epochs": 2,
        "max_updates": None,
        "learning_rates": dict(a1["training"]["stages"][0]["learning_rates"]),
        "weight_decay": float(a1["training"]["weight_decay"]),
        "grad_clip": float(a1["training"]["grad_clip"]),
        "gradient_diagnostics_every": int(a1["training"]["gradient_diagnostics_every"]),
        "output_root": "",
    }
    base = {
        "schema": "molvid.learned_memory.train.v1",
        "arm": "",
        "data": {
            "manifest_root": str(args.manifest_root.resolve()),
            "raw_root": str(args.raw_root.resolve()),
            "data_hash": dataset.identity["data_hash"],
        },
        "codec": {
            "checkpoint": str(args.codec.resolve()),
            "sha256": recipe["teacher_sha256"],
        },
        "statistics": {
            "path": str(args.statistics.resolve()),
            "sha256": recipe["statistics_sha256"],
            "statistics_hash": a1["statistics"]["statistics_hash"],
            "normalization_data_hash": a1["data"]["normalization_source_data_hash"],
        },
        "model": {
            "history_backend": "learned_memory",
            "analysis_mode": "physical",
            "depth": 4,
            "heads": 8,
            "scalar_width": 256,
            "vector_width": 128,
            "geometry_enabled": True,
            "motion_enabled": False,
        },
        "parent": {
            "path": str(parent),
            "sha256": recipe["A1_checkpoint"]["sha256"],
            "mode": "model_only_warm_start_retire_legacy_history_and_M",
            "reset": ["optimizer", "scheduler", "cursor", "rng"],
        },
        "loss": dict(a1["loss"]),
        "sampled_auxiliary": dict(a1["sampled_auxiliary"]),
        "training": training,
        "release": {
            "recipe_release": str(args.recipe_release.resolve()),
            "recipe_release_sha256": sha256_file(args.recipe_release),
            "capacity_release": str(args.capacity_release.resolve()),
            "capacity_release_sha256": sha256_file(args.capacity_release),
            "A1_configuration_sha256": sha256_file(args.a1_config),
            "reviewed_A_source_commit": recipe["reviewed_source_commit"],
        },
        "budget": {
            "lane_total_gpu_hours": 48.0,
            "formal_gpu_hours": 32.0,
            "evaluation_gpu_hours": 8.0,
            "pilot_gpu_hours": 8.0,
            "formal_fallback": "both arms one effective epoch before launch if paired profile exceeds32GPUh",
        },
    }
    sampler = TrajectoryCappedBatchSampler(
        dataset,
        max_tokens=training["max_atom_frames_per_gpu"],
        clips_per_trajectory=training["clips_per_trajectory"],
        seed=training["seed"],
        shuffle=True,
        num_replicas=1,
        rank=0,
        drop_last=False,
        oversize_policy="partition",
    )
    paired = {}
    updates = []
    for epoch in range(training["effective_epochs"]):
        sampler.set_epoch(epoch)
        batches = list(iter(sampler))
        updates.append(len(batches))
        manifests = {}
        for arm in ("M-short", "M-long"):
            manifests[arm] = learned_memory_execution_manifest(
                learned_memory_execution_schedule(
                    batches, sampler.specs, arm=arm,
                    seed=training["seed"], epoch=epoch, cadence=8,
                )
            )
        if manifests["M-short"]["paired_schedule_sha256"] != manifests["M-long"]["paired_schedule_sha256"]:
            raise RuntimeError("short/long paired schedules differ")
        paired[str(epoch)] = {
            "paired_schedule_sha256": manifests["M-short"]["paired_schedule_sha256"],
            "short_execution_sha256": manifests["M-short"]["sha256"],
            "long_execution_sha256": manifests["M-long"]["sha256"],
            "updates": len(batches),
        }
    records = {}
    for arm in ("M-short", "M-long"):
        records[arm] = {}
        for phase, maximum in (("pilot", args.pilot_updates), ("formal", None)):
            config = copy.deepcopy(base)
            config["arm"] = arm
            config["training"]["max_updates"] = maximum
            config["training"]["output_root"] = str(
                (args.output / phase / arm).resolve()
            )
            path = args.output / f"{phase}_{arm}.json"
            atomic_write_json(path, config)
            records[arm][phase] = {
                "config": str(path.resolve()),
                "config_sha256": sha256_file(path),
                "output_root": config["training"]["output_root"],
            }
    frozen = {
        "schema": "molvid.learned_memory.frozen_configs.v1",
        "recipe_release_sha256": sha256_file(args.recipe_release),
        "capacity_release_sha256": sha256_file(args.capacity_release),
        "A1_configuration_sha256": sha256_file(args.a1_config),
        "parent_sha256": recipe["A1_checkpoint"]["sha256"],
        "data": dataset.identity,
        "effective_epochs": training["effective_epochs"],
        "updates_per_epoch": updates,
        "formal_total_updates": sum(updates),
        "paired_epochs": paired,
        "arms": records,
        "test_opened": False,
    }
    atomic_write_json(args.output / "frozen_configs.json", frozen)
    print(json.dumps({
        "output": str(args.output),
        "data_hash": dataset.identity["data_hash"],
        "updates_per_epoch": updates,
        "formal_total_updates": sum(updates),
        "paired_hash": canonical_hash(paired),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
