"""Freeze executable, matched C4/C8 pilot and formal runtime configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from molvid.cli.train_frame_joint import (
    _open_data,
    _sampler,
    _validate_frame_joint_config,
)
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file
from tools.gm_capacity_common import (
    CAPACITY_DEPTHS,
    GEOMETRY_BLOCKS,
    PARENT_BLOCK_MAPPING,
    load_capacity_release,
)


SEED = 20260928
PILOT_UPDATES = 128
EVALUATION_FAMILIES = {
    "ordinary": {
        "store": Path(
            "/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views/clip_store/valid"
        ),
        "manifest": Path(
            "/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views/paired_view_manifest.json"
        ),
    },
    "fixed_history": {
        "store": Path(
            "/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views/clip_store/valid"
        ),
        "manifest": Path(
            "/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views/fixed_history_manifest.json"
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = load_capacity_release(args.release)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("schema") != "molvid.capacity.profile.v1" or profile.get("passed") is not True:
        raise ValueError("capacity configs require a completed profile")
    if profile.get("parent_checkpoint_sha256") != bundle.checkpoint_sha256:
        raise ValueError("capacity profile and release parent differ")
    max_atom_frames = int(profile["max_atom_frames_per_gpu"])
    effective_epochs = float(profile["chosen_effective_epochs"])
    formal_updates = int(profile["formal_updates_per_arm"])
    if effective_epochs <= 0 or formal_updates < 1:
        raise ValueError("capacity profile selected no formal exposure")

    raw = copy.deepcopy(bundle.configuration)
    raw["training"].update({
        "seed": SEED,
        "deterministic": True,
        "max_atom_frames_per_gpu": max_atom_frames,
        "gradient_diagnostics_every": 500,
        "checkpoint_flow_field": True,
    })
    train, _, data_hash, _, relationship, close_data = _open_data(raw["data"])
    try:
        sampler = _sampler(train, raw["training"], seed=SEED, rank=0, world=1)
        updates_per_epoch = []
        schedules = []
        epoch = 0
        covered_updates = 0
        while covered_updates < formal_updates:
            sampler.set_epoch(epoch)
            updates_per_epoch.append(len(sampler))
            schedules.append(canonical_hash(sampler.global_batches))
            covered_updates += len(sampler)
            epoch += 1
    finally:
        close_data()
    if formal_updates > sum(updates_per_epoch):
        raise RuntimeError("profile formal exposure exceeds the inspected sampler schedules")
    evaluation_recipe = {
        "schema": "molvid.capacity.evaluation_recipe.v1",
        "physical": copy.deepcopy(raw["sampled_auxiliary"]["physical"]),
        "source_capacity_release_sha256": sha256_file(bundle.release_path),
    }
    evaluation_recipe_path = args.output / "evaluation_recipe.json"
    atomic_write_json(evaluation_recipe_path, evaluation_recipe)
    evaluation_families = {}
    for name, paths in EVALUATION_FAMILIES.items():
        evaluation_families[name] = {
            "store": str(paths["store"]),
            "store_index_sha256": sha256_file(paths["store"] / "index.txt"),
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
        }
    manifest: dict[str, object] = {
        "schema": "molvid.capacity.frozen_configs.v1",
        "capacity_release": str(bundle.release_path),
        "capacity_release_sha256": sha256_file(bundle.release_path),
        "recipe_release": str(bundle.recipe_release_path),
        "recipe_release_sha256": sha256_file(bundle.recipe_release_path),
        "parent_path": str(bundle.checkpoint_path),
        "parent_sha256": bundle.checkpoint_sha256,
        "parent_formal_updates": int(bundle.release["formal_updates"]),
        "parent_effective_epochs": int(bundle.release["effective_epochs"]),
        "profile": str(args.profile.resolve()),
        "profile_sha256": sha256_file(args.profile),
        "data_hash": data_hash,
        "data_relationship": relationship,
        "max_atom_frames_per_gpu": max_atom_frames,
        "effective_epochs": effective_epochs,
        "updates_per_epoch": updates_per_epoch,
        "global_batch_schedule_hashes": schedules,
        "formal_total_updates": formal_updates,
        "budget_adjustment": copy.deepcopy(profile.get("budget_adjustment")),
        "pilot_updates": PILOT_UPDATES,
        "seed": SEED,
        "shared_loss": copy.deepcopy(raw["loss"]),
        "shared_sampled_auxiliary": copy.deepcopy(raw["sampled_auxiliary"]),
        "evaluation_recipe": str(evaluation_recipe_path.resolve()),
        "evaluation_recipe_sha256": sha256_file(evaluation_recipe_path),
        "evaluation_families": evaluation_families,
        "arms": {},
        "test_opened": False,
    }
    for depth in CAPACITY_DEPTHS:
        arm = f"C{depth}"
        config = copy.deepcopy(raw)
        config["model"].update({
            "depth": depth,
            "geometry_enabled": True,
            "motion_enabled": True,
            "geometry_block_indices": list(GEOMETRY_BLOCKS[depth]),
        })
        config["training"]["warm_start_block_mapping"] = list(
            PARENT_BLOCK_MAPPING[depth]
        )
        records = {}
        for phase, updates in (("pilot", PILOT_UPDATES), ("formal", formal_updates)):
            current = copy.deepcopy(config)
            current["training"]["stages"] = [{
                **copy.deepcopy(config["training"]["stages"][0]),
                "name": "continuation",
                "updates": updates,
            }]
            current["training"]["output_root"] = str(
                (args.output / phase / arm).resolve()
            )
            if phase == "pilot":
                current["training"]["gradient_diagnostics_every"] = 32
            _validate_frame_joint_config(current)
            path = args.output / f"{phase}_{arm}.json"
            atomic_write_json(path, current)
            records[phase] = {
                "config": str(path.resolve()),
                "config_sha256": sha256_file(path),
                "output_root": current["training"]["output_root"],
                "updates": updates,
            }
        manifest["arms"][arm] = {
            "depth": depth,
            "geometry_block_indices_one_based": list(GEOMETRY_BLOCKS[depth]),
            "parent_block_mapping_one_based": list(PARENT_BLOCK_MAPPING[depth]),
            "initialization": (
                "exact A1 model-only warm start"
                if depth == 4
                else "parent blocks 1/2/3/4 -> 2/4/6/8; odd blocks identity; new G at 2/6 zero contribution"
            ),
            **records,
        }
    atomic_write_json(args.output / "frozen_configs.json", manifest)
    print(json.dumps({
        "frozen_configs": str(args.output / "frozen_configs.json"),
        "formal_total_updates": formal_updates,
        "effective_epochs": effective_epochs,
        "config_hashes": {
            arm: {
                phase: manifest["arms"][arm][phase]["config_sha256"]
                for phase in ("pilot", "formal")
            }
            for arm in manifest["arms"]
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
