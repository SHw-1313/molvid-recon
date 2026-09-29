"""Validate and freeze the paired released-A1 static-decoder arm configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from molvid.cli.train_frame_joint import _validate_frame_joint_config
from molvid.config import load_config
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()


def _dynamic_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(raw))
    result.pop("static_decoder_auxiliary", None)
    result.pop("experiment_warm_start", None)
    training = result["training"]
    training["output_root"] = "<arm-output>"
    for stage in training["stages"]:
        stage["updates"] = "<matched-short-updates>"
    return result


def _static_shared_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    static = dict(raw["static_decoder_auxiliary"])
    return {
        name: static[name]
        for name in (
            "cadence", "seed", "max_atoms_per_update", "max_samples_per_update",
            "min_source_atoms", "max_source_atoms", "exposure_target",
            "perturbation", "loss", "gamma_calibration",
        )
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if sha256_file(args.parent) != args.parent_sha256:
        raise ValueError("paired experiment parent SHA-256 differs")
    paths = {
        "baseline": args.baseline,
        "internal_md": args.internal,
        "external_pdb": args.external,
    }
    configs = {
        name: load_config(path, schema="molvid.frame_joint.train.v1")
        for name, path in paths.items()
    }
    for raw in configs.values():
        _validate_frame_joint_config(raw)
    baseline_static = configs["baseline"].get("static_decoder_auxiliary", {})
    if baseline_static not in ({}, {"enabled": False}):
        raise ValueError("paired baseline must have static auxiliary disabled")
    if configs["internal_md"]["static_decoder_auxiliary"]["source"] != "internal_md":
        raise ValueError("internal arm source differs")
    if configs["external_pdb"]["static_decoder_auxiliary"]["source"] != "external_static":
        raise ValueError("external arm source differs")
    experiment_parents = {
        name: raw.get("experiment_warm_start") for name, raw in configs.items()
    }
    if any(not isinstance(value, Mapping) for value in experiment_parents.values()):
        raise ValueError("every paired arm must freeze its warm-start parent")
    if len({canonical_hash(value) for value in experiment_parents.values()}) != 1:
        raise ValueError("paired arms differ in their warm-start parent")
    frozen_parent = next(iter(experiment_parents.values()))
    if Path(frozen_parent["path"]).resolve() != args.parent.resolve():
        raise ValueError("config warm-start path differs from the paired parent")
    if frozen_parent["sha256"] != args.parent_sha256:
        raise ValueError("config warm-start SHA-256 differs from the paired parent")
    if frozen_parent["code_commit"] != _commit():
        raise ValueError("config warm-start code identity differs from the current commit")

    dynamic_hashes = {
        name: canonical_hash(_dynamic_contract(raw)) for name, raw in configs.items()
    }
    if len(set(dynamic_hashes.values())) != 1:
        raise ValueError("paired arms differ outside output/static sections")
    reference = load_config(args.reference_config, schema="molvid.frame_joint.train.v1")
    if canonical_hash(_dynamic_contract(reference)) != next(iter(dynamic_hashes.values())):
        raise ValueError("paired dynamic recipe differs from released A1")
    totals = {
        name: sum(int(stage["updates"]) for stage in raw["training"]["stages"])
        for name, raw in configs.items()
    }
    if len(set(totals.values())) != 1 or next(iter(totals.values())) not in {500, 1000}:
        raise ValueError("paired arms require a uniform frozen 500/1000 update budget")
    target_updates = next(iter(totals.values())) // 4
    for name in ("internal_md", "external_pdb"):
        target = configs[name]["static_decoder_auxiliary"]["exposure_target"]
        if int(target["static_updates"]) != target_updates:
            raise ValueError("static exposure target differs from the frozen dynamic budget")
    shared = {
        name: canonical_hash(_static_shared_contract(configs[name]))
        for name in ("internal_md", "external_pdb")
    }
    if len(set(shared.values())) != 1:
        raise ValueError("internal/external static objective or RNG budget differs")

    payload = {
        "schema": "molvid.static_decoder.paired_experiment.v1",
        "code_commit": _commit(),
        "reference_A1_config": str(args.reference_config.resolve()),
        "reference_A1_config_sha256": sha256_file(args.reference_config),
        "parent": str(args.parent.resolve()),
        "parent_sha256": args.parent_sha256,
        "parent_mode": "model_only_warm_start_with_optimizer_scheduler_cursor_rng_reset",
        "dynamic_contract_hash": next(iter(dynamic_hashes.values())),
        "dynamic_updates_per_arm": next(iter(totals.values())),
        "arms": {
            name: {
                "config": str(path.resolve()),
                "config_sha256": sha256_file(path),
                "output_root": configs[name]["training"]["output_root"],
                "static_source": (
                    None
                    if name == "baseline"
                    else configs[name]["static_decoder_auxiliary"]["source"]
                ),
            }
            for name, path in paths.items()
        },
        "static_shared_contract_hash": next(iter(shared.values())),
        "same_dynamic_order_and_rng_seed": True,
        "same_dynamic_lr_progress": True,
        "same_static_update_count": True,
        "same_static_sample_atom_and_supervised_atom_target": True,
        "launch_contract": {
            name: {
                "argv": [
                    "python", "-m", "molvid.cli.train_frame_joint",
                    "--config", str(path.resolve()),
                    "--warm-start", str(args.parent.resolve()),
                    "--warm-start-sha256", args.parent_sha256,
                    "--checkpoint-every", "500",
                ],
                "code_commit": _commit(),
                "config_sha256": sha256_file(path),
                "output_root": configs[name]["training"]["output_root"],
                "resume_requires_checkpoint_sha256": True,
            }
            for name, path in paths.items()
        },
        "sealed_test_trajectories_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
