"""Freeze one or two paired B epochs from measured pilot GPU time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molvid.runtime import atomic_write_json, sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-configs", type=Path, required=True)
    parser.add_argument("--short-pilot-manifest", type=Path, required=True)
    parser.add_argument("--long-pilot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-factor", type=float, default=1.15)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.safety_factor < 1.0:
        raise ValueError("budget safety factor must be at least one")
    original_frozen_sha = sha256_file(args.frozen_configs)
    frozen = json.loads(args.frozen_configs.read_text(encoding="utf-8"))
    manifests = {
        "M-short": json.loads(args.short_pilot_manifest.read_text(encoding="utf-8")),
        "M-long": json.loads(args.long_pilot_manifest.read_text(encoding="utf-8")),
    }
    seconds_per_update: dict[str, float] = {}
    for arm, manifest in manifests.items():
        if (
            manifest.get("schema") != "molvid.learned_memory.train_manifest.v1"
            or manifest.get("status") != "completed"
            or manifest.get("arm") != arm
        ):
            raise ValueError(f"{arm} pilot manifest is incomplete or mismatched")
        updates = int(manifest["successful_updates"])
        if updates < 1:
            raise ValueError(f"{arm} pilot contains no successful updates")
        pilot_config = Path(frozen["arms"][arm]["pilot"]["config"])
        if (
            sha256_file(pilot_config) != frozen["arms"][arm]["pilot"]["config_sha256"]
            or manifest["config_sha256"] != sha256_file(pilot_config)
        ):
            raise ValueError(f"{arm} pilot did not use the frozen pilot config")
        seconds_per_update[arm] = (
            float(manifest["training_seconds_this_invocation"]) / updates
        )
    original_updates = [int(value) for value in frozen["updates_per_epoch"]]
    if len(original_updates) != 2:
        raise ValueError("budget finalizer expects the reviewed two-epoch candidate")
    projections = {}
    for epochs in (1, 2):
        updates = sum(original_updates[:epochs])
        projections[str(epochs)] = (
            sum(seconds_per_update.values())
            * updates
            * float(args.safety_factor)
            / 3600.0
        )
    cap = 32.0
    chosen_epochs = 2 if projections["2"] <= cap else 1
    if projections[str(chosen_epochs)] > cap:
        raise RuntimeError("even one paired epoch exceeds the32 GPU-hour formal cap")
    for arm in ("M-short", "M-long"):
        formal_path = Path(frozen["arms"][arm]["formal"]["config"])
        config = json.loads(formal_path.read_text(encoding="utf-8"))
        if config["training"]["max_updates"] is not None:
            raise ValueError("formal config unexpectedly has a max_updates pilot bound")
        config["training"]["effective_epochs"] = chosen_epochs
        atomic_write_json(formal_path, config)
        frozen["arms"][arm]["formal"]["config_sha256"] = sha256_file(formal_path)
    frozen["effective_epochs"] = chosen_epochs
    frozen["updates_per_epoch"] = original_updates[:chosen_epochs]
    frozen["formal_total_updates"] = sum(original_updates[:chosen_epochs])
    frozen["paired_epochs"] = {
        key: value for key, value in frozen["paired_epochs"].items()
        if int(key) < chosen_epochs
    }
    atomic_write_json(args.frozen_configs, frozen)
    decision = {
        "schema": "molvid.learned_memory.formal_budget_decision.v1",
        "original_frozen_configs_sha256": original_frozen_sha,
        "final_frozen_configs_sha256": sha256_file(args.frozen_configs),
        "pilot_manifests": {
            arm: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "successful_updates": manifests[arm]["successful_updates"],
                "gpu_hours": manifests[arm]["gpu_hours_this_invocation"],
                "seconds_per_update_including_setup": seconds_per_update[arm],
            }
            for arm, path in (
                ("M-short", args.short_pilot_manifest),
                ("M-long", args.long_pilot_manifest),
            )
        },
        "safety_factor": args.safety_factor,
        "formal_cap_gpu_hours": cap,
        "projected_paired_gpu_hours": projections,
        "chosen_effective_epochs": chosen_epochs,
        "chosen_updates_per_arm": sum(original_updates[:chosen_epochs]),
        "rule": "use two paired epochs only when conservative pilot projection is <=32 GPUh; otherwise freeze both to one before review/launch",
    }
    atomic_write_json(args.output, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
