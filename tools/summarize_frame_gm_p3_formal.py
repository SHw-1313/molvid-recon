"""Validate and summarize the paired formal P3 J0/J1 training runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from molvid.config import load_config
from molvid.runtime import atomic_write_json, sha256_file


ARMS = ("J0", "J1")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--resolved-experiment", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def validate_paired_configs(
    j0: Mapping[str, Any],
    j1: Mapping[str, Any],
    *,
    expected_step: int,
) -> None:
    """Require P3 arms to differ only by output root and sampled auxiliary."""
    if expected_step <= 0:
        raise ValueError("expected formal step must be positive")
    for section in ("data", "codec", "statistics", "model", "loss"):
        if j0.get(section) != j1.get(section):
            raise ValueError(f"J0/J1 {section} configs differ")
    training = []
    for arm, raw in (("J0", j0), ("J1", j1)):
        current = dict(raw.get("training", {}))
        output_root = current.pop("output_root", None)
        if not isinstance(output_root, str) or not output_root:
            raise ValueError(f"{arm} output root is missing")
        if current.get("deterministic") is not True:
            raise ValueError(f"{arm} formal training must be deterministic")
        stages = current.get("stages")
        if not isinstance(stages, list) or sum(
            int(stage.get("updates", -1))
            for stage in stages
            if isinstance(stage, Mapping)
        ) != expected_step:
            raise ValueError(f"{arm} formal update count differs")
        training.append(current)
    if training[0] != training[1]:
        raise ValueError("J0/J1 main training configs differ")

    j0_sampled = j0.get("sampled_auxiliary", {})
    if not isinstance(j0_sampled, Mapping) or j0_sampled.get("enabled", False) is not False:
        raise ValueError("J0 must have sampled auxiliary disabled")
    sampled = j1.get("sampled_auxiliary")
    if not isinstance(sampled, Mapping) or sampled.get("enabled") is not True:
        raise ValueError("J1 must have sampled auxiliary enabled")
    if (
        sampled.get("cadence") != 8
        or sampled.get("draws") != 2
        or sampled.get("euler_steps") != 4
        or sampled.get("activation_checkpoint") is not True
        or sampled.get("calibration_batches") != 8
        or sampled.get("energy_gradient_target_ratio") != 0.05
        or sampled.get("observed_bond_gradient_target_ratio") != 0.05
    ):
        raise ValueError("J1 sampled auxiliary differs from the frozen P3 contract")
    for name in ("energy_weight", "observed_bond_weight"):
        value = sampled.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"J1 {name} must be resolved numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"J1 {name} must be finite and non-negative")
    scales = sampled.get("feature_scales")
    if not isinstance(scales, Mapping) or set(scales) != {
        "residue_rmsf_A",
        "displacement_squared_A2",
        "internal_distance_increment_A",
        "internal_distance_increment_product_A2",
    }:
        raise ValueError("J1 feature scales are not resolved")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in scales.values()
    ):
        raise ValueError("J1 feature scales must be finite and positive")


def _validate_metrics(path: Path, *, arm: str, expected_step: int) -> dict[str, Any]:
    count = 0
    sampled_steps: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if row.get("step") != count or not _all_finite(row):
                raise ValueError(f"{arm} formal training metrics are incomplete or non-finite")
            active = row.get("sampled_branch_active")
            if not isinstance(active, bool):
                raise ValueError(f"{arm} formal metrics lack sampled cadence state")
            if active:
                sampled_steps.append(count)
    if count != expected_step:
        raise ValueError(f"{arm} formal metric row count differs")
    expected = [] if arm == "J0" else list(range(8, expected_step + 1, 8))
    if sampled_steps != expected:
        raise ValueError(f"{arm} formal sampled cadence differs")
    return {
        "row_count": count,
        "sampled_active_count": len(sampled_steps),
        "sampled_first_step": sampled_steps[0] if sampled_steps else None,
        "sampled_last_step": sampled_steps[-1] if sampled_steps else None,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = _arguments()
    if args.expected_step <= 0:
        raise ValueError("expected formal step must be positive")
    resolved = _read_json(args.resolved_experiment)
    manifest_path = args.formal_root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if resolved.get("schema") != "molvid.frame_gm.p3_resolved_experiment.v1":
        raise ValueError("unexpected P3 resolved-experiment schema")
    if manifest.get("schema") != "molvid.frame_gm.p3_formal_run.v1":
        raise ValueError("unexpected P3 formal-run schema")
    if manifest.get("status") != "COMPLETE" or manifest.get("sealed_test_opened") is not False:
        raise ValueError("P3 formal training is incomplete or opened sealed test data")
    if (
        manifest.get("candidate_commit") != resolved.get("candidate_commit")
        or manifest.get("parent", {}).get("sha256") != resolved.get("parent", {}).get("sha256")
        or manifest.get("successful_updates_per_arm") != args.expected_step
    ):
        raise ValueError("P3 formal identity differs from resolved experiment")
    arms_manifest = manifest.get("arms")
    configs = resolved.get("formal_training", {}).get("configs")
    if (
        not isinstance(arms_manifest, Mapping)
        or not isinstance(configs, Mapping)
        or set(arms_manifest) != set(ARMS)
        or set(configs) != set(ARMS)
    ):
        raise ValueError("P3 formal arm/config provenance is missing")

    raw_configs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        config_path = Path(str(configs[arm].get("path", "")))
        config_sha = sha256_file(config_path)
        if (
            config_sha != configs[arm].get("sha256")
            or config_sha != arms_manifest[arm].get("config_sha256")
            or str(config_path) != arms_manifest[arm].get("config")
        ):
            raise ValueError(f"P3 formal config provenance differs for {arm}")
        raw_configs[arm] = load_config(
            config_path, schema="molvid.frame_joint.train.v1"
        )
    validate_paired_configs(
        raw_configs["J0"], raw_configs["J1"], expected_step=args.expected_step
    )

    arm_results: dict[str, Any] = {}
    exposures: dict[str, Any] = {}
    samplers: dict[str, Any] = {}
    for arm in ARMS:
        run = args.formal_root / arm
        arm_manifest = arms_manifest[arm]
        if arm_manifest.get("status") != "COMPLETE":
            raise ValueError(f"P3 formal arm {arm} is incomplete")
        checkpoint = run / f"frame_joint_step_{args.expected_step:08d}.pt"
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != arm_manifest.get("final_checkpoint_sha256"):
            raise ValueError(f"P3 formal checkpoint provenance differs for {arm}")
        exposure_path = run / "exposure.json"
        sampler_path = run / "sampler_manifest.json"
        resolved_path = run / "resolved_config.json"
        target_path = run / "target_encoder_provenance.json"
        warm_path = run / "warm_start_report.json"
        exposures[arm] = _read_json(exposure_path)
        samplers[arm] = _read_json(sampler_path)
        runtime = _read_json(resolved_path)
        target = _read_json(target_path)
        warm = _read_json(warm_path)
        if exposures[arm].get("successful_updates") != args.expected_step:
            raise ValueError(f"P3 formal exposure is incomplete for {arm}")
        if samplers[arm].get("test_opened") is not False:
            raise ValueError(f"P3 formal sampler opened sealed test for {arm}")
        if target.get("code_commit") != resolved.get("candidate_commit"):
            raise ValueError(f"P3 formal code provenance differs for {arm}")
        if warm.get("source_sha256") != resolved.get("parent", {}).get("sha256"):
            raise ValueError(f"P3 formal parent differs for {arm}")
        actual_config = {key: value for key, value in runtime.items() if key != "resolved"}
        if actual_config != raw_configs[arm]:
            raise ValueError(f"P3 formal runtime config differs for {arm}")
        child = runtime.get("resolved", {})
        if (
            child.get("total_updates") != args.expected_step
            or child.get("continuation_parent", {}).get("sha256")
            != resolved.get("parent", {}).get("sha256")
            or child.get("data_hash") != resolved.get("data", {}).get("derived_data_hash")
            or (arm == "J1") != isinstance(child.get("sampled_auxiliary_contract"), Mapping)
        ):
            raise ValueError(f"P3 formal resolved config differs for {arm}")
        metrics = _validate_metrics(
            run / "train_metrics.jsonl", arm=arm, expected_step=args.expected_step
        )
        arm_results[arm] = {
            "config": str(Path(str(configs[arm]["path"])).resolve()),
            "config_sha256": configs[arm]["sha256"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "training_metrics": metrics,
            "exposure_sha256": sha256_file(exposure_path),
            "sampler_manifest_sha256": sha256_file(sampler_path),
            "resolved_config_sha256": sha256_file(resolved_path),
            "target_encoder_provenance_sha256": sha256_file(target_path),
            "warm_start_report_sha256": sha256_file(warm_path),
        }
    if exposures["J0"] != exposures["J1"] or samplers["J0"] != samplers["J1"]:
        raise ValueError("P3 formal sampler schedule or exposure differs")
    output = {
        "schema": "molvid.frame_gm.p3_formal_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_commit": resolved["candidate_commit"],
        "parent": resolved["parent"],
        "successful_updates_per_arm": args.expected_step,
        "arms": arm_results,
        "paired_main_stream": {
            "exposure_exact": True,
            "sampler_manifest_exact": True,
        },
        "resolved_experiment": str(args.resolved_experiment.resolve()),
        "resolved_experiment_sha256": sha256_file(args.resolved_experiment),
        "formal_run_manifest": str(manifest_path.resolve()),
        "formal_run_manifest_sha256": sha256_file(manifest_path),
        "test_opened": False,
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
