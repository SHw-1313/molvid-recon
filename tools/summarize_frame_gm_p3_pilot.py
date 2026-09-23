"""Build fail-closed, hash-bound evidence for the paired P3 J0/J1 pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from molvid.runtime import atomic_write_json, sha256_file


ARMS = ("J0", "J1")
EXPECTED_SAMPLED_STEPS = tuple(range(8, 65, 8))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--target-mutation", type=Path, required=True)
    parser.add_argument("--ddp-check", type=Path, required=True)
    parser.add_argument("--continuation-check", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--parent-arm", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--targeted-tests-passed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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


def _tree_max_abs(left: Any, right: Any) -> float:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return float("inf")
        left = left.detach().cpu()
        right = right.detach().cpu()
        if left.dtype.is_floating_point:
            return float((left - right).abs().max()) if left.numel() else 0.0
        return 0.0 if torch.equal(left, right) else float("inf")
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if (
            not isinstance(left, np.ndarray)
            or not isinstance(right, np.ndarray)
            or left.shape != right.shape
            or left.dtype != right.dtype
        ):
            return float("inf")
        if not left.size:
            return 0.0
        if np.issubdtype(left.dtype, np.floating):
            return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        return 0.0 if np.array_equal(left, right) else float("inf")
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max((_tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_tree_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0 if left == right else float("inf")


def _resume_equivalence(primary: Path, reference: Path) -> dict[str, Any]:
    first = torch.load(primary, map_location="cpu", weights_only=False)
    second = torch.load(reference, map_location="cpu", weights_only=False)
    fields = {
        "model_state_max_abs": _tree_max_abs(first["model_state"], second["model_state"]),
        "optimizer_state_max_abs": _tree_max_abs(
            first["optimizer_state"], second["optimizer_state"]
        ),
        "scheduler_state_max_abs": _tree_max_abs(
            first.get("scheduler_state"), second.get("scheduler_state")
        ),
        "cursor_max_abs": _tree_max_abs(first["cursor"], second["cursor"]),
        "contracts_max_abs": _tree_max_abs(first["contracts"], second["contracts"]),
        "extra_state_max_abs": _tree_max_abs(
            first["extra_state"], second["extra_state"]
        ),
        "step_equal": first["step"] == second["step"] == 64,
    }
    fields["exact"] = all(
        value if isinstance(value, bool) else value == 0.0
        for value in fields.values()
    )
    return fields


def _calibration_summary(root: Path) -> dict[str, Any]:
    feature_path = root / "sampled_feature_scale_calibration.json"
    gradient_path = root / "sampled_auxiliary_gradient_calibration.json"
    feature = _read_json(feature_path)
    gradient = _read_json(gradient_path)
    if (
        feature.get("batch_count") != 8
        or gradient.get("batch_count") != 8
        or feature.get("split") != "train"
        or gradient.get("split") != "train"
        or feature.get("test_opened") is not False
        or gradient.get("test_opened") is not False
    ):
        raise ValueError("P3 calibration did not use exactly eight sealed train batches")
    per_batch = gradient.get("gradient_norms_per_batch", {})
    if set(per_batch) != {"flow", "energy", "observed_bond"} or any(
        len(values) != 8 or any(not math.isfinite(float(value)) for value in values)
        for values in per_batch.values()
    ):
        raise ValueError("P3 calibration lacks eight finite raw gradient norms")
    achieved = gradient.get("achieved_mean_gradient_ratios", {})
    if any(
        not math.isclose(float(achieved.get(name, math.nan)), 0.05, rel_tol=1e-6)
        for name in ("energy_to_flow", "observed_bond_to_flow")
    ):
        raise ValueError("P3 resolved auxiliary weights do not achieve 0.05/0.05")
    return {
        "feature_scales": feature["scales"],
        "resolved_weights": gradient["resolved_weights"],
        "achieved_mean_gradient_ratios": achieved,
        "feature_artifact": str(feature_path.resolve()),
        "feature_artifact_sha256": sha256_file(feature_path),
        "gradient_artifact": str(gradient_path.resolve()),
        "gradient_artifact_sha256": sha256_file(gradient_path),
    }


def main() -> int:
    args = _arguments()
    arms: dict[str, Any] = {}
    metrics_by_arm: dict[str, list[dict[str, Any]]] = {}
    exposures: dict[str, Any] = {}
    samplers: dict[str, Any] = {}
    for arm in ARMS:
        run = args.root / arm
        metrics_path = run / "train_metrics.jsonl"
        metrics = _read_jsonl(metrics_path)
        if [row.get("step") for row in metrics] != list(range(1, 65)):
            raise ValueError(f"{arm} pilot does not contain exactly steps 1..64")
        if not _all_finite(metrics):
            raise ValueError(f"{arm} pilot contains a non-finite metric")
        active = tuple(
            int(row["step"]) for row in metrics if row["sampled_branch_active"]
        )
        expected_active = () if arm == "J0" else EXPECTED_SAMPLED_STEPS
        if active != expected_active:
            raise ValueError(f"{arm} sampled cadence differs: {active}")
        if arm == "J1":
            for row in metrics:
                if row["sampled_branch_active"]:
                    diagnostics = row.get("sampled_diagnostics", {})
                    if diagnostics.get("draws") != 2 or diagnostics.get("euler_steps") != 4:
                        raise ValueError("J1 sampled branch differs from K=2/Euler4")
                elif (
                    row["weighted_sampled_energy"] != 0.0
                    or row["weighted_sampled_observed_bond"] != 0.0
                ):
                    raise ValueError("J1 auxiliary contributed outside its cadence")
        checkpoint = run / "frame_joint_step_00000064.pt"
        provenance_path = run / "target_encoder_provenance.json"
        provenance = _read_json(provenance_path)
        if provenance.get("code_commit") != args.candidate_commit:
            raise ValueError(f"{arm} pilot provenance differs from candidate commit")
        warm_path = run / "warm_start_report.json"
        warm = _read_json(warm_path)
        if warm.get("source_sha256") != args.parent_sha256:
            raise ValueError(f"{arm} warm-start parent differs")
        resolved_path = run / "resolved_config.json"
        resolved = _read_json(resolved_path)
        sampled_contract = resolved.get("resolved", {}).get(
            "sampled_auxiliary_contract"
        )
        if (arm == "J1") != isinstance(sampled_contract, Mapping):
            raise ValueError(f"{arm} resolved sampled contract differs")
        exposures[arm] = _read_json(run / "exposure.json")
        samplers[arm] = _read_json(run / "sampler_manifest.json")
        metrics_by_arm[arm] = metrics
        arms[arm] = {
            "successful_updates": len(metrics),
            "sampled_active_steps": list(active),
            "mean_loss": fmean(float(row["loss"]) for row in metrics),
            "mean_grad_norm": fmean(float(row["grad_norm"]) for row in metrics),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metrics_sha256": sha256_file(metrics_path),
            "exposure_sha256": sha256_file(run / "exposure.json"),
            "sampler_manifest_sha256": sha256_file(run / "sampler_manifest.json"),
            "resolved_config_sha256": sha256_file(resolved_path),
            "warm_start_report_sha256": sha256_file(warm_path),
            "target_encoder_provenance_sha256": sha256_file(provenance_path),
            "sampled_contract": sampled_contract,
        }
    if exposures["J0"] != exposures["J1"] or samplers["J0"] != samplers["J1"]:
        raise ValueError("J0/J1 exposure or sampler manifest differs")
    for index in range(64):
        left = metrics_by_arm["J0"][index]
        right = metrics_by_arm["J1"][index]
        if (
            left["sample_ids"] != right["sample_ids"]
            or left["history_frames"] != right["history_frames"]
            or left["flow_time_mean"] != right["flow_time_mean"]
            or left["time_bucket_id"] != right["time_bucket_id"]
        ):
            raise ValueError(f"J0/J1 main data or flow RNG differs at step {index + 1}")

    primary_checkpoint = args.root / "J1" / "frame_joint_step_00000064.pt"
    reference_checkpoint = args.reference_root / "frame_joint_step_00000064.pt"
    resume = _resume_equivalence(primary_checkpoint, reference_checkpoint)
    reference_metrics = args.reference_root / "train_metrics.jsonl"
    if sha256_file(args.root / "J1" / "train_metrics.jsonl") != sha256_file(
        reference_metrics
    ):
        raise ValueError("interrupted and uninterrupted J1 metric streams differ")
    if not resume["exact"]:
        raise ValueError("interrupted and uninterrupted J1 checkpoints differ")

    calibration = _calibration_summary(args.calibration_root)
    profile = _read_json(args.profile)
    mutation = _read_json(args.target_mutation)
    ddp = _read_json(args.ddp_check)
    continuation = _read_json(args.continuation_check)
    if (
        profile.get("metrics", {}).get("sampled_branch_active") is not True
        or profile.get("teacher_state_unchanged") is not True
        or mutation.get("passed") is not True
        or ddp.get("passed") is not True
        or ddp.get("sampled_branch_enabled") is not True
        or continuation.get("passed") is not True
        or continuation.get("sampled_auxiliary_enabled") is not True
    ):
        raise ValueError("one or more P3 CUDA evidence artifacts failed")
    output = {
        "schema": "molvid.frame_gm.p3_pilot_summary.v1",
        "candidate_commit": args.candidate_commit,
        "parent": {"arm": args.parent_arm, "sha256": args.parent_sha256},
        "arms": arms,
        "paired_main_stream": {
            "exposure_exact": True,
            "sampler_manifest_exact": True,
            "sample_ids_history_flow_time_and_bucket_exact_for_64_steps": True,
        },
        "calibration": calibration,
        "strict_resume": {
            **resume,
            "reference_checkpoint": str(reference_checkpoint.resolve()),
            "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
            "metrics_stream_exact": True,
        },
        "cuda_evidence": {
            "largest_sample_profile": {
                "path": str(args.profile.resolve()),
                "sha256": sha256_file(args.profile),
                "summary": profile,
            },
            "target_mutation": {
                "path": str(args.target_mutation.resolve()),
                "sha256": sha256_file(args.target_mutation),
                "summary": mutation,
            },
            "ddp_resume": {
                "path": str(args.ddp_check.resolve()),
                "sha256": sha256_file(args.ddp_check),
                "summary": ddp,
            },
            "continuation": {
                "path": str(args.continuation_check.resolve()),
                "sha256": sha256_file(args.continuation_check),
                "summary": continuation,
            },
        },
        "targeted_tests": {"passed": args.targeted_tests_passed, "failed": 0},
        "test_opened": False,
        "formal_training_started": False,
    }
    atomic_write_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "arms": list(arms),
        "resume_exact": resume["exact"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
