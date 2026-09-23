"""Compare formal P3 J1 against J0 and gate the optional bond fork."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from molvid.runtime import atomic_write_json, sha256_file
from tools.select_frame_gm_p2_parent import load_arm_values


METRICS = (
    "bond_rmse_A",
    "residue_rmsf_mae_A",
    "msd_curve_mae_A2",
)
RULE_VERSION = "frame_gm_p3_bond_fork_system_paired_bootstrap.v1"
FROZEN_RULE = {
    "version": RULE_VERSION,
    "comparison": "J1 source-sampled distribution minus J0 standard continuation",
    "bootstrap_unit": "system",
    "confidence_interval": "paired two-sided percentile 95%",
    "trigger": (
        "bond RMSE upper CI < 0 and residue RMSF MAE lower CI > 0 "
        "and fixed-history MSD curve MAE lower CI > 0"
    ),
    "action_when_true": (
        "start one symmetric 0.25-effective-epoch J1 bond keep/release fork"
    ),
    "action_when_false": "do not start a bond fork",
    "posthoc_metric_changes_allowed": False,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    return parser.parse_args()


def _ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size < max(100, int(0.95 * values.size)):
        raise ValueError("insufficient finite system-bootstrap replicates")
    return [float(value) for value in np.percentile(finite, [2.5, 97.5])]


def compare_from_system_values(
    j0: Mapping[str, Any],
    j1: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Apply the frozen optional-fork rule to paired per-system values."""
    if bootstrap_samples < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    systems = list(j0.get("systems", ()))
    if len(systems) != 8 or list(j1.get("systems", ())) != systems:
        raise ValueError("J0/J1 must contain the same ordered set of eight systems")
    arrays: dict[str, dict[str, np.ndarray]] = {"J0": {}, "J1": {}}
    for arm, values in (("J0", j0), ("J1", j1)):
        for metric in METRICS:
            array = np.asarray(values.get(metric), dtype=np.float64)
            if array.shape != (len(systems),) or not np.isfinite(array).all():
                raise ValueError(f"invalid {metric} values for {arm}")
            arrays[arm][metric] = array
        target = np.asarray(values.get("rmsf_target_A"), dtype=np.float64)
        if target.shape != (len(systems),) or not np.isfinite(target).all():
            raise ValueError(f"invalid rmsf_target_A values for {arm}")
        arrays[arm]["rmsf_target_A"] = target
    if not np.allclose(
        arrays["J0"]["rmsf_target_A"],
        arrays["J1"]["rmsf_target_A"],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("J0/J1 RMSF targets are not paired")

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(systems), size=(bootstrap_samples, len(systems)))
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        per_system_delta = arrays["J1"][metric] - arrays["J0"][metric]
        comparisons[metric] = {
            "J0_mean": float(arrays["J0"][metric].mean()),
            "J1_mean": float(arrays["J1"][metric].mean()),
            "J1_minus_J0": float(per_system_delta.mean()),
            "paired_bootstrap_95pct_ci": _ci(per_system_delta[indices].mean(axis=1)),
            "per_system_J1_minus_J0": {
                system: float(value)
                for system, value in zip(systems, per_system_delta, strict=True)
            },
        }
    bond_improved = comparisons["bond_rmse_A"]["paired_bootstrap_95pct_ci"][1] < 0.0
    rmsf_worsened = (
        comparisons["residue_rmsf_mae_A"]["paired_bootstrap_95pct_ci"][0] > 0.0
    )
    time_worsened = (
        comparisons["msd_curve_mae_A2"]["paired_bootstrap_95pct_ci"][0] > 0.0
    )
    trigger = bool(bond_improved and rmsf_worsened and time_worsened)
    return {
        "systems": systems,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "comparisons": comparisons,
        "trigger_components": {
            "bond_geometry_clearly_improved": bool(bond_improved),
            "residue_rmsf_group_clearly_worsened": bool(rmsf_worsened),
            "fixed_history_time_group_clearly_worsened": bool(time_worsened),
        },
        "trigger_bond_keep_release": trigger,
        "decision": (
            "run one symmetric bond keep/release fork"
            if trigger
            else "do not run a bond keep/release fork"
        ),
    }


def main() -> int:
    args = _arguments()
    formal = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    if (
        not isinstance(formal, Mapping)
        or formal.get("schema") != "molvid.frame_gm.p3_formal_summary.v1"
        or formal.get("test_opened") is not False
        or set(formal.get("arms", {})) != {"J0", "J1"}
        or formal.get("paired_main_stream")
        != {"exposure_exact": True, "sampler_manifest_exact": True}
    ):
        raise ValueError("P3 formal summary is missing or invalid")
    checkpoints = {
        arm: Path(str(formal["arms"][arm].get("checkpoint", "")))
        for arm in ("J0", "J1")
    }
    checkpoint_sha = {}
    for arm, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha[arm] = sha256_file(checkpoint)
        if checkpoint_sha[arm] != formal["arms"][arm].get("checkpoint_sha256"):
            raise ValueError(f"{arm} checkpoint differs from P3 formal summary")
    values = {
        arm: load_arm_values(
            args.legacy_root / arm,
            args.fixed_root / arm,
            expected_checkpoint_sha256=checkpoint_sha[arm],
        )
        for arm in ("J0", "J1")
    }
    targets = values["J0"]["rmsf_target_by_view"]
    other_targets = values["J1"]["rmsf_target_by_view"]
    if set(targets) != set(other_targets) or any(
        not math.isclose(targets[key], other_targets[key], rel_tol=1e-6, abs_tol=1e-8)
        for key in targets
    ):
        raise ValueError("J0/J1 per-view RMSF targets are not paired")
    for mode in ("legacy_run", "fixed_run"):
        if (
            values["J0"]["source"][mode]["protocol_identity"]
            != values["J1"]["source"][mode]["protocol_identity"]
        ):
            raise ValueError(f"J0/J1 evaluation protocol identity differs for {mode}")
    result = compare_from_system_values(
        values["J0"],
        values["J1"],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = {
        "schema": "molvid.frame_gm.p3_bond_fork_decision.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule": FROZEN_RULE,
        "metric_direction": "all compared quantities are errors; lower is better",
        "aggregation": (
            "saved draw/seed -> anchor -> replica -> system; formal views mean within "
            "system; paired percentile bootstrap resamples systems"
        ),
        "checkpoints": {
            arm: {"path": str(checkpoints[arm].resolve()), "sha256": checkpoint_sha[arm]}
            for arm in ("J0", "J1")
        },
        "formal_summary": str(args.formal_summary.resolve()),
        "formal_summary_sha256": sha256_file(args.formal_summary),
        "metric_sources": {
            arm: values[arm]["source"] for arm in ("J0", "J1")
        },
        **result,
        "test_opened": False,
        "unresolved_issues": [
            "P3 comparison is exploratory model selection, not confirmatory evidence",
            "Only the predeclared three-way trigger may authorize the optional short fork",
        ],
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
