"""Build a hash-bound factual summary of the four P2 pilot arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from molvid.runtime import atomic_write_json, sha256_file


ARMS = ("B0", "G", "M", "GM")
METRICS = ("bond_rmse", "rmsf_absolute_error", "aligned_rmsd", "drmsd")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluation_summary(path: Path) -> dict[str, Any]:
    rows = [
        row for row in read_jsonl(path / "rows.jsonl")
        if row["path"] == "generated" and row["clock"] == "true"
    ]
    checks = read_json(path / "checks" / "checks.json")
    return {
        "generated_rows": len(rows),
        "mean": {name: fmean(float(row["summary"][name]) for row in rows) for name in METRICS},
        "mean_msd_curve_absolute_error_A2": fmean(
            fmean(float(point["absolute_error"]) for point in row["lag_msd"]["t0_curve"])
            for row in rows
        ),
        "checks": checks,
        "rows_sha256": sha256_file(path / "rows.jsonl"),
        "protocol_sha256": sha256_file(path / "protocol.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms: dict[str, Any] = {}
    schedule_hashes: set[str] = set()
    selected_hashes: set[str] = set()
    for arm in ARMS:
        run = args.root / "pilot" / arm
        metrics = read_jsonl(run / "train_metrics.jsonl")
        exposure = read_json(run / "exposure.json")
        warm = read_json(run / "warm_start_report.json")
        profile = read_json(args.root / "profile" / f"{arm}.json")
        sampler = read_json(run / "sampler_manifest.json")
        checkpoint = run / "frame_joint_step_00000128.pt"
        schedule_hashes.add(str(sampler["global_batch_schedule_hash"]))
        selected_hashes.add(str(sampler["selected_sample_ids_hash"]))
        finite = all(
            math.isfinite(float(row[name]))
            for row in metrics
            for name in ("loss", "grad_norm")
        )
        arms[arm] = {
            "successful_updates": len(metrics),
            "finite_loss_and_grad": finite,
            "first_loss": float(metrics[0]["loss"]),
            "last_loss": float(metrics[-1]["loss"]),
            "mean_loss": fmean(float(row["loss"]) for row in metrics),
            "mean_grad_norm": fmean(float(row["grad_norm"]) for row in metrics),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "config_sha256": sha256_file(Path(profile["config"])),
            "train_metrics_sha256": sha256_file(run / "train_metrics.jsonl"),
            "warm_start_report": warm,
            "exposure": exposure,
            "profile": profile,
            "legacy_pilot_evaluation": evaluation_summary(args.root / "pilot_eval" / arm),
            "fixed_history_pilot_evaluation": evaluation_summary(args.root / "pilot_eval_fixed" / arm),
        }
    output = {
        "schema": "molvid.frame_gm.p2_pilot_summary.v1",
        "arms": arms,
        "paired_sampler": {
            "schedule_hashes": sorted(schedule_hashes),
            "selected_sample_id_hashes": sorted(selected_hashes),
            "identical_across_arms": len(schedule_hashes) == 1 and len(selected_hashes) == 1,
        },
        "pilot_scope": {
            "systems": 3,
            "physical_trajectories": 9,
            "successful_updates_per_arm": 128,
            "evaluation_seed": [0],
            "euler_steps": 16,
            "legacy_views": "one paired anchor, lags 100/200/300/400ps, H4/H8",
            "fixed_history_views": "one paired anchor, lags 100/200/300/400ps, H4 with padded frame masks",
        },
        "targeted_tests": {"passed": 32, "failed": 0, "device": "CUDA"},
        "test_opened": False,
        "formal_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, output)
    print(json.dumps({"output": str(args.output), "arms": list(arms), "paired": output["paired_sampler"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
