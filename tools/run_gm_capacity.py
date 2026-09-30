"""Run matched C4/C8 pilots or reviewed formal training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import torch

from molvid.runtime import atomic_write_json, canonical_hash, sha256_file


FORMAL_TRAIN_BUDGET_SECONDS = 32.0 * 3600.0
PILOT_BUDGET_SECONDS = 8.0 * 3600.0
EVALUATION_BUDGET_SECONDS = 8.0 * 3600.0
CHECKPOINT_PATTERN = re.compile(r"frame_joint_step_(\d{8})\.pt$")


def _checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def _checkpoints(output: Path) -> list[tuple[int, Path]]:
    values = [
        (_checkpoint_step(path), path) for path in output.glob("frame_joint_step_*.pt")
    ]
    return sorted((step, path) for step, path in values if step is not None)


def _stage_seconds(ledger, prefix: str) -> float:
    return sum(
        float(stage.get("elapsed_seconds", 0.0))
        for stage in ledger["stages"]
        if stage["name"].startswith(prefix)
        and stage.get("status") in {"completed", "budget_stopped", "failed"}
    )


def _load_rank_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rank_states = payload["extra_state"].get("rank_states", {})
    return rank_states.get("0", {})


def _verify_paired_checkpoint(first: Path, second: Path) -> dict:
    left = _load_rank_state(first)
    right = _load_rank_state(second)
    left_exposure = left.get("exposure_state")
    right_exposure = right.get("exposure_state")
    if left_exposure != right_exposure:
        raise RuntimeError("C4/C8 checkpoints do not have identical global exposure")
    for name in ("training_generator_state", "sampled_generator_state"):
        if not torch.equal(left[name], right[name]):
            raise RuntimeError(f"C4/C8 checkpoints differ in {name}")
    return {
        "exposure_sha256": canonical_hash(left_exposure),
        "training_generator_equal": True,
        "sampled_generator_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--review-gate", type=Path)
    parser.add_argument("--arms", nargs="+", default=("C4", "C8"))
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("an explicit lane-C GPU UUID assignment is required")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    if frozen.get("schema") != "molvid.capacity.frozen_configs.v1":
        raise ValueError("capacity frozen-config schema differs")
    if tuple(args.arms) != ("C4", "C8"):
        raise ValueError("capacity execution requires the matched C4 then C8 pair")
    root = args.frozen.parent
    profile = json.loads(Path(frozen["profile"]).read_text(encoding="utf-8"))
    if sha256_file(frozen["profile"]) != frozen["profile_sha256"]:
        raise ValueError("capacity profile changed after config freeze")
    if args.phase == "formal":
        if args.review_gate is None:
            raise ValueError("formal capacity training requires an independent review gate")
        gate = json.loads(args.review_gate.read_text(encoding="utf-8"))
        if gate.get("verdict") != "PASS" or gate.get("independent_session") is not True:
            raise ValueError("capacity review gate is not an independent PASS")
        if gate.get("frozen_configs_sha256") != sha256_file(args.frozen):
            raise ValueError("reviewed frozen configs differ")
        if gate.get("parent_checkpoint_sha256") != frozen["parent_sha256"]:
            raise ValueError("reviewed capacity parent differs")
        if sha256_file(gate["review_path"]) != gate["review_sha256"]:
            raise ValueError("capacity review report identity differs")
        changed = subprocess.check_output(
            [
                "git", "diff", str(gate["candidate_commit"]), "--",
                "molvid", "tools", "configs", "tests",
            ],
            text=True,
        )
        if changed:
            raise ValueError("runtime source differs from the reviewed capacity candidate")

    ledger_path = root / f"{args.phase}_cost_ledger.json"
    ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else {"schema": "molvid.capacity.stage_costs.v1", "phase": args.phase, "stages": []}
    )

    def run_stage(
        name: str,
        command: list[str],
        *,
        output: Path | None = None,
        train_arm: str | None = None,
    ) -> tuple[str, Path | None]:
        log = root / f"{name}.log"
        if log.exists():
            raise FileExistsError(f"refusing to overwrite existing stage log {log}")
        if output is not None and output.exists():
            raise FileExistsError(f"refusing to overwrite existing output {output}")
        budget = (
            PILOT_BUDGET_SECONDS
            if args.phase == "pilot"
            else FORMAL_TRAIN_BUDGET_SECONDS
            if train_arm is not None
            else EVALUATION_BUDGET_SECONDS
        )
        prefix = "pilot_" if args.phase == "pilot" else "formal_" if train_arm else "eval_"
        used = _stage_seconds(ledger, prefix)
        if used >= budget:
            raise RuntimeError(f"{prefix.rstrip('_')} budget is exhausted")
        record = {
            "name": name,
            "command": command,
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "started_unix": time.time(),
            "status": "running",
        }
        ledger["stages"].append(record)
        atomic_write_json(ledger_path, ledger)
        stopped = False
        last_checkpoint: Path | None = None
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            atomic_write_json(ledger_path, ledger)
            print(json.dumps({"stage": name, "pid": process.pid, "log": str(log)}), flush=True)
            while True:
                code = process.poll()
                if code is not None:
                    break
                if output is not None:
                    checkpoints = _checkpoints(output)
                    if checkpoints and checkpoints[-1][1] != last_checkpoint:
                        last_checkpoint = checkpoints[-1][1]
                        record["last_checkpoint"] = str(last_checkpoint)
                        record["last_checkpoint_step"] = checkpoints[-1][0]
                        elapsed = time.time() - record["started_unix"]
                        if args.phase == "formal" and train_arm is not None:
                            seconds_per_update = float(
                                profile["arms"][train_arm]["weighted_seconds_per_update"]
                            )
                            reserve = 120.0 + 1.5 * 1000.0 * seconds_per_update
                            if used + elapsed + reserve >= budget:
                                record["budget_stop_reason"] = "next_checkpoint_reserve"
                                process.send_signal(signal.SIGTERM)
                                code = process.wait()
                                stopped = True
                                break
                        atomic_write_json(ledger_path, ledger)
                time.sleep(1.0)
        elapsed = time.time() - record["started_unix"]
        record.update({
            "exit_code": code,
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "status": "budget_stopped" if stopped else "completed" if code == 0 else "failed",
        })
        atomic_write_json(ledger_path, ledger)
        if not stopped and code != 0:
            raise RuntimeError(f"stage {name} failed; inspect {log}")
        return record["status"], last_checkpoint

    retained: dict[str, tuple[int, Path]] = {}
    for arm in args.arms:
        record = frozen["arms"][arm][args.phase]
        config = Path(record["config"])
        if sha256_file(config) != record["config_sha256"]:
            raise ValueError(f"{arm} {args.phase} config changed after freeze")
        output = Path(record["output_root"])
        checkpoint_every = "64" if args.phase == "pilot" else "1000"
        status, last_checkpoint = run_stage(
            f"{args.phase}_{arm}_train",
            [
                sys.executable,
                "-u",
                "-m",
                "molvid.cli.train_frame_joint",
                "--config",
                str(config),
                "--warm-start",
                str(frozen["parent_path"]),
                "--warm-start-sha256",
                str(frozen["parent_sha256"]),
                "--checkpoint-every",
                checkpoint_every,
            ],
            output=output,
            train_arm=arm,
        )
        checkpoints = _checkpoints(output)
        if not checkpoints:
            raise RuntimeError(f"{arm} produced no durable checkpoint")
        step, checkpoint = checkpoints[-1]
        warm = json.loads((output / "warm_start_report.json").read_text(encoding="utf-8"))
        if warm.get("dit_block_mapping_one_based") != frozen["arms"][arm]["parent_block_mapping_one_based"]:
            raise RuntimeError(f"{arm} warm-start mapping differs from the frozen protocol")
        retained[arm] = (step, checkpoint)
        atomic_write_json(output / "retained_identity.json", {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "successful_updates": step,
            "config_sha256": record["config_sha256"],
            "parent_sha256": frozen["parent_sha256"],
            "status": status,
        })

    paired_step = min(value[0] for value in retained.values())
    paired = {}
    for arm in args.arms:
        candidates = dict(_checkpoints(Path(frozen["arms"][arm][args.phase]["output_root"])))
        if paired_step not in candidates:
            raise RuntimeError(f"{arm} lacks the matched checkpoint at step {paired_step}")
        path = candidates[paired_step]
        paired[arm] = {
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "successful_updates": paired_step,
        }
    paired.update(_verify_paired_checkpoint(
        Path(paired["C4"]["checkpoint"]), Path(paired["C8"]["checkpoint"])
    ))
    paired.update({
        "schema": "molvid.capacity.paired_identity.v1",
        "phase": args.phase,
        "parent_sha256": frozen["parent_sha256"],
        "test_opened": False,
    })
    atomic_write_json(root / f"{args.phase}_paired_identity.json", paired)

    if args.phase == "formal":
        recipe = Path(frozen["evaluation_recipe"])
        if sha256_file(recipe) != frozen["evaluation_recipe_sha256"]:
            raise ValueError("capacity evaluation recipe changed after freeze")
        for arm in args.arms:
            for family, source in frozen["evaluation_families"].items():
                if sha256_file(source["manifest"]) != source["manifest_sha256"]:
                    raise ValueError(f"{family} evaluation manifest changed")
                if sha256_file(Path(source["store"]) / "index.txt") != source["store_index_sha256"]:
                    raise ValueError(f"{family} evaluation store changed")
                run_stage(
                    f"eval_{arm}_{family}",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "tools.evaluate_gm_a",
                        "--checkpoint",
                        paired[arm]["checkpoint"],
                        "--checkpoint-sha256",
                        paired[arm]["checkpoint_sha256"],
                        "--recipe",
                        str(recipe),
                        "--store",
                        source["store"],
                        "--manifest",
                        source["manifest"],
                        "--output",
                        str(root / "evaluation" / arm / family),
                    ],
                    output=root / "evaluation" / arm / family,
                )


if __name__ == "__main__":
    main()
