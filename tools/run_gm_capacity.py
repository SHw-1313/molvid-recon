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
from tools.evaluate_gm_a import (
    BUDGET_STOP_EXIT_CODE,
    evaluation_protocol,
    validate_completed_evaluation,
)
from tools.gm_capacity_common import capacity_source_identity


FORMAL_TRAIN_BUDGET_SECONDS = 32.0 * 3600.0
PILOT_BUDGET_SECONDS = 8.0 * 3600.0
EVALUATION_BUDGET_SECONDS = 8.0 * 3600.0
EVALUATION_BOUNDARY_RESERVE_SECONDS = 300.0
EVALUATION_HARD_RESERVE_SECONDS = 30.0
LEDGER_HEARTBEAT_SECONDS = 5.0
CHECKPOINT_PATTERN = re.compile(r"frame_joint_step_(\d{8})\.pt$")


def _checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def _checkpoints(output: Path) -> list[tuple[int, Path]]:
    values = [
        (_checkpoint_step(path), path) for path in output.glob("frame_joint_step_*.pt")
    ]
    return sorted((step, path) for step, path in values if step is not None)


def _prepare_training_attempt(
    *,
    phase: str,
    arm: str,
    output: Path,
    arm_stages: list[dict],
    recover: bool,
) -> dict:
    """Choose a fresh/restart/resume transition without overwriting evidence."""

    existing = _checkpoints(output) if output.exists() else []
    archive = None
    if output.exists() and not existing:
        if not recover:
            raise FileExistsError(
                f"{arm} has a pre-checkpoint output; use --recover for a safe restart"
            )
        recoverable = [
            stage for stage in arm_stages
            if stage.get("status") in {"interrupted", "failed"}
        ]
        if not recoverable:
            raise RuntimeError(
                f"{arm} pre-checkpoint output has no interrupted/failed ledger attempt"
            )
        archive_root = output.parent / "_abandoned_precheckpoint"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / f"{output.name}_attempt_{len(arm_stages) - 1:02d}"
        if archive.exists():
            raise FileExistsError(f"refusing to overwrite abandoned attempt {archive}")
        output.rename(archive)
        recoverable[-1]["abandoned_output_archive"] = str(archive)
        recoverable[-1]["recovery_transition"] = "restart_from_frozen_parent"
    attempt_index = len(arm_stages)
    if existing:
        step, checkpoint = existing[-1]
        return {
            "kind": "resume",
            "step": step,
            "checkpoint": checkpoint,
            "stage_name": f"{phase}_{arm}_resume_{step:08d}_{attempt_index:02d}",
            "allow_existing_output": True,
            "archive": archive,
            "checkpoints": existing,
        }
    return {
        "kind": "warm_start",
        "step": 0,
        "checkpoint": None,
        "stage_name": f"{phase}_{arm}_warm_start_{attempt_index:02d}",
        "allow_existing_output": False,
        "archive": archive,
        "checkpoints": [],
    }


def _training_command(
    *,
    config: Path,
    checkpoint_every: str,
    attempt: dict,
    frozen: dict,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "molvid.cli.train_frame_joint",
        "--config",
        str(config),
    ]
    if attempt["kind"] == "resume":
        checkpoint = Path(attempt["checkpoint"])
        command.extend([
            "--resume",
            str(checkpoint),
            "--resume-sha256",
            sha256_file(checkpoint),
        ])
    else:
        command.extend([
            "--warm-start",
            str(frozen["parent_path"]),
            "--warm-start-sha256",
            str(frozen["parent_sha256"]),
        ])
    command.extend(["--checkpoint-every", checkpoint_every])
    return command


def _evaluation_deadlines(*, now: float, used: float) -> tuple[float, float]:
    """Return a resumable row-boundary stop and a hard cap below 8 GPU-hours."""

    remaining = EVALUATION_BUDGET_SECONDS - used
    if remaining <= EVALUATION_BOUNDARY_RESERVE_SECONDS:
        raise RuntimeError("evaluation budget lacks the row-boundary reserve")
    return (
        now + remaining - EVALUATION_BOUNDARY_RESERVE_SECONDS,
        now + remaining - EVALUATION_HARD_RESERVE_SECONDS,
    )


def _finalize_stale_stage(record: dict, *, source_commit: str) -> None:
    """Charge a crashed attempt through its last heartbeat plus one safe interval."""

    observed = max(
        float(record.get("elapsed_seconds_at_checkpoint", 0.0)),
        float(record.get("elapsed_seconds_observed", 0.0)),
    )
    elapsed = observed + LEDGER_HEARTBEAT_SECONDS
    record.update({
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "status": "interrupted",
        "elapsed_seconds_observed_before_crash": observed,
        "unobserved_runtime_charged_seconds": LEDGER_HEARTBEAT_SECONDS,
        "budget_charge_is_conservative": True,
        "recovered_by_source_commit": source_commit,
    })


def _stage_seconds(ledger, prefix: str) -> float:
    return sum(
        float(stage.get("elapsed_seconds", 0.0))
        for stage in ledger["stages"]
        if stage["name"].startswith(prefix)
        and stage.get("status") in {"completed", "budget_stopped", "failed", "interrupted"}
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


def validate_review_gate(
    gate_path: Path,
    *,
    frozen: dict,
    frozen_path: Path,
    source_identity: dict,
) -> dict:
    """Validate the independent formal gate before any output mutation."""

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("independent_session") is not True:
        raise ValueError("capacity review gate is not an independent PASS")
    if gate.get("frozen_configs_sha256") != sha256_file(frozen_path):
        raise ValueError("reviewed frozen configs differ")
    if gate.get("parent_checkpoint_sha256") != frozen["parent_sha256"]:
        raise ValueError("reviewed capacity parent differs")
    if gate.get("candidate_commit") != source_identity["commit"]:
        raise ValueError("review gate candidate differs from the executing source commit")
    if sha256_file(gate["review_path"]) != gate["review_sha256"]:
        raise ValueError("capacity review report identity differs")
    root = Path(__file__).resolve().parents[1]
    changed = subprocess.check_output(
        [
            "git", "diff", str(gate["candidate_commit"]), "--",
            "molvid", "tools", "configs", "tests",
        ],
        cwd=root,
        text=True,
    )
    if changed:
        raise ValueError("runtime source differs from the reviewed capacity candidate")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--review-gate", type=Path)
    parser.add_argument(
        "--recover",
        action="store_true",
        help="resume interrupted arms/evaluations from durable artifacts",
    )
    parser.add_argument("--arms", nargs="+", default=("C4", "C8"))
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("an explicit lane-C GPU UUID assignment is required")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    if frozen.get("schema") != "molvid.capacity.frozen_configs.v1":
        raise ValueError("capacity frozen-config schema differs")
    source_identity = capacity_source_identity()
    if frozen.get("source_identity") != source_identity:
        raise ValueError("capacity frozen configs were not produced by this clean source commit")
    if tuple(args.arms) != ("C4", "C8"):
        raise ValueError("capacity execution requires the matched C4 then C8 pair")
    root = args.frozen.parent
    profile = json.loads(Path(frozen["profile"]).read_text(encoding="utf-8"))
    if sha256_file(frozen["profile"]) != frozen["profile_sha256"]:
        raise ValueError("capacity profile changed after config freeze")
    if args.phase == "formal":
        if args.review_gate is None:
            raise ValueError("formal capacity training requires an independent review gate")
        validate_review_gate(
            args.review_gate,
            frozen=frozen,
            frozen_path=args.frozen,
            source_identity=source_identity,
        )

    ledger_path = root / f"{args.phase}_cost_ledger.json"
    ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else {
            "schema": "molvid.capacity.stage_costs.v1",
            "phase": args.phase,
            "source_identity": source_identity,
            "stages": [],
        }
    )
    if ledger.get("source_identity") != source_identity:
        raise ValueError("capacity cost ledger source identity differs")
    stale = [stage for stage in ledger["stages"] if stage.get("status") == "running"]
    if stale and not args.recover:
        raise RuntimeError("capacity ledger contains an interrupted stage; rerun with --recover")
    for record in stale:
        _finalize_stale_stage(record, source_commit=source_identity["commit"])
    if stale:
        atomic_write_json(ledger_path, ledger)

    def run_stage(
        name: str,
        command: list[str],
        *,
        output: Path | None = None,
        train_arm: str | None = None,
        allow_existing_output: bool = False,
    ) -> tuple[str, Path | None]:
        log = root / f"{name}.log"
        if log.exists():
            raise FileExistsError(f"refusing to overwrite existing stage log {log}")
        if output is not None and output.exists() and not allow_existing_output:
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
        evaluation_boundary_stop = None
        evaluation_hard_stop = None
        if args.phase == "formal" and train_arm is None:
            evaluation_boundary_stop, evaluation_hard_stop = _evaluation_deadlines(
                now=time.time(), used=used
            )
            command = [
                *command,
                "--stop-before-unix",
                repr(evaluation_boundary_stop),
            ]
        record = {
            "name": name,
            "command": command,
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "started_unix": time.time(),
            "status": "running",
        }
        if evaluation_boundary_stop is not None:
            record.update({
                "evaluation_boundary_stop_unix": evaluation_boundary_stop,
                "evaluation_hard_stop_unix": evaluation_hard_stop,
                "evaluation_boundary_reserve_seconds": EVALUATION_BOUNDARY_RESERVE_SECONDS,
                "evaluation_hard_reserve_seconds": EVALUATION_HARD_RESERVE_SECONDS,
            })
        ledger["stages"].append(record)
        atomic_write_json(ledger_path, ledger)
        stopped = False
        last_checkpoint: Path | None = None
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            atomic_write_json(ledger_path, ledger)
            print(json.dumps({"stage": name, "pid": process.pid, "log": str(log)}), flush=True)
            last_heartbeat = time.time()
            try:
                while True:
                    code = process.poll()
                    if code is not None:
                        break
                    now = time.time()
                    elapsed = now - record["started_unix"]
                    if now - last_heartbeat >= LEDGER_HEARTBEAT_SECONDS:
                        record["elapsed_seconds_observed"] = elapsed
                        record["gpu_hours_observed"] = elapsed / 3600.0
                        atomic_write_json(ledger_path, ledger)
                        last_heartbeat = now
                    if evaluation_hard_stop is not None and now >= evaluation_hard_stop:
                        record["budget_stop_reason"] = "evaluation_hard_deadline"
                        process.send_signal(signal.SIGTERM)
                        try:
                            code = process.wait(timeout=20.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            code = process.wait()
                        stopped = True
                        break
                    if output is not None:
                        checkpoints = _checkpoints(output)
                        if checkpoints and checkpoints[-1][1] != last_checkpoint:
                            last_checkpoint = checkpoints[-1][1]
                            record["last_checkpoint"] = str(last_checkpoint)
                            record["last_checkpoint_step"] = checkpoints[-1][0]
                            elapsed = time.time() - record["started_unix"]
                            record["elapsed_seconds_at_checkpoint"] = elapsed
                            record["gpu_hours_at_checkpoint"] = elapsed / 3600.0
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
            except BaseException:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=30.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                elapsed = time.time() - record["started_unix"]
                record.update({
                    "exit_code": process.returncode,
                    "elapsed_seconds": elapsed,
                    "gpu_hours": elapsed / 3600.0,
                    "status": "interrupted",
                })
                atomic_write_json(ledger_path, ledger)
                raise
        if (
            evaluation_boundary_stop is not None
            and code == BUDGET_STOP_EXIT_CODE
        ):
            stopped = True
            record["budget_stop_reason"] = "evaluation_row_boundary"
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
        arm_stages = [
            stage for stage in ledger["stages"]
            if stage["name"].startswith(f"{args.phase}_{arm}_")
        ]
        attempt = _prepare_training_attempt(
            phase=args.phase,
            arm=arm,
            output=output,
            arm_stages=arm_stages,
            recover=args.recover,
        )
        if attempt["archive"] is not None:
            atomic_write_json(ledger_path, ledger)
        existing = attempt["checkpoints"]
        target_updates = int(record["updates"])
        status = "completed"
        last_checkpoint = existing[-1][1] if existing else None
        if existing and not args.recover:
            raise FileExistsError(
                f"{arm} has durable checkpoints; use --recover for an exact continuation"
            )
        if existing and existing[-1][0] > target_updates:
            raise RuntimeError(f"{arm} checkpoint exceeds the frozen update target")
        budget_stopped = any(stage.get("status") == "budget_stopped" for stage in arm_stages)
        if budget_stopped:
            status = "budget_stopped"
        if not existing or (existing[-1][0] < target_updates and not budget_stopped):
            command = _training_command(
                config=config,
                checkpoint_every=checkpoint_every,
                attempt=attempt,
                frozen=frozen,
            )
            status, last_checkpoint = run_stage(
                attempt["stage_name"],
                command,
                output=output,
                train_arm=arm,
                allow_existing_output=attempt["allow_existing_output"],
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
            "source_identity": source_identity,
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
        "source_identity": source_identity,
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
                evaluation_output = root / "evaluation" / arm / family
                summary_path = evaluation_output / "summary.json"
                expected_protocol = evaluation_protocol(
                    checkpoint_sha256=paired[arm]["checkpoint_sha256"],
                    recipe=recipe,
                    store=Path(source["store"]),
                    manifest=Path(source["manifest"]),
                )
                if summary_path.exists():
                    if not args.recover:
                        raise FileExistsError(
                            f"{arm} {family} evaluation exists; use --recover to verify and reuse it"
                        )
                    validate_completed_evaluation(
                        evaluation_output,
                        expected_protocol=expected_protocol,
                        expected_family=family,
                        store=Path(source["store"]),
                    )
                    continue
                evaluation_exists = evaluation_output.exists()
                if evaluation_exists and not args.recover:
                    raise FileExistsError(
                        f"{arm} {family} evaluation is partial; use --recover"
                    )
                eval_stages = [
                    stage for stage in ledger["stages"]
                    if stage["name"].startswith(f"eval_{arm}_{family}")
                ]
                command = [
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
                    str(evaluation_output),
                ]
                if evaluation_exists:
                    command.append("--resume")
                status, _ = run_stage(
                    f"eval_{arm}_{family}_{len(eval_stages):02d}",
                    command,
                    output=evaluation_output,
                    allow_existing_output=evaluation_exists,
                )
                if status == "budget_stopped":
                    raise RuntimeError(
                        "evaluation allocation stopped at a resumable boundary; "
                        "the 8 GPU-hour cap is exhausted"
                    )
                validate_completed_evaluation(
                    evaluation_output,
                    expected_protocol=expected_protocol,
                    expected_family=family,
                    store=Path(source["store"]),
                )


if __name__ == "__main__":
    main()
