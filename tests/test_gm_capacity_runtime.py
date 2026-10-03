from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
from torch import nn

from molvid.runtime import sha256_file
from molvid.training.joint import FrameJointTrainer
from tools.evaluate_gm_a import (
    evaluation_protocol,
    validate_completed_evaluation,
)
from tools.run_gm_capacity import (
    CHILD_LAUNCH_PROTOCOL,
    EVALUATION_BOUNDARY_RESERVE_SECONDS,
    EVALUATION_BUDGET_SECONDS,
    EVALUATION_HARD_RESERVE_SECONDS,
    _evaluation_deadlines,
    _prepare_training_attempt,
    _process_start_ticks,
    _recover_stale_stage,
    _release_gated_child,
    _spawn_gated_child,
    _training_command,
    validate_frozen_release_files,
    validate_review_gate,
)


class _AdaLN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modulation = nn.Linear(2, 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)


class _Motion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_residual_gate = nn.Parameter(torch.zeros(()))
        self.vector_residual_gate = nn.Parameter(torch.zeros(()))
        self.inner = nn.Linear(2, 2)


class _Geometry(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_gate = nn.Parameter(torch.zeros(()))
        self.vector_residual_gate = nn.Parameter(torch.zeros(()))
        self.inner = nn.Linear(2, 2)


class _Block(nn.Module):
    def __init__(self, geometry: bool) -> None:
        super().__init__()
        self.history_adaln = _AdaLN()
        self.spatial_adaln = _AdaLN()
        self.temporal_adaln = _AdaLN()
        self.ffn_adaln = _AdaLN()
        self.motion_context = _Motion()
        self.local_geometry = _Geometry() if geometry else None
        self.inner = nn.Linear(2, 2)


class _DiT(nn.Module):
    def __init__(self, depth: int, geometry: tuple[int, ...]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Block(index + 1 in geometry) for index in range(depth)]
        )


class _FakeCapacityModel(nn.Module):
    def __init__(self, depth: int, geometry: tuple[int, ...]) -> None:
        super().__init__()
        self.history_backend = "legacy"
        self.statistics_hash = "stats"
        self.shared = nn.Linear(2, 2)
        self.dit = _DiT(depth, geometry)
        self._depth = depth
        self._geometry = geometry

    def contract(self) -> dict:
        return {
            "schema_version": "molvid.frame_joint.model.v2",
            "dit": {
                "schema_version": "molvid.frame_joint.dit.v2",
                "depth": self._depth,
                "scalar_width": 2,
                "vector_width": 2,
                "heads": 1,
                "geometry_enabled": True,
                "motion_enabled": True,
                "local_geometry_blocks_one_based": list(self._geometry),
            },
        }


def test_capacity_warm_start_maps_every_parent_block_and_reports_new_residuals(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1)
    parent = _FakeCapacityModel(4, (2, 4))
    torch.manual_seed(2)
    target = _FakeCapacityModel(8, (2, 4, 6, 8))
    checkpoint = tmp_path / "parent.pt"
    torch.save({
        "schema_version": "molvid.training.checkpoint.v1",
        "step": 17,
        "contracts": {
            "statistics_hash": "stats",
            "teacher_artifact_sha256": "teacher",
            "data_hash": "data",
            "model": parent.contract(),
        },
        "model_state": parent.state_dict(),
    }, checkpoint)
    trainer = object.__new__(FrameJointTrainer)
    trainer.base_model = target
    trainer.teacher_artifact_sha256 = "teacher"
    trainer.step = -1
    trainer.successful_updates = -1
    trainer.continuation_parent = None
    report = trainer.load_warm_start(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        dit_block_mapping=(2, 4, 6, 8),
    )
    assert report.dit_block_mapping_one_based == (2, 4, 6, 8)
    assert report.identity_initialized_blocks_one_based == (1, 3, 5, 7)
    assert report.zero_initialized_geometry_blocks_one_based == (2, 6)
    assert not report.unexpected
    assert not report.shape_mismatch
    for source, destination in enumerate((1, 3, 5, 7)):
        source_prefix = f"dit.blocks.{source}."
        destination_prefix = f"dit.blocks.{destination}."
        for name, value in parent.state_dict().items():
            if name.startswith(source_prefix):
                target_name = destination_prefix + name.removeprefix(source_prefix)
                if target_name in target.state_dict():
                    assert torch.equal(value, target.state_dict()[target_name])


def test_capacity_review_gate_rejects_missing_and_stale_identity(tmp_path: Path) -> None:
    frozen_path = tmp_path / "frozen.json"
    frozen = {"parent_sha256": "parent"}
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(FileNotFoundError):
        validate_review_gate(
            tmp_path / "missing.json",
            frozen=frozen,
            frozen_path=frozen_path,
            source_identity={"commit": commit},
        )
    review = tmp_path / "REVIEW.md"
    review.write_text("PASS\n", encoding="utf-8")
    gate = {
        "verdict": "PASS",
        "independent_session": True,
        "frozen_configs_sha256": "stale",
        "parent_checkpoint_sha256": "parent",
        "candidate_commit": commit,
        "review_path": str(review),
        "review_sha256": sha256_file(review),
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen configs differ"):
        validate_review_gate(
            gate_path,
            frozen=frozen,
            frozen_path=frozen_path,
            source_identity={"commit": commit},
        )


def test_capacity_runner_archives_precheckpoint_attempt_and_restarts_parent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal" / "C4"
    output.mkdir(parents=True)
    (output / "train_metrics.jsonl").write_text("partial\n", encoding="utf-8")
    original_log = tmp_path / "formal_C4_warm_start_00.log"
    original_log.write_text("interrupted\n", encoding="utf-8")
    stages = [{
        "name": "formal_C4_warm_start_00",
        "status": "interrupted",
    }]
    attempt = _prepare_training_attempt(
        phase="formal",
        arm="C4",
        output=output,
        arm_stages=stages,
        recover=True,
    )
    assert attempt["kind"] == "warm_start"
    assert attempt["stage_name"] == "formal_C4_warm_start_01"
    assert not output.exists()
    archive = Path(attempt["archive"])
    assert (archive / "train_metrics.jsonl").read_text(encoding="utf-8") == "partial\n"
    assert original_log.read_text(encoding="utf-8") == "interrupted\n"
    assert stages[0]["abandoned_output_archive"] == str(archive)
    command = _training_command(
        config=tmp_path / "formal_C4.json",
        checkpoint_every="1000",
        attempt=attempt,
        frozen={"parent_path": "parent.pt", "parent_sha256": "parent-sha"},
    )
    assert "--warm-start" in command
    assert "--resume" not in command


def test_capacity_runner_second_interruption_resumes_latest_durable_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal" / "C8"
    output.mkdir(parents=True)
    checkpoint = output / "frame_joint_step_00001000.pt"
    checkpoint.write_bytes(b"durable")
    stages = [
        {"name": "formal_C8_warm_start_00", "status": "interrupted"},
        {"name": "formal_C8_resume_00001000_01", "status": "interrupted"},
    ]
    attempt = _prepare_training_attempt(
        phase="formal",
        arm="C8",
        output=output,
        arm_stages=stages,
        recover=True,
    )
    assert attempt["kind"] == "resume"
    assert attempt["checkpoint"] == checkpoint
    assert attempt["stage_name"] == "formal_C8_resume_00001000_02"
    assert attempt["archive"] is None
    command = _training_command(
        config=tmp_path / "formal_C8.json",
        checkpoint_every="1000",
        attempt=attempt,
        frozen={"parent_path": "parent.pt", "parent_sha256": "parent-sha"},
    )
    assert "--resume" in command
    assert "--warm-start" not in command
    assert command[command.index("--resume-sha256") + 1] == sha256_file(checkpoint)


def _write_completed_evaluation(tmp_path: Path) -> tuple[Path, dict, Path]:
    recipe = tmp_path / "recipe.json"
    manifest = tmp_path / "manifest.json"
    store = tmp_path / "store"
    output = tmp_path / "evaluation"
    recipe.write_text("{}\n", encoding="utf-8")
    manifest.write_text('{"test_opened": false}\n', encoding="utf-8")
    store.mkdir()
    (store / "index.txt").write_text("sample_H4\t0\t1\t1\t16\tdt_100ps\n", encoding="utf-8")
    (output / "coordinates").mkdir(parents=True)
    (output / "profiles").mkdir()
    protocol = evaluation_protocol(
        checkpoint_sha256="checkpoint",
        recipe=recipe,
        store=store,
        manifest=manifest,
    )
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    rows = []
    for path, seed in (
        ("persistence", None),
        ("oracle", None),
        ("generated", 0),
        ("generated", 1),
        ("generated", 2),
    ):
        name = f"{path}-{seed}.npz"
        (output / "coordinates" / name).write_bytes(b"coordinate")
        (output / "profiles" / name).write_bytes(b"profile")
        rows.append({
            "row_key": f"sample_H4|H4|{path}|seed={seed}",
            "path": path,
            "seed": seed,
            "metric_family": {"family": "ordinary"},
            "coordinate_file": name,
            "profile_file": name,
        })
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "per_system.csv").write_text("system\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "family": "ordinary",
        "records": 1,
        "rows": 5,
        "checkpoint_sha256": "checkpoint",
        "protocol_sha256": sha256_file(protocol_path),
        "test_opened": False,
    }), encoding="utf-8")
    return output, protocol, store


def test_completed_evaluation_reuse_requires_full_protocol_and_coverage(
    tmp_path: Path,
) -> None:
    output, protocol, store = _write_completed_evaluation(tmp_path)
    verified = validate_completed_evaluation(
        output,
        expected_protocol=protocol,
        expected_family="ordinary",
        store=store,
    )
    assert verified["records"] == 1
    stale = dict(protocol)
    stale["recipe_sha256"] = "stale"
    with pytest.raises(ValueError, match="protocol differs"):
        validate_completed_evaluation(
            output,
            expected_protocol=stale,
            expected_family="ordinary",
            store=store,
        )
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rows"] = 4
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="row coverage differs"):
        validate_completed_evaluation(
            output,
            expected_protocol=protocol,
            expected_family="ordinary",
            store=store,
        )


def test_evaluation_deadlines_preserve_row_and_hard_cap_reserves() -> None:
    boundary, hard = _evaluation_deadlines(now=100.0, used=3600.0)
    assert boundary == pytest.approx(
        100.0 + EVALUATION_BUDGET_SECONDS - 3600.0
        - EVALUATION_BOUNDARY_RESERVE_SECONDS
    )
    assert hard == pytest.approx(
        100.0 + EVALUATION_BUDGET_SECONDS - 3600.0
        - EVALUATION_HARD_RESERVE_SECONDS
    )
    assert boundary < hard
    with pytest.raises(RuntimeError, match="row-boundary reserve"):
        _evaluation_deadlines(
            now=100.0,
            used=EVALUATION_BUDGET_SECONDS - EVALUATION_BOUNDARY_RESERVE_SECONDS,
        )


def test_recovery_refuses_a_stage_owned_by_a_live_runner() -> None:
    record = {
        "name": "pilot_C4_warm_start_00",
        "status": "running",
        "runner_pid": os.getpid(),
        "runner_start_ticks": _process_start_ticks(os.getpid()),
    }
    with pytest.raises(RuntimeError, match="live runner"):
        _recover_stale_stage(record, source_commit="candidate")


def test_recovery_refuses_identityless_running_stage() -> None:
    record = {
        "name": "pilot_C4_warm_start_00",
        "status": "running",
        "started_unix": time.time() - 1.0,
    }
    with pytest.raises(RuntimeError, match="lacks durable child identity"):
        _recover_stale_stage(record, source_commit="candidate")


def test_launch_gate_parent_death_prevents_exec_before_identity(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "command_executed"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
        str(marker),
    ]
    process, gate_write_fd = _spawn_gated_child(
        command,
        stdout=subprocess.DEVNULL,
    )
    record = {
        "name": "pilot_C4_warm_start_00",
        "status": "launching",
        "started_unix": time.time(),
        "child_launch_protocol": CHILD_LAUNCH_PROTOCOL,
    }
    # Closing the inherited pipe without the release token models the runner
    # dying after Popen but before the child identity is durably recorded.
    os.close(gate_write_fd)
    assert process.wait(timeout=5.0) == 125
    _recover_stale_stage(record, source_commit="candidate")
    assert record["status"] == "interrupted"
    assert record["child_exec_prevented_by_launch_gate"] is True
    assert record["child_was_alive_at_recovery"] is False
    assert not marker.exists()

    attempt = _prepare_training_attempt(
        phase="pilot",
        arm="C4",
        output=tmp_path / "pilot" / "C4",
        arm_stages=[record],
        recover=True,
    )
    assert attempt["kind"] == "warm_start"
    assert process.poll() is not None


def test_launch_gate_release_executes_the_recorded_command(tmp_path: Path) -> None:
    marker = tmp_path / "command_executed"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
        str(marker),
    ]
    process, gate_write_fd = _spawn_gated_child(
        command,
        stdout=subprocess.DEVNULL,
    )
    assert not marker.exists()
    _release_gated_child(gate_write_fd)
    assert process.wait(timeout=5.0) == 0
    assert marker.read_text(encoding="utf-8") == "ok"


def test_recovery_stops_live_child_before_archiving_output(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    started = time.time()
    try:
        child_start_ticks = _process_start_ticks(process.pid)
        assert child_start_ticks is not None
        record = {
            "name": "pilot_C4_warm_start_00",
            "status": "running",
            "started_unix": started,
            "elapsed_seconds_observed": 0.01,
            "child_pid": process.pid,
            "child_start_ticks": child_start_ticks,
            "child_process_group_id": os.getpgid(process.pid),
            "child_session_id": os.getsid(process.pid),
        }
        _recover_stale_stage(record, source_commit="candidate")
        process.wait(timeout=5.0)
        assert record["status"] == "interrupted"
        assert record["child_was_alive_at_recovery"] is True
        assert record["child_stop_signal"] == "SIGTERM"
        assert record["elapsed_seconds"] >= record["elapsed_seconds_observed"]
        assert record["gpu_hours"] == pytest.approx(
            record["elapsed_seconds"] / 3600.0
        )
        assert record["budget_charge_is_conservative"] is True
        assert record["recovered_by_source_commit"] == "candidate"

        output = tmp_path / "pilot" / "C4"
        output.mkdir(parents=True)
        (output / "partial.txt").write_text("stopped", encoding="utf-8")
        attempt = _prepare_training_attempt(
            phase="pilot",
            arm="C4",
            output=output,
            arm_stages=[record],
            recover=True,
        )
        assert attempt["kind"] == "warm_start"
        assert attempt["archive"] is not None
        assert not output.exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)


def test_frozen_release_files_are_rehashed_at_runner_start(tmp_path: Path) -> None:
    capacity = tmp_path / "capacity_release.json"
    recipe = tmp_path / "recipe_release.json"
    capacity.write_text("capacity", encoding="utf-8")
    recipe.write_text("recipe", encoding="utf-8")
    frozen = {
        "capacity_release": str(capacity),
        "capacity_release_sha256": sha256_file(capacity),
        "recipe_release": str(recipe),
        "recipe_release_sha256": sha256_file(recipe),
    }
    validate_frozen_release_files(frozen)
    recipe.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="recipe release changed"):
        validate_frozen_release_files(frozen)

def test_completed_evaluation_accepts_d_seed_scope_and_rejects_missing_draw(tmp_path):
    output, protocol, store = _write_completed_evaluation(tmp_path)
    protocol["seeds"] = [0]
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    rows_path = output / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows = [row for row in rows if row["seed"] in (None, 0)]
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(rows=3, protocol_sha256=sha256_file(protocol_path))
    summary_path.write_text(json.dumps(summary))
    result = validate_completed_evaluation(
        output, expected_protocol=protocol, expected_family="ordinary", store=store,
    )
    assert result["rows"] == 3
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(ValueError, match="coverage differs"):
        validate_completed_evaluation(
            output, expected_protocol=protocol, expected_family="ordinary", store=store,
        )
