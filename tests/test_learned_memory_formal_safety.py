"""Bounded safety checks for formal raw I/O and paired GPU accounting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from molvid.data.learned_memory_formal import _validate_offset_sidecar
from molvid.runtime import sha256_file
from molvid.training.formal_budget import PairedFormalBudget


def test_offset_sidecar_rejects_uniform_shift(tmp_path: Path) -> None:
    offset_root = tmp_path / "offsets"
    sidecar = offset_root / "system" / "trajectory.xtc.offsets.npz"
    sidecar.parent.mkdir(parents=True)
    xtc = tmp_path / "trajectory.xtc"
    xtc.write_bytes(bytes(64))
    source = xtc.stat()

    def write(offsets: np.ndarray) -> None:
        with sidecar.open("wb") as handle:
            np.savez_compressed(
                handle,
                offsets=offsets,
                xtc_size=np.asarray(source.st_size, dtype=np.int64),
                xtc_mtime_ns=np.asarray(source.st_mtime_ns, dtype=np.int64),
            )

    write(np.asarray([0, 16, 32, 48], dtype=np.int64))
    record = {
        "frames": 4,
        "sidecar": str(sidecar),
        "sidecar_sha256": sha256_file(sidecar),
        "splits": ["train"],
        "xtc": str(xtc),
        "xtc_mtime_ns": source.st_mtime_ns,
        "xtc_size": source.st_size,
    }
    _validate_offset_sidecar(
        xtc=xtc,
        sidecar=sidecar,
        offset_root=offset_root,
        record=record,
        minimum_frames=4,
    )
    write(np.asarray([4, 20, 36, 52], dtype=np.int64))
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_offset_sidecar(
            xtc=xtc,
            sidecar=sidecar,
            offset_root=offset_root,
            record=record,
            minimum_frames=4,
        )
    shifted_record = {**record, "sidecar_sha256": sha256_file(sidecar)}
    with pytest.raises(ValueError, match="offsets are invalid"):
        _validate_offset_sidecar(
            xtc=xtc,
            sidecar=sidecar,
            offset_root=offset_root,
            record=shifted_record,
            minimum_frames=4,
        )


def test_paired_budget_accumulates_failure_resume_and_both_arms(tmp_path: Path) -> None:
    now = [100.0]
    clock = lambda: now[0]
    ledger = tmp_path / "paired.json"
    first = PairedFormalBudget.start(
        ledger,
        pair_contract_hash="pair",
        arm="M-short",
        config_sha256="short",
        code_commit="code",
        cap_gpu_hours=1.0,
        planned_updates=2,
        resume_checkpoint_sha256=None,
        clock=clock,
    )
    now[0] = 105.0
    checkpoint_state = first.checkpoint_state()
    now[0] = 110.0
    failed = first.finish(
        status="failed_or_interrupted",
        successful_updates=1,
        failure={"type": "SyntheticInterrupt", "message": "bounded test"},
    )
    assert failed["cumulative_gpu_hours"] == pytest.approx(10.0 / 3600.0)

    now[0] = 120.0
    resumed = PairedFormalBudget.start(
        ledger,
        pair_contract_hash="pair",
        arm="M-short",
        config_sha256="short",
        code_commit="code",
        cap_gpu_hours=1.0,
        planned_updates=2,
        resume_checkpoint_sha256="checkpoint",
        clock=clock,
    )
    resumed.validate_resume_state(checkpoint_state)
    now[0] = 130.0
    resumed.finish(status="completed", successful_updates=2, failure=None)

    now[0] = 140.0
    long_arm = PairedFormalBudget.start(
        ledger,
        pair_contract_hash="pair",
        arm="M-long",
        config_sha256="long",
        code_commit="code",
        cap_gpu_hours=1.0,
        planned_updates=2,
        resume_checkpoint_sha256=None,
        clock=clock,
    )
    now[0] = 150.0
    final = long_arm.finish(status="completed", successful_updates=2, failure=None)
    assert final["pair_status"] == "completed"
    assert final["cumulative_gpu_hours"] == pytest.approx(30.0 / 3600.0)
    stored = json.loads(ledger.read_text())
    assert [row["status"] for row in stored["invocations"]] == [
        "failed_or_interrupted", "completed", "completed"
    ]


def test_paired_budget_stops_entire_pair_at_cap(tmp_path: Path) -> None:
    now = [0.0]
    clock = lambda: now[0]
    ledger = tmp_path / "paired.json"
    short = PairedFormalBudget.start(
        ledger,
        pair_contract_hash="pair",
        arm="M-short",
        config_sha256="short",
        code_commit="code",
        cap_gpu_hours=0.02,
        planned_updates=1,
        resume_checkpoint_sha256=None,
        clock=clock,
    )
    now[0] = 60.0
    short.finish(status="completed", successful_updates=1, failure=None)
    long_arm = PairedFormalBudget.start(
        ledger,
        pair_contract_hash="pair",
        arm="M-long",
        config_sha256="long",
        code_commit="code",
        cap_gpu_hours=0.02,
        planned_updates=1,
        resume_checkpoint_sha256=None,
        clock=clock,
    )
    now[0] = 73.0
    assert long_arm.exhausted()
    final = long_arm.finish(
        status="budget_exhausted",
        successful_updates=0,
        failure={"type": "FormalBudgetExceeded", "message": "bounded test"},
    )
    assert final["pair_status"] == "budget_exhausted"
    with pytest.raises(RuntimeError, match="already budget_exhausted"):
        PairedFormalBudget.start(
            ledger,
            pair_contract_hash="pair",
            arm="M-long",
            config_sha256="long",
            code_commit="code",
            cap_gpu_hours=0.02,
            planned_updates=1,
            resume_checkpoint_sha256="checkpoint",
            clock=clock,
        )
