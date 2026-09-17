from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.dit_reassessment import (
    aggregate_generated_rows,
    reassessment_seed,
    reassessment_trajectory_metrics,
    reaggregate_legacy_jsonl,
)


def _batch(*, atoms: int = 4, systems: int = 1, frames: int = 16) -> SimpleNamespace:
    if systems == 1:
        abid = torch.zeros(atoms, dtype=torch.long)
    else:
        abid = torch.repeat_interleave(torch.arange(systems), atoms)
    total_atoms = int(abid.numel())
    return SimpleNamespace(
        frame_mask=torch.ones(systems, frames, dtype=torch.bool),
        abid=abid,
        loss_mask=torch.ones(total_atoms, dtype=torch.bool),
        align_mask=torch.ones(total_atoms, dtype=torch.bool),
        bond_index=torch.empty(2, 0, dtype=torch.long),
        time_ps=torch.arange(frames, dtype=torch.float32).repeat(systems, 1),
        delta_time_ps=torch.ones(systems, frames - 1, dtype=torch.float32),
    )


def _coordinates(frames: int = 16) -> torch.Tensor:
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return base.unsqueeze(0).repeat(frames, 1, 1)


def test_reassessment_seed_excludes_arm_steps_batch_and_checkpoint() -> None:
    first = reassessment_seed(17, "sys_R1_w000030", 8, 2)
    assert first == reassessment_seed(17, "sys_R1_w000030", 8, 2)
    assert first != reassessment_seed(17, "sys_R1_w000030", 4, 2)
    assert first != reassessment_seed(17, "sys_R1_w000030", 8, 3)


def test_legacy_main_rows_are_not_split_into_system_rows(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    arm_root = root / "gaussian"
    arm_root.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"small-test-checkpoint")
    rows = [
        {
            "sample_id": ["sys_R1_w000030", "other_R1_w000030"],
            "history_frames": 4,
            "steps": 16,
            "draw_id": 0,
            "generation": {"seed": 1},
            "diagnostic_metrics": {"future": {"aligned_rmsd": 2.0}},
        },
        {
            "sample_id": "sys_R1_w000030",
            "history_frames": 4,
            "steps": 16,
            "draw_id": 0,
            "generation": {"seed": 1},
            "diagnostic_metrics": {"future": {"aligned_rmsd": 3.0}},
        },
    ]
    (arm_root / "generation_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = reaggregate_legacy_jsonl(
        root,
        checkpoint_paths={"gaussian": checkpoint},
    )
    groups = result["arms"]["gaussian"]["groups"]
    main = next(group for group in groups if group["row_type"] == "main_batch")
    subset = next(group for group in groups if group["row_type"] == "subset_clip")
    assert main["system_count"] is None
    assert main["aggregation"] == "batch_row_equal; system_equal_not_available"
    assert subset["system_count"] == 1


def test_contact_occupancy_is_time_averaged_not_frame_disagreement() -> None:
    batch = _batch()
    target = _coordinates()
    prediction = target.clone()
    # The first nonbond pair changes contact state in disjoint future frames,
    # while both trajectories have the same time-averaged occupancy.
    for frame in range(4, 8):
        target[frame, 1, 0] = 1.0
        prediction[frame, 1, 0] = 10.0
    for frame in range(8, 12):
        target[frame, 1, 0] = 10.0
        prediction[frame, 1, 0] = 1.0
    metrics = reassessment_trajectory_metrics(prediction, target, batch, 4)
    future = metrics["future"]
    assert future["contact_occupancy_mae"] == 0.0
    assert future["contact_frame_disagreement_legacy"] > 0.0


def test_degenerate_velocity_acf_and_torsion_are_null_with_reasons() -> None:
    batch = _batch()
    coordinates = _coordinates()
    metrics = reassessment_trajectory_metrics(coordinates, coordinates, batch, 4)
    velocity = metrics["future"]["velocity"]
    assert velocity["velocity_lag1_acf"]["prediction"] is None
    assert "constant" in velocity["velocity_lag1_acf"]["prediction_reason"]
    assert metrics["torsion"]["value"] is None
    assert metrics["torsion"]["reason"] == "torsion_index is not present"


def test_drmsd_excludes_cross_system_pairs() -> None:
    single = _coordinates()
    target = torch.cat((single, single + torch.tensor([100.0, 0.0, 0.0])), dim=1)
    prediction = target.clone()
    prediction[:, 4:] += torch.tensor([0.0, 50.0, 0.0])
    batch = _batch(atoms=4, systems=2)
    metrics = reassessment_trajectory_metrics(prediction, target, batch, 4)
    assert metrics["future"]["drmsd"] == 0.0


def test_generated_aggregation_is_draw_clip_then_system_equal() -> None:
    def row(system: str, sample_id: str, draw: int, value: float) -> dict:
        return {
            "system": system,
            "sample_id": sample_id,
            "draw": draw,
            "metrics": {"future": {"aligned_rmsd": value}},
        }

    result = aggregate_generated_rows(
        [
            row("A", "A_R1_w000030", 0, 1.0),
            row("A", "A_R1_w000030", 1, 1.0),
            row("A", "A_R1_w000031", 0, 3.0),
            row("A", "A_R1_w000031", 1, 3.0),
            row("B", "B_R1_w000030", 0, 10.0),
        ]
    )
    assert result["system_count"] == 2
    assert result["system_equal"]["aligned_rmsd"] == 6.0
