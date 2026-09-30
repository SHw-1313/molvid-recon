from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn

from molvid.runtime import sha256_file
from molvid.training.joint import FrameJointTrainer
from tools.run_gm_capacity import validate_review_gate


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
