from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from molvid.codec.frame import FrozenFrameTeacher
from molvid.data.batch import ClipValidationError, collate_clip_records, validate_clip_record
from molvid.evaluation.temporal import system_mean_rmsf_summary, temporal_metrics_v3
from molvid.geometry.frames import pack_frame_nodes


def _record(*, frames: int = 8, valid_frames: int | None = None) -> dict:
    atoms = 4
    valid = frames if valid_frames is None else int(valid_frames)
    coordinates = np.zeros((frames, atoms, 3), dtype=np.float32)
    coordinates[:, :, 0] = np.arange(atoms, dtype=np.float32)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": "fixed",
        "source": "test",
        "split": "valid",
        "task": "trajectory",
        "coordinate_unit": "angstrom",
        "time_bucket_id": "fixed_history_dt_100ps",
        "time_ps": np.arange(frames, dtype=np.float32) * 100.0,
        "delta_time_ps": np.full(frames - 1, 100.0, dtype=np.float32),
        "frame_mask": np.arange(frames) < valid,
        "x": coordinates,
        "bpos": coordinates.copy(),
        "atype": np.ones(atoms, dtype=np.int64),
        "btype": np.ones(atoms, dtype=np.int64),
        "block_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "component_id": np.zeros(atoms, dtype=np.int64),
        "atom_source_index": np.arange(atoms, dtype=np.int64),
        "atom_identity": [f"a{index}" for index in range(atoms)],
        "edge_mask": np.zeros(atoms, dtype=np.int64),
        "loss_mask": np.ones(atoms, dtype=np.bool_),
        "align_mask": np.asarray([True, True, True, False]),
        "bond_index": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
    }


def test_frame_mask_is_a_serialized_valid_prefix() -> None:
    record = validate_clip_record(_record(valid_frames=6))
    assert record["frame_mask"].tolist() == [True] * 6 + [False] * 2
    invalid = _record(valid_frames=6)
    invalid["frame_mask"] = np.asarray([True, False, True, False, False, False, False, False])
    with pytest.raises(ClipValidationError, match="valid prefix"):
        validate_clip_record(invalid)


def test_padded_frames_do_not_enter_frame_graph_nodes() -> None:
    batch = collate_clip_records([_record(valid_frames=6)])
    nodes = pack_frame_nodes(batch)
    assert nodes.num_nodes == 6 * batch.atom_count
    assert nodes.dense_index.tolist() == list(range(6 * batch.atom_count))
    assert torch.all(nodes.dense_to_compact[6 * batch.atom_count :] == -1)


def test_frozen_teacher_zeroes_partial_and_padding_only_chunks() -> None:
    class Encoder(nn.Module):
        def forward(self, batch):
            shape = (batch.frames, batch.atom_count, 2)
            return SimpleNamespace(
                h=batch.x.new_ones(shape),
                v=batch.x.new_ones((*shape[:2], 3, shape[-1])),
            )

    class Stem(nn.Module):
        def forward(self, coordinates):
            return coordinates.new_ones((*coordinates.shape, 2))

    batch = collate_clip_records([_record(frames=16, valid_frames=7)])
    teacher = FrozenFrameTeacher(Encoder(), Stem())
    latent, _origin = teacher(batch)
    assert torch.all(latent.h[:7] == 1)
    assert torch.all(latent.v[:7] == 2)
    assert torch.all(latent.h[7:] == 0)
    assert torch.all(latent.v[7:] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_true_dt_and_padding_control_increment_units_on_cuda() -> None:
    device = torch.device("cuda")
    batch = collate_clip_records([_record(valid_frames=6)]).to(device)
    target = batch.x.clone()
    prediction = target.clone()
    # The first three atoms define the rigid reference; the fourth moves by
    # exactly 1 Å per valid 100 ps query interval.
    prediction[4, 3, 1] = 1.0
    prediction[5, 3, 1] = 2.0
    prediction[6:, 3, 1] = 1.0e6  # invalid padding must not contribute
    result = temporal_metrics_v3(prediction, target, batch, history_frames=4).metrics
    increments = result["increments"]
    assert increments["intervals_ps"] == [100.0]
    # One of four evaluated atoms has a 1 A error, so the Euclidean atomwise
    # RMSE is 0.5 A; the true 100 ps interval scales it to 0.005 A/ps.
    assert increments["displacement_increment_rmse_A"] == pytest.approx(0.5, abs=2.0e-6)
    assert increments["finite_difference_velocity_rmse_A_per_ps"] == pytest.approx(0.005, abs=2.0e-8)
    assert increments["valid_atom_increments"] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_common_observed_reference_removes_only_rigid_motion_on_cuda() -> None:
    device = torch.device("cuda")
    batch = collate_clip_records([_record(valid_frames=8)]).to(device)
    target = batch.x.clone()
    target[4:, 3, 1] = torch.arange(1, 5, device=device, dtype=torch.float32)
    prediction = target.clone()
    for frame in range(batch.frames):
        angle = 0.11 * frame
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        prediction[frame] = prediction[frame] @ rotation + torch.tensor([5.0, -2.0, 3.0], device=device)
    result = temporal_metrics_v3(prediction, target, batch, history_frames=4).metrics
    assert result["alignment"]["prediction_fit_target_future"] is False
    assert result["increments"]["displacement_increment_rmse_A"] < 2.0e-5
    assert result["msd"]["curve_mae_A2"] < 2.0e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_motion_is_valid_rmsf_but_correlation_is_unavailable() -> None:
    device = torch.device("cuda")
    batch = collate_clip_records([_record(valid_frames=8)]).to(device)
    result = temporal_metrics_v3(batch.x, batch.x, batch, history_frames=4).metrics
    rmsf = result["rmsf"]
    assert rmsf["available"] is True
    assert rmsf["system_mean_prediction_A"] == 0.0
    assert rmsf["system_mean_target_A"] == 0.0
    assert rmsf["atom_profile"]["pearson"]["available"] is False
    assert rmsf["atom_profile"]["pearson"]["reason"] == "zero_variance"


def test_system_spread_is_distinct_from_mean_amplitude() -> None:
    summary = system_mean_rmsf_summary([
        {"system_mean_prediction_A": 1.0, "system_mean_target_A": 1.0},
        {"system_mean_prediction_A": 3.0, "system_mean_target_A": 2.0},
        {"system_mean_prediction_A": 5.0, "system_mean_target_A": 3.0},
    ])
    assert summary["pearson"]["value"] == pytest.approx(1.0)
    assert summary["spearman"]["value"] == pytest.approx(1.0)
    assert summary["spread_ratio_prediction_over_target"] == pytest.approx(2.0)
