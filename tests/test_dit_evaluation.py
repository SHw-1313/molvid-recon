from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from data.clip_dataset import TRAJECTORY_TASK
from evaluation.dit_evaluation import evaluate_oracle_vs_generated, trajectory_metrics


def _batch(frames: int = 8, *, ragged: bool = False) -> dict[str, torch.Tensor]:
    batch_size = 2 if ragged else 1
    atoms = 4 if ragged else 3
    abid = torch.tensor([0, 0, 1, 1], dtype=torch.long) if ragged else torch.zeros(atoms, dtype=torch.long)
    frame_mask = torch.ones(batch_size, frames, dtype=torch.bool)
    if ragged:
        frame_mask[1, 6:] = False
    return {
        "frame_mask": frame_mask,
        "loss_mask": torch.tensor([True, False, True, True], dtype=torch.bool)
        if ragged
        else torch.ones(atoms, dtype=torch.bool),
        "align_mask": torch.ones(atoms, dtype=torch.bool),
        "abid": abid,
        "bond_index": torch.empty(2, 0, dtype=torch.long),
        "delta_time_ps": torch.ones(batch_size, frames - 1),
        "time_ps": torch.arange(frames, dtype=torch.float32).expand(batch_size, -1).clone(),
        "task": torch.full((batch_size,), TRAJECTORY_TASK, dtype=torch.long),
    }


def test_future_metrics_ignore_observed_errors_and_use_declared_intervals() -> None:
    torch.manual_seed(101)
    batch = _batch()
    target = torch.randn(8, 3, 3)
    observed_error = target.clone()
    observed_error[:4, 0, 0] += 1000.0
    clean = trajectory_metrics(target, target, batch, history_frames=4)
    observed_only = trajectory_metrics(observed_error, target, batch, history_frames=4)
    assert observed_only["future"]["rmsd"] == clean["future"]["rmsd"]
    assert observed_only["temporal"]["future"]["rmsf"] == clean["temporal"]["future"]["rmsf"]
    assert observed_only["full_diagnostic"] == observed_only["temporal"]["full_diagnostic"]
    assert observed_only["evaluation_protocol"]["observed_frame_interval"] == [0, 4]
    assert observed_only["evaluation_protocol"]["future_frame_interval"] == [4, 8]
    assert observed_only["evaluation_protocol"]["boundary_frame_interval"] == [3, 4]


def test_future_errors_change_future_metrics_and_h4_h8_select_different_frames() -> None:
    batch = _batch()
    target = torch.zeros(8, 3, 3)
    future_bad = target.clone()
    future_bad[4, 0, 0] = 10.0
    h4 = trajectory_metrics(future_bad, target, batch, history_frames=4)
    h8 = trajectory_metrics(future_bad, target, batch, history_frames=8)
    assert h4["future"]["rmsd"] > 0.0
    assert h8["future"]["rmsd"] == 0.0
    assert h4["evaluation_protocol"]["future_frame_interval"] == [4, 8]
    assert h8["evaluation_protocol"]["future_frame_interval"] == [8, 8]
    assert h4["boundary"]["predicted_step_magnitude"] is not None
    assert h4["boundary"]["target_step_magnitude"] == 0.0
    assert h4["boundary"]["step_difference"] > 0.0


def test_ragged_frame_and_loss_masks_exclude_invalid_future_errors() -> None:
    batch = _batch(ragged=True)
    target = torch.zeros(8, 4, 3)
    changed = target.clone()
    changed[5, 1, 1] = 1000.0
    changed[7, 2, 1] = 1000.0
    clean = trajectory_metrics(target, target, batch, history_frames=4)
    result = trajectory_metrics(changed, target, batch, history_frames=4)
    assert result["future"]["rmsd"] == clean["future"]["rmsd"]
    valid_changed = changed.clone()
    valid_changed[5, 0, 1] = 3.0
    changed_result = trajectory_metrics(valid_changed, target, batch, history_frames=4)
    assert changed_result["future"]["rmsd"] > result["future"]["rmsd"]


def test_h0_has_all_frames_as_future_and_no_boundary_transition() -> None:
    batch = _batch()
    target = torch.zeros(8, 3, 3)
    result = trajectory_metrics(target, target, batch, history_frames=0)
    assert result["evaluation_protocol"]["observed_frame_interval"] == [0, 0]
    assert result["evaluation_protocol"]["future_frame_interval"] == [0, 8]
    assert result["boundary"]["available"] is False
    assert result["boundary"]["frame_interval"] is None


def test_history_policy_rejects_partial_block_h2() -> None:
    batch = _batch()
    target = torch.zeros(8, 3, 3)
    with pytest.raises(ValueError, match="H=0, H=4, or H=8"):
        trajectory_metrics(target, target, batch, history_frames=2)


def test_oracle_and_generated_evaluation_thread_history_frames() -> None:
    batch = _batch()
    target = torch.zeros(8, 3, 3)

    class IdentityCodec:
        def eval(self):
            return self

        def decode(self, latent):
            return latent

    result = evaluate_oracle_vs_generated(
        IdentityCodec(),
        oracle_latent=target,
        generated_latent=target,
        batch=SimpleNamespace(**{**batch, "x": target}),
        history_frames=4,
    )
    assert result.codec_oracle["evaluation_protocol"]["future_frame_interval"] == [4, 8]
    assert result.generated_result["rmsf"]["future"] == result.codec_oracle["rmsf"]["future"]
