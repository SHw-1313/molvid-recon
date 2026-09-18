"""Reconstruction loss extraction against the current codec objective."""

from __future__ import annotations

import pytest
import torch

from data.clip_dataset import collate_clip_records as old_collate
from trainer.codec_losses import (
    compute_codec_losses as old_compute,
    fit_time_bucket_normalization as old_fit,
)
from trainer.codec_trainer import TimeBucketSpec as OldTimeBucketSpec
from molvid.data.batch import collate_clip_records
from molvid.losses.reconstruction import (
    TimeBucketSpec,
    acceleration_loss,
    compute_codec_losses,
    fit_time_bucket_normalization,
    velocity_loss,
)
from test_codec_training import _record


@pytest.mark.parametrize("task", ["trajectory", "static"])
def test_reconstruction_loss_and_gradient_match_old(task):
    assert torch.cuda.is_available(), "P4a loss parity requires CUDA"
    times = torch.tensor([0.0]) if task == "static" else torch.tensor([0.0, 1.0, 3.0, 6.0])
    record = _record(task=task, times=times)
    old_batch = old_collate([record]).to("cuda")
    new_batch = collate_clip_records([record]).to("cuda")
    old_pred = old_batch.x.detach().clone().requires_grad_()
    new_pred = new_batch.x.detach().clone().requires_grad_()
    old_pred.data[0, 0, 0] += 0.5
    new_pred.data[0, 0, 0] += 0.5
    if task == "trajectory":
        old_pred.data[2, 1, 1] -= 0.25
        new_pred.data[2, 1, 1] -= 0.25
    weights = {"coordinate": 1.0, "local": 0.3, "bond": 0.4, "velocity": 0.5, "acceleration": 0.6}
    old_stats = old_fit([old_batch], min_count=1, epsilon=1e-4)
    new_stats = fit_time_bucket_normalization([new_batch], min_count=1, epsilon=1e-4)
    assert old_stats.keys() == new_stats.keys()
    for bucket in old_stats:
        assert old_stats[bucket].as_dict() == new_stats[bucket].as_dict()
    expected = old_compute(old_pred, old_batch, weights=weights, normalization=old_stats)
    actual = compute_codec_losses(new_pred, new_batch, weights=weights, normalization=new_stats)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    expected["total"].backward()
    actual["total"].backward()
    torch.testing.assert_close(old_pred.grad, new_pred.grad, rtol=0, atol=0)
    if task == "static":
        assert velocity_loss(new_pred, new_batch.x, new_batch).item() == 0
        assert acceleration_loss(new_pred, new_batch.x, new_batch).item() == 0


def test_loss_mask_and_normalization_fallback_match_old():
    record = _record()
    old_batch = old_collate([record])
    new_batch = collate_clip_records([record])
    old_batch.loss_mask[1] = False
    new_batch.loss_mask[1] = False
    old_stats = old_fit([old_batch], min_count=10_000, epsilon=1e-4)
    new_stats = fit_time_bucket_normalization([new_batch], min_count=10_000, epsilon=1e-4)
    assert old_stats["dt_100ps"].as_dict() == new_stats["dt_100ps"].as_dict()
    old_prediction = old_batch.x.clone()
    new_prediction = new_batch.x.clone()
    old_prediction[:, 1] += 7.0
    new_prediction[:, 1] += 7.0
    expected = old_compute(old_prediction, old_batch, normalization=old_stats)
    actual = compute_codec_losses(new_prediction, new_batch, normalization=new_stats)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    assert actual["total"].item() == 0.0


def test_time_bucket_spec_contract_matches_old():
    old = OldTimeBucketSpec("dt_100ps", 100.0, 1.0, 0.75)
    new = TimeBucketSpec("dt_100ps", 100.0, 1.0, 0.75)
    assert old.as_dict() == new.as_dict()
    with pytest.raises(ValueError, match="non-negative"):
        TimeBucketSpec("invalid", -1.0, 1.0)
