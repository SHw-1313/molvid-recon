"""Codec trainer one-step and new-format resume gates."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from data.clip_dataset import collate_clip_records as old_collate
from trainer.codec_trainer import (
    CodecTrainConfig as OldTrainConfig,
    CodecTrainer as OldTrainer,
    PVBCodecModel,
    TimeBucketSpec as OldTimeBucketSpec,
)
from molvid.checkpoints import _mapped_key
from molvid.codec.model import TrajectoryCodec
from molvid.data.batch import collate_clip_records
from molvid.losses.reconstruction import TimeBucketSpec
from molvid.training.codec import CodecTrainConfig, CodecTrainer
from test_migration import _record


@pytest.fixture
def deterministic_cuda():
    assert torch.cuda.is_available(), "P4a codec training gate requires CUDA"
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _models():
    common = dict(
        hidden_channels=8, spatial_layers=1, temporal_codec_mode="ratio4_state_detail",
        num_rbf=8, num_heads=2, coordinate_stem="centered_vector",
        freeze_frame_encoder=True,
    )
    torch.manual_seed(713)
    old = PVBCodecModel(**common, temporal_layers=1, temporal_ratio=4).cuda()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(713)
    new = TrajectoryCodec(**common).cuda()
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[_mapped_key(name)], rtol=0, atol=0)
    return old, new


def _configs():
    common = dict(
        lr=2e-4, weight_decay=0.01, max_steps=2, warmup_steps=2,
        grad_clip=1.0, device="cuda",
        loss_schedule=((0, {"coordinate": 1.0, "bond": 0.1, "velocity": 0.1}),),
    )
    old = OldTrainConfig(**common, bucket_specs=(OldTimeBucketSpec("dt_100ps", 100.0, 1.0),))
    new = CodecTrainConfig(**common, bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),))
    assert old.as_dict() == new.as_dict()
    return old, new


def _equal_optimizer(old, new):
    first = old.state_dict()
    second = new.state_dict()
    assert first["param_groups"] == second["param_groups"]
    assert first["state"].keys() == second["state"].keys()
    for parameter_id in first["state"]:
        for key, value in first["state"][parameter_id].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second["state"][parameter_id][key], rtol=0, atol=0)
            else:
                assert value == second["state"][parameter_id][key]


def test_codec_trainer_one_step_matches_old_and_preserves_frozen_encoder(deterministic_cuda):
    old_model, new_model = _models()
    old_batch = old_collate([_record()])
    new_batch = collate_clip_records([_record()])
    old_config, new_config = _configs()
    old = OldTrainer(old_model, [old_batch], config=old_config, device="cuda")
    new = CodecTrainer(new_model, [new_batch], config=new_config, device="cuda")
    old_metrics = old.optimizer_step(old_batch)
    new_metrics = new.optimizer_step(new_batch)
    assert old_metrics == new_metrics
    assert old.step == new.step == 1
    for name, value in old.model.state_dict().items():
        torch.testing.assert_close(value, new.model.state_dict()[_mapped_key(name)], rtol=0, atol=0)
    _equal_optimizer(old.optimizer, new.optimizer)
    assert all(parameter.grad is None for parameter in new.model.frame_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in new.model.frame_encoder.parameters())


def test_new_codec_checkpoint_restores_next_step_rng_and_optimizer(tmp_path, deterministic_cuda):
    random.seed(713)
    np.random.seed(713)
    torch.manual_seed(713)
    torch.cuda.manual_seed_all(713)
    _, model = _models()
    batch = collate_clip_records([_record()])
    _, config = _configs()
    trainer = CodecTrainer(model, [batch], config=config, device="cuda")
    trainer.optimizer_step(batch)
    path = trainer.save_checkpoint(tmp_path / "codec_step1.pt")
    expected_draws = (
        random.random(), float(np.random.rand()), torch.randn(2),
        torch.randn(2, device="cuda"),
    )
    expected_metrics = trainer.optimizer_step(batch)
    expected_model = copy.deepcopy(trainer.model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())

    _, fresh = _models()
    resumed = CodecTrainer(fresh, [batch], config=config, device="cuda")
    resumed.load_checkpoint(path)
    assert resumed.step == 1
    actual_draws = (
        random.random(), float(np.random.rand()), torch.randn(2),
        torch.randn(2, device="cuda"),
    )
    for expected, actual in zip(expected_draws, actual_draws):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        else:
            assert expected == actual
    actual_metrics = resumed.optimizer_step(batch)
    assert expected_metrics == actual_metrics
    for name in expected_model:
        torch.testing.assert_close(expected_model[name], resumed.model.state_dict()[name], rtol=0, atol=0)
    assert resumed.optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, values in expected_optimizer["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][parameter_id][key], rtol=0, atol=0)

def test_corrupt_optimizer_moment_fails_without_mutating_live_state(tmp_path, deterministic_cuda):
    from molvid.checkpoints import capture_rng_state

    _, model = _models()
    batch = collate_clip_records([_record()])
    _, config = _configs()
    trainer = CodecTrainer(model, [batch], config=config, device="cuda")
    trainer.optimizer_step(batch)
    checkpoint = trainer.save_checkpoint(tmp_path / "good.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_slot = next(iter(payload["optimizer_state"]["state"].values()))
    first_slot["exp_avg"] = torch.zeros(1)
    corrupt = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt)
    before_model = copy.deepcopy(trainer.model.state_dict())
    before_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    before_rng = capture_rng_state()
    before_cursor = (trainer.step, trainer.epoch, trainer.batch_in_epoch)
    with pytest.raises(ValueError, match="moment"):
        trainer.load_checkpoint(corrupt)
    assert (trainer.step, trainer.epoch, trainer.batch_in_epoch) == before_cursor
    for name, value in before_model.items():
        torch.testing.assert_close(value, trainer.model.state_dict()[name], rtol=0, atol=0)
    actual_optimizer = trainer.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == before_optimizer["param_groups"]
    for parameter_id, slot in before_optimizer["state"].items():
        for name, value in slot.items():
            torch.testing.assert_close(value, actual_optimizer["state"][parameter_id][name], rtol=0, atol=0)
    after_rng = capture_rng_state()
    assert before_rng["python"] == after_rng["python"]
    assert np.array_equal(before_rng["numpy"][1], after_rng["numpy"][1])
    torch.testing.assert_close(before_rng["torch"], after_rng["torch"], rtol=0, atol=0)
    for first, second in zip(before_rng["cuda"], after_rng["cuda"]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

def test_codec_run_restores_mid_epoch_cursor_and_next_batch(tmp_path):
    class ScaleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.9))

        def forward(self, batch):
            return batch.x * self.scale

    first_record = _record()
    second_record = _record()
    second_record["x"] = second_record["x"] * 2.0
    second_record["bpos"] = second_record["x"].copy()
    batches = [collate_clip_records([first_record]), collate_clip_records([second_record])]
    config = CodecTrainConfig(
        lr=1e-3, max_steps=2, device="cpu",
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, {"coordinate": 1.0}),),
    )
    trainer = CodecTrainer(ScaleModel(), batches, config=config)
    trainer.run(max_steps=1)
    assert (trainer.step, trainer.epoch, trainer.batch_in_epoch) == (1, 0, 1)
    path = trainer.save_checkpoint(tmp_path / "mid_epoch.pt")
    expected = trainer.run(max_steps=2)
    expected_weight = trainer.model.scale.detach().clone()
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    resumed = CodecTrainer(ScaleModel(), batches, config=config)
    resumed.load_checkpoint(path)
    assert (resumed.step, resumed.epoch, resumed.batch_in_epoch) == (1, 0, 1)
    actual = resumed.run(max_steps=2)
    assert actual == expected
    torch.testing.assert_close(resumed.model.scale, expected_weight, rtol=0, atol=0)
    assert resumed.optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, values in expected_optimizer["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][parameter_id][name], rtol=0, atol=0)
