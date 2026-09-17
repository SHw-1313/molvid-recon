from pathlib import Path

import pytest

from data.clip_batching import ClipSpecTable
from scripts.run_state_detail_dit_pilot import (
    HISTORY_SCHEDULE,
    PILOT_SCHEMA,
    SAMPLE_ID_RE,
    VALIDATION_WINDOWS,
    _batch_schedule_hash,
    _canonical_hash,
    _make_validation_plan,
    _next_train_batch,
    _reject_test_path,
)
from trainer.dit_trainer import DiTTrainConfig


def test_pilot_contract_constants_and_hash_are_deterministic():
    assert PILOT_SCHEMA == "pvb.dit.state_detail.t1_pilot.v1"
    assert HISTORY_SCHEDULE == (4, 8)
    assert VALIDATION_WINDOWS == (0, 30, 61)
    assert _canonical_hash({"b": 2, "a": 1}) == _canonical_hash({"a": 1, "b": 2})


def test_pilot_rejects_test_paths():
    _reject_test_path(Path("/data/clip_store/train"), "train")
    with pytest.raises(RuntimeError, match="test split"):
        _reject_test_path(Path("/data/clip_store/test"), "store")


def test_validation_plan_contains_all_systems_replicas_and_fixed_windows():
    systems = [f"atlas_{index:04d}_A" for index in range(8)]
    rows = []
    for system in systems:
        for replica in ("R1", "R2", "R3"):
            for window in range(62):
                rows.append((f"{system}_{replica}_w{window:06d}", 0, 1, 4, 16, "dt_100ps"))

    class FakeDataset:
        def __init__(self, values):
            self._index = values

        def __len__(self):
            return len(self._index)

        def __getitem__(self, index):
            raise AssertionError("the validation-plan test must not open clip payloads")

        def clip_spec_table(self):
            return ClipSpecTable.from_clip_index(self._index)

    plan = _make_validation_plan(FakeDataset(rows))
    assert plan.windows == VALIDATION_WINDOWS
    assert len(plan.systems) == 8
    assert len(plan.selected_sample_ids) == 8 * 3 * len(VALIDATION_WINDOWS)
    assert all(SAMPLE_ID_RE.match(value) for value in plan.selected_sample_ids)
    assert plan.batches


def test_pilot_config_allows_profile_budget_but_probe_config_stays_bounded():
    DiTTrainConfig(
        ratio=2,
        mode="ratio2_state_detail",
        max_steps=200,
        metadata={"phase": "t1_pilot"},
    ).validate()
    with pytest.raises(ValueError, match="at most 100"):
        DiTTrainConfig(
            ratio=2,
            mode="ratio2_state_detail",
            max_steps=101,
        ).validate()


def test_consumed_training_schedule_hash_matches_contract_hash():
    class FakeSampler:
        def __init__(self):
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        @property
        def selected_sample_ids(self):
            return (f"trajectory_w{self.epoch:06d}",)

        @property
        def global_batches(self):
            return ((self.epoch, self.epoch + 1),)

        def state_dict(self):
            return {"epoch": self.epoch, "schema_version": "test"}

    sampler = FakeSampler()
    expected = _batch_schedule_hash(sampler, epoch=0)
    indices, consumed = _next_train_batch(None, sampler, {"epoch": 0, "batch_index": 0})
    assert indices == (0, 1)
    assert consumed == expected
