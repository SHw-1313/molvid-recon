from __future__ import annotations

import numpy as np

from data.clip_batching import (
    ClipItemSpec,
    ClipSpecTable,
    TaskAwareClipBatchSampler,
    make_clip_dataloader,
    summarize_clip_batch,
)
from data.clip_dataset import (
    ClipMMapDataset,
    ClipMMapWriter,
    collate_clip_records,
)


def _specs():
    items = [
        ClipItemSpec(
            index=0,
            atoms=2,
            frames=1,
            task="static",
            time_bucket_id="static",
            sample_id="s0",
        ),
    ]
    items.extend(
        ClipItemSpec(
            index=index,
            atoms=3,
            frames=4,
            task="trajectory",
            time_bucket_id="dt_100ps",
            native_delta_time_ps=100.0,
            physical_clip_span_ps=300.0,
            sample_id=f"t100_{index}",
        )
        for index in range(1, 7)
    )
    items.extend(
        ClipItemSpec(
            index=index,
            atoms=3,
            frames=4,
            task="trajectory",
            time_bucket_id="dt_1ns",
            native_delta_time_ps=1000.0,
            physical_clip_span_ps=3000.0,
            sample_id=f"t1ns_{index}",
        )
        for index in range(7, 10)
    )
    return ClipSpecTable.from_specs(items)


def _record(*, sample_id, task, bucket, frames, atoms, dt):
    x = np.zeros((frames, atoms, 3), dtype=np.float32)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "task": task,
        "time_bucket_id": bucket,
        "time_ps": np.arange(frames, dtype=np.float32) * dt,
        "delta_time_ps": np.full(max(0, frames - 1), dt, dtype=np.float32),
        "x": x,
        "bpos": x.copy(),
        "atype": np.arange(atoms, dtype=np.int64),
        "btype": np.arange(atoms, dtype=np.int64),
        "block_id": np.arange(atoms, dtype=np.int64),
        "component_id": np.zeros(atoms, dtype=np.int64),
        "atom_source_index": np.arange(atoms, dtype=np.int64),
        "atom_identity": [f"{sample_id}:a{i}" for i in range(atoms)],
        "edge_mask": np.zeros(atoms, dtype=np.int64),
        "loss_mask": np.ones(atoms, dtype=np.bool_),
        "align_mask": np.asarray([True] + [False] * (atoms - 1)),
        "bond_index": np.empty((2, 0), dtype=np.int64),
    }


def test_tn_complexity_and_homogeneous_batches_are_deterministic():
    specs = _specs()
    sampler_a = TaskAwareClipBatchSampler(
        specs,
        max_tokens=24,
        batches_per_epoch=12,
        seed=17,
        replacement=True,
    )
    sampler_b = TaskAwareClipBatchSampler(
        specs,
        max_tokens=24,
        batches_per_epoch=12,
        seed=17,
        replacement=True,
    )
    batches_a = list(sampler_a)
    assert batches_a == list(sampler_b)
    assert batches_a
    by_index = {spec.index: spec for spec in specs}
    for batch in batches_a:
        keys = {by_index[index].group_key for index in batch}
        assert len(keys) == 1
        assert sum(by_index[index].effective_tokens for index in batch) <= 24


def test_sampling_weights_are_group_level_and_can_exclude_large_groups():
    sampler = TaskAwareClipBatchSampler(
        _specs(),
        max_tokens=24,
        batches_per_epoch=20,
        seed=2,
        task_weights={"static": 1.0, "trajectory": 0.0},
        replacement=True,
    )
    by_index = {spec.index: spec for spec in sampler.specs}
    assert all(by_index[index].task == 0 for batch in sampler for index in batch)

    sampler = TaskAwareClipBatchSampler(
        _specs(),
        max_tokens=24,
        batches_per_epoch=20,
        seed=2,
        time_bucket_weights={"dt_100ps": 0.0, "dt_1ns": 1.0, "static": 0.0},
        replacement=True,
    )
    by_index = {spec.index: spec for spec in sampler.specs}
    assert all(
        by_index[index].time_bucket_id == "dt_1ns"
        for batch in sampler
        for index in batch
    )


def test_ddp_shards_global_batches_instead_of_repeating_them():
    items = [
        ClipItemSpec(
            index=index,
            atoms=2,
            frames=1,
            task="static",
            time_bucket_id="static",
        )
        for index in range(12)
    ]
    specs = ClipSpecTable.from_specs(items)
    rank0 = TaskAwareClipBatchSampler(
        specs,
        max_tokens=4,
        batches_per_epoch=6,
        num_replicas=2,
        rank=0,
        seed=9,
        replacement=False,
    )
    rank1 = TaskAwareClipBatchSampler(
        specs,
        max_tokens=4,
        batches_per_epoch=6,
        num_replicas=2,
        rank=1,
        seed=9,
        replacement=False,
    )
    batches0 = {tuple(batch) for batch in rank0}
    batches1 = {tuple(batch) for batch in rank1}
    assert batches0
    assert batches1
    assert batches0.isdisjoint(batches1)
    assert batches0 | batches1 == {
        tuple(batch) for batch in rank0.global_batches
    }


def test_batch_log_contains_physical_time_and_effective_tokens():
    batch = collate_clip_records(
        [
            _record(
                sample_id="a",
                task="trajectory",
                bucket="dt_100ps",
                frames=4,
                atoms=2,
                dt=100.0,
            ),
            _record(
                sample_id="b",
                task="trajectory",
                bucket="dt_100ps",
                frames=4,
                atoms=3,
                dt=100.0,
            ),
        ]
    )
    record = summarize_clip_batch(batch)
    assert record["atoms"] == 5
    assert record["frames"] == 4
    assert record["effective_tokens"] == 20
    assert record["task_type"] == "trajectory"
    assert record["native_delta_time_ps"] == 100.0
    assert record["physical_clip_span_ps"] == 300.0
    assert record["sample_id"] == ["a", "b"]


def test_indexed_clip_specs_drive_loader(tmp_path):
    root = tmp_path / "clips"
    with ClipMMapWriter(root) as writer:
        writer.append(
            _record(
                sample_id="indexed",
                task="trajectory",
                bucket="dt_100ps",
                frames=4,
                atoms=2,
                dt=100.0,
            )
        )
    dataset = ClipMMapDataset(root)
    table = dataset.clip_spec_table()
    assert len(table) == 1
    assert table[0].effective_tokens == 8
    assert table[0].time_bucket_id == "dt_100ps"

    loader = make_clip_dataloader(
        dataset,
        max_tokens=8,
        batches_per_epoch=1,
        replacement=False,
    )
    batch = next(iter(loader))
    assert batch.sample_id == ("indexed",)
    assert summarize_clip_batch(batch)["effective_tokens"] == 8
    dataset.close()



def test_no_replacement_loader_covers_all_groups_without_silent_tail_drop():
    specs = ClipSpecTable.from_specs(
        [
            ClipItemSpec(
                index=index,
                atoms=atoms,
                frames=16,
                task="trajectory",
                time_bucket_id="dt_100ps",
            )
            for index, atoms in enumerate([887] * 13 + [1503] * 13 + [2499] * 13)
        ]
    )
    sampler = TaskAwareClipBatchSampler(
        specs,
        max_tokens=80000,
        shuffle=False,
        replacement=False,
    )
    seen = [index for batch in sampler for index in batch]
    assert seen == list(range(39))
