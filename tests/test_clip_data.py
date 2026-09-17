from __future__ import annotations

import numpy as np
import pytest

from data.clip_dataset import (
    ClipMMapDataset,
    ClipMMapWriter,
    ClipValidationError,
    StaticClipDataset,
    canonical_time_fields,
    collate_clip_records,
    legacy_static_record_to_clip,
    validate_clip_record,
)
from data.mmap_dataset import create_mmap


def _record(
    *,
    sample_id: str = "sample",
    atoms: int = 4,
    frames: int = 4,
    bucket: str = "dt_100ps",
    task: str = "trajectory",
    dt: float = 100,
):
    x = np.arange(frames * atoms * 3, dtype=np.float32).reshape(frames, atoms, 3)
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
        "atom_identity": [f"A:{i}" for i in range(atoms)],
        "edge_mask": np.zeros(atoms, dtype=np.int64),
        "loss_mask": np.ones(atoms, dtype=np.bool_),
        "align_mask": np.asarray([True] + [False] * (atoms - 1)),
        "bond_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    }


def test_irregular_time_grid_is_canonical():
    time, delta = canonical_time_fields([0, 80, 250, 500])
    np.testing.assert_allclose(time, [0, 80, 250, 500])
    np.testing.assert_allclose(delta, [80, 170, 250])


def test_time_grid_rejects_mismatch():
    with pytest.raises(ClipValidationError, match="disagrees"):
        canonical_time_fields([0, 100, 200], [100, 101])


def test_collate_packs_atoms_and_offsets_bonds():
    batch = collate_clip_records([_record(atoms=3), _record(sample_id="b", atoms=5)])
    assert tuple(batch.x.shape) == (4, 8, 3)
    assert batch.atom_ptr.tolist() == [0, 3, 8]
    assert batch.abid.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    np.testing.assert_array_equal(
        batch.bond_index.numpy(), [[0, 1, 3, 4], [1, 0, 4, 3]]
    )


def test_collate_rejects_mixed_bucket_and_task():
    with pytest.raises(ClipValidationError, match="buckets"):
        collate_clip_records([_record(), _record(sample_id="b", bucket="dt_80ps")])
    with pytest.raises(ClipValidationError, match="task"):
        collate_clip_records([_record(), _record(sample_id="b", task="static", frames=1)])


def test_writer_round_trip(tmp_path):
    root = tmp_path / "store"
    with ClipMMapWriter(root) as writer:
        writer.append(_record())
    dataset = ClipMMapDataset(root)
    loaded = dataset[0]
    validate_clip_record(loaded)
    assert isinstance(loaded["x"], np.ndarray)
    assert isinstance(loaded["bond_index"], np.ndarray)
    assert loaded["sample_id"] == "sample"
    assert len(dataset) == 1
    dataset.close()


def test_pair_record_is_not_upgraded():
    record = _record()
    record["x0"] = record["x"][0]
    with pytest.raises(ClipValidationError, match="pair records"):
        validate_clip_record(record)


def test_static_clip_is_single_frame_and_uses_explicit_bucket():
    record = _record(frames=1, bucket="dt_1ns", task="static")
    validate_clip_record(record)
    assert record["x"].shape == (1, 4, 3)
    assert record["time_ps"].tolist() == [0.0]
    assert record["delta_time_ps"].size == 0


def test_trajectory_clip_has_fixed_sixteen_frame_shape():
    record = _record(frames=16)
    validate_clip_record(record)
    assert record["x"].shape == (16, 4, 3)
    assert record["time_ps"].shape == (16,)
    assert record["delta_time_ps"].shape == (15,)


def test_one_ns_trajectory_fixture_keeps_physical_timestamps():
    record = _record(frames=16, bucket="dt_1ns", dt=1000)
    validate_clip_record(record)
    np.testing.assert_allclose(record["time_ps"], np.arange(16) * 1000)
    np.testing.assert_allclose(record["delta_time_ps"], np.full(15, 1000))


def test_legacy_static_record_maps_to_one_frame_without_replication():
    x0 = np.arange(12, dtype=np.float32).reshape(4, 3)
    b0 = np.asarray([x0[0], x0[0], x0[2], x0[2]], dtype=np.float32)
    record = legacy_static_record_to_clip(
        {
            "x0": x0.tolist(),
            "b0": b0.tolist(),
            "atype": [1, 2, 3, 4],
            "btype": [7, 7, 8, 8],
            "edge_mask": [0, 0, 1, 1],
            "mask": [True, True, False, True],
            "bond_index": [[0, 1], [1, 0]],
        },
        sample_id="static_test_0",
        source="pdbbind",
        split="train",
        legacy_id="mol_0",
    )
    validate_clip_record(record)
    assert record["task"] == 0
    assert record["x"].shape == (1, 4, 3)
    assert record["bpos"].shape == (1, 4, 3)
    assert record["delta_time_ps"].shape == (0,)
    assert record["block_id"].tolist() == [0, 0, 1, 1]
    assert record["component_id"].tolist() == [0, 0, 1, 1]
    assert record["loss_mask"].tolist() == [True, True, False, True]


def test_static_dataset_wraps_existing_mmap_store(tmp_path):
    root = tmp_path / "legacy"
    x0 = np.arange(9, dtype=np.float32).reshape(3, 3).tolist()
    create_mmap(
        iter(
            [
                (
                    "mol_0",
                    {
                        "x0": x0,
                        "b0": x0,
                        "atype": [1, 2, 3],
                        "btype": [4, 5, 6],
                        "bond_index": [[0, 1], [1, 0]],
                    },
                    [3],
                )
            ]
        ),
        str(root),
    )
    dataset = StaticClipDataset(root, source="ani1x", split="train")
    clip = dataset[0]
    assert len(dataset) == 1
    assert clip["sample_id"] == "static_ani1x_train_00000000"
    assert clip["x"].shape == (1, 3, 3)
    batch = dataset.collate_fn([clip])
    assert tuple(batch.x.shape) == (1, 3, 3)
    assert batch.task.tolist() == [0]
