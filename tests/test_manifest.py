from __future__ import annotations

import numpy as np
import pytest

from molvid.data.manifest import (
    build_manifest,
    check_split_overlap,
    load_datasets,
    make_validation_plan,
)
from molvid.data.store import ClipMMapWriter
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file


def _record(sample_id: str):
    x = np.zeros((2, 2, 3), dtype=np.float32)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": np.array([0.0, 100.0], dtype=np.float32),
        "delta_time_ps": np.array([100.0], dtype=np.float32),
        "x": x,
        "bpos": x.copy(),
        "atype": np.array([5, 7]),
        "btype": np.array([5, 7]),
        "block_id": np.array([0, 1]),
        "component_id": np.array([0, 0]),
        "atom_source_index": np.array([0, 1]),
        "atom_identity": [f"{sample_id}:0", f"{sample_id}:1"],
        "edge_mask": np.array([0, 0]),
        "loss_mask": np.array([True, True]),
        "align_mask": np.array([True, True]),
        "bond_index": np.array([[0, 1], [1, 0]]),
    }


def test_frozen_manifest_loads_only_train_valid_and_preserves_schedule(tmp_path):
    split_ids = {
        "train": ["atlas_train_R1_w000000"],
        "valid": ["atlas_valid_R1_w000000", "atlas_valid_R1_w000001"],
        "test": ["atlas_test_R1_w000000"],
    }
    source_splits = {
        split: {
            "selected_systems": [f"atlas_{split}"],
            "sample_ids": ids,
        }
        for split, ids in split_ids.items()
    }
    manifest = build_manifest(
        source_splits,
        source_root=tmp_path,
        frames_per_clip=2,
        time_bucket_id="dt_100ps",
        max_tokens=8,
    )
    atomic_write_json(tmp_path / "manifest.json", manifest)
    materialized = {"materialized": True, "splits": {}}
    for split in ("train", "valid"):
        root = tmp_path / "clip_store" / split
        with ClipMMapWriter(root) as writer:
            for sample_id in split_ids[split]:
                writer.append(_record(sample_id))
        materialized["splits"][split] = {
            "count": len(split_ids[split]),
            "index_sha256": sha256_file(root / "index.txt"),
        }
    materialized["splits"]["test"] = {"count": 1}
    materialized["materialization_sha256"] = canonical_hash(materialized)
    atomic_write_json(tmp_path / "materialization.json", materialized)
    assert not (tmp_path / "clip_store/test").exists()
    datasets = load_datasets(tmp_path)
    assert len(datasets.train) == 1
    assert len(datasets.valid) == 2
    plan = make_validation_plan(
        datasets.valid,
        systems=("atlas_valid",),
        windows=(0, 1),
        replicas=("R1",),
        max_tokens=8,
        seed=7,
    )
    assert len(plan.selected_sample_ids) == 2
    assert sorted(index for batch in plan.batches for index in batch) == [0, 1]
    datasets.close()
    assert not (tmp_path / "clip_store/test").exists()


def test_manifest_rejects_cross_split_system_and_sample_overlap():
    source_splits = {
        split: {"selected_systems": [split], "sample_ids": [f"{split}_sample"]}
        for split in ("train", "valid", "test")
    }
    source_splits["valid"]["selected_systems"] = ["train"]
    with pytest.raises(ValueError, match="overlap"):
        check_split_overlap(source_splits)
    source_splits["valid"]["selected_systems"] = ["valid"]
    source_splits["test"]["sample_ids"] = ["valid_sample"]
    with pytest.raises(ValueError, match="overlap"):
        check_split_overlap(source_splits)
