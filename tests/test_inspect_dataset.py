"""Read-only dataset inspection, including a sealed test split."""

from __future__ import annotations

import json

from molvid.data.manifest import build_manifest
from molvid.data.store import ClipMMapDataset, ClipMMapWriter
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file
from test_manifest import _record
from tools.inspect_dataset import inspect_split, main


def test_inspect_split_counts_actual_clips_and_marks_partial_scan(tmp_path):
    root = tmp_path / "store"
    with ClipMMapWriter(root) as writer:
        writer.append(_record("atlas_A_R1_w000000"))
        writer.append(_record("atlas_A_R1_w000001"))
    dataset = ClipMMapDataset(root)
    try:
        full = inspect_split(dataset)
        assert full["clips_available"] == full["clips_scanned"] == 2
        assert full["systems_scanned"] == full["trajectories_scanned"] == 1
        assert full["atoms_min"] == full["atoms_max"] == 2
        assert full["frames_min"] == full["frames_max"] == 2
        assert full["interval_ps_min"] == full["interval_ps_max"] == 100.0
        assert full["by_time_bucket"] == {"dt_100ps": 2}
        assert not full["partial"]
        limited = inspect_split(dataset, max_clips=1)
        assert limited["clips_scanned"] == 1 and limited["partial"]
    finally:
        dataset.close()


def test_manifest_inspection_never_opens_sealed_test_store(tmp_path, capsys):
    split_ids = {
        "train": ["atlas_train_R1_w000000"],
        "valid": ["atlas_valid_R1_w000000"],
        "test": ["atlas_test_R1_w000000"],
    }
    manifest = build_manifest(
        {split: {"selected_systems": [f"atlas_{split}"], "sample_ids": ids}
         for split, ids in split_ids.items()},
        source_root=tmp_path, frames_per_clip=2, time_bucket_id="dt_100ps",
        max_tokens=8,
    )
    atomic_write_json(tmp_path / "manifest.json", manifest)
    materialization = {"materialized": True, "splits": {}}
    for split in ("train", "valid"):
        root = tmp_path / "clip_store" / split
        with ClipMMapWriter(root) as writer:
            writer.append(_record(split_ids[split][0]))
        materialization["splits"][split] = {
            "count": 1, "index_sha256": sha256_file(root / "index.txt"),
        }
    materialization["splits"]["test"] = {"count": 1}
    materialization["materialization_sha256"] = canonical_hash(materialization)
    atomic_write_json(tmp_path / "materialization.json", materialization)
    assert main(["--manifest-root", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["splits"]) == {"train", "valid"}
    assert result["splits"]["valid"]["clips_scanned"] == 1
    assert not (tmp_path / "clip_store" / "test").exists()
