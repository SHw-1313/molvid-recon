"""Historical full-dataset sampler cursor against the new sampler implementation."""

from __future__ import annotations

from pathlib import Path

import torch

from data.clip_dataset import ClipMMapDataset as OldDataset
from molvid.data.store import ClipMMapDataset as NewDataset
from molvid.data.sampling import TrajectoryCappedBatchSampler as NewSampler
from molvid.runtime import canonical_hash, sha256_file
from scripts.run_state_detail_codec_v2_t1 import TrajectoryCappedBatchSampler as OldSampler
from test_migration import HISTORICAL_DIT, HISTORICAL_DIT_SHA


MANIFEST = Path(
    "/workspace/molvid-dit-capacity-data-v1/outputs/"
    "dit_capacity_data_v1/data_manifest_20260914"
)


def _next(sampler, cursor):
    while True:
        sampler.set_epoch(cursor["epoch"])
        batches = sampler.global_batches
        if cursor["batch_index"] < len(batches):
            result = batches[cursor["batch_index"]]
            cursor["batch_index"] += 1
            return result
        cursor["epoch"] += 1
        cursor["batch_index"] = 0


def test_historical_cursor_next_real_batches_match_new_sampler():
    assert MANIFEST.is_dir(), "frozen train/validation manifest is required"
    assert sha256_file(HISTORICAL_DIT) == HISTORICAL_DIT_SHA
    payload = torch.load(HISTORICAL_DIT, map_location="cpu", weights_only=False)
    assert payload["config"]["data_hash"]
    saved = payload["sequential"]
    assert saved["test_payload_opened"] is False
    assert saved["cursor"] == {"epoch": 2, "batch_index": 1336}
    parent_path = Path(saved["parent_checkpoint"])
    assert sha256_file(parent_path) == saved["parent_checkpoint_sha256"]
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    assert parent["capacity"]["schedule_hash"] == saved["schedule_hash"]
    assert parent["capacity"]["cursor"] == saved["cursor"]
    assert parent["capacity"]["data_hash"] != payload["config"]["data_hash"]
    old_train = OldDataset(MANIFEST / "clip_store/train48")
    new_train = NewDataset(MANIFEST / "clip_store/train48")
    try:
        assert tuple(row[0] for row in old_train._index) == tuple(
            row[0] for row in new_train._index
        )
        kwargs = dict(max_tokens=80000, clips_per_trajectory=24, seed=20260914, shuffle=False)
        old = OldSampler(old_train, **kwargs)
        new = NewSampler(new_train, **kwargs)
        epoch = int(saved["cursor"]["epoch"])
        old.set_epoch(epoch)
        new.set_epoch(epoch)
        assert old.state_dict() == new.state_dict()
        assert old.global_batches == new.global_batches
        expected_hash = canonical_hash({
            "schema": "pvb.dit.capacity_data.schedule.v1",
            "epoch": epoch,
            "batches": [list(batch) for batch in old.global_batches],
            "selected_sample_ids": list(old.selected_sample_ids),
        })
        assert expected_hash == saved["schedule_hash"]
        old_cursor = dict(saved["cursor"])
        new_cursor = dict(saved["cursor"])
        for _ in range(3):
            assert _next(old, old_cursor) == _next(new, new_cursor)
        assert old_cursor == new_cursor
    finally:
        old_train.close()
        new_train.close()
