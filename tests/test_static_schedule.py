"""Metadata-only checks for the independent static source schedule."""

from __future__ import annotations

import pytest
import torch

from molvid.data.sampling import ClipItemSpec, ClipSpecTable
from molvid.training.static_schedule import (
    StaticBatchSchedule,
    external_static_pool,
    internal_md_static_pool,
)


class _SpecDataset:
    def __init__(self) -> None:
        specs = []
        for system in ("alpha", "beta"):
            for replica in ("R1", "R2", "R3"):
                for window in range(2):
                    index = len(specs)
                    specs.append(
                        ClipItemSpec(
                            index=index,
                            atoms=100 + index,
                            frames=16,
                            task="trajectory",
                            time_bucket_id="dt_100ps",
                            sample_id=f"atlas_{system}_{replica}_w{window:03d}",
                        )
                    )
        self._specs = ClipSpecTable.from_specs(specs)

    def clip_spec_table(self) -> ClipSpecTable:
        return self._specs


def test_internal_pool_selects_one_midpoint_per_complete_trajectory() -> None:
    entries = internal_md_static_pool(_SpecDataset())
    assert len(entries) == 6
    assert {(entry.system_id, entry.replica) for entry in entries} == {
        (system, replica)
        for system in ("alpha", "beta")
        for replica in ("R1", "R2", "R3")
    }
    assert all(entry.frame_index == 8 for entry in entries)
    assert all(entry.source_sample_id.endswith("w000") for entry in entries)


def test_static_schedule_commits_only_success_and_resumes_exactly() -> None:
    entries = internal_md_static_pool(_SpecDataset())
    schedule = StaticBatchSchedule(entries, seed=17, max_atoms_per_update=250)
    first = schedule.next_entries()
    with pytest.raises(RuntimeError, match="uncommitted"):
        schedule.next_entries()
    with pytest.raises(RuntimeError, match="cannot checkpoint"):
        schedule.state_dict()
    with pytest.raises(RuntimeError, match="different source"):
        schedule.record_success(tuple(reversed(first)))
    schedule.record_success(first)
    state = schedule.state_dict()
    expected_next = schedule.next_entries()

    restored = StaticBatchSchedule(entries, seed=17, max_atoms_per_update=250)
    restored.load_state_dict(state)
    assert restored.next_entries() == expected_next
    with pytest.raises(ValueError, match="pool_hash"):
        changed = dict(state)
        changed["pool_hash"] = "different"
        StaticBatchSchedule(entries, seed=17, max_atoms_per_update=250).load_state_dict(
            changed
        )


def test_external_pool_opens_only_the_preselected_records() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.opened = []

        def __getitem__(self, index: int):
            self.opened.append(index)
            return {
                "x": torch.zeros((1, 10 + index % 3, 3)),
                "sample_id": f"static_pdb_train_{index:08d}",
                "legacy_id": f"pdb_{index:08d}",
            }

    dataset = Dataset()
    indices = tuple(range(2000))
    entries = external_static_pool(dataset, indices)
    assert dataset.opened == list(indices)
    assert len(entries) == 2000
    assert entries[0].source_kind == "external_static"
    assert entries[0].system_id == "pdb_00000000"

    with pytest.raises(ValueError, match="duplicate"):
        external_static_pool(dataset, (0,) * 2000)
