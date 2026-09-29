"""Metadata-only checks for the independent static source schedule."""

from __future__ import annotations

import pytest

from molvid.data.sampling import ClipItemSpec, ClipSpecTable
from molvid.training.static_schedule import (
    StaticBatchSchedule,
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
