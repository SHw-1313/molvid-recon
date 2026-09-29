"""Deterministic paired execution schedule for B learned-memory arms."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ..data.sampling import ClipSpecTable
from ..runtime import canonical_hash


SCHEMA = "molvid.learned_memory.execution.v1"
OBSERVED_DT_PS = (100, 200, 400)
QUERY_LAG_PS = (100, 200, 400)
LONG_SPANS_PS = (800, 3200, 12800)


@dataclass(frozen=True)
class LearnedMemoryExecutionEntry:
    global_batch_index: int
    indices: tuple[int, ...]
    history_span_ps: int
    observed_dt_ps: int
    query_lag_ps: int
    auxiliary: bool


def learned_memory_execution_schedule(
    batches: Sequence[Sequence[int]],
    specs: ClipSpecTable,
    *,
    arm: str,
    seed: int,
    epoch: int,
    cadence: int = 8,
) -> tuple[LearnedMemoryExecutionEntry, ...]:
    """Pair anchors/dt/query/aux exactly; only M-long varies the span."""
    if arm not in {"M-short", "M-long"} or cadence < 1:
        raise ValueError("arm must be M-short/M-long and cadence must be positive")
    by_index = {item.index: item for item in specs}
    for indices in batches:
        if not indices:
            raise ValueError("empty learned-memory execution batch")
        keys = {by_index[int(index)].group_key for index in indices}
        if len(keys) != 1:
            raise ValueError("learned-memory execution batch is not homogeneous")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, epoch, 0xB20260929])
    )
    order = rng.permutation(len(batches)).tolist()
    entries: list[LearnedMemoryExecutionEntry | None] = [None] * len(batches)
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for offset, position in enumerate(order):
        # Cycling the full 3x3x3 grid makes both marginals and long spans
        # differ by at most one batch.  Short ignores only the span component.
        combined = offset % 27
        span_index = combined // 9
        remainder = combined % 9
        observed_dt = OBSERVED_DT_PS[remainder // 3]
        query_lag = QUERY_LAG_PS[remainder % 3]
        span = LONG_SPANS_PS[span_index] if arm == "M-long" else 800
        entries[position] = LearnedMemoryExecutionEntry(
            global_batch_index=position,
            indices=tuple(int(index) for index in batches[position]),
            history_span_ps=span,
            observed_dt_ps=observed_dt,
            query_lag_ps=query_lag,
            auxiliary=False,
        )
        strata[(observed_dt, query_lag)].append(position)
    for positions in strata.values():
        count = int(np.floor(len(positions) / cadence + 0.5))
        for position in rng.choice(positions, size=count, replace=False).tolist():
            entry = entries[position]
            assert entry is not None
            entries[position] = LearnedMemoryExecutionEntry(
                **{**asdict(entry), "auxiliary": True}
            )
    return tuple(entry for entry in entries if entry is not None)


def learned_memory_execution_manifest(
    entries: Sequence[LearnedMemoryExecutionEntry],
) -> dict:
    coverage: dict[str, dict[str, int]] = {}
    paired_rows = []
    rows = []
    for entry in entries:
        key = (
            f"span_{entry.history_span_ps}ps|history_dt_{entry.observed_dt_ps}ps|"
            f"query_{entry.query_lag_ps}ps"
        )
        value = coverage.setdefault(
            key, {"updates": 0, "clips": 0, "aux_updates": 0}
        )
        value["updates"] += 1
        value["clips"] += len(entry.indices)
        value["aux_updates"] += int(entry.auxiliary)
        row = asdict(entry)
        rows.append(row)
        paired_rows.append(
            {
                key: row[key]
                for key in (
                    "global_batch_index", "indices", "observed_dt_ps",
                    "query_lag_ps", "auxiliary",
                )
            }
        )
    return {
        "schema": SCHEMA,
        "entries": rows,
        "sha256": canonical_hash(rows),
        "paired_schedule_sha256": canonical_hash(paired_rows),
        "coverage": coverage,
        "auxiliary_rounding": "nearest integer within history-dt/query-lag strata",
    }
