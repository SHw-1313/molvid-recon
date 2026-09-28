"""Deterministic global batch/H/aux schedule, independent of model RNG.

The cursor advances only after a successful update. Numerical failures abort;
resuming the last checkpoint restores cursor and both model RNG streams. DDP
ranks take every world-size-th entry, matching the packed sampler's sharding.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ..data.sampling import ClipSpecTable
from ..runtime import canonical_hash


EXECUTION_SCHEMA = "molvid.global_execution.stratified.v1"


@dataclass(frozen=True)
class ExecutionEntry:
    global_batch_index: int
    indices: tuple[int, ...]
    view: str
    time_bucket_id: str
    history_frames: int
    auxiliary: bool


def execution_schedule(
    batches: Sequence[Sequence[int]], specs: ClipSpecTable, *,
    histories: Sequence[int], seed: int, epoch: int, cadence: int = 8,
) -> tuple[ExecutionEntry, ...]:
    """Balance H within each ordinary lag, then select rounded n/8 per stratum.

    Rounding is nearest integer (ties upward), explicitly visible in coverage.
    The auxiliary mask is created even for A0, so schedules match across arms.
    No torch/CUDA RNG is read or modified here.
    """
    if not histories or any(h < 1 for h in histories) or cadence < 1:
        raise ValueError("positive histories and cadence are required")
    by_index = {item.index: item for item in specs}
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0xA20260928]))
    groups = defaultdict(list)
    for position, indices in enumerate(batches):
        if not indices:
            raise ValueError("empty execution batch")
        keys = {by_index[index].group_key for index in indices}
        if len(keys) != 1:
            raise ValueError("execution batch is not homogeneous")
        groups[next(iter(keys))].append(position)
    entries = [None] * len(batches)
    strata = defaultdict(list)
    for (_, frames, bucket), positions in sorted(groups.items()):
        order = rng.permutation(positions).tolist()
        fixed = bucket.startswith("fixed_history_dt_")
        view = "fixed_history" if fixed else "ordinary"
        for offset, position in enumerate(order):
            history = 4 if fixed else int(histories[offset % len(histories)])
            if history >= frames:
                raise ValueError("scheduled history leaves no query frames")
            entries[position] = ExecutionEntry(
                position, tuple(batches[position]), view, bucket, history, False,
            )
            strata[(view, bucket, history)].append(position)
    for positions in strata.values():
        count = int(np.floor(len(positions) / cadence + 0.5))
        for position in rng.choice(positions, size=count, replace=False).tolist():
            entry = entries[position]
            entries[position] = ExecutionEntry(**{**asdict(entry), "auxiliary": True})
    return tuple(entries)


def execution_manifest(entries: Sequence[ExecutionEntry]) -> dict:
    coverage = {}
    for entry in entries:
        key = f"{entry.view}|{entry.time_bucket_id}|H{entry.history_frames}"
        row = coverage.setdefault(key, {"updates": 0, "clips": 0, "aux_updates": 0})
        row["updates"] += 1
        row["clips"] += len(entry.indices)
        row["aux_updates"] += int(entry.auxiliary)
    rows = [asdict(entry) for entry in entries]
    return {"schema": EXECUTION_SCHEMA, "entries": rows,
            "sha256": canonical_hash(rows), "coverage": coverage,
            "rounding": "nearest integer n/cadence; half upward"}
