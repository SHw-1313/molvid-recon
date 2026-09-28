"""B raw ATLAS window contracts, independent of common training samplers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..geometry.coordinates import kabsch_align
from .io import iter_atlas_frames


@dataclass(frozen=True)
class RawHistoryWindow:
    """Paired arm indices on the same native 10 ps trajectory clock."""
    anchor_index: int
    history_span_ps: int
    observed_dt_ps: int
    query_lag_ps: int
    split: str
    future_count: int = 12

    def __post_init__(self):
        if self.split not in ("train", "valid"):
            raise ValueError("only train/valid raw trajectories are permitted")
        if self.history_span_ps not in (800, 3200, 12800) or self.observed_dt_ps not in (100, 200, 400):
            raise ValueError("unsupported B history span or observed sampling interval")
        allowed = (100, 200, 400) if self.split == "train" else (100, 200, 300, 400)
        if self.query_lag_ps not in allowed or self.future_count != 12:
            raise ValueError("B uses Q12; 300 ps queries are valid-only")
        if self.anchor_index < 1280:
            raise ValueError("both arms require the same 12.8 ns-past eligible anchors")

    @property
    def history_indices(self):
        return np.arange(self.anchor_index - self.history_span_ps // 10,
                         self.anchor_index + 1, self.observed_dt_ps // 10, dtype=np.int64)

    @property
    def anchor_indices(self):
        return np.array([self.anchor_index - 80, self.anchor_index], dtype=np.int64)

    @property
    def future_indices(self):
        return self.anchor_index + np.arange(1, 13, dtype=np.int64) * (self.query_lag_ps // 10)

    def validate_coverage(self, frame_count: int):
        # Common future eligibility even for a 100 ps query.
        if self.anchor_index + 480 >= frame_count:
            raise ValueError("anchor lacks the common Q12/400 ps future coverage")

    def manifest(self):
        return dict(anchor_raw_index=self.anchor_index, history_span_ps=self.history_span_ps,
                    observed_dt_ps=self.observed_dt_ps, query_lag_ps=self.query_lag_ps,
                    split=self.split, future_count=self.future_count,
                    history_raw_indices=self.history_indices.tolist(),
                    geometry_anchor_raw_indices=self.anchor_indices.tolist(),
                    future_raw_indices=self.future_indices.tolist(),
                    history_time_ps=((self.history_indices - self.anchor_index) * 10).tolist(),
                    anchor_time_ps=[-800, 0], query_time_ps=((self.future_indices - self.anchor_index) * 10).tolist())


def read_raw_window_bank(xtc: Path, pdb: Path, indices: np.ndarray, *, atom_source_index: np.ndarray):
    """Read actual requested frames, checking every native timestamp through max index.

    Does not claim coverage of the unscanned trajectory tail. CPU work here is
    raw I/O only, never a model fallback. Returned coordinates are Angstrom.
    """
    needed = set(int(i) for i in indices)
    if not needed or min(needed) < 0:
        raise ValueError("requested indices must be nonempty and nonnegative")
    frames, times = {}, {}
    first_time = None
    reader = iter_atlas_frames(xtc, pdb, source_stride=1, expected_native_dt_ps=10, tolerance_ps=1e-3)
    try:
        for index, (xyz, time) in enumerate(reader):
            if first_time is None:
                first_time = time
            if not np.isclose(time - first_time, index * 10, rtol=0, atol=1e-3):
                raise ValueError(f"raw physical time mismatch at {xtc}:{index}")
            if index in needed:
                selected = xyz[atom_source_index]
                if not np.isfinite(selected).all():
                    raise ValueError("nonfinite raw coordinates")
                frames[index], times[index] = selected, time
            if index >= max(needed):
                break
    finally:
        reader.close()
    if needed != set(frames):
        raise ValueError(f"missing requested raw past/future indices: {sorted(needed-set(frames))[:5]}")
    ordered = sorted(needed)
    return np.stack([frames[i] for i in ordered]), np.array([times[i] for i in ordered]), np.array(ordered)


def align_bank_to_observed(bank: np.ndarray, indices: np.ndarray, anchor_index: int, align_mask: np.ndarray):
    """Every frame uses the same last-observed backbone reference.

    Future frames are transformed independently against observed geometry;
    changing future cannot change any history/anchor coordinates or origin.
    """
    reference = bank[np.flatnonzero(indices == anchor_index).item(), align_mask].astype(np.float64)
    result = np.empty_like(bank)
    for i, frame in enumerate(bank):
        rotation, translation = kabsch_align(frame[align_mask].astype(np.float64), reference)
        result[i] = frame @ rotation + translation
    return result
