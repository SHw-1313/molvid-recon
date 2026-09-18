"""Raw topology, trajectory and static-store I/O boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np


def read_topology(path: str | Path) -> Any:
    import mdtraj as md

    return md.load(str(path)).topology

def iter_atlas_frames(
    xtc_path: str | Path,
    pdb_path: str | Path,
    *,
    source_stride: int,
    expected_native_dt_ps: float,
    tolerance_ps: float,
    chunk_size: int = 256,
) -> Iterator[tuple[np.ndarray, float]]:
    """Stream ATLAS frames while checking the XTC physical clock."""

    import mdtraj as md
    from .preprocess import SourceDataError

    previous_time: float | None = None
    raw_index = 0
    observed_dt: float | None = None
    for trajectory in md.iterload(
        str(xtc_path), top=str(pdb_path), chunk=chunk_size
    ):
        coordinates = np.asarray(trajectory.xyz, dtype=np.float32) * 10.0
        times = np.asarray(trajectory.time, dtype=np.float64)
        if coordinates.shape[0] != times.size or times.size == 0:
            raise SourceDataError(f"invalid XTC chunk: {xtc_path}")
        for frame, time in zip(coordinates, times):
            if previous_time is not None:
                delta = float(time - previous_time)
                if delta <= 0:
                    raise SourceDataError(f"non-monotonic ATLAS time in {xtc_path}")
                if observed_dt is None:
                    observed_dt = delta
                elif not np.isclose(
                    delta, observed_dt, rtol=1e-5, atol=tolerance_ps
                ):
                    raise SourceDataError(f"irregular ATLAS time grid in {xtc_path}")
            if raw_index % source_stride == 0:
                yield frame, float(time)
            previous_time = float(time)
            raw_index += 1
    if observed_dt is None:
        raise SourceDataError(f"ATLAS trajectory has fewer than two frames: {xtc_path}")
    if not np.isclose(
        observed_dt, expected_native_dt_ps, rtol=1e-4, atol=tolerance_ps
    ):
        raise SourceDataError(
            f"ATLAS native dt mismatch for {xtc_path}: observed {observed_dt}, "
            f"expected {expected_native_dt_ps} ps"
        )


def iter_misato_frames(
    coordinates: np.ndarray,
    *,
    source_stride: int,
    native_dt_ps: float,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield MISATO frames with the operator-supplied verified time interval."""

    from .preprocess import SourceDataError

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise SourceDataError("MISATO trajectory_coordinates must have shape [T, N, 3]")
    for index in range(0, coordinates.shape[0], source_stride):
        yield np.asarray(coordinates[index], dtype=np.float32), float(index * native_dt_ps)



def read_trajectory(
    source: str,
    data: str | Path | np.ndarray,
    *,
    topology_path: str | Path | None = None,
    source_stride: int = 1,
    native_dt_ps: float,
    tolerance_ps: float = 1e-3,
) -> Iterator[tuple[np.ndarray, float]]:
    """Read ATLAS XTC (nm to Å) or verified-interval MISATO coordinates."""

    if source_stride < 1 or native_dt_ps <= 0:
        raise ValueError("source stride and native interval must be positive")
    if source == "atlas":
        if topology_path is None or isinstance(data, np.ndarray):
            raise ValueError("ATLAS XTC requires a path and topology_path")
        return iter_atlas_frames(
            data,
            topology_path,
            source_stride=source_stride,
            expected_native_dt_ps=native_dt_ps,
            tolerance_ps=tolerance_ps,
        )
    if source == "misato":
        return iter_misato_frames(
            np.asarray(data, dtype=np.float32),
            source_stride=source_stride,
            native_dt_ps=native_dt_ps,
        )
    raise ValueError(f"unsupported trajectory source: {source!r}")


def read_static_records(
    root: str | Path, *, source: str, split: str
) -> Iterator[dict[str, Any]]:
    """Yield T=1 clips from an existing compressed static mmap store."""

    from .store import StaticClipDataset

    dataset = StaticClipDataset(root, source=source, split=split)
    for index in range(len(dataset)):
        yield dataset[index]


def write_trajectory(
    path: str | Path, coordinates: np.ndarray, time_ps: np.ndarray
) -> None:
    """Export an ordered Å/ps trajectory as an NPZ artifact."""

    from .batch import canonical_time_fields

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("trajectory export must use .npz")
    x = np.asarray(coordinates, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("trajectory coordinates must have shape [T, N, 3]")
    time, _ = canonical_time_fields(time_ps)
    if x.shape[0] != time.size:
        raise ValueError("trajectory coordinates and timestamps disagree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=x, time_ps=time, coordinate_unit="angstrom")
