"""Raw trajectory to ordered, physically timed clip records."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..geometry.coordinates import align_and_center
from .batch import canonical_time_fields
from .chemistry import get_block_from_complex, get_block_from_top
from .io import read_topology, read_trajectory, read_static_records
from .store import ClipMMapWriter, STORAGE_FORMAT, StaticClipDataset

class SourceDataError(RuntimeError):
    """Raised when raw trajectory metadata or topology cannot be verified."""


@dataclass(frozen=True)
class ClipPreprocessConfig:
    clip_len: int = 16
    window_stride: int = 16
    source_stride: int = 1
    atlas_native_dt_ps: float = 10.0
    misato_native_dt_ps: float = 80.0
    timestamp_tolerance_ps: float = 1e-3

    def validate(self) -> None:
        if self.clip_len < 2:
            raise ValueError("clip_len must be at least 2 for trajectory clips")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")
        if self.source_stride < 1:
            raise ValueError("source_stride must be positive")
        if self.atlas_native_dt_ps <= 0 or self.misato_native_dt_ps <= 0:
            raise ValueError("native time intervals must be positive")


def stable_unit_hash(value: str, *, seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def select_fraction(value: str, *, fraction: float, seed: int) -> bool:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return stable_unit_hash(value, seed=seed, salt="select") < fraction


def atlas_split(system_id: str, *, seed: int) -> str:
    value = stable_unit_hash(system_id, seed=seed, salt="split")
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "valid"
    return "test"


def _bucket_id(delta_time_ps: float) -> str:
    rounded = float(f"{delta_time_ps:.6g}")
    if rounded.is_integer():
        return f"dt_{int(rounded)}ps"
    return f"dt_{rounded:g}ps"


def _atom_identity(atom: Any) -> str:
    residue = atom.residue
    chain = residue.chain
    chain_id = str(getattr(chain, "chain_id", "") or getattr(chain, "index", 0))
    insertion = str(getattr(residue, "insertion_code", "") or "")
    return (
        f"{chain_id}:{int(residue.resSeq)}:{insertion}:{residue.name}:"
        f"{atom.name}:{int(atom.index)}"
    )


def topology_metadata(top: Any, *, complex_topology: bool) -> dict[str, Any]:
    """Return atom/block/component metadata in the model's stable atom order."""

    if complex_topology:
        atype, btype, atom_index, block_index, bond_index, edge_mask = (
            get_block_from_complex(top)
        )
    else:
        atype, btype, atom_index, block_index, bond_index = get_block_from_top(top)
        edge_mask = np.zeros(len(atom_index), dtype=np.int64)

    atom_index = np.asarray(atom_index, dtype=np.int64)
    block_index = np.asarray(block_index, dtype=np.int64)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    if atom_index.size == 0:
        raise SourceDataError("topology contains no supported non-hydrogen atoms")
    if atom_index.size != block_index.size or atom_index.size != edge_mask.size:
        raise SourceDataError("topology parser returned inconsistent atom metadata")

    atoms = [top.atom(int(index)) for index in atom_index]
    block_id = np.asarray([atom.residue.index for atom in atoms], dtype=np.int64)
    component_id = (edge_mask > 0).astype(np.int64)
    align_mask = np.asarray(
        [
            component == 0
            and atom.name.strip().upper() in {"N", "CA", "C", "O"}
            for atom, component in zip(atoms, component_id)
        ],
        dtype=np.bool_,
    )
    if int(align_mask.sum()) < 3:
        align_mask = np.asarray(
            [component == 0 and atom.name.strip().upper() == "CA"
             for atom, component in zip(atoms, component_id)],
            dtype=np.bool_,
        )
    if int(align_mask.sum()) < 1:
        raise SourceDataError("topology has no protein alignment atoms")

    bonds = np.asarray(bond_index, dtype=np.int64)
    if bonds.size == 0:
        bonds = np.empty((2, 0), dtype=np.int64)
    elif bonds.ndim != 2 or bonds.shape[0] != 2:
        raise SourceDataError("topology parser returned invalid bond_index")

    identities = [_atom_identity(atom) for atom in atoms]
    if len(set(identities)) != len(identities):
        raise SourceDataError("topology contains duplicate stable atom identities")
    fingerprint_payload = {
        "atom_index": atom_index.tolist(),
        "atype": np.asarray(atype, dtype=np.int64).tolist(),
        "btype": np.asarray(btype, dtype=np.int64).tolist(),
        "block_index": block_index.tolist(),
        "edge_mask": edge_mask.tolist(),
        "bond_index": bonds.tolist(),
        "atom_identity": identities,
    }
    topology_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "atype": np.asarray(atype, dtype=np.int64),
        "btype": np.asarray(btype, dtype=np.int64),
        "atom_source_index": atom_index,
        "block_source_index": block_index,
        "block_id": block_id,
        "component_id": component_id,
        "edge_mask": edge_mask,
        "align_mask": align_mask,
        "loss_mask": np.ones(atom_index.size, dtype=np.bool_),
        "bond_index": bonds,
        "atom_identity": identities,
        "topology_fingerprint": topology_fingerprint,
    }


def iterate_windows(
    samples: Iterable[tuple[np.ndarray, float]],
    *,
    clip_len: int,
    window_stride: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    frame_window: deque[np.ndarray] = deque()
    time_window: deque[float] = deque()
    start = 0
    for frame, time in samples:
        frame_window.append(np.asarray(frame, dtype=np.float32))
        time_window.append(float(time))
        if len(frame_window) < clip_len:
            continue
        if start % window_stride == 0:
            yield np.stack(tuple(frame_window), axis=0), np.asarray(
                tuple(time_window), dtype=np.float64
            )
        frame_window.popleft()
        time_window.popleft()
        start += 1


def make_clip_record(
    frames: np.ndarray,
    absolute_times: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    sample_id: str,
    source: str,
    system_id: str,
    replica: str,
    split: str,
    source_stride: int,
    native_dt_ps: float,
    timestamp_provenance: str,
) -> dict[str, Any]:
    """Transform one raw window into a validated serialized clip record."""

    if frames.ndim != 3:
        raise SourceDataError("clip frames must have shape [T, N, 3]")
    atom_index = metadata["atom_source_index"]
    block_index = metadata["block_source_index"]
    if np.max(atom_index) >= frames.shape[1] or np.max(block_index) >= frames.shape[1]:
        raise SourceDataError("topology atom index exceeds trajectory atom count")
    x = frames[:, atom_index]
    bpos = frames[:, block_index]
    try:
        x, bpos, transform = align_and_center(x, bpos, metadata["align_mask"])
    except ValueError as error:
        raise SourceDataError(str(error)) from error
    time_ps, delta_time_ps = canonical_time_fields(
        absolute_times - absolute_times[0]
    )
    sampled_dt_ps = float(delta_time_ps[0]) if delta_time_ps.size else native_dt_ps
    record = {
        "schema_version": "pvb.clip.v1",
        "storage_format": STORAGE_FORMAT,
        "sample_id": sample_id,
        "source": source,
        "system_id": system_id,
        "replica": replica,
        "split": split,
        "task": "trajectory",
        "coordinate_unit": "angstrom",
        "timestamp_provenance": timestamp_provenance,
        "native_delta_time_ps": float(native_dt_ps),
        "source_stride": int(source_stride),
        "sampled_delta_time_ps": sampled_dt_ps,
        "time_bucket_id": _bucket_id(sampled_dt_ps),
        "time_ps": time_ps,
        "delta_time_ps": delta_time_ps,
        "x": x,
        "bpos": bpos,
        "atype": metadata["atype"],
        "btype": metadata["btype"],
        "block_id": metadata["block_id"],
        "component_id": metadata["component_id"],
        "atom_source_index": metadata["atom_source_index"],
        "atom_identity": list(metadata["atom_identity"]),
        "edge_mask": metadata["edge_mask"],
        "loss_mask": metadata["loss_mask"],
        "align_mask": metadata["align_mask"],
        "bond_index": metadata["bond_index"],
        "topology_fingerprint": metadata["topology_fingerprint"],
        "alignment": transform,
    }
    return record


def iter_atlas_records(
    atlas_root: str | Path,
    system_id: str,
    *,
    split: str,
    config: ClipPreprocessConfig,
    replicas: Sequence[str] = ("R1", "R2", "R3"),
) -> Iterator[dict[str, Any]]:
    root = Path(atlas_root)
    system_dir = root / system_id
    pdb_path = system_dir / f"{system_id}.pdb"
    if not pdb_path.exists():
        raise SourceDataError(f"missing ATLAS topology: {pdb_path}")
    top = read_topology(pdb_path)
    metadata = topology_metadata(top, complex_topology=False)
    for replica in replicas:
        xtc_path = system_dir / f"{system_id}_prod_{replica}_fit.xtc"
        if not xtc_path.exists():
            raise SourceDataError(f"missing ATLAS trajectory: {xtc_path}")
        frames = read_trajectory(
            "atlas",
            xtc_path,
            topology_path=pdb_path,
            source_stride=config.source_stride,
            native_dt_ps=config.atlas_native_dt_ps,
            tolerance_ps=config.timestamp_tolerance_ps,
        )
        for window_index, (window_frames, window_times) in enumerate(
            iterate_windows(
                frames,
                clip_len=config.clip_len,
                window_stride=config.window_stride,
            )
        ):
            yield make_clip_record(
                window_frames,
                window_times,
                metadata,
                sample_id=f"atlas_{system_id}_{replica}_w{window_index:06d}",
                source="atlas",
                system_id=system_id,
                replica=replica,
                split=split,
                source_stride=config.source_stride,
                native_dt_ps=config.atlas_native_dt_ps,
                timestamp_provenance="ATLAS_XTC_time_ps",
            )


def _misato_top_path(root: Path, system_id: str) -> Path:
    return root / "parameter_restart_files_MD" / system_id.lower() / f"{system_id.upper()}.pdb"


def iter_misato_records(
    misato_root: str | Path,
    system_id: str,
    *,
    split: str,
    hdf5: Any,
    config: ClipPreprocessConfig,
) -> Iterator[dict[str, Any]]:
    root = Path(misato_root)
    top_path = _misato_top_path(root, system_id)
    if not top_path.exists():
        raise SourceDataError(f"missing MISATO topology: {top_path}")
    key = system_id.upper()
    if key not in hdf5:
        raise SourceDataError(f"missing MISATO HDF5 key: {key}")
    coordinates = np.asarray(hdf5[key]["trajectory_coordinates"], dtype=np.float32)
    top = read_topology(top_path)
    if top.n_atoms != coordinates.shape[1]:
        raise SourceDataError(
            f"MISATO topology/coordinate atom mismatch for {key}: "
            f"{top.n_atoms} != {coordinates.shape[1]}"
        )
    metadata = topology_metadata(top, complex_topology=True)
    frames = read_trajectory(
        "misato",
        coordinates,
        source_stride=config.source_stride,
        native_dt_ps=config.misato_native_dt_ps,
    )
    for window_index, (window_frames, window_times) in enumerate(
        iterate_windows(
            frames,
            clip_len=config.clip_len,
            window_stride=config.window_stride,
        )
    ):
        yield make_clip_record(
            window_frames,
            window_times,
            metadata,
            sample_id=f"misato_{key}_w{window_index:06d}",
            source="misato",
            system_id=key,
            replica="MD",
            split=split,
            source_stride=config.source_stride,
            native_dt_ps=config.misato_native_dt_ps,
            timestamp_provenance="MISATO_Handoff_verified_native_dt_ps",
        )


def write_records(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> int:
    count = 0
    with ClipMMapWriter(output_dir, overwrite=overwrite) as writer:
        for record in records:
            writer.append(record)
            count += 1
    return count

def preprocess_atlas(
    raw_root: str | Path,
    system_id: str,
    output_dir: str | Path,
    *,
    split: str,
    config: ClipPreprocessConfig,
    replicas: Sequence[str] = ("R1", "R2", "R3"),
    overwrite: bool = False,
) -> int:
    config.validate()
    return write_records(
        iter_atlas_records(raw_root, system_id, split=split, config=config, replicas=replicas),
        output_dir,
        overwrite=overwrite,
    )


def preprocess_misato(
    raw_root: str | Path,
    system_id: str,
    output_dir: str | Path,
    *,
    split: str,
    config: ClipPreprocessConfig,
    overwrite: bool = False,
) -> int:
    import h5py

    config.validate()
    with h5py.File(Path(raw_root) / "MD.hdf5", "r") as hdf5:
        return write_records(
            iter_misato_records(raw_root, system_id, split=split, hdf5=hdf5, config=config),
            output_dir,
            overwrite=overwrite,
        )


def preprocess_static(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    source: str,
    split: str,
    overwrite: bool = False,
) -> int:
    return write_records(
        read_static_records(raw_root, source=source, split=split),
        output_dir,
        overwrite=overwrite,
    )
