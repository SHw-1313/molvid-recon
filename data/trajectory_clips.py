"""Raw ATLAS and MISATO to ordered packed-clip preprocessing.

This module is intentionally separate from ``atlas_dataset.py`` and
``misato_dataset.py``.  Those modules implement the legacy one-step pair
datasets and remain untouched by this path.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import mdtraj as md
import numpy as np

from data.clip_dataset import (
    ClipMMapWriter,
    STORAGE_FORMAT,
    canonical_time_fields,
    source_inventory_hash,
)
from utils.bio_utils import get_block_from_complex, get_block_from_top


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


def topology_metadata(top: md.Topology, *, complex_topology: bool) -> dict[str, Any]:
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


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise SourceDataError("Kabsch inputs must have matching [M, 3] shapes")
    if mobile.shape[0] == 0:
        raise SourceDataError("Kabsch requires at least one alignment atom")
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = mobile_centered.T @ reference_centered
    u, _, vh = np.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    translation = reference_center - mobile_center @ rotation
    return rotation.astype(np.float64), translation.astype(np.float64)


def align_and_center(
    x: np.ndarray,
    bpos: np.ndarray,
    align_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align every frame to its clip's first frame and center once."""

    if x.ndim != 3 or x.shape[-1] != 3 or bpos.shape != x.shape:
        raise SourceDataError("coordinates must have matching [T, N, 3] shapes")
    reference = x[0, align_mask].astype(np.float64)
    aligned_x = np.empty_like(x, dtype=np.float32)
    aligned_bpos = np.empty_like(bpos, dtype=np.float32)
    rotations: list[list[list[float]]] = []
    translations: list[list[float]] = []
    for frame_idx in range(x.shape[0]):
        if frame_idx == 0:
            rotation = np.eye(3, dtype=np.float64)
            translation = np.zeros(3, dtype=np.float64)
        else:
            rotation, translation = _kabsch(
                x[frame_idx, align_mask].astype(np.float64), reference
            )
        aligned_x[frame_idx] = x[frame_idx] @ rotation + translation
        aligned_bpos[frame_idx] = bpos[frame_idx] @ rotation + translation
        rotations.append(rotation.tolist())
        translations.append(translation.tolist())
    center = aligned_x[0].mean(axis=0).astype(np.float32)
    aligned_x -= center
    aligned_bpos -= center
    return aligned_x, aligned_bpos, {
        "rotation": rotations,
        "translation": translations,
        "center_angstrom": center.tolist(),
        "alignment_atom_count": int(align_mask.sum()),
    }


def _windows(
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

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise SourceDataError("MISATO trajectory_coordinates must have shape [T, N, 3]")
    for index in range(0, coordinates.shape[0], source_stride):
        yield np.asarray(coordinates[index], dtype=np.float32), float(index * native_dt_ps)


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
    x, bpos, transform = align_and_center(x, bpos, metadata["align_mask"])
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
    top = md.load(str(pdb_path)).topology
    metadata = topology_metadata(top, complex_topology=False)
    for replica in replicas:
        xtc_path = system_dir / f"{system_id}_prod_{replica}_fit.xtc"
        if not xtc_path.exists():
            raise SourceDataError(f"missing ATLAS trajectory: {xtc_path}")
        frames = iter_atlas_frames(
            xtc_path,
            pdb_path,
            source_stride=config.source_stride,
            expected_native_dt_ps=config.atlas_native_dt_ps,
            tolerance_ps=config.timestamp_tolerance_ps,
        )
        for window_index, (window_frames, window_times) in enumerate(
            _windows(
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
    hdf5: h5py.File,
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
    top = md.load(str(top_path)).topology
    if top.n_atoms != coordinates.shape[1]:
        raise SourceDataError(
            f"MISATO topology/coordinate atom mismatch for {key}: "
            f"{top.n_atoms} != {coordinates.shape[1]}"
        )
    metadata = topology_metadata(top, complex_topology=True)
    frames = iter_misato_frames(
        coordinates,
        source_stride=config.source_stride,
        native_dt_ps=config.misato_native_dt_ps,
    )
    for window_index, (window_frames, window_times) in enumerate(
        _windows(
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


def _inventory(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(set(paths)):
        stat = path.stat()
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return entries


def write_split_manifest(
    output_path: str | Path,
    *,
    source: str,
    raw_root: str | Path,
    config: ClipPreprocessConfig,
    source_version: str,
    timestamp_provenance: str,
    topology_policy: str,
    split_policy: str,
    systems: Sequence[str],
    inventory: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    manifest = {
        "schema_version": "pvb.clip.v1",
        "storage_format": STORAGE_FORMAT,
        "source": source,
        "raw_root": str(Path(raw_root).resolve()),
        "source_version": source_version,
        "source_inventory_sha256": source_inventory_hash(inventory),
        "coordinate_unit": "angstrom",
        "timestamp_provenance": timestamp_provenance,
        "native_delta_time_ps": (
            config.atlas_native_dt_ps if source == "atlas" else config.misato_native_dt_ps
        ),
        "clip_len": config.clip_len,
        "window_stride": config.window_stride,
        "source_stride": config.source_stride,
        "topology_policy": topology_policy,
        "split_policy": split_policy,
        "systems": list(systems),
        "inventory": list(inventory),
        "counts": dict(counts),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


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
