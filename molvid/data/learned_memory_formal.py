"""Paired raw-ATLAS exposure and I/O for learned-memory training.

The frozen 100 ps store supplies only cheap trajectory/window identities and
atom counts.  Coordinates are read from the original 10 ps XTC trajectory so
history sampling remains independent of the query lag.  Targets and the two
geometry anchors are materialized separately and aligned to the same last
observed reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..runtime import canonical_hash, sha256_file
from .batch import collate_clip_records
from .preprocess import topology_metadata
from .io import read_topology
from .raw_history import RawHistoryWindow, pack_pilot_inputs
from .sampling import ClipItemSpec, ClipSpecTable


_DT_ID = re.compile(
    r"^atlas_(?P<system>.+)_(?P<replica>R[123])_dt_(?P<dt>100|300)ps_w(?P<window>[0-9]+)$"
)
_RAW_ID = re.compile(
    r"^atlas_(?P<system>.+)_(?P<replica>R[123])_w(?P<window>[0-9]+)$"
)


@dataclass(frozen=True)
class RawMemoryItem:
    index: int
    system: str
    replica: str
    window_index: int
    anchor_raw_index: int
    maximum_evidenced_raw_index: int
    atoms: int
    sample_id: str
    xtc: Path
    pdb: Path

    @property
    def trajectory_id(self) -> str:
        return f"atlas_{self.system}_{self.replica}"


class RawMemoryDataset:
    """Frozen 12.8 ns-past/Q12-400 ps-future eligible train anchors."""

    def __init__(
        self,
        *,
        manifest_root: Path,
        raw_root: Path,
        offset_root: Path | None = None,
        split: str = "train",
    ) -> None:
        if split not in {"train", "valid"}:
            raise ValueError("raw learned-memory dataset supports train/valid only")
        self.manifest_root = Path(manifest_root).resolve()
        self.raw_root = Path(raw_root).resolve()
        self.offset_root = Path(offset_root).resolve() if offset_root is not None else None
        self._offset_paths: dict[str, Path] = {}
        self.split = split
        manifest_path = self.manifest_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("test_sampling", {}).get("opened") is not False:
            raise RuntimeError("sealed test contract is not intact")
        selected = tuple(
            str(value).removeprefix("atlas_")
            for value in manifest["source_splits"][split]["selected_systems"]
        )
        expected_systems = 192 if split == "train" else 8
        if len(selected) != expected_systems or len(set(selected)) != expected_systems:
            raise ValueError(
                f"expected frozen {expected_systems} {split} systems, got {len(selected)}"
            )
        store_name = "train" if split == "train" else "valid"
        index_path = self.manifest_root / "clip_store" / store_name / "index.txt"
        by_trajectory: dict[tuple[str, str], list[tuple[int, int]]] = {}
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                match = _DT_ID.fullmatch(fields[0])
                if match is None:
                    continue
                expected_index_dt = 100 if split == "train" else 300
                if int(match.group("dt")) != expected_index_dt:
                    continue
                system = match.group("system")
                if system not in selected:
                    continue
                by_trajectory.setdefault((system, match.group("replica")), []).append(
                    (int(match.group("window")), int(fields[3]))
                )
        expected_trajectories = expected_systems * 3
        if len(by_trajectory) != expected_trajectories:
            raise ValueError(
                f"expected {expected_trajectories} frozen trajectories, got {len(by_trajectory)}"
            )
        items: list[RawMemoryItem] = []
        inventory: list[dict[str, Any]] = []
        for (system, replica), windows in sorted(by_trajectory.items()):
            ordered = sorted(windows)
            if len({atoms for _, atoms in ordered}) != 1:
                raise ValueError("atom count changes inside a physical trajectory")
            maximum = ordered[-1][0]
            atoms = ordered[0][1]
            available = {window for window, _ in ordered}
            # The frozen train index is dt100 (160 raw frames per window); the
            # validation index is dt300 (480 raw frames per window).  Both only
            # select anchors with 12.8 ns raw past and 4.8 ns raw future.
            window_stride_raw = 160 if split == "train" else 480
            minimum_window = math.ceil(1280 / window_stride_raw)
            maximum_window = maximum - math.ceil(480 / window_stride_raw)
            eligible = [
                window for window in sorted(available)
                if minimum_window <= window <= maximum_window
            ]
            minimum_anchors = 24 if split == "train" else 2
            if len(eligible) < minimum_anchors:
                raise ValueError(
                    f"trajectory {system}/{replica} has fewer than{minimum_anchors} eligible anchors"
                )
            pdb = self.raw_root / system / f"{system}.pdb"
            xtc = self.raw_root / system / f"{system}_prod_{replica}_fit.xtc"
            for path in (pdb, xtc):
                if not path.is_file():
                    raise FileNotFoundError(path)
            inventory.extend(
                {
                    "path": str(path),
                    "size": int(path.stat().st_size),
                    "mtime_ns": int(path.stat().st_mtime_ns),
                }
                for path in (pdb, xtc)
            )
            for window in eligible:
                sample_id = f"atlas_{system}_{replica}_w{window:06d}"
                items.append(
                    RawMemoryItem(
                        index=len(items), system=system, replica=replica,
                        window_index=window,
                        anchor_raw_index=window * window_stride_raw,
                        maximum_evidenced_raw_index=(
                            maximum * window_stride_raw + window_stride_raw - 10
                        ),
                        atoms=atoms, sample_id=sample_id, xtc=xtc, pdb=pdb,
                    )
                )
        self.items = tuple(items)
        self._specs = ClipSpecTable.from_specs(
            ClipItemSpec(
                index=item.index,
                atoms=item.atoms,
                # Two observed teacher anchors plus Q12 targets determine the
                # packed CUDA cost.  Raw history is encoded by the shallow path.
                frames=14,
                task="trajectory",
                time_bucket_id="raw_memory_q12",
                native_delta_time_ps=10.0,
                physical_clip_span_ps=4800.0,
                sample_id=item.sample_id,
            )
            for item in self.items
        )
        self.identity = {
            "schema": "molvid.learned_memory.raw_dataset.v1",
            "split": split,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "store_index": str(index_path),
            "store_index_sha256": sha256_file(index_path),
            "raw_root": str(self.raw_root),
            "systems": expected_systems,
            "trajectories": expected_trajectories,
            "eligible_anchors": len(self.items),
            "anchor_rule": (
                f"dt{100 if split == 'train' else 300}_window_start="
                f"w*{160 if split == 'train' else 480}; past>=1280raw; future>=480raw"
            ),
            "raw_inventory_hash": canonical_hash(inventory),
            "test_opened": False,
        }
        if self.offset_root is not None:
            offset_manifest = self.offset_root / "manifest.json"
            if not offset_manifest.is_file():
                raise FileNotFoundError(offset_manifest)
            offset_identity = json.loads(offset_manifest.read_text(encoding="utf-8"))
            if offset_identity.get("schema") != "molvid.raw_xtc_offset_cache.v1":
                raise ValueError("unsupported XTC offset cache schema")
            if Path(offset_identity.get("raw_root", "")).resolve() != self.raw_root:
                raise ValueError("XTC offset cache raw root differs")
            if Path(offset_identity.get("manifest_root", "")).resolve() != self.manifest_root:
                raise ValueError("XTC offset cache source manifest differs")
            available = set(offset_identity.get("splits", ()))
            if split not in available:
                raise ValueError(f"XTC offset cache does not cover {split!r}")
            records = offset_identity.get("records")
            if (
                not isinstance(records, list)
                or len(records) != int(offset_identity.get("trajectories", -1))
            ):
                raise ValueError("XTC offset cache records are incomplete")
            by_xtc: dict[str, Mapping[str, Any]] = {}
            for record in records:
                if not isinstance(record, Mapping) or not isinstance(record.get("xtc"), str):
                    raise ValueError("XTC offset cache record is invalid")
                key = str(Path(record["xtc"]).resolve())
                if key in by_xtc:
                    raise ValueError(f"duplicate XTC offset cache record: {key}")
                by_xtc[key] = record
            required: dict[str, RawMemoryItem] = {}
            maximum_required: dict[str, int] = {}
            for item in self.items:
                key = str(item.xtc.resolve())
                required.setdefault(key, item)
                maximum_required[key] = max(
                    maximum_required.get(key, -1), item.maximum_evidenced_raw_index
                )
            for key, item in sorted(required.items()):
                record = by_xtc.get(key)
                if record is None or split not in set(record.get("splits", ())):
                    raise ValueError(f"XTC offset cache does not bind {split} trajectory {key}")
                sidecar = self.offset_root / item.system / f"{item.xtc.name}.offsets.npz"
                _validate_offset_sidecar(
                    xtc=item.xtc,
                    sidecar=sidecar,
                    offset_root=self.offset_root,
                    record=record,
                    minimum_frames=maximum_required[key] + 1,
                )
                self._offset_paths[key] = sidecar
            self.identity["xtc_offset_cache"] = {
                "root": str(self.offset_root),
                "manifest": str(offset_manifest),
                "manifest_sha256": sha256_file(offset_manifest),
                "cache_hash": offset_identity.get("cache_hash"),
            }
        self.identity["data_hash"] = canonical_hash(self.identity)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RawMemoryItem:
        return self.items[int(index)]

    def clip_spec_table(self) -> ClipSpecTable:
        return self._specs

    def load_inputs(
        self,
        indices: Sequence[int],
        *,
        history_span_ps: int,
        observed_dt_ps: int,
        query_lag_ps: int,
    ):
        values = [
            _load_item(
                self.items[int(index)],
                RawHistoryWindow(
                    self.items[int(index)].anchor_raw_index,
                    history_span_ps,
                    observed_dt_ps,
                    query_lag_ps,
                    self.split,
                ),
                offset_path=self._offset_paths.get(
                    str(self.items[int(index)].xtc.resolve())
                ),
            )
            for index in indices
        ]
        return pack_pilot_inputs(values)


def _validate_offset_sidecar(
    *,
    xtc: Path,
    sidecar: Path,
    offset_root: Path,
    record: Mapping[str, Any],
    minimum_frames: int,
) -> None:
    """Bind one mutable sidecar byte-for-byte to its frozen manifest record."""

    expected_keys = {
        "frames", "sidecar", "sidecar_sha256", "splits", "xtc",
        "xtc_mtime_ns", "xtc_size",
    }
    if set(record) != expected_keys:
        raise ValueError("XTC offset cache record keys differ")
    xtc = Path(xtc).resolve()
    sidecar = Path(sidecar).resolve()
    offset_root = Path(offset_root).resolve()
    if Path(str(record["xtc"])).resolve() != xtc:
        raise ValueError("XTC offset record trajectory path differs")
    declared_sidecar = Path(str(record["sidecar"]))
    if declared_sidecar.is_absolute():
        sidecar_matches = declared_sidecar.resolve() == sidecar
    else:
        relative = sidecar.relative_to(offset_root)
        suffix = (offset_root.name, *relative.parts)
        sidecar_matches = tuple(declared_sidecar.parts[-len(suffix):]) == suffix
    if not sidecar_matches:
        raise ValueError("XTC offset record sidecar path differs")
    source = xtc.stat()
    if (
        int(record["xtc_size"]) != int(source.st_size)
        or int(record["xtc_mtime_ns"]) != int(source.st_mtime_ns)
    ):
        raise ValueError("XTC offset record source identity differs")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    if sha256_file(sidecar) != str(record["sidecar_sha256"]):
        raise ValueError("XTC offset sidecar SHA-256 differs")
    with np.load(sidecar, allow_pickle=False) as cache:
        if set(cache.files) != {"offsets", "xtc_size", "xtc_mtime_ns"}:
            raise ValueError("XTC offset sidecar fields differ")
        offsets = np.asarray(cache["offsets"])
        cached_size = np.asarray(cache["xtc_size"])
        cached_mtime = np.asarray(cache["xtc_mtime_ns"])
    if (
        offsets.dtype != np.int64
        or offsets.ndim != 1
        or offsets.size != int(record["frames"])
        or offsets.size < int(minimum_frames)
        or offsets.size == 0
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) <= 0)
        or np.any(offsets % 4)
        or int(offsets[-1]) >= int(source.st_size)
    ):
        raise ValueError("XTC offset sidecar offsets are invalid")
    if (
        cached_size.shape != ()
        or cached_mtime.shape != ()
        or int(cached_size) != int(record["xtc_size"])
        or int(cached_mtime) != int(record["xtc_mtime_ns"])
    ):
        raise ValueError("XTC offset sidecar embedded source identity differs")


@lru_cache(maxsize=256)
def _metadata(pdb: str) -> Mapping[str, Any]:
    return topology_metadata(read_topology(Path(pdb)), complex_topology=False)


def _read_raw_frames(
    xtc: str,
    atom_source_index: tuple[int, ...],
    raw_indices: np.ndarray,
    offset_path: str | None = None,
):
    """Decode only the requested raw-XTC frames using persistent byte offsets.

    Formal epochs touch every physical trajectory but only a small part of
    each 100 ns file at a time.  Reading and retaining the complete evidenced
    prefix makes the first step for a cold file scan almost the entire XTC and
    leaves CUDA idle for minutes.  XTC seek is indexed, so a bounded interval
    preserves the raw-coordinate contract while keeping I/O proportional to
    the actual history and future requested by this update.
    """
    from mdtraj.formats import XTCTrajectoryFile

    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    if (
        raw_indices.ndim != 1
        or raw_indices.size == 0
        or np.any(np.diff(raw_indices) <= 0)
        or np.any(raw_indices % 10)
    ):
        raise ValueError("requested raw XTC indices must be sorted unique 100 ps points")
    coordinates: list[np.ndarray] = []
    times: list[np.ndarray] = []
    with XTCTrajectoryFile(xtc, "r") as handle:
        if offset_path is not None:
            with np.load(offset_path, allow_pickle=False) as cache:
                source = Path(xtc).stat()
                if (
                    int(cache["xtc_size"]) != int(source.st_size)
                    or int(cache["xtc_mtime_ns"]) != int(source.st_mtime_ns)
                ):
                    raise ValueError("XTC offset sidecar source identity differs")
                offsets = np.asarray(cache["offsets"], dtype=np.int64)
                if int(raw_indices[-1]) >= offsets.size:
                    raise ValueError("XTC offset sidecar ends before requested frame")
                # MDTraj's offsets setter does not update its private n_frames.
                # Exact stride-1 reads avoid the broken len(self) path used by
                # efficient striding while still making seek use the sidecar.
                handle.offsets = offsets
        for raw_index in raw_indices.tolist():
            handle.seek(int(raw_index))
            xyz, absolute_time_ps, _step, _box = handle.read(
                n_frames=1,
                stride=1,
                atom_indices=np.asarray(atom_source_index, dtype=np.int64),
            )
            if xyz.shape[0] != 1:
                raise ValueError("raw XTC ended before a requested formal frame")
            coordinates.append(xyz)
            times.append(absolute_time_ps)
    coordinates_nm = np.concatenate(coordinates, axis=0)
    times = np.concatenate(times).astype(np.float64, copy=False)
    if not np.allclose(
        times - times[0], (raw_indices - raw_indices[0]) * 10,
        rtol=0.0, atol=1e-3,
    ):
        raise ValueError("raw XTC interval differs from native10 ps time")
    return np.asarray(coordinates_nm, dtype=np.float32), times, raw_indices


def _align_to_observed_batched(
    coordinates: np.ndarray,
    indices: np.ndarray,
    anchor_index: int,
    align_mask: np.ndarray,
) -> np.ndarray:
    """Batched float64 Kabsch against one observed reference."""
    selected = coordinates[:, align_mask].astype(np.float64)
    reference = selected[int(np.flatnonzero(indices == anchor_index).item())]
    mobile_center = selected.mean(axis=1, keepdims=True)
    reference_center = reference.mean(axis=0, keepdims=True)
    mobile = selected - mobile_center
    target = reference - reference_center
    covariance = np.einsum("fmi,mj->fij", mobile, target)
    u, _singular, vh = np.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vh
    reflected = np.linalg.det(rotation) < 0
    if reflected.any():
        u[reflected, :, -1] *= -1
        rotation[reflected] = u[reflected] @ vh[reflected]
    translation = reference_center[0] - np.einsum(
        "fi,fij->fj", mobile_center[:, 0], rotation
    )
    aligned = np.einsum(
        "fni,fij->fnj", coordinates.astype(np.float64), rotation
    ) + translation[:, None]
    return aligned.astype(coordinates.dtype)


def _load_item(
    item: RawMemoryItem,
    window: RawHistoryWindow,
    *,
    offset_path: Path | None = None,
):
    """Read one strided XTC range and split observed/anchor/target objects."""
    import torch
    from ..codec.learned_memory import ChemicalMapping, RawHistoryObservation
    from ..geometry.types import StaticTopologyMetadata

    meta = _metadata(str(item.pdb))
    atom_source_index = np.asarray(meta["atom_source_index"], dtype=np.int64)
    minimum = int(window.history_indices[0])
    maximum = int(window.future_indices[-1])
    if minimum % 10 or maximum % 10 or (maximum - minimum) % 10:
        raise ValueError("formal B windows must lie on the100 ps read grid")
    if offset_path is not None:
        if not offset_path.is_file():
            raise FileNotFoundError(offset_path)
    requested = np.unique(np.concatenate(
        (window.history_indices, window.anchor_indices, window.future_indices)
    ))
    coordinates_nm, absolute_time_ps, read_indices = _read_raw_frames(
        str(item.xtc),
        tuple(int(value) for value in atom_source_index.tolist()),
        requested,
        str(offset_path) if offset_path is not None else None,
    )
    selected = np.searchsorted(read_indices, requested)
    if not np.array_equal(read_indices[selected], requested):
        raise ValueError("requested formal frames do not lie on the100 ps read grid")
    expected_time = (requested - window.anchor_index) * 10
    anchor_position = int(np.flatnonzero(requested == window.anchor_index).item())
    observed_time = np.asarray(absolute_time_ps, dtype=np.float64)
    if not np.allclose(
        observed_time[selected] - observed_time[selected[anchor_position]], expected_time,
        rtol=0.0, atol=1e-3,
    ):
        raise ValueError("raw XTC physical clock differs from the native10 ps contract")
    bank = _align_to_observed_batched(
        np.asarray(coordinates_nm[selected], dtype=np.float32) * 10.0,
        requested,
        window.anchor_index,
        np.asarray(meta["align_mask"], dtype=np.bool_),
    )
    raw_indices = requested

    def positions(requested: np.ndarray) -> np.ndarray:
        value = np.searchsorted(raw_indices, requested)
        if (value >= raw_indices.size).any() or not np.array_equal(raw_indices[value], requested):
            raise ValueError("strided XTC range lacks a requested physical frame")
        return value

    common = {
        key: np.asarray(meta[key])
        for key in (
            "atype", "btype", "block_id", "component_id", "bond_index",
            "align_mask", "loss_mask", "atom_source_index",
        )
    }
    common.update(
        sample_id=item.sample_id,
        source="atlas",
        system_id=item.system,
        replica=item.replica,
        split=window.split,
        task="trajectory",
        coordinate_unit="angstrom",
        native_delta_time_ps=10.0,
        edge_mask=np.asarray(meta["edge_mask"]),
        atom_identity=list(meta["atom_identity"]),
        time_bucket_id=f"raw_query_dt_{window.query_lag_ps}ps",
    )

    def clip(requested: np.ndarray):
        x = bank[positions(requested)]
        bpos = np.empty_like(x)
        for group in np.unique(meta["block_id"]):
            atoms = np.asarray(meta["block_id"]) == group
            bpos[:, atoms] = x[:, atoms].mean(axis=1, keepdims=True)
        return collate_clip_records(
            [{
                **common,
                "x": x,
                "bpos": bpos,
                "time_ps": (requested - window.anchor_index) * 10,
                "sampled_delta_time_ps": float(
                    np.diff(requested).min() * 10 if requested.size > 1 else 10
                ),
            }]
        )

    anchors = clip(window.anchor_indices)
    future = clip(window.future_indices)
    topology = StaticTopologyMetadata.from_batch(anchors)
    history = torch.from_numpy(bank[positions(window.history_indices)])
    raw = RawHistoryObservation(
        history,
        torch.as_tensor(
            (window.history_indices - window.anchor_index) * 10,
            dtype=torch.float32,
        )[None],
        torch.ones((1, history.shape[0]), dtype=torch.bool),
        topology,
        ChemicalMapping.from_topology(topology),
        history[-1],
    )
    return raw, anchors, future


def parse_raw_sample_id(sample_id: str) -> tuple[str, str, int]:
    match = _RAW_ID.fullmatch(str(sample_id))
    if match is None:
        raise ValueError(f"invalid raw-memory sample id: {sample_id}")
    return match.group("system"), match.group("replica"), int(match.group("window"))
