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


_DT100_ID = re.compile(
    r"^atlas_(?P<system>.+)_(?P<replica>R[123])_dt_100ps_w(?P<window>[0-9]+)$"
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
        split: str = "train",
    ) -> None:
        if split not in {"train", "valid"}:
            raise ValueError("raw learned-memory dataset supports train/valid only")
        self.manifest_root = Path(manifest_root).resolve()
        self.raw_root = Path(raw_root).resolve()
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
                match = _DT100_ID.fullmatch(fields[0])
                if match is None:
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
            # The dt100 store uses source_stride=10 and window_stride=16.
            # w>=8 supplies 1280 raw past frames.  w<=max-3 ensures that the
            # raw XTC prefix already evidenced by the store covers +480 frames.
            eligible = [window for window in sorted(available) if 8 <= window <= maximum - 3]
            if len(eligible) < 24:
                raise ValueError(f"trajectory {system}/{replica} has fewer than24 eligible anchors")
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
                        window_index=window, anchor_raw_index=window * 160,
                        maximum_evidenced_raw_index=maximum * 160 + 150,
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
            "anchor_rule": "dt100_window_start=w*160; w>=8; w<=maximum_w-3",
            "raw_inventory_hash": canonical_hash(inventory),
            "test_opened": False,
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
            )
            for index in indices
        ]
        return pack_pilot_inputs(values)


@lru_cache(maxsize=256)
def _metadata(pdb: str) -> Mapping[str, Any]:
    return topology_metadata(read_topology(Path(pdb)), complex_topology=False)


def _read_100ps_window(
    xtc: str,
    atom_source_index: tuple[int, ...],
    minimum_raw_index: int,
    maximum_raw_index: int,
):
    """Decode only the requested raw-XTC interval on the 100 ps grid.

    Formal epochs touch every physical trajectory but only a small part of
    each 100 ns file at a time.  Reading and retaining the complete evidenced
    prefix makes the first step for a cold file scan almost the entire XTC and
    leaves CUDA idle for minutes.  XTC seek is indexed, so a bounded interval
    preserves the raw-coordinate contract while keeping I/O proportional to
    the actual history and future requested by this update.
    """
    from mdtraj.formats import XTCTrajectoryFile

    if minimum_raw_index % 10 or maximum_raw_index % 10:
        raise ValueError("raw XTC interval must lie on the100 ps grid")
    count = (maximum_raw_index - minimum_raw_index) // 10 + 1
    with XTCTrajectoryFile(xtc, "r") as handle:
        handle.seek(minimum_raw_index)
        coordinates_nm, absolute_time_ps, _step, _box = handle.read(
            n_frames=count,
            stride=10,
            atom_indices=np.asarray(atom_source_index, dtype=np.int64),
        )
    if coordinates_nm.shape[0] != count:
        raise ValueError("raw XTC ended before the requested formal interval")
    raw_indices = np.arange(
        minimum_raw_index, maximum_raw_index + 1, 10, dtype=np.int64
    )
    times = np.asarray(absolute_time_ps, dtype=np.float64)
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


def _load_item(item: RawMemoryItem, window: RawHistoryWindow):
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
    coordinates_nm, absolute_time_ps, read_indices = _read_100ps_window(
        str(item.xtc),
        tuple(int(value) for value in atom_source_index.tolist()),
        minimum,
        maximum,
    )
    requested = np.unique(np.concatenate(
        (window.history_indices, window.anchor_indices, window.future_indices)
    ))
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
