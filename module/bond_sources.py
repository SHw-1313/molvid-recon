"""Static graph-input bond sources for the controlled Phase-B ablation."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from .neighbor_graph import CudaRadiusNeighborList


def topology_id_from_record(record: Mapping[str, Any], *, require_stable: bool = False) -> str:
    """Return a stable identifier shared by clips/windows/replicas of a system."""

    explicit = record.get("topology_id")
    if explicit not in (None, ""):
        return str(explicit)
    system = record.get("system_id")
    fingerprint = record.get("topology_fingerprint")
    if system not in (None, "") and fingerprint not in (None, ""):
        return f"{system}::{fingerprint}"
    if system not in (None, ""):
        return str(system)
    if fingerprint not in (None, ""):
        return str(fingerprint)
    if require_stable:
        raise RuntimeError(
            "distance_only bond construction requires a stable topology_id, "
            "system_id, or topology_fingerprint on every sample"
        )
    return str(record.get("sample_id", ""))


def _identity_hash(record: Mapping[str, Any], *, atom_count: int) -> str:
    identities = record.get("atom_identity")
    if not isinstance(identities, list) or len(identities) != int(atom_count):
        raise RuntimeError(
            "distance_only references require one atom_identity entry per atom"
        )
    payload = json.dumps([str(value) for value in identities], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_canonical_reference_index(
    dataset: Any,
    *,
    source_split: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Select deterministic references from one explicitly selected split.

    ``source_split`` is provenance, not a filter inferred from record order.
    If records carry a split field it is checked against the explicit value;
    callers must pass the train dataset when constructing training references.
    """

    references: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        record = dataset[index]
        if source_split is not None and "split" in record:
            actual_split = str(record["split"])
            if actual_split != str(source_split):
                raise RuntimeError(
                    f"canonical reference record {index} has split {actual_split!r}, "
                    f"expected {source_split!r}"
                )
        key = topology_id_from_record(record, require_stable=True)
        sample_id = str(record.get("sample_id", index))
        coordinates = torch.as_tensor(record["x"][0], dtype=torch.float32).contiguous()
        if coordinates.device.type != "cpu":
            raise RuntimeError(
                "canonical distance-only references must be selected from CPU dataset coordinates"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise RuntimeError(f"canonical reference {key!r} must have shape [N, 3]")
        if not torch.isfinite(coordinates).all():
            raise RuntimeError(f"canonical reference {key!r} contains non-finite coordinates")
        atom_count = int(coordinates.shape[0])
        coordinate_hash = hashlib.sha256(coordinates.numpy().tobytes()).hexdigest()
        identity_hash = _identity_hash(record, atom_count=atom_count)
        current = references.get(key)
        if current is not None and sample_id == str(current["sample_id"]):
            if (
                int(current["atom_count"]) != atom_count
                or str(current["coordinate_sha256"]) != coordinate_hash
                or str(current["atom_identity_sha256"]) != identity_hash
            ):
                raise RuntimeError(
                    f"canonical reference sample_id {sample_id!r} has conflicting data for {key!r}"
                )
        if current is None or sample_id < str(current["sample_id"]):
            references[key] = {
                "sample_id": sample_id,
                "coordinates": coordinates,
                "atom_count": atom_count,
                "coordinate_sha256": coordinate_hash,
                "atom_identity_sha256": identity_hash,
                "source_split": str(source_split) if source_split is not None else str(record.get("split", "unknown")),
            }
    if not references:
        raise RuntimeError("distance_only mode found no canonical references")
    return references


class DistanceOnlyBondCache:
    """Bounded CUDA cache of canonical-reference distance-inferred bonds."""

    def __init__(
        self,
        *,
        min_distance_angstrom: float = 0.5,
        max_distance_angstrom: float = 2.2,
        max_num_neighbors: int = 64,
        capacity: int = 4096,
    ) -> None:
        if not 0 <= float(min_distance_angstrom) < float(max_distance_angstrom):
            raise ValueError("invalid distance-only bond interval")
        if int(max_num_neighbors) < 1 or int(capacity) < 1:
            raise ValueError("distance-only cache limits must be positive")
        self.min_distance = float(min_distance_angstrom)
        self.max_distance = float(max_distance_angstrom)
        self.max_num_neighbors = int(max_num_neighbors)
        self.capacity = int(capacity)
        self._entries: OrderedDict[tuple[str, str, str, float, float, int], Tensor] = OrderedDict()
        self._entry_metadata: dict[tuple[str, str, str, float, float, int], dict[str, Any]] = {}
        self._active_keys: dict[tuple[str, str], tuple[str, str, str, float, float, int]] = {}
        # Canonical CPU references remain available after a bounded CUDA
        # entry is evicted. A later materialize() deterministically rebuilds
        # the same frozen graph instead of failing on the next dataset pass.
        self._canonical_sources: dict[
            str, tuple[str, Tensor, int, str | None, str | None, str | None]
        ] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def register_reference(
        self,
        topology_id: str,
        coordinates: Tensor,
        *,
        device: torch.device,
        atom_identity_sha256: str | None = None,
        sample_id: str | None = None,
        source_split: str | None = None,
    ) -> dict[str, Any]:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "distance_only bond inference requires CUDA; refusing a CPU fallback"
            )
        source = torch.as_tensor(coordinates, dtype=torch.float32).contiguous()
        if source.device.type != "cpu":
            raise RuntimeError(
                "canonical reference hashing must start from CPU dataset coordinates; "
                "GPU-to-CPU graph round trips are forbidden"
            )
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("canonical reference must have shape [N, 3]")
        reference_hash = hashlib.sha256(source.numpy().tobytes()).hexdigest()
        identity_hash = (
            None if atom_identity_sha256 in (None, "") else str(atom_identity_sha256)
        )
        if identity_hash is not None:
            if len(identity_hash) != 64 or any(value not in "0123456789abcdef" for value in identity_hash.lower()):
                raise ValueError("atom_identity_sha256 must be a SHA-256 hex digest")
            identity_hash = identity_hash.lower()
        reference = source.to(device=device, dtype=torch.float32)
        if not torch.isfinite(reference).all():
            raise ValueError("canonical reference contains non-finite coordinates")
        # The graph kernel and all geometry are FP32.  Use the next FP32
        # number above the requested upper boundary because radius_graph has
        # an exclusive upper comparison; then enforce the closed interval
        # explicitly.  The lower boundary remains strict.
        min_distance_fp32 = torch.tensor(
            self.min_distance, dtype=torch.float32
        ).item()
        max_distance_fp32 = torch.nextafter(
            torch.tensor(self.max_distance, dtype=torch.float32),
            torch.tensor(float("inf"), dtype=torch.float32),
        ).item()
        finder = CudaRadiusNeighborList(
            cutoff_lower=min_distance_fp32,
            cutoff_upper=max_distance_fp32,
            max_num_neighbors=self.max_num_neighbors,
            loop=False,
            strict_cap=True,
        )
        with torch.no_grad():
            neighbors = finder(
                reference,
                torch.zeros(reference.shape[0], device=device, dtype=torch.long),
                batch_size=1,
            )
            lower = torch.tensor(min_distance_fp32, device=device, dtype=torch.float32)
            upper = torch.tensor(max_distance_fp32, device=device, dtype=torch.float32)
            closed_keep = (neighbors.edge_weight > lower) & (neighbors.edge_weight <= upper)
            edge_index = neighbors.edge_index[:, closed_keep]
            if edge_index.numel():
                counts = torch.bincount(
                    edge_index[1], minlength=int(reference.shape[0])
                )
                if int(counts.max().item()) >= self.max_num_neighbors:
                    raise RuntimeError(
                        f"distance-only neighbor cap {self.max_num_neighbors} is saturated "
                        f"for topology_id {topology_id!r}"
                    )
                pairs = torch.sort(edge_index, dim=0).values
                codes = pairs[0] * int(reference.shape[0]) + pairs[1]
                order = torch.argsort(codes)
                sorted_codes = codes[order]
                keep = torch.ones(sorted_codes.shape, device=device, dtype=torch.bool)
                if keep.numel() > 1:
                    keep[1:] = sorted_codes[1:] != sorted_codes[:-1]
                undirected = pairs[:, order[keep]]
                directed = torch.cat([undirected, undirected.flip(0)], dim=1)
            else:
                directed = torch.empty((2, 0), device=device, dtype=torch.long)
        atom_count = int(source.shape[0])
        inferred_bond_hash = hashlib.sha256(
            directed.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        active_key = (str(device), str(topology_id))
        previous_source = self._canonical_sources.get(str(topology_id))
        if previous_source is not None:
            if previous_source[0] != reference_hash:
                raise RuntimeError(
                    f"distance-only topology_id {topology_id!r} was registered with "
                    "conflicting canonical reference coordinates"
                )
            if (
                previous_source[3] is not None
                and identity_hash is not None
                and previous_source[3] != identity_hash
            ):
                raise RuntimeError(
                    f"distance-only topology_id {topology_id!r} was registered with "
                    "conflicting atom identity metadata"
                )
        effective_identity = identity_hash if identity_hash is not None else (
            previous_source[3] if previous_source is not None else None
        )
        effective_sample_id = sample_id if sample_id is not None else (
            previous_source[4] if previous_source is not None else None
        )
        effective_source_split = source_split if source_split is not None else (
            previous_source[5] if previous_source is not None else None
        )
        self._canonical_sources[str(topology_id)] = (
            reference_hash,
            source.detach().clone(),
            atom_count,
            effective_identity,
            effective_sample_id,
            effective_source_split,
        )
        key = (
            str(device),
            str(topology_id),
            reference_hash,
            self.min_distance,
            self.max_distance,
            self.max_num_neighbors,
        )
        previous_key = self._active_keys.get(active_key)
        if previous_key is not None and previous_key != key:
            raise RuntimeError(
                f"distance-only topology_id {topology_id!r} was registered with "
                "a conflicting canonical reference or cutoff"
            )
        metadata = {
            "topology_id": str(topology_id),
            "atom_count": atom_count,
            "coordinate_sha256": reference_hash,
            "inferred_bond_sha256": inferred_bond_hash,
            "sample_id": effective_sample_id,
            "source_split": effective_source_split,
            "atom_identity_sha256": effective_identity,
        }
        existing = self._entries.get(key)
        if existing is not None:
            if not torch.equal(existing, directed):
                raise RuntimeError(
                    f"distance-only topology_id {topology_id!r} has conflicting inferred bonds"
                )
            self._entry_metadata[key] = metadata
            self._active_keys[active_key] = key
            self._entries.move_to_end(key)
            return metadata.copy()
        self._entries[key] = directed.detach()
        self._entry_metadata[key] = metadata
        self._active_keys[active_key] = key
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            evicted_key, _ = self._entries.popitem(last=False)
            self._entry_metadata.pop(evicted_key, None)
            self._evictions += 1
            for active, active_entry in tuple(self._active_keys.items()):
                if active_entry == evicted_key:
                    del self._active_keys[active]
        return metadata.copy()

    def materialize(
        self,
        topology_id: str,
        *,
        device: torch.device,
        atom_count: int,
        atom_identity_sha256: str | None = None,
    ) -> Tensor:
        active_key = (str(device), str(topology_id))
        key = self._active_keys.get(active_key)
        value = self._entries.get(key) if key is not None else None
        if value is None:
            source_info = self._canonical_sources.get(str(topology_id))
            if source_info is not None:
                (
                    _reference_hash,
                    source,
                    expected_atoms,
                    expected_identity,
                    expected_sample_id,
                    expected_split,
                ) = source_info
                if int(atom_count) != expected_atoms:
                    raise RuntimeError(
                        f"distance-only atom count mismatch for {topology_id!r}: "
                        f"batch={int(atom_count)}, reference={expected_atoms}"
                    )
                self._misses += 1
                self.register_reference(
                    str(topology_id),
                    source,
                    device=device,
                    atom_identity_sha256=expected_identity,
                    sample_id=expected_sample_id,
                    source_split=expected_split,
                )
                key = self._active_keys.get(active_key)
                value = self._entries.get(key) if key is not None else None
            if value is None:
                self._misses += 1
                raise RuntimeError(
                    f"distance-only bonds for topology_id {topology_id!r} were not precomputed; "
                    "register canonical CPU references before inference"
                )
        metadata = self._entry_metadata.get(key)
        if metadata is None:
            raise RuntimeError(f"distance-only cache metadata is missing for {topology_id!r}")
        if int(metadata["atom_count"]) != int(atom_count):
            raise RuntimeError(
                f"distance-only atom count mismatch for {topology_id!r}: "
                f"batch={int(atom_count)}, reference={metadata['atom_count']}"
            )
        expected_identity = metadata.get("atom_identity_sha256")
        if expected_identity is not None:
            actual_identity = (
                None if atom_identity_sha256 in (None, "") else str(atom_identity_sha256)
            )
            if actual_identity != expected_identity:
                raise RuntimeError(
                    f"distance-only atom identity mismatch for {topology_id!r}: "
                    f"batch={actual_identity!r}, reference={expected_identity!r}"
                )
        self._hits += 1
        self._entries.move_to_end(key)
        return value

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "canonical_sources": len(self._canonical_sources),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "capacity": self.capacity,
        }


__all__ = [
    "DistanceOnlyBondCache",
    "build_canonical_reference_index",
    "topology_id_from_record",
]
