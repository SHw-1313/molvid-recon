"""Bounded coordinate-independent topology caches for graph input."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class TopologyEntry:
    topology_id: str
    atom_count: int
    bond_index_cpu: Tensor
    bond_codes_cpu: Tensor


class BoundedTopologyCache:
    """CPU canonical topology cache plus bounded per-device materialization.

    The cache is deliberately not an ``nn.Module`` and contains no registered
    tensors, so it cannot alter a model state dict.  Canonical entries are
    registered from CPU dataset batches before those batches are transferred to
    CUDA.  A CUDA forward therefore never performs a GPU-to-CPU graph round
    trip.
    """

    def __init__(self, *, max_canonical_entries: int = 4096, max_device_entries: int = 4096):
        if max_canonical_entries < 1 or max_device_entries < 1:
            raise ValueError("topology cache capacities must be positive")
        self.max_canonical_entries = int(max_canonical_entries)
        self.max_device_entries = int(max_device_entries)
        self._canonical: OrderedDict[str, TopologyEntry] = OrderedDict()
        self._device: OrderedDict[tuple[str, str], Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _normalize_bonds(bond_index: Tensor, atom_count: int) -> tuple[Tensor, Tensor]:
        bond = torch.as_tensor(bond_index, dtype=torch.long)
        if bond.numel() == 0:
            bond = torch.empty((2, 0), dtype=torch.long)
        if bond.ndim != 2 or bond.shape[0] != 2:
            raise ValueError("bond_index must have shape [2, E]")
        if torch.any(bond < 0) or torch.any(bond >= int(atom_count)):
            raise ValueError("topology bond contains an out-of-range atom")
        if bond.numel() and torch.any(bond[0] == bond[1]):
            raise ValueError("topology bond_index must not contain self edges")
        if bond.numel() == 0:
            return bond.contiguous(), torch.empty((0,), dtype=torch.long)
        codes = bond[0] * int(atom_count) + bond[1]
        order = torch.argsort(codes)
        sorted_codes = codes[order]
        keep = torch.ones(sorted_codes.shape, dtype=torch.bool)
        if keep.numel() > 1:
            keep[1:] = sorted_codes[1:] != sorted_codes[:-1]
        normalized = bond[:, order[keep]].contiguous()
        return normalized, sorted_codes[keep].contiguous()

    def register(self, topology_id: str, bond_index: Tensor, *, atom_count: int) -> None:
        key = str(topology_id)
        if bond_index.device.type != "cpu":
            raise RuntimeError(
                "topology cache registration must happen before CUDA transfer; "
                "GPU-to-CPU graph round trips are forbidden"
            )
        normalized, codes = self._normalize_bonds(bond_index, atom_count)
        existing = self._canonical.get(key)
        if existing is not None:
            if existing.atom_count != int(atom_count) or not torch.equal(
                existing.bond_index_cpu, normalized
            ):
                raise RuntimeError(
                    f"topology_id {key!r} was observed with conflicting atom/bond topology"
                )
            self._canonical.move_to_end(key)
            return
        self._canonical[key] = TopologyEntry(
            topology_id=key,
            atom_count=int(atom_count),
            bond_index_cpu=normalized,
            bond_codes_cpu=codes,
        )
        self._canonical.move_to_end(key)
        while len(self._canonical) > self.max_canonical_entries:
            evicted, _ = self._canonical.popitem(last=False)
            self._evictions += 1
            for device_key in tuple(self._device):
                if device_key[1] == evicted:
                    del self._device[device_key]

    def register_packed_batch(self, batch: Any) -> None:
        """Register all per-sample local bonds from a CPU ``ClipBatch``."""

        topology_ids = tuple(str(value) for value in batch.topology_id)
        atom_ptr = torch.as_tensor(batch.atom_ptr, dtype=torch.long)
        abid = torch.as_tensor(batch.abid, dtype=torch.long)
        bond_index = torch.as_tensor(batch.bond_index, dtype=torch.long)
        if atom_ptr.device.type != "cpu" or abid.device.type != "cpu" or bond_index.device.type != "cpu":
            raise RuntimeError("register_packed_batch requires CPU batch tensors")
        if bond_index.numel() and torch.any(
            abid[bond_index[0]] != abid[bond_index[1]]
        ):
            raise RuntimeError("packed bond index crosses sample boundaries")
        for sample_index, topology_id in enumerate(topology_ids):
            start = int(atom_ptr[sample_index])
            stop = int(atom_ptr[sample_index + 1])
            if bond_index.numel():
                sample_mask = abid[bond_index[0]] == sample_index
                local = bond_index[:, sample_mask] - start
            else:
                local = torch.empty((2, 0), dtype=torch.long)
            self.register(topology_id, local, atom_count=stop - start)

    def is_registered(self, topology_id: str, *, atom_count: int | None = None) -> bool:
        entry = self._canonical.get(str(topology_id))
        if entry is None:
            return False
        if atom_count is not None and entry.atom_count != int(atom_count):
            return False
        self._canonical.move_to_end(str(topology_id))
        return True

    def materialize(
        self,
        topology_id: str,
        *,
        device: torch.device,
        atom_count: int,
        allow_cpu_test: bool = False,
    ) -> Tensor:
        key = str(topology_id)
        if device.type != "cuda" and not allow_cpu_test:
            raise RuntimeError(
                "production topology materialization requires CUDA; "
                "CPU materialization is only allowed for the explicit dense_test backend"
            )
        entry = self._canonical.get(key)
        if entry is None:
            self._misses += 1
            raise RuntimeError(
                f"topology_id {key!r} is not registered; register the CPU batch "
                "before transferring it to CUDA"
            )
        if entry.atom_count != int(atom_count):
            raise RuntimeError(f"topology_id {key!r} atom count changed")
        self._canonical.move_to_end(key)
        device_key = (str(device), key)
        cached = self._device.get(device_key)
        if cached is not None:
            self._hits += 1
            self._device.move_to_end(device_key)
            return cached
        self._misses += 1
        materialized = entry.bond_index_cpu.to(device=device, dtype=torch.long)
        self._device[device_key] = materialized
        self._device.move_to_end(device_key)
        while len(self._device) > self.max_device_entries:
            self._device.popitem(last=False)
            self._evictions += 1
        return materialized

    def stats(self) -> dict[str, int]:
        return {
            "canonical_entries": len(self._canonical),
            "device_entries": len(self._device),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }


__all__ = ["BoundedTopologyCache", "TopologyEntry"]
