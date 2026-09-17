"""Bounded source-center cache used by the source A/B runner profiles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import Tensor

from .state_detail_latent_adapter import LatentFieldSet


CACHE_SCHEMA = "pvb.dit.state_detail.latent_cache.v1"


def cache_key(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bytes(fields: LatentFieldSet) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for value in fields.as_dict().values())


def _cpu_copy(fields: LatentFieldSet) -> LatentFieldSet:
    return fields.map(lambda value: value.detach().to(device="cpu", dtype=torch.float32).contiguous())


@dataclass(frozen=True)
class CacheStats:
    mode: str
    capacity_bytes: int
    entries: int
    bytes: int
    hits: int
    misses: int
    evictions: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "mode": self.mode,
            "capacity_bytes": self.capacity_bytes,
            "entries": self.entries,
            "bytes": self.bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class LatentFieldCache:
    """A deterministic, bounded CPU RAM cache for normalized source centers."""

    def __init__(self, *, mode: str = "disabled", max_bytes: int = 8 * 1024**3) -> None:
        if mode not in ("disabled", "ram"):
            raise ValueError("cache mode must be disabled or ram")
        if int(max_bytes) < 1:
            raise ValueError("cache capacity must be positive")
        self.mode = str(mode)
        self.capacity_bytes = int(max_bytes)
        self._entries: OrderedDict[str, tuple[LatentFieldSet, int]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str, *, device: torch.device, dtype: torch.dtype) -> LatentFieldSet | None:
        if self.mode == "disabled":
            self.misses += 1
            return None
        value = self._entries.get(str(key))
        if value is None:
            self.misses += 1
            return None
        self._entries.move_to_end(str(key))
        self.hits += 1
        return value[0].map(lambda item: item.to(device=device, dtype=dtype))

    def put(self, key: str, fields: LatentFieldSet) -> None:
        if self.mode == "disabled":
            return
        stored = _cpu_copy(fields)
        size = _bytes(stored)
        if size > self.capacity_bytes:
            return
        old = self._entries.pop(str(key), None)
        if old is not None:
            self._bytes -= old[1]
        self._entries[str(key)] = (stored, size)
        self._bytes += size
        while self._bytes > self.capacity_bytes and self._entries:
            _key, (_value, old_size) = self._entries.popitem(last=False)
            self._bytes -= old_size
            self.evictions += 1

    def stats(self) -> CacheStats:
        return CacheStats(
            mode=self.mode,
            capacity_bytes=self.capacity_bytes,
            entries=len(self._entries),
            bytes=self._bytes,
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
        )


__all__ = ["CACHE_SCHEMA", "CacheStats", "LatentFieldCache", "cache_key"]
