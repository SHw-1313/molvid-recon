"""Deterministic data and RNG scheduling for interleaved static updates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import re
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from ..data.sampling import get_clip_specs
from ..data.static import trajectory_record_to_static
from ..runtime import canonical_hash


@dataclass(frozen=True)
class StaticSourceEntry:
    """One auditable clean structure source; time is provenance, not a label."""

    dataset_index: int
    source_sample_id: str
    atoms: int
    frame_index: int
    system_id: str
    replica: str
    source_kind: str

    def contract(self) -> dict[str, Any]:
        return asdict(self)


def internal_md_static_pool(dataset: Any) -> tuple[StaticSourceEntry, ...]:
    """Select one midpoint frame for every unique ATLAS train trajectory."""

    selected: dict[tuple[str, str], StaticSourceEntry] = {}
    for item in get_clip_specs(dataset):
        match = re.match(r"^atlas_(.+)_(R[123])_", str(item.sample_id))
        if match is None:
            continue
        system, replica = match.groups()
        key = system, replica
        selected.setdefault(
            key,
            StaticSourceEntry(
                dataset_index=int(item.index),
                source_sample_id=str(item.sample_id),
                atoms=int(item.atoms),
                frame_index=int(item.frames) // 2,
                system_id=system,
                replica=replica,
                source_kind="internal_md_train",
            ),
        )
    entries = tuple(selected[key] for key in sorted(selected))
    systems = {entry.system_id for entry in entries}
    if not entries or len(entries) != 3 * len(systems):
        raise ValueError("internal static pool must contain one entry per system/replica")
    for system in systems:
        if {entry.replica for entry in entries if entry.system_id == system} != {
            "R1", "R2", "R3"
        }:
            raise ValueError("internal static pool lacks a complete three-replica system")
    return entries


def external_static_pool(
    dataset: Any,
    selected_indices: Sequence[int],
) -> tuple[StaticSourceEntry, ...]:
    """Register a pre-audited external T=1 selection without opening targets."""

    specs = get_clip_specs(dataset)
    entries = []
    for raw_index in selected_indices:
        index = int(raw_index)
        spec = specs[index]
        if int(spec.frames) != 1:
            raise ValueError("external static eligibility selected a non-static record")
        entries.append(
            StaticSourceEntry(
                dataset_index=index,
                source_sample_id=str(spec.sample_id or f"external_{index:08d}"),
                atoms=int(spec.atoms),
                frame_index=0,
                system_id="external",
                replica="not_applicable",
                source_kind="external_static",
            )
        )
    if not 2000 <= len(entries) <= 5000:
        raise ValueError("external static selection must contain 2000--5000 structures")
    if len({entry.dataset_index for entry in entries}) != len(entries):
        raise ValueError("external static selection contains duplicate indices")
    return tuple(entries)


def materialize_static_records(
    dataset: Any,
    entries: Sequence[StaticSourceEntry],
) -> list[Mapping[str, Any]]:
    """Open only the scheduled sources and retain explicit provenance."""

    result = []
    for entry in entries:
        record = dataset[entry.dataset_index]
        if entry.source_kind == "internal_md_train":
            if str(record.get("sample_id", "")) != entry.source_sample_id:
                raise RuntimeError("internal static dataset index changed sample identity")
            result.append(
                trajectory_record_to_static(
                    record,
                    frame_index=entry.frame_index,
                    sample_id=(
                        f"static_decoder::{entry.system_id}::{entry.replica}::"
                        f"{entry.frame_index}"
                    ),
                )
            )
        else:
            if int(record["x"].shape[0]) != 1:
                raise RuntimeError("external static source is not T=1")
            result.append(record)
    return result


class StaticBatchSchedule:
    """Independent epoch/cursor schedule packed under an atom-count bound."""

    def __init__(
        self,
        entries: Sequence[StaticSourceEntry],
        *,
        seed: int,
        max_atoms_per_update: int,
    ) -> None:
        self.entries = tuple(entries)
        self.seed = int(seed)
        self.max_atoms_per_update = int(max_atoms_per_update)
        if not self.entries or self.max_atoms_per_update < 1:
            raise ValueError("static schedule requires entries and a positive atom bound")
        if max(entry.atoms for entry in self.entries) > self.max_atoms_per_update:
            raise ValueError("static atom bound is smaller than an eligible source")
        self.pool_hash = canonical_hash([entry.contract() for entry in self.entries])
        self.epoch = 0
        self.batch_index = 0
        self.successful_static_updates = 0
        self.exposed_samples = 0
        self.exposed_atoms = 0
        self._batches = self._build_epoch(self.epoch)
        self._pending_indices: tuple[int, ...] | None = None

    def _build_epoch(self, epoch: int) -> tuple[tuple[int, ...], ...]:
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + 104729 * int(epoch)
        )
        order = torch.randperm(len(self.entries), generator=generator).tolist()
        batches: list[tuple[int, ...]] = []
        current: list[int] = []
        atoms = 0
        for index in order:
            count = self.entries[index].atoms
            if current and atoms + count > self.max_atoms_per_update:
                batches.append(tuple(current))
                current, atoms = [], 0
            current.append(index)
            atoms += count
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def next_entries(self) -> tuple[StaticSourceEntry, ...]:
        if self._pending_indices is not None:
            raise RuntimeError("static schedule already has an uncommitted batch")
        if self.batch_index >= len(self._batches):
            self.epoch += 1
            self.batch_index = 0
            self._batches = self._build_epoch(self.epoch)
        self._pending_indices = self._batches[self.batch_index]
        return tuple(self.entries[index] for index in self._pending_indices)

    def record_success(self, entries: Sequence[StaticSourceEntry]) -> None:
        if self._pending_indices is None:
            raise RuntimeError("static schedule has no pending batch to commit")
        expected = tuple(self.entries[index] for index in self._pending_indices)
        if tuple(entries) != expected:
            raise RuntimeError("static schedule committed a different source batch")
        self.batch_index += 1
        self._pending_indices = None
        self.successful_static_updates += 1
        self.exposed_samples += len(entries)
        self.exposed_atoms += sum(entry.atoms for entry in entries)

    def state_dict(self) -> dict[str, Any]:
        if self._pending_indices is not None:
            raise RuntimeError("cannot checkpoint an uncommitted static batch")
        return {
            "schema": "molvid.static_decoder.schedule.v1",
            "seed": self.seed,
            "max_atoms_per_update": self.max_atoms_per_update,
            "pool_hash": self.pool_hash,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "successful_static_updates": self.successful_static_updates,
            "exposed_samples": self.exposed_samples,
            "exposed_atoms": self.exposed_atoms,
            "epoch_batch_hash": canonical_hash(self._batches),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema": "molvid.static_decoder.schedule.v1",
            "seed": self.seed,
            "max_atoms_per_update": self.max_atoms_per_update,
            "pool_hash": self.pool_hash,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"static schedule resume differs at {name}")
        self.epoch = int(state["epoch"])
        self.batch_index = int(state["batch_index"])
        self.successful_static_updates = int(state["successful_static_updates"])
        self.exposed_samples = int(state["exposed_samples"])
        self.exposed_atoms = int(state["exposed_atoms"])
        self._batches = self._build_epoch(self.epoch)
        self._pending_indices = None
        if state.get("epoch_batch_hash") != canonical_hash(self._batches):
            raise ValueError("static schedule epoch batches differ on resume")
        if not 0 <= self.batch_index <= len(self._batches):
            raise ValueError("static schedule resume cursor is out of range")

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "molvid.static_decoder.schedule_contract.v1",
            "seed": self.seed,
            "max_atoms_per_update": self.max_atoms_per_update,
            "pool_hash": self.pool_hash,
            "pool_size": len(self.entries),
            "source_kind": sorted({entry.source_kind for entry in self.entries}),
        }


class StaticRNGStreams:
    """Noise and module RNG state isolated from all later dynamic updates."""

    def __init__(self, *, device: torch.device, seed: int) -> None:
        if device.type != "cuda":
            raise RuntimeError("formal static RNG streams require CUDA")
        self.device = device
        self.seed = int(seed)
        self.perturbation = torch.Generator(device=device).manual_seed(self.seed)
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.cuda.manual_seed(self.seed + 1)
            self.model_state = torch.cuda.get_rng_state(device).detach().cpu()

    @contextmanager
    def model_random_context(self) -> Iterator[None]:
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            torch.cuda.set_rng_state(self.model_state, self.device)
            yield
            self.model_state = torch.cuda.get_rng_state(self.device).detach().cpu()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "molvid.static_decoder.rng.v1",
            "seed": self.seed,
            "perturbation_generator_state": self.perturbation.get_state().detach().cpu(),
            "model_rng_state": self.model_state.detach().cpu(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "molvid.static_decoder.rng.v1":
            raise ValueError("static RNG resume schema differs")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("static RNG resume seed differs")
        perturbation = state.get("perturbation_generator_state")
        model = state.get("model_rng_state")
        if not isinstance(perturbation, Tensor) or not isinstance(model, Tensor):
            raise ValueError("static RNG resume lacks tensor states")
        self.perturbation.set_state(perturbation.detach().cpu())
        self.model_state = model.detach().cpu()
