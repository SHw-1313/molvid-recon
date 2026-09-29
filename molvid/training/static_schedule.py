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
    supervised_atoms: int

    def contract(self) -> dict[str, Any]:
        return asdict(self)


def internal_md_static_pool(
    dataset: Any,
    *,
    max_source_atoms: int | None = None,
) -> tuple[StaticSourceEntry, ...]:
    """Select one midpoint frame from each eligible ATLAS train trajectory."""

    selected: dict[tuple[str, str], StaticSourceEntry] = {}
    for item in get_clip_specs(dataset):
        match = re.match(r"^atlas_(.+)_(R[123])_", str(item.sample_id))
        if match is None:
            continue
        system, replica = match.groups()
        key = system, replica
        if max_source_atoms is not None and int(item.atoms) > int(max_source_atoms):
            continue
        if key in selected:
            continue
        record = dataset[int(item.index)]
        loss_mask = torch.as_tensor(record["loss_mask"], dtype=torch.bool)
        supervised_atoms = int(loss_mask.sum())
        if loss_mask.shape != (int(item.atoms),) or supervised_atoms < 1:
            raise ValueError("internal static source has an invalid loss mask")
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
                supervised_atoms=supervised_atoms,
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
    """Register only the pre-audited external records selected for training."""

    indices = tuple(int(value) for value in selected_indices)
    if not 2000 <= len(indices) <= 5000:
        raise ValueError("external static selection must contain 2000--5000 structures")
    if len(set(indices)) != len(indices):
        raise ValueError("external static selection contains duplicate indices")
    entries = []
    for index in indices:
        record = dataset[index]
        coordinates = torch.as_tensor(record["x"])
        if coordinates.ndim != 3 or int(coordinates.shape[0]) != 1:
            raise ValueError("external static eligibility selected a non-static record")
        supervised_atoms = int(
            torch.as_tensor(record["loss_mask"], dtype=torch.bool).sum()
        )
        if supervised_atoms < 1 or supervised_atoms > int(coordinates.shape[1]):
            raise ValueError("external static eligibility selected an invalid loss mask")
        entries.append(
            StaticSourceEntry(
                dataset_index=index,
                source_sample_id=str(record.get("sample_id", f"external_{index:08d}")),
                atoms=int(coordinates.shape[1]),
                frame_index=0,
                system_id=str(record.get("legacy_id", "external")),
                replica="not_applicable",
                source_kind="external_static",
                supervised_atoms=supervised_atoms,
            )
        )
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
            if int(torch.as_tensor(record["loss_mask"], dtype=torch.bool).sum()) != entry.supervised_atoms:
                raise RuntimeError("internal static dataset index changed supervised atom count")
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
            if str(record.get("sample_id", "")) != entry.source_sample_id:
                raise RuntimeError("external static dataset index changed sample identity")
            if int(record["x"].shape[1]) != entry.atoms:
                raise RuntimeError("external static dataset index changed atom count")
            if int(torch.as_tensor(record["loss_mask"], dtype=torch.bool).sum()) != entry.supervised_atoms:
                raise RuntimeError("external static dataset index changed supervised atom count")
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
        self.exposed_supervised_atoms = 0
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
        self.exposed_supervised_atoms += sum(entry.supervised_atoms for entry in entries)

    def preview_exposure(self, updates: int) -> dict[str, int]:
        """Return the deterministic exposure without advancing this schedule."""

        if isinstance(updates, bool) or int(updates) < 1:
            raise ValueError("static exposure preview requires positive updates")
        epoch = 0
        batch_index = 0
        batches = self._build_epoch(epoch)
        samples = atoms = supervised_atoms = 0
        for _ in range(int(updates)):
            if batch_index >= len(batches):
                epoch += 1
                batch_index = 0
                batches = self._build_epoch(epoch)
            indices = batches[batch_index]
            batch_index += 1
            samples += len(indices)
            atoms += sum(self.entries[index].atoms for index in indices)
            supervised_atoms += sum(
                self.entries[index].supervised_atoms for index in indices
            )
        return {
            "static_updates": int(updates),
            "samples": samples,
            "atoms": atoms,
            "supervised_atoms": supervised_atoms,
        }

    def state_dict(self) -> dict[str, Any]:
        if self._pending_indices is not None:
            raise RuntimeError("cannot checkpoint an uncommitted static batch")
        return {
            "schema": "molvid.static_decoder.schedule.v2",
            "seed": self.seed,
            "max_atoms_per_update": self.max_atoms_per_update,
            "pool_hash": self.pool_hash,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "successful_static_updates": self.successful_static_updates,
            "exposed_samples": self.exposed_samples,
            "exposed_atoms": self.exposed_atoms,
            "exposed_supervised_atoms": self.exposed_supervised_atoms,
            "epoch_batch_hash": canonical_hash(self._batches),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema": "molvid.static_decoder.schedule.v2",
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
        self.exposed_supervised_atoms = int(state["exposed_supervised_atoms"])
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
            "supervision": "loss_mask atoms only",
        }


def validate_static_exposure_plan(
    plan: Mapping[str, int],
    target: Mapping[str, int],
) -> None:
    """Fail closed unless a planned arm meets the shared exposure target."""

    expected_keys = {
        "static_updates", "samples", "atoms", "supervised_atoms",
        "sample_tolerance", "atom_tolerance", "supervised_atom_tolerance",
    }
    if set(target) != expected_keys:
        raise ValueError("static exposure target keys differ from the frozen contract")
    for name, value in target.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"static exposure target {name} must be a non-negative integer")
    if target["static_updates"] < 1:
        raise ValueError("static exposure target requires positive static updates")
    if int(plan["static_updates"]) != target["static_updates"]:
        raise ValueError("static update exposure differs from the shared target")
    checks = (
        ("samples", "sample_tolerance"),
        ("atoms", "atom_tolerance"),
        ("supervised_atoms", "supervised_atom_tolerance"),
    )
    for name, tolerance in checks:
        if abs(int(plan[name]) - target[name]) > target[tolerance]:
            raise ValueError(f"static {name} exposure differs from the shared target")


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
