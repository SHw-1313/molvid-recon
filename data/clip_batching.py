"""Task-aware dynamic batching for the packed multi-frame clip path.

The legacy PVB loaders have their own batching policy and are intentionally
not routed through this module.  A clip batch is first grouped by
``(task, frames, time_bucket_id)`` and then packed under a token-like bound.
The initial cost model is deliberately simple: one clip costs ``T * N``.

The sampler creates one deterministic global batch schedule and shards that
schedule by batch id for distributed training.  This is important: applying
``DistributedSampler`` to an already formed batch dataset can make every rank
iterate over the same logical batch.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import partial
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from .clip_dataset import (
    ClipBatch,
    collate_clip_records,
    STATIC_TASK,
    TASK_NAMES,
    TRAJECTORY_TASK,
    validate_clip_record,
)


_BUCKET_RE = re.compile(r"^dt_(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ps|ns)$")


def task_id(value: Any) -> int:
    """Normalize the public task spelling used by sampler configurations."""

    if isinstance(value, str):
        lowered = value.lower()
        for numeric, name in TASK_NAMES.items():
            if lowered == name:
                return numeric
    if isinstance(value, (int, np.integer)) and int(value) in TASK_NAMES:
        return int(value)
    raise ValueError(f"unknown clip task: {value!r}")


def task_name(value: Any) -> str:
    return TASK_NAMES[task_id(value)]


def bucket_delta_time_ps(bucket: str) -> float | None:
    """Decode the conventional bucket id for cheap sampler-side metadata.

    The actual batch log always uses the tensors in ``ClipBatch``.  This
    helper is only a fast estimate for stores whose index already contains a
    bucket id, and therefore returns ``None`` for custom bucket names.
    """

    if bucket == "static":
        return None
    match = _BUCKET_RE.match(str(bucket))
    if match is None:
        return None
    value = float(match.group("value"))
    if match.group("unit") == "ns":
        value *= 1000.0
    return value


@dataclass(frozen=True)
class ClipItemSpec:
    """Small sampler-side description of one clip.

    ``effective_tokens`` is the initial T*N complexity estimate.  It is not
    a model vocabulary token count and must not be confused with a compressed
    latent length.
    """

    index: int
    atoms: int
    frames: int
    task: int | str
    time_bucket_id: str
    native_delta_time_ps: float | None = None
    physical_clip_span_ps: float | None = None
    sample_id: str = ""

    def __post_init__(self) -> None:
        normalized_task = task_id(self.task)
        if int(self.index) < 0:
            raise ValueError("clip index must be non-negative")
        if int(self.atoms) < 1:
            raise ValueError("clip atoms must be positive")
        if int(self.frames) < 1:
            raise ValueError("clip frames must be positive")
        if normalized_task == STATIC_TASK and int(self.frames) != 1:
            raise ValueError("static clip specs must have T=1")
        if normalized_task == TRAJECTORY_TASK and int(self.frames) < 2:
            raise ValueError("trajectory clip specs must have T>=2")
        if not str(self.time_bucket_id):
            raise ValueError("clip time_bucket_id must be non-empty")
        if self.native_delta_time_ps is not None:
            value = float(self.native_delta_time_ps)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("native_delta_time_ps must be finite and positive")
            object.__setattr__(self, "native_delta_time_ps", value)
        if self.physical_clip_span_ps is not None:
            value = float(self.physical_clip_span_ps)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "physical_clip_span_ps must be finite and non-negative"
                )
            object.__setattr__(self, "physical_clip_span_ps", value)
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "atoms", int(self.atoms))
        object.__setattr__(self, "frames", int(self.frames))
        object.__setattr__(self, "task", normalized_task)
        object.__setattr__(self, "time_bucket_id", str(self.time_bucket_id))

    @property
    def effective_tokens(self) -> int:
        return self.atoms * self.frames

    @property
    def task_type(self) -> str:
        return TASK_NAMES[self.task]

    @property
    def group_key(self) -> tuple[int, int, str]:
        return self.task, self.frames, self.time_bucket_id


class ClipSpecTable(Sequence[ClipItemSpec]):
    """Compact columnar collection of clip specs.

    Large static stores can contain millions of records.  Keeping one Python
    dataclass and one Python integer per column is unnecessarily expensive, so
    this table stores numeric columns in NumPy arrays and only one string per
    homogeneous group.
    """

    def __init__(
        self,
        *,
        indices: Sequence[int] | np.ndarray,
        atoms: Sequence[int] | np.ndarray,
        frames: Sequence[int] | np.ndarray,
        tasks: Sequence[int] | np.ndarray,
        bucket_group_ids: Sequence[int] | np.ndarray,
        group_keys: Sequence[tuple[int, int, str]],
        native_delta_time_ps: Sequence[float] | np.ndarray | None = None,
        physical_clip_span_ps: Sequence[float] | np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.atoms = np.asarray(atoms, dtype=np.int64)
        self.frames = np.asarray(frames, dtype=np.int64)
        self.tasks = np.asarray(tasks, dtype=np.int64)
        self.bucket_group_ids = np.asarray(bucket_group_ids, dtype=np.int32)
        self.group_keys = tuple(
            (task_id(task), int(frame), str(bucket))
            for task, frame, bucket in group_keys
        )
        size = self.indices.size
        columns = {
            "atoms": self.atoms,
            "frames": self.frames,
            "tasks": self.tasks,
            "bucket_group_ids": self.bucket_group_ids,
        }
        if any(column.ndim != 1 or column.size != size for column in columns.values()):
            raise ValueError("clip spec columns must be equally sized 1-D arrays")
        if any(group_id < 0 or group_id >= len(self.group_keys)
               for group_id in self.bucket_group_ids.tolist()):
            raise ValueError("clip spec group id is out of range")
        if native_delta_time_ps is None:
            self.native_delta_time_ps = np.asarray(
                [
                    bucket_delta_time_ps(self.group_keys[group_id][2])
                    or np.nan
                    for group_id in self.bucket_group_ids
                ],
                dtype=np.float64,
            )
        else:
            self.native_delta_time_ps = np.asarray(
                native_delta_time_ps, dtype=np.float64
            )
            if self.native_delta_time_ps.ndim != 1 or self.native_delta_time_ps.size != size:
                raise ValueError("native_delta_time_ps must match spec count")
        if physical_clip_span_ps is None:
            self.physical_clip_span_ps = np.asarray(
                [
                    (int(self.frames[index]) - 1) * float(delta)
                    if math.isfinite(float(delta))
                    else np.nan
                    for index, delta in enumerate(self.native_delta_time_ps)
                ],
                dtype=np.float64,
            )
        else:
            self.physical_clip_span_ps = np.asarray(
                physical_clip_span_ps, dtype=np.float64
            )
            if self.physical_clip_span_ps.ndim != 1 or self.physical_clip_span_ps.size != size:
                raise ValueError("physical_clip_span_ps must match spec count")
        if sample_ids is None:
            self.sample_ids = None
        else:
            self.sample_ids = tuple(str(sample_id) for sample_id in sample_ids)
            if len(self.sample_ids) != size:
                raise ValueError("sample_ids must match spec count")

    @classmethod
    def from_specs(cls, specs: Iterable[ClipItemSpec]) -> "ClipSpecTable":
        items = list(specs)
        group_ids: dict[tuple[int, int, str], int] = {}
        group_keys: list[tuple[int, int, str]] = []
        bucket_ids: list[int] = []
        for item in items:
            key = item.group_key
            if key not in group_ids:
                group_ids[key] = len(group_keys)
                group_keys.append(key)
            bucket_ids.append(group_ids[key])
        return cls(
            indices=[item.index for item in items],
            atoms=[item.atoms for item in items],
            frames=[item.frames for item in items],
            tasks=[task_id(item.task) for item in items],
            bucket_group_ids=bucket_ids,
            group_keys=group_keys,
            native_delta_time_ps=[
                np.nan if item.native_delta_time_ps is None else item.native_delta_time_ps
                for item in items
            ],
            physical_clip_span_ps=[
                np.nan if item.physical_clip_span_ps is None else item.physical_clip_span_ps
                for item in items
            ],
            sample_ids=[item.sample_id for item in items],
        )

    @classmethod
    def from_clip_index(
        cls, entries: Sequence[tuple[str, int, int, int, int, str]]
    ) -> "ClipSpecTable":
        specs: list[ClipItemSpec] = []
        for index, entry in enumerate(entries):
            sample_id, _start, _end, atoms, frames, bucket = entry
            normalized_bucket = str(bucket)
            task = STATIC_TASK if normalized_bucket == "static" else TRAJECTORY_TASK
            delta = bucket_delta_time_ps(normalized_bucket)
            specs.append(
                ClipItemSpec(
                    index=index,
                    atoms=int(atoms),
                    frames=int(frames),
                    task=task,
                    time_bucket_id=normalized_bucket,
                    native_delta_time_ps=delta,
                    physical_clip_span_ps=(int(frames) - 1) * delta
                    if delta is not None
                    else 0.0,
                    sample_id=str(sample_id),
                )
            )
        return cls.from_specs(specs)

    @classmethod
    def from_static_index(
        cls,
        atom_counts: Sequence[int] | np.ndarray,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> "ClipSpecTable":
        atoms = np.asarray(atom_counts, dtype=np.int64)
        if atoms.ndim != 1:
            raise ValueError("static atom_counts must be 1-D")
        count = int(atoms.size)
        return cls(
            indices=np.arange(count, dtype=np.int64),
            atoms=atoms,
            frames=np.ones(count, dtype=np.int64),
            tasks=np.full(count, STATIC_TASK, dtype=np.int64),
            bucket_group_ids=np.zeros(count, dtype=np.int32),
            group_keys=((STATIC_TASK, 1, "static"),),
            native_delta_time_ps=np.full(count, np.nan, dtype=np.float64),
            physical_clip_span_ps=np.zeros(count, dtype=np.float64),
            sample_ids=sample_ids,
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> ClipItemSpec:
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        group_id = int(self.bucket_group_ids[position])
        native = float(self.native_delta_time_ps[position])
        span = float(self.physical_clip_span_ps[position])
        return ClipItemSpec(
            index=int(self.indices[position]),
            atoms=int(self.atoms[position]),
            frames=int(self.frames[position]),
            task=int(self.tasks[position]),
            time_bucket_id=self.group_keys[group_id][2],
            native_delta_time_ps=native if math.isfinite(native) else None,
            physical_clip_span_ps=span if math.isfinite(span) else None,
            sample_id=(self.sample_ids[position] if self.sample_ids is not None else ""),
        )

    def __iter__(self) -> Iterator[ClipItemSpec]:
        for position in range(len(self)):
            yield self[position]

    def group_positions(self) -> "OrderedDict[tuple[int, int, str], np.ndarray]":
        groups: "OrderedDict[tuple[int, int, str], np.ndarray]" = OrderedDict()
        for group_id, key in enumerate(self.group_keys):
            positions = np.flatnonzero(self.bucket_group_ids == group_id)
            if positions.size:
                groups[key] = positions.astype(np.int64, copy=False)
        return groups


def _record_spec(record: Mapping[str, Any], index: int) -> ClipItemSpec:
    normalized = validate_clip_record(record)
    time = np.asarray(normalized["time_ps"], dtype=np.float64)
    delta = np.asarray(normalized["delta_time_ps"], dtype=np.float64)
    task = task_id(normalized["task"])
    native = normalized.get("sampled_delta_time_ps", None)
    if native is None and delta.size:
        native = float(delta[0])
    return ClipItemSpec(
        index=index,
        atoms=int(normalized["x"].shape[1]),
        frames=int(normalized["x"].shape[0]),
        task=task,
        time_bucket_id=str(normalized["time_bucket_id"]),
        native_delta_time_ps=native,
        physical_clip_span_ps=float(time[-1] - time[0]),
        sample_id=str(normalized.get("sample_id", index)),
    )


def get_clip_specs(dataset: Any) -> ClipSpecTable:
    """Get cheap sampler metadata from a clip dataset.

    Dataset implementations may provide ``clip_spec_table``.  A generic
    fallback is retained for small synthetic datasets and validates records
    as it reads them; production clip stores use their indexed fast path.
    """

    if isinstance(dataset, ClipSpecTable):
        return dataset
    method = getattr(dataset, "clip_spec_table", None)
    if callable(method):
        table = method()
        if not isinstance(table, ClipSpecTable):
            raise TypeError("clip_spec_table() must return ClipSpecTable")
        return table
    if isinstance(dataset, torch.utils.data.Subset):
        base_table = get_clip_specs(dataset.dataset)
        specs = []
        for local_index, base_index in enumerate(dataset.indices):
            item = base_table[int(base_index)]
            specs.append(
                ClipItemSpec(
                    index=local_index,
                    atoms=item.atoms,
                    frames=item.frames,
                    task=item.task,
                    time_bucket_id=item.time_bucket_id,
                    native_delta_time_ps=item.native_delta_time_ps,
                    physical_clip_span_ps=item.physical_clip_span_ps,
                    sample_id=item.sample_id,
                )
            )
        return ClipSpecTable.from_specs(specs)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        tables: list[ClipSpecTable] = []
        offset = 0
        for child in dataset.datasets:
            child_table = get_clip_specs(child)
            items = []
            for item in child_table:
                items.append(
                    ClipItemSpec(
                        index=item.index + offset,
                        atoms=item.atoms,
                        frames=item.frames,
                        task=item.task,
                        time_bucket_id=item.time_bucket_id,
                        native_delta_time_ps=item.native_delta_time_ps,
                        physical_clip_span_ps=item.physical_clip_span_ps,
                        sample_id=item.sample_id,
                    )
                )
            tables.append(ClipSpecTable.from_specs(items))
            offset += len(child_table)
        return _concat_spec_tables(tables)
    specs = (_record_spec(dataset[index], index) for index in range(len(dataset)))
    return ClipSpecTable.from_specs(specs)


def _concat_spec_tables(tables: Sequence[ClipSpecTable]) -> ClipSpecTable:
    items: list[ClipItemSpec] = []
    for table in tables:
        items.extend(table)
    return ClipSpecTable.from_specs(items)


def _weight_value(mapping: Mapping[Any, float] | None, keys: Sequence[Any]) -> float:
    if mapping is None:
        return 1.0
    value: Any = None
    for key in keys:
        if key in mapping:
            value = mapping[key]
            break
    if value is None:
        return 1.0
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"sampling weights must be finite and non-negative: {value}")
    return value


class TaskAwareClipBatchSampler(Sampler[list[int]]):
    """Pack homogeneous clip batches under a deterministic ``T*N`` bound.

    Parameters
    ----------
    dataset:
        A clip dataset, a ``ConcatDataset`` of clip datasets, or a
        :class:`ClipSpecTable`.
    max_tokens:
        Maximum sum of ``frames * atoms`` in one batch.  ``ubound_per_batch``
        and ``max_complexity`` are accepted as explicit aliases for callers
        migrating from the legacy dynamic wrapper.
    task_weights, time_bucket_weights:
        Batch-level weights.  They are applied to homogeneous groups and are
        not multiplied by group or dataset size.  Set ``batches_per_epoch``
        to control the amount of sampling independently of data-set size.
    replacement:
        Whether a group's item cursor may wrap when a weighted schedule asks
        for more draws than that group contains.  Replacement is useful for
        balancing a small static set against a large trajectory set; it is
        still deterministic under ``seed`` and ``epoch``.
    """

    def __init__(
        self,
        dataset: Any,
        max_tokens: int | None = None,
        *,
        ubound_per_batch: int | None = None,
        max_complexity: int | None = None,
        task_weights: Mapping[Any, float] | None = None,
        time_bucket_weights: Mapping[str, float] | None = None,
        bucket_weights: Mapping[str, float] | None = None,
        batches_per_epoch: int | None = None,
        epoch_size: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
        replacement: bool = True,
        drop_last: bool = False,
        oversize_policy: str = "error",
    ) -> None:
        if max_tokens is None:
            max_tokens = ubound_per_batch
        if max_tokens is None:
            max_tokens = max_complexity
        if max_tokens is None:
            raise ValueError("max_tokens is required")
        if ubound_per_batch is not None and int(ubound_per_batch) != int(max_tokens):
            raise ValueError("max_tokens and ubound_per_batch disagree")
        if epoch_size is not None:
            if batches_per_epoch is not None and int(epoch_size) != int(batches_per_epoch):
                raise ValueError("batches_per_epoch and epoch_size disagree")
            batches_per_epoch = epoch_size
        if int(max_tokens) < 1:
            raise ValueError("max_tokens must be positive")
        if batches_per_epoch is not None and int(batches_per_epoch) < 1:
            raise ValueError("batches_per_epoch must be positive")
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if int(num_replicas) < 1:
            raise ValueError("num_replicas must be positive")
        if int(rank) < 0 or int(rank) >= int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas)")
        if time_bucket_weights is not None and bucket_weights is not None:
            if dict(time_bucket_weights) != dict(bucket_weights):
                raise ValueError("time_bucket_weights and bucket_weights disagree")
        if time_bucket_weights is None:
            time_bucket_weights = bucket_weights

        self.specs = get_clip_specs(dataset)
        self.max_tokens = int(max_tokens)
        self.task_weights = dict(task_weights or {})
        self.time_bucket_weights = dict(time_bucket_weights or {})
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.oversize_policy = str(oversize_policy)
        if self.oversize_policy not in {"error", "partition"}:
            raise ValueError("oversize_policy must be 'error' or 'partition'")
        self._batches_per_epoch_explicit = batches_per_epoch is not None
        self.epoch = 0
        self._global_cache: tuple[int, list[list[int]]] | None = None
        self._validate_weights()
        all_positions = np.arange(len(self.specs), dtype=np.int64)
        self._oversize_positions = np.asarray(
            [
                int(position)
                for position in all_positions.tolist()
                if self.specs[position].effective_tokens > self.max_tokens
            ],
            dtype=np.int64,
        )
        if self.oversize_policy == "error" and self._oversize_positions.size:
            details = [
                {
                    "index": int(position),
                    "sample_id": self.specs[int(position)].sample_id,
                    "effective_tokens": self.specs[int(position)].effective_tokens,
                }
                for position in self._oversize_positions[:32].tolist()
            ]
            raise ValueError(
                f"{self._oversize_positions.size} clips exceed max_tokens={self.max_tokens}; "
                f"oversize_policy='error' requires an explicit larger budget or partition mode: {details}"
            )
        self._valid_positions = all_positions
        if self._valid_positions.size == 0:
            raise ValueError("dataset contains no clips")
        self._groups = self._build_groups()
        self._group_weights = self._build_group_weights()
        self._batches_per_epoch = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else self._default_batches_per_epoch()
        )

    def _validate_weights(self) -> None:
        for value in list(self.task_weights.values()) + list(self.time_bucket_weights.values()):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("sampling weights must be finite and non-negative")

    def _build_groups(self) -> "OrderedDict[tuple[int, int, str], np.ndarray]":
        groups: "OrderedDict[tuple[int, int, str], list[int]]" = OrderedDict()
        for position in self._valid_positions.tolist():
            key = self.specs[position].group_key
            groups.setdefault(key, []).append(int(position))
        return OrderedDict(
            (key, np.asarray(positions, dtype=np.int64))
            for key, positions in groups.items()
        )

    def _build_group_weights(self) -> np.ndarray:
        weights = []
        for task, _frames, bucket in self._groups:
            task_weight = _weight_value(
                self.task_weights, (task, TASK_NAMES[task], str(task))
            )
            bucket_weight = _weight_value(self.time_bucket_weights, (bucket,))
            weights.append(task_weight * bucket_weight)
        values = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(values).all() or float(values.sum()) <= 0:
            raise ValueError("at least one clip group must have positive sampling weight")
        return values / values.sum()

    def _default_batches_per_epoch(self) -> int:
        # Estimate a full pass without using group weights.  Each group has a
        # separate homogeneity constraint, so group costs are summed rather
        # than pretending all records can share one batch.
        count = 0
        for positions in self._groups.values():
            total = int(self.specs.atoms[positions] @ self.specs.frames[positions])
            count += max(1, math.ceil(total / self.max_tokens))
        return max(1, count)

    def _next_group_item(
        self,
        key: tuple[int, int, str],
        orders: dict[tuple[int, int, str], np.ndarray],
        cursors: dict[tuple[int, int, str], int],
        rng: np.random.Generator,
    ) -> int | None:
        order = orders[key]
        cursor = cursors[key]
        if cursor >= order.size:
            if not self.replacement:
                return None
            order = order.copy()
            if self.shuffle:
                rng.shuffle(order)
            cursors[key] = 0
            orders[key] = order
            cursor = 0
        position = int(order[cursor])
        cursors[key] = cursor + 1
        return position

    def _make_global_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if not self.replacement:
            # Evaluation must cover every eligible record exactly once.  A
            # weighted random choice of groups with a fixed batch budget can
            # exhaust the budget before the last group is reached, silently
            # dropping validation clips.  Pack each homogeneous group in
            # order; optional shuffling only changes order within/among groups.
            keys = list(self._groups)
            if self.shuffle:
                rng.shuffle(keys)
            result: list[list[int]] = []
            for key in keys:
                positions = self._groups[key].copy()
                if self.shuffle:
                    rng.shuffle(positions)
                batch: list[int] = []
                cost = 0
                for position in positions.tolist():
                    item_cost = int(self.specs[position].effective_tokens)
                    if item_cost > self.max_tokens:
                        if batch:
                            result.append(batch)
                            batch = []
                            cost = 0
                        result.append([int(self.specs[position].index)])
                        continue
                    if batch and cost + item_cost > self.max_tokens:
                        result.append(batch)
                        batch = []
                        cost = 0
                    batch.append(int(self.specs[position].index))
                    cost += item_cost
                if batch:
                    result.append(batch)
            if self._batches_per_epoch_explicit and self._batches_per_epoch < len(result):
                # A no-replacement epoch must not silently discard records.
                # A larger requested batch budget is harmless; a smaller one
                # is an explicit configuration error.
                raise ValueError(
                    "batches_per_epoch is smaller than the exact no-replacement "
                    f"schedule ({self._batches_per_epoch} < {len(result)})"
                )
            return result

        orders: dict[tuple[int, int, str], np.ndarray] = {}
        cursors: dict[tuple[int, int, str], int] = {}
        for key, positions in self._groups.items():
            order = positions.copy()
            if self.shuffle:
                rng.shuffle(order)
            orders[key] = order
            cursors[key] = 0

        result: list[list[int]] = []
        group_keys = list(self._groups)
        group_weights = {
            key: float(self._group_weights[index])
            for index, key in enumerate(group_keys)
            if float(self._group_weights[index]) > 0
        }
        available_keys = list(group_weights)
        for _batch_id in range(self._batches_per_epoch):
            if not available_keys:
                break
            available_weights = np.asarray(
                [group_weights[key] for key in available_keys],
                dtype=np.float64,
            )
            available_weights /= available_weights.sum()
            group_index = int(rng.choice(len(available_keys), p=available_weights))
            key = available_keys[group_index]
            batch: list[int] = []
            cost = 0
            seen_in_batch: set[int] = set()
            while True:
                position = self._next_group_item(key, orders, cursors, rng)
                if position is None:
                    break
                if position in seen_in_batch:
                    break
                item_cost = int(self.specs[position].effective_tokens)
                if item_cost > self.max_tokens:
                    if batch:
                        # The cursor has advanced. Put it back so this
                        # oversize sample is emitted alone next.
                        cursors[key] -= 1
                        break
                    batch.append(int(self.specs[position].index))
                    seen_in_batch.add(position)
                    cost = item_cost
                    break
                if cost + item_cost > self.max_tokens:
                    # The cursor has advanced.  Put this item back so the
                    # next batch can use it, preserving no-replacement order.
                    cursors[key] -= 1
                    break
                batch.append(int(self.specs[position].index))
                seen_in_batch.add(position)
                cost += item_cost
                if cursors[key] >= orders[key].size:
                    break
            if not batch:
                # A group can only reach this branch when it was exhausted
                # without replacement.  Remove it and redraw this batch.
                if not self.replacement:
                    available_keys.pop(group_index)
                    continue
                raise RuntimeError("clip sampler failed to form a non-empty batch")
            result.append(batch)
            if not self.replacement and cursors[key] >= orders[key].size:
                available_keys.pop(group_index)
        return result

    def _local_batches(self) -> list[list[int]]:
        if self._global_cache is None or self._global_cache[0] != self.epoch:
            self._global_cache = (self.epoch, self._make_global_batches())
        global_batches = self._global_cache[1]
        local = [
            batch for global_index, batch in enumerate(global_batches)
            if global_index % self.num_replicas == self.rank
        ]
        if self.drop_last:
            target = len(global_batches) // self.num_replicas
            local = local[:target]
        return local

    @property
    def global_batches(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(batch) for batch in self._global_cache[1]) if self._global_cache else tuple(
            tuple(batch) for batch in self._make_global_batches()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._global_cache = None

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._local_batches())

    def __len__(self) -> int:
        return len(self._local_batches())

    def eligibility_report(self) -> dict[str, Any]:
        oversize = [
            {
                "index": int(position),
                "sample_id": self.specs[int(position)].sample_id,
                "effective_tokens": int(self.specs[int(position)].effective_tokens),
            }
            for position in self._oversize_positions.tolist()
        ]
        return {
            "max_tokens": int(self.max_tokens),
            "oversize_policy": self.oversize_policy,
            "total_clips": int(len(self.specs)),
            "in_budget_clips": int(len(self.specs) - self._oversize_positions.size),
            "scheduled_clips": int(len(self.specs)),
            "oversize_clips": int(self._oversize_positions.size),
            "oversize": oversize,
            "batches_per_epoch": int(len(self)),
            "replacement": bool(self.replacement),
            "epoch": int(self.epoch),
        }

    def _spec_signature(self) -> dict[str, Any]:
        sample_ids = self.specs.sample_ids or ()
        return {
            "count": len(self.specs),
            "indices_sha256": hashlib.sha256(self.specs.indices.tobytes()).hexdigest(),
            "atoms_sha256": hashlib.sha256(self.specs.atoms.tobytes()).hexdigest(),
            "frames_sha256": hashlib.sha256(self.specs.frames.tobytes()).hexdigest(),
            "tasks_sha256": hashlib.sha256(self.specs.tasks.tobytes()).hexdigest(),
            "bucket_group_ids_sha256": hashlib.sha256(self.specs.bucket_group_ids.tobytes()).hexdigest(),
            "group_keys": [list(key) for key in self.specs.group_keys],
            "sample_ids_sha256": hashlib.sha256(
                json.dumps(list(sample_ids), separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "seed": int(self.seed),
            "replacement": bool(self.replacement),
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "oversize_policy": self.oversize_policy,
            "max_tokens": int(self.max_tokens),
            "num_replicas": int(self.num_replicas),
            "rank": int(self.rank),
            "batches_per_epoch": int(self._batches_per_epoch),
            "batches_per_epoch_explicit": bool(self._batches_per_epoch_explicit),
            "task_weights": {str(key): float(value) for key, value in self.task_weights.items()},
            "time_bucket_weights": {str(key): float(value) for key, value in self.time_bucket_weights.items()},
            "spec_signature": self._spec_signature(),
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("sampler state must be a mapping")
        exact = {
            "seed": (int, self.seed, "seed"),
            "replacement": (bool, self.replacement, "replacement"),
            "shuffle": (bool, self.shuffle, "shuffle"),
            "drop_last": (bool, self.drop_last, "drop_last"),
            "oversize_policy": (str, self.oversize_policy, "oversize_policy"),
            "max_tokens": (int, self.max_tokens, "max_tokens"),
            "num_replicas": (int, self.num_replicas, "num_replicas"),
            "rank": (int, self.rank, "rank"),
            "batches_per_epoch": (int, self._batches_per_epoch, "batches_per_epoch"),
            "batches_per_epoch_explicit": (bool, self._batches_per_epoch_explicit, "batches_per_epoch_explicit"),
        }
        for key, (_kind, expected, label) in exact.items():
            if key not in state:
                raise ValueError(f"sampler checkpoint is missing {label}")
            actual = state[key]
            if isinstance(expected, bool):
                matches = isinstance(actual, bool) and actual == expected
            elif isinstance(expected, int):
                matches = isinstance(actual, (int, np.integer)) and int(actual) == expected
            else:
                matches = str(actual) == expected
            if not matches:
                raise ValueError(f"sampler {label} differs from checkpoint")
        if dict(state.get("task_weights", {})) != {
            str(key): float(value) for key, value in self.task_weights.items()
        }:
            raise ValueError("sampler task_weights differ from checkpoint")
        if dict(state.get("time_bucket_weights", {})) != {
            str(key): float(value) for key, value in self.time_bucket_weights.items()
        }:
            raise ValueError("sampler time_bucket_weights differ from checkpoint")
        if state.get("spec_signature") != self._spec_signature():
            raise ValueError("sampler dataset/specification differs from checkpoint")
        epoch = state.get("epoch")
        if not isinstance(epoch, (int, np.integer)) or int(epoch) < 0:
            raise ValueError("sampler epoch must be a non-negative integer")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        self.set_epoch(int(state["epoch"]))

    def batch_complexity(self, batch: Sequence[int]) -> int:
        by_index = {int(self.specs[position].index): position for position in range(len(self.specs))}
        return sum(self.specs[by_index[int(index)]].effective_tokens for index in batch)


# Short aliases make the intent discoverable without preserving the legacy
# DynamicBatchWrapper API or changing its behavior.
ClipBatchSampler = TaskAwareClipBatchSampler
TaskAwareBatchSampler = TaskAwareClipBatchSampler


def make_clip_dataloader(
    dataset: Any,
    *,
    max_tokens: int | None = None,
    sampler: TaskAwareClipBatchSampler | None = None,
    collate_fn: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    trusted_store_fast_path: bool = False,
    strict_record_validation: bool = True,
    oversize_policy: str = "error",
    **sampler_kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader using the T03 batch sampler.

    batch_sampler owns the full list of indices for each minibatch, so no
    second distributed sampler or batch-size argument should be supplied.
    """
    if sampler is not None and (
        max_tokens is not None
        or sampler_kwargs
        or oversize_policy != "error"
    ):
        raise ValueError("pass either sampler or sampler construction arguments")
    if trusted_store_fast_path and strict_record_validation:
        raise ValueError(
            "trusted_store_fast_path requires strict_record_validation=False"
        )
    if sampler is None:
        sampler = TaskAwareClipBatchSampler(
            dataset,
            max_tokens=max_tokens,
            oversize_policy=oversize_policy,
            **sampler_kwargs,
        )
    if collate_fn is None:
        collate_fn = getattr(dataset, "collate_fn", None) or collate_clip_records
    if trusted_store_fast_path:
        if collate_fn is not collate_clip_records:
            raise ValueError(
                "trusted_store_fast_path is only valid with collate_clip_records"
            )
        collate_fn = partial(collate_clip_records, validate=False)
    worker_count = int(num_workers)
    if worker_count < 0:
        raise ValueError("num_workers must be non-negative")
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collate_fn,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
    }
    if worker_count:
        if int(prefetch_factor) < 1:
            raise ValueError("prefetch_factor must be positive")
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    elif persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")
    return torch.utils.data.DataLoader(**loader_kwargs)


build_clip_dataloader = make_clip_dataloader
create_clip_dataloader = make_clip_dataloader


def _as_scalar_or_list(values: Sequence[float], *, atol: float = 1e-6) -> float | list[float]:
    if not values:
        return []
    first = float(values[0])
    if all(math.isclose(first, float(value), rel_tol=1e-6, abs_tol=atol) for value in values[1:]):
        return first
    return [float(value) for value in values]


def summarize_clip_batch(batch: ClipBatch) -> dict[str, Any]:
    """Return JSON-safe batch diagnostics required by T03."""

    if not isinstance(batch, ClipBatch):
        raise TypeError("summarize_clip_batch expects ClipBatch")
    atom_counts = (
        batch.atom_ptr[1:] - batch.atom_ptr[:-1]
    ).detach().cpu().tolist()
    frames = int(batch.frames)
    per_sample_tokens = [int(frames * atoms) for atoms in atom_counts]
    task_values = batch.task.detach().cpu().tolist()
    task_values = [int(value) for value in task_values]
    task_types = [TASK_NAMES[value] for value in task_values]
    bucket_values = [str(value) for value in batch.time_bucket_id]
    spans = (
        batch.time_ps[:, -1] - batch.time_ps[:, 0]
    ).detach().cpu().tolist()
    if frames > 1:
        delta_values = batch.delta_time_ps.detach().cpu().reshape(-1).tolist()
        native_delta: float | list[float] | None = _as_scalar_or_list(delta_values)
    else:
        native_delta = None
    return {
        "batch_size": int(batch.batch_size),
        "atoms": int(batch.atom_count),
        "atoms_per_sample": [int(value) for value in atom_counts],
        "frames": frames,
        "effective_tokens": int(sum(per_sample_tokens)),
        "effective_tokens_per_sample": per_sample_tokens,
        "task_type": task_types[0] if len(set(task_types)) == 1 else task_types,
        "task_id": task_values[0] if len(set(task_values)) == 1 else task_values,
        "time_bucket_id": bucket_values[0] if len(set(bucket_values)) == 1 else bucket_values,
        "native_delta_time_ps": native_delta,
        "physical_clip_span_ps": _as_scalar_or_list(spans),
        "sample_id": list(batch.sample_id),
    }


def clip_batch_log_record(batch: ClipBatch) -> dict[str, Any]:
    """Compatibility-readable name for :func:`summarize_clip_batch`."""

    return summarize_clip_batch(batch)


class ClipBatchLogger:
    """Callable batch logger suitable for a training loop or a test sink."""

    def __init__(
        self,
        sink: Callable[[Mapping[str, Any]], Any] | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.sink = sink
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, batch: ClipBatch) -> dict[str, Any]:
        record = summarize_clip_batch(batch)
        if self.sink is not None:
            self.sink(record)
        else:
            self.logger.info("clip_batch %s", record)
        return record


def log_clip_batch(
    batch: ClipBatch,
    *,
    sink: Callable[[Mapping[str, Any]], Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ClipBatchLogger(sink=sink, logger=logger)(batch)


__all__ = [
    "build_clip_dataloader",
    "ClipBatchLogger",
    "ClipBatchSampler",
    "create_clip_dataloader",
    "ClipItemSpec",
    "ClipSpecTable",
    "TaskAwareBatchSampler",
    "TaskAwareClipBatchSampler",
    "bucket_delta_time_ps",
    "clip_batch_log_record",
    "get_clip_specs",
    "log_clip_batch",
    "summarize_clip_batch",
    "task_id",
    "task_name",
]
