"""Prepare CPU topology before moving codec or DiT batches to the device."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..data.batch import ClipBatch


def _to_device(value: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    if isinstance(value, ClipBatch):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=non_blocking)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _to_device(
                    getattr(value, field.name),
                    device,
                    non_blocking=non_blocking,
                )
                for field in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {
            key: _to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _to_device(item, device, non_blocking=non_blocking) for item in value
        )
    if isinstance(value, list):
        return [
            _to_device(item, device, non_blocking=non_blocking) for item in value
        ]
    return value


def prepare_batch_then_to_device(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    """Run the CPU-only registration phase before any CUDA transfer."""

    prepare = getattr(model, "prepare_batch", None)
    if callable(prepare):
        prepare(batch)
    return _to_device(batch, device, non_blocking=non_blocking)
