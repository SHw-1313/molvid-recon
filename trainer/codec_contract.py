"""Canonical JSON contracts shared by codec checkpoints and evaluators."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def json_safe(value: Any, *, path: str = "value") -> Any:
    """Return a deterministic JSON-compatible copy or fail loudly."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported contract value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def contract_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def contract_differences(expected: Any, actual: Any, *, path: str = "contract") -> list[str]:
    """Describe semantic differences without relying on mapping insertion order."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            child = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing value")
            else:
                differences.extend(contract_differences(expected[key], actual[key], path=child))
        return differences
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        differences = []
        if len(expected) != len(actual):
            differences.append(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(contract_differences(left, right, path=f"{path}[{index}]"))
        return differences
    if expected != actual:
        return [f"{path}: {actual!r} != {expected!r}"]
    return []


def require_contract_equal(expected: Any, actual: Any, *, label: str) -> None:
    differences = contract_differences(expected, actual, path=label)
    if differences:
        preview = "; ".join(differences[:8])
        suffix = "" if len(differences) <= 8 else f"; ... ({len(differences)} differences)"
        raise ValueError(f"{label} mismatch: {preview}{suffix}")


__all__ = [
    "canonical_json",
    "contract_differences",
    "contract_hash",
    "json_safe",
    "require_contract_equal",
]
