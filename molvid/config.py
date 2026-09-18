"""YAML loading and explicit project-relative path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def validate_config(
    config: Mapping[str, Any],
    *,
    schema: str | None = None,
    required_sections: Sequence[str] = (),
) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("configuration root must be a mapping")
    actual_schema = config.get("schema", config.get("schema_version"))
    if not isinstance(actual_schema, str) or not actual_schema:
        raise ValueError("configuration requires a non-empty schema")
    if schema is not None and actual_schema != schema:
        raise ValueError(f"unsupported configuration schema: {actual_schema!r}")
    for name in required_sections:
        if not isinstance(config.get(name), Mapping):
            raise ValueError(f"configuration section {name!r} must be a mapping")


def load_config(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    validate_config(value, schema=schema)
    return dict(value)


def resolve_config(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
    path_fields: Sequence[str],
) -> dict[str, Any]:
    """Resolve only explicitly named dotted path fields, including path lists."""

    validate_config(config)
    resolved = deepcopy(dict(config))
    root = Path(project_root).resolve()
    for dotted in path_fields:
        section: dict[str, Any] = resolved
        parts = dotted.split(".")
        for part in parts[:-1]:
            value = section.get(part)
            if not isinstance(value, dict):
                raise ValueError(f"missing configuration path: {dotted}")
            section = value
        key = parts[-1]
        if key not in section:
            raise ValueError(f"missing configuration path: {dotted}")
        value = section[key]
        if value is None:
            continue

        def absolute(item: str | Path) -> str:
            path = Path(item)
            return str(path.resolve() if path.is_absolute() else (root / path).resolve())

        if isinstance(value, str):
            section[key] = absolute(value)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            section[key] = [absolute(item) for item in value]
        else:
            raise ValueError(f"configuration path {dotted!r} must be a string or string list")
    return resolved
