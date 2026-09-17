#!/usr/bin/env python3
"""Inventory tracked binary artifacts and enforce the repository size policy.

The check is read-only. It reports every tracked generated/checkpoint binary
with byte size, SHA-256, and a path-derived purpose. CI should run
python scripts/check_tracked_artifacts.py --check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_MAX_BYTES = 64 * 1024 * 1024
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pkl", ".pickle"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def purpose_for(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "baseline_contract" in parts:
        return "engineering acceptance baseline/repaired reference"
    if "overfit" in parts:
        return "selected-system overfit checkpoint or artifact"
    if "outputs" in parts or "checkpoint" in name or "codec_step" in name:
        return "generated training/evaluation artifact"
    return "tracked binary/model asset; operator classification required"


def is_generated_checkpoint(path: Path) -> bool:
    """Identify generated checkpoint paths without banning small source assets."""

    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "outputs" in parts or "checkpoint" in name or "codec_step" in name


def inventory(root: Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    )
    paths = [Path(value) for value in raw.decode().split("\0") if value]
    artifacts = []
    for relative in paths:
        if relative.suffix.lower() not in BINARY_SUFFIXES:
            continue
        absolute = root / relative
        if not absolute.is_file():
            continue
        artifacts.append(
            {
                "path": relative.as_posix(),
                "bytes": absolute.stat().st_size,
                "sha256": sha256_file(absolute),
                "purpose": purpose_for(relative),
            }
        )
    artifacts.sort(key=lambda item: item["path"])
    return {
        "schema_version": "pvb.repository.tracked_artifact_inventory.v1",
        "max_bytes_default": DEFAULT_MAX_BYTES,
        "tracked_binary_count": len(artifacts),
        "tracked_binary_bytes": sum(int(item["bytes"]) for item in artifacts),
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = inventory(root)
    violations = [
        item for item in result["artifacts"]
        if int(item["bytes"]) > int(args.max_bytes)
        or (
            item["path"].lower().endswith((".pt", ".pth", ".ckpt"))
            and is_generated_checkpoint(Path(item["path"]))
        )
    ]
    result["policy"] = {
        "max_bytes": int(args.max_bytes),
        "generated_checkpoint_suffixes": [".pt", ".pth", ".ckpt"],
        "generated_checkpoint_path_policy": "outputs/** or checkpoint/codec_step filenames",
        "violations": violations,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.check and violations:
        print(
            f"tracked artifact policy failed: {len(violations)} violation(s)",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
