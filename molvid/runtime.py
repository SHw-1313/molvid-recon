"""Device, RNG and atomic-result primitives shared by training and evaluation."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch


def configure_device(device: str | torch.device = "cuda", *, deterministic: bool = False) -> torch.device:
    selected = torch.device("cuda" if str(device) == "auto" else device)
    if deterministic and selected.type == "cuda":
        # This must be present before the first cuBLAS handle is created;
        # otherwise deterministic matrix multiplication raises at runtime.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback is permitted")
        if selected.index is not None:
            torch.cuda.set_device(selected)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return selected


def seed_all(seed: int) -> None:
    """Preserve the existing Python, NumPy, Torch, CUDA seed call order."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"unsupported precision: {precision}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BF16 training requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def append_metrics(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
