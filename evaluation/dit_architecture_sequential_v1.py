"""Independent loading and paired-seed helpers for architecture sequential v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    StateDetailLatentAdapter,
    contract_hash,
)
from trainer.dit_trainer import DIT_CHECKPOINT_SCHEMA, module_state_hash


PHASE = "dit_architecture_sequential_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_seed(
    master_seed: int,
    sample_id: str,
    history_frames: int,
    draw_id: int,
) -> int:
    payload = "\0".join(
        (
            "dit-architecture-sequential-v1-epsilon",
            str(int(master_seed)),
            str(sample_id),
            str(int(history_frames)),
            str(int(draw_id)),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def trainable_storage_pointers(module: nn.Module) -> set[int]:
    return {
        parameter.data_ptr()
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def assert_independent_trainers(first: Any, second: Any) -> None:
    if first is second or first.model is second.model or first.adapter is second.adapter:
        raise RuntimeError("paired trainers must own distinct trainer, model, and adapter objects")
    if first.optimizer is second.optimizer or first.scaler is second.scaler:
        raise RuntimeError("paired trainers must own distinct optimizer and scaler objects")
    overlap = trainable_storage_pointers(first.model) & trainable_storage_pointers(second.model)
    if overlap:
        raise RuntimeError("paired trainers share trainable parameter storage")
    for trainer in (first, second):
        if trainer.model.adapter is not trainer.adapter:
            raise RuntimeError("trainer model does not own the trainer adapter")


@dataclass(frozen=True)
class LoadedSequentialModel:
    model: MolecularDiT
    adapter: StateDetailLatentAdapter
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_hash: str
    adapter_state_hash: str
    model_contract_hash: str
    step: int
    successful_updates: int


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"checkpoint {label} mismatch: {value!r} != {expected!r}")


def load_sequential_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    expected_data_hash: str,
    expected_codec_hash: str,
    expected_statistics_hash: str,
) -> LoadedSequentialModel:
    """Build, load, and return one fully independent evaluation model."""

    checkpoint_path = Path(path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("sequential checkpoint must be a mapping")
    _expect(payload.get("schema"), DIT_CHECKPOINT_SCHEMA, "schema")
    _expect(payload.get("data_hash"), expected_data_hash, "data hash")
    _expect(payload.get("codec_hash"), expected_codec_hash, "codec hash")
    _expect(payload.get("statistics_hash"), expected_statistics_hash, "statistics hash")
    config = payload.get("config")
    model_contract = payload.get("model_contract")
    adapter_contract = payload.get("adapter_contract")
    model_state = payload.get("model_state")
    sequential = payload.get("sequential")
    if not all(
        isinstance(value, Mapping)
        for value in (config, model_contract, adapter_contract, model_state, sequential)
    ):
        raise ValueError("sequential checkpoint lacks config/model/adapter/provenance state")
    _expect(sequential.get("schema"), "pvb.dit.architecture_sequential.v1.checkpoint.v1", "sequential schema")
    if not isinstance(sequential.get("training_generator_state"), torch.Tensor):
        raise ValueError("sequential checkpoint lacks its independent training generator state")
    if not isinstance(sequential.get("cursor"), Mapping) or not sequential.get("schedule_hash"):
        raise ValueError("sequential checkpoint lacks its sampler cursor or schedule hash")
    _expect(config.get("metadata", {}).get("phase"), PHASE, "phase")
    if "ffn_norm_source" not in model_contract:
        raise ValueError("sequential checkpoint lacks the explicit Round 1 norm-source contract")
    norm_source = str(model_contract["ffn_norm_source"])
    _expect(config.get("metadata", {}).get("ffn_norm_source"), norm_source, "config norm source")
    _expect(sequential.get("unique_variable", {}).get("ffn_norm_source"), norm_source, "provenance norm source")
    ratio = int(config["ratio"])
    adapter = StateDetailLatentAdapter(
        codec_width=int(config["codec_width"]),
        scalar_width=int(config["scalar_width"]),
        vector_width=int(config["vector_width"]),
        ratio=ratio,
    ).to(device)
    if dict(adapter.contract()) != dict(adapter_contract):
        raise ValueError("checkpoint adapter contract does not match a fresh adapter")
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=int(config["scalar_width"]),
        vector_width=int(config["vector_width"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        ffn_multiplier=int(config["ffn_multiplier"]),
        dropout=float(config["dropout"]),
        execution_backend=str(config.get("metadata", {}).get("execution_backend", "factorized_v2")),
        ffn_norm_source=str(model_contract["ffn_norm_source"]),
    ).to(device)
    _expect(
        contract_hash(model.contract()),
        payload.get("model_contract_hash"),
        "model contract hash",
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    step = int(payload.get("step", -1))
    successful_updates = int(payload.get("successful_optimizer_updates", -1))
    if step < 0 or successful_updates != step:
        raise ValueError("checkpoint step and successful update count disagree")
    return LoadedSequentialModel(
        model=model,
        adapter=adapter,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        model_state_hash=module_state_hash(model),
        adapter_state_hash=module_state_hash(adapter),
        model_contract_hash=contract_hash(model.contract()),
        step=step,
        successful_updates=successful_updates,
    )


__all__ = [
    "LoadedSequentialModel",
    "PHASE",
    "assert_independent_trainers",
    "generation_seed",
    "load_sequential_checkpoint",
    "sha256_file",
    "trainable_storage_pointers",
]
