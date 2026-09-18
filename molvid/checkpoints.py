"""Strict codec artifact conversion and training-state serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .codec.model import TrajectoryCodec
from .codec.types import RATIO_FOR_MODE
from .runtime import sha256_file


_CODEC_SCHEMA = "pvb.codec.checkpoint.v2"
_CONTRACT_SCHEMA = "pvb.codec.model_contract.v4"
_NEW_SCHEMA = "molvid.training.checkpoint.v1"
_KEY_PREFIXES = (
    ("state_detail_codec.coordinate_head.", "coordinate_head."),
    ("coordinate_vector_stem.projection.", "coordinate_stem.projection."),
    ("frame_encoder.spatial_encoder.", "frame_encoder.backbone."),
    ("state_detail_codec.", "temporal_codec."),
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    source_sha256: str
    source_schema: str
    target_schema: str
    weight_count: int
    optimizer_parameter_count: int
    optimizer_moment_count: int


@dataclass
class CodecArtifact:
    model: TrajectoryCodec
    optimizer: torch.optim.Optimizer
    report: CheckpointLoadReport
    step: int
    epoch: int
    batch_in_epoch: int
    sampler_state: Mapping[str, Any] | None
    normalization_stats: Mapping[str, Any]
    config: Mapping[str, Any]
    model_contract: Mapping[str, Any]
    optimizer_contract: Mapping[str, Any]


def _mapped_key(name: str) -> str:
    for source, target in _KEY_PREFIXES:
        if name.startswith(source):
            return target + name[len(source):]
    return name


def _model_from_contract(contract: Any) -> TrajectoryCodec:
    if not isinstance(contract, Mapping):
        raise ValueError("codec checkpoint is missing its model contract")
    if contract.get("schema_version") != _CONTRACT_SCHEMA:
        raise ValueError("unsupported codec model contract schema")
    if contract.get("model_type") != "trainer.codec_trainer.PVBCodecModel":
        raise ValueError("codec model contract type is invalid")
    config = contract.get("constructor")
    if not isinstance(config, Mapping):
        raise ValueError("codec model contract has no constructor")
    if config.get("spatial_backbone") != "torchmd_et":
        raise ValueError("codec artifact requires the current TorchMD backbone")
    if config.get("neighbor_backend") != "cuda_radius":
        raise ValueError("codec artifact requires the CUDA radius graph")
    if int(config.get("lmax", 0)) != 1 or int(config.get("temporal_layers", 0)) != 1:
        raise ValueError("codec artifact has an unsupported spatial/temporal stack")
    if config.get("spatial_execution") != {"mode": "full"}:
        raise ValueError("codec artifact requires full spatial execution")
    if bool(config.get("use_spatial_refiner")):
        raise ValueError("codec artifact uses the retired spatial refiner")
    mode = str(config.get("temporal_codec_mode", ""))
    if mode not in RATIO_FOR_MODE or int(config.get("temporal_ratio", 0)) != RATIO_FOR_MODE[mode]:
        raise ValueError("codec mode and temporal ratio disagree")
    coordinate_stem = str(config.get("coordinate_stem", "none"))
    if coordinate_stem not in {"none", "centered_vector"}:
        raise ValueError("unsupported codec coordinate stem")
    dtype_name = str(config.get("spatial_dtype", ""))
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("unsupported codec spatial dtype")
    bond_config = config.get("bond_construction")
    if not isinstance(bond_config, Mapping):
        raise ValueError("codec bond construction contract is missing")
    return TrajectoryCodec(
        hidden_channels=int(config["hidden_channels"]),
        spatial_layers=int(config["spatial_layers"]),
        temporal_codec_mode=mode,
        num_rbf=int(config["num_rbf"]),
        num_heads=int(config["num_heads"]),
        cutoff_lower=float(config["cutoff_lower"]),
        cutoff_upper=float(config["cutoff_upper"]),
        max_num_neighbors=int(config["max_num_neighbors"]),
        bond_construction=bond_config,
        topology_cache_capacity=int(config["topology_cache_capacity"]),
        topology_device_cache_capacity=int(config["topology_device_cache_capacity"]),
        distance_bond_min=float(bond_config.get("min_distance_angstrom", config["distance_bond_min"])),
        distance_bond_max=float(bond_config.get("max_distance_angstrom", config["distance_bond_max"])),
        distance_bond_max_num_neighbors=int(bond_config.get("max_num_neighbors", config["distance_bond_max_num_neighbors"])),
        distance_bond_cache_capacity=int(config["distance_bond_cache_capacity"]),
        spatial_dtype=getattr(torch, dtype_name),
        freeze_frame_encoder=bool(config.get("freeze_frame_encoder")),
        frame_encoder_source_hash=config.get("frame_encoder_source_hash"),
        coordinate_stem=coordinate_stem,
    )


def _load_strict_model(model: TrajectoryCodec, state: Any) -> tuple[str, ...]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("codec checkpoint model_state must be a nonempty mapping")
    target = model.state_dict()
    renamed: dict[str, Tensor] = {}
    source_names: list[str] = []
    for source_name, value in state.items():
        if not isinstance(source_name, str) or not isinstance(value, Tensor):
            raise ValueError("codec model_state must contain only named tensors")
        target_name = _mapped_key(source_name)
        if target_name in renamed:
            raise ValueError(f"codec checkpoint key collision at {target_name!r}")
        renamed[target_name] = value
        source_names.append(source_name)
    if list(renamed) != list(target):
        missing = sorted(set(target) - set(renamed))
        unexpected = sorted(set(renamed) - set(target))
        raise ValueError(f"codec state keys/order mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    for name, value in renamed.items():
        expected = target[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"codec state shape/dtype mismatch at {name!r}")
    model.load_state_dict(renamed, strict=True)
    for name, value in renamed.items():
        if not torch.equal(model.state_dict()[name].cpu(), value.cpu()):
            raise RuntimeError(f"codec weight changed while loading {name!r}")
    return tuple(source_names)


def _load_strict_optimizer(
    model: TrajectoryCodec,
    state: Any,
    contract: Any,
    config: Any,
    source_state_names: tuple[str, ...],
    *,
    step: int,
) -> torch.optim.Optimizer:
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("codec optimizer state and contract are required")
    if not isinstance(config, Mapping) or not isinstance(config.get("training"), Mapping):
        raise ValueError("codec training config is required for optimizer validation")
    training = config["training"]
    if contract.get("algorithm") != "AdamW":
        raise ValueError("codec optimizer is not AdamW")
    if float(contract.get("base_lr", -1)) != float(training["lr"]):
        raise ValueError("codec optimizer learning rate disagrees with config")
    if float(contract.get("weight_decay", -1)) != float(training["weight_decay"]):
        raise ValueError("codec optimizer weight decay disagrees with config")
    if int(contract.get("warmup_steps", -1)) != int(training["warmup_steps"]):
        raise ValueError("codec optimizer warmup disagrees with config")
    groups = state.get("param_groups")
    moment_state = state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(moment_state, Mapping):
        raise ValueError("codec optimizer requires one complete parameter group")
    old_names = [name for name in source_state_names if _mapped_key(name) in dict(model.named_parameters())]
    new_names = list(dict(model.named_parameters()))
    if [_mapped_key(name) for name in old_names] != new_names:
        raise ValueError("codec optimizer parameter name/order migration is not one-to-one")
    ids = groups[0].get("params")
    if not isinstance(ids, list) or len(ids) != len(old_names) or len(set(ids)) != len(ids):
        raise ValueError("codec optimizer parameter IDs are incomplete or duplicated")
    if contract.get("param_group_count") != 1 or contract.get("param_group_sizes") != [len(ids)]:
        raise ValueError("codec optimizer group sizes disagree with contract")
    if contract.get("current_lrs") != [float(groups[0].get("lr", float("nan")))]:
        raise ValueError("codec optimizer group LR disagrees with contract")
    if float(groups[0].get("weight_decay", float("nan"))) != float(training["weight_decay"]):
        raise ValueError("codec optimizer group decay disagrees with config")
    if set(moment_state) - set(ids):
        raise ValueError("codec optimizer moments refer to unknown parameter IDs")
    params = list(model.parameters())
    for index, parameter_id in enumerate(ids):
        slot = moment_state.get(parameter_id)
        if slot is None:
            if step > 0 and params[index].requires_grad:
                raise ValueError(f"trainable codec parameter {new_names[index]!r} has no optimizer state")
            continue
        if not isinstance(slot, Mapping) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("codec AdamW moment fields are incomplete")
        if not params[index].requires_grad:
            raise ValueError("frozen codec parameter unexpectedly has optimizer moments")
        for name in ("exp_avg", "exp_avg_sq"):
            value = slot[name]
            if not isinstance(value, Tensor) or value.shape != params[index].shape or value.dtype != params[index].dtype:
                raise ValueError(f"codec optimizer {name} shape/dtype differs at {new_names[index]!r}")
        if not isinstance(slot["step"], Tensor) or slot["step"].numel() != 1:
            raise ValueError("codec optimizer step is invalid")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer.load_state_dict(state)
    loaded = optimizer.state_dict()
    if loaded["param_groups"] != groups:
        raise RuntimeError("codec optimizer parameter group changed while loading")
    for parameter_id, original in moment_state.items():
        restored = loaded["state"][parameter_id]
        for name, value in original.items():
            if isinstance(value, Tensor):
                if not torch.equal(value.cpu(), restored[name].cpu()):
                    raise RuntimeError(f"codec optimizer moment changed at parameter {parameter_id}")
            elif value != restored[name]:
                raise RuntimeError(f"codec optimizer state changed at parameter {parameter_id}")
    return optimizer


def load_codec_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> CodecArtifact:
    """Load an approved old codec only after hash, schema and full state validation."""

    source = Path(path)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError("codec checkpoint SHA-256 differs from the approved source")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _CODEC_SCHEMA:
        raise ValueError("unsupported codec checkpoint schema")
    required = {"step", "epoch", "batch_in_epoch", "precision", "sampler_state", "model_contract", "distance_reference_contract", "optimizer_contract", "model_state", "optimizer_state", "normalization_stats", "config"}
    if required - set(payload):
        raise ValueError(f"codec checkpoint missing fields: {sorted(required - set(payload))}")
    if not isinstance(payload["normalization_stats"], Mapping):
        raise ValueError("codec normalization statistics must be a mapping")
    model = _model_from_contract(payload["model_contract"])
    names = _load_strict_model(model, payload["model_state"])
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA codec artifact device is unavailable")
    model.to(selected_device)
    step = int(payload["step"])
    if step < 0 or int(payload["epoch"]) < 0 or int(payload["batch_in_epoch"]) < 0:
        raise ValueError("codec checkpoint counters must be nonnegative")
    optimizer = _load_strict_optimizer(
        model, payload["optimizer_state"], payload["optimizer_contract"], payload["config"], names, step=step
    )
    report = CheckpointLoadReport(
        source_sha256=actual_sha256,
        source_schema=_CODEC_SCHEMA,
        target_schema=_NEW_SCHEMA,
        weight_count=len(names),
        optimizer_parameter_count=len(optimizer.param_groups[0]["params"]),
        optimizer_moment_count=len(optimizer.state),
    )
    return CodecArtifact(
        model=model,
        optimizer=optimizer,
        report=report,
        step=step,
        epoch=int(payload["epoch"]),
        batch_in_epoch=int(payload["batch_in_epoch"]),
        sampler_state=payload["sampler_state"],
        normalization_stats=payload["normalization_stats"],
        config=payload["config"],
        model_contract=payload["model_contract"],
        optimizer_contract=payload["optimizer_contract"],
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture all process RNG streams used by codec and DiT training."""

    import random
    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams, including CUDA streams, from a checked checkpoint."""

    import random
    import numpy as np

    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("checkpoint RNG state is incomplete")
    if not isinstance(state["torch"], Tensor):
        raise ValueError("checkpoint torch RNG state must be a tensor")
    cuda = state["cuda"]
    if cuda is not None and (
        not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()
    ):
        raise RuntimeError("checkpoint CUDA RNG streams do not match available devices")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if cuda is not None:
        for device_index, value in enumerate(cuda):
            if not isinstance(value, Tensor):
                raise ValueError("checkpoint CUDA RNG state must contain tensors")
            torch.cuda.set_rng_state(
                value.detach().to(device="cpu", dtype=torch.uint8),
                device=device_index,
            )


def _optimizer_parameter_names(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    name_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    names: list[list[str]] = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            name = name_by_identity.get(id(parameter))
            if name is None:
                raise ValueError("optimizer owns a parameter outside the model")
            group_names.append(name)
        names.append(group_names)
    if len({name for group in names for name in group}) != sum(map(len, names)):
        raise ValueError("optimizer parameter groups contain duplicate model parameters")
    return names


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cursor: Mapping[str, Any],
    contracts: Mapping[str, Any],
    scheduler: Any = None,
    scaler: Any = None,
    sampler_state: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a new-format checkpoint with explicit cursor and RNG."""

    import os

    if int(step) < 0:
        raise ValueError("checkpoint step must be nonnegative")
    if not isinstance(cursor, Mapping) or not isinstance(contracts, Mapping):
        raise ValueError("checkpoint cursor and contracts must be mappings")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _NEW_SCHEMA,
        "step": int(step),
        "cursor": dict(cursor),
        "contracts": dict(contracts),
        "parameter_names": _optimizer_parameter_names(model, optimizer),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "sampler_state": sampler_state,
        "extra_state": dict(extra_state or {}),
        "rng_state": capture_rng_state(),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_contracts: Mapping[str, Any],
    scheduler: Any = None,
    scaler: Any = None,
    expected_sha256: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Validate the complete new checkpoint before restoring model and train state."""

    import copy

    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("training checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _NEW_SCHEMA:
        raise ValueError("unsupported training checkpoint schema")
    required = {
        "step", "cursor", "contracts", "parameter_names", "model_state",
        "optimizer_state", "scheduler_state", "scaler_state",
        "sampler_state", "extra_state", "rng_state",
    }
    if required - set(payload):
        raise ValueError(f"training checkpoint missing fields: {sorted(required - set(payload))}")
    if payload["contracts"] != dict(expected_contracts):
        raise ValueError("training checkpoint contracts differ")
    if not isinstance(payload["cursor"], Mapping) or int(payload["step"]) < 0:
        raise ValueError("training checkpoint cursor/step is invalid")
    if payload["parameter_names"] != _optimizer_parameter_names(model, optimizer):
        raise ValueError("training optimizer parameter names/order differ")
    saved_model = payload["model_state"]
    target_model = model.state_dict()
    if not isinstance(saved_model, Mapping) or list(saved_model) != list(target_model):
        raise ValueError("training model state keys/order differ")
    for name, value in saved_model.items():
        target = target_model[name]
        if not isinstance(value, Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"training model shape/dtype mismatch at {name!r}")
    saved_optimizer = payload["optimizer_state"]
    if not isinstance(saved_optimizer, Mapping) or not isinstance(saved_optimizer.get("param_groups"), list):
        raise ValueError("training optimizer state is incomplete")
    if [len(group["params"]) for group in saved_optimizer["param_groups"]] != [
        len(group["params"]) for group in optimizer.param_groups
    ]:
        raise ValueError("training optimizer group sizes differ")
    if (payload["scheduler_state"] is None) != (scheduler is None):
        raise ValueError("training scheduler presence differs")
    if (payload["scaler_state"] is None) != (scaler is None):
        raise ValueError("training scaler presence differs")
    old_model = copy.deepcopy(target_model)
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_scheduler = None if scheduler is None else copy.deepcopy(scheduler.state_dict())
    old_scaler = None if scaler is None else copy.deepcopy(scaler.state_dict())
    old_rng = capture_rng_state() if restore_rng else None
    try:
        model.load_state_dict(saved_model, strict=True)
        optimizer.load_state_dict(saved_optimizer)
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler_state"])
        if restore_rng:
            restore_rng_state(payload["rng_state"])
    except Exception:
        model.load_state_dict(old_model, strict=True)
        optimizer.load_state_dict(old_optimizer)
        if scheduler is not None:
            scheduler.load_state_dict(old_scheduler)
        if scaler is not None:
            scaler.load_state_dict(old_scaler)
        if old_rng is not None:
            restore_rng_state(old_rng)
        raise
    return dict(payload)
