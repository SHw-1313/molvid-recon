"""Strict codec artifact conversion and training-state serialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


def _module_tensor_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frame_target_provenance(
    artifact: CodecArtifact,
    *,
    source_path: str | Path,
    code_commit: str,
    cuda_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact Haar-preceding modules used by Frame Joint v1."""

    constructor = artifact.model_contract.get("constructor")
    if not isinstance(constructor, Mapping):
        raise ValueError("codec artifact lacks its verified constructor contract")
    return {
        "schema_version": "molvid.frame_joint.target_encoder_provenance.v1",
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": artifact.report.source_sha256,
        "source_schema": artifact.report.source_schema,
        "artifact_step": int(artifact.step),
        "artifact_epoch": int(artifact.epoch),
        "historical_final_result_step": 45844,
        "extracted_modules": ["frame_encoder", "coordinate_stem"],
        "name_mappings": [
            {"source_prefix": source, "target_prefix": target}
            for source, target in _KEY_PREFIXES
            if source.startswith("frame_encoder") or source.startswith("coordinate_vector_stem")
        ],
        "frame_encoder_state_sha256": _module_tensor_hash(artifact.model.frame_encoder),
        "coordinate_stem_state_sha256": _module_tensor_hash(artifact.model.coordinate_stem),
        "constructor": dict(constructor),
        "coordinate_convention": "one frame-zero loss-mask centroid per sample; same origin for all frames",
        "feature_boundary": "frame_encoder_plus_coordinate_stem_before_haar",
        "frozen": True,
        "code_commit": str(code_commit),
        "cuda_check": dict(cuda_check or {}),
    }


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


def load_dit_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Strictly inspect and transfer one verified historical DiT state.

    This is an explicit artifact-conversion boundary, not a runtime fallback
    for the new training checkpoint format.
    """

    import copy
    from .dit.model import MolecularDiT
    from .latent.adapter import StateDetailLatentAdapter
    from .latent.statistics import LatentStatistics
    from .latent.types import contract_hash

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("DiT checkpoint SHA-256 differs from the approved source")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != "pvb.dit.state_detail.checkpoint.v2":
        raise ValueError("unsupported historical DiT checkpoint schema")
    required = {
        "step", "successful_optimizer_updates", "config", "model_contract",
        "model_contract_hash", "adapter_contract", "adapter_contract_hash",
        "model_state", "optimizer_state", "statistics_state", "statistics_hash",
        "codec_hash", "data_hash", "source_contract", "frozen_hashes",
        "scheduler_state", "scaler_state", "rng_state", "sequential",
    }
    if required - set(payload):
        raise ValueError(f"historical DiT checkpoint missing fields: {sorted(required - set(payload))}")
    config = payload["config"]
    if not isinstance(config, Mapping) or not isinstance(config.get("metadata"), Mapping):
        raise ValueError("historical DiT config is incomplete")
    if int(payload["step"]) < 1 or int(payload["step"]) != int(payload["successful_optimizer_updates"]):
        raise ValueError("historical DiT update count differs from step")
    if payload["scheduler_state"] is not None or not isinstance(payload["scaler_state"], Mapping):
        raise ValueError("historical DiT scheduler/scaler contract differs")
    if not isinstance(payload["rng_state"], Mapping) or set(payload["rng_state"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("historical DiT RNG state is incomplete")
    sequential = payload["sequential"]
    if not isinstance(sequential, Mapping) or not isinstance(sequential.get("cursor"), Mapping) or not isinstance(sequential.get("training_generator_state"), Tensor):
        raise ValueError("historical DiT sequential cursor/generator is incomplete")
    statistics = LatentStatistics.from_state_dict(payload["statistics_state"])
    if statistics.hash != payload["statistics_hash"] or statistics.hash != config["stats_hash"]:
        raise ValueError("historical DiT statistics hash differs")
    if config["data_hash"] != payload["data_hash"] or config["codec_hash"] != payload["codec_hash"]:
        raise ValueError("historical DiT data/codec provenance differs")
    source = payload["source_contract"]
    if not isinstance(source, Mapping) or source.get("source_mode") != config["source_mode"] or source.get("normalization_hash") != statistics.hash:
        raise ValueError("historical DiT source contract differs")
    metadata = config["metadata"]
    adapter = StateDetailLatentAdapter(
        codec_width=int(config["codec_width"]),
        scalar_width=int(config["scalar_width"]),
        vector_width=int(config["vector_width"]),
        ratio=int(config["ratio"]),
        mode=str(config["mode"]),
    )
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=int(config["scalar_width"]),
        vector_width=int(config["vector_width"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        ffn_multiplier=int(config["ffn_multiplier"]),
        dropout=float(config["dropout"]),
        execution_backend=str(metadata["execution_backend"]),
        ffn_norm_source=str(metadata["ffn_norm_source"]),
    )
    if model.contract() != payload["model_contract"] or contract_hash(model.contract()) != payload["model_contract_hash"]:
        raise ValueError("historical DiT model contract differs")
    if adapter.contract() != payload["adapter_contract"] or contract_hash(adapter.contract()) != payload["adapter_contract_hash"]:
        raise ValueError("historical DiT adapter contract differs")
    state = payload["model_state"]
    expected = model.state_dict()
    if not isinstance(state, Mapping) or list(state) != list(expected):
        raise ValueError("historical DiT model keys/order differ")
    for name, value in state.items():
        if not isinstance(value, Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"historical DiT model shape/dtype differs at {name!r}")
    saved_optimizer = payload["optimizer_state"]
    if not isinstance(saved_optimizer, Mapping):
        raise ValueError("historical DiT optimizer is incomplete")
    groups = saved_optimizer.get("param_groups")
    slots = saved_optimizer.get("state")
    parameters = list(model.parameters())
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(slots, Mapping):
        raise ValueError("historical DiT optimizer group/state is incomplete")
    ids = groups[0].get("params")
    if not isinstance(ids, list) or ids != list(range(len(parameters))) or set(slots) != set(ids):
        raise ValueError("historical DiT optimizer parameter IDs/moments are incomplete")
    if float(groups[0].get("lr", -1)) != float(config["learning_rate"]) or float(groups[0].get("weight_decay", -1)) != float(config["weight_decay"]):
        raise ValueError("historical DiT optimizer hyperparameters disagree")
    for index, parameter in enumerate(parameters):
        slot = slots[index]
        if not isinstance(slot, Mapping) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"historical DiT AdamW moments are incomplete at {index}")
        if not isinstance(slot["step"], Tensor) or slot["step"].numel() != 1 or int(slot["step"].item()) != int(payload["step"]):
            raise ValueError(f"historical DiT optimizer step differs at {index}")
        for name in ("exp_avg", "exp_avg_sq"):
            value = slot[name]
            if not isinstance(value, Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"historical DiT {name} shape/dtype differs at {index}")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA DiT artifact device is unavailable")
    model.to(selected_device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
    for name, value in state.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), value):
            raise RuntimeError(f"historical DiT weight changed while loading {name!r}")
    loaded = optimizer.state_dict()
    if loaded["param_groups"] != groups:
        raise RuntimeError("historical DiT optimizer group changed while loading")
    for index, slot in slots.items():
        for name, value in slot.items():
            if not torch.equal(loaded["state"][index][name].detach().cpu(), value):
                raise RuntimeError(f"historical DiT optimizer moment changed at {index}")
    return {
        "model": model,
        "optimizer": optimizer,
        "statistics": statistics.to(device=selected_device),
        "payload": payload,
        "source_sha256": actual_sha256,
        "weight_count": len(state),
        "optimizer_parameter_count": len(ids),
        "optimizer_moment_count": len(slots),
    }

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
    if not isinstance(saved_optimizer, Mapping):
        raise ValueError("training optimizer state is incomplete")
    saved_groups = saved_optimizer.get("param_groups")
    saved_slots = saved_optimizer.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_slots, Mapping):
        raise ValueError("training optimizer state is incomplete")
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("training optimizer group count differs")
    parameters_by_id: dict[int, nn.Parameter] = {}
    amsgrad_by_id: dict[int, bool] = {}
    for saved_group, current_group in zip(saved_groups, optimizer.param_groups):
        ids = saved_group.get("params") if isinstance(saved_group, Mapping) else None
        parameters = current_group["params"]
        if not isinstance(ids, list) or len(ids) != len(parameters):
            raise ValueError("training optimizer group sizes differ")
        for key in ("weight_decay", "betas", "eps", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
            if key in current_group and saved_group.get(key) != current_group[key]:
                raise ValueError(f"training optimizer hyperparameter {key!r} differs")
        for parameter_id, parameter in zip(ids, parameters):
            if not isinstance(parameter_id, int) or parameter_id in parameters_by_id:
                raise ValueError("training optimizer parameter IDs are invalid")
            parameters_by_id[parameter_id] = parameter
            amsgrad_by_id[parameter_id] = bool(saved_group.get("amsgrad", False))
    if set(saved_slots) - set(parameters_by_id):
        raise ValueError("training optimizer moments refer to unknown parameters")
    if isinstance(optimizer, torch.optim.AdamW):
        for parameter_id, slot in saved_slots.items():
            parameter = parameters_by_id[parameter_id]
            required = {"step", "exp_avg", "exp_avg_sq"}
            if amsgrad_by_id[parameter_id]:
                required.add("max_exp_avg_sq")
            if not parameter.requires_grad or not isinstance(slot, Mapping) or set(slot) != required:
                raise ValueError("training AdamW moments are incomplete or belong to frozen parameters")
            step_value = slot["step"]
            if not isinstance(step_value, Tensor) or step_value.numel() != 1:
                raise ValueError("training AdamW step is invalid")
            for name in required - {"step"}:
                value = slot[name]
                if not isinstance(value, Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(f"training AdamW moment {name!r} shape/dtype differs")
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


def load_frame_joint_inference(
    path: str | Path,
    *,
    expected_sha256: str,
    codec_path: str | Path,
    codec_sha256: str,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Strictly reconstruct a Frame Joint model for sampling/evaluation."""

    if sha256_file(path) != expected_sha256:
        raise ValueError("Frame Joint checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _NEW_SCHEMA:
        raise ValueError("unsupported Frame Joint checkpoint schema")
    contracts = payload.get("contracts")
    extra = payload.get("extra_state")
    if not isinstance(contracts, Mapping) or contracts.get("trainer") != "molvid.frame_joint.v1":
        raise ValueError("checkpoint is not a Frame Joint v1 artifact")
    if not isinstance(extra, Mapping) or not isinstance(extra.get("statistics_state"), Mapping):
        raise ValueError("Frame Joint checkpoint lacks latent statistics")
    from .latent.statistics import FrameLatentStatistics
    from .model import FrameJointModel

    statistics = FrameLatentStatistics.from_state_dict(extra["statistics_state"])
    if statistics.hash != contracts.get("statistics_hash"):
        raise ValueError("Frame Joint statistics differ from checkpoint contract")
    if str(contracts.get("teacher_artifact_sha256")) != str(codec_sha256):
        raise ValueError("Frame Joint target codec identity differs")
    artifact = load_codec_artifact(
        codec_path,
        expected_sha256=codec_sha256,
        device=device,
    )
    model_contract = contracts.get("model")
    dit_contract = model_contract.get("dit") if isinstance(model_contract, Mapping) else None
    if not isinstance(dit_contract, Mapping):
        raise ValueError("Frame Joint checkpoint lacks its model constructor contract")
    model_schema = str(model_contract.get("schema_version"))
    dit_schema = str(dit_contract.get("schema_version"))
    if model_schema == "molvid.frame_joint.model.v1" and dit_schema == "molvid.frame_joint.dit.v1":
        geometry_enabled = False
        motion_enabled = False
    elif model_schema == "molvid.frame_joint.model.v2" and dit_schema == "molvid.frame_joint.dit.v2":
        geometry_enabled = model_contract.get("geometry_enabled")
        motion_enabled = model_contract.get("motion_enabled")
        if not isinstance(geometry_enabled, bool) or not isinstance(motion_enabled, bool):
            raise ValueError("Frame Joint v2 checkpoint has invalid G/M switches")
        if geometry_enabled != dit_contract.get("geometry_enabled") or motion_enabled != dit_contract.get("motion_enabled"):
            raise ValueError("Frame Joint model and DiT G/M switches differ")
    else:
        raise ValueError("unsupported Frame Joint model/DiT contract schemas")
    model = FrameJointModel.from_codec(
        artifact.model,
        statistics.to(device=device),
        scalar_width=int(dit_contract["scalar_width"]),
        vector_width=int(dit_contract["vector_width"]),
        depth=int(dit_contract["depth"]),
        heads=int(dit_contract["heads"]),
        geometry_enabled=geometry_enabled,
        motion_enabled=motion_enabled,
    ).to(device)
    if dict(model.contract()) != dict(model_contract):
        raise ValueError("reconstructed Frame Joint model contract differs")
    state = payload.get("model_state")
    expected = model.state_dict()
    if not isinstance(state, Mapping) or list(state) != list(expected):
        raise ValueError("Frame Joint model state keys/order differ")
    for name, value in state.items():
        target = expected[name]
        if not isinstance(value, Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Frame Joint state shape/dtype differs at {name!r}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return {
        "model": model,
        "statistics": model.statistics,
        "codec_artifact": artifact,
        "payload": payload,
        "source_sha256": expected_sha256,
    }


def load_dit_inference(
    path: str | Path,
    *,
    expected_sha256: str,
    codec_path: str | Path,
    codec_sha256: str,
    device: str | torch.device,
) -> dict[str, Any]:
    """Validate and open a new-format DiT checkpoint for observed-only sampling.

    The full optimizer, frozen-module and statistics contracts are checked
    through the same trainer loader used for continuation. Model construction
    leaves the caller's Python/NumPy/Torch/CUDA RNG streams unchanged.
    """

    from .dit.model import MolecularDiT
    from .latent.adapter import StateDetailLatentAdapter
    from .latent.statistics import LatentStatistics
    from .training.dit import DiTTrainConfig, DiTTrainer, module_state_hash

    if sha256_file(path) != expected_sha256:
        raise ValueError("DiT inference checkpoint SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _NEW_SCHEMA:
        raise ValueError("DiT inference requires a new-format training checkpoint")
    contracts = payload.get("contracts")
    extra = payload.get("extra_state")
    if not isinstance(contracts, Mapping) or contracts.get("trainer") != "molvid.dit":
        raise ValueError("checkpoint is not a Molvid DiT training artifact")
    if not isinstance(extra, Mapping) or not isinstance(extra.get("statistics_state"), Mapping):
        raise ValueError("DiT inference checkpoint lacks statistics")
    configuration = contracts.get("config")
    if not isinstance(configuration, Mapping) or not isinstance(configuration.get("metadata"), Mapping):
        raise ValueError("DiT inference checkpoint lacks a model configuration")
    metadata = configuration["metadata"]
    if metadata.get("codec_checkpoint_sha256") != codec_sha256:
        raise ValueError("codec checkpoint identity differs from the DiT checkpoint")
    selected_device = torch.device(device)
    prior_rng = capture_rng_state()
    try:
        codec_artifact = load_codec_artifact(
            codec_path, expected_sha256=codec_sha256, device=selected_device
        )
        codec = codec_artifact.model.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        statistics = LatentStatistics.from_state_dict(extra["statistics_state"]).to(device=selected_device)
        if statistics.hash != configuration.get("stats_hash"):
            raise ValueError("DiT inference statistics hash differs")
        if statistics.provenance.get("codec_checkpoint_sha256") != codec_sha256:
            raise ValueError("DiT inference statistics belong to another codec")
        if statistics.provenance.get("data_hash") != configuration.get("data_hash"):
            raise ValueError("DiT inference statistics belong to another dataset")
        if statistics.width != codec.temporal_codec.channels or statistics.ratio != codec.temporal_ratio:
            raise ValueError("codec and DiT inference statistics disagree")
        config_values = dict(configuration)
        config_values["observation_mixture"] = tuple(config_values["observation_mixture"])
        config_values["history_probabilities"] = tuple(config_values["history_probabilities"])
        config = DiTTrainConfig(**config_values)
        adapter = StateDetailLatentAdapter(
            codec_width=config.codec_width,
            scalar_width=config.scalar_width,
            vector_width=config.vector_width,
            ratio=config.ratio,
            mode=config.mode,
        )
        model = MolecularDiT(
            adapter=adapter, scalar_width=config.scalar_width,
            vector_width=config.vector_width, depth=config.depth,
            heads=config.heads, ffn_multiplier=config.ffn_multiplier,
            dropout=config.dropout,
            execution_backend=str(metadata["execution_backend"]),
            ffn_norm_source=str(metadata["ffn_norm_source"]),
        ).to(selected_device)
        trainer = DiTTrainer(
            model, adapter, config=config, statistics=statistics,
            codec=codec, frame_encoder=codec.frame_encoder,
        )
        loaded = trainer.load_checkpoint(path, expected_sha256=expected_sha256)
        if module_state_hash(codec) != config.codec_hash:
            raise ValueError("DiT inference codec weight hash differs")
        codec.eval()
        model.eval()
        return {
            "codec": codec,
            "model": model,
            "adapter": adapter,
            "statistics": statistics,
            "data_hash": config.data_hash,
            "codec_hash": config.codec_hash,
            "step": trainer.step,
            "cursor": loaded["cursor"],
            "source_mode": config.source_mode,
            "center_kind": config.center_kind,
        }
    finally:
        restore_rng_state(prior_rng)
