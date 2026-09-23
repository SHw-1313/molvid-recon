"""Fit statistics and train the staged Frame Joint v1 model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, Subset

from ..checkpoints import capture_rng_state, frame_target_provenance, load_codec_artifact
from ..codec.frame import FrozenFrameTeacher
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.manifest import load_datasets
from ..data.sampling import (
    TaskAwareClipBatchSampler,
    TrajectoryCappedBatchSampler,
    get_clip_specs,
)
from ..data.store import ClipMMapDataset
from ..latent.statistics import FrameLatentStatistics
from ..model import FrameJointModel
from ..runtime import atomic_write_json, canonical_hash, configure_device, seed_all, sha256_file
from ..training.batches import prepare_batch_then_to_device, prepare_frame_joint_batch
from ..training.joint import (
    FrameJointTrainer,
    JointLossConfig,
    SampledAuxiliaryConfig,
    STAGES,
    calibrate_generated_bond_weight,
    calibrate_sampled_auxiliary_weights,
    fit_sampled_feature_scales,
)


SCHEMA = "molvid.frame_joint.train.v1"

_SECTION_KEYS = {
    "data": {
        "manifest_root", "train_view", "train_store", "valid_store", "base_data_hash",
        "normalization_source_data_hash", "expected_data_hash", "train_systems", "derived_train",
    },
    "codec": {"checkpoint", "sha256"},
    "statistics": {"path", "sha256", "statistics_hash", "max_atom_frames_per_gpu"},
    "model": {"scalar_width", "vector_width", "depth", "heads", "geometry_enabled", "motion_enabled"},
    "loss": {
        "bond_enabled", "generated_bond", "clean_coordinate", "clean_bond", "near_coordinate",
        "near_bond", "generated_bond_min_flow_time", "near_min_flow_time",
        "calibration_target_ratio", "inherit_parent",
    },
    "training": {
        "device", "deterministic", "precision", "seed", "max_atom_frames_per_gpu",
        "clips_per_trajectory", "time_bucket_weights", "history_order", "weight_decay",
        "grad_clip", "output_root", "stages",
    },
}
_DERIVED_KEYS = {"store", "manifest", "manifest_sha256", "store_index_sha256", "record_count"}
_STAGE_KEYS = {"name", "updates", "learning_rates"}
_LEARNING_RATE_KEYS = {"history_encoder", "dit", "decoder"}
_SAMPLED_AUXILIARY_KEYS = {
    "enabled", "cadence", "draws", "euler_steps", "activation_checkpoint",
    "energy_weight", "observed_bond_weight", "feature_scales",
    "calibration_batches", "energy_gradient_target_ratio",
    "observed_bond_gradient_target_ratio",
}
_EXPOSURE_KEYS = {
    "schema", "successful_updates", "trajectory_clip_exposure", "bucket_clip_exposure",
    "view_history_clip_exposure", "valid_atom_frames",
}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} keys: {unknown}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_bool(mapping: Mapping[str, Any], name: str, path: str) -> None:
    if name in mapping and not isinstance(mapping[name], bool):
        raise ValueError(f"{path}.{name} must be a boolean")


def _require_string(value: Any, path: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{path} must be a {'string' if allow_empty else 'nonempty string'}")


def _require_number(
    value: Any,
    path: str,
    *,
    integer: bool = False,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ValueError(f"{path} must be {'an integer' if integer else 'numeric'}")
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0) or (nonnegative and numeric < 0):
        qualifier = " and positive" if positive else " and non-negative" if nonnegative else ""
        raise ValueError(f"{path} must be finite{qualifier}")


def _validate_frame_joint_config(raw: Mapping[str, Any]) -> None:
    """Reject unknown or mistyped training settings before data or CUDA setup."""

    _reject_unknown(raw, {"schema", "sampled_auxiliary", *_SECTION_KEYS}, "root")
    for section, allowed in _SECTION_KEYS.items():
        value = _require_mapping(raw.get(section), section)
        _reject_unknown(value, allowed, section)

    data = raw["data"]
    manifest_mode = "manifest_root" in data
    direct_mode = "train_store" in data or "valid_store" in data
    if manifest_mode == direct_mode:
        raise ValueError("data must select exactly one of manifest_root or train_store/valid_store")
    if manifest_mode:
        _require_string(data["manifest_root"], "data.manifest_root")
    if direct_mode:
        for name in ("train_store", "valid_store"):
            _require_string(data.get(name), f"data.{name}")
    for name in (
        "train_view", "base_data_hash", "normalization_source_data_hash", "expected_data_hash"
    ):
        if name in data:
            _require_string(data[name], f"data.{name}")
    if "train_systems" in data and (
        not isinstance(data["train_systems"], list)
        or not all(isinstance(value, str) and value for value in data["train_systems"])
    ):
        raise ValueError("data.train_systems must be a list of nonempty strings")
    if "derived_train" in data:
        derived = data["derived_train"]
        if not isinstance(derived, list) or not derived:
            raise ValueError("data.derived_train must be a nonempty list")
        for index, item in enumerate(derived):
            value = _require_mapping(item, f"data.derived_train[{index}]")
            _reject_unknown(value, _DERIVED_KEYS, f"data.derived_train[{index}]")
            missing = sorted(_DERIVED_KEYS - set(value))
            if missing:
                raise ValueError(f"data.derived_train[{index}] missing keys: {missing}")
            for name in ("store", "manifest", "manifest_sha256", "store_index_sha256"):
                _require_string(value[name], f"data.derived_train[{index}].{name}")
            _require_number(value["record_count"], f"data.derived_train[{index}].record_count", integer=True, positive=True)

    codec = raw["codec"]
    for name in ("checkpoint", "sha256"):
        _require_string(codec.get(name), f"codec.{name}")
    statistics = raw["statistics"]
    _require_string(statistics.get("path"), "statistics.path")
    for name in ("sha256", "statistics_hash"):
        if name in statistics:
            _require_string(statistics[name], f"statistics.{name}", allow_empty=True)
    if "max_atom_frames_per_gpu" in statistics:
        _require_number(statistics["max_atom_frames_per_gpu"], "statistics.max_atom_frames_per_gpu", integer=True, positive=True)

    model = raw["model"]
    for name in ("scalar_width", "vector_width", "depth", "heads"):
        if name in model:
            _require_number(model[name], f"model.{name}", integer=True, positive=True)
    for name in ("geometry_enabled", "motion_enabled"):
        _require_bool(model, name, "model")

    loss = raw["loss"]
    for name in ("bond_enabled", "inherit_parent"):
        _require_bool(loss, name, "loss")
    for name, value in loss.items():
        if name in {"bond_enabled", "inherit_parent"}:
            continue
        if name == "generated_bond" and value == "calibrate":
            continue
        _require_number(value, f"loss.{name}", nonnegative=True)

    training = raw["training"]
    _require_bool(training, "deterministic", "training")
    for name in ("device", "output_root"):
        _require_string(training.get(name), f"training.{name}")
    for name in ("seed", "max_atom_frames_per_gpu", "clips_per_trajectory"):
        if name not in training:
            raise ValueError(f"training requires {name}")
        _require_number(training[name], f"training.{name}", integer=True, positive=name != "seed")
    for name in ("weight_decay", "grad_clip"):
        if name in training:
            _require_number(training[name], f"training.{name}", nonnegative=True)
    if training.get("precision", "fp32") not in {"fp32", "bf16"}:
        raise ValueError("training.precision must be fp32 or bf16")
    histories = training.get("history_order", (4, 8))
    if not isinstance(histories, list) or not histories:
        raise ValueError("training.history_order must be a nonempty list")
    for index, value in enumerate(histories):
        _require_number(value, f"training.history_order[{index}]", integer=True, positive=True)
    weights = training.get("time_bucket_weights")
    if weights is not None:
        weights = _require_mapping(weights, "training.time_bucket_weights")
        for name, value in weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("training.time_bucket_weights keys must be nonempty strings")
            _require_number(value, f"training.time_bucket_weights.{name}", positive=True)
    stages = training.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("training.stages must be a nonempty list")
    for index, item in enumerate(stages):
        stage = _require_mapping(item, f"training.stages[{index}]")
        _reject_unknown(stage, _STAGE_KEYS, f"training.stages[{index}]")
        if stage.get("name") not in STAGES:
            raise ValueError(f"training.stages[{index}].name is unsupported")
        _require_number(stage.get("updates"), f"training.stages[{index}].updates", integer=True, positive=True)
        rates = _require_mapping(stage.get("learning_rates"), f"training.stages[{index}].learning_rates")
        _reject_unknown(rates, _LEARNING_RATE_KEYS, f"training.stages[{index}].learning_rates")
        missing = sorted(_LEARNING_RATE_KEYS - set(rates))
        if missing:
            raise ValueError(f"training.stages[{index}].learning_rates missing keys: {missing}")
        for name, value in rates.items():
            _require_number(
                value,
                f"training.stages[{index}].learning_rates.{name}",
                nonnegative=True,
            )

    sampled = _require_mapping(raw.get("sampled_auxiliary", {}), "sampled_auxiliary")
    _reject_unknown(sampled, _SAMPLED_AUXILIARY_KEYS, "sampled_auxiliary")
    for name in ("enabled", "activation_checkpoint"):
        _require_bool(sampled, name, "sampled_auxiliary")
    for name, frozen in (("cadence", 8), ("draws", 2), ("euler_steps", 4)):
        if name in sampled:
            _require_number(
                sampled[name], f"sampled_auxiliary.{name}", integer=True, positive=True
            )
            if int(sampled[name]) != frozen:
                raise ValueError(
                    "sampled auxiliary is frozen at cadence=8, K=2, Euler steps=4"
                )
    if "calibration_batches" in sampled:
        _require_number(
            sampled["calibration_batches"],
            "sampled_auxiliary.calibration_batches",
            integer=True,
            positive=True,
        )
        if int(sampled["calibration_batches"]) != 8:
            raise ValueError("sampled auxiliary calibration is frozen at eight train batches")
    for name in ("energy_gradient_target_ratio", "observed_bond_gradient_target_ratio"):
        if name in sampled:
            _require_number(sampled[name], f"sampled_auxiliary.{name}", positive=True)
            if not math.isclose(float(sampled[name]), 0.05, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("sampled auxiliary gradient target ratios are frozen at 0.05")
    for name in ("energy_weight", "observed_bond_weight"):
        value = sampled.get(name, 0.0)
        if value != "calibrate":
            _require_number(value, f"sampled_auxiliary.{name}", nonnegative=True)
    feature_scales = sampled.get("feature_scales")
    if feature_scales != "fit" and feature_scales is not None:
        values = _require_mapping(feature_scales, "sampled_auxiliary.feature_scales")
        if set(values) != {
            "residue_rmsf_A",
            "displacement_squared_A2",
            "internal_distance_increment_A",
            "internal_distance_increment_product_A2",
        }:
            raise ValueError("sampled auxiliary feature scales have unexpected keys")
        for name, value in values.items():
            _require_number(
                value, f"sampled_auxiliary.feature_scales.{name}", positive=True
            )
    enabled = bool(sampled.get("enabled", False))
    if enabled and feature_scales is None:
        raise ValueError("enabled sampled auxiliary requires feature_scales or 'fit'")
    calibrated_weights = [
        sampled.get(name) == "calibrate"
        for name in ("energy_weight", "observed_bond_weight")
    ]
    if any(calibrated_weights) and not all(calibrated_weights):
        raise ValueError("sampled auxiliary weights must be calibrated together")


def _verified_derived_identity(
    item: Mapping[str, Any], manifest: Mapping[str, Any], store_path: Path, record_count: int
) -> dict[str, Any]:
    actual_index = sha256_file(store_path / "index.txt")
    for source, expected in (
        ("derived manifest", manifest.get("store_index_sha256")),
        ("training config", item.get("store_index_sha256")),
    ):
        if str(expected) != actual_index:
            raise ValueError(f"{source} store index SHA-256 differs")
    for source, expected in (
        ("derived manifest", manifest.get("record_count")),
        ("training config", item.get("record_count")),
    ):
        if isinstance(expected, bool) or not isinstance(expected, int) or expected != record_count:
            raise ValueError(f"{source} record count differs")
    return {"index_sha256": actual_index, "record_count": record_count}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-statistics", action="store_true")
    parser.add_argument("--fit-statistics-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--continuation-parent", type=Path)
    parser.add_argument("--continuation-parent-sha256")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--warm-start-sha256")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _distributed_device(training: Mapping[str, Any]) -> tuple[torch.device, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Frame Joint DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = configure_device(
            f"cuda:{local_rank}", deterministic=bool(training.get("deterministic", False))
        )
    else:
        device = configure_device(
            str(training.get("device", "cuda")),
            deterministic=bool(training.get("deterministic", False)),
        )
    return device, rank, local_rank, world


def _direct_data_hash(train: ClipMMapDataset, valid: ClipMMapDataset) -> str:
    return canonical_hash({
        "schema": "molvid.frame_joint.direct_clip_stores.v1",
        "train_index_sha256": sha256_file(train.root / "index.txt"),
        "valid_index_sha256": sha256_file(valid.root / "index.txt"),
        "train_count": len(train),
        "valid_count": len(valid),
        "test_opened": False,
    })


def _open_data(data: Mapping[str, Any]):
    manifest_root = data.get("manifest_root")
    if manifest_root:
        splits = load_datasets(manifest_root, train_view=str(data.get("train_view", "train")))
        expected_base = str(data.get("base_data_hash", ""))
        if expected_base and expected_base != splits.data_hash:
            splits.close()
            raise ValueError("base data hash differs from the derived-view contract")
        derived = data.get("derived_train", ())
        if derived:
            if not isinstance(derived, list) or not all(isinstance(item, Mapping) for item in derived):
                splits.close()
                raise ValueError("data.derived_train must be a list of mappings")
            stores: list[ClipMMapDataset] = []
            identities: list[dict[str, Any]] = []
            try:
                for item in derived:
                    store_path = Path(item["store"])
                    manifest_path = Path(item["manifest"])
                    expected_manifest = str(item["manifest_sha256"])
                    actual_manifest = sha256_file(manifest_path)
                    if actual_manifest != expected_manifest:
                        raise ValueError("derived-view manifest SHA-256 differs")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("test_opened") is not False or manifest.get("split") != "train":
                        raise ValueError("derived training view must declare split=train and sealed test")
                    if manifest.get("source_data_hash") != splits.data_hash:
                        raise ValueError("derived training view points to a different base data identity")
                    store = ClipMMapDataset(store_path)
                    stores.append(store)
                    store_identity = _verified_derived_identity(item, manifest, store_path, len(store))
                    identities.append({
                        "store": str(store_path.resolve()),
                        **store_identity,
                        "manifest": str(manifest_path.resolve()),
                        "manifest_sha256": actual_manifest,
                        "contract_hash": manifest.get("contract_hash"),
                    })
                train: Any = ConcatDataset([splits.train, *stores])
                requested_systems = tuple(str(value) for value in data.get("train_systems", ()))
                if requested_systems:
                    requested = set(requested_systems)
                    selected_indices = []
                    selected_trajectories: set[str] = set()
                    for item in get_clip_specs(train):
                        match = re.match(r"^(atlas_.+)_R([123])_", item.sample_id)
                        if match is not None and match.group(1) in requested:
                            selected_indices.append(int(item.index))
                            selected_trajectories.add(f"{match.group(1)}_R{match.group(2)}")
                    if set(requested_systems) != {
                        value.rsplit("_R", 1)[0] for value in selected_trajectories
                    } or len(selected_trajectories) != 3 * len(requested_systems):
                        raise ValueError("pilot train_systems do not form a complete three-replica grid")
                    train = Subset(train, selected_indices)
                current_hash = canonical_hash({
                    "schema": "molvid.frame_joint.derived_train_data.v1",
                    "base_data_hash": splits.data_hash,
                    "derived_train": identities,
                    "train_systems": list(requested_systems),
                    "valid_index_sha256": splits.valid_index_hash,
                    "test_opened": False,
                })
                expected_hash = str(data.get("expected_data_hash", ""))
                if expected_hash and expected_hash != current_hash:
                    raise ValueError("derived training data hash differs from the configured identity")
                normalization_hash = str(data.get("normalization_source_data_hash", ""))
                if normalization_hash != splits.data_hash:
                    raise ValueError(
                        "derived training requires normalization_source_data_hash equal to base_data_hash"
                    )
                def close() -> None:
                    for store in stores:
                        store.close()
                    splits.close()

                relationship = {
                    "mode": "authorized_train_only_derived_views",
                    "base_data_hash": splits.data_hash,
                    "normalization_source_data_hash": normalization_hash,
                    "derived_train": identities,
                    "train_systems": list(requested_systems),
                }
                return train, splits.valid, current_hash, normalization_hash, relationship, close
            except Exception:
                for store in stores:
                    store.close()
                splits.close()
                raise
        return splits.train, splits.valid, splits.data_hash, splits.data_hash, None, splits.close
    train = ClipMMapDataset(data["train_store"])
    valid = ClipMMapDataset(data["valid_store"])
    data_hash = _direct_data_hash(train, valid)
    return train, valid, data_hash, data_hash, None, lambda: (train.close(), valid.close())


def _statistics_stream(
    teacher: FrozenFrameTeacher,
    dataset: ClipMMapDataset,
    *,
    device: torch.device,
    max_tokens: int,
):
    sampler = TaskAwareClipBatchSampler(
        dataset,
        max_tokens=max_tokens,
        seed=0,
        shuffle=False,
        replacement=False,
        oversize_policy="partition",
    )
    for indices in sampler.global_batches:
        batch = collate_clip_records([dataset[index] for index in indices])
        coordinate = prepare_batch_then_to_device(teacher, batch, device)
        latent, _ = teacher(coordinate)
        yield latent


def _fit_statistics(
    artifact: Any,
    dataset: ClipMMapDataset,
    *,
    device: torch.device,
    path: Path,
    data_hash: str,
    max_tokens: int,
) -> FrameLatentStatistics:
    teacher = FrozenFrameTeacher.from_codec(artifact.model).to(device).eval()
    statistics = FrameLatentStatistics.fit(
        _statistics_stream(teacher, dataset, device=device, max_tokens=max_tokens),
        provenance={
            "scope": "complete_training_store",
            "training_clip_count": len(dataset),
            "data_hash": data_hash,
            "codec_checkpoint_sha256": artifact.report.source_sha256,
            "vector_mean_subtraction": False,
            "vector_xyz_shared_scale": True,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statistics.state_dict(), path)
    atomic_write_json(path.with_suffix(".json"), {
        "statistics_hash": statistics.hash,
        "file_sha256": sha256_file(path),
        "provenance": dict(statistics.provenance),
    })
    return statistics.to(device)


def _load_statistics(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    normalization_data_hash: str,
    codec_sha256: str,
) -> FrameLatentStatistics:
    path = Path(config["path"])
    expected_file = str(config.get("sha256", ""))
    if expected_file and expected_file != sha256_file(path):
        raise ValueError("frame statistics file SHA-256 differs")
    statistics = FrameLatentStatistics.from_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    expected_contract = str(config.get("statistics_hash", ""))
    if expected_contract and expected_contract != statistics.hash:
        raise ValueError("frame statistics contract hash differs")
    if statistics.provenance.get("data_hash") != normalization_data_hash:
        raise ValueError("frame statistics normalization source differs")
    if statistics.provenance.get("codec_checkpoint_sha256") != codec_sha256:
        raise ValueError("frame statistics target encoder differs")
    return statistics.to(device)


def _stages(training: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = training.get("stages")
    if not isinstance(values, list) or not values:
        raise ValueError("training.stages must be a nonempty list")
    result: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, Mapping) or str(item.get("name")) not in STAGES:
            raise ValueError("each stage requires a supported name")
        updates = int(item.get("updates", 0))
        rates = item.get("learning_rates")
        if updates < 1 or not isinstance(rates, Mapping):
            raise ValueError("each stage requires positive updates and learning_rates")
        resolved_rates = {
            name: float(rates[name]) for name in ("history_encoder", "dit", "decoder")
        }
        if any(not math.isfinite(value) or value < 0 for value in resolved_rates.values()):
            raise ValueError("stage learning rates must be finite and non-negative")
        result.append({"name": str(item["name"]), "updates": updates, "learning_rates": resolved_rates})
    return tuple(result)


def _stage_at(stages: Sequence[Mapping[str, Any]], step: int) -> tuple[Mapping[str, Any], int]:
    offset = 0
    for stage in stages:
        end = offset + int(stage["updates"])
        if step < end:
            return stage, step - offset
        offset = end
    return stages[-1], int(stages[-1]["updates"]) - 1


def _scheduled_rates(stage: Mapping[str, Any], position: int) -> dict[str, float]:
    updates = int(stage["updates"])
    warmup = max(1, math.ceil(0.05 * updates))
    if position < warmup:
        factor = float(position + 1) / warmup
    else:
        progress = float(position - warmup) / max(1, updates - warmup - 1)
        factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return {name: float(value) * factor for name, value in stage["learning_rates"].items()}


def _sampler(
    dataset: Any,
    training: Mapping[str, Any],
    *,
    seed: int,
    rank: int,
    world: int,
) -> TrajectoryCappedBatchSampler:
    return TrajectoryCappedBatchSampler(
        dataset,
        max_tokens=int(training["max_atom_frames_per_gpu"]),
        clips_per_trajectory=int(training["clips_per_trajectory"]),
        seed=seed,
        shuffle=True,
        num_replicas=world,
        rank=rank,
        drop_last=world > 1,
        time_bucket_weights=training.get("time_bucket_weights"),
    )


def _calibration_stream(
    model: FrameJointModel,
    dataset: ClipMMapDataset,
    training: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
    histories: Sequence[int],
    count: int = 16,
    sampler_seed_offset: int = 7000,
):
    sampler = _sampler(
        dataset, training, seed=seed + sampler_seed_offset, rank=0, world=1
    )
    yielded = 0
    epoch = 0
    while yielded < count:
        sampler.set_epoch(epoch)
        for indices in sampler:
            cpu_batch = collate_clip_records([dataset[index] for index in indices])
            history = int(histories[yielded % len(histories)])
            yield prepare_frame_joint_batch(
                model.target_teacher,
                cpu_batch,
                device=device,
                normalizer=model,
                history_frames=history,
            )
            yielded += 1
            if yielded == count:
                return
        epoch += 1


def _trajectory(sample_id: str) -> str:
    value = str(sample_id).rsplit("_w", 1)[0]
    return re.sub(r"_dt_[0-9]+(?:\.[0-9]+)?ps$", "", value)


def _exposure_state(
    exposures: Mapping[str, int],
    bucket_exposures: Mapping[str, int],
    view_history_exposures: Mapping[str, int],
    *,
    successful_updates: int,
    valid_atom_frames: int,
) -> dict[str, Any]:
    return {
        "schema": "molvid.frame_joint.exposure.rank.v1",
        "successful_updates": int(successful_updates),
        "trajectory_clip_exposure": dict(sorted((str(key), int(value)) for key, value in exposures.items())),
        "bucket_clip_exposure": dict(sorted((str(key), int(value)) for key, value in bucket_exposures.items())),
        "view_history_clip_exposure": dict(
            sorted((str(key), int(value)) for key, value in view_history_exposures.items())
        ),
        "valid_atom_frames": int(valid_atom_frames),
    }


def _validate_exposure_state(value: Any, *, successful_updates: int) -> Mapping[str, Any]:
    state = _require_mapping(value, "checkpoint exposure_state")
    _reject_unknown(state, _EXPOSURE_KEYS, "checkpoint exposure_state")
    if state.get("schema") != "molvid.frame_joint.exposure.rank.v1":
        raise ValueError("checkpoint exposure_state schema differs")
    if state.get("successful_updates") != successful_updates:
        raise ValueError("checkpoint exposure_state update count differs")
    for name in ("trajectory_clip_exposure", "bucket_clip_exposure", "view_history_clip_exposure"):
        counts = _require_mapping(state.get(name), f"checkpoint exposure_state.{name}")
        for key, count in counts.items():
            if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"checkpoint exposure_state.{name} is invalid")
    count = state.get("valid_atom_frames")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("checkpoint exposure_state.valid_atom_frames is invalid")
    return state


def _merge_exposure_states(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trajectories: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    views: Counter[str] = Counter()
    valid_atom_frames = 0
    successful_updates: int | None = None
    for value in values:
        updates = int(value["successful_updates"])
        if successful_updates is None:
            successful_updates = updates
        elif successful_updates != updates:
            raise ValueError("rank exposure states disagree on successful updates")
        trajectories.update(value["trajectory_clip_exposure"])
        buckets.update(value["bucket_clip_exposure"])
        views.update(value["view_history_clip_exposure"])
        valid_atom_frames += int(value["valid_atom_frames"])
    return {
        "successful_updates": int(successful_updates or 0),
        "trajectory_clip_exposure": dict(sorted(trajectories.items())),
        "bucket_clip_exposure": dict(sorted(buckets.items())),
        "view_history_clip_exposure": dict(sorted(views.items())),
        "valid_atom_frames": valid_atom_frames,
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
    ).strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parent_modes = [args.resume is not None, args.continuation_parent is not None, args.warm_start is not None]
    if sum(parent_modes) > 1:
        raise ValueError("resume, continuation parent, and warm start are mutually exclusive")
    if args.continuation_parent and not args.continuation_parent_sha256:
        raise ValueError("continuation parent requires its SHA-256")
    if args.warm_start and not args.warm_start_sha256:
        raise ValueError("warm start requires its SHA-256")
    if args.fit_statistics_only:
        args.fit_statistics = True
    raw = load_config(args.config, schema=SCHEMA)
    _validate_frame_joint_config(raw)
    training = dict(raw["training"])
    if args.output_root is not None:
        training["output_root"] = str(args.output_root)
    device, rank, local_rank, world = _distributed_device(training)
    if args.fit_statistics and world != 1:
        raise ValueError("fit statistics in one CUDA process before launching DDP")
    seed = int(training["seed"])
    seed_all(seed + rank)
    train, valid, data_hash, normalization_data_hash, data_relationship, close_data = _open_data(raw["data"])
    del valid  # Validation is opened for identity only; this command never opens test.
    try:
        codec_config = raw["codec"]
        artifact = load_codec_artifact(
            codec_config["checkpoint"],
            expected_sha256=str(codec_config["sha256"]),
            device=device,
        )
        codec = artifact.model.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        stats_path = Path(raw["statistics"]["path"])
        if args.fit_statistics:
            statistics = _fit_statistics(
                artifact,
                train,
                device=device,
                path=stats_path,
                data_hash=data_hash,
                max_tokens=int(raw["statistics"].get("max_atom_frames_per_gpu", 80000)),
            )
            print(json.dumps({
                "statistics": str(stats_path),
                "statistics_hash": statistics.hash,
                "sha256": sha256_file(stats_path),
            }, sort_keys=True))
            if args.fit_statistics_only:
                return 0
        else:
            statistics = _load_statistics(
                raw["statistics"],
                device=device,
                normalization_data_hash=normalization_data_hash,
                codec_sha256=artifact.report.source_sha256,
            )

        seed_all(seed + rank)
        model_config = raw["model"]
        model = FrameJointModel.from_codec(
            codec,
            statistics,
            scalar_width=int(model_config.get("scalar_width", 256)),
            vector_width=int(model_config.get("vector_width", 128)),
            depth=int(model_config.get("depth", 4)),
            heads=int(model_config.get("heads", 8)),
            geometry_enabled=bool(model_config.get("geometry_enabled", False)),
            motion_enabled=bool(model_config.get("motion_enabled", False)),
        ).to(device)
        stages = _stages(training)
        histories = tuple(int(value) for value in training.get("history_order", (4, 8)))
        if not histories or any(value < 1 for value in histories):
            raise ValueError("history_order must contain positive H values")
        schedule_contract = {
            "schema_version": "molvid.frame_joint.schedule.v1",
            "stages": [
                {
                    "name": str(stage["name"]),
                    "updates": int(stage["updates"]),
                    "learning_rates": dict(stage["learning_rates"]),
                }
                for stage in stages
            ],
            "history_order": list(histories),
            "learning_rate_schedule": "linear_warmup_5pct_cosine_to_0.1.v1",
        }
        total_updates = sum(int(stage["updates"]) for stage in stages)
        if args.max_steps is not None:
            total_updates = min(total_updates, int(args.max_steps))
        initial_stage = stages[0]
        resume_payload = None
        parent_preview = None
        if args.resume:
            resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            initial_stage = {**stages[0], "name": str(resume_payload["contracts"]["stage"])}
        elif args.continuation_parent:
            parent_preview = torch.load(
                args.continuation_parent, map_location="cpu", weights_only=False
            )
            if any(stage["name"] not in {"continuation", "frozen_decoder"} for stage in stages):
                raise ValueError(
                    "a continuation config may contain only continuation or frozen_decoder stages"
                )
            initial_stage = stages[0]
        elif args.warm_start:
            parent_preview = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        wrapped: torch.nn.Module = model
        if world > 1:
            wrapped = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                # The frozen teacher owns rank-local topology caches and the
                # trainable model has no running-stat buffers.
                broadcast_buffers=False,
                # Stages deliberately freeze whole components after DDP is
                # constructed, then enable them again at later boundaries.
                find_unused_parameters=True,
            )
        loss_mapping = dict(raw["loss"])
        inherit_parent_loss = bool(loss_mapping.pop("inherit_parent", False))
        if resume_payload is not None:
            loss_mapping = dict(resume_payload["contracts"]["loss"])
        elif inherit_parent_loss:
            if parent_preview is None:
                raise ValueError("loss.inherit_parent requires a continuation parent or warm start")
            loss_mapping = dict(parent_preview["contracts"]["loss"])
        calibration_requested = loss_mapping.get("generated_bond") in (None, "calibrate")
        if resume_payload is None and calibration_requested:
            loss_mapping["generated_bond"] = 0.0
        loss_config = JointLossConfig.resolve(loss_mapping)
        sampled_raw = dict(raw.get("sampled_auxiliary", {}))
        sampled_enabled = bool(sampled_raw.get("enabled", False))
        sampled_weight_calibration = (
            sampled_enabled and sampled_raw.get("energy_weight") == "calibrate"
        )
        sampled_scale_fitting = (
            sampled_enabled and sampled_raw.get("feature_scales") == "fit"
        )
        sampled_runtime_keys = {
            "enabled", "cadence", "draws", "euler_steps", "activation_checkpoint",
            "energy_weight", "observed_bond_weight", "feature_scales",
        }
        if resume_payload is not None:
            resumed_sampled = resume_payload["contracts"].get("sampled_auxiliary")
            if sampled_enabled != isinstance(resumed_sampled, Mapping):
                raise ValueError(
                    "resume sampled auxiliary enablement differs from checkpoint"
                )
            sampled_config = SampledAuxiliaryConfig.resolve(
                {
                    key: value
                    for key, value in resumed_sampled.items()
                    if key in sampled_runtime_keys
                }
                if isinstance(resumed_sampled, Mapping)
                else None
            )
            sampled_weight_calibration = False
            sampled_scale_fitting = False
        elif sampled_enabled:
            provisional = {
                key: value
                for key, value in sampled_raw.items()
                if key in sampled_runtime_keys
            }
            if sampled_scale_fitting:
                provisional["feature_scales"] = {
                    "residue_rmsf_A": 1.0,
                    "displacement_squared_A2": 1.0,
                    "internal_distance_increment_A": 1.0,
                    "internal_distance_increment_product_A2": 1.0,
                }
            if sampled_weight_calibration:
                provisional["energy_weight"] = 0.0
                provisional["observed_bond_weight"] = 0.0
            sampled_config = SampledAuxiliaryConfig.resolve(provisional)
        else:
            sampled_config = SampledAuxiliaryConfig()
        base_rates = {
            name: max(float(stage["learning_rates"][name]) for stage in stages)
            for name in ("history_encoder", "dit", "decoder")
        }
        trainer = FrameJointTrainer(
            wrapped,
            loss_config=loss_config,
            learning_rates=base_rates,
            weight_decay=float(training.get("weight_decay", 0.01)),
            grad_clip=float(training.get("grad_clip", 1.0)),
            stage=str(initial_stage["name"]),
            amp=str(training.get("precision", "fp32")) == "bf16",
            data_hash=data_hash,
            teacher_artifact_sha256=artifact.report.source_sha256,
            schedule_contract=schedule_contract,
            continuation_parent=(
                resume_payload["contracts"].get("continuation_parent")
                if resume_payload is not None
                else None
            ),
            sampled_config=sampled_config,
        )
        generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
        sampled_generator = torch.Generator(device=device).manual_seed(
            seed + 2000 + rank
        )
        sampler = _sampler(train, training, seed=seed, rank=rank, world=world)
        epoch = batch_index = ordinary_batch_index = 0
        sampler.set_epoch(epoch)
        warm_start_report = None
        resumed_exposure: Mapping[str, Any] | None = None
        if args.resume:
            loaded = trainer.load_checkpoint(
                args.resume,
                generator=generator,
                sampled_generator=sampled_generator,
                expected_sha256=args.resume_sha256,
                rank=rank,
            )
            rank_states = loaded["extra_state"].get("rank_states", {})
            rank_state = rank_states.get(str(rank), {}) if isinstance(rank_states, Mapping) else {}
            rank_cursor = rank_state.get("cursor") if isinstance(rank_state, Mapping) else None
            cursor = rank_cursor if isinstance(rank_cursor, Mapping) else loaded["cursor"]
            resumed_exposure = _validate_exposure_state(
                rank_state.get("exposure_state") if isinstance(rank_state, Mapping) else None,
                successful_updates=trainer.successful_updates,
            )
            epoch, batch_index = int(cursor["epoch"]), int(cursor["batch_index"])
            ordinary_batch_index = int(cursor.get("ordinary_batch_index", trainer.step))
            sampler.set_epoch(epoch)
            sampler.validate_state_dict(cursor["sampler_state"])
            if canonical_hash(sampler.global_batches) != cursor["global_batch_schedule_hash"]:
                raise ValueError("resume batch schedule differs")
        elif args.continuation_parent:
            loaded = trainer.load_continuation_parent(
                args.continuation_parent,
                generator=generator,
                expected_sha256=args.continuation_parent_sha256,
                rank=rank,
            )
            rank_states = loaded["extra_state"].get("rank_states", {})
            rank_cursor = rank_states.get(str(rank), {}).get("cursor") if isinstance(rank_states, Mapping) else None
            cursor = rank_cursor if isinstance(rank_cursor, Mapping) else loaded["cursor"]
            epoch, batch_index = int(cursor["epoch"]), int(cursor["batch_index"])
            sampler.set_epoch(epoch)
            sampler.validate_state_dict(cursor["sampler_state"])
            if canonical_hash(sampler.global_batches) != cursor["global_batch_schedule_hash"]:
                raise ValueError("continuation parent batch schedule differs")
        elif args.warm_start:
            warm_start_report = trainer.load_warm_start(
                args.warm_start,
                expected_sha256=args.warm_start_sha256,
            )
        output_root = Path(training["output_root"])
        sampled_scale_diagnostics: dict[str, Any] | None = None
        sampled_weight_diagnostics: dict[str, Any] | None = None
        if sampled_scale_fitting or sampled_weight_calibration:
            if world != 1:
                raise ValueError(
                    "fit sampled auxiliary calibration in one CUDA process, "
                    "then use the resolved numeric contract for DDP"
                )
            scales = trainer.sampled_config.feature_scales
            if sampled_scale_fitting:
                scales, sampled_scale_diagnostics = fit_sampled_feature_scales(
                    _calibration_stream(
                        model,
                        train,
                        training,
                        device=device,
                        seed=seed,
                        histories=histories,
                        count=8,
                        sampler_seed_offset=17000,
                    )
                )
            if scales is None:
                raise ValueError("sampled auxiliary calibration lacks feature scales")
            energy_weight = trainer.sampled_config.energy_weight
            observed_bond_weight = trainer.sampled_config.observed_bond_weight
            if sampled_weight_calibration:
                (
                    (energy_weight, observed_bond_weight),
                    sampled_weight_diagnostics,
                ) = calibrate_sampled_auxiliary_weights(
                    model,
                    _calibration_stream(
                        model,
                        train,
                        training,
                        device=device,
                        seed=seed,
                        histories=histories,
                        count=8,
                        sampler_seed_offset=17000,
                    ),
                    main_generator=torch.Generator(device=device).manual_seed(
                        seed + 18000
                    ),
                    sampled_generator=torch.Generator(device=device).manual_seed(
                        seed + 19000
                    ),
                    feature_scales=scales,
                    energy_target_ratio=float(
                        sampled_raw.get("energy_gradient_target_ratio", 0.05)
                    ),
                    observed_bond_target_ratio=float(
                        sampled_raw.get(
                            "observed_bond_gradient_target_ratio", 0.05
                        )
                    ),
                    checkpoint_sampled_steps=trainer.sampled_config.activation_checkpoint,
                )
            trainer.sampled_config = SampledAuxiliaryConfig.resolve({
                "enabled": True,
                "cadence": trainer.sampled_config.cadence,
                "draws": trainer.sampled_config.draws,
                "euler_steps": trainer.sampled_config.euler_steps,
                "activation_checkpoint": trainer.sampled_config.activation_checkpoint,
                "energy_weight": energy_weight,
                "observed_bond_weight": observed_bond_weight,
                "feature_scales": scales.contract(),
            })
        resolved_config = {
            **raw,
            "resolved": {
                "data_hash": data_hash,
                "normalization_source_data_hash": normalization_data_hash,
                "data_relationship": data_relationship,
                "statistics_hash": statistics.hash,
                "statistics_sha256": sha256_file(stats_path),
                "codec_artifact_step": artifact.step,
                "world_size": world,
                "total_updates": total_updates,
                "output_root": str(output_root),
                "continuation_parent": trainer.continuation_parent,
                "loss_contract": trainer.loss_config.contract(),
                "sampled_auxiliary_contract": (
                    trainer.sampled_config.contract()
                    if trainer.sampled_config.enabled
                    else None
                ),
            },
        }
        if rank == 0:
            output_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output_root / "resolved_config.json", resolved_config)
            if warm_start_report is not None:
                atomic_write_json(output_root / "warm_start_report.json", warm_start_report.as_dict())
            if sampled_scale_diagnostics is not None:
                atomic_write_json(
                    output_root / "sampled_feature_scale_calibration.json",
                    sampled_scale_diagnostics,
                )
            if sampled_weight_diagnostics is not None:
                atomic_write_json(
                    output_root / "sampled_auxiliary_gradient_calibration.json",
                    sampled_weight_diagnostics,
                )
            specs = get_clip_specs(train)
            by_index = {int(item.index): item for item in specs}
            sampler_manifest = {
                "schema": "molvid.frame_gm.p2_sampler_schedule.v1",
                "epoch": epoch,
                "sampler_state": sampler.state_dict(),
                "global_batch_schedule": [list(batch) for batch in sampler.global_batches],
                "global_batch_schedule_hash": canonical_hash(sampler.global_batches),
                "selected_sample_ids": list(sampler.selected_sample_ids),
                "selected_sample_ids_hash": canonical_hash(sampler.selected_sample_ids),
                "selected_bucket_counts": dict(sorted(Counter(
                    by_index[index].time_bucket_id
                    for batch in sampler.global_batches
                    for index in batch
                ).items())),
                "test_opened": False,
            }
            atomic_write_json(output_root / "sampler_manifest.json", sampler_manifest)
            atomic_write_json(
                output_root / "target_encoder_provenance.json",
                frame_target_provenance(
                    artifact,
                    source_path=codec_config["checkpoint"],
                    code_commit=_git_commit(),
                ),
            )
        exposures: Counter[str] = Counter(
            {} if resumed_exposure is None else resumed_exposure["trajectory_clip_exposure"]
        )
        bucket_exposures: Counter[str] = Counter(
            {} if resumed_exposure is None else resumed_exposure["bucket_clip_exposure"]
        )
        view_history_exposures: Counter[str] = Counter(
            {} if resumed_exposure is None else resumed_exposure["view_history_clip_exposure"]
        )
        valid_atom_frames = int(0 if resumed_exposure is None else resumed_exposure["valid_atom_frames"])
        if args.dry_run:
            total_updates = min(total_updates, trainer.step + 1)
        last: dict[str, Any] = {}
        calibration_done = not calibration_requested or not loss_config.bond_enabled or loss_config.generated_bond > 0
        while trainer.step < total_updates:
            batches = list(iter(sampler))
            if batch_index >= len(batches):
                epoch += 1
                batch_index = 0
                sampler.set_epoch(epoch)
                batches = list(iter(sampler))
            indices = batches[batch_index]
            cpu_batch = collate_clip_records([train[index] for index in indices])
            buckets = set(str(value) for value in cpu_batch.time_bucket_id)
            if len(buckets) != 1:
                raise RuntimeError("Frame Joint sampler produced a mixed time-bucket batch")
            bucket = next(iter(buckets))
            fixed_history = bucket.startswith("fixed_history_dt_")
            if fixed_history:
                history = 4
            else:
                history = histories[ordinary_batch_index % len(histories)]
                ordinary_batch_index += 1
            if history >= cpu_batch.frames:
                raise ValueError("sampled history must leave future query frames")
            stage, stage_position = _stage_at(stages, trainer.step)
            if trainer.stage != stage["name"]:
                trainer.configure_stage(str(stage["name"]))
            if stage["name"] in {"joint", "continuation"} and not calibration_done:
                coefficient = torch.zeros((), device=device)
                diagnostics: dict[str, Any] = {}
                if rank == 0:
                    coefficient_value, diagnostics = calibrate_generated_bond_weight(
                        model,
                        _calibration_stream(
                            model,
                            train,
                            training,
                            device=device,
                            seed=seed,
                            histories=histories,
                        ),
                        generator=torch.Generator(device=device).manual_seed(seed + 8000),
                        target_ratio=float(raw["loss"].get("calibration_target_ratio", 0.1)),
                    )
                    coefficient.fill_(coefficient_value)
                if world > 1:
                    dist.broadcast(coefficient, src=0)
                resolved_loss = dict(raw["loss"])
                resolved_loss.pop("calibration_target_ratio", None)
                resolved_loss["generated_bond"] = float(coefficient)
                trainer.loss_config = JointLossConfig.resolve(resolved_loss)
                calibration_done = True
                if rank == 0:
                    atomic_write_json(output_root / "bond_calibration.json", diagnostics)
                    resolved_config["resolved"]["loss_contract"] = trainer.loss_config.contract()
                    atomic_write_json(output_root / "resolved_config.json", resolved_config)
            trainer.set_learning_rates(_scheduled_rates(stage, stage_position))
            prepared = prepare_frame_joint_batch(
                model.target_teacher,
                cpu_batch,
                device=device,
                normalizer=model,
                history_frames=history,
            )
            last = trainer.train_step(
                prepared,
                generator=generator,
                sampled_generator=sampled_generator,
            )
            last.update({
                "epoch": epoch,
                "batch_index": batch_index,
                "history_frames": history,
                "sample_ids": list(cpu_batch.sample_id),
                "atom_frames": cpu_batch.atom_count * cpu_batch.frames,
                "valid_atom_frames": int(
                    cpu_batch.frame_mask.index_select(0, cpu_batch.abid).transpose(0, 1).sum()
                ),
                "view_kind": "fixed_history" if fixed_history else "legacy_uniform",
                "time_bucket_id": bucket,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in trainer.optimizer.param_groups
                },
            })
            for sample_id in cpu_batch.sample_id:
                exposures[_trajectory(sample_id)] += 1
            bucket_exposures[bucket] += cpu_batch.batch_size
            view_history_exposures[
                f"{'fixed_history' if fixed_history else 'legacy_uniform'}|H{history}"
            ] += cpu_batch.batch_size
            valid_atom_frames += int(
                cpu_batch.frame_mask.index_select(0, cpu_batch.abid).transpose(0, 1).sum()
            )
            batch_index += 1
            if rank == 0:
                with (output_root / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(last, sort_keys=True) + "\n")
            save_now = args.checkpoint_every and trainer.step % args.checkpoint_every == 0
            if save_now or trainer.step == total_updates:
                cursor = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "ordinary_batch_index": ordinary_batch_index,
                    "sampler_state": sampler.state_dict(),
                    "global_batch_schedule_hash": canonical_hash(sampler.global_batches),
                }
                local_state = {
                    "cursor": cursor,
                    "rng_state": capture_rng_state(),
                    "training_generator_state": generator.get_state().detach().cpu(),
                    "sampled_generator_state": (
                        sampled_generator.get_state().detach().cpu()
                    ),
                    "exposure_state": _exposure_state(
                        exposures,
                        bucket_exposures,
                        view_history_exposures,
                        successful_updates=trainer.successful_updates,
                        valid_atom_frames=valid_atom_frames,
                    ),
                }
                states: list[Any] = [None for _ in range(world)]
                if world > 1:
                    dist.all_gather_object(states, local_state)
                else:
                    states[0] = local_state
                if rank == 0:
                    trainer.save_checkpoint(
                        output_root / f"frame_joint_step_{trainer.step:08d}.pt",
                        cursor=cursor,
                        generator=generator,
                        sampled_generator=sampled_generator,
                        rank_states={str(index): state for index, state in enumerate(states)},
                    )
                if world > 1:
                    dist.barrier()
        local_exposure = _exposure_state(
            exposures,
            bucket_exposures,
            view_history_exposures,
            successful_updates=trainer.successful_updates,
            valid_atom_frames=valid_atom_frames,
        )
        rank_exposures: list[Any] = [None for _ in range(world)]
        if world > 1:
            dist.all_gather_object(rank_exposures, local_exposure)
        else:
            rank_exposures[0] = local_exposure
        if rank == 0:
            merged_exposure = _merge_exposure_states(rank_exposures)
            atomic_write_json(output_root / "exposure.json", {
                "schema": "molvid.frame_joint.exposure.v2",
                **merged_exposure,
                "clips_per_trajectory_per_epoch": int(training["clips_per_trajectory"]),
                "world_size": world,
            })
            print(json.dumps({"step": trainer.step, "metrics": last, "output_root": str(output_root)}, sort_keys=True))
        return 0
    finally:
        close_data()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
