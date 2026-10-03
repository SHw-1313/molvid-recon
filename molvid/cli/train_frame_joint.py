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
import time
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
from ..data.store import ClipMMapDataset, StaticClipDataset
from ..latent.statistics import FrameLatentStatistics
from ..losses.sampled_physical import PhysicalAuxiliaryConfig
from ..model import FrameJointModel
from ..runtime import atomic_write_json, canonical_hash, configure_device, seed_all, sha256_file
from ..training.batches import prepare_batch_then_to_device, prepare_frame_joint_batch
from ..training.execution import EXECUTION_SCHEMA, execution_manifest, execution_schedule
from ..training.joint import (
    FrameJointTrainer,
    JointLossConfig,
    SampledAuxiliaryConfig,
    STAGES,
    calibrate_generated_bond_weight,
    calibrate_sampled_auxiliary_weights,
    fit_sampled_feature_scales,
)
from ..training.static_decoder import (
    StaticDecoderAuxiliaryTrainer,
    StaticDecoderLossConfig,
    StaticPerturbationConfig,
    prepare_static_decoder_batch,
    static_clash_loss_function,
)
from ..training.static_schedule import (
    StaticBatchSchedule,
    StaticRNGStreams,
    external_static_pool,
    internal_md_static_pool,
    materialize_static_records,
    validate_static_exposure_plan,
)


SCHEMA = "molvid.frame_joint.train.v1"

_SECTION_KEYS = {
    "data": {
        "manifest_root", "train_view", "train_store", "valid_store", "base_data_hash",
        "normalization_source_data_hash", "expected_data_hash", "train_systems", "derived_train",
    },
    "codec": {"checkpoint", "sha256"},
    "statistics": {"path", "sha256", "statistics_hash", "max_atom_frames_per_gpu"},
    "model": {
        "scalar_width", "vector_width", "depth", "heads", "geometry_enabled",
        "motion_enabled", "geometry_block_indices",
    },
    "loss": {
        "bond_enabled", "generated_bond", "clean_coordinate", "clean_bond", "near_coordinate",
        "near_bond", "generated_bond_min_flow_time", "near_min_flow_time",
        "calibration_target_ratio", "inherit_parent",
    },
    "training": {
        "device", "deterministic", "precision", "seed", "max_atom_frames_per_gpu",
        "clips_per_trajectory", "time_bucket_weights", "history_order", "weight_decay",
        "grad_clip", "output_root", "stages",
        "gradient_diagnostics_every", "warm_start_block_mapping",
        "checkpoint_flow_field",
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
    "physical", "streamed_atom_frames_threshold",
}
_STATIC_AUXILIARY_KEYS = {
    "enabled", "source", "cadence", "seed", "max_atoms_per_update",
    "max_samples_per_update", "min_source_atoms", "max_source_atoms",
    "expected_pool_size", "exposure_target", "perturbation", "loss",
    "gamma_calibration", "external",
}
_STATIC_GAMMA_KEYS = {
    "path", "sha256", "parent_sha256", "code_commit",
    "reference_config_sha256", "recipe_release_sha256",
    "auxiliary_recipe_sha256",
}
_STATIC_EXTERNAL_KEYS = {
    "store", "store_index_sha256", "store_data_sha256", "source", "split",
    "selection_report", "selection_report_sha256", "selected_count",
    "selected_indices_hash", "selection_seed", "code_commit",
    "eligibility_manifest", "eligibility_sha256", "near_homology_policy",
}
_STATIC_EXPOSURE_TARGET_KEYS = {
    "static_updates", "samples", "atoms", "supervised_atoms",
    "sample_tolerance", "atom_tolerance", "supervised_atom_tolerance",
}
_EXPERIMENT_WARM_START_KEYS = {"path", "sha256", "code_commit", "mode"}
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

    _reject_unknown(
        raw,
        {
            "schema", "sampled_auxiliary", "static_decoder_auxiliary",
            "experiment_warm_start", *_SECTION_KEYS,
        },
        "root",
    )
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
    geometry_indices = model.get("geometry_block_indices")
    if geometry_indices is not None:
        if not isinstance(geometry_indices, list) or not geometry_indices:
            raise ValueError("model.geometry_block_indices must be a nonempty list")
        for index, value in enumerate(geometry_indices):
            _require_number(
                value,
                f"model.geometry_block_indices[{index}]",
                integer=True,
                positive=True,
            )
        if geometry_indices != sorted(set(geometry_indices)):
            raise ValueError("model.geometry_block_indices must be unique and increasing")
        if not bool(model.get("geometry_enabled", False)):
            raise ValueError("model.geometry_block_indices require geometry_enabled=true")
        if int(geometry_indices[-1]) > int(model.get("depth", 4)):
            raise ValueError("model.geometry_block_indices exceed model.depth")

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
    _require_bool(training, "checkpoint_flow_field", "training")
    for name in ("device", "output_root"):
        _require_string(training.get(name), f"training.{name}")
    for name in ("seed", "max_atom_frames_per_gpu", "clips_per_trajectory"):
        if name not in training:
            raise ValueError(f"training requires {name}")
        _require_number(training[name], f"training.{name}", integer=True, positive=name != "seed")
    for name in ("weight_decay", "grad_clip"):
        if name in training:
            _require_number(training[name], f"training.{name}", nonnegative=True)
    if "gradient_diagnostics_every" in training:
        _require_number(training["gradient_diagnostics_every"], "training.gradient_diagnostics_every", integer=True, nonnegative=True)
    block_mapping = training.get("warm_start_block_mapping")
    if block_mapping is not None:
        if not isinstance(block_mapping, list) or not block_mapping:
            raise ValueError("training.warm_start_block_mapping must be a nonempty list")
        for index, value in enumerate(block_mapping):
            _require_number(
                value,
                f"training.warm_start_block_mapping[{index}]",
                integer=True,
                positive=True,
            )
        if block_mapping != sorted(set(block_mapping)):
            raise ValueError("training.warm_start_block_mapping must be unique and increasing")
        if int(block_mapping[-1]) > int(model.get("depth", 4)):
            raise ValueError("training.warm_start_block_mapping exceeds model.depth")
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
    for name, allowed in (("cadence", (8,)), ("draws", (2,)), ("euler_steps", (4, 8))):
        if name in sampled:
            _require_number(
                sampled[name], f"sampled_auxiliary.{name}", integer=True, positive=True
            )
            if int(sampled[name]) not in allowed:
                raise ValueError(
                    "sampled auxiliary schedule is frozen at cadence=8, K=2, "
                    "Euler steps=4 or 8"
                )
    if "streamed_atom_frames_threshold" in sampled:
        _require_number(
            sampled["streamed_atom_frames_threshold"],
            "sampled_auxiliary.streamed_atom_frames_threshold",
            integer=True, nonnegative=True,
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
    if sampled.get("physical") is not None:
        PhysicalAuxiliaryConfig.resolve(_require_mapping(sampled["physical"], "sampled_auxiliary.physical"))
    energy_requested = sampled.get("energy_weight", 0.0)
    if enabled and feature_scales is None and (energy_requested == "calibrate" or float(energy_requested) > 0):
        raise ValueError("enabled energy score requires feature_scales or 'fit'")
    calibrated_weights = [
        sampled.get(name) == "calibrate"
        for name in ("energy_weight", "observed_bond_weight")
    ]
    if any(calibrated_weights) and not all(calibrated_weights):
        raise ValueError("sampled auxiliary weights must be calibrated together")

    experiment_parent = raw.get("experiment_warm_start")
    if experiment_parent is not None:
        experiment_parent = _require_mapping(
            experiment_parent, "experiment_warm_start"
        )
        _reject_unknown(
            experiment_parent,
            _EXPERIMENT_WARM_START_KEYS,
            "experiment_warm_start",
        )
        if set(experiment_parent) != _EXPERIMENT_WARM_START_KEYS:
            raise ValueError("experiment warm-start identity is incomplete")
        for name in _EXPERIMENT_WARM_START_KEYS:
            _require_string(
                experiment_parent[name], f"experiment_warm_start.{name}"
            )
        if experiment_parent["mode"] != (
            "model_only_warm_start_with_optimizer_scheduler_cursor_rng_reset"
        ):
            raise ValueError("experiment warm-start mode differs from the paired contract")

    static = _require_mapping(
        raw.get("static_decoder_auxiliary", {}), "static_decoder_auxiliary"
    )
    _reject_unknown(static, _STATIC_AUXILIARY_KEYS, "static_decoder_auxiliary")
    _require_bool(static, "enabled", "static_decoder_auxiliary")
    if not bool(static.get("enabled", False)):
        if set(static) - {"enabled"}:
            raise ValueError("disabled static decoder auxiliary accepts only enabled=false")
        return
    required = _STATIC_AUXILIARY_KEYS - {"enabled", "external"}
    missing = sorted(required - set(static))
    if missing:
        raise ValueError(f"static_decoder_auxiliary missing keys: {missing}")
    if static["source"] not in {"internal_md", "external_static"}:
        raise ValueError("static decoder source must be internal_md or external_static")
    for name in (
        "cadence", "seed", "max_atoms_per_update", "max_samples_per_update",
        "min_source_atoms", "max_source_atoms", "expected_pool_size",
    ):
        _require_number(
            static[name],
            f"static_decoder_auxiliary.{name}",
            integer=True,
            positive=name != "seed",
        )
    if int(static["cadence"]) != 4:
        raise ValueError("static decoder auxiliary cadence is frozen at four dynamic updates")
    if not (
        int(static["min_source_atoms"])
        <= int(static["max_source_atoms"])
        <= int(static["max_atoms_per_update"])
    ):
        raise ValueError("static source atom bounds differ from the per-update atom cap")
    exposure_target = _require_mapping(
        static["exposure_target"], "static_decoder_auxiliary.exposure_target"
    )
    _reject_unknown(
        exposure_target,
        _STATIC_EXPOSURE_TARGET_KEYS,
        "static_decoder_auxiliary.exposure_target",
    )
    if set(exposure_target) != _STATIC_EXPOSURE_TARGET_KEYS:
        raise ValueError("static exposure target identity is incomplete")
    for name, value in exposure_target.items():
        _require_number(
            value,
            f"static_decoder_auxiliary.exposure_target.{name}",
            integer=True,
            nonnegative=True,
        )
    if int(exposure_target["static_updates"]) < 1:
        raise ValueError("static exposure target requires positive static updates")
    StaticPerturbationConfig.resolve(
        _require_mapping(static["perturbation"], "static_decoder_auxiliary.perturbation")
    )
    StaticDecoderLossConfig.resolve(
        _require_mapping(static["loss"], "static_decoder_auxiliary.loss")
    )
    gamma = _require_mapping(
        static["gamma_calibration"], "static_decoder_auxiliary.gamma_calibration"
    )
    _reject_unknown(gamma, _STATIC_GAMMA_KEYS, "static_decoder_auxiliary.gamma_calibration")
    if set(gamma) != _STATIC_GAMMA_KEYS:
        raise ValueError("static gamma calibration identity is incomplete")
    for name in _STATIC_GAMMA_KEYS:
        _require_string(gamma[name], f"static_decoder_auxiliary.gamma_calibration.{name}")
    if static["source"] == "internal_md":
        if int(static["expected_pool_size"]) % 3:
            raise ValueError("internal static pool must contain complete three-replica systems")
        if "external" in static:
            raise ValueError("internal static source cannot define an external store")
    else:
        external = _require_mapping(
            static.get("external"), "static_decoder_auxiliary.external"
        )
        _reject_unknown(external, _STATIC_EXTERNAL_KEYS, "static_decoder_auxiliary.external")
        if set(external) != _STATIC_EXTERNAL_KEYS:
            raise ValueError("external static source identity is incomplete")
        for name in _STATIC_EXTERNAL_KEYS - {"selected_count", "selection_seed"}:
            _require_string(external[name], f"static_decoder_auxiliary.external.{name}")
        for name in ("selected_count", "selection_seed"):
            _require_number(
                external[name],
                f"static_decoder_auxiliary.external.{name}",
                integer=True,
                positive=name == "selected_count",
            )
        if int(external["selected_count"]) != int(static["expected_pool_size"]):
            raise ValueError("external selected count differs from the expected pool size")
        if external["near_homology_policy"] not in {
            "excluded",
            "waived_by_user_20260929",
        }:
            raise ValueError("external near-homology policy is unsupported")


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


def _validate_parent_mode(
    training: Mapping[str, Any],
    *,
    resume: str | Path | None,
    continuation_parent: str | Path | None,
    warm_start: str | Path | None,
) -> None:
    """Validate construction provenance without blocking exact resume.

    ``warm_start_block_mapping`` describes how a fresh child is constructed.
    It remains in the frozen config and checkpoint provenance, but an exact
    resume restores the already-constructed child and must not apply it again.
    """

    modes = [resume is not None, continuation_parent is not None, warm_start is not None]
    if sum(modes) > 1:
        raise ValueError("resume, continuation parent, and warm start are mutually exclusive")
    if (
        training.get("warm_start_block_mapping") is not None
        and warm_start is None
        and resume is None
    ):
        raise ValueError(
            "warm_start_block_mapping requires --warm-start on a fresh run or --resume"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.continuation_parent and not args.continuation_parent_sha256:
        raise ValueError("continuation parent requires its SHA-256")
    if args.warm_start and not args.warm_start_sha256:
        raise ValueError("warm start requires its SHA-256")
    if args.resume and not args.resume_sha256:
        raise ValueError("resume requires its SHA-256")
    if args.fit_statistics_only:
        args.fit_statistics = True
    raw = load_config(args.config, schema=SCHEMA)
    _validate_frame_joint_config(raw)
    experiment_parent = raw.get("experiment_warm_start")
    if experiment_parent is not None:
        if experiment_parent["code_commit"] != _git_commit():
            raise ValueError("paired experiment code differs from the frozen commit")
        if args.continuation_parent:
            raise ValueError("paired experiment requires model-only warm start, not continuation")
        if not args.resume:
            if not args.warm_start or not args.warm_start_sha256:
                raise ValueError("paired experiment requires its frozen warm-start parent")
            if Path(args.warm_start).resolve() != Path(experiment_parent["path"]).resolve():
                raise ValueError("paired experiment warm-start path differs")
            if args.warm_start_sha256 != experiment_parent["sha256"]:
                raise ValueError("paired experiment warm-start SHA-256 differs")
    training = dict(raw["training"])
    _validate_parent_mode(
        training,
        resume=args.resume,
        continuation_parent=args.continuation_parent,
        warm_start=args.warm_start,
    )
    if args.output_root is not None:
        training["output_root"] = str(args.output_root)
    device, rank, local_rank, world = _distributed_device(training)
    if world != 1:
        raise ValueError(
            "stratified execution currently supports one process per arm only; "
            "rank-local aux masks require a separately validated DDP normalization protocol"
        )
    if args.fit_statistics and world != 1:
        raise ValueError("fit statistics in one CUDA process before launching DDP")
    seed = int(training["seed"])
    seed_all(seed + rank)
    train, valid, data_hash, normalization_data_hash, data_relationship, close_data = _open_data(raw["data"])
    del valid  # Validation is opened for identity only; this command never opens test.
    static_external_dataset: StaticClipDataset | None = None
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
            geometry_block_indices=model_config.get("geometry_block_indices"),
        ).to(device)
        stages = _stages(training)
        histories = tuple(int(value) for value in training.get("history_order", (4, 8)))
        if not histories or any(value < 1 for value in histories):
            raise ValueError("history_order must contain positive H values")
        schedule_contract = {
            "schema_version": "molvid.frame_joint.schedule.v2",
            "execution_schema": EXECUTION_SCHEMA,
            "execution_seed": seed,
            "auxiliary_cadence": 8,
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
        configured_total_updates = sum(int(stage["updates"]) for stage in stages)
        total_updates = configured_total_updates
        if args.max_steps is not None:
            total_updates = min(total_updates, int(args.max_steps))
        initial_stage = stages[0]
        resume_payload = None
        parent_preview = None
        if args.resume:
            resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            if experiment_parent is not None:
                resumed_parent = resume_payload.get("contracts", {}).get(
                    "continuation_parent"
                )
                if not isinstance(resumed_parent, Mapping) or resumed_parent.get(
                    "sha256"
                ) != experiment_parent["sha256"]:
                    raise ValueError("paired resume does not descend from the frozen parent")
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
            # Explicit terms override inherited terms independently. In
            # particular endpoint=0 must not disable clean/near bond losses.
            inherited = dict(parent_preview["contracts"]["loss"])
            overrides = dict(loss_mapping)
            loss_mapping = {**inherited, **overrides}
            print(json.dumps({"loss_resolution": "parent_plus_explicit_terms",
                              "overrides": overrides, "resolved": loss_mapping}, sort_keys=True))
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
            "physical", "streamed_atom_frames_threshold",
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
        static_raw = dict(raw.get("static_decoder_auxiliary", {}))
        static_enabled = bool(static_raw.get("enabled", False))
        static_source_dataset: Any | None = None
        static_schedule: StaticBatchSchedule | None = None
        static_rng: StaticRNGStreams | None = None
        static_loss_config: StaticDecoderLossConfig | None = None
        static_perturbation: StaticPerturbationConfig | None = None
        static_contract: dict[str, Any] | None = None
        static_gamma_report: Mapping[str, Any] | None = None
        static_entries: Sequence[Any] = ()
        if static_enabled:
            if world != 1:
                raise ValueError(
                    "formal static auxiliary currently requires one CUDA rank"
                )
            if args.continuation_parent:
                raise ValueError(
                    "a changed static objective requires model-only warm start, not exact continuation"
                )
            if args.resume:
                resumed_static = resume_payload["contracts"].get(
                    "static_decoder_auxiliary"
                )
                if not isinstance(resumed_static, Mapping):
                    raise ValueError("static resume checkpoint lacks its contract")
            elif not args.warm_start or not args.warm_start_sha256:
                raise ValueError(
                    "formal static auxiliary requires an explicit released-A1 model-only warm start"
                )
            if sampled_config.physical is None:
                raise ValueError("static clash requires the released A physical recipe")
            static_loss_config = StaticDecoderLossConfig.resolve(static_raw["loss"])
            static_perturbation = StaticPerturbationConfig.resolve(
                static_raw["perturbation"]
            )
            if not math.isclose(
                static_loss_config.bond,
                loss_config.clean_bond,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("static bond coefficient differs from A clean_bond")
            if not math.isclose(
                static_loss_config.clash,
                sampled_config.physical.clash_weight,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("static clash coefficient differs from released A clash")
            gamma_identity = static_raw["gamma_calibration"]
            gamma_path = Path(gamma_identity["path"])
            if sha256_file(gamma_path) != str(gamma_identity["sha256"]):
                raise ValueError("static gamma calibration SHA-256 differs")
            static_gamma_report = json.loads(gamma_path.read_text(encoding="utf-8"))
            if static_gamma_report.get("schema") != "molvid.static_decoder.gamma_calibration.v1":
                raise ValueError("static gamma calibration schema differs")
            if static_gamma_report.get("parent_role") != "released_A1":
                raise ValueError("formal static gamma was not calibrated from released A1")
            if static_gamma_report.get("parent_sha256") != gamma_identity["parent_sha256"]:
                raise ValueError("static gamma parent identity differs")
            if args.resume:
                resumed_parent = resume_payload["contracts"].get("continuation_parent")
                if not isinstance(resumed_parent, Mapping) or resumed_parent.get(
                    "sha256"
                ) != gamma_identity["parent_sha256"]:
                    raise ValueError("static resume does not descend from the gamma parent")
            elif args.warm_start_sha256 != gamma_identity["parent_sha256"]:
                raise ValueError("formal warm start differs from static gamma parent")
            if gamma_identity["code_commit"] != _git_commit():
                raise ValueError("formal static code differs from gamma calibration")
            identity_checks = {
                "code_commit": gamma_identity["code_commit"],
                "source_config_sha256": gamma_identity["reference_config_sha256"],
                "recipe_release_sha256": gamma_identity["recipe_release_sha256"],
                "auxiliary_recipe_sha256": gamma_identity["auxiliary_recipe_sha256"],
                "dynamic_data_hash": data_hash,
                "normalization_source_data_hash": normalization_data_hash,
                "statistics_sha256": sha256_file(stats_path),
                "statistics_hash": statistics.hash,
                "teacher_sha256": artifact.report.source_sha256,
            }
            for name, expected in identity_checks.items():
                if static_gamma_report.get(name) != expected:
                    raise ValueError(f"static gamma identity differs at {name}")
            contract_checks = {
                "dynamic_model_contract": model.contract(),
                "dynamic_loss_contract": loss_config.contract(),
                "sampled_auxiliary_contract": sampled_config.contract(),
                "static_perturbation_contract": static_perturbation.contract(),
                "static_loss_contract_before_gamma": {
                    **static_loss_config.contract(),
                    "gamma": 1.0,
                },
            }
            for name, expected in contract_checks.items():
                if canonical_hash(static_gamma_report.get(name)) != canonical_hash(expected):
                    raise ValueError(f"static gamma contract differs at {name}")
            if not math.isclose(
                float(static_gamma_report["resolved_gamma"]),
                static_loss_config.gamma,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("static gamma config differs from calibration")
            if not math.isclose(
                float(static_gamma_report.get("static_clean_probability", -1.0)),
                static_perturbation.clean_probability,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("static gamma clean mixture differs from training")
            if int(static_gamma_report.get("dynamic_sampled_active_count", 0)) < 1:
                raise ValueError("static gamma denominator omitted A actual-source clash")
            if static_raw["source"] == "internal_md":
                entries = internal_md_static_pool(
                    train,
                    min_source_atoms=int(static_raw["min_source_atoms"]),
                    max_source_atoms=int(static_raw["max_source_atoms"]),
                )
                static_source_dataset = train
            else:
                external = static_raw["external"]
                store = Path(external["store"])
                if sha256_file(store / "index.txt") != external["store_index_sha256"]:
                    raise ValueError("external static store index SHA-256 differs")
                if sha256_file(store / "data.bin") != external["store_data_sha256"]:
                    raise ValueError("external static store payload SHA-256 differs")
                selection_report = Path(external["selection_report"])
                if sha256_file(selection_report) != external["selection_report_sha256"]:
                    raise ValueError("external static selection report SHA-256 differs")
                eligibility_path = Path(external["eligibility_manifest"])
                if sha256_file(eligibility_path) != external["eligibility_sha256"]:
                    raise ValueError("external static eligibility SHA-256 differs")
                eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
                required_eligibility = {
                    "schema": "molvid.static_decoder.external_eligibility.v1",
                    "status": "eligible",
                    "store_index_sha256": external["store_index_sha256"],
                    "store_data_sha256": external["store_data_sha256"],
                    "source": external["source"],
                    "split": external["split"],
                    "store": str(store.resolve()),
                    "selection_report": str(selection_report.resolve()),
                    "selection_report_sha256": external["selection_report_sha256"],
                    "selected_count": int(external["selected_count"]),
                    "selected_indices_hash": external["selected_indices_hash"],
                    "selection_seed": int(external["selection_seed"]),
                    "code_commit": external["code_commit"],
                    "dynamic_data_hash": data_hash,
                    "normalization_source_data_hash": normalization_data_hash,
                    "direct_overlap_excluded": True,
                    "near_homology_policy": external["near_homology_policy"],
                    "sealed_test_trajectories_opened": False,
                }
                for name, value in required_eligibility.items():
                    if eligibility.get(name) != value:
                        raise ValueError(f"external eligibility differs at {name}")
                selected_indices = eligibility.get("selected_indices")
                if (
                    not isinstance(selected_indices, list)
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in selected_indices)
                    or len(selected_indices) != int(external["selected_count"])
                    or canonical_hash(selected_indices) != external["selected_indices_hash"]
                ):
                    raise ValueError("external eligibility selected indices differ")
                quality = eligibility.get("quality_requirements", {})
                if (
                    quality.get("minimum_atoms") != int(static_raw["min_source_atoms"])
                    or
                    quality.get("maximum_atoms") != int(static_raw["max_source_atoms"])
                    or quality.get("all_atoms_supervised_by_loss_mask") is not True
                ):
                    raise ValueError("external eligibility supervision/atom cap differs")
                if external["near_homology_policy"] == "excluded":
                    if eligibility.get("near_homology_excluded") is not True:
                        raise ValueError("external eligibility lacks near-homology exclusion")
                elif eligibility.get("near_homology_authorization") != (
                    "explicit user waiver on 2026-09-29 for this bounded pilot"
                ):
                    raise ValueError("external near-homology waiver identity differs")
                static_external_dataset = StaticClipDataset(
                    store,
                    source=external["source"],
                    split=external["split"],
                )
                entries = external_static_pool(
                    static_external_dataset,
                    selected_indices,
                )
                if any(
                    not int(static_raw["min_source_atoms"])
                    <= entry.atoms
                    <= int(static_raw["max_source_atoms"])
                    for entry in entries
                ):
                    raise ValueError("external static pool violates frozen atom bounds")
                static_source_dataset = static_external_dataset
            if len(entries) != int(static_raw["expected_pool_size"]):
                raise ValueError("static pool size differs from the frozen config")
            static_entries = entries
            static_schedule = StaticBatchSchedule(
                entries,
                seed=int(static_raw["seed"]),
                max_atoms_per_update=int(static_raw["max_atoms_per_update"]),
                max_samples_per_update=int(static_raw["max_samples_per_update"]),
            )
            if configured_total_updates % int(static_raw["cadence"]):
                raise ValueError("dynamic budget is not divisible by static cadence")
            planned_static_updates = configured_total_updates // int(static_raw["cadence"])
            static_exposure_plan = static_schedule.preview_exposure(
                planned_static_updates
            )
            validate_static_exposure_plan(
                static_exposure_plan,
                static_raw["exposure_target"],
            )
            static_rng = StaticRNGStreams(
                device=device,
                seed=int(static_raw["seed"]) + 1,
            )
            static_contract = {
                "schema": "molvid.static_decoder.formal_auxiliary.v1",
                "enabled": True,
                "source": static_raw["source"],
                "cadence_dynamic_updates": int(static_raw["cadence"]),
                "perturbation": static_perturbation.contract(),
                "loss": static_loss_config.contract(),
                "gamma_calibration": {
                    **dict(gamma_identity),
                    "resolved_gamma": float(static_gamma_report["resolved_gamma"]),
                },
                "schedule": static_schedule.contract(),
                "exposure_target": dict(static_raw["exposure_target"]),
                "planned_exposure": static_exposure_plan,
                "dynamic_rng_isolation": "torch.random.fork_rng plus independent generators",
                "scheduler_advanced_by_static_updates": False,
                "teacher_history_motion_dit_updated_by_static": False,
            }
            if static_raw["source"] == "external_static":
                static_contract["external_identity"] = {
                    **dict(static_raw["external"]),
                    "store": str(Path(static_raw["external"]["store"]).resolve()),
                    "eligibility_manifest": str(
                        Path(
                            static_raw["external"]["eligibility_manifest"]
                        ).resolve()
                    ),
                }
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
            checkpoint_flow_field=bool(training.get("checkpoint_flow_field", False)),
            static_auxiliary_contract=static_contract,
        )
        static_trainer = (
            None
            if not static_enabled
            else StaticDecoderAuxiliaryTrainer(
                model,
                loss_config=static_loss_config,
                optimizer=trainer.optimizer,
                grad_clip=float(training.get("grad_clip", 1.0)),
                amp=str(training.get("precision", "fp32")) == "bf16",
                clash_loss_function=static_clash_loss_function(sampled_config.physical),
            )
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
            if static_enabled:
                static_state = rank_state.get("static_auxiliary_state")
                if not isinstance(static_state, Mapping):
                    raise ValueError("static auxiliary resume lacks rank-local state")
                static_schedule.load_state_dict(static_state.get("schedule", {}))
                static_rng.load_state_dict(static_state.get("rng", {}))
                static_trainer.successful_updates = (
                    static_schedule.successful_static_updates
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
                sampled_generator=sampled_generator,
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
                dit_block_mapping=training.get("warm_start_block_mapping"),
            )
            # Architecture construction consumes a depth-dependent amount of
            # global RNG. Model-only continuations reset that stream after the
            # audited load so C4/C8 differ only in architecture, not stochastic
            # state. Source and auxiliary noise already use dedicated generators.
            seed_all(seed + rank)
        output_root = Path(training["output_root"])
        def build_execution():
            global_batches = sampler.global_batches
            if sampler.drop_last:
                global_batches = global_batches[:len(global_batches) // world * world]
            entries = execution_schedule(
                global_batches, sampler.specs, histories=histories, seed=seed,
                epoch=sampler.epoch, cadence=8,
            )
            manifest = execution_manifest(entries)
            if rank == 0:
                atomic_write_json(output_root / f"execution_epoch_{sampler.epoch:04d}.json", manifest)
            return entries, manifest["sha256"]

        execution, execution_hash = build_execution()
        if args.resume or args.continuation_parent:
            if cursor.get("execution_schedule_hash") != execution_hash:
                raise ValueError("execution H/aux schedule differs; use model-only warm start for new objectives")
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
                    euler_steps=trainer.sampled_config.euler_steps,
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
                "static_decoder_auxiliary_contract": static_contract,
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
            if static_enabled:
                atomic_write_json(
                    output_root / "static_source_pool.json",
                    {
                        "schema": "molvid.static_decoder.source_pool.v1",
                        "contract": static_schedule.contract(),
                        "entries": [entry.contract() for entry in static_entries],
                        "test_opened": False,
                    },
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
        gradient_diagnostics_every = int(training.get("gradient_diagnostics_every", 0))
        while trainer.step < total_updates:
            update_started = time.perf_counter()
            batches = list(iter(sampler))
            if batch_index >= len(batches):
                epoch += 1
                batch_index = 0
                sampler.set_epoch(epoch)
                batches = list(iter(sampler))
                execution, execution_hash = build_execution()
            indices = batches[batch_index]
            entry = execution[batch_index * world + rank]
            if tuple(indices) != entry.indices:
                raise RuntimeError("execution schedule and sampler cursor disagree")
            io_started = time.perf_counter()
            cpu_batch = collate_clip_records([train[index] for index in indices])
            io_seconds = time.perf_counter() - io_started
            buckets = set(str(value) for value in cpu_batch.time_bucket_id)
            if len(buckets) != 1:
                raise RuntimeError("Frame Joint sampler produced a mixed time-bucket batch")
            bucket = next(iter(buckets))
            fixed_history = bucket.startswith("fixed_history_dt_")
            history = entry.history_frames
            if not fixed_history:
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
                resolved_loss = dict(loss_mapping)
                resolved_loss.pop("inherit_parent", None)
                resolved_loss.pop("calibration_target_ratio", None)
                resolved_loss["generated_bond"] = float(coefficient)
                trainer.loss_config = JointLossConfig.resolve(resolved_loss)
                calibration_done = True
                if rank == 0:
                    atomic_write_json(output_root / "bond_calibration.json", diagnostics)
                    resolved_config["resolved"]["loss_contract"] = trainer.loss_config.contract()
                    atomic_write_json(output_root / "resolved_config.json", resolved_config)
            trainer.set_learning_rates(_scheduled_rates(stage, stage_position))
            prepare_started = time.perf_counter()
            prepared = prepare_frame_joint_batch(
                model.target_teacher,
                cpu_batch,
                device=device,
                normalizer=model,
                history_frames=history,
            )
            torch.cuda.synchronize(device)
            prepare_seconds = time.perf_counter() - prepare_started
            train_started = time.perf_counter()
            last = trainer.train_step(
                prepared,
                generator=generator,
                sampled_generator=sampled_generator,
                auxiliary_active=entry.auxiliary,
                diagnose_gradients=bool(gradient_diagnostics_every and (trainer.step + 1) % gradient_diagnostics_every == 0),
            )
            last.update({
                "io_seconds": io_seconds,
                "prepare_seconds": prepare_seconds,
                "train_seconds": time.perf_counter() - train_started,
                "update_seconds": time.perf_counter() - update_started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "execution_schedule_hash": execution_hash,
                "scheduled_auxiliary": entry.auxiliary,
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
            if (
                static_enabled
                and trainer.successful_updates % int(static_raw["cadence"]) == 0
            ):
                if trainer.stage != "continuation":
                    raise RuntimeError("formal static auxiliary is valid only in continuation")
                source_entries = static_schedule.next_entries()
                static_io_started = time.perf_counter()
                cpu_static_batch = collate_clip_records(
                    materialize_static_records(static_source_dataset, source_entries)
                )
                static_io_seconds = time.perf_counter() - static_io_started
                global_cpu_rng = torch.random.get_rng_state().clone()
                global_cuda_rng = torch.cuda.get_rng_state(device).clone()
                static_started = time.perf_counter()
                with static_rng.model_random_context():
                    prepared_static = prepare_static_decoder_batch(
                        model.target_teacher,
                        cpu_static_batch,
                        device=device,
                        generator=static_rng.perturbation,
                        perturbation=static_perturbation,
                    )
                    static_metrics = static_trainer.train_step(prepared_static)
                if not torch.equal(torch.random.get_rng_state(), global_cpu_rng):
                    raise RuntimeError("static update changed the dynamic CPU RNG stream")
                if not torch.equal(torch.cuda.get_rng_state(device), global_cuda_rng):
                    raise RuntimeError("static update changed the dynamic CUDA RNG stream")
                static_schedule.record_success(source_entries)
                static_metrics.update(
                    {
                        "dynamic_step": trainer.step,
                        "source": static_raw["source"],
                        "source_pool_hash": static_schedule.pool_hash,
                        "source_sample_ids": [
                            entry.source_sample_id for entry in source_entries
                        ],
                        "source_dataset_indices": [
                            entry.dataset_index for entry in source_entries
                        ],
                        "source_frame_indices": [
                            entry.frame_index for entry in source_entries
                        ],
                        "static_io_seconds": static_io_seconds,
                        "static_update_seconds": time.perf_counter() - static_started,
                        "static_exposed_samples_total": static_schedule.exposed_samples,
                        "static_exposed_atoms_total": static_schedule.exposed_atoms,
                        "static_exposed_supervised_atoms_total": (
                            static_schedule.exposed_supervised_atoms
                        ),
                    }
                )
                last["static_auxiliary_applied"] = True
                last["static_successful_updates"] = (
                    static_schedule.successful_static_updates
                )
                last["update_seconds"] = time.perf_counter() - update_started
                if rank == 0:
                    with (output_root / "static_metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(static_metrics, sort_keys=True) + "\n")
            if rank == 0:
                with (output_root / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(last, sort_keys=True) + "\n")
            save_now = args.checkpoint_every and trainer.step % args.checkpoint_every == 0
            if save_now or trainer.step == total_updates or batch_index == len(batches):
                cursor = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "ordinary_batch_index": ordinary_batch_index,
                    "sampler_state": sampler.state_dict(),
                    "global_batch_schedule_hash": canonical_hash(sampler.global_batches),
                    "execution_schedule_hash": execution_hash,
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
                if static_enabled:
                    local_state["static_auxiliary_state"] = {
                        "schedule": static_schedule.state_dict(),
                        "rng": static_rng.state_dict(),
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
            if static_enabled:
                actual_static_exposure = {
                    "static_updates": static_schedule.successful_static_updates,
                    "samples": static_schedule.exposed_samples,
                    "atoms": static_schedule.exposed_atoms,
                    "supervised_atoms": static_schedule.exposed_supervised_atoms,
                }
                if trainer.step == configured_total_updates:
                    validate_static_exposure_plan(
                        actual_static_exposure,
                        static_raw["exposure_target"],
                    )
                atomic_write_json(
                    output_root / "static_exposure.json",
                    {
                        "schema": "molvid.static_decoder.exposure.v1",
                        **static_schedule.state_dict(),
                        "target": dict(static_raw["exposure_target"]),
                        "planned": static_exposure_plan,
                        "actual": actual_static_exposure,
                        "cadence_dynamic_updates": int(static_raw["cadence"]),
                        "test_opened": False,
                    },
                )
            print(json.dumps({"step": trainer.step, "metrics": last, "output_root": str(output_root)}, sort_keys=True))
        return 0
    finally:
        close_data()
        if static_external_dataset is not None:
            del static_external_dataset
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
