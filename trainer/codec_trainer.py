"""Isolated trainer and end-to-end model wrapper for the PVB clip codec."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from data.clip_dataset import ClipBatch, TASK_NAMES
from module.coordinate_decoder import CodecLatent, JointMultiFrameDecoder, LatentConditionedSpatialRefiner
from module.multiframe_codec import PVBFrameEncoder
from module.state_detail_codec_v2 import (
    CenteredCoordinateVectorStem,
    ORIGIN_RULE,
    RATIO_FOR_MODE,
    STATE_DETAIL_MODES,
    MatchedPoolingCodecV2,
    StaticTopologyMetadata,
    StateDetailCodecV2,
    center_coordinates,
)
from module.temporal_codec import CausalTemporalEncoder

from .codec_contract import json_safe, require_contract_equal
from .codec_losses import (
    BucketNormalization,
    CodecLossWeights,
    compute_codec_losses,
    fit_time_bucket_normalization,
)


CODEC_CONFIG_SCHEMA = "pvb.codec.config.v1"
CODEC_CHECKPOINT_SCHEMA = "pvb.codec.checkpoint.v2"
LEGACY_CODEC_CHECKPOINT_SCHEMA = "pvb.codec.checkpoint.v1"
CODEC_MODEL_CONTRACT_SCHEMA = "pvb.codec.model_contract.v1"
CODEC_MODEL_CONTRACT_SCHEMA_V2 = "pvb.codec.model_contract.v2"
CODEC_MODEL_CONTRACT_SCHEMA_V3 = "pvb.codec.model_contract.v3"
CODEC_MODEL_CONTRACT_SCHEMA_V4 = "pvb.codec.model_contract.v4"
CODEC_DISTANCE_REFERENCE_SCHEMA = "pvb.codec.distance_reference.v1"
PVB_MODEL_CONFIG_KEYS = (
    "hidden_channels",
    "spatial_layers",
    "spatial_backbone",
    "temporal_layers",
    "temporal_ratio",
    "temporal_codec_mode",
    "num_rbf",
    "num_heads",
    "lmax",
    "vertex",
    "trainable_rbf",
    "vecnorm_type",
    "vertex_type",
    "rbf_type",
    "trainable_vecnorm",
    "cutoff_lower",
    "cutoff_upper",
    "max_num_neighbors",
    "neighbor_backend",
    "bond_construction",
    "spatial_execution",
    "use_spatial_refiner",
    "time_scale_ps",
    "topology_cache_capacity",
    "topology_device_cache_capacity",
    "distance_bond_min",
    "distance_bond_max",
    "distance_bond_max_num_neighbors",
    "distance_bond_cache_capacity",
    "spatial_dtype",
    "frame_encoder_checkpoint",
    "freeze_frame_encoder",
    "frame_encoder_source_hash",
    "coordinate_stem",
)


@dataclass(frozen=True)
class TimeBucketSpec:
    bucket_id: str
    center_ps: float
    tolerance_ps: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.bucket_id):
            raise ValueError("time bucket id must be non-empty")
        if not math.isfinite(float(self.center_ps)) or float(self.center_ps) < 0:
            raise ValueError("time bucket center must be finite and non-negative")
        if not math.isfinite(float(self.tolerance_ps)) or float(self.tolerance_ps) < 0:
            raise ValueError("time bucket tolerance must be finite and non-negative")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0:
            raise ValueError("time bucket weight must be finite and non-negative")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "id": str(self.bucket_id),
            "center_ps": float(self.center_ps),
            "tolerance_ps": float(self.tolerance_ps),
            "weight": float(self.weight),
        }


def validate_codec_config(config: Mapping[str, Any]) -> tuple[TimeBucketSpec, ...]:
    """Validate the explicit physical-time and normalization configuration."""

    if str(config.get("schema_version", "")) != CODEC_CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported codec config schema; expected {CODEC_CONFIG_SCHEMA!r}"
        )
    time_config = config.get("time")
    if not isinstance(time_config, Mapping):
        raise ValueError("codec config requires a time mapping")
    if str(time_config.get("unit", "")) != "ps":
        raise ValueError("codec time.unit must be the canonical 'ps'")
    continuous_scale = float(time_config.get("continuous_scale_ps", 0.0))
    if not math.isfinite(continuous_scale) or continuous_scale <= 0:
        raise ValueError("time.continuous_scale_ps must be finite and positive")
    buckets = time_config.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("codec config requires at least one time bucket")
    specs = tuple(
        TimeBucketSpec(
            bucket_id=str(item.get("id", "")),
            center_ps=float(item.get("center_ps", 0.0)),
            tolerance_ps=float(item.get("tolerance_ps", 0.0)),
            weight=float(item.get("weight", 1.0)),
        )
        for item in buckets
        if isinstance(item, Mapping)
    )
    if len(specs) != len(buckets) or len({item.bucket_id for item in specs}) != len(specs):
        raise ValueError("time bucket entries must be mappings with unique ids")
    normalization = time_config.get("normalization", {})
    if not isinstance(normalization, Mapping):
        raise ValueError("time.normalization must be a mapping")
    min_count = int(normalization.get("min_count", 1))
    epsilon = float(normalization.get("epsilon", 0.0))
    if min_count < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization min_count must be positive and epsilon must be positive")
    return specs


@dataclass
class CodecTrainConfig:
    """Validated runtime subset of ``config/codec.yaml``."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1
    grad_clip: Optional[float] = 1.0
    warmup_steps: int = 0
    device: str = "cpu"
    loss_schedule: tuple[tuple[int, CodecLossWeights], ...] = ((0, CodecLossWeights()),)
    bucket_specs: tuple[TimeBucketSpec, ...] = (
        TimeBucketSpec("static", 0.0, 0.0, 1.0),
        TimeBucketSpec("dt_100ps", 100.0, 1.0, 1.0),
    )
    continuous_scale_ps: float = 100.0
    normalization_min_count: int = 32
    normalization_epsilon: float = 1e-6
    precision: str = "fp32"

    def __post_init__(self) -> None:
        self.loss_schedule = tuple(
            (int(start), weights if isinstance(weights, CodecLossWeights) else CodecLossWeights.from_mapping(weights))
            for start, weights in self.loss_schedule
        )
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0:
            raise ValueError("weight decay must be finite and non-negative")
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if self.grad_clip is not None and (not math.isfinite(float(self.grad_clip)) or float(self.grad_clip) <= 0):
            raise ValueError("grad_clip must be positive when supplied")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not self.loss_schedule or self.loss_schedule[0][0] != 0:
            raise ValueError("loss schedule must start at step zero")
        if any(start < 0 for start, _ in self.loss_schedule):
            raise ValueError("loss schedule steps must be non-negative")
        if any(self.loss_schedule[i][0] >= self.loss_schedule[i + 1][0] for i in range(len(self.loss_schedule) - 1)):
            raise ValueError("loss schedule steps must be strictly increasing")
        if not math.isfinite(float(self.continuous_scale_ps)) or float(self.continuous_scale_ps) <= 0:
            raise ValueError("continuous_scale_ps must be finite and positive")
        if int(self.normalization_min_count) < 1 or float(self.normalization_epsilon) <= 0:
            raise ValueError("normalization guards must be positive")
        self.precision = str(self.precision).lower()
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CodecTrainConfig":
        specs = validate_codec_config(config)
        time_config = config["time"]
        training = config.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("training config must be a mapping")
        base_weights = CodecLossWeights.from_mapping(training.get("loss_weights"))
        stages = training.get("loss_schedule")
        if stages is None:
            schedule = ((0, base_weights),)
        else:
            if not isinstance(stages, list) or not stages:
                raise ValueError("training.loss_schedule must be a non-empty list")
            parsed = []
            for stage in stages:
                if not isinstance(stage, Mapping):
                    raise ValueError("each loss schedule stage must be a mapping")
                stage_weights = stage.get("weights", stage)
                parsed.append((int(stage.get("start_step", 0)), CodecLossWeights.from_mapping(stage_weights)))
            schedule = tuple(sorted(parsed, key=lambda item: item[0]))
        normalization = time_config.get("normalization", {})
        grad_clip = training.get("grad_clip", 1.0)
        return cls(
            lr=float(training.get("lr", 1e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_steps=int(training.get("max_steps", 1)),
            grad_clip=None if grad_clip is None else float(grad_clip),
            warmup_steps=int(training.get("warmup_steps", 0)),
            device=str(training.get("device", "cpu")),
            loss_schedule=schedule,
            bucket_specs=specs,
            continuous_scale_ps=float(time_config["continuous_scale_ps"]),
            normalization_min_count=int(normalization.get("min_count", 32)),
            normalization_epsilon=float(normalization.get("epsilon", 1e-6)),
            precision=str(training.get("precision", "fp32")).lower(),
        )

    def weights_at(self, step: int) -> CodecLossWeights:
        selected = self.loss_schedule[0][1]
        for start, weights in self.loss_schedule:
            if int(step) >= start:
                selected = weights
            else:
                break
        return selected

    def bucket(self, bucket_id: str) -> TimeBucketSpec:
        for spec in self.bucket_specs:
            if spec.bucket_id == str(bucket_id):
                return spec
        raise ValueError(f"unconfigured time bucket: {bucket_id!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CODEC_CONFIG_SCHEMA,
            "time": {
                "unit": "ps",
                "continuous_scale_ps": float(self.continuous_scale_ps),
                "buckets": [spec.as_dict() for spec in self.bucket_specs],
                "normalization": {
                    "min_count": int(self.normalization_min_count),
                    "epsilon": float(self.normalization_epsilon),
                },
            },
            "training": {
                "lr": float(self.lr),
                "weight_decay": float(self.weight_decay),
                "max_steps": int(self.max_steps),
                "grad_clip": self.grad_clip,
                "warmup_steps": int(self.warmup_steps),
                "device": self.device,
                "precision": self.precision,
                "loss_schedule": [
                    {"start_step": start, "weights": weights.as_dict()}
                    for start, weights in self.loss_schedule
                ],
            },
        }


class PVBCodecModel(nn.Module):
    """Compose the existing PVB spatial backbone with T05/T06 codec modules."""

    def __init__(
        self,
        hidden_channels: int = 128,
        spatial_layers: int = 2,
        spatial_backbone: str = "torchmd_et",
        temporal_layers: int = 1,
        temporal_ratio: int | None = None,
        temporal_codec_mode: str = "legacy",
        num_rbf: int = 50,
        num_heads: int = 8,
        lmax: int = 1,
        vertex: bool = True,
        trainable_rbf: bool = False,
        vecnorm_type: str | None = None,
        vertex_type: str | None = None,
        rbf_type: str | None = None,
        trainable_vecnorm: bool = False,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        neighbor_backend: str = "cuda_radius",
        bond_construction: Mapping[str, Any] | str | None = None,
        use_spatial_refiner: bool = False,
        time_scale_ps: float = 100.0,
        topology_cache_capacity: int = 4096,
        topology_device_cache_capacity: int = 4096,
        distance_bond_min: float = 0.5,
        distance_bond_max: float = 2.2,
        distance_bond_max_num_neighbors: int = 64,
        distance_bond_cache_capacity: int = 4096,
        spatial_execution: Mapping[str, Any] | str | None = None,
        spatial_dtype: torch.dtype = torch.float32,
        frame_encoder_checkpoint: str | Path | None = None,
        freeze_frame_encoder: bool = False,
        frame_encoder_source_hash: str | None = None,
        coordinate_stem: str | None = None,
    ) -> None:
        super().__init__()
        temporal_mode = str(temporal_codec_mode).lower()
        if temporal_mode != "legacy" and temporal_mode not in STATE_DETAIL_MODES:
            raise ValueError(
                f"unsupported temporal_codec_mode {temporal_codec_mode!r}; expected "
                f"'legacy' or one of {STATE_DETAIL_MODES}"
            )
        if coordinate_stem is None:
            resolved_coordinate_stem = "none" if temporal_mode == "legacy" else "centered_vector"
        else:
            resolved_coordinate_stem = str(coordinate_stem).lower()
        if resolved_coordinate_stem not in {"none", "centered_vector"}:
            raise ValueError(
                "coordinate_stem must be 'none' or 'centered_vector'"
            )
        if temporal_mode == "legacy" and resolved_coordinate_stem != "none":
            raise ValueError("legacy codec does not support coordinate_stem")
        if temporal_ratio is None:
            resolved_temporal_ratio = (
                RATIO_FOR_MODE[temporal_mode] if temporal_mode != "legacy" else 1
            )
        else:
            resolved_temporal_ratio = int(temporal_ratio)
        if temporal_mode != "legacy":
            expected_ratio = RATIO_FOR_MODE[temporal_mode]
            if resolved_temporal_ratio != expected_ratio:
                raise ValueError(
                    f"temporal_codec_mode={temporal_mode!r} requires "
                    f"temporal_ratio={expected_ratio}, got {resolved_temporal_ratio}"
                )
            if int(temporal_layers) != 1:
                raise ValueError(
                    "state/detail codec controls require temporal_layers=1; "
                    "cross-block temporal stacks are not part of this phase"
                )
            if use_spatial_refiner:
                raise ValueError(
                    "state/detail codec controls require use_spatial_refiner=false"
                )
        if int(temporal_layers) < 0:
            raise ValueError("temporal_layers must be non-negative")
        if not isinstance(spatial_dtype, torch.dtype):
            raise TypeError("spatial_dtype must be a torch.dtype")
        if spatial_execution is None:
            spatial_execution_config: dict[str, Any] = {"mode": "full"}
        elif isinstance(spatial_execution, str):
            spatial_execution_config = {"mode": spatial_execution}
        elif isinstance(spatial_execution, Mapping):
            spatial_execution_config = dict(spatial_execution)
        else:
            raise TypeError("spatial_execution must be a mode string or mapping")
        spatial_mode = str(spatial_execution_config.get("mode", "full"))
        if spatial_mode != "full":
            raise NotImplementedError(
                "only spatial_execution.mode='full' is implemented in Phase A/B; "
                "halo_subgraph is the optional Phase-C stretch"
            )
        if isinstance(bond_construction, str):
            bond_config = {"mode": bond_construction}
        elif isinstance(bond_construction, Mapping):
            bond_config = dict(bond_construction)
        elif bond_construction is None:
            bond_config = {"mode": "topology"}
        else:
            raise TypeError("bond_construction must be a mode string or mapping")
        distance_bond_min = float(
            bond_config.get("min_distance_angstrom", distance_bond_min)
        )
        distance_bond_max = float(
            bond_config.get("max_distance_angstrom", distance_bond_max)
        )
        distance_bond_max_num_neighbors = int(
            bond_config.get("max_num_neighbors", distance_bond_max_num_neighbors)
        )
        self.frame_encoder = PVBFrameEncoder(
            hidden_channels=hidden_channels,
            num_layers=spatial_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            spatial_backbone=spatial_backbone,
            lmax=lmax,
            vertex=vertex,
            trainable_rbf=trainable_rbf,
            vecnorm_type=vecnorm_type,
            vertex_type=vertex_type,
            rbf_type=rbf_type,
            trainable_vecnorm=trainable_vecnorm,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            neighbor_backend=neighbor_backend,
            bond_construction=bond_config,
            topology_cache_capacity=topology_cache_capacity,
            topology_device_cache_capacity=topology_device_cache_capacity,
            distance_bond_min=distance_bond_min,
            distance_bond_max=distance_bond_max,
            distance_bond_max_num_neighbors=distance_bond_max_num_neighbors,
            distance_bond_cache_capacity=distance_bond_cache_capacity,
            dtype=spatial_dtype,
        )
        self.freeze_frame_encoder = bool(freeze_frame_encoder)
        self.frame_encoder_source_hash = (
            None
            if frame_encoder_source_hash in (None, "")
            else str(frame_encoder_source_hash)
        )
        self.frame_encoder_checkpoint_report = None
        if frame_encoder_checkpoint is not None:
            checkpoint_path = Path(frame_encoder_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"frame encoder checkpoint does not exist: {checkpoint_path}"
                )
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if self.frame_encoder_source_hash is not None and digest != self.frame_encoder_source_hash:
                raise ValueError(
                    "frame encoder checkpoint hash disagrees with frame_encoder_source_hash"
                )
            self.frame_encoder_source_hash = digest
            self.frame_encoder_checkpoint_report = self.frame_encoder.load_checkpoint(
                checkpoint_path
            )
            if self.frame_encoder_checkpoint_report.missing_keys:
                raise ValueError(
                    "frame encoder checkpoint is incomplete; missing keys: "
                    f"{self.frame_encoder_checkpoint_report.missing_keys[:8]}"
                )
            if self.frame_encoder_checkpoint_report.shape_mismatch:
                raise ValueError(
                    "frame encoder checkpoint contains shape mismatches: "
                    f"{self.frame_encoder_checkpoint_report.shape_mismatch[:4]}"
                )
        if self.freeze_frame_encoder:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad_(False)
            self.frame_encoder.eval()

        self.temporal_codec_mode = temporal_mode
        self.temporal_ratio = resolved_temporal_ratio
        self.coordinate_stem = resolved_coordinate_stem
        self.coordinate_vector_stem = (
            CenteredCoordinateVectorStem(hidden_channels)
            if resolved_coordinate_stem == "centered_vector"
            else None
        )
        self.temporal_encoder = None
        self.decoder = None
        self.state_detail_codec = None
        if temporal_mode == "legacy":
            self.temporal_encoder = CausalTemporalEncoder(
                hidden_channels,
                hidden_channels,
                ratio=resolved_temporal_ratio,
                num_layers=temporal_layers,
                num_heads=num_heads,
                time_scale_ps=time_scale_ps,
            )
            refiner = None
            if use_spatial_refiner:
                refiner = LatentConditionedSpatialRefiner(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels=hidden_channels,
                    num_layers=spatial_layers,
                    num_rbf=num_rbf,
                    num_heads=num_heads,
                    cutoff_lower=cutoff_lower,
                    cutoff_upper=cutoff_upper,
                    max_num_neighbors=max_num_neighbors,
                )
            self.decoder = JointMultiFrameDecoder(
                hidden_channels,
                hidden_channels,
                temporal_layers=temporal_layers,
                num_heads=num_heads,
                time_scale_ps=time_scale_ps,
                spatial_refiner=refiner,
            )
        elif temporal_mode == "ratio4_matched_pooling":
            self.state_detail_codec = MatchedPoolingCodecV2(hidden_channels)
        else:
            self.state_detail_codec = StateDetailCodecV2(
                hidden_channels,
                mode=temporal_mode,
            )
        constructor_config: dict[str, Any] = {
                "hidden_channels": int(hidden_channels),
                "spatial_layers": int(spatial_layers),
                "spatial_backbone": str(spatial_backbone),
                "temporal_layers": int(temporal_layers),
                "temporal_ratio": int(resolved_temporal_ratio),
                "num_rbf": int(num_rbf),
                "num_heads": int(num_heads),
                "lmax": int(lmax),
                "vertex": bool(vertex),
                "trainable_rbf": bool(trainable_rbf),
                "vecnorm_type": vecnorm_type,
                "cutoff_lower": float(cutoff_lower),
                "cutoff_upper": float(cutoff_upper),
                "max_num_neighbors": int(max_num_neighbors),
                "neighbor_backend": str(neighbor_backend),
                "bond_construction": bond_config,
                "use_spatial_refiner": bool(use_spatial_refiner),
                "time_scale_ps": float(time_scale_ps),
                "topology_cache_capacity": int(topology_cache_capacity),
                "topology_device_cache_capacity": int(topology_device_cache_capacity),
                "distance_bond_min": float(distance_bond_min),
                "distance_bond_max": float(distance_bond_max),
                "distance_bond_max_num_neighbors": int(distance_bond_max_num_neighbors),
                "distance_bond_cache_capacity": int(distance_bond_cache_capacity),
                "spatial_execution": spatial_execution_config,
                "spatial_dtype": str(spatial_dtype).replace("torch.", ""),
        }
        if str(spatial_backbone).lower() in {
            "visnet_v2_radius",
            "visnet_v2_bonded",
        }:
            constructor_config.update(
                {
                    "vertex_type": self.frame_encoder.vertex_type,
                    "rbf_type": self.frame_encoder.rbf_type,
                    "trainable_vecnorm": bool(
                        self.frame_encoder.trainable_vecnorm
                    ),
                }
            )
        if temporal_mode != "legacy":
            constructor_config.update(
                {
                    "temporal_codec_mode": temporal_mode,
                    "freeze_frame_encoder": bool(self.freeze_frame_encoder),
                    "frame_encoder_source_hash": self.frame_encoder_source_hash,
                }
            )
            if resolved_coordinate_stem != "none":
                constructor_config["coordinate_stem"] = resolved_coordinate_stem
        self._constructor_config = json_safe(constructor_config)
        self._distance_reference_contract: dict[str, Any] | None = None

    def model_contract(self) -> dict[str, Any]:
        """Return every constructor option that changes forward semantics."""

        constructor = json_safe(self._constructor_config)
        graph = {
            "schema_version": "pvb.codec.graph_contract.v1",
            "spatial_backbone": constructor["spatial_backbone"],
            "graph_mode": self.frame_encoder.graph_mode,
            "bond_construction": constructor["bond_construction"],
            "neighbor_backend": constructor["neighbor_backend"],
            "cutoff_lower": constructor["cutoff_lower"],
            "cutoff_upper": constructor["cutoff_upper"],
            "max_num_neighbors": constructor["max_num_neighbors"],
            "distance_bond_min": constructor["distance_bond_min"],
            "distance_bond_max": constructor["distance_bond_max"],
            "distance_bond_max_num_neighbors": constructor["distance_bond_max_num_neighbors"],
            "topology_cache_capacity": constructor["topology_cache_capacity"],
            "topology_device_cache_capacity": constructor["topology_device_cache_capacity"],
            "distance_bond_cache_capacity": constructor["distance_bond_cache_capacity"],
            "spatial_execution": constructor["spatial_execution"],
        }
        is_v2 = constructor["spatial_backbone"] in {
            "visnet_v2_radius",
            "visnet_v2_bonded",
        }
        is_state_detail = constructor.get("temporal_codec_mode", "legacy") != "legacy"
        is_repaired_state_detail = (
            is_state_detail and constructor.get("coordinate_stem", "none") != "none"
        )
        if is_v2:
            spatial_architecture = {
                "backbone": constructor["spatial_backbone"],
                "hidden_channels": constructor["hidden_channels"],
                "layers": constructor["spatial_layers"],
                "num_rbf": constructor["num_rbf"],
                "num_heads": constructor["num_heads"],
                "lmax": constructor["lmax"],
                "vertex": constructor["vertex"],
                "vertex_type": constructor["vertex_type"],
                "rbf_type": constructor["rbf_type"],
                "trainable_rbf": constructor["trainable_rbf"],
                "vecnorm_type": constructor["vecnorm_type"],
                "trainable_vecnorm": constructor["trainable_vecnorm"],
                "dtype": constructor["spatial_dtype"],
                "spatial_refiner": constructor["use_spatial_refiner"],
                "representation": "AI2BMD ViSNet ViS-MP",
                "reference_provenance": {
                    "primary_commit": "497efaa190ee6f6cbc6030710c44208a01ece52d",
                    "cross_check_commit": "79d33965a40b7fa83616a9f598a0f8619f25d939",
                },
                "internal_vector_components": 3
                if constructor["lmax"] == 1
                else 8,
                "public_vector_components": 3,
            }
        else:
            spatial_architecture = {
                "backbone": constructor["spatial_backbone"],
                "hidden_channels": constructor["hidden_channels"],
                "layers": constructor["spatial_layers"],
                "num_rbf": constructor["num_rbf"],
                "num_heads": constructor["num_heads"],
                "lmax": constructor["lmax"],
                "vertex": constructor["vertex"],
                "trainable_rbf": constructor["trainable_rbf"],
                "vecnorm_type": constructor["vecnorm_type"],
                "dtype": constructor["spatial_dtype"],
                "spatial_refiner": constructor["use_spatial_refiner"],
            }
        architecture_temporal = {
            "hidden_channels": constructor["hidden_channels"],
            "layers": constructor["temporal_layers"],
            "ratio": constructor["temporal_ratio"],
            "num_heads": constructor["num_heads"],
            "time_scale_ps": constructor["time_scale_ps"],
        }
        architecture_decoder = {
            "temporal_layers": constructor["temporal_layers"],
            "num_heads": constructor["num_heads"],
            "time_scale_ps": constructor["time_scale_ps"],
        }
        if is_state_detail:
            architecture_temporal = {
                "codec_mode": constructor["temporal_codec_mode"],
                "ratio": constructor["temporal_ratio"],
                "layers": constructor["temporal_layers"],
                "hidden_channels": constructor["hidden_channels"],
                "block_local": True,
                "cross_block_attention": False,
                "coefficient_order": (
                    []
                    if constructor["temporal_ratio"] == 1
                    else ["D01"]
                    if constructor["temporal_ratio"] == 2
                    else ["Dmid", "D01", "D23"]
                ),
                "active_feature_volume_per_atom": (
                    16 * constructor["hidden_channels"]
                    if constructor["temporal_ratio"] in {1, 2}
                    else 8 * constructor["hidden_channels"]
                ),
            }
            if is_repaired_state_detail:
                architecture_temporal["coordinate_stem"] = constructor["coordinate_stem"]
            if constructor["temporal_codec_mode"] == "ratio4_matched_pooling":
                architecture_temporal["pooling_semantics"] = "linear_two_bank_pooling"
            architecture_decoder = {
                "name": "StateDetailNoAnchorDecoder",
                "coordinate_head": "shared_framewise_equivariant",
                "origin_rule": ORIGIN_RULE,
                "per_atom_anchor": False,
                "target_coordinates": False,
                "spatial_refiner": False,
            }
            if is_repaired_state_detail:
                architecture_decoder["coordinate_stem"] = constructor["coordinate_stem"]
        return {
            "schema_version": (
                CODEC_MODEL_CONTRACT_SCHEMA_V4
                if is_repaired_state_detail
                else CODEC_MODEL_CONTRACT_SCHEMA_V3
                if is_state_detail
                else CODEC_MODEL_CONTRACT_SCHEMA_V2
                if is_v2
                else CODEC_MODEL_CONTRACT_SCHEMA
            ),
            "model_type": "trainer.codec_trainer.PVBCodecModel",
            "constructor": constructor,
            "architecture": {
                "spatial": spatial_architecture,
                "temporal": architecture_temporal,
                "decoder": architecture_decoder,
            },
            "graph": graph,
        }

    @classmethod
    def from_model_contract(cls, contract: Mapping[str, Any]) -> "PVBCodecModel":
        schema = str(contract.get("schema_version", ""))
        if schema not in {
            CODEC_MODEL_CONTRACT_SCHEMA,
            CODEC_MODEL_CONTRACT_SCHEMA_V2,
            CODEC_MODEL_CONTRACT_SCHEMA_V3,
            CODEC_MODEL_CONTRACT_SCHEMA_V4,
        }:
            raise ValueError(
                "unsupported PVB model contract; expected v1, v2, v3, or v4 schema"
            )
        if str(contract.get("model_type", "")) != "trainer.codec_trainer.PVBCodecModel":
            raise ValueError("checkpoint model contract is not for PVBCodecModel")
        constructor = contract.get("constructor")
        if not isinstance(constructor, Mapping):
            raise ValueError("PVB model contract is missing its constructor mapping")
        constructor = dict(constructor)
        if schema == CODEC_MODEL_CONTRACT_SCHEMA_V3 and str(
            constructor.get("temporal_codec_mode", "legacy")
        ) != "legacy":
            # v3 state/detail checkpoints predate the learned coordinate stem.
            # Loading them must preserve their exact no-stem semantics.
            constructor.setdefault("coordinate_stem", "none")
        if schema == CODEC_MODEL_CONTRACT_SCHEMA_V4 and str(
            constructor.get("coordinate_stem", "")
        ) != "centered_vector":
            raise ValueError("v4 state/detail contracts require coordinate_stem='centered_vector'")
        dtype_name = str(constructor.pop("spatial_dtype", "float32"))
        try:
            spatial_dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported checkpoint spatial dtype {dtype_name!r}") from exc
        if not isinstance(spatial_dtype, torch.dtype):
            raise ValueError(f"checkpoint spatial dtype {dtype_name!r} is not a torch dtype")
        constructor["spatial_dtype"] = spatial_dtype
        model = cls(**constructor)
        require_contract_equal(contract, model.model_contract(), label="model contract")
        return model

    def prepare_batch(self, batch: ClipBatch) -> None:
        self.frame_encoder.prepare_batch(batch)

    def prepare_distance_bonds(
        self,
        references: Mapping[str, Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> None:
        manifest = self.frame_encoder.prepare_distance_bonds(references, device=device)
        self._distance_reference_contract = {
            "schema_version": CODEC_DISTANCE_REFERENCE_SCHEMA,
            "policy": "canonical_reference",
            "references": sorted(manifest, key=lambda item: str(item["topology_id"])),
        }

    def distance_reference_contract(self) -> dict[str, Any] | None:
        return None if self._distance_reference_contract is None else json_safe(self._distance_reference_contract)

    @staticmethod
    def _batch_field(batch: Any, name: str) -> Any:
        if isinstance(batch, Mapping):
            if name not in batch:
                raise ValueError(f"batch is missing required field {name!r}")
            return batch[name]
        if not hasattr(batch, name):
            raise ValueError(f"batch is missing required field {name!r}")
        return getattr(batch, name)

    def _centered_batch(self, batch: ClipBatch) -> tuple[ClipBatch | Mapping[str, Any], Tensor]:
        """Center spatial inputs while retaining only one origin per sample."""

        x = torch.as_tensor(self._batch_field(batch, "x"))
        centered, origin = center_coordinates(
            x,
            frame_mask=self._batch_field(batch, "frame_mask"),
            abid=self._batch_field(batch, "abid"),
            atom_mask=self._batch_field(batch, "loss_mask"),
        )
        if isinstance(batch, ClipBatch):
            return replace(batch, x=centered), origin
        if isinstance(batch, Mapping):
            value = dict(batch)
            value["x"] = centered
            return value, origin
        raise TypeError("codec batch must be a ClipBatch or mapping")

    @staticmethod
    def _topology_metadata(batch: ClipBatch) -> StaticTopologyMetadata:
        """Return static N-axis chemistry, never the frame-expanded graph."""

        return StaticTopologyMetadata.from_batch(batch)

    def _state_detail_encode(self, batch: ClipBatch):
        if self.temporal_codec_mode == "legacy":
            raise RuntimeError("state/detail encode is unavailable for the legacy codec")
        centered_batch, origin = self._centered_batch(batch)
        encoded = self.frame_encoder(centered_batch)
        if self.coordinate_vector_stem is not None:
            coordinate_vectors = self.coordinate_vector_stem(centered_batch.x).to(
                dtype=encoded.v.dtype
            )
            encoded = replace(encoded, v=encoded.v + coordinate_vectors)
        latent = self.state_detail_codec.encode(
            encoded.h,
            encoded.v,
            time_ps=self._batch_field(batch, "time_ps"),
            frame_mask=self._batch_field(batch, "frame_mask"),
            abid=self._batch_field(batch, "abid"),
            sample_origin=origin,
            topology=self._topology_metadata(batch),
        )
        return latent, encoded

    def encode(self, batch: ClipBatch):
        """Encode a new codec batch into its structured no-anchor latent."""

        latent, _encoded = self._state_detail_encode(batch)
        return latent

    def decode(self, latent: Any):
        """Decode a structured new latent without accepting target coordinates."""

        if self.temporal_codec_mode == "legacy":
            raise RuntimeError("state/detail decode is unavailable for the legacy codec")
        return self.state_detail_codec.decode(latent)

    def _decode_state_detail(self, batch: ClipBatch):
        latent, encoded = self._state_detail_encode(batch)
        return encoded, self.state_detail_codec.decode(latent)

    def _decode_encoded(self, encoded: Any, batch: ClipBatch):
        state = self.temporal_encoder(
            encoded.h,
            encoded.v,
            batch.time_ps,
            frame_mask=batch.frame_mask,
            abid=batch.abid,
        )
        graph = encoded.graph
        topology = {
            "z": graph.z,
            "b": graph.b,
            "batch": graph.batch,
            "edge_index": graph.edge_index,
            "bond_type": graph.bond_type,
        }
        latent = CodecLatent.from_temporal_state(
            state,
            x_anchor=batch.x[0],
            topology=topology,
        )
        return self.decoder(
            latent,
            target_time_ps=batch.time_ps,
            target_mask=batch.frame_mask,
        )

    def forward_with_encoded(self, batch: ClipBatch):
        if self.temporal_codec_mode != "legacy":
            return self._decode_state_detail(batch)
        encoded = self.frame_encoder(batch)
        return encoded, self._decode_encoded(encoded, batch)

    def forward(self, batch: ClipBatch):
        if self.temporal_codec_mode != "legacy":
            return self._decode_state_detail(batch)[1]
        return self._decode_encoded(self.frame_encoder(batch), batch)

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self.freeze_frame_encoder:
            self.frame_encoder.eval()
        return result


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


class CodecTrainer:
    """Small self-contained optimizer loop with versioned checkpoint state."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[Any],
        valid_loader: Optional[Iterable[Any]] = None,
        config: CodecTrainConfig | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
        normalization_stats: Mapping[str, BucketNormalization | Mapping[str, Any]] | None = None,
        non_blocking_transfer: bool = True,
    ) -> None:
        self.config = (
            config if isinstance(config, CodecTrainConfig)
            else CodecTrainConfig.from_mapping(config) if config is not None
            else CodecTrainConfig()
        )
        selected_device = device or self.config.device
        if str(selected_device).lower() == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "training.device='auto' requires CUDA; GPU is unavailable and "
                    "the production trainer will not fall back to CPU"
                )
            selected_device = "cuda"
        if str(selected_device).lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"requested CUDA device {selected_device!r} is unavailable; "
                "the production trainer will not fall back to CPU"
            )
        self.device = torch.device(selected_device)
        if self.config.precision == "bf16" and self.device.type != "cuda":
            raise RuntimeError(
                "BF16 codec training requires CUDA; refusing a CPU precision fallback"
            )
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.normalization_stats: dict[str, BucketNormalization] = {}
        for key, value in (normalization_stats or {}).items():
            self.normalization_stats[str(key)] = value if isinstance(value, BucketNormalization) else BucketNormalization.from_dict(value)
        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._train_iterator: Optional[Iterable[Any]] = None

    def fit_normalization(self, batches: Optional[Iterable[Any]] = None) -> dict[str, BucketNormalization]:
        source = self.train_loader if batches is None else batches
        self.normalization_stats = fit_time_bucket_normalization(
            source,
            min_count=self.config.normalization_min_count,
            epsilon=self.config.normalization_epsilon,
        )
        return self.normalization_stats

    def _validate_batch_clock(self, batch: Any) -> float:
        bucket_ids = tuple(str(item) for item in batch.time_bucket_id)
        if not bucket_ids:
            raise ValueError("codec batch must contain at least one time bucket")
        delta_time = batch.delta_time_ps
        frame_mask = batch.frame_mask
        if isinstance(delta_time, Tensor) and delta_time.device.type != "cpu":
            raise RuntimeError(
                "batch clock validation must run on the CPU batch before transfer"
            )
        if isinstance(frame_mask, Tensor) and frame_mask.device.type != "cpu":
            raise RuntimeError(
                "batch frame-mask validation must run on the CPU batch before transfer"
            )
        frames = int(batch.frames) if hasattr(batch, "frames") else int(batch.x.shape[0])
        weights = []
        for sample_index, bucket_id in enumerate(bucket_ids):
            spec = self.config.bucket(bucket_id)
            weights.append(spec.weight)
            if frames > 1:
                valid_delta = delta_time[sample_index][frame_mask[sample_index, 1:]]
                if valid_delta.numel() and torch.any(
                    (valid_delta - spec.center_ps).abs() > spec.tolerance_ps
                ):
                    raise ValueError(
                        f"batch delta_time_ps does not match bucket {bucket_id!r} center/tolerance"
                    )
        return float(sum(weights) / len(weights))

    def _autocast_context(self):
        if self.config.precision == "bf16":
            if self.device.type != "cuda":
                raise RuntimeError(
                    "BF16 autocast requires CUDA; CPU fallback is not permitted"
                )
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _prepare_batch(self, batch: Any) -> None:
        prepare = getattr(self.model, "prepare_batch", None)
        if callable(prepare):
            prepare(batch)

    def _loss_for_batch(self, batch: Any) -> tuple[dict[str, Tensor], float, Any]:
        # Validate immutable clock metadata while it is still on the CPU. This
        # keeps the CUDA transfer/forward path free of Python CUDA scalar reads.
        bucket_weight = self._validate_batch_clock(batch)
        batch = prepare_batch_then_to_device(
            self.model,
            batch,
            self.device,
            non_blocking=self.non_blocking_transfer,
        )
        with self._autocast_context():
            output = self.model(batch)
        losses = compute_codec_losses(
            output,
            batch,
            weights=self.config.weights_at(self.step),
            normalization=self.normalization_stats,
        )
        losses["total"] = losses["total"] * bucket_weight
        return losses, bucket_weight, batch

    @staticmethod
    def _metrics(losses: Mapping[str, Tensor], batch: Any) -> dict[str, float]:
        metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
        host_task_ids = tuple(getattr(batch, "host_task_ids", ()))
        if host_task_ids:
            task_ids = tuple(sorted(set(int(value) for value in host_task_ids)))
        else:
            task = torch.as_tensor(batch.task, dtype=torch.long).flatten()
            if task.device.type != "cpu":
                raise RuntimeError(
                    "CUDA metric task labels require host_task_ids from CPU collation"
                )
            task_ids = tuple(sorted(set(int(value) for value in task.tolist())))
        for task_id in task_ids:
            task_name = TASK_NAMES[int(task_id)]
            metrics[f"task_{task_name}_total"] = metrics["total"]
        for bucket_id in dict.fromkeys(str(item) for item in batch.time_bucket_id):
            safe_bucket = bucket_id.replace("/", "_")
            metrics[f"bucket_{safe_bucket}_velocity"] = metrics["velocity"]
            metrics[f"bucket_{safe_bucket}_velocity_raw"] = metrics["velocity_raw"]
            metrics[f"bucket_{safe_bucket}_acceleration"] = metrics["acceleration"]
            metrics[f"bucket_{safe_bucket}_acceleration_raw"] = metrics["acceleration_raw"]
        return metrics

    def optimizer_step(self, batch: Any) -> dict[str, float]:
        self.model.train()
        next_step = self.step + 1
        if self.config.warmup_steps:
            scale = min(1.0, next_step / float(self.config.warmup_steps))
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr * scale
        self.optimizer.zero_grad(set_to_none=True)
        losses, _, moved_batch = self._loss_for_batch(batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("codec loss is NaN or Inf before backward")
        total.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("codec gradient is NaN or Inf")
        self.optimizer.step()
        self.step = next_step
        return self._metrics(losses, moved_batch)

    @torch.no_grad()
    def evaluate_batch(self, batch: Any) -> dict[str, float]:
        self.model.eval()
        losses, _, moved_batch = self._loss_for_batch(batch)
        return self._metrics(losses, moved_batch)

    def _batch_sampler(self):
        return getattr(self.train_loader, "batch_sampler", None)

    def _new_train_iterator(self) -> Iterable[Any]:
        sampler = self._batch_sampler()
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)
        iterator = iter(self.train_loader)
        for _ in range(int(self.batch_in_epoch)):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "checkpoint batch_in_epoch exceeds the current epoch schedule"
                ) from exc
        return iterator

    def run(
        self,
        max_steps: Optional[int] = None,
        *,
        log_path: str | Path | None = None,
        log_every: int = 1,
    ) -> dict[str, float]:
        """Optimize until max_steps and optionally append step metrics as JSONL."""

        target = int(max_steps if max_steps is not None else self.config.max_steps)
        if target < self.step:
            raise ValueError("max_steps cannot be less than the current checkpoint step")
        if int(log_every) < 1:
            raise ValueError("log_every must be positive")
        if len(self.train_loader) < 1:
            raise RuntimeError("training loader contains no batches")
        log_handle = None
        if log_path is not None:
            destination = Path(log_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            log_handle = destination.open("a", encoding="utf-8")
        if self._train_iterator is None:
            self._train_iterator = self._new_train_iterator()
        last: dict[str, float] = {}
        try:
            while self.step < target:
                if self._train_iterator is None:
                    self._train_iterator = self._new_train_iterator()
                try:
                    batch = next(self._train_iterator)
                except StopIteration:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self._train_iterator = self._new_train_iterator()
                    batch = next(self._train_iterator)
                last = self.optimizer_step(batch)
                self.batch_in_epoch += 1
                finished_epoch = self.batch_in_epoch >= len(self.train_loader)
                logged_epoch = self.epoch
                if finished_epoch:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self._train_iterator = None
                if log_handle is not None and (
                    self.step % int(log_every) == 0 or self.step == target
                ):
                    record = {
                        "schema_version": "pvb.codec.train.v1",
                        "split": "train",
                        "step": int(self.step),
                        "epoch": int(logged_epoch),
                        "batch_in_epoch": int(self.batch_in_epoch),
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "metrics": last,
                    }
                    log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    log_handle.flush()
        finally:
            if log_handle is not None:
                log_handle.close()
        return last

    def _model_contract(self) -> dict[str, Any]:
        method = getattr(self.model, "model_contract", None)
        if callable(method):
            contract = method()
            if not isinstance(contract, Mapping):
                raise TypeError("model_contract() must return a mapping")
            return json_safe(contract)
        signature = []
        for name, parameter in self.model.named_parameters():
            signature.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype).replace("torch.", ""),
                    "requires_grad": bool(parameter.requires_grad),
                }
            )
        return {
            "schema_version": CODEC_MODEL_CONTRACT_SCHEMA,
            "model_type": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "generic_state_signature": signature,
        }

    def _distance_reference_contract(self) -> dict[str, Any] | None:
        method = getattr(self.model, "distance_reference_contract", None)
        if not callable(method):
            return None
        value = method()
        return None if value is None else json_safe(value)

    def _optimizer_contract(self) -> dict[str, Any]:
        groups = self.optimizer.param_groups
        return {
            "algorithm": "AdamW",
            "base_lr": float(self.config.lr),
            "weight_decay": float(self.config.weight_decay),
            "warmup_steps": int(self.config.warmup_steps),
            "param_group_count": len(groups),
            "param_group_sizes": [len(group["params"]) for group in groups],
            "current_lrs": [float(group["lr"]) for group in groups],
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": CODEC_CHECKPOINT_SCHEMA,
            "step": int(self.step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "precision": self.config.precision,
            "sampler_state": (
                self._batch_sampler().state_dict()
                if self._batch_sampler() is not None
                and hasattr(self._batch_sampler(), "state_dict")
                else None
            ),
            "model_contract": self._model_contract(),
            "distance_reference_contract": self._distance_reference_contract(),
            "optimizer_contract": self._optimizer_contract(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "normalization_stats": {
                key: value.as_dict() for key, value in self.normalization_stats.items()
            },
            "config": self.config.as_dict(),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), destination)
        return destination

    @staticmethod
    def _config_for_resume(config: CodecTrainConfig) -> dict[str, Any]:
        value = config.as_dict()
        # Extending the target step count is the only training override that
        # preserves the exact optimizer/loss/sampler contract.
        value["training"] = dict(value["training"])
        value["training"].pop("max_steps", None)
        return value

    @staticmethod
    def _validate_state_mapping(model: nn.Module, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint model_state must be a mapping")
        expected = model.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected).difference(state))
            unexpected = sorted(set(state).difference(expected))
            raise ValueError(
                f"checkpoint model_state keys differ: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
        for name, value in state.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"checkpoint model_state[{name!r}] is not a tensor")
            target = expected[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"checkpoint model_state[{name!r}] shape {tuple(value.shape)} "
                    f"does not match {tuple(target.shape)}"
                )
            if value.dtype != target.dtype:
                raise ValueError(
                    f"checkpoint model_state[{name!r}] dtype {value.dtype} "
                    f"does not match {target.dtype}"
                )

    def _validate_optimizer_state(
        self,
        state: Any,
        saved_contract: Any,
        saved_config: CodecTrainConfig,
        *,
        saved_step: int,
        allow_legacy_contract: bool = False,
    ) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint optimizer_state must be a mapping")
        if saved_contract is None and allow_legacy_contract:
            # v1 has no contract field. The explicit legacy path still uses
            # the known AdamW schema and validates its serialized groups below;
            # it never guesses a graph/model configuration.
            legacy_groups = state.get("param_groups")
            if not isinstance(legacy_groups, list):
                raise ValueError("legacy checkpoint optimizer_state is missing param_groups")
            saved_contract = {
                "algorithm": "AdamW",
                "base_lr": float(saved_config.lr),
                "weight_decay": float(saved_config.weight_decay),
                "warmup_steps": int(saved_config.warmup_steps),
                "param_group_count": len(legacy_groups),
                "param_group_sizes": [len(group.get("params", ())) for group in legacy_groups],
                "current_lrs": [float(group.get("lr", float("nan"))) for group in legacy_groups],
            }
        if not isinstance(saved_contract, Mapping):
            raise ValueError("checkpoint optimizer_contract must be a mapping")
        if int(saved_step) < 0:
            raise ValueError("checkpoint step must be non-negative")
        expected_lr = float(saved_config.lr)
        if int(saved_step) > 0 and int(saved_config.warmup_steps):
            expected_lr *= min(1.0, int(saved_step) / float(saved_config.warmup_steps))
        expected_contract = {
            "algorithm": "AdamW",
            "base_lr": float(saved_config.lr),
            "weight_decay": float(saved_config.weight_decay),
            "warmup_steps": int(saved_config.warmup_steps),
            "param_group_count": len(self.optimizer.param_groups),
            "param_group_sizes": [
                len(group["params"]) for group in self.optimizer.param_groups
            ],
            "current_lrs": [expected_lr for _ in self.optimizer.param_groups],
        }
        require_contract_equal(
            expected_contract,
            saved_contract,
            label="optimizer contract",
        )
        saved_groups = state.get("param_groups")
        current_groups = self.optimizer.state_dict().get("param_groups")
        if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
            raise ValueError("optimizer checkpoint is missing param_groups")
        if len(saved_groups) != len(current_groups):
            raise ValueError("optimizer param-group count differs from current model")
        for index, (saved_group, current_group) in enumerate(zip(saved_groups, current_groups)):
            if len(saved_group.get("params", ())) != len(current_group.get("params", ())):
                raise ValueError(f"optimizer param-group {index} parameter count differs")
            if "weight_decay" not in saved_group or not math.isclose(
                float(saved_group["weight_decay"]), float(saved_config.weight_decay), rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(f"optimizer param-group {index} weight_decay disagrees with config")
            for key in ("betas", "eps", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
                if key in saved_group and key in current_group and saved_group[key] != current_group[key]:
                    raise ValueError(f"optimizer param-group {index} hyperparameter {key!r} differs")
            if "lr" not in saved_group or not math.isfinite(float(saved_group["lr"])):
                raise ValueError(f"optimizer param-group {index} has an invalid learning rate")

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        allow_legacy: bool = False,
        legacy_bond_mode: str | None = None,
    ) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("codec checkpoint must be a mapping")
        schema = str(payload.get("schema_version", ""))
        legacy = schema == LEGACY_CODEC_CHECKPOINT_SCHEMA
        if schema != CODEC_CHECKPOINT_SCHEMA and not legacy:
            raise ValueError(
                f"unsupported codec checkpoint schema; expected {CODEC_CHECKPOINT_SCHEMA!r}"
            )
        if legacy and not allow_legacy:
            raise ValueError(
                "legacy codec checkpoint v1 has no model/graph contract; pass "
                "allow_legacy=True and an explicit legacy_bond_mode to load it"
            )
        required = {
            "step",
            "epoch",
            "batch_in_epoch",
            "precision",
            "sampler_state",
            "model_state",
            "optimizer_state",
            "normalization_stats",
            "config",
        }
        if not legacy:
            required.update({"model_contract", "distance_reference_contract", "optimizer_contract"})
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"codec checkpoint is missing fields: {sorted(missing)}")

        if legacy and isinstance(self.model, PVBCodecModel):
            if legacy_bond_mode not in {"topology", "distance_only"}:
                raise ValueError(
                    "loading codec checkpoint v1 requires explicit legacy_bond_mode="
                    "'topology' or 'distance_only'"
                )
            actual_mode = self.model.frame_encoder.bond_construction_mode
            if actual_mode != legacy_bond_mode:
                raise ValueError(
                    f"legacy checkpoint graph mode {legacy_bond_mode!r} conflicts with "
                    f"current model mode {actual_mode!r}"
                )
        saved_config = CodecTrainConfig.from_mapping(payload["config"])
        if int(self.config.max_steps) < int(saved_config.max_steps):
            raise ValueError(
                "resume max_steps cannot be shorter than checkpoint max_steps; "
                "only an extension is allowed"
            )
        require_contract_equal(
            self._config_for_resume(saved_config),
            self._config_for_resume(self.config),
            label="training config",
        )
        if str(payload["precision"]).lower() != saved_config.precision:
            raise ValueError("checkpoint precision disagrees with checkpoint config")
        if str(payload["precision"]).lower() != self.config.precision:
            raise ValueError("codec checkpoint precision differs from current config")
        if not legacy:
            require_contract_equal(
                payload["model_contract"],
                self._model_contract(),
                label="model contract",
            )
            require_contract_equal(
                payload["distance_reference_contract"],
                self._distance_reference_contract(),
                label="distance reference contract",
            )
        try:
            step = int(payload["step"])
            epoch = int(payload["epoch"])
            batch_in_epoch = int(payload["batch_in_epoch"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint counters must be integers") from exc
        if step < 0 or epoch < 0 or batch_in_epoch < 0:
            raise ValueError("checkpoint counters must be non-negative")
        self._validate_state_mapping(self.model, payload["model_state"])
        self._validate_optimizer_state(
            payload["optimizer_state"],
            payload.get("optimizer_contract"),
            saved_config,
            saved_step=step,
            allow_legacy_contract=legacy,
        )
        if batch_in_epoch >= len(self.train_loader):
            raise ValueError("checkpoint batch_in_epoch must be smaller than current loader length")
        if not isinstance(payload["normalization_stats"], Mapping):
            raise ValueError("checkpoint normalization_stats must be a mapping")
        new_normalization = {
            str(key): BucketNormalization.from_dict(value)
            for key, value in payload["normalization_stats"].items()
        }
        sampler = self._batch_sampler()
        saved_sampler = payload["sampler_state"]
        if saved_sampler is not None:
            if sampler is None or not hasattr(sampler, "load_state_dict"):
                raise ValueError("checkpoint contains sampler state but loader cannot restore it")
            validator = getattr(sampler, "validate_state_dict", None)
            if callable(validator):
                validator(saved_sampler)

        # All preflight checks above run before mutation. The rollback snapshot
        # also protects against a third-party load_state_dict implementation
        # failing after partially copying tensors.
        model_snapshot = copy.deepcopy(self.model.state_dict())
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        sampler_snapshot = copy.deepcopy(sampler.state_dict()) if sampler is not None and hasattr(sampler, "state_dict") else None
        old_step, old_epoch, old_batch = self.step, self.epoch, self.batch_in_epoch
        old_normalization = copy.deepcopy(self.normalization_stats)
        old_iterator = self._train_iterator
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.step = step
            self.epoch = epoch
            self.batch_in_epoch = batch_in_epoch
            if saved_sampler is not None:
                sampler.load_state_dict(saved_sampler)
            self.normalization_stats = new_normalization
            self._train_iterator = None
        except Exception:
            self.model.load_state_dict(model_snapshot, strict=True)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.step, self.epoch, self.batch_in_epoch = old_step, old_epoch, old_batch
            if sampler is not None and sampler_snapshot is not None:
                sampler.load_state_dict(sampler_snapshot)
            self.normalization_stats = old_normalization
            self._train_iterator = old_iterator
            raise


__all__ = [
    "CODEC_CHECKPOINT_SCHEMA",
    "CODEC_CONFIG_SCHEMA",
    "CODEC_DISTANCE_REFERENCE_SCHEMA",
    "CODEC_MODEL_CONTRACT_SCHEMA",
    "CODEC_MODEL_CONTRACT_SCHEMA_V2",
    "CODEC_MODEL_CONTRACT_SCHEMA_V3",
    "CODEC_MODEL_CONTRACT_SCHEMA_V4",
    "LEGACY_CODEC_CHECKPOINT_SCHEMA",
    "PVB_MODEL_CONFIG_KEYS",
    "CodecTrainConfig",
    "CodecTrainer",
    "PVBCodecModel",
    "TimeBucketSpec",
    "prepare_batch_then_to_device",
    "validate_codec_config",
]
