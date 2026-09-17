#!/usr/bin/env python3
"""Run the three T08 codec controls and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.codec_evaluation import evaluate_controls, model_control, write_report
from module.bond_sources import build_canonical_reference_index
from train_codec import _fractional_subset
from trainer.codec_contract import require_contract_equal
from trainer.codec_losses import compute_codec_losses
from trainer.codec_trainer import (
    CODEC_CHECKPOINT_SCHEMA,
    LEGACY_CODEC_CHECKPOINT_SCHEMA,
    PVB_MODEL_CONFIG_KEYS,
    CodecTrainConfig,
    PVBCodecModel,
    prepare_batch_then_to_device,
    _to_device,
)


def _dataset(roots: Sequence[str]) -> Any:
    stores = [ClipMMapDataset(root) for root in roots]
    if not stores:
        raise ValueError("evaluation requires data.valid_roots in the codec YAML")
    return stores[0] if len(stores) == 1 else ConcatDataset(stores)


def _model_from_yaml(
    model_config: Mapping[str, Any],
    *,
    temporal_layers: int,
    temporal_ratio: int,
) -> PVBCodecModel:
    common = {
        key: model_config[key]
        for key in PVB_MODEL_CONFIG_KEYS
        if key in model_config and key not in {"temporal_layers", "temporal_ratio"}
    }
    if "spatial_dtype" in common:
        dtype_name = str(common["spatial_dtype"]).removeprefix("torch.")
        try:
            common["spatial_dtype"] = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported model.spatial_dtype {dtype_name!r}") from exc
        if not isinstance(common["spatial_dtype"], torch.dtype):
            raise ValueError(f"model.spatial_dtype {dtype_name!r} is not a torch dtype")
    return PVBCodecModel(
        **common,
        temporal_layers=int(temporal_layers),
        temporal_ratio=int(temporal_ratio),
    )


def _predictor(model: PVBCodecModel, device: torch.device):
    def predict(batch):
        # Topology registration is CPU-only and must precede every transfer.
        moved = prepare_batch_then_to_device(model, batch, device)
        with torch.no_grad():
            output = model(moved)
        return output.x_hat.detach().cpu()

    return predict


def _loss_evaluator(
    device: torch.device,
    config: CodecTrainConfig,
    normalization: Mapping[str, Any],
    step: int,
):
    def evaluate(prediction: torch.Tensor, batch: Any) -> dict[str, float]:
        bucket_ids = tuple(str(item) for item in batch.time_bucket_id)
        if not bucket_ids or len(set(bucket_ids)) != 1:
            raise ValueError("validation loss requires homogeneous time buckets")
        moved = _to_device(batch, device)
        coordinates = prediction.to(device)
        losses = compute_codec_losses(
            coordinates,
            moved,
            weights=config.weights_at(step),
            normalization=normalization,
        )
        losses["total"] = losses["total"] * config.bucket(bucket_ids[0]).weight
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    return evaluate


def _load_control(
    name: str,
    *,
    model_config: Mapping[str, Any],
    temporal_layers: int,
    temporal_ratio: int,
    checkpoint: Path | None,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> tuple[PVBCodecModel, dict[str, Any] | None]:
    yaml_model = _model_from_yaml(
        model_config,
        temporal_layers=temporal_layers,
        temporal_ratio=temporal_ratio,
    )
    if checkpoint is None:
        return yaml_model, None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} checkpoint is not a mapping: {checkpoint}")
    schema = str(payload.get("schema_version", ""))
    if schema == CODEC_CHECKPOINT_SCHEMA:
        required = {"model_contract", "model_state", "config", "normalization_stats", "optimizer_contract"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"{name} checkpoint is missing fields {sorted(missing)}")
        require_contract_equal(
            payload["model_contract"],
            yaml_model.model_contract(),
            label=f"{name} YAML/control model contract",
        )
        model = PVBCodecModel.from_model_contract(payload["model_contract"])
    elif schema == LEGACY_CODEC_CHECKPOINT_SCHEMA:
        if not allow_legacy or legacy_bond_mode is None:
            raise ValueError(
                f"{name} is legacy checkpoint v1 without a model/graph contract; "
                "pass --allow-legacy-checkpoint and explicit --legacy-bond-mode"
            )
        actual_mode = yaml_model.frame_encoder.bond_construction_mode
        if actual_mode != legacy_bond_mode:
            raise ValueError(
                f"{name} legacy graph mode {legacy_bond_mode!r} conflicts with YAML "
                f"mode {actual_mode!r}"
            )
        model = yaml_model
    else:
        raise ValueError(
            f"{name} checkpoint schema {schema!r} is unsupported; expected "
            f"{CODEC_CHECKPOINT_SCHEMA!r}"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    return model, dict(payload)


def _model_bond_mode(model_config: Mapping[str, Any]) -> str:
    value = model_config.get("bond_construction", {})
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("mode", "topology"))
    return "topology"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/codec.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--json", type=Path, default=Path("eval/codec_eval.json"))
    parser.add_argument("--markdown", type=Path, default=Path("eval/codec_eval.md"))
    parser.add_argument("--ratio1-no-temporal-checkpoint", type=Path, default=None)
    parser.add_argument("--ratio1-temporal-checkpoint", type=Path, default=None)
    parser.add_argument("--ratio4-temporal-checkpoint", type=Path, default=None)
    parser.add_argument("--valid-root", action="append", default=None)
    parser.add_argument("--subset-fraction", type=float, default=None, help="deterministically evaluate on this fraction of the validation dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bond-mode", choices=("topology", "distance_only"), default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--legacy-bond-mode", choices=("topology", "distance_only"), default=None)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("codec YAML root must be a mapping")
    if args.device is not None:
        raw.setdefault("training", {})["device"] = args.device
    if args.valid_root is not None:
        raw.setdefault("data", {})["valid_roots"] = args.valid_root
    if args.seed is not None:
        raw.setdefault("data", {})["seed"] = args.seed
    model_config = raw.get("model", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("model config must be a mapping")
    configured_backbone = str(
        model_config.get("spatial_backbone", "torchmd_et")
    ).lower()
    configured_mode = _model_bond_mode(model_config)
    if args.bond_mode is not None and args.bond_mode != configured_mode:
        raise ValueError(
            f"--bond-mode={args.bond_mode!r} conflicts with YAML bond_construction.mode={configured_mode!r}"
        )
    requested_mode = args.bond_mode or configured_mode
    if configured_backbone == "visnet_radius":
        if args.bond_mode not in (None, "topology"):
            raise ValueError(
                "visnet_radius does not accept an external bond-construction mode"
            )
        requested_mode = "native_radius"
    if requested_mode not in {"topology", "distance_only", "native_radius"}:
        raise ValueError(f"unsupported bond construction mode: {requested_mode!r}")
    if args.legacy_bond_mode is not None and args.legacy_bond_mode != requested_mode:
        raise ValueError("--legacy-bond-mode conflicts with the configured bond mode")

    runtime = dict(raw)
    train_config = CodecTrainConfig.from_mapping(runtime)
    selected = args.device or train_config.device
    if str(selected).lower() == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "EVAL_HARD_FAIL: training.device='auto' requires CUDA; "
                "evaluation has no CPU fallback"
            )
        selected = "cuda"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EVAL_HARD_FAIL: requested CUDA device is unavailable")
    if device.type != "cuda" and str(model_config.get("neighbor_backend", "cuda_radius")) == "cuda_radius":
        raise RuntimeError(
            "EVAL_HARD_FAIL: cuda_radius production evaluation requires CUDA; "
            "select the explicit dense_test backend only for CPU tests"
        )
    seed = int(raw.get("data", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_config = raw.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data config must be a mapping")
    dataset = _dataset(data_config.get("valid_roots", []))
    dataset, subset_info = _fractional_subset(dataset, args.subset_fraction, seed + 1)
    loader = make_clip_dataloader(
        dataset,
        max_tokens=int(data_config.get("max_tokens", 4096)),
        collate_fn=collate_clip_records,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        persistent_workers=bool(data_config.get("persistent_workers", False)),
        prefetch_factor=int(data_config.get("prefetch_factor", 2)),
        trusted_store_fast_path=bool(data_config.get("trusted_store_fast_path", False)),
        strict_record_validation=bool(data_config.get("strict_record_validation", True)),
        oversize_policy=str(data_config.get("oversize_policy", "error")),
        seed=int(data_config.get("seed", 0)),
        shuffle=False,
        replacement=False,
    )

    references = None
    if requested_mode == "distance_only":
        train_roots = data_config.get("train_roots", [])
        if not train_roots:
            raise ValueError(
                "distance-only evaluation requires data.train_roots to validate the "
                "checkpoint's frozen training reference manifest"
            )
        references = build_canonical_reference_index(
            _dataset(train_roots), source_split="train"
        )

    controls = []
    checkpoint_paths = {
        "ratio1_no_temporal": args.ratio1_no_temporal_checkpoint,
        "ratio1_temporal": args.ratio1_temporal_checkpoint,
        "ratio4_temporal": args.ratio4_temporal_checkpoint,
    }
    for name, ratio, layers, temporal in (
        ("ratio1_no_temporal", 1, 0, False),
        ("ratio1_temporal", 1, 1, True),
        ("ratio4_temporal", 4, 1, True),
    ):
        model, payload = _load_control(
            name,
            model_config=model_config,
            temporal_layers=layers,
            temporal_ratio=ratio,
            checkpoint=checkpoint_paths[name],
            allow_legacy=args.allow_legacy_checkpoint,
            legacy_bond_mode=args.legacy_bond_mode,
        )
        model = model.to(device).eval()
        if requested_mode == "distance_only":
            model.prepare_distance_bonds(references, device=device)
            if payload is not None and str(payload.get("schema_version")) == CODEC_CHECKPOINT_SCHEMA:
                require_contract_equal(
                    payload.get("distance_reference_contract"),
                    model.distance_reference_contract(),
                    label=f"{name} distance reference contract",
                )
        loss_evaluator = None
        if payload is not None:
            checkpoint_config = CodecTrainConfig.from_mapping(payload["config"])
            loss_evaluator = _loss_evaluator(
                device,
                checkpoint_config,
                payload.get("normalization_stats", {}),
                int(payload.get("step", checkpoint_config.max_steps)),
            )
        controls.append(
            model_control(
                name,
                _predictor(model, device),
                ratio=ratio,
                temporal=temporal,
                loss_evaluator=loss_evaluator,
            )
        )

    report = evaluate_controls(controls, loader, max_batches=args.max_batches, device=device)
    report["evaluation_data"] = {
        "subset": subset_info,
        "max_batches": args.max_batches,
        "bond_construction_mode": requested_mode,
        "distance_reference_policy": (
            "canonical references selected from train split only; validation coordinates "
            "are never used for graph construction"
            if requested_mode == "distance_only" else None
        ),
    }
    write_report(report, args.json, args.markdown)
    print(f"wrote {args.json} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
