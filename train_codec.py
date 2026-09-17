#!/usr/bin/env python3
"""Launch the isolated multi-frame codec trainer.

The script deliberately consumes only the versioned clip stores.  Legacy
``train.py`` and its pair-record model path are not routed through this entry
point.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, Subset

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from module.bond_sources import build_canonical_reference_index
from trainer.codec_trainer import (
    PVB_MODEL_CONFIG_KEYS,
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("codec YAML must contain a mapping at its root")
    return config


def _dataset(roots: Sequence[str]) -> Any:
    stores = [ClipMMapDataset(root) for root in roots]
    if not stores:
        raise ValueError("codec data roots are empty; set data.train_roots in the YAML")
    return stores[0] if len(stores) == 1 else ConcatDataset(stores)


def _fractional_subset(dataset: Any, fraction: float | None, seed: int) -> tuple[Any, dict[str, int | float]]:
    if fraction is None:
        return dataset, {"full_records": len(dataset), "selected_records": len(dataset), "fraction": 1.0}
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--subset-fraction must be in (0, 1]")
    selected = max(1, int(len(dataset) * fraction))
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(dataset), size=selected, replace=False)).tolist()
    return Subset(dataset, indices), {
        "full_records": len(dataset),
        "selected_records": selected,
        "fraction": fraction,
    }


def _loader(dataset: Any, data_config: dict[str, Any], *, training: bool) -> Any:
    acceptance = data_config.get("acceptance_sampling", {})
    if not isinstance(acceptance, dict):
        raise ValueError("data.acceptance_sampling must be a mapping")
    replacement = bool(
        acceptance.get("replacement", training)
    ) if training else False
    return make_clip_dataloader(
        dataset,
        max_tokens=int(data_config["max_tokens"]),
        collate_fn=collate_clip_records,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        persistent_workers=bool(data_config.get("persistent_workers", False)),
        prefetch_factor=int(data_config.get("prefetch_factor", 2)),
        trusted_store_fast_path=bool(data_config.get("trusted_store_fast_path", False)),
        strict_record_validation=bool(data_config.get("strict_record_validation", True)),
        oversize_policy=str(data_config.get("oversize_policy", "error")),
        seed=int(data_config.get("seed", 0)) + (0 if training else 1),
        shuffle=bool(training),
        replacement=replacement,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/codec.yaml"))
    parser.add_argument("--device", default=None, help="override training.device")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--legacy-bond-mode", choices=("topology", "distance_only"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="run this many complete train-loader epochs")
    parser.add_argument("--subset-fraction", type=float, default=None, help="deterministically train/evaluate on this fraction of each dataset")
    parser.add_argument("--fit-normalization", action="store_true")
    parser.add_argument("--normalization-batches", type=int, default=None, help="bound normalization fitting to a deterministic number of train batches")
    parser.add_argument("--temporal-ratio", type=int, default=None)
    parser.add_argument("--temporal-layers", type=int, default=None)
    parser.add_argument("--train-root", action="append", default=None)
    parser.add_argument("--valid-root", action="append", default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None, help="append per-step training metrics as JSONL")
    parser.add_argument("--log-every", type=int, default=1, help="write one training record every N optimizer steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate one batch without optimizing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw = _load_config(args.config)
    if args.device is not None:
        raw.setdefault("training", {})["device"] = args.device
    if args.temporal_ratio is not None:
        raw.setdefault("model", {})["temporal_ratio"] = args.temporal_ratio
    if args.temporal_layers is not None:
        raw.setdefault("model", {})["temporal_layers"] = args.temporal_layers
    if args.train_root is not None:
        raw.setdefault("data", {})["train_roots"] = args.train_root
    if args.valid_root is not None:
        raw.setdefault("data", {})["valid_roots"] = args.valid_root
    if args.save_dir is not None:
        raw.setdefault("training", {})["save_dir"] = str(args.save_dir)
    if args.seed is not None:
        raw.setdefault("data", {})["seed"] = args.seed
    data_seed = int(raw.get("data", {}).get("seed", 0))
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    if args.max_steps is not None and args.max_epochs is not None:
        raise ValueError("--max-steps and --max-epochs are mutually exclusive")
    if args.resume is not None and args.fit_normalization:
        raise ValueError(
            "--fit-normalization cannot be combined with --resume; "
            "normalization statistics are part of the checkpoint contract"
        )
    if args.max_epochs is not None and int(args.max_epochs) < 1:
        raise ValueError("--max-epochs must be positive")
    data_config = raw.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("data config must be a mapping")
    train_dataset = _dataset(data_config.get("train_roots", []))
    train_dataset, train_subset_info = _fractional_subset(
        train_dataset, args.subset_fraction, data_seed
    )
    train_loader = _loader(train_dataset, data_config, training=True)
    valid_loader = None
    valid_subset_info = None
    valid_roots = data_config.get("valid_roots", [])
    if valid_roots:
        valid_dataset, valid_subset_info = _fractional_subset(
            _dataset(valid_roots), args.subset_fraction, data_seed + 1
        )
        valid_loader = _loader(valid_dataset, data_config, training=False)

    batches_per_epoch = len(train_loader)
    max_steps = args.max_steps
    if args.max_epochs is not None:
        max_steps = int(args.max_epochs) * batches_per_epoch
    train_config = CodecTrainConfig.from_mapping(raw)
    if max_steps is not None:
        train_config.max_steps = int(max_steps)
    print(json.dumps({
        "train_subset": train_subset_info,
        "valid_subset": valid_subset_info,
        "batches_per_epoch": batches_per_epoch,
        "requested_epochs": args.max_epochs,
        "requested_steps": max_steps,
    }, sort_keys=True))

    model_config = raw.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model_kwargs = {
        key: model_config[key] for key in PVB_MODEL_CONFIG_KEYS if key in model_config
    }
    if "spatial_dtype" in model_kwargs:
        dtype_name = str(model_kwargs["spatial_dtype"]).removeprefix("torch.")
        try:
            model_kwargs["spatial_dtype"] = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported model.spatial_dtype {dtype_name!r}") from exc
        if not isinstance(model_kwargs["spatial_dtype"], torch.dtype):
            raise ValueError(f"model.spatial_dtype {dtype_name!r} is not a torch dtype")
    model = PVBCodecModel(**model_kwargs)
    trainer = CodecTrainer(
        model,
        train_loader,
        valid_loader,
        train_config,
        non_blocking_transfer=bool(data_config.get("non_blocking_transfer", True)),
    )
    model_bond_mode = model.frame_encoder.bond_construction_mode
    if model.frame_encoder.graph_mode == "native_radius":
        # Native radius ViSNet has no external topology/distance preparation.
        # Its graph is built exactly once inside forward_native.
        if model_bond_mode != "native_radius":
            raise RuntimeError("native-radius backbone has an invalid graph mode")
    elif model_bond_mode == "distance_only":
        # The frozen distance graph is a train-split artifact. Validation is
        # intentionally never part of canonical selection or cache setup.
        references = build_canonical_reference_index(
            train_dataset,
            source_split="train",
        )
        trainer.model.prepare_distance_bonds(references, device=trainer.device)
    elif model_bond_mode != "topology":
        raise ValueError(f"unsupported bond construction mode: {model_bond_mode!r}")
    eligibility = train_loader.batch_sampler.eligibility_report()
    distance_reference_contract = trainer.model.distance_reference_contract()
    protocol = dict(eligibility)
    if distance_reference_contract is not None:
        protocol["distance_reference_contract"] = distance_reference_contract
    print(json.dumps({"train_eligibility": protocol}, sort_keys=True))
    if args.resume is not None:
        trainer.load_checkpoint(
            args.resume,
            allow_legacy=args.allow_legacy_checkpoint,
            legacy_bond_mode=args.legacy_bond_mode,
        )
    normalization_elapsed_s = 0.0
    if args.fit_normalization:
        normalization_started = time.perf_counter()
        if args.normalization_batches is not None:
            if args.normalization_batches < 1:
                raise ValueError("--normalization-batches must be positive")
            normalization_source = islice(train_loader, args.normalization_batches)
        else:
            normalization_source = train_loader
        stats = trainer.fit_normalization(normalization_source)
        normalization_elapsed_s = time.perf_counter() - normalization_started
        print(json.dumps({
            "normalization_elapsed_s": normalization_elapsed_s,
            "normalization_stats": {key: value.as_dict() for key, value in stats.items()},
        }, indent=2))
    if args.dry_run:
        batch = next(iter(train_loader))
        print(json.dumps(trainer.evaluate_batch(batch), sort_keys=True))
        return 0
    save_dir = Path(raw.get("training", {}).get("save_dir", "ckpt/codec"))
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "eligibility.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path = args.log_path if args.log_path is not None else save_dir / "train_metrics.jsonl"
    training_started = time.perf_counter()
    metrics = trainer.run(
        max_steps=max_steps,
        log_path=log_path,
        log_every=args.log_every,
    )
    training_elapsed_s = time.perf_counter() - training_started
    checkpoint = save_dir / f"codec_step_{trainer.step:08d}.pt"
    trainer.save_checkpoint(checkpoint)
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "log": str(log_path),
        "step": trainer.step,
        "completed_epochs": trainer.step // batches_per_epoch,
        "batches_per_epoch": batches_per_epoch,
        "normalization_elapsed_s": normalization_elapsed_s,
        "training_elapsed_s": training_elapsed_s,
        "total_elapsed_s": normalization_elapsed_s + training_elapsed_s,
        "metrics": metrics,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
