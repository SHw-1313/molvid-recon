"""Train the current trajectory codec from versioned clip stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import ConcatDataset

from ..codec.model import build_codec
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.sampling import make_clip_dataloader
from ..data.store import ClipMMapDataset
from ..geometry.topology import build_canonical_reference_index
from ..runtime import configure_device, seed_all
from ..training.codec import CodecTrainConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/codec_train.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-root", action="append")
    parser.add_argument("--valid-root", action="append")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--fit-normalization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _stores(roots: Sequence[str]) -> tuple[Any, list[ClipMMapDataset]]:
    stores = [ClipMMapDataset(root) for root in roots]
    if not stores:
        raise ValueError("training requires at least one train clip store")
    return (stores[0] if len(stores) == 1 else ConcatDataset(stores)), stores


def _loader(dataset: Any, data: dict[str, Any], *, training: bool):
    return make_clip_dataloader(
        dataset,
        max_tokens=int(data["max_tokens"]),
        collate_fn=collate_clip_records,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False)),
        seed=int(data.get("seed", 0)) + (0 if training else 1),
        shuffle=training,
        replacement=training,
        oversize_policy="error",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume_sha256 and args.resume is None:
        raise ValueError("--resume-sha256 requires --resume")
    raw = load_config(args.config, schema="molvid.codec.config.v1")
    data = raw.get("data")
    model_config = raw.get("model")
    training = raw.get("training")
    if not all(isinstance(item, dict) for item in (data, model_config, training)):
        raise ValueError("codec configuration requires data, model and training mappings")
    if args.train_root:
        data["train_roots"] = args.train_root
    if args.valid_root:
        data["valid_roots"] = args.valid_root
    if args.device:
        training["device"] = args.device
    if args.max_steps is not None:
        training["max_steps"] = int(args.max_steps)
    if args.save_dir:
        training["save_dir"] = str(args.save_dir)
    if args.resume and args.fit_normalization:
        raise ValueError("normalization fitting cannot be combined with resume")
    config = CodecTrainConfig.from_mapping(raw)
    device = configure_device(config.device)
    seed_all(int(data.get("seed", 0)))
    kwargs = dict(model_config)
    if "spatial_dtype" in kwargs:
        dtype_name = str(kwargs["spatial_dtype"]).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"invalid model.spatial_dtype {dtype_name!r}")
        kwargs["spatial_dtype"] = dtype
    # The current constructor rejects retired legacy/refiner settings rather
    # than silently selecting another architecture.
    model = build_codec(kwargs)
    train_dataset, train_stores = _stores(data.get("train_roots", ()))
    valid_stores: list[ClipMMapDataset] = []
    try:
        train_loader = _loader(train_dataset, data, training=True)
        valid_loader = None
        if data.get("valid_roots"):
            valid_dataset, valid_stores = _stores(data["valid_roots"])
            valid_loader = _loader(valid_dataset, data, training=False)
        from ..training.codec import CodecTrainer

        trainer = CodecTrainer(model, train_loader, valid_loader, config)
        bond_mode = model.frame_encoder.bond_construction_mode
        if bond_mode == "distance_only":
            references = build_canonical_reference_index(train_dataset, source_split="train")
            model.prepare_distance_bonds(references, device=trainer.device)
        elif bond_mode != "topology":
            raise ValueError(f"unsupported current bond construction mode {bond_mode!r}")
        if args.resume:
            trainer.load_checkpoint(args.resume, expected_sha256=args.resume_sha256)
        if args.fit_normalization:
            trainer.fit_normalization()
        if args.dry_run:
            metrics = trainer.evaluate_batch(next(iter(train_loader)))
            print(json.dumps({"dry_run": True, "metrics": metrics}, sort_keys=True))
            return 0
        save_dir = Path(training.get("save_dir", "runs/molvid_codec"))
        log_path = save_dir / "train_metrics.jsonl"
        metrics = trainer.run(max_steps=config.max_steps, log_path=log_path, log_every=args.log_every)
        checkpoint = trainer.save_checkpoint(save_dir / f"codec_step_{trainer.step:08d}.pt")
        print(json.dumps({
            "checkpoint": str(checkpoint), "step": trainer.step,
            "log": str(log_path), "metrics": metrics,
        }, sort_keys=True))
        return 0
    finally:
        for store in (*train_stores, *valid_stores):
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
