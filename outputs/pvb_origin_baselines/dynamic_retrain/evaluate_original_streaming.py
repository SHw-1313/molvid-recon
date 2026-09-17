"""Evaluate a PVB_origin dynamic checkpoint on the matching valid clip split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-valid", required=True)
    parser.add_argument("--misato-valid", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--pair-bound", type=int, default=5000)
    parser.add_argument("--atlas-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_batches < 1 or args.max_batches % 2:
        raise ValueError("--max-batches must be a positive even number")

    ref_repo = Path(args.ref_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(output_dir / "wandb_eval")
    sys.path.insert(0, str(ref_repo))
    sys.path.insert(0, str(output_dir))
    os.chdir(ref_repo)

    from data import DynamicBatchWrapper, MixDatasetWrapper
    from data.collate import collate_fn
    from trainer import DynamicTrainer, TrainConfig
    from streaming_pair_dataset import StreamingAdjacentPairDataset
    import utils.random_seed as random_seed

    random_seed.SEED = args.seed
    random_seed.setup_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    source_roots = {"atlas_dt_100ps": Path(args.atlas_valid).resolve()}
    if not args.atlas_only:
        source_roots["misato_dt_80ps"] = Path(args.misato_valid).resolve()
    per_source = {}
    source_loaders = {}
    source_datasets = {}
    source_wrappers = {}
    batches_per_source = args.max_batches if args.atlas_only else args.max_batches // 2
    for source, root in source_roots.items():
        dataset = StreamingAdjacentPairDataset(str(root))
        dataset.collate_fn = collate_fn
        wrapper = DynamicBatchWrapper(
            dataset,
            complexity="n",
            ubound_per_batch=args.pair_bound,
            same_origin=False,
        )
        if len(wrapper) < batches_per_source:
            raise RuntimeError(
                f"{source} has only {len(wrapper)} dynamic batches, "
                f"need {batches_per_source}"
            )
        source_datasets[source] = dataset
        source_wrappers[source] = wrapper
        source_loaders[source] = DataLoader(
            wrapper,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA evaluation but torch.cuda.is_available() is false")
    model = torch.load(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    config = TrainConfig(
        str(output_dir / "eval_trainer"),
        lr=1.0e-4,
        max_epoch=1,
        warmup=100,
        patience=1,
        grad_clip=1.0,
        save_topk=-1,
    )
    trainer = DynamicTrainer(model, None, None, config)
    trainer.local_rank = -1

    started = time.time()
    all_losses: list[float] = []
    for source, loader in source_loaders.items():
        losses: list[float] = []
        pair_records = 0
        atoms = 0
        selected_clips: set[int] = set()
        wrapper = source_wrappers[source]
        dataset = source_datasets[source]
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches_per_source:
                break
            batch_items = wrapper.batch_indexes[batch_index]
            pair_records += len(batch_items)
            atoms += sum(dataset.get_len(index) for index in batch_items)
            selected_clips.update(dataset._locate(index)[0] for index in batch_items)
            batch = trainer.to_device(batch, device)
            with torch.no_grad():
                loss = trainer.valid_step(batch, batch_index)
            value = float(loss.detach().cpu().item())
            losses.append(value)
            all_losses.append(value)
        per_source[source] = {
            "batches": len(losses),
            "pair_records": pair_records,
            "unique_clips": len(selected_clips),
            "atoms": atoms,
            "loss_mean": _mean(losses),
            "kl_loss_mean": _mean(
                [float(value) for value in trainer.writer_buffer.get("KL Loss/valid", [])]
            ),
            "rec_vel_loss_mean": _mean(
                [float(value) for value in trainer.writer_buffer.get("rec. vel Loss/valid", [])]
            ),
            "rec_drf_loss_mean": _mean(
                [float(value) for value in trainer.writer_buffer.get("rec. drf Loss/valid", [])]
            ),
        }
        trainer.writer_buffer = {}

    report = {
        "schema_version": "pvb.origin.baseline_d.eval.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "ref_repo": str(ref_repo),
        "seed": args.seed,
        "device": str(device),
        "atlas_only": args.atlas_only,
        "pair_policy": "all_adjacent_pairs_streamed_from_each_T16_clip",
        "pair_bound": args.pair_bound,
        "max_batches_total": args.max_batches,
        "batches_per_source": batches_per_source,
        "valid_roots": {name: str(path) for name, path in source_roots.items()},
        "per_source": per_source,
        "overall_loss_mean": _mean(all_losses),
        "wall_time_s": time.time() - started,
    }
    out_path = output_dir / "baseline_d_valid_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    for dataset in source_datasets.values():
        dataset.close()


if __name__ == "__main__":
    main()
