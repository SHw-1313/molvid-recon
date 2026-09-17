#!/usr/bin/env python3
"""Reproducible T09 codec smoke: one device, two buckets, checkpoint, DDP."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from data.clip_dataset import ClipMMapDataset, collate_clip_records
from trainer.codec_losses import compute_codec_losses
from trainer.codec_trainer import CodecTrainConfig, CodecTrainer, PVBCodecModel, TimeBucketSpec, _to_device


def _synthetic_record(frames: int, atoms: int, delta: float, bucket: str) -> dict[str, Any]:
    time_ps = torch.arange(frames, dtype=torch.float32) * delta
    base = torch.arange(atoms, dtype=torch.float32).view(1, atoms, 1)
    x = torch.zeros(frames, atoms, 3)
    x[:, :, 0] = base[..., 0] + 0.05 * torch.sin(time_ps[:, None] / max(delta, 1.0))
    x[:, :, 1] = (base[..., 0] % 3) * 0.8 + 0.02 * torch.cos(time_ps[:, None] / max(delta, 1.0))
    x[:, :, 2] = (base[..., 0] % 2) * 0.7
    source = torch.arange(max(0, atoms - 1), dtype=torch.long)
    bonds = torch.stack([source, source + 1], dim=0) if source.numel() else torch.empty(2, 0, dtype=torch.long)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": f"smoke_{bucket}",
        "task": "trajectory",
        "time_bucket_id": bucket,
        "time_ps": time_ps.numpy(),
        "delta_time_ps": torch.diff(time_ps).numpy(),
        "x": x.numpy(),
        "bpos": x.numpy(),
        "atype": torch.ones(atoms, dtype=torch.long).numpy(),
        "btype": torch.zeros(atoms, dtype=torch.long).numpy(),
        "block_id": torch.arange(atoms, dtype=torch.long).numpy(),
        "component_id": torch.zeros(atoms, dtype=torch.long).numpy(),
        "atom_source_index": torch.arange(atoms, dtype=torch.long).numpy(),
        "atom_identity": [f"{bucket}:atom:{i}" for i in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long).numpy(),
        "loss_mask": torch.ones(atoms, dtype=torch.bool).numpy(),
        "align_mask": torch.ones(atoms, dtype=torch.bool).numpy(),
        "bond_index": bonds.numpy(),
    }


def _batches(root: str | None) -> list[Any]:
    if root is not None:
        dataset = ClipMMapDataset(root)
        return [collate_clip_records([dataset[0]])]
    return [
        collate_clip_records([_synthetic_record(16, 8, 100.0, "dt_100ps")]),
        collate_clip_records([_synthetic_record(16, 8, 1000.0, "dt_1ns")]),
    ]


def run_one_device(device: str, root: str | None, checkpoint: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    selected = torch.device(device)
    batches = _batches(root)
    model = PVBCodecModel(
        hidden_channels=16,
        spatial_layers=1,
        temporal_layers=1,
        temporal_ratio=4,
        num_rbf=8,
        num_heads=2,
        max_num_neighbors=16,
        neighbor_backend="cuda_radius" if selected.type == "cuda" else "dense_test",
        time_scale_ps=100.0,
    )
    config = CodecTrainConfig(
        lr=1e-4,
        max_steps=len(batches),
        grad_clip=1.0,
        device=str(selected),
        bucket_specs=(
            TimeBucketSpec("dt_80ps", 80.0, 0.5),
            TimeBucketSpec("dt_100ps", 100.0, 0.5),
            TimeBucketSpec("dt_1ns", 1000.0, 0.5),
        ),
        loss_schedule=((0, {"coordinate": 1.0, "velocity": 0.1}),),
    )
    trainer = CodecTrainer(model, batches, config=config, device=selected)
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    start = time.perf_counter()
    train_metrics = [trainer.optimizer_step(batch) for batch in batches]
    validation_metrics = [trainer.evaluate_batch(batch) for batch in batches]
    elapsed = time.perf_counter() - start
    trainer.save_checkpoint(checkpoint)
    resumed = CodecTrainer(
        PVBCodecModel(
            hidden_channels=16,
            spatial_layers=1,
            temporal_layers=1,
            temporal_ratio=4,
            num_rbf=8,
            num_heads=2,
            max_num_neighbors=16,
            neighbor_backend="cuda_radius" if selected.type == "cuda" else "dense_test",
            time_scale_ps=100.0,
        ),
        batches,
        config=config,
        device=selected,
    )
    resumed.load_checkpoint(checkpoint)
    peak = int(torch.cuda.max_memory_allocated(selected)) if selected.type == "cuda" else 0
    return {
        "device": str(selected),
        "frames": 16,
        "temporal_ratio": 4,
        "buckets": ["dt_100ps", "dt_1ns"] if root is None else list(batches[0].time_bucket_id),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "wall_time_s": elapsed,
        "peak_memory_bytes": peak,
        "checkpoint": str(checkpoint),
        "resumed_step": resumed.step,
    }


def _ddp_codec_worker(rank: int, devices: tuple[int, ...], init_file: str, result_file: str) -> None:
    device_id = int(devices[rank])
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=len(devices),
    )
    try:
        torch.manual_seed(100 + rank)
        torch.cuda.manual_seed_all(100 + rank)
        cpu_batch = _batches(None)[rank % 2]
        model = PVBCodecModel(
            hidden_channels=16,
            spatial_layers=1,
            temporal_layers=1,
            temporal_ratio=4,
            num_rbf=8,
            num_heads=2,
            max_num_neighbors=16,
            neighbor_backend="cuda_radius",
            time_scale_ps=100.0,
        ).to(device)
        # DDP does not proxy the explicit CPU registration phase. Register on
        # the underlying model before moving any topology-bearing fields.
        model.prepare_batch(cpu_batch)
        batch = _to_device(cpu_batch, device)
        ddp_model = DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
        config = CodecTrainConfig(
            lr=1e-4,
            max_steps=1,
            grad_clip=1.0,
            device=str(device),
            bucket_specs=(
                TimeBucketSpec("dt_80ps", 80.0, 0.5),
                TimeBucketSpec("dt_100ps", 100.0, 0.5),
                TimeBucketSpec("dt_1ns", 1000.0, 0.5),
            ),
            loss_schedule=((0, {"coordinate": 1.0, "velocity": 0.1}),),
        )
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config.lr)
        optimizer.zero_grad(set_to_none=True)
        output = ddp_model(batch)
        losses = compute_codec_losses(output, batch, weights=config.weights_at(0))
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite NCCL DDP smoke loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        optimizer.step()
        marker = torch.tensor([float(rank)], device=device)
        dist.all_reduce(marker, op=dist.ReduceOp.SUM)
        expected = float(len(devices) * (len(devices) - 1) / 2)
        if float(marker.item()) != expected:
            raise RuntimeError("NCCL DDP ranks did not participate in one collective")
        peak = torch.tensor([torch.cuda.max_memory_allocated(device)], dtype=torch.long, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            Path(result_file).write_text(
                json.dumps({"loss": float(total.detach().cpu()), "peak_memory_bytes": int(peak.item())}) + "\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


def run_ddp_gpu(devices: tuple[int, ...] = (0, 1)) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the NCCL DDP smoke")
    if len(devices) < 2:
        raise ValueError("NCCL DDP smoke requires at least two GPU ids")
    if any(device < 0 or device >= torch.cuda.device_count() for device in devices):
        raise ValueError(f"invalid GPU ids {devices}; device_count={torch.cuda.device_count()}")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="pvb_nccl_ddp_") as directory:
        init_file = str(Path(directory) / "init")
        result_file = str(Path(directory) / "result.json")
        mp.spawn(
            _ddp_codec_worker,
            args=(tuple(devices), init_file, result_file),
            nprocs=len(devices),
            join=True,
        )
        details = json.loads(Path(result_file).read_text(encoding="utf-8"))
    return {
        "backend": "nccl",
        "world_size": len(devices),
        "devices": list(devices),
        "passed": True,
        **details,
    }


def _ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    torch.manual_seed(100 + rank)
    model = DistributedDataParallel(nn.Linear(1, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[float(rank + 1)]])
    target = torch.tensor([[1.0]])
    optimizer.zero_grad(set_to_none=True)
    (model(x) - target).square().mean().backward()
    optimizer.step()
    marker = torch.tensor([float(rank)])
    dist.all_reduce(marker, op=dist.ReduceOp.SUM)
    if float(marker.item()) != float(world_size * (world_size - 1) / 2):
        raise RuntimeError("DDP ranks did not participate in one collective")
    dist.destroy_process_group()


def run_ddp(world_size: int = 2) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pvb_ddp_") as directory:
        init_file = str(Path(directory) / "init")
        mp.spawn(_ddp_worker, args=(world_size, init_file), nprocs=world_size, join=True)
    return {"backend": "gloo", "world_size": world_size, "passed": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/pvb_codec_smoke.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cpu-fallback", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-gpus", default=None, help="comma-separated GPU ids for the real NCCL DDP smoke")
    args = parser.parse_args(argv)
    if args.ddp:
        if args.ddp_gpus:
            devices = tuple(int(item.strip()) for item in args.ddp_gpus.split(",") if item.strip())
            result = {"ddp": run_ddp_gpu(devices)}
        else:
            result = {"ddp": run_ddp()}
    else:
        if args.device == "auto":
            if torch.cuda.is_available():
                args.device = "cuda:0"
            elif args.cpu_fallback:
                args.device = "cpu"
            else:
                raise RuntimeError("CUDA is unavailable; pass --cpu-fallback only for a CPU diagnostic")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device but torch.cuda.is_available() is false")
        result = run_one_device(args.device, args.data_root, args.checkpoint)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
