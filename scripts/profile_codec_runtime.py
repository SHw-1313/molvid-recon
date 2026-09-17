#!/usr/bin/env python3
"""Profile the CUDA codec hot path after warm-up.

The command is CUDA-hard-fail and records data-wait, step latency,
aten::_local_scalar_dense, and synchronization events.  Supply
--baseline-json from a separately captured pre-change run to get a before /
after comparison; the script never fabricates a baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from module.bond_sources import build_canonical_reference_index
from scripts.run_engineering_acceptance import (
    DEVICE,
    _make_model,
    _real_batch,
    require_cuda,
)
from trainer.codec_trainer import CodecTrainConfig, CodecTrainer, TimeBucketSpec
from trainer.codec_losses import CodecLossWeights


ROOT = Path(__file__).resolve().parents[1]


def profile_runtime(mode: str, *, warmup: int, steps: int, output: Path, baseline: Path | None) -> dict[str, Any]:
    if warmup < 0 or steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")
    runtime = require_cuda()
    _, batch_cpu, records = _real_batch()
    model = _make_model(
        {"temporal_layers": 1, "temporal_ratio": 4, "full": True},
        bond_mode=mode,
    )
    if mode == "distance_only":
        refs = build_canonical_reference_index(records, source_split="train")
        model.prepare_distance_bonds(refs, device=DEVICE)
    config = CodecTrainConfig(
        lr=1.0e-4,
        weight_decay=1.0e-6,
        max_steps=warmup + steps,
        grad_clip=1.0,
        warmup_steps=20,
        device="cuda",
        precision="fp32",
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0, 1.0),),
        loss_schedule=((0, CodecLossWeights(coordinate=1.0)),),
        normalization_min_count=1,
    )
    trainer = CodecTrainer(model, [batch_cpu], config=config, device=DEVICE)
    iterator = iter([batch_cpu] * (warmup + steps))
    active_latencies: list[float] = []
    data_wait_s = 0.0
    profiler_schedule = schedule(wait=warmup, warmup=0, active=steps, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for index in range(warmup + steps):
            wait_start = time.perf_counter()
            with record_function("data_wait"):
                batch = next(iterator)
            data_wait_s += time.perf_counter() - wait_start
            started = time.perf_counter()
            with record_function("optimizer_step"):
                trainer.optimizer_step(batch)
            torch.cuda.synchronize()
            if index >= warmup:
                active_latencies.append(time.perf_counter() - started)
            prof.step()

    event_rows = []
    for event in prof.key_averages():
        event_rows.append(
            {
                "key": event.key,
                "count": int(event.count),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "cpu_time_us": float(event.cpu_time_total),
                "self_cuda_time_us": float(getattr(event, "self_device_time_total", 0.0)),
                "cuda_time_us": float(getattr(event, "device_time_total", 0.0)),
            }
        )
    def total_matching(*needles: str, field: str) -> float:
        return float(
            sum(
                row[field]
                for row in event_rows
                if any(needle in row["key"] for needle in needles)
            )
        )
    current = {
        "runtime": runtime,
        "mode": mode,
        "warmup_steps": warmup,
        "profiled_steps": steps,
        "data_wait_total_s": data_wait_s,
        "data_wait_mean_s": data_wait_s / max(warmup + steps, 1),
        "step_latency_s": active_latencies,
        "step_latency_median_s": float(np.median(np.asarray(active_latencies))),
        "step_latency_mean_s": float(np.mean(np.asarray(active_latencies))),
        "aten_local_scalar_dense": {
            "self_cpu_time_us": total_matching("aten::_local_scalar_dense", field="self_cpu_time_us"),
            "count": sum(row["count"] for row in event_rows if row["key"] == "aten::_local_scalar_dense"),
        },
        "cuda_synchronization": {
            "self_cpu_time_us": total_matching(
                "cudaDeviceSynchronize",
                "cudaStreamSynchronize",
                "aten::cuda_synchronize",
                "synchronize",
                field="self_cpu_time_us",
            ),
            "cuda_time_us": total_matching(
                "cudaDeviceSynchronize",
                "cudaStreamSynchronize",
                "aten::cuda_synchronize",
                "synchronize",
                field="cuda_time_us",
            ),
        },
        "events": event_rows,
    }
    result: dict[str, Any] = {
        "schema_version": "pvb.codec.runtime_profile.v1",
        "before": None,
        "after": current,
        "comparison": None,
    }
    if baseline is not None:
        before = json.loads(baseline.read_text(encoding="utf-8"))
        result["before"] = before
        before_metrics = before.get("after", before)
        result["comparison"] = {
            "step_latency_median_ratio": current["step_latency_median_s"] / max(float(before_metrics["step_latency_median_s"]), 1e-12),
            "data_wait_mean_ratio": current["data_wait_mean_s"] / max(float(before_metrics["data_wait_mean_s"]), 1e-12),
            "local_scalar_cpu_time_ratio": current["aten_local_scalar_dense"]["self_cpu_time_us"] / max(float(before_metrics["aten_local_scalar_dense"]["self_cpu_time_us"]), 1e-12),
            "sync_cpu_time_ratio": current["cuda_synchronization"]["self_cpu_time_us"] / max(float(before_metrics["cuda_synchronization"]["self_cpu_time_us"]), 1e-12),
        }
    else:
        result["comparison_note"] = (
            "No pre-change profiler was supplied.  The report is an after-only "
            "measurement; do not interpret it as a before/after improvement."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del trainer, model, batch_cpu
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("topology", "distance_only"), default="topology")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/engineering_v1/profiling" / f"{args.mode}.json"
    result = profile_runtime(
        args.mode,
        warmup=args.warmup,
        steps=args.steps,
        output=output,
        baseline=args.baseline_json,
    )
    print(json.dumps({"path": str(output), "mode": result["after"]["mode"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
