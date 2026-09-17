#!/usr/bin/env python
"""Probe one lazy origin-format pair at selected atom counts on the target GPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/PVB/outputs/pvb_origin_baselines/static_retrain")
CKPT = "/data1/repo/PVB/ckpt/pdbbind_pretrain/version_1/checkpoint/epoch191_step113472.ckpt"


def load_adapter():
    import importlib.util
    import sys
    name = "pvb_origin_streaming_adapter_probe"
    spec = importlib.util.spec_from_file_location(name, ROOT / "streaming_origin_adapter_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--targets", nargs="+", type=int, default=[4000, 8000, 12000, 16000, 20000])
    args = parser.parse_args()

    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    import sys
    sys.path.insert(0, "/workspace/PVB_origin")
    from data.collate import collate_fn
    adapter = load_adapter()
    root = f"/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/{args.source}/dt_{'100' if args.source == 'atlas' else '80'}ps/{args.split}"
    ds = adapter.StreamingClipPairDataset(root, pairs_per_clip=1)
    selected = {}
    for i in range(len(ds)):
        n = ds.get_len(i)
        for target in args.targets:
            if target not in selected and n >= target:
                selected[target] = i
    selected = dict(sorted(selected.items()))
    model = torch.load(CKPT, map_location="cpu")
    model = model.cuda()
    model.train()
    output = {"source": args.source, "split": args.split, "targets": {}}
    for target, index in selected.items():
        torch.cuda.empty_cache()
        gc.collect()
        record = ds[index]
        batch = collate_fn([[record]])
        n = int(batch["x0"].shape[0])
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        row = {"index": index, "atoms": n}
        try:
            loss, parts = model._train({k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items()}, mode="md")
            loss.backward()
            row.update({"status": "ok", "loss": float(loss.detach().cpu()), "peak_bytes": int(torch.cuda.max_memory_allocated()), "seconds": time.time() - started})
        except RuntimeError as exc:
            row.update({"status": "runtime_error", "error": str(exc)[:500], "peak_bytes": int(torch.cuda.max_memory_allocated()), "seconds": time.time() - started})
        output["targets"][str(target)] = row
        del batch, record
        model.zero_grad(set_to_none=True)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
