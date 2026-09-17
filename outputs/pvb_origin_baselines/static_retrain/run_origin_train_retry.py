#!/usr/bin/env python
"""Retry Baseline C with the lazy full-coverage data adapter."""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path


REFERENCE_ROOT = Path("/workspace/PVB_origin")
OUTPUT_ROOT = Path("/workspace/PVB/outputs/pvb_origin_baselines/static_retrain")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--pairs-per-clip", type=int, default=1)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    sys.path.insert(0, str(REFERENCE_ROOT))
    adapter = _load(OUTPUT_ROOT / "streaming_origin_adapter_v2.py", "pvb_origin_streaming_adapter_retry")
    full_batch = _load(OUTPUT_ROOT / "full_coverage_dynamic_wrapper.py", "pvb_origin_full_coverage_batch_retry")

    import data as origin_data
    from utils import random_seed

    random_seed.SEED = args.seed
    origin_data.UniDataset = lambda path: adapter.make_streaming_unidataset(
        path, pairs_per_clip=args.pairs_per_clip
    )
    origin_data.DynamicBatchWrapper = full_batch.FullCoverageDynamicBatchWrapper
    adapter.StreamingClipPairDataset.collate_fn = None

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(OUTPUT_ROOT / "wandb_retry"))
    os.environ.setdefault("WANDB_SILENT", "true")
    os.chdir(REFERENCE_ROOT)
    sys.argv = [
        str(REFERENCE_ROOT / "train.py"),
        "--config", str(Path(args.config).resolve()),
        "--gpus", *[str(gpu) for gpu in args.gpus],
    ]
    runpy.run_path(str(REFERENCE_ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
