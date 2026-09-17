#!/usr/bin/env python
"""Audit the reference DynamicBatchWrapper selection for pilot500."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

REFERENCE_ROOT = Path("/workspace/PVB_origin")
OUTPUT_ROOT = Path("/workspace/PVB/outputs/pvb_origin_baselines/static_retrain")
sys.path.insert(0, str(REFERENCE_ROOT))


def load_adapter():
    spec = importlib.util.spec_from_file_location(
        "pvb_origin_streaming_adapter_audit", OUTPUT_ROOT / "streaming_origin_adapter_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    from utils.random_seed import setup_seed
    from data.dataset_wrapper import DynamicBatchWrapper

    setup_seed(20260810)
    adapter = load_adapter()
    paths = [
        ("atlas_train", "/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps/train"),
        ("atlas_valid", "/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps/valid"),
        ("misato_train", "/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/misato/dt_80ps/train"),
        ("misato_valid", "/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/misato/dt_80ps/valid"),
    ]
    rows = {}
    for name, path in paths:
        dataset = adapter.StreamingClipPairDataset(path, pairs_per_clip=1)
        wrapper = DynamicBatchWrapper(dataset, complexity="n", ubound_per_batch=5000, same_origin=False)
        selected_batches = min(500, len(wrapper))
        selected_items = sum(len(wrapper.batch_indexes[i]) for i in range(selected_batches))
        atoms = [dataset.get_len(i) for i in range(len(dataset))]
        rows[name] = {
            "clips_and_pairs": len(dataset),
            "dynamic_batches_before_max_batches": len(wrapper),
            "pilot_batches_selected": selected_batches,
            "pairs_selected_by_max_batches": selected_items,
            "pairs_over_5000_atoms": sum(x > 5000 for x in atoms),
            "pairs_at_or_under_5000_atoms": sum(x <= 5000 for x in atoms),
            "max_atoms": max(atoms),
        }
    print(json.dumps({"seed": 20260810, "ubound_per_batch": 5000, "max_batches": 500, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
