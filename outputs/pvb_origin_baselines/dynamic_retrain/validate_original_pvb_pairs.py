#!/usr/bin/env python3
"""Read migrated stores through the unmodified PVB_origin data path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store", action="append", required=True)
    parser.add_argument("--max-batches", type=int, default=2)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.repo))
    from data import DynamicBatchWrapper, UniDataset, collate_fn

    results = []
    for store in args.store:
        dataset = UniDataset(str(store))
        wrapped = DynamicBatchWrapper(
            dataset, complexity="n", ubound_per_batch=5000, same_origin=False
        )
        loader = DataLoader(
            wrapped,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        batches = []
        for index, batch in enumerate(loader):
            batches.append(
                {
                    "batch_index": index,
                    "atoms": int(batch["atype"].numel()),
                    "graphs": int(batch["abid"].max().item()) + 1,
                    "x0_shape": list(batch["x0"].shape),
                    "x1_shape": list(batch["x1"].shape),
                    "bond_shape": list(batch["bond_index"].shape),
                    "edge_mask_sum": int(batch["edge_mask"].sum().item()),
                    "mask_sum": int(batch["mask"].sum().item()),
                }
            )
            if index + 1 >= args.max_batches:
                break
        results.append(
            {
                "store": str(store),
                "records": len(dataset),
                "dynamic_batches": len(wrapped),
                "checked_batches": batches,
            }
        )
    print(json.dumps({"schema_version": "pvb.origin.pair-validation.v1", "stores": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
