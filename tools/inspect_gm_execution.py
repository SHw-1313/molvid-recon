"""Inspect train/valid index metadata and proposed A schedules; no model execution."""

import argparse
from collections import Counter
from pathlib import Path

from molvid.cli.train_frame_joint import SCHEMA, _open_data, _sampler, _trajectory
from molvid.config import load_config
from molvid.data.sampling import get_clip_specs
from molvid.runtime import atomic_write_json
from molvid.training.execution import execution_manifest, execution_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    raw = load_config(args.config, schema=SCHEMA)
    train, valid, data_hash, normalization_hash, relationship, close = _open_data(raw["data"])
    try:
        card = {"schema": "molvid.A.data_card.v1", "data_hash": data_hash,
                "normalization_data_hash": normalization_hash, "relationship": relationship,
                "sealed_test_opened": False, "status": "metadata_verified_model_not_run"}
        for name, dataset in (("train", train), ("valid", valid)):
            specs = get_clip_specs(dataset)
            trajectories = sorted({_trajectory(item.sample_id) for item in specs})
            systems = sorted({value.rsplit("_R", 1)[0] for value in trajectories})
            card[name] = {"clips": len(specs), "trajectories": len(trajectories),
                          "systems": len(systems), "system_ids": systems,
                          "bucket_clip_counts": dict(Counter(item.time_bucket_id for item in specs))}
        sampler = _sampler(train, raw["training"], seed=args.seed, rank=0, world=1)
        card["proposed_epochs"] = []
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            entries = execution_schedule(sampler.global_batches, sampler.specs,
                                         histories=(4, 8), seed=args.seed, epoch=epoch)
            manifest = execution_manifest(entries)
            exposure = Counter(_trajectory(sampler.specs[index].sample_id)
                               for entry in entries for index in entry.indices)
            card["proposed_epochs"].append({
                "epoch": epoch, "updates": len(entries), "clips": sum(exposure.values()),
                "trajectory_min_clips": min(exposure.values()), "trajectory_max_clips": max(exposure.values()),
                "coverage": manifest["coverage"], "schedule_hash": manifest["sha256"],
                "status": "proposed_not_frozen_pending_profile",
            })
            atomic_write_json(args.output / f"proposed_execution_epoch_{epoch:04d}.json", manifest)
        atomic_write_json(args.output / "data_card.json", card)
        print({"data_hash": data_hash, "train_systems": card["train"]["systems"],
               "valid_systems": card["valid"]["systems"], "epochs": card["proposed_epochs"]}, flush=True)
    finally:
        close()


if __name__ == "__main__":
    main()
