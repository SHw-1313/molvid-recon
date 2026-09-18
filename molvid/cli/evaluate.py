"""Evaluate codec-oracle and generated trajectories on the verified validation split."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from ..checkpoints import load_dit_inference
from ..data.batch import collate_clip_records
from ..data.io import write_trajectory
from ..data.manifest import load_datasets
from ..evaluation.report import aggregate_by_time_bucket, plot_report, write_report
from ..evaluation.runner import EvalConfig, RolloutTrack, evaluate_generation, evaluate_rollout
from ..runtime import configure_device


_SAMPLE = re.compile(r"^(?P<system>.+)_R[0-9]+_w[0-9]+$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--valid-index", type=int, required=True)
    parser.add_argument("--history", type=int, choices=(4, 8), default=8)
    parser.add_argument("--steps", type=int, choices=(8, 16), default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-track", type=Path, help="reference NPZ with x=[8+8S,N,3]")
    parser.add_argument("--rollout-seed", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = configure_device(args.device, deterministic=True)
    loaded = load_dit_inference(
        args.checkpoint, expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec, codec_sha256=args.codec_sha256, device=device,
    )
    splits = load_datasets(args.manifest_root)
    try:
        if splits.data_hash != loaded["data_hash"]:
            raise ValueError("evaluation manifest differs from the DiT training data contract")
        if args.valid_index < 0 or args.valid_index >= len(splits.valid):
            raise IndexError("validation clip index is out of range")
        batch = collate_clip_records([splits.valid[args.valid_index]])
        config = EvalConfig(
            history_frames=args.history, steps=args.steps, seed=args.seed,
            source_mode=loaded["source_mode"], center_kind=loaded["center_kind"],
        )
        result = evaluate_generation(
            loaded["codec"], loaded["model"], loaded["adapter"], loaded["statistics"],
            batch, config=config,
            codec_hash=loaded["codec_hash"], data_hash=loaded["data_hash"],
        )
        sample_id = str(batch.sample_id[0])
        match = _SAMPLE.match(sample_id)
        system = match.group("system") if match else sample_id
        report = {
            "schema_version": "molvid.dit.evaluation.v1",
            "checkpoint_step": loaded["step"],
            "sample_id": sample_id,
            "system": system,
            "time_bucket_id": str(batch.time_bucket_id[0]),
            **result,
        }
        report["by_time_bucket"] = aggregate_by_time_bucket([{
            "system": system,
            "sample_id": sample_id,
            "draw": 0,
            "time_bucket_id": str(batch.time_bucket_id[0]),
            "metrics": result["generated_result"],
        }])
        if args.rollout_track:
            if not args.rollout_seed:
                raise ValueError("--rollout-track requires --rollout-seed values")
            with np.load(args.rollout_track, allow_pickle=False) as payload:
                if "x" not in payload.files:
                    raise ValueError("rollout reference NPZ lacks x")
                reference = np.asarray(payload["x"], dtype=np.float32)
            track = RolloutTrack(
                batch, batch.x.new_tensor(reference), system,
                sample_id.rsplit("_", 2)[-2] if "_R" in sample_id else "",
            )
            rollout_result = evaluate_rollout(
                loaded["codec"], loaded["model"], loaded["adapter"], loaded["statistics"],
                track, seeds=tuple(args.rollout_seed), steps=args.steps,
                codec_hash=loaded["codec_hash"], data_hash=loaded["data_hash"],
                source_mode=loaded["source_mode"], center_kind=loaded["center_kind"],
            )
            prediction = rollout_result.pop("prediction")
            intervals = batch.delta_time_ps[0].cpu().numpy()
            if not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-3):
                raise ValueError("rollout export requires a regular physical time grid")
            delta = float(intervals[0])
            time_ps = float(batch.time_ps[0, 0]) + np.arange(prediction.shape[0], dtype=np.float32) * delta
            write_trajectory(
                args.output_root / "rollout.npz",
                prediction.detach().cpu().numpy(), time_ps,
            )
            report["rollout"] = rollout_result
        elif args.rollout_seed:
            raise ValueError("--rollout-seed requires --rollout-track")
        write_report(
            report, args.output_root / "metrics.json", args.output_root / "report.md"
        )
        if args.plot:
            plot_report(report, args.output_root / "future_aligned_rmsd.png")
        return 0
    finally:
        splits.close()


if __name__ == "__main__":
    raise SystemExit(main())
