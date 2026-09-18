"""Generate a clip or short rollout from an explicit observed-prefix NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..checkpoints import load_dit_inference
from ..config import cli_config_argv
from ..data.batch import collate_clip_records
from ..data.io import write_trajectory
from ..data.manifest import load_datasets
from ..generation import rollout, sample_clip
from ..runtime import atomic_write_json, configure_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--valid-index", type=int, required=True)
    parser.add_argument("--prefix", type=Path, required=True, help="NPZ containing only x=[H,N,3]")
    parser.add_argument("--history", type=int, choices=(4, 8), default=8)
    parser.add_argument("--steps", type=int, choices=(8, 16), default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-seed", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _prefix(path: Path, *, history: int, atoms: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"x"}:
            raise ValueError("observed-prefix NPZ must contain only x")
        value = np.asarray(payload["x"], dtype=np.float32)
    if value.shape != (history, atoms, 3) or not np.isfinite(value).all():
        raise ValueError("observed prefix must have finite shape [H,N,3]")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    words = list(argv) if argv is not None else sys.argv[1:]
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    config_path = preliminary.parse_known_args(words)[0].config
    defaults = cli_config_argv(
        config_path, schema="molvid.sample.v1",
        allowed_fields=(
            "checkpoint", "checkpoint_sha256", "codec", "codec_sha256",
            "manifest_root", "valid_index", "prefix", "history", "steps",
            "seed", "rollout_seed", "device", "output",
        ),
        path_fields=("checkpoint", "codec", "manifest_root", "prefix", "output"),
    ) if config_path is not None else []
    args = _parser().parse_args([*defaults, *words])
    device = configure_device(args.device, deterministic=True)
    loaded = load_dit_inference(
        args.checkpoint, expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec, codec_sha256=args.codec_sha256, device=device,
    )
    splits = load_datasets(args.manifest_root)
    try:
        if splits.data_hash != loaded["data_hash"]:
            raise ValueError("sampling manifest differs from the DiT training data contract")
        if args.valid_index < 0 or args.valid_index >= len(splits.valid):
            raise IndexError("validation clip index is out of range")
        record = dict(splits.valid[args.valid_index])
        prefix = _prefix(args.prefix, history=args.history, atoms=int(record["x"].shape[1]))
        # Erase every reference coordinate before constructing the generation
        # template. Sampling never receives a true future coordinate array.
        scaffold = np.concatenate(
            (prefix, np.repeat(prefix[-1:], 16 - args.history, axis=0)), axis=0
        )
        record["x"] = scaffold
        record["bpos"] = scaffold.copy()
        template = collate_clip_records([record])
        if args.rollout_seed:
            if args.history != 8:
                raise ValueError("rollout requires H8")
            prediction, metadata = rollout(
                loaded["codec"], loaded["model"], loaded["adapter"], loaded["statistics"],
                template=template, prefix_coordinates=torch.from_numpy(prefix),
                seeds=tuple(args.rollout_seed), steps=args.steps,
                codec_hash=loaded["codec_hash"], data_hash=loaded["data_hash"],
                source_mode=loaded["source_mode"], center_kind=loaded["center_kind"],
            )
            intervals = template.delta_time_ps[0].cpu().numpy()
            if not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-3):
                raise ValueError("rollout export requires a regular physical time grid")
            delta = float(intervals[0])
            time_ps = float(template.time_ps[0, 0]) + np.arange(prediction.shape[0], dtype=np.float32) * delta
        else:
            prediction, metadata = sample_clip(
                loaded["codec"], loaded["model"], loaded["adapter"], loaded["statistics"],
                template=template, prefix_coordinates=torch.from_numpy(prefix),
                history_frames=args.history, steps=args.steps, seed=args.seed,
                codec_hash=loaded["codec_hash"], data_hash=loaded["data_hash"],
                source_mode=loaded["source_mode"], center_kind=loaded["center_kind"],
            )
            time_ps = template.time_ps[0].cpu().numpy()
        write_trajectory(args.output, prediction.detach().cpu().numpy(), time_ps)
        atomic_write_json(
            args.output.with_suffix(".json"),
            {
                "checkpoint_step": loaded["step"],
                "sample_id": str(record["sample_id"]),
                "history_frames": args.history,
                "generation": metadata,
                "future_condition": "observed_prefix_only",
            },
        )
        return 0
    finally:
        splits.close()


if __name__ == "__main__":
    raise SystemExit(main())
