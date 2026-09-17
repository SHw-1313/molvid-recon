"""Run the clean PVB_origin train.py on the streaming clip adapter."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()

    ref_repo = Path(args.ref_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(output_dir / "wandb")
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    sys.path.insert(0, str(ref_repo))
    sys.path.insert(0, str(output_dir))
    os.chdir(ref_repo)
    from streaming_pair_dataset import StreamingAdjacentPairDataset

    import data as original_data
    from data.collate import collate_fn as original_collate_fn
    import utils.random_seed as random_seed

    # This is the only loader substitution.  train.py, DynamicTrainer, and
    # dyVAE._train are then executed from the untouched reference checkout.
    StreamingAdjacentPairDataset.collate_fn = staticmethod(original_collate_fn)
    original_data.UniDataset = StreamingAdjacentPairDataset
    random_seed.SEED = args.seed

    sys.argv = [
        str(ref_repo / "train.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--gpus",
        args.gpus,
    ]
    runpy.run_path(str(ref_repo / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
