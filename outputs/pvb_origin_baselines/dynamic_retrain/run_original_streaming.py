"""Launch the unmodified PVB_origin train.py with an output-scoped loader shim."""

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

    # Import the reference package from its clean checkout.  Only the class
    # bound to data.UniDataset is replaced; all model, collate, and trainer
    # code remains the original PVB_origin implementation.
    sys.path.insert(0, str(ref_repo))
    sys.path.insert(0, str(output_dir))
    os.chdir(ref_repo)
    from streaming_pair_dataset import StreamingAdjacentPairDataset

    import data as original_data
    import utils.random_seed as random_seed

    random_seed.SEED = args.seed
    original_data.UniDataset = StreamingAdjacentPairDataset

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
