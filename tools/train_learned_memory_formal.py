"""Run one frozen B learned-memory training arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molvid.training.learned_memory_formal import run_learned_memory_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    if (args.resume is None) != (args.resume_sha256 is None):
        parser.error("--resume and --resume-sha256 must be provided together")
    result = run_learned_memory_training(
        args.config,
        resume=args.resume,
        resume_sha256=args.resume_sha256,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
