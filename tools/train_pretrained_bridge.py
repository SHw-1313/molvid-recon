"""Launch the actual pretrained-geometry mixed-training experiment."""
import argparse
import json
from pathlib import Path
from molvid.training.bridge_runner import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--warmstart', type=Path)
    parser.add_argument('--diagnose-step', type=int, help='Locate one actual failed update without changing its arithmetic')
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), output=args.output, seed=args.seed,
          resume=args.resume, diagnose_step=args.diagnose_step, warmstart=args.warmstart)
