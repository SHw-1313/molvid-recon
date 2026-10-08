"""Train paired reference-coordinate flow or EDM from an explicit recipe."""
import argparse
import json
from pathlib import Path
from molvid.training.coordinate_state import train


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    initialization=p.add_mutually_exclusive_group()
    initialization.add_argument('--resume',type=Path,help='Exact continuation including optimizer, RNG and data visits')
    initialization.add_argument('--warm-start',type=Path,help='Compatible fixed-final coordinate weights only; new stage and optimizer')
    a=p.parse_args()
    train(json.loads(a.config.read_text()),a.output,a.seed,a.resume,a.warm_start)
