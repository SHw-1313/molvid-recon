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
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    train(json.loads(a.config.read_text()),a.output,a.seed,a.resume)
