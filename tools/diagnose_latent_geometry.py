"""CLI for current-A1 train-only representation diagnosis."""
import argparse
from pathlib import Path
from molvid.evaluation.latent_geometry import diagnose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    diagnose(args.cache,args.output)
