"""Build observed-only A1 features and separate real-MD targets for U/K."""
import argparse
from pathlib import Path
from molvid.training.predictive_data import build_cache

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--windows',type=int,default=2)
    args=p.parse_args()
    build_cache(args.output,windows=args.windows)
