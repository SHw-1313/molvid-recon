"""Controlled static gradient routing experiment on the fixed ABCD/A1 base."""
import argparse
from pathlib import Path
from molvid.training.static_response import run_experiment


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=512)
    parser.add_argument('--seed',type=int,default=20261004)
    parser.add_argument('--arms',nargs='+',choices=['S0','Sdec','Strunk'],default=['S0','Sdec','Strunk'])
    parser.add_argument('--no-evaluation',action='store_true',help='numerical smoke only, never a quality result')
    args=parser.parse_args()
    run_experiment(args.output,updates=args.updates,seed=args.seed,
                   arms=tuple(args.arms),evaluate=not args.no_evaluation)
