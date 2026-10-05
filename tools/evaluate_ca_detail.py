import argparse
from pathlib import Path
from molvid.evaluation.ca_detail import evaluate_detail

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate_detail(a.cache,a.training_root,a.output)
