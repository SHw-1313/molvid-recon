import argparse
from pathlib import Path
from molvid.evaluation.review_diagnostics import run_diagnostic

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run_diagnostic(a.output)
