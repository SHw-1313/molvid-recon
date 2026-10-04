import argparse
from pathlib import Path
from molvid.evaluation.review_diagnostics import run_diagnostic

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--solver-extension',action='store_true');a=p.parse_args()
    run_diagnostic(a.output,solver_extension=a.solver_extension)
