"""Train U heads only; parent Molvid remains frozen and unchanged."""
import argparse
from pathlib import Path
from molvid.training.temporal_moments import train_u
from molvid.training.probe_wait import wait_for_cache

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103])
    p.add_argument('--wait-cache',action='store_true')
    a=p.parse_args()
    if a.wait_cache: wait_for_cache(a.cache,a.output)
    train_u(a.cache,a.output,steps=a.steps,seeds=tuple(a.seeds))
