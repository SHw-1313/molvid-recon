"""Train the registered current-coordinate geometry intervention."""
import argparse
from pathlib import Path
from molvid.training.current_recovery import train_recovery

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['static','mixed'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=2048)
    a=p.parse_args();train_recovery(a.cache,a.output,arm=a.arm,seed=a.seed,steps=a.steps)
