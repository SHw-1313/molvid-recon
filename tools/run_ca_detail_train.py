import argparse
from pathlib import Path
from molvid.training.ca_detail import train_detail

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['clean','mixed'],required=True);p.add_argument('--steps',type=int,default=512)
    p.add_argument('--seeds',nargs='+',type=int,default=[71,72]);a=p.parse_args()
    train_detail(a.cache,a.output,arm=a.arm,steps=a.steps,seeds=tuple(a.seeds))
