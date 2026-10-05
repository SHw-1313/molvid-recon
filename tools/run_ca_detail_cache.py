import argparse
from pathlib import Path
from molvid.training.ca_detail_data import build_ca_cache

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-systems',type=int,default=32);a=p.parse_args();build_ca_cache(a.output,a.train_systems)
