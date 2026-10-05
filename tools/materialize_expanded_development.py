import argparse
from pathlib import Path
from molvid.data.expanded_development import materialize

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    materialize(args.manifest,args.output)
