"""N4 bounded existing-reference metadata and sampling audit."""
import argparse
from molvid.evaluation.reference_qualification import qualify_references

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=4000);a=p.parse_args()
    qualify_references(a.output,a.frames)
