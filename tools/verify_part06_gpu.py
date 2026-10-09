"""Run GPU phases sequentially; retain one read-only loaded model in-process."""
from pathlib import Path
import argparse
import importlib
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'script'/'part06'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','integration'],default='all')
    args=parser.parse_args()
    scripts=['34_model_benchmark','33_profile_model']
    if args.phase=='all':scripts=['28_environment','31_verify_and_benchmark','32_selection',*scripts]
    # Individual mains retain their own CLI defaults, rather than receiving this flag.
    sys.argv=[sys.argv[0]]
    for name in scripts:
        print('RUN',name,flush=True)
        importlib.import_module(name).main()


if __name__=='__main__':main()
