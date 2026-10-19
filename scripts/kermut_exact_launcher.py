import argparse
from pathlib import Path
import runpy
import sys

import gpytorch

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("script")
parser.add_argument("overrides", nargs=argparse.REMAINDER)
args = parser.parse_args()
script = (ROOT / args.script).resolve()
assert script.is_relative_to(ROOT / "external/kermut/src/experiments") and script.is_file()
sys.argv = [str(script), *args.overrides]
with gpytorch.settings.max_cholesky_size(8192), gpytorch.settings.fast_computations(
        covar_root_decomposition=False, log_prob=False, solves=False):
    runpy.run_path(str(script), run_name="__main__")
