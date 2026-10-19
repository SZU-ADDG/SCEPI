import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/environment"
OUTPUT.mkdir(parents=True, exist_ok=True)
packages = ["torch", "transformers", "gpytorch", "linear_operator", "esm", "esm-msr", "pymochi",
            "mavenn", "tensorflow", "keras", "numpy", "pandas", "scipy", "biopython", "h5py", "pyreadr", "openpyxl"]
versions = {name: importlib.metadata.version(name) for name in packages}
sources = []
for name in ["ThermoMPNN-D", "kermut", "ProteinGym", "esm-msr", "pymochi", "mochi-v1.1",
             "esm", "RFA", "FLIP", "mutation-predictability", "ProteinMPNN", "nuclease_design", "fuzzy_specificity"]:
    path = ROOT / "external" / name
    sources.append({
        "name": name, "path": str(path.relative_to(ROOT)),
        "url": subprocess.check_output(["git", "-C", str(path), "remote", "get-url", "origin"], text=True).strip(),
        "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
    })
check = subprocess.run([sys.executable, "-m", "pip", "check"], check=True, capture_output=True, text=True)
(OUTPUT / "pip_check.txt").write_text(check.stdout)
(OUTPUT / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
(OUTPUT / "conda_packages.json").write_text(subprocess.check_output([
    "/data1/doomx/anaconda3/bin/conda", "list", "--prefix", "/data1/doomx/anaconda3/envs/recomb-3-epistasis", "--json",
], text=True))
(OUTPUT / "source_repositories.json").write_text(json.dumps(sources, indent=2))
result = {
    "project_root": str(ROOT), "environment": "/data1/doomx/anaconda3/envs/recomb-3-epistasis", "python": sys.version,
    "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "versions": versions, "pip_check": check.stdout.strip(),
    "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip().splitlines(),
}
(OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
