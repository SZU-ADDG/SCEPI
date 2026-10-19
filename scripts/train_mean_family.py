import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.mean_family import MeanFamilyCache, fit
from scripts.train_single import family_permissions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mean_ridge", "mean_mlp"], required=True)
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-penalty", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    suffix = (f"a{args.ridge_penalty:g}" if args.method == "mean_ridge"
              else f"lr{args.learning_rate:g}_w{args.width}_wd{args.weight_decay:g}")
    name = suffix + f"_seed{args.seed}"
    output = ROOT / "runs" / ("mean_family_smoke" if args.smoke else "mean_family")
    directory = output / f"family{args.threshold}" / f"fold{args.fold}" / args.method / name
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "manifest.json").exists()
    permissions = family_permissions(args.threshold, args.fold)
    (directory / "permissions.json").write_text(json.dumps(permissions, indent=2))
    (directory / "config.json").write_text(json.dumps(vars(args), indent=2))
    started = time.monotonic()
    cache = MeanFamilyCache()
    details = fit(cache, permissions, directory, args.method, args.seed, args.learning_rate, args.width,
                  args.weight_decay, args.ridge_penalty, args.smoke)
    manifest = {"complete": True, "smoke": args.smoke, "config": vars(args), **details,
                "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint": str((directory / "best.pt").relative_to(ROOT)),
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "elapsed_seconds": time.monotonic() - started}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
