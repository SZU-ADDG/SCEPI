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
from epistasis.thermo_training import FEATURES, ThermoCache, fit_source
from scripts.train_single import family_permissions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["epistatic", "additive"], required=True)
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    name = f"lr{args.learning_rate:g}_wd{args.weight_decay:g}_seed{args.seed}"
    output = ROOT / "runs" / ("thermo_family_smoke" if args.smoke else "thermo_family")
    directory = output / f"family{args.threshold}" / f"fold{args.fold}" / args.mode / name
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "manifest.json").exists()
    permissions = family_permissions(args.threshold, args.fold)
    (directory / "permissions.json").write_text(json.dumps(permissions, indent=2))
    (directory / "config.json").write_text(json.dumps(vars(args), indent=2))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    cache = ThermoCache(FEATURES / "direct", args.mode)
    augmentation = None
    if args.mode == "epistatic":
        path = FEATURES / f"family{args.threshold}" / f"fold{args.fold}" / f"seed{args.seed}"
        original = json.loads((path / "augmentation_manifest.json").read_text())
        assert original["complete"] and set(original["source_training_proteins"]) == set(permissions["train"])
        augmentation = ThermoCache(path / "augmentation", args.mode, augmentation=True, smoke=args.smoke)
    started = time.monotonic()
    details = fit_source(args.mode, cache, augmentation, permissions, args.seed, args.learning_rate,
                         args.weight_decay, directory, args.smoke)
    manifest = {"complete": True, "smoke": args.smoke, "config": vars(args), **details,
                "source_commit": commit,
                "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/ThermoMPNN-D"), "rev-parse", "HEAD"], text=True).strip(),
                "pretrained_parameters": "Frozen ProteinMPNN encoder only; prediction head initialized for this source split.",
                "checkpoint": str((directory / "best.pt").relative_to(ROOT)),
                "torch": torch.__version__, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "gpu_name": torch.cuda.get_device_name(), "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "elapsed_seconds": time.monotonic() - started}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
