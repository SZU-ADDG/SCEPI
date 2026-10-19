import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs


def training_job(method, threshold, fold, seed, settings, evaluate=False):
    suffix = (f"a{settings['ridge_penalty']:g}" if method == "mean_ridge"
              else f"lr{settings['learning_rate']:g}_w{settings['width']}_wd{settings['weight_decay']:g}")
    run = Path("runs/mean_family") / f"family{threshold}" / f"fold{fold}" / method / (suffix + f"_seed{seed}")
    command = [sys.executable, "scripts/train_mean_family.py", "--method", method, "--threshold", str(threshold),
               "--fold", str(fold), "--seed", str(seed)]
    for key, value in settings.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    stages = [{"name": "training", "command": command, "manifest": str(run / "manifest.json")}]
    if evaluate:
        for partition in ["validation", "test"]:
            stages.append({"name": "evaluation_" + partition,
                           "command": [sys.executable, "scripts/evaluate_mean_family.py", "--run", str(run), "--partition", partition],
                           "manifest": str(run / ("evaluation_" + partition) / "manifest.json")})
    return {"name": str(run), "method": method, "threshold": threshold, "fold": fold, "seed": seed,
            "settings": settings, "run": str(run), "stages": stages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mean_ridge", "mean_mlp"], required=True)
    parser.add_argument("--stage", choices=["grid", "repeats"], required=True)
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[25, 30], default=[30, 25])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    if args.method == "mean_ridge":
        settings = [{"ridge_penalty": float(value)} for value in np.logspace(-6, 5, 12)]
    else:
        settings = [{"learning_rate": rate, "width": width, "weight_decay": decay}
                    for rate, width, decay in itertools.product([1e-4, 3e-4, 1e-3], [128, 256], [1e-4, 0.01])]
    grid = [training_job(args.method, threshold, fold, 11, setting)
            for threshold, fold, setting in itertools.product(args.thresholds, args.folds, settings)]
    selected = {}
    if args.stage == "grid":
        jobs = grid
    else:
        for threshold, fold in itertools.product(args.thresholds, args.folds):
            candidates = []
            for job in grid:
                if (job["threshold"], job["fold"]) != (threshold, fold):
                    continue
                manifest = json.loads((ROOT / job["run"] / "manifest.json").read_text())
                assert manifest["complete"] and not manifest["smoke"]
                candidates.append({**job, "validation_score": manifest["validation_score"]})
            assert len(candidates) == 12
            selected[f"family{threshold}/fold{fold}"] = min(
                candidates, key=lambda value: (value["validation_score"], value["run"]))
        jobs = [training_job(args.method, item["threshold"], item["fold"], seed, item["settings"], evaluate=True)
                for item in selected.values() for seed in [11, 23, 47, 89, 131]]
    suffix = "_thresholds" + "_".join(map(str, args.thresholds)) + "_folds" + "_".join(map(str, args.folds))
    output = ROOT / "runs/mean_family" / (args.stage + "_" + args.method + suffix)
    output.mkdir(parents=True, exist_ok=True)
    if selected:
        (output / "selected_hyperparameters.json").write_text(json.dumps(selected, indent=2))
    run_jobs(ROOT, jobs, output)


if __name__ == "__main__":
    main()
