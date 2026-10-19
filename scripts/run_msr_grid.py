import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs


def training_job(threshold, fold, seed, learning_rate, weight_decay, evaluate=False):
    name = f"lr{learning_rate:g}_wd{weight_decay:g}_seed{seed}"
    run = Path("runs/msr_family") / f"family{threshold}" / f"fold{fold}" / name
    command = [sys.executable, "scripts/train_msr.py", "--threshold", str(threshold),
               "--fold", str(fold), "--seed", str(seed), "--learning-rate", str(learning_rate),
               "--weight-decay", str(weight_decay)]
    stages = [{"name": "training", "command": command, "manifest": str(run / "manifest.json")}]
    if evaluate:
        for partition in ["validation", "test"]:
            stages.append({"name": "evaluation_" + partition,
                           "command": [sys.executable, "scripts/evaluate_msr_family.py", "--run", str(run), "--partition", partition],
                           "manifest": str(run / ("evaluation_" + partition) / "manifest.json")})
    return {"name": str(run), "threshold": threshold, "fold": fold, "seed": seed,
            "learning_rate": learning_rate, "weight_decay": weight_decay, "run": str(run), "stages": stages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["grid", "repeats"], required=True)
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[25, 30], default=[30, 25])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    grid = [training_job(threshold, fold, 11, rate, decay)
            for threshold, fold, rate, decay in itertools.product(
                args.thresholds, args.folds, [5e-5, 1e-4, 2e-4], [0.0, 1e-4, 1e-3, 0.01])]
    selected = {}
    if args.stage == "grid":
        jobs = grid
    else:
        for threshold, fold in itertools.product(args.thresholds, args.folds):
            candidates = []
            for job in grid:
                if (job["threshold"], job["fold"]) != (threshold, fold):
                    continue
                information = json.loads((ROOT / job["run"] / "manifest.json").read_text())
                assert information["complete"] and not information["smoke"]
                candidates.append({**job, "validation_score": information["validation_score"]})
            assert len(candidates) == 12
            selected[f"family{threshold}/fold{fold}"] = min(
                candidates, key=lambda value: (value["validation_score"], value["run"]))
        jobs = [training_job(item["threshold"], item["fold"], seed, item["learning_rate"], item["weight_decay"], evaluate=True)
                for item in selected.values() for seed in [11, 23, 47, 89, 131]]
    suffix = "_thresholds" + "_".join(map(str, args.thresholds)) + "_folds" + "_".join(map(str, args.folds))
    output = ROOT / "runs/msr_family" / (args.stage + suffix)
    output.mkdir(parents=True, exist_ok=True)
    if selected:
        (output / "selected_hyperparameters.json").write_text(json.dumps(selected, indent=2))
    run_jobs(ROOT, jobs, output)


if __name__ == "__main__":
    main()
