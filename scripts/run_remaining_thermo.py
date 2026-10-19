import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 23, 47, 89, 131]

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["cache25", "source", "targets"], required=True)
args = parser.parse_args()
status = ROOT / "runs" / ("remaining_thermo_" + args.stage + ".json")
records = []


def wait_for(paths, name):
    while True:
        complete = [path.exists() for path in paths]
        status.write_text(json.dumps({"stage": name, "completed_dependencies": sum(complete),
                                      "expected_dependencies": len(paths), "completed_stages": records}, indent=2))
        if all(complete):
            assert all(json.loads(path.read_text())["complete"] for path in paths)
            return
        time.sleep(30)


def execute(command, name, memory):
    log = ROOT / "runs" / (name + ".log")
    with log.open("w") as stream:
        subprocess.run([sys.executable, "scripts/run_when_capacity.py", "--memory-gib", str(memory),
                        "--status", "runs/" + name + "_capacity.json", sys.executable, *command],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    records.append(name)
    status.write_text(json.dumps({"stage": name, "complete": True, "completed_stages": records}, indent=2))


def caches(threshold, fold, seeds):
    return [ROOT / "data/features/thermo/tsuboyama" / f"family{threshold}" / f"fold{fold}" /
            f"seed{seed}/augmentation/full/epistatic/manifest_rank{rank}.json"
            for seed in seeds for rank in range(8)]


if args.stage == "cache25":
    wait_for([path for fold in range(5) for path in caches(30, fold, SEEDS)], "family30_caches")
    execute(["scripts/launch_eight.py", "--log-name", "thermo_aug_family25", "scripts/cache_thermo.py",
             "--phase", "augmentation", "--mode", "epistatic", "--thresholds", "25"],
            "thermo_aug_family25", 60)
elif args.stage == "source":
    for threshold, folds in [(30, [2, 3, 4]), (25, list(range(5)))]:
        for fold in folds:
            wait_for(caches(threshold, fold, [11]), f"family{threshold}_fold{fold}_grid_inputs")
            base = ["scripts/run_thermo_grid.py", "--mode", "epistatic", "--thresholds", str(threshold), "--folds", str(fold)]
            execute([*base, "--stage", "grid"], f"thermo_epistatic_grid_family{threshold}_fold{fold}", 60)
            wait_for(caches(threshold, fold, SEEDS), f"family{threshold}_fold{fold}_repeat_inputs")
            execute([*base, "--stage", "repeats"], f"thermo_epistatic_repeats_family{threshold}_fold{fold}", 60)
else:
    for head in ["epistatic", "additive"]:
        for threshold in [30, 25]:
            for fold in range(5):
                if head == "epistatic" and threshold == 30 and fold == 0:
                    continue
                suffix = f"thresholds{threshold}_folds{fold}" if head == "epistatic" else "thresholds30_25_folds0_1_2_3_4"
                wait_for([ROOT / "runs/thermo_family" / ("repeats_" + head + "_" + suffix) / "manifest.json"],
                         f"{head}_family{threshold}_fold{fold}_source")
                for mode in ["S"] + (["C"] if threshold == 30 else []):
                    name = f"thermo_target_{head}_{mode}_family{threshold}_fold{fold}"
                    execute(["scripts/launch_eight.py", "--log-name", name, "scripts/run_thermo_targets.py",
                             "--head", head, "--mode", mode, "--threshold", str(threshold), "--folds", str(fold)],
                            name, 60)
status.write_text(json.dumps({"complete": True, "completed_stages": records}, indent=2))
