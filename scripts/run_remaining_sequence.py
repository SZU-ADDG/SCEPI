import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["mean_targets", "msr_source"], required=True)
parser.add_argument("--initial-msr-pid", type=int)
args = parser.parse_args()
status = ROOT / "runs" / ("remaining_sequence_" + args.stage + ".json")
records = []


def wait_for(path, name, pid=None):
    while not path.exists():
        if pid is not None:
            assert psutil.pid_exists(pid), f"Initial training process {pid} ended without {path}"
        status.write_text(json.dumps({"state": "waiting_for_dependency", "stage": name,
                                      "dependency": str(path.relative_to(ROOT)), "completed_stages": records}, indent=2))
        time.sleep(30)
    information = json.loads(path.read_text())
    assert information["complete"] and not information.get("smoke", False)


def execute(command, name, memory, gpu_memory=0):
    with (ROOT / "runs" / (name + ".log")).open("w") as stream:
        subprocess.run([sys.executable, "scripts/run_when_capacity.py", "--memory-gib", str(memory),
                        "--gpu-memory-gib", str(gpu_memory), "--status", "runs/" + name + "_capacity.json",
                        sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    records.append(name)
    status.write_text(json.dumps({"state": "stage_complete", "completed_stages": records}, indent=2))


if args.stage == "mean_targets":
    proof = json.loads((ROOT / "runs/mean_family_target_smoke/verification/manifest.json").read_text())
    assert proof["complete"] and set(proof["methods"]) == {"mean_ridge", "mean_mlp"} and set(proof["modes"]) == {"S", "C"}
    for method in ["mean_ridge", "mean_mlp"]:
        wait_for(ROOT / "runs/mean_family" / f"repeats_{method}_thresholds30_25_folds0_1_2_3_4/manifest.json",
                 method + "_source_repeats")
        for threshold, mode in [(30, "S"), (30, "C"), (25, "S")]:
            name = f"mean_family_target_{method}_{mode}_family{threshold}"
            execute(["scripts/launch_eight.py", "--log-name", name, "scripts/run_mean_family_targets.py",
                     "--method", method, "--mode", mode, "--threshold", str(threshold)], name, 60)
else:
    assert args.initial_msr_pid is not None
    wait_for(ROOT / "runs/msr_family/family30/fold0/lr0.0002_wd0_seed11/manifest.json",
             "initial_msr_source_training", args.initial_msr_pid)
    for stage in ["grid", "repeats"]:
        execute(["scripts/run_msr_grid.py", "--stage", stage], "msr_family_" + stage, 120, 22)
status.write_text(json.dumps({"complete": True, "completed_stages": records}, indent=2))
