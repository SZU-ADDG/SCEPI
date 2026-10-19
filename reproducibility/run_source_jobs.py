import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_stage(stage):
    destination = ROOT / stage["manifest"]
    if destination.exists():
        result = json.loads(destination.read_text())
        assert result["complete"] and not result.get("smoke", False), destination
        print(json.dumps({"stage": stage["name"], "state": "already_complete", "manifest": stage["manifest"]}), flush=True)
        return
    command = [sys.executable, *stage["arguments"]]
    print(json.dumps({"stage": stage["name"], "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    assert json.loads(destination.read_text())["complete"], destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--arm", choices=["interaction", "direct_residual", "no_pair_measured"])
    parser.add_argument("--threshold", type=int, choices=[30, 25])
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=[11, 23, 47, 89, 131])
    args = parser.parse_args()
    jobs = json.loads((ROOT / "reproducibility/source_jobs.json").read_text())
    assert len(jobs) == 150
    if args.list:
        print(json.dumps([{key: row[key] for key in ["arm", "threshold", "fold", "seed", "run"]} for row in jobs], indent=2))
        return
    assert all(value is not None for value in [args.arm, args.threshold, args.fold, args.seed])
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    selected = [row for row in jobs if row["arm"] == args.arm and row["threshold"] == args.threshold
                and row["fold"] == args.fold and row["seed"] == args.seed]
    assert len(selected) == 1
    for stage in selected[0]["stages"]:
        run_stage(stage)


if __name__ == "__main__":
    main()
