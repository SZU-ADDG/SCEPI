import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/original_main"


def execute(name, script, options, gpus):
    directory = OUTPUT / "sources" / name
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "scripts/run_when_capacity.py", "--memory-gib", "60",
               "--gpu-memory-gib", "2", "--gpus", *[str(gpu) for gpu in gpus],
               "--status", str((directory / "capacity.json").relative_to(ROOT)),
               sys.executable, script, *options]
    environment = dict(os.environ)
    if len(gpus) == 1:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    with (directory / "process.log").open("a") as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    information = {"complete": True, "command": command, "gpus": gpus}
    (directory / "manifest.json").write_text(json.dumps(information, indent=2))
    return information


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["grids", "transfer_recovery"], required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = []
    status = OUTPUT / ("source_" + args.stage + ".json")
    if args.stage == "grids":
        experiments = [
            ("full_g0_f0_j0_residual", ["--objective", "residual"]),
            ("full_g0_f0_j0_phenotype", ["--objective", "phenotype"]),
            ("full_g1_f0_j0_geopermuted", ["--geometry", "--geometry-intervention", "permuted"]),
            ("full_g1_f0_j0_georewired", ["--geometry", "--geometry-intervention", "rewired"]),
        ]
        for tag, options in experiments:
            status.write_text(json.dumps({"state": "waiting_or_running", "stage": tag,
                                          "completed_stages": completed}, indent=2))
            execute("grid_" + tag, "scripts/run_pair_grid.py", options, list(range(8)))
            information = json.loads((ROOT / "runs/e3" / ("pair_grid_" + tag) / "manifest.json").read_text())
            assert information["complete"] and information["trials"] == 120
            assert set(information["config"]["thresholds"]) == {25, 30}
            completed.append(tag)
    else:
        prior = ROOT / "runs/shared_targets_all_full"
        statuses = json.loads((prior / "exit_codes.json").read_text())
        records = json.loads((prior / "processes.json").read_text())
        assert statuses == [0, 1, 0, 0, 0, 0, 0, 1]
        assert not any(psutil.pid_exists(entry["pid"]) for entry in records)
        pattern = "manifest_unseen_pair_random_two_pair_secondary_C128_C512_C2048_Call_seeds11_23_47_89_131_rank{}.json"
        for rank in [0, 2, 3, 4, 5, 6]:
            information = json.loads((ROOT / "runs/shared_target" / pattern.format(rank)).read_text())
            assert information["complete"] and not information["smoke"]
        status.write_text(json.dumps({"state": "waiting_or_running", "recovered_ranks": [1, 7],
                                      "unchanged_world_size": 8}, indent=2))

        def recover(rank):
            result = execute(f"shared_transfer_rank{rank}", "scripts/run_shared_targets.py",
                             ["--rank", str(rank), "--world-size", "8"], [rank])
            information = json.loads((ROOT / "runs/shared_target" / pattern.format(rank)).read_text())
            assert information["complete"] and not information["smoke"]
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = list(pool.map(recover, [1, 7]))
    status.write_text(json.dumps({"complete": True, "stage": args.stage,
                                  "completed_stages": completed}, indent=2))


if __name__ == "__main__":
    main()
