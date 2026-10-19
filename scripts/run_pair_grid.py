import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[25, 30], default=[30, 25])
    parser.add_argument("--representation", choices=["full", "no_cross", "wt", "identity"], default="full")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--flexible", action="store_true")
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--geometry-intervention", choices=["none", "permuted", "rewired"], default="none")
    parser.add_argument("--objective", choices=["combined", "residual", "phenotype"], default="combined")
    args = parser.parse_args()
    return args


def grid_tag(args):
    control = (f"_geo{args.geometry_intervention}" if args.geometry_intervention != "none" else "") + ("_n1" if args.no_interaction else "")
    tag = f"{args.representation}_g{int(args.geometry)}_f{int(args.flexible)}_j{int(args.joint)}{control}"
    return tag + ("_" + args.objective if args.objective != "combined" else "")


def run_grid(args):
    control_tag = (f"_geo{args.geometry_intervention}" if args.geometry_intervention != "none" else "") + ("_n1" if args.no_interaction else "")
    jobs = []
    for threshold in args.thresholds:
        for fold in range(5):
            for projection in [64, 128]:
                for learning_rate in [1e-4, 3e-4, 1e-3]:
                    for weight in [0.1, 1.0]:
                        name = f"{args.representation}_p{projection}_lr{learning_rate:g}_ly{weight:g}_g{int(args.geometry)}_f{int(args.flexible)}_j{int(args.joint)}{control_tag}_{args.objective}_seed11"
                        run = ROOT / "runs/e3" / f"family{threshold}" / f"fold{fold}" / "pair" / name
                        command = [
                            sys.executable, "scripts/train_pair.py", "--threshold", str(threshold),
                            "--fold", str(fold), "--projection", str(projection), "--learning-rate", str(learning_rate),
                            "--lambda-y", str(weight), "--seed", "11", "--representation", args.representation,
                            "--objective", args.objective, "--geometry-intervention", args.geometry_intervention,
                        ]
                        for option in ["geometry", "flexible", "joint", "no_interaction"]:
                            if getattr(args, option):
                                command.append("--" + option.replace("_", "-"))
                        jobs.append({"threshold": threshold, "fold": fold, "name": name, "run": str(run.relative_to(ROOT)), "command": command})
    tag = grid_tag(args)
    destination = ROOT / "runs/e3" / f"pair_grid_{tag}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "jobs.json").write_text(json.dumps(jobs, indent=2))
    pending = deque(jobs)
    lock = threading.Lock()
    stop = threading.Event()
    started = time.monotonic()

    def worker(gpu):
        completed = []
        while not stop.is_set():
            with lock:
                if not pending:
                    return completed
                job = pending.popleft()
            run = ROOT / job["run"]
            run.mkdir(parents=True, exist_ok=True)
            manifest_path = run / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                assert manifest["complete"] and not manifest["smoke"]
                completed.append({"run": job["run"], "state": "already_complete", "validation_score": manifest["validation_score"]})
                continue
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(json.dumps({"state": "running", "gpu": gpu, "run": job["run"]}), flush=True)
            with (run / "process.log").open("w") as stream:
                result = subprocess.run(job["command"], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                stop.set()
                raise RuntimeError(f"Pair-effect training failed with code {result.returncode}: {job['run']}")
            manifest = json.loads(manifest_path.read_text())
            completed.append({"run": job["run"], "state": "complete", "gpu": gpu, "validation_score": manifest["validation_score"]})
            print(json.dumps(completed[-1]), flush=True)
        return completed

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(8)))
    flattened = [job for worker_results in results for job in worker_results]
    assert len(flattened) == len(jobs)
    selected = {}
    for threshold in args.thresholds:
        for fold in range(5):
            candidates = [item for item in flattened if f"family{threshold}/fold{fold}/" in item["run"]]
            selected[f"family{threshold}/fold{fold}"] = min(candidates, key=lambda item: (item["validation_score"], item["run"]))
    manifest = {
        "complete": True, "trials": len(jobs), "gpus": list(range(8)), "config": vars(args),
        "elapsed_seconds": time.monotonic() - started, "results": flattened, "selected": selected,
        "selection_scope": "Source validation families only. No held-out test labels are evaluated by this grid.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"complete": True, "trials": len(jobs), "selected": selected}, indent=2))


def main():
    args = arguments()
    directory = ROOT / "runs/e3" / ("pair_grid_" + grid_tag(args))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_grid(args)


if __name__ == "__main__":
    main()
