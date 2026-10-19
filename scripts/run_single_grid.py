import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[25, 30], default=[30, 25])
    parser.add_argument("--representation", choices=["full", "no_cross", "wt", "identity"], default="full")
    args = parser.parse_args()
    jobs = []
    for threshold in args.thresholds:
        for fold in range(5):
            for projection in [64, 128]:
                for learning_rate in [1e-4, 3e-4, 1e-3]:
                    name = f"{args.representation}_p{projection}_lr{learning_rate:g}_seed11"
                    run = ROOT / "runs/e3" / f"family{threshold}" / f"fold{fold}" / "single" / name
                    command = [
                        sys.executable, "scripts/train_single.py", "--threshold", str(threshold),
                        "--fold", str(fold), "--projection", str(projection),
                        "--learning-rate", str(learning_rate), "--seed", "11", "--representation", args.representation,
                    ]
                    jobs.append({"threshold": threshold, "fold": fold, "name": name, "run": str(run.relative_to(ROOT)), "command": command})
    destination = ROOT / "runs/e3" / f"single_grid_{args.representation}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "jobs.json").write_text(json.dumps(jobs, indent=2))
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    stop = threading.Event()
    pending_lock = threading.Lock()
    started = time.monotonic()

    def worker(gpu):
        completed = []
        while not stop.is_set():
            with pending_lock:
                if pending.empty():
                    return completed
                job = pending.get_nowait()
            run = ROOT / job["run"]
            run.mkdir(parents=True, exist_ok=True)
            manifest_path = run / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                assert manifest["complete"] and not manifest["smoke"]
                completed.append({"run": job["run"], "state": "already_complete", "validation_macro_rmse": manifest["validation_macro_rmse"]})
                continue
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(json.dumps({"state": "running", "gpu": gpu, "run": job["run"]}), flush=True)
            with (run / "process.log").open("w") as stream:
                result = subprocess.run(job["command"], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                stop.set()
                raise RuntimeError(f"Single-effect training failed with code {result.returncode}: {job['run']}")
            manifest = json.loads(manifest_path.read_text())
            completed.append({"run": job["run"], "state": "complete", "gpu": gpu, "validation_macro_rmse": manifest["validation_macro_rmse"]})
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
            best = min(candidates, key=lambda item: (item["validation_macro_rmse"], item["run"]))
            selected[f"family{threshold}/fold{fold}"] = best
    manifest = {"complete": True, "trials": len(jobs), "gpus": list(range(8)), "elapsed_seconds": time.monotonic() - started, "results": flattened, "selected": selected}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"complete": True, "trials": len(jobs), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
