import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mochi_targets import tasks_for


def task_directory(base, dataset, job):
    return ROOT / "runs" / base / dataset / "order2" / job["protein_id"] / f"seed{job['seed']}" / job["scheme"] / job["condition"]


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-host-memory-gib", type=float, default=64)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--policy")
    parser.add_argument("--admission-interval-seconds", type=float, default=0)
    args = parser.parse_args()
    assert args.workers > 0 and args.minimum_host_memory_gib > 0
    bounded = args.policy is not None
    policy = json.loads((ROOT / args.policy).read_text()) if bounded else None
    fixed_default = bounded and policy["penalty_candidates"] == 1
    family = "rfa_fixed" if fixed_default else "rfa_bounded" if bounded else "rfa_accelerated"
    output = ROOT / "runs" / (family + "_20261018" if bounded else "rfa_acceleration_20261018")
    output.mkdir(parents=True, exist_ok=True)
    validation = json.loads((output / "validation.json").read_text())
    assert validation["complete"]
    snapshot = json.loads((ROOT / "runs/rfa_acceleration_20261018/preserved_processes.json").read_text())
    original_directories = {part for item in snapshot for part in item["cmdline"] if part.startswith(str(ROOT / "runs"))}
    plan = output / "tasks.json"
    tasks = json.loads(plan.read_text())["tasks"] if plan.exists() else []
    if bounded and not plan.exists():
        prior_plan = "runs/rfa_bounded_20261018/tasks.json" if fixed_default else "runs/rfa_acceleration_20261018/tasks.json"
        tasks = json.loads((ROOT / prior_plan).read_text())["tasks"]
    if not tasks:
        for dataset, mode in [("functional_doubles", "C"), ("flip_primary", "flip")]:
            for index, job in enumerate(tasks_for(dataset, mode)):
                original = task_directory("rfa", dataset, job)
                accelerated = task_directory("rfa_accelerated", dataset, job)
                if (original / "manifest.json").exists() or (accelerated / "manifest.json").exists():
                    continue
                tasks.append({"dataset": dataset, "mode": mode, "index": index,
                              "protein": job["protein_id"], "seed": job["seed"], "scheme": job["scheme"], "condition": job["condition"],
                              "original": str(original.relative_to(ROOT)), "accelerated": str(accelerated.relative_to(ROOT)),
                              "already_running_original": str(original) in original_directories})
    dimensions = {}
    for task in tasks:
        key = (task["dataset"], task["protein"])
        if key not in dimensions:
            singles = pd.read_parquet(ROOT / "data/features/higher" / key[0] / key[1] / "single/metadata.parquet", columns=["positions"])
            sizes = singles.groupby(singles.positions.map(lambda value: int(value[0]))).size().to_numpy() + 1
            dimensions[key] = int(1 + sizes.sum() + (sizes.sum() ** 2 - (sizes ** 2).sum()) // 2)
        task["estimated_coefficients"] = dimensions[key]
    if bounded:
        task_inputs = output / "task_inputs"
        task_inputs.mkdir(exist_ok=True)
        by_dataset = {}
        for task in tasks:
            path = ROOT / task["task_file"] if fixed_default else task_inputs / f"{task['dataset']}_task{task['index']}.json"
            task["task_file"] = str(path.relative_to(ROOT))
            if path.exists():
                payload = json.loads(path.read_text())
                assert (payload["dataset"], payload["mode"], payload["index"]) == (task["dataset"], task["mode"], task["index"])
                continue
            if task["dataset"] not in by_dataset:
                by_dataset[task["dataset"]] = tasks_for(task["dataset"], task["mode"])
            job = by_dataset[task["dataset"]][task["index"]]
            assert (job["protein_id"], job["seed"], job["scheme"], job["condition"]) == (task["protein"], task["seed"], task["scheme"], task["condition"])
            write_json(path, {"dataset": task["dataset"], "mode": task["mode"], "index": task["index"], "job": job})
    queue, finished, active = [], [], []
    for task in tasks:
        task["destination"] = str((ROOT / "runs" / family / Path(task["accelerated"]).relative_to("runs/rfa_accelerated")).relative_to(ROOT))
        if bounded:
            task["already_running_original"] = False
        original = ROOT / task["original"] / "manifest.json"
        accelerated = ROOT / task["accelerated"] / "manifest.json"
        destination = ROOT / task["destination"] / "manifest.json"
        retained = ROOT / "runs/rfa_bounded" / Path(task["accelerated"]).relative_to("runs/rfa_accelerated") / "manifest.json"
        if original.exists() or accelerated.exists() or destination.exists() or (fixed_default and retained.exists()):
            manifest = original if original.exists() else accelerated if accelerated.exists() else destination if destination.exists() else retained
            assert json.loads(manifest.read_text())["complete"]
            finished.append({**task, "state": "completed_by_original" if manifest == original else "complete",
                             "manifest": str(manifest.relative_to(ROOT))})
        else:
            queue.append(task)
    queue.sort(key=lambda task: (task["already_running_original"],
                                 fixed_default and not (ROOT / "runs/rfa_bounded" / Path(task["accelerated"]).relative_to("runs/rfa_accelerated") / "trial_models/trial1.rds").exists(),
                                 task["estimated_coefficients"], task["dataset"], task["index"]))
    write_json(plan, {"complete": True, "tasks": tasks, "total": len(tasks),
                                      "original_jobs_preserved": len(snapshot), "config": vars(args)})
    if args.prepare_only:
        print(json.dumps({"tasks": len(queue), "completed": len(finished), "dimensions": {"/".join(key): value for key, value in dimensions.items()},
                          "original_jobs_preserved": len(snapshot)}), flush=True)
        return
    for process in psutil.process_iter(["pid", "cwd", "cmdline", "create_time", "status"], ad_value=None):
        command = process.info["cmdline"]
        if process.info["cwd"] != str(ROOT) or not command or "scripts/run_mochi_targets.py" not in command or "--rfa-accelerated" not in command:
            continue
        if bounded != ("--rfa-policy" in command):
            continue
        if bounded and command[command.index("--rfa-policy") + 1] != args.policy:
            continue
        dataset = command[command.index("--dataset") + 1]
        index = int(command[command.index("--task-index") + 1])
        selected = [task for task in queue if task["dataset"] == dataset and task["index"] == index]
        assert len(selected) == 1, (process.pid, selected)
        task = selected[0]
        queue.remove(task)
        active.append({"task": task, "process": process, "stream": None, "adopted": True,
                       "log": str((output / f"{dataset}_task{index}.log").relative_to(ROOT)),
                       "started": process.info["create_time"]})
    assert len(active) <= args.workers
    started = time.time()
    last_launch = 0
    failures = []
    while queue or active:
        remaining = []
        live = {process.pid: process.info for process in psutil.process_iter(["create_time", "status"], ad_value=None)}
        for entry in active:
            process = entry["process"]
            adopted = entry.get("adopted", False)
            running = process.pid in live and live[process.pid]["status"] != psutil.STATUS_ZOMBIE if adopted else process.poll() is None
            if running:
                if adopted:
                    assert live[process.pid]["create_time"] == entry["started"]
                remaining.append(entry)
                continue
            if not adopted:
                entry["stream"].close()
                if process.returncode != 0:
                    failures.append({**entry["task"], "returncode": process.returncode, "log": entry["log"]})
                    print(json.dumps({"state": "failed", **failures[-1]}), flush=True)
                    continue
            task = entry["task"]
            manifest = ROOT / task["destination"] / "manifest.json"
            information = json.loads(manifest.read_text())
            assert information["complete"] and information["accelerated"]
            finished.append({**task, "state": "complete", "elapsed_seconds": time.time() - entry["started"]})
            print(json.dumps(finished[-1]), flush=True)
        active = remaining
        while queue and len(active) < args.workers and psutil.virtual_memory().available / 1024 ** 3 >= args.minimum_host_memory_gib and time.time() - last_launch >= args.admission_interval_seconds:
            task = queue.pop(0)
            original = ROOT / task["original"] / "manifest.json"
            if original.exists():
                assert json.loads(original.read_text())["complete"]
                finished.append({**task, "state": "completed_by_original"})
                continue
            command = [sys.executable, "scripts/run_mochi_targets.py", "--dataset", task["dataset"],
                       "--mode", task["mode"], "--order", "2", "--method", "rfa", "--device", "cpu",
                       "--rank", "0", "--world-size", "1", "--task-index", str(task["index"]), "--rfa-accelerated"]
            if bounded:
                command += ["--rfa-policy", args.policy, "--task-file", task["task_file"]]
            log = output / f"{task['dataset']}_task{task['index']}.log"
            stream = log.open("a")
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            last_launch = time.time()
            active.append({"task": task, "process": process, "stream": stream, "log": str(log.relative_to(ROOT)), "started": time.time()})
            print(json.dumps({"state": "started", "pid": process.pid, **task}), flush=True)
        state = {"complete": not queue and not active and not failures, "started": started, "elapsed_seconds": time.time() - started,
                 "completed": len(finished), "queued": len(queue), "active": len(active), "finished": finished,
                 "failures": failures,
                 "processes": [{**entry["task"], "pid": entry["process"].pid, "log": entry["log"],
                                "elapsed_seconds": time.time() - entry["started"]} for entry in active],
                 "available_host_memory_gib": psutil.virtual_memory().available / 1024 ** 3,
                 "validation": str((output / "validation.json").relative_to(ROOT)), "config": vars(args)}
        write_json(output / "current.json", state)
        if not queue and not active:
            write_json(output / "manifest.json", state)
            assert not failures, failures
        else:
            time.sleep(min(15, args.admission_interval_seconds) if args.admission_interval_seconds else 15)


if __name__ == "__main__":
    main()
