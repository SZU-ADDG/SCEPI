from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def run_jobs(root, jobs, output, use_gpu=True, gpu_memory_gib=0):
    root = Path(root).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_jobs = []
    for job in jobs:
        stages = []
        for stage in job["stages"]:
            command = list(stage["command"])
            if Path(command[0]).name in ["python", "python3", "python3.11"]:
                command[0] = sys.executable
            stages.append({**stage, "command": command})
        resolved_jobs.append({**job, "stages": stages})
    jobs = resolved_jobs
    (output / "jobs.json").write_text(json.dumps(jobs, indent=2))
    pending = deque(jobs)
    lock = threading.Lock()
    stop = threading.Event()
    started = time.monotonic()

    def worker(gpu):
        results = []
        while not stop.is_set():
            with lock:
                if not pending:
                    return results
                job = pending.popleft()
            for stage in job["stages"]:
                manifest_path = (root / stage["manifest"]).resolve()
                assert manifest_path.is_relative_to(root)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    assert manifest["complete"] and not manifest.get("smoke", False)
                    continue
                print(json.dumps({"state": "running", "job": job["name"], "stage": stage["name"], "worker": gpu, "gpu": gpu if use_gpu else None}), flush=True)
                with (manifest_path.parent / (stage["name"] + ".process.log")).open("a") as stream:
                    command = stage["command"]
                    stage_memory_gib = stage.get("gpu_memory_gib", gpu_memory_gib)
                    if use_gpu and stage_memory_gib > 0:
                        capacity_path = manifest_path.parent / (stage["name"] + ".capacity.json")
                        command = [stage["command"][0], "scripts/run_when_capacity.py", "--memory-gib", "16",
                                   "--gpu-memory-gib", str(stage_memory_gib), "--gpus", str(gpu),
                                   "--status", str(capacity_path.relative_to(root)), *stage["command"]]
                    result = subprocess.run(command, cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu) if use_gpu else ""}, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    stop.set()
                    raise RuntimeError(f"Job {job['name']} stage {stage['name']} failed: exit {result.returncode}")
                manifest = json.loads(manifest_path.read_text())
                assert manifest["complete"] and not manifest.get("smoke", False)
            results.append({"name": job["name"], "worker": gpu, "gpu": gpu if use_gpu else None, "complete": True})
            print(json.dumps(results[-1]), flush=True)
        return results

    with ThreadPoolExecutor(max_workers=8) as pool:
        grouped = list(pool.map(worker, range(8)))
    results = [entry for group in grouped for entry in group]
    assert len(results) == len(jobs)
    manifest = {
        "complete": True, "jobs": len(jobs), "results": results, "workers": 8, "gpus": list(range(8)) if use_gpu else [],
        "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
