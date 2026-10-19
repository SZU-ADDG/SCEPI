import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs


def main():
    cache = ROOT / "data/features/stabilityarc/tsuboyama/manifest.json"
    while not cache.exists():
        state = json.loads((ROOT / "runs/stabilityarc_cache_20261013_capacity.json").read_text())
        assert state["state"] != "failed", state
        active = any(any(argument.endswith("scripts/cache_stabilityarc.py") for argument in (process.info["cmdline"] or []))
                     for process in psutil.process_iter(["cmdline"]))
        assert active or cache.exists(), "Frozen-feature extraction exited without its manifest"
        time.sleep(30)
    assert json.loads(cache.read_text())["complete"]
    subprocess.run([sys.executable, "scripts/check_stabilityarc.py"], cwd=ROOT, check=True)
    jobs = []
    for threshold, fold, seed in itertools.product([30, 25], range(5), [11, 23, 47, 89, 131]):
        arguments = ["--threshold", str(threshold), "--fold", str(fold), "--seed", str(seed)]
        source = f"runs/stabilityarc_source/family{threshold}/fold{fold}/seed{seed}"
        target = f"runs/stabilityarc_target/family{threshold}/fold{fold}/seed{seed}"
        jobs.append({"name": f"stabilityarc_family{threshold}_fold{fold}_seed{seed}", "stages": [
            {"name": "source_training", "command": [sys.executable, "scripts/train_stabilityarc.py", *arguments],
             "manifest": source + "/manifest.json", "gpu_memory_gib": 4},
            {"name": "target_single_adaptation", "command": [sys.executable, "scripts/run_stabilityarc_targets.py", *arguments],
             "manifest": target + "/manifest.json", "gpu_memory_gib": 6}]})
    run_jobs(ROOT, jobs, ROOT / "runs/stabilityarc_complete_20261013", gpu_memory_gib=4)
    subprocess.run([sys.executable, "scripts/aggregate_stabilityarc.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
