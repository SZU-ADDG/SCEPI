import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["gpu", "controller"], required=True)
args = parser.parse_args()
status = ROOT / "runs/remaining_kermut.json"
if args.stage == "gpu":
    processes, streams = [], []
    output = ROOT / "runs/kermut_singles_remaining"
    output.mkdir(parents=True, exist_ok=True)
    for rank in range(1, 8):
        stream = (output / f"gpu{rank}.log").open("w")
        command = [sys.executable, "scripts/reproduce_kermut.py", "--scope", "singles",
                   "--rank", str(rank), "--world-size", "8"]
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(rank)},
                                   stdout=stream, stderr=subprocess.STDOUT)
        streams.append(stream)
        processes.append(process)
    (output / "processes.json").write_text(json.dumps([{"rank": rank, "gpu": rank, "pid": process.pid}
                                                      for rank, process in zip(range(1, 8), processes)], indent=2))
    statuses = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    (output / "exit_codes.json").write_text(json.dumps(statuses))
    assert statuses == [0] * 7, statuses
    manifests = [ROOT / "runs/e1_kermut/singles" / f"manifest_rank{rank}.json" for rank in range(1, 8)]
    assert all(json.loads(path.read_text())["complete"] for path in manifests)
    (output / "manifest.json").write_text(json.dumps({"complete": True, "ranks": list(range(1, 8)),
                                                       "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}, indent=2))
else:
    stages = [
        ("kermut_singles_remaining", 120, 2, ["scripts/run_remaining_kermut.py", "--stage", "gpu"]),
        ("kermut_original_cpu", 240, 0, ["scripts/reproduce_kermut.py", "--scope", "cpu", "--rank", "0", "--world-size", "1"]),
    ]
    completed = []
    for name, memory, gpu_memory, command in stages:
        status.write_text(json.dumps({"state": "waiting_or_running", "stage": name, "completed_stages": completed}, indent=2))
        with (ROOT / "runs" / (name + ".log")).open("w") as stream:
            subprocess.run([sys.executable, "scripts/run_when_capacity.py", "--memory-gib", str(memory),
                            "--gpu-memory-gib", str(gpu_memory), "--status", "runs/" + name + "_capacity.json",
                            sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        completed.append(name)
    rank_zero = ROOT / "runs/e1_kermut/singles/manifest_rank0.json"
    while not rank_zero.exists():
        status.write_text(json.dumps({"state": "waiting_for_dependency", "dependency": str(rank_zero.relative_to(ROOT)),
                                      "completed_stages": completed}, indent=2))
        time.sleep(30)
    assert json.loads(rank_zero.read_text())["complete"]
    status.write_text(json.dumps({"complete": True, "completed_stages": completed,
                                  "original_single_protocol_workers": 8, "original_cpu_workers": 1}, indent=2))
