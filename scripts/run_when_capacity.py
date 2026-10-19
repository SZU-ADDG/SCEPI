import argparse
import csv
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--memory-gib", type=float, required=True)
parser.add_argument("--gpu-memory-gib", type=float, default=0)
parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
parser.add_argument("--status", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
args.gpus = [int(gpu) for gpu in args.gpus.split(",")]
assert args.memory_gib > 0 and args.gpu_memory_gib >= 0
assert args.gpus and len(set(args.gpus)) == len(args.gpus) and set(args.gpus) <= set(range(8))
assert args.command and args.command[0] == sys.executable
status = (ROOT / args.status).resolve()
assert status.is_relative_to(ROOT)
status.parent.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
admission = (ROOT / "runs/capacity_checks.lock").open("a")
while True:
    fcntl.flock(admission, fcntl.LOCK_EX)
    memory = psutil.virtual_memory().available / 1024 ** 3
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    gpus = {int(row[0]): float(row[1]) / 1024 for row in csv.reader(io.StringIO(output))}
    assert set(gpus) == set(range(8))
    ready = memory >= args.memory_gib and min(gpus[gpu] for gpu in args.gpus) >= args.gpu_memory_gib
    record = {"state": "running" if ready else "waiting_for_memory",
              "command": args.command, "minimum_host_memory_gib": args.memory_gib,
              "minimum_each_gpu_memory_gib": args.gpu_memory_gib,
              "admission_gpus": args.gpus,
              "available_host_memory_gib": memory, "available_gpu_memory_gib": gpus,
              "waited_seconds": time.monotonic() - started}
    status.write_text(json.dumps(record, indent=2))
    if ready:
        print(json.dumps(record), flush=True)
        fcntl.flock(admission, fcntl.LOCK_UN)
        result = subprocess.run(args.command, cwd=ROOT)
        record.update(state="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        status.write_text(json.dumps(record, indent=2))
        assert result.returncode == 0, result.returncode
        break
    fcntl.flock(admission, fcntl.LOCK_UN)
    time.sleep(30)
