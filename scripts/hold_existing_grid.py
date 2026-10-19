import argparse
import fcntl
import json
from pathlib import Path
import subprocess

import psutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--tag", required=True)
args = parser.parse_args()
process = psutil.Process(args.pid)
assert "scripts/run_pair_grid.py" in process.cmdline()
assert Path(process.cwd()) == ROOT
directory = ROOT / "runs/e3" / ("pair_grid_" + args.tag)
assert directory.exists()
with (directory / "dispatch.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    information = {"state": "holding_for_existing_training", "pid": args.pid, "tag": args.tag,
                   "existing_command": process.cmdline()}
    (directory / "existing_process_lock.json").write_text(json.dumps(information, indent=2))
    print(json.dumps(information), flush=True)
    subprocess.run(["tail", "--pid", str(args.pid), "--sleep-interval", "2", "-f", "/dev/null"], check=True)
    information.update(state="complete", complete=True)
    (directory / "existing_process_lock.json").write_text(json.dumps(information, indent=2))
