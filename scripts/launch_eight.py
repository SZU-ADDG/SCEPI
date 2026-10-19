import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--log-name")
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.gpus and len(set(args.gpus)) == len(args.gpus) and set(args.gpus) <= set(range(8))
    name = args.log_name or Path(args.script).stem
    for flag in ["--dataset", "--stage", "--mode", "--scope"]:
        if flag in args.arguments:
            name += "_" + args.arguments[args.arguments.index(flag) + 1]
    logdir = ROOT / "runs" / (name + ("_smoke" if "--smoke" in args.arguments else "_full"))
    logdir.mkdir(parents=True, exist_ok=True)
    pending = iter(range(8))
    running = {}
    processes = []
    handles = []
    statuses = [None] * 8
    failed = False

    def launch(rank, gpu):
        handle = (logdir / f"gpu{gpu}_rank{rank}.log").open("w")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        command = [sys.executable, args.script, "--rank", str(rank), "--world-size", "8", *args.arguments]
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        handles.append(handle)
        running[process] = (rank, gpu, handle)
        processes.append({"gpu": gpu, "rank": rank, "pid": process.pid})
        (logdir / "processes.json").write_text(json.dumps(processes, indent=2))

    for gpu in args.gpus:
        try:
            launch(next(pending), gpu)
        except StopIteration:
            break

    while running:
        for process, (rank, gpu, _) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            statuses[rank] = code
            del running[process]
            if code != 0:
                failed = True
        if failed:
            for process in running:
                process.terminate()
            for process, (rank, _, _) in list(running.items()):
                statuses[rank] = process.wait()
                del running[process]
            break
        for gpu in args.gpus:
            if any(active_gpu == gpu for _, active_gpu, _ in running.values()):
                continue
            try:
                launch(next(pending), gpu)
            except StopIteration:
                break
        if running:
            time.sleep(2)

    for handle in handles:
        handle.close()
    (logdir / "exit_codes.json").write_text(json.dumps(statuses))
    assert statuses == [0] * 8, statuses


if __name__ == "__main__":
    main()
