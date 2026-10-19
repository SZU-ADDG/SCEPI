import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu-memory-gib", type=float, required=True)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "processes.json").exists()
    processes = []
    for rank in range(8):
        command = [sys.executable, "scripts/run_when_capacity.py", "--memory-gib", str(args.memory_gib),
                   "--gpu-memory-gib", str(args.gpu_memory_gib), "--gpus", str(rank),
                   "--status", str((output / f"capacity_rank{rank}.json").relative_to(ROOT)),
                   sys.executable, args.script, "--rank", str(rank), "--world-size", "8", *args.arguments]
        with (output / f"rank{rank}.log").open("w") as stream:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((rank, process))
    (output / "processes.json").write_text(json.dumps([{"rank": rank, "gpu": rank, "pid": process.pid}
                                                       for rank, process in processes], indent=2))
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda item: item[1].wait(), processes))
    (output / "exit_codes.json").write_text(json.dumps(codes))
    assert codes == [0] * 8, codes


if __name__ == "__main__":
    main()
