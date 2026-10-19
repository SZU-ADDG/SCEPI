from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import psutil

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python"


def resume(pid, expected, command):
    if pid and psutil.pid_exists(pid):
        process = psutil.Process(pid)
        assert any(argument.endswith(expected) for argument in process.cmdline())
        print({"waiting_for_existing_process": pid, "script": expected}, flush=True)
        psutil.wait_procs([process])
    print({"resuming": command}, flush=True)
    subprocess.run([PYTHON, *command], cwd=ROOT, check=True)


def main():
    jobs = [(205923, "paper_full_target_evidence.py", ["scripts/paper_full_target_evidence.py", "--workers", "8"]),
            (150377, "run_sall_full_target.py", ["scripts/run_sall_full_target.py", "--workers", "8"]),
            (0, "", ["scripts/paper_full_target_evidence.py", "--workers", "8", "--thresholds", "30",
                     "--summary-name", "family30_snapshot"])]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(resume, *job) for job in jobs]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
