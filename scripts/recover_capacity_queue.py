import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
parser.add_argument("--mean-flip", action="store_true")
args = parser.parse_args()
output = ROOT / "runs/compensation_main"
output.mkdir(parents=True, exist_ok=True)

if args.mean_flip:
    for method in ["mean_ridge", "mean_mlp"]:
        name = "mean_flip_" + ("ridge" if method == "mean_ridge" else "mlp")
        with (ROOT / "runs" / (name + "_full.log")).open("a") as stream:
            subprocess.run([
                sys.executable, "scripts/run_when_capacity.py", "--memory-gib", "60",
                "--gpu-memory-gib", "2", "--status", "runs/" + name + "_capacity.json",
                sys.executable, "scripts/launch_eight.py", "--log-name", name,
                "scripts/run_mean_flip.py", "--method", method,
            ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    sys.exit(0)

locks = json.loads(subprocess.check_output([
    "lslocks", "-J", "-o", "PID,COMMAND,TYPE,MODE,PATH,BLOCKER",
], text=True))["locks"]
plans = []
owners = set()
for item in locks:
    if item["type"] != "FLOCK" or item["mode"] != "WRITE*" or item["blocker"] is None:
        continue
    process = psutil.Process(item["pid"])
    command = process.cmdline()
    if "scripts/run_when_capacity.py" not in command or Path(process.cwd()) != ROOT:
        continue
    assert process.username() == psutil.Process().username() and not process.children()
    blocker = psutil.Process(item["blocker"])
    assert "scripts/run_when_capacity.py" in blocker.cmdline() and Path(blocker.cwd()) == ROOT
    assert str(ROOT / "runs/capacity_admission.lock") in [entry.path for entry in blocker.open_files()]
    assert blocker.children()
    owners.add(blocker.pid)
    parent = process.parent()
    parent_command = parent.cmdline()
    controller = parent_command[0].endswith("python") and any(
        part.startswith("scripts/run_remaining_") for part in parent_command
    )
    if controller:
        assert parent.children() == [process]
        resume = [sys.executable, *parent_command[1:]]
    elif "scripts/run_mean_flip.py" in command:
        resume = [sys.executable, "scripts/recover_capacity_queue.py", "--mean-flip"]
    else:
        resume = [sys.executable, *command[1:]]
    plans.append({"waiting_pid": process.pid, "controller_pid": parent.pid if controller else None,
                  "resume": resume, "blocker_pid": blocker.pid})

assert plans and len({tuple(plan["resume"]) for plan in plans}) == len(plans)
record = {"apply": args.apply, "waiting_stages": plans, "preserved_running_capacity_owners": sorted(owners),
          "new_admission_lock": "runs/capacity_checks.lock"}
(output / "queue_recovery_plan.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
if not args.apply:
    sys.exit(0)

probe = subprocess.run(["flock", "--nonblock", str(ROOT / "runs/capacity_checks.lock"), "true"])
assert probe.returncode == 0
for plan in plans:
    process = psutil.Process(plan["waiting_pid"])
    assert not process.children()
    process.terminate()
    process.wait(timeout=10)
    if plan["controller_pid"] is not None:
        parent = psutil.Process(plan["controller_pid"])
        parent.wait(timeout=10)
    name = "queue_resume_" + str(plan["waiting_pid"])
    with (output / (name + ".log")).open("a") as stream:
        replacement = subprocess.Popen(plan["resume"], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
    plan["resumed_pid"] = replacement.pid
assert all(psutil.pid_exists(pid) for pid in owners)
record.update(complete=True)
(output / "queue_recovery.json").write_text(json.dumps(record, indent=2))
print(json.dumps({"complete": True, "resumed_waiting_stages": len(plans),
                  "running_experiments_preserved": True}), flush=True)
