import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from epistasis.evaluation_scope import authorized_jobs, job_name
from run_mochi_targets import tasks_for

OUTPUT = ROOT / "runs/prediction_completion_20261016"
PYTHON = "/data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python"
MATRICES = ["original_main", "compensation_main", "sall_full_target"]
SOURCE = "runs/msr_family/family30/fold0/lr0.0002_wd0_seed11"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def project_processes():
    return [p.info for p in psutil.process_iter(["pid", "cmdline", "cwd"], ad_value=None)
            if p.info["cwd"] == str(ROOT) and p.info["cmdline"]]


def check_msr_policy():
    information = json.loads((ROOT / SOURCE / "manifest.json").read_text())
    assert information["checkpoint_ready"] and information["best_epoch"] == 9
    assert information["user_requested_stop"] and not information["training_complete"]
    assert (ROOT / information["checkpoint"]).is_file()
    completion = json.loads((ROOT / "runs/msr_family/execution_policy_20261015.json").read_text())
    assert completion["complete"] and completion["selected_round"] == 10
    assert not any(any(script in p["cmdline"] for script in
                       ["scripts/train_msr.py", "scripts/run_msr_grid.py", "scripts/run_msr_targets.py"])
                   for p in project_processes())


def prepare():
    assert not (OUTPUT / "plan.json").exists()
    check_msr_policy()
    states = {name: json.loads((ROOT / "runs" / name / "current.json").read_text()) for name in MATRICES}
    missing = {name: {path for record in state["waiting_dependencies"] for path in record["dependencies"]
                     if not (ROOT / path).is_file()} for name, state in states.items()}
    policy = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "user_instruction": "Use the round-10 ESM-MSR checkpoint without further training; resume other missing predictions and retain running experiments.",
              "esm_msr": {"further_training_allowed": False, "selected_round": 10,
                          "source_manifest": SOURCE + "/manifest.json",
                          "completed_evaluation_policy": "runs/msr_family/execution_policy_20261015.json",
                          "scope": "family30/fold0/seed11; existing source validation, source test and S-all target evaluations only",
                          "discontinuation_reason": "Other source folds/repeats and new target fits require additional training. Reusing this checkpoint across source folds would violate held-out family permissions."},
              "matrices": {}}
    for name in MATRICES:
        jobs = json.loads((ROOT / "runs" / name / "jobs.json").read_text())
        waiting = {record.get("job", "/".join(str(record.get(key)) for key in ["arm", "threshold", "seed", "fold", "condition"]))
                   for record in states[name]["waiting_dependencies"]}
        discontinued = [job_name(job) for job in jobs
                        if job["arm"] in ["esm_msr", "esm_msr_C"] and job_name(job) in waiting]
        policy["matrices"][name] = {"original_expected_evaluations": len(jobs),
                                    "discontinued_job_names": discontinued,
                                    "authorized_evaluations": len(jobs) - len(discontinued)}
    pending = missing["original_main"]
    kermut = {path for path in pending if path.startswith("runs/kermut_target/C/")}
    proteins = pd.read_parquet(ROOT / "data/processed/tsuboyama/proteins.parquet")
    doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet", columns=["protein_id"])
    protein_ids = proteins.loc[proteins.protein_id.isin(doubles.protein_id)].sort_values("protein_id").protein_id.tolist()
    tasks = []
    for rank in sorted({protein_ids.index(Path(path).parts[3]) % 8 for path in kermut}):
        expected = sorted(path for path in kermut if protein_ids.index(Path(path).parts[3]) % 8 == rank)
        tasks.append({"name": f"kermut_rank{rank}", "method": "kermut", "gpu_memory_gib": 24,
                      "command": [PYTHON, "scripts/run_kermut_targets.py", "--mode", "C", "--conditions", "Call",
                                  "--rank", str(rank), "--world-size", "8"], "expected_manifests": expected})
    mavenn = {path for path in pending if path.startswith("runs/mavenn/functional_doubles/order2/")}
    covered = set()
    for index, job in enumerate(tasks_for("functional_doubles", "C")):
        directory = Path("runs/mavenn/functional_doubles/order2") / job["protein_id"] / f"seed{job['seed']}" / job["scheme"] / job["condition"]
        manifest = str(directory / "manifest.json")
        if manifest not in mavenn:
            continue
        tasks.append({"name": f"mavenn_task{index:04d}", "method": "mavenn", "gpu_memory_gib": 12,
                      "command": [PYTHON, "scripts/run_mochi_targets.py", "--method", "mavenn", "--dataset", "functional_doubles",
                                  "--mode", "C", "--order", "2", "--device", "cuda", "--rank", "0", "--world-size", "1",
                                  "--task-index", str(index)], "expected_manifests": [manifest]})
        covered.add(manifest)
    assert covered == mavenn
    assert all(Path(path).parts[-2] == "Call" for path in kermut)
    assert not any("scripts/run_kermut_targets.py" in p["cmdline"] or
                   ("scripts/run_mochi_targets.py" in p["cmdline"] and "mavenn" in p["cmdline"])
                   for p in project_processes())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(ROOT / "runs/evaluation_scope.json", policy)
    plan = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "run_path": str(OUTPUT), "tasks": tasks,
            "missing_kermut_predictions": len(kermut), "missing_mavenn_predictions": len(mavenn),
            "preserved_processes": project_processes(), "scope_policy": "runs/evaluation_scope.json"}
    save(OUTPUT / "plan.json", plan)
    print(json.dumps({"prepared": True, "tasks": len(tasks), "missing_kermut": len(kermut),
                      "missing_mavenn": len(mavenn), "discontinued": {key: len(value["discontinued_job_names"])
                                                                       for key, value in policy["matrices"].items()}}), flush=True)


def verify():
    check_msr_policy()
    plan = json.loads((OUTPUT / "plan.json").read_text())
    policy = json.loads((ROOT / "runs/evaluation_scope.json").read_text())
    coverage = {}
    for name in MATRICES:
        all_jobs = json.loads((ROOT / "runs" / name / "jobs.json").read_text())
        active, discontinued = authorized_jobs(ROOT, name, all_jobs)
        assert all(job["arm"] in ["esm_msr", "esm_msr_C"] for job in all_jobs if job_name(job) in discontinued)
        assert len(active) == policy["matrices"][name]["authorized_evaluations"]
        coverage[name] = {"original": len(all_jobs), "authorized": len(active), "discontinued": len(discontinued)}
    completed = Counter()
    remaining = Counter()
    for task in plan["tasks"]:
        assert task["command"][0] == PYTHON
        assert not any("msr" in argument for argument in task["command"])
        for relative in task["expected_manifests"]:
            path = ROOT / relative
            if path.exists():
                assert json.loads(path.read_text())["complete"]
                completed[task["method"]] += 1
            else:
                remaining[task["method"]] += 1
    live = {item["pid"]: item for item in project_processes()}
    rfa_running, rfa_completed = 0, 0
    for item in plan["preserved_processes"]:
        if not any(argument.endswith("/scripts/fit_rfa.R") for argument in item["cmdline"]):
            continue
        if item["pid"] in live:
            assert item["cmdline"] == live[item["pid"]]["cmdline"]
            rfa_running += 1
        else:
            manifest = Path(item["cmdline"][-2]) / "manifest.json"
            assert json.loads(manifest.read_text())["complete"]
            rfa_completed += 1
    reused = ROOT / "runs/mavenn/functional_doubles/order2/manifest_task0_rank0.json"
    reuse = json.loads(reused.read_text())
    assert reuse["complete"] and len(reuse["outputs"]) == 1
    original = json.loads((ROOT / reuse["outputs"][0]).read_text())
    assert original["complete"] and original["trials"] == 12
    rfa_coverage = json.loads((OUTPUT / "rfa_coverage.json").read_text())
    assert not rfa_coverage["unclaimed"] and not rfa_coverage["duplicate_claims"]
    assert rfa_coverage["missing_predictions"] == rfa_coverage["claimed_missing_predictions"]
    progress = {}
    for name in MATRICES:
        current = json.loads((ROOT / "runs" / name / "current.json").read_text())
        previous = json.loads((OUTPUT / (name + ".before_scope.json")).read_text())
        progress[name] = {key: current[key] for key in
                          ["expected_evaluations", "authorized_evaluations", "discontinued_evaluations",
                           "completed_evaluations", "waiting_evaluations", "active_scope_complete"]}
        progress[name]["new_completed_evaluations"] = current["completed_evaluations"] - previous["completed_evaluations"]
    return {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": plan["source_commit"], "run_path": str(OUTPUT),
            "working_tree_modified": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
            "msr_checkpoint_round": 10, "msr_further_training": False, "scope": coverage,
            "existing_mavenn_task_reused": reuse["outputs"][0],
            "preserved_rfa_processes_running": rfa_running, "preserved_rfa_processes_completed": rfa_completed,
            "rfa_missing_predictions_covered": rfa_coverage["claimed_missing_predictions"],
            "rfa_coverage_manifest": "runs/prediction_completion_20261016/rfa_coverage.json",
            "evaluation_progress": progress,
            "completed_predictions": dict(completed), "remaining_predictions": dict(remaining)}


def restart_evaluators():
    check_msr_policy()
    assert not (OUTPUT / "evaluator_processes.json").exists()
    scripts = {"original_main": "scripts/run_original_evaluation.py",
               "compensation_main": "scripts/run_compensation_evaluation.py",
               "sall_full_target": "scripts/run_sall_full_target.py"}
    previous = {}
    processes = project_processes()
    for name, script in scripts.items():
        matches = [p for p in processes if script in p["cmdline"]]
        assert len(matches) == 1, (name, matches)
        process = psutil.Process(matches[0]["pid"])
        state = json.loads((ROOT / "runs" / name / "current.json").read_text())
        assert state["ready_evaluations"] == 0 and not process.children(recursive=True), name
        previous[name] = process
        save(OUTPUT / (name + ".before_scope.json"), state)
    records = {}
    for name, script in scripts.items():
        process = previous[name]
        process.terminate()
        assert process.wait(timeout=10) in [-15, None]
        command = [PYTHON, script] + (["--workers", "8"] if name == "sall_full_target" else [])
        with (OUTPUT / (name + ".log")).open("x") as stream:
            replacement = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records[name] = {"previous_waiting_controller_pid": process.pid, "pid": replacement.pid,
                         "command": command, "experiment_processes_interrupted": 0}
    save(OUTPUT / "evaluator_processes.json", records)
    print(json.dumps(records), flush=True)


def handoff_reporting():
    check_msr_policy()
    target = OUTPUT / "evaluator_reporting_handoff.json"
    assert not target.exists()
    definitions = {"original_main": ("scripts/run_original_evaluation.py", "scripts/aggregate_original_main.py"),
                   "compensation_main": ("scripts/run_compensation_evaluation.py", "scripts/aggregate_compensation_main.py")}
    records = {}
    processes = project_processes()
    for name, (script, aggregation) in definitions.items():
        matches = [item for item in processes if script in item["cmdline"]]
        assert len(matches) == 1
        controller = psutil.Process(matches[0]["pid"])
        children = controller.children()
        assert len(children) == 1 and aggregation in children[0].cmdline()
        child = children[0]
        created = child.create_time()
        controller.terminate()
        controller.wait(timeout=10)
        assert child.is_running() and child.create_time() == created
        with (OUTPUT / (name + ".async.log")).open("x") as stream:
            replacement = subprocess.Popen([PYTHON, script], cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records[name] = {"previous_waiting_controller_pid": controller.pid, "pid": replacement.pid,
                         "preserved_aggregation_pid": child.pid, "experiments_interrupted": 0}
    save(target, records)
    print(json.dumps(records), flush=True)


def run():
    assert str(Path(sys.executable).resolve()) == str(Path(PYTHON).resolve())
    assert not (OUTPUT / "status.json").exists()
    check_msr_policy()
    plan = json.loads((OUTPUT / "plan.json").read_text())
    queue = deque(plan["tasks"])
    state = {"state": "restoring_predictions", "controller_pid": os.getpid(), "scope_policy": "runs/evaluation_scope.json",
             "gpu_policy": "One recovery process per GPU; admit by available VRAM and RAM.",
             "tasks": {task["name"]: {"state": "queued", "method": task["method"]} for task in queue},
             "further_msr_training_allowed": False}
    running = {}
    while queue or running:
        for gpu, (task, process, stream) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            record = state["tasks"][task["name"]]
            record.update(state="complete" if code == 0 else "failed", returncode=code)
            if code == 0:
                assert all(json.loads((ROOT / path).read_text())["complete"] for path in task["expected_manifests"])
            del running[gpu]
            print(json.dumps({"task": task["name"], "state": record["state"], "returncode": code}), flush=True)
        for gpu in range(8):
            if gpu in running or not queue:
                continue
            task = queue.popleft()
            manifests = [ROOT / path for path in task["expected_manifests"]]
            if all(path.is_file() for path in manifests):
                assert all(json.loads(path.read_text())["complete"] for path in manifests)
                state["tasks"][task["name"]].update(state="complete", reused_completed_results=True)
                continue
            for path in manifests:
                if not path.exists() and path.parent.exists():
                    archive = OUTPUT / "preserved_partial" / path.parent.relative_to(ROOT)
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    assert not archive.exists()
                    path.parent.rename(archive)
            command = [PYTHON, "scripts/run_when_capacity.py", "--memory-gib", "16", "--gpu-memory-gib", str(task["gpu_memory_gib"]),
                       "--gpus", str(gpu), "--status", str((OUTPUT / (task["name"] + ".capacity.json")).relative_to(ROOT)), *task["command"]]
            log = OUTPUT / (task["name"] + ".log")
            stream = log.open("x")
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                       stdout=stream, stderr=subprocess.STDOUT)
            running[gpu] = (task, process, stream)
            state["tasks"][task["name"]].update(state="running_or_waiting_for_memory", pid=process.pid, gpu=gpu,
                                                  log=str(log.relative_to(ROOT)), command=command)
        state.update(updated_at_utc=datetime.now(timezone.utc).isoformat(), queued_tasks=len(queue), running_tasks=len(running))
        save(OUTPUT / "status.json", state)
        if running:
            time.sleep(5)
    failures = [name for name, record in state["tasks"].items() if record["state"] == "failed"]
    state.update(state="failed" if failures else "waiting_for_remaining_evaluations", failed_tasks=failures)
    save(OUTPUT / "status.json", state)
    assert not failures, failures
    while True:
        current = {name: json.loads((ROOT / "runs" / name / "current.json").read_text()) for name in MATRICES}
        state.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
                     evaluations={name: {key: value for key, value in item.items() if key != "waiting_dependencies"}
                                  for name, item in current.items()})
        if all(item["active_scope_complete"] for item in current.values()):
            state.update(state="complete", complete=True, verification=verify())
            save(OUTPUT / "status.json", state)
            print(json.dumps({"complete": True, "scope": "authorized experiments; discontinued ESM-MSR fits remain explicitly excluded"}), flush=True)
            return
        save(OUTPUT / "status.json", state)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "restart-evaluators", "handoff-reporting", "run", "verify"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "restart-evaluators":
        restart_evaluators()
    elif args.action == "handoff-reporting":
        handoff_reporting()
    elif args.action == "run":
        run()
    else:
        verification = verify()
        save(OUTPUT / "verification.json", verification)
        print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
