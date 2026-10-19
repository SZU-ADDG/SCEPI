import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python"
SOURCE = ROOT / "runs/msr_family/family30/fold0/lr0.0002_wd0_seed11"
POLICY = ROOT / "runs/msr_family/execution_policy_20261015.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pid", type=int, default=9352)
    args = parser.parse_args()
    previous_policy = json.loads(POLICY.read_text())
    request = json.loads((SOURCE / "config.json").read_text())["request"]
    assert request == {"threshold": 30, "fold": 0, "seed": 11, "learning_rate": 2e-4,
                       "weight_decay": 0.0, "smoke": False}
    assert not psutil.pid_exists(args.source_pid), "Default source training must be stopped before evaluation"
    stopped = [234080, 234097, 234100, 234114, 234122, 234126, 237651]
    assert not any(psutil.pid_exists(pid) for pid in [233984, 4077664, *stopped])
    preserved = []
    for directory in sorted(SOURCE.parent.glob("*_seed11")):
        checkpoint = directory / "best.ckpt"
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["state_dict"]
        metrics = pd.read_csv(directory / "logs/version_0/metrics.csv")
        preserved.append({"run": str(directory.relative_to(ROOT)), "checkpoint_bytes": checkpoint.stat().st_size,
                          "checkpoint_readable": True, "latest_epoch": int(metrics.epoch.dropna().iloc[-1]),
                          "latest_step": int(metrics.step.dropna().iloc[-1])})
    assert len(preserved) == 8
    history = pd.read_csv(SOURCE / "history.csv")
    best = history.loc[history.validation_score.idxmin()]
    assert int(best.epoch) == 9, "The user selected the tenth-round validation checkpoint"
    validation = pd.read_parquet(SOURCE / "validation_metrics.parquet")
    score = float(validation.loc[validation.main_eligible, "epsilon_rmse"].mean())
    assert abs(score - float(best.validation_score)) < 1e-12
    training = pd.read_parquet(SOURCE / "training_variants.parquet")
    configuration = json.loads((SOURCE / "config.json").read_text())["author"]
    source = {"complete": True, "checkpoint_ready": True, "training_complete": False,
              "user_requested_stop": True, "termination_reason": "user_stop_use_best_validation_checkpoint",
              "smoke": False, "config": request,
              "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/esm-msr"), "rev-parse", "HEAD"], text=True).strip(),
              "training_variants": len(training), "full_training_variants": len(training), "training_fraction": 1.0,
              "epochs": len(history), "last_completed_epoch": int(history.epoch.max()),
              "last_logged_epoch": next(item["latest_epoch"] for item in preserved if item["run"] == str(SOURCE.relative_to(ROOT))),
              "best_epoch": int(best.epoch), "validation_score": score,
              "checkpoint": str((SOURCE / "best.ckpt").relative_to(ROOT)),
              "selection": "Best source-validation macro epsilon RMSE among completed epochs; user stopped further source training.",
              "target_labels_used": 0, "training_precision": configuration["precision"], "evaluation_precision": "fp32",
              "stopped_training_pid": args.source_pid, "stop_confirmed_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest = SOURCE / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        assert previous["user_requested_stop"] and previous["best_epoch"] == source["best_epoch"]
        assert previous["config"] == request and previous["validation_score"] == score
        source = previous
    else:
        manifest.write_text(json.dumps(source, indent=2))
    state = {"state": "evaluating_saved_checkpoint", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "controller_pid": os.getpid(), "source_run": str(SOURCE.relative_to(ROOT)), "retained_config": request,
             "selected_epoch": 9, "selected_round": 10, "source_training_stopped": True,
             "hyperparameter_search_enabled": False, "maximum_source_training_processes": 0,
             "stopped_controller_pids": [233984, 4077664, 3999341], "stopped_training_pids": [*stopped, args.source_pid],
             "preserved_artifacts": preserved, "completed_stages": [], "stages": {},
             "target_scope": {"threshold": 30, "fold": 0, "seed": 11, "condition": "Sall", "workers": 8},
             "full_main_matrix_complete": False}
    commands = [
        ("default_source_validation", 0, SOURCE / "evaluation_validation/manifest.json",
         ["scripts/evaluate_msr_family.py", "--run", str(SOURCE), "--partition", "validation"]),
        ("default_source_test", 1, SOURCE / "evaluation_test/manifest.json",
         ["scripts/evaluate_msr_family.py", "--run", str(SOURCE), "--partition", "test"]),
    ]
    for rank in range(8):
        commands.append((
            f"default_target_Sall_rank{rank}", rank,
            ROOT / f"runs/msr_target/family30/S/manifest_folds0_seeds11_conditionsSall_rank{rank}.json",
            ["scripts/run_msr_targets.py", "--mode", "S", "--threshold", "30", "--folds", "0", "--seeds", "11",
             "--conditions", "Sall", "--rank", str(rank), "--world-size", "8", "--source-run", str(SOURCE)]))
    running = {}
    for name, gpu, output, command in commands:
        if output.exists():
            assert json.loads(output.read_text())["complete"]
            state["completed_stages"].append(name)
            state["stages"][name] = {"state": "complete", "manifest": str(output.relative_to(ROOT))}
            continue
        capacity = ROOT / f"runs/msr_family/{name}.capacity.json"
        previous = previous_policy.get("stages", {}).get(name)
        if previous and previous["state"] == "running" and psutil.pid_exists(previous["pid"]):
            process = psutil.Process(previous["pid"])
            assert process.username() == psutil.Process().username()
            assert "scripts/run_when_capacity.py" in process.cmdline()
            assert str(capacity.relative_to(ROOT)) in process.cmdline()
            running[name] = (None, None, output, previous["pid"], process.create_time())
            state["stages"][name] = {**previous, "adopted_existing_process": True}
            continue
        admission = [PYTHON, "scripts/run_when_capacity.py", "--memory-gib", "16", "--gpu-memory-gib", "22",
                     "--gpus", str(gpu), "--status", f"runs/msr_family/{name}.capacity.json", PYTHON, *command]
        stream = (ROOT / "runs/msr_family" / (name + ".log")).open("a")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), USE_TF="0")
        process = subprocess.Popen(admission, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        running[name] = (process, stream, output, process.pid, psutil.Process(process.pid).create_time())
        state["stages"][name] = {"state": "running", "pid": process.pid, "gpu": gpu,
                                 "manifest": str(output.relative_to(ROOT))}
    POLICY.write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    while running:
        for name, (process, stream, output, pid, created) in list(running.items()):
            if process is None:
                capacity = json.loads((ROOT / f"runs/msr_family/{name}.capacity.json").read_text())
                if capacity["state"] in ["complete", "failed"]:
                    code = capacity["returncode"]
                else:
                    assert psutil.pid_exists(pid) and psutil.Process(pid).create_time() == created, name
                    continue
            else:
                code = process.poll()
            if code is None:
                continue
            if stream is not None:
                stream.close()
            state["stages"][name].update(state="complete" if code == 0 else "failed", returncode=code)
            state.update(updated_at_utc=datetime.now(timezone.utc).isoformat())
            if code != 0:
                state.update(state="failed", failed_stage=name)
                POLICY.write_text(json.dumps(state, indent=2))
                sys.exit(code)
            assert json.loads(output.read_text())["complete"], name
            state["completed_stages"].append(name)
            del running[name]
            POLICY.write_text(json.dumps(state, indent=2))
            print(json.dumps({"complete": True, "stage": name}), flush=True)
        if running:
            time.sleep(30)
    state.update(state="complete", complete=True, updated_at_utc=datetime.now(timezone.utc).isoformat())
    POLICY.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
