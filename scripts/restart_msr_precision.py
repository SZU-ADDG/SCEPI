import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source-pid", type=int, required=True)
parser.add_argument("--controller-pid", type=int, required=True)
args = parser.parse_args()
proof = json.loads((ROOT / "runs/msr_target_batch_diagnostic/verification.json").read_text())
assert proof["complete"] and proof["maximum_reload_difference"] < 1e-5
assert proof["training_precision_restored"]
archive = ROOT / "runs/msr_precision_before_fp32"
assert not archive.exists()
for pid, expected in [(args.controller_pid, "scripts/run_remaining_sequence.py"),
                      (args.source_pid, "scripts/train_msr.py")]:
    if not psutil.pid_exists(pid):
        continue
    process = psutil.Process(pid)
    assert process.username() == psutil.Process().username()
    assert Path(process.cwd()).is_relative_to(ROOT) and expected in process.cmdline()
    assert not process.children()
    process.terminate()
    process.wait(timeout=300)
archive.mkdir()
paths = [
    ("runs/msr_family/family30/fold0/lr0.0002_wd0_seed11", "source_formal"),
    ("runs/msr_family_smoke/family30/fold0/lr0.0002_wd0_seed11", "source_smoke"),
    ("runs/msr_target_smoke", "target_smoke"),
]
for relative, name in paths:
    (ROOT / relative).rename(archive / name)
(archive / "archive.json").write_text(json.dumps({
    "complete": True, "reason": "BF16 predictions depended on batch size; FP32 evaluation passed independent reload.",
    "formal_results_included": False, "prior_paths": dict(paths),
    "verification": "runs/msr_target_batch_diagnostic/verification.json",
}, indent=2))
status = ROOT / "runs/msr_precision_pipeline.json"
completed = []


def execute(name, command, gpu="4"):
    status.write_text(json.dumps({"state": "running", "stage": name, "completed_stages": completed}, indent=2))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, USE_TF="0")
    with (ROOT / "runs" / (name + ".log")).open("w") as stream:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    completed.append(name)


source = "runs/msr_family_smoke/family30/fold0/lr0.0002_wd0_seed11"
execute("msr_source_fp32_smoke", ["scripts/train_msr.py", "--fold", "0", "--smoke"])
execute("msr_source_fp32_verification", ["scripts/evaluate_msr_family.py", "--run", source,
                                      "--partition", "validation", "--verify-smoke"])
for mode, condition in [("S", "S32"), ("C", "C128")]:
    execute("msr_target_" + mode + "_fp32_smoke", [
        "scripts/run_msr_targets.py", "--mode", mode, "--folds", "0", "--seeds", "11",
        "--conditions", condition, "--rank", "0", "--world-size", "8", "--source-run", source, "--smoke"])
execute("msr_target_fp32_verification", ["scripts/verify_msr_targets.py", "--run-root", "runs/msr_target_smoke"])
target_proof = json.loads((ROOT / "runs/msr_target_smoke/verification/manifest.json").read_text())
assert target_proof["complete"] and set(target_proof["modes"]) == {"S", "C"}
assert target_proof["verified_fits"] >= 2 and target_proof["maximum_prediction_difference"] < 1e-5
execute("msr_source_initial_fp32", ["scripts/train_msr.py", "--fold", "0"])
execute("remaining_sequence_msr_source_fp32", ["scripts/run_remaining_sequence.py", "--stage", "msr_source",
                                             "--initial-msr-pid", str(os.getpid())])
for threshold, mode in [(30, "S"), (30, "C"), (25, "S")]:
    name = f"msr_target_{mode}_family{threshold}"
    execute(name + "_full", [
        "scripts/run_when_capacity.py", "--memory-gib", "120", "--gpu-memory-gib", "22",
        "--status", "runs/" + name + "_capacity.json",
        sys.executable, "scripts/launch_eight.py", "--log-name", name,
        "scripts/run_msr_targets.py", "--mode", mode, "--threshold", str(threshold)])
status.write_text(json.dumps({"complete": True, "completed_stages": completed}, indent=2))
