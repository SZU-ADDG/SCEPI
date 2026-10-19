import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["controller", "interventions"], required=True)
args = parser.parse_args()
source_root = ROOT / "runs/e3/pair_repeats_full_g1_f0_j0"
status = ROOT / "runs/remaining_geometry.json"
completed = []


def execute(name, command, memory=60):
    status.write_text(json.dumps({"state": "waiting_or_running", "stage": name, "completed_stages": completed}, indent=2))
    with (ROOT / "runs" / (name + ".log")).open("w") as stream:
        subprocess.run([sys.executable, "scripts/run_when_capacity.py", "--memory-gib", str(memory),
                        "--gpu-memory-gib", "2", "--status", "runs/" + name + "_capacity.json",
                        sys.executable, *command], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    completed.append(name)


if args.stage == "interventions":
    information = json.loads((source_root / "manifest.json").read_text())
    assert information["complete"]
    selected = json.loads((source_root / "selected_hyperparameters.json").read_text())
    jobs = []
    for key, entry in sorted(selected.items()):
        chosen = ROOT / entry["run"]
        for seed in [11, 23, 47, 89, 131]:
            source = chosen.with_name(chosen.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
            source_info = json.loads((source / "manifest.json").read_text())
            assert source_info["complete"] and not source_info["smoke"] and source_info["config"]["geometry"]
            for intervention in ["permuted", "rewired"]:
                name = key + f"/seed{seed}/" + intervention
                relative = str(source.relative_to(ROOT))
                jobs.append({"name": name, "stages": [{
                    "name": "evaluation_test_" + intervention,
                    "command": [sys.executable, "scripts/evaluate_family.py", "--run", relative,
                                "--partition", "test", "--geometry-intervention", intervention],
                    "manifest": relative + "/evaluation_test_" + intervention + "/manifest.json",
                }]})
    assert len(jobs) == 100
    run_jobs(ROOT, jobs, ROOT / "runs/e3/geometry_inference_interventions")
else:
    dependency = source_root / "manifest.json"
    while not dependency.exists():
        status.write_text(json.dumps({"state": "waiting_for_dependency",
                                      "dependency": str(dependency.relative_to(ROOT))}, indent=2))
        time.sleep(30)
    assert json.loads(dependency.read_text())["complete"]
    execute("geometry_inference_interventions", ["scripts/run_geometry_followups.py", "--stage", "interventions"])
    for supervision in ["transfer", "target_only"]:
        name = "shared_target_geometry" if supervision == "transfer" else "shared_target_only_geometry"
        execute(name + "_smoke", ["scripts/run_shared_targets.py", "--rank", "0", "--world-size", "8",
                                 "--seeds", "11", "--geometry", "--supervision", supervision, "--smoke"], 20)
        with (ROOT / "runs" / (name + "_verification.log")).open("w") as stream:
            subprocess.run([sys.executable, "scripts/verify_physical_target_only.py",
                            "--run-root", "runs/" + name + "_smoke", "--supervision", supervision],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        proof = json.loads((ROOT / "runs" / (name + "_smoke/verification/manifest.json")).read_text())
        assert proof["complete"] and proof["verified_fits"] > 0 and proof["source_family_exclusion"]
        assert proof["maximum_prediction_difference"] < 1e-5
        execute(name + "_formal", ["scripts/launch_eight.py", "--log-name", name,
                                  "scripts/run_shared_targets.py", "--geometry", "--supervision", supervision])
    status.write_text(json.dumps({"complete": True, "completed_stages": completed}, indent=2))
