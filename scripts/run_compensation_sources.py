import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["main", "matched", "delegated_grids", "geometry_inference"], required=True)
args = parser.parse_args()
output = ROOT / "runs/compensation_main"
output.mkdir(parents=True, exist_ok=True)
status = output / ("source_" + args.stage + ".json")
completed = []


def wait_for(path):
    while not path.exists():
        status.write_text(json.dumps({"state": "waiting_for_dependency", "dependency": str(path.relative_to(ROOT)),
                                      "completed_stages": completed}, indent=2))
        time.sleep(30)
    result = json.loads(path.read_text())
    assert result["complete"] and not result.get("smoke", False)
    return result


def execute(name, script, options):
    status.write_text(json.dumps({"state": "waiting_or_running", "stage": name, "completed_stages": completed}, indent=2))
    with (output / (name + ".log")).open("a") as stream:
        subprocess.run([
            sys.executable, "scripts/run_when_capacity.py", "--memory-gib", "60", "--gpu-memory-gib", "2",
            "--status", "runs/compensation_main/" + name + "_capacity.json",
            sys.executable, script, *options,
        ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    completed.append(name)


if args.stage == "main":
    grid = wait_for(ROOT / "runs/e3/pair_grid_full_g0_f1_j1/manifest.json")
    assert 30 in grid["config"]["thresholds"] and grid["trials"] == 60 * len(grid["config"]["thresholds"])
    execute("interaction_flexible_joint_family30_repeats", "scripts/run_family_repeats.py",
            ["--stage", "pair", "--thresholds", "30", "--flexible", "--joint"])
    execute("interaction_flexible_joint_both_family_grid", "scripts/run_pair_grid.py", ["--flexible", "--joint"])
    execute("interaction_flexible_joint_family25_repeats", "scripts/run_family_repeats.py",
            ["--stage", "pair", "--thresholds", "25", "--flexible", "--joint"])
elif args.stage in ["matched", "delegated_grids"]:
    delegated = [
        ("full_g0_f1_j0", ["--flexible"]),
        ("full_g0_f0_j1", ["--joint"]),
    ]
    matched = [
        ("no_cross_g0_f1_j1", ["--representation", "no_cross", "--flexible", "--joint"]),
        ("wt_g0_f1_j1", ["--representation", "wt", "--flexible", "--joint"]),
        ("identity_g0_f1_j1", ["--representation", "identity", "--flexible", "--joint"]),
        ("full_g0_f1_j1_residual", ["--flexible", "--joint", "--objective", "residual"]),
        ("full_g0_f1_j1_phenotype", ["--flexible", "--joint", "--objective", "phenotype"]),
        ("full_g1_f1_j1", ["--geometry", "--flexible", "--joint"]),
        ("full_g1_f1_j1_geopermuted", ["--geometry", "--flexible", "--joint", "--geometry-intervention", "permuted"]),
        ("full_g1_f1_j1_georewired", ["--geometry", "--flexible", "--joint", "--geometry-intervention", "rewired"]),
    ]
    for tag, options in matched if args.stage == "matched" else delegated:
        execute("grid_" + tag, "scripts/run_pair_grid.py", options)
        information = wait_for(ROOT / "runs/e3" / ("pair_grid_" + tag) / "manifest.json")
        assert information["trials"] == 120 and set(information["config"]["thresholds"]) == {25, 30}
        if args.stage == "matched":
            execute("repeats_" + tag, "scripts/run_family_repeats.py", ["--stage", "pair", *options])
else:
    base = ROOT / "runs/e3/pair_repeats_full_g1_f1_j1"
    wait_for(base / "manifest.json")
    selected = json.loads((base / "selected_hyperparameters.json").read_text())
    jobs = []
    for key, entry in sorted(selected.items()):
        chosen = ROOT / entry["run"]
        for seed in [11, 23, 47, 89, 131]:
            source = chosen.with_name(chosen.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
            information = wait_for(source / "manifest.json")
            assert information["config"]["geometry"] and information["config"]["flexible"] and information["config"]["joint"]
            for intervention in ["permuted", "rewired"]:
                relative = str(source.relative_to(ROOT))
                jobs.append({"name": key + f"/seed{seed}/" + intervention, "stages": [{
                    "name": "evaluation_test_" + intervention,
                    "command": [sys.executable, "scripts/evaluate_family.py", "--run", relative,
                                "--partition", "test", "--geometry-intervention", intervention],
                    "manifest": relative + "/evaluation_test_" + intervention + "/manifest.json",
                }]})
    assert len(jobs) == 100
    run_jobs(ROOT, jobs, ROOT / "runs/e3/compensation_geometry_interventions")
status.write_text(json.dumps({"complete": True, "completed_stages": completed}, indent=2))
