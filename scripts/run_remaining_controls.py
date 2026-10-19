import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["source_repeats", "no_cross_grid", "physical_target_only", "flip"], required=True)
args = parser.parse_args()
status = ROOT / "runs" / ("remaining_controls_" + args.stage + ".json")
completed = json.loads(status.read_text()).get("completed_stages", []) if status.exists() else []


def wait_for(path, allow_smoke=False):
    while not path.exists():
        status.write_text(json.dumps({"state": "waiting_for_dependency", "dependency": str(path.relative_to(ROOT)),
                                      "completed_stages": completed}, indent=2))
        time.sleep(30)
    information = json.loads(path.read_text())
    assert information["complete"] and (allow_smoke or not information.get("smoke", False))
    return information


def execute(command, name, memory, gpu_memory=2, gpus=None):
    status.write_text(json.dumps({"state": "waiting_or_running", "stage": name, "completed_stages": completed}, indent=2))
    capacity_command = [sys.executable, "scripts/run_when_capacity.py", "--memory-gib", str(memory),
                        "--gpu-memory-gib", str(gpu_memory), "--status", "runs/" + name + "_capacity.json"]
    if gpus is not None:
        capacity_command.extend(["--gpus", ",".join(str(gpu) for gpu in gpus)])
    capacity_command.extend([sys.executable, *command])
    with (ROOT / "runs" / (name + ".log")).open("w") as stream:
        subprocess.run(capacity_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    completed.append(name)


def completed_flip_mavenn_order2():
    output_root = ROOT / "runs" / "mavenn" / "flip_primary" / "order2"
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    recorded_outputs = set()
    for rank in range(8):
        path = output_root / f"manifest_rank{rank}.json"
        if not path.is_file():
            return False
        rank_manifest = json.loads(path.read_text())
        config = rank_manifest.get("config", {})
        if (not rank_manifest.get("complete") or rank_manifest.get("smoke") or
                rank_manifest.get("source_commit") != source_commit or config.get("rank") != rank or
                config.get("world_size") != 8 or config.get("method") != "mavenn" or
                config.get("dataset") != "flip_primary" or config.get("mode") != "flip" or
                config.get("order") != 2):
            return False
        for relative_path in rank_manifest.get("outputs", []):
            if relative_path in recorded_outputs:
                return False
            output_path = ROOT / relative_path
            if not output_path.is_file():
                return False
            output_manifest = json.loads(output_path.read_text())
            if (not output_manifest.get("complete") or output_manifest.get("smoke") or
                    output_manifest.get("source_commit") != source_commit or
                    output_manifest.get("method") != "mavenn_order2" or
                    output_manifest.get("dataset") != "flip_primary"):
                return False
            recorded_outputs.add(relative_path)
    existing_outputs = {path.relative_to(ROOT).as_posix()
                        for path in output_root.glob("*/seed*/*/*/manifest.json")}
    return bool(recorded_outputs) and recorded_outputs == existing_outputs


if args.stage == "no_cross_grid":
    current = wait_for(ROOT / "runs/e3/pair_grid_no_cross_g0_f0_j0/manifest.json")
    if set(current["config"]["thresholds"]) != {25, 30}:
        execute(["scripts/run_pair_grid.py", "--representation", "no_cross"], "no_cross_both_thresholds", 60)
    final = wait_for(ROOT / "runs/e3/pair_grid_no_cross_g0_f0_j0/manifest.json")
    assert set(final["config"]["thresholds"]) == {25, 30} and final["trials"] == 120
elif args.stage == "source_repeats":
    experiments = [
        ("full_g1_f0_j0", ["--geometry"]),
        ("no_cross_g0_f0_j0", ["--representation", "no_cross"]),
        ("wt_g0_f0_j0", ["--representation", "wt"]),
        ("identity_g0_f0_j0", ["--representation", "identity"]),
        ("full_g0_f0_j1_n1", ["--joint", "--no-interaction"]),
        ("full_g0_f1_j1_n1", ["--flexible", "--joint", "--no-interaction"]),
        ("full_g0_f1_j0", ["--flexible"]),
        ("full_g0_f0_j1", ["--joint"]),
        ("full_g0_f0_j0_residual", ["--objective", "residual"]),
        ("full_g0_f0_j0_phenotype", ["--objective", "phenotype"]),
        ("full_g1_f0_j0_geopermuted", ["--geometry", "--geometry-intervention", "permuted"]),
        ("full_g1_f0_j0_georewired", ["--geometry", "--geometry-intervention", "rewired"]),
    ]
    for name, options in experiments:
        if "repeats_" + name in completed:
            continue
        if name.startswith("no_cross"):
            wait_for(ROOT / "runs/remaining_controls_no_cross_grid.json")
        grid = wait_for(ROOT / "runs/e3" / ("pair_grid_" + name) / "manifest.json")
        assert grid["trials"] == 120 and set(grid["config"]["thresholds"]) == {25, 30}
        execute(["scripts/run_family_repeats.py", "--stage", "pair", *options], "repeats_" + name, 60)
elif args.stage == "physical_target_only":
    name = "manifest_unseen_pair_random_two_pair_secondary_C128_C512_C2048_Call_seeds11_23_47_89_131_rank0.json"
    smoke = wait_for(ROOT / "runs/shared_target_only_smoke" / name, allow_smoke=True)
    subprocess.run([sys.executable, "scripts/verify_physical_target_only.py", "--run-root", "runs/shared_target_only_smoke"], cwd=ROOT, check=True)
    proof = wait_for(ROOT / "runs/shared_target_only_smoke/verification/manifest.json", allow_smoke=True)
    expected = sum(json.loads((ROOT / path).read_text())["evaluated"] for path in smoke["outputs"])
    assert proof["verified_fits"] == expected and expected > 0
    if "shared_target_only_formal" not in completed:
        execute(["scripts/launch_eight.py", "--log-name", "shared_target_only", "scripts/run_shared_targets.py",
                 "--supervision", "target_only"], "shared_target_only_formal", 60)
else:
    if "flip_target_only" not in completed:
        execute(["scripts/launch_eight.py", "--log-name", "flip_target_only", "scripts/run_functional_adaptation.py",
                 "--dataset", "flip_primary", "--supervision", "target_only"], "flip_target_only", 100)
    for method in ["mochi", "mavenn"]:
        for order in [1, 2]:
            name = f"flip_{method}_order{order}"
            if name in completed:
                continue
            if name == "flip_mavenn_order2" and completed_flip_mavenn_order2():
                completed.append(name)
                continue
            command = ["scripts/launch_eight.py", "--log-name", name, "scripts/run_mochi_targets.py",
                       "--method", method, "--dataset", "flip_primary", "--mode", "flip", "--order", str(order)]
            gpu_memory = 2
            gpus = None
            if method == "mavenn" and order == 2:
                gpus = list(range(8))
                command = ["scripts/launch_eight.py", "--gpus", *[str(gpu) for gpu in gpus],
                           "--log-name", name + "_memresume_20261013", "scripts/run_mochi_targets.py",
                           "--method", method, "--dataset", "flip_primary", "--mode", "flip", "--order", str(order)]
                gpu_memory = 2
            execute(command, name, 120, gpu_memory=gpu_memory, gpus=gpus)
status.write_text(json.dumps({"complete": True, "completed_stages": completed}, indent=2))
