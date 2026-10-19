from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/method_reporting_20261015"
CORE = {"interaction", "additive", "global_nonlinear", "remove_interaction", "frozen_main",
        "identity_mapping", "reference_context", "identity_features", "no_cross", "residual_only",
        "phenotype_only", "geometry", "geometry_permuted_retrained", "geometry_rewired_retrained",
        "geometry_permuted_inference", "geometry_rewired_inference"}


def owned(arm):
    return arm in CORE or arm.startswith(("original_", "interaction_", "transfer_", "target_only_"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrices = []
    for name in ["original_main", "compensation_main", "sall_full_target"]:
        directory = ROOT / "runs" / name
        jobs = json.loads((directory / "jobs.json").read_text())
        total, completed, missing, pending_external = Counter(), Counter(), [], Counter()
        for job in jobs:
            if name == "sall_full_target":
                path = (directory / "evaluations" / job["arm"] / f"family{job['threshold']}"
                        / f"seed{job['seed']}" / f"fold{job['fold']}" / "manifest.json")
                identity = f"{job['arm']}/family{job['threshold']}/seed{job['seed']}/fold{job['fold']}"
            else:
                path = ROOT / job["stages"][0]["manifest"]
                identity = job["name"]
            done = path.is_file()
            if done:
                manifest = json.loads(path.read_text())
                assert manifest["complete"] and not manifest.get("smoke", False), path
            if owned(job["arm"]):
                total[job["arm"]] += 1
                if done:
                    completed[job["arm"]] += 1
                else:
                    missing.append({"job": identity, "manifest": str(path.relative_to(ROOT))})
            elif not done:
                pending_external[(job["arm"], job.get("condition", ""))] += 1
        matrices.append({"matrix": name, "expected_method_evaluations": sum(total.values()),
                         "completed_method_evaluations": sum(completed.values()),
                         "method_grids": [{"arm": arm, "expected": count, "completed": completed[arm]}
                                          for arm, count in sorted(total.items())],
                         "missing_method_evaluations": missing,
                         "pending_external_evaluations": [{"arm": arm, "condition": condition, "evaluations": count}
                                                          for (arm, condition), count in sorted(pending_external.items())]})
    rows = json.loads((ROOT / "runs/original_main/paired_summary.json").read_text())
    selected = []
    for row in rows:
        if row["dataset"] != "tsuboyama" or row["threshold"] != 30 or row["family_scope"] != "all_groups":
            continue
        if row.get("mutation_order") != "all" or row["metric"] != "epsilon_rmse":
            continue
        family = row["scheme"] == "family" and row["reference"] == "interaction"
        combination = row["scheme"] == "unseen_pair" and row["reference"] == "interaction_C"
        if not (family or combination):
            continue
        if row["control"] not in {"additive", "global_nonlinear", "thermo_epistatic", "kermut",
                                  "mochi_order2_C", "thermo_epistatic_C", "interaction_C_r0"}:
            continue
        assert row["evaluation_grid_complete"] and row["evaluated_repeats"] == 5
        selected.append(row)
    assert selected
    (OUTPUT / "original_comparisons.json").write_text(json.dumps(selected, indent=2))
    statistics = []
    verified = []
    controls = {"additive", "global_nonlinear", "thermo_epistatic", "kermut", "mean_mlp", "mean_ridge", "stabilityarc_kermut"}
    for threshold in [30, 25]:
        directory = ROOT / "runs/sall_full_target/paper_evidence" / f"verified_family{threshold}"
        proof = json.loads((directory / "manifest.json").read_text())
        assert proof["complete"] and proof["verified_grids"] == 600 and len(proof["arms"]) == 24
        verified.append(str((directory / "manifest.json").relative_to(ROOT)))
        shutil.copyfile(directory / "paper_table.csv", OUTPUT / f"family{threshold}_table.csv")
        for row in json.loads((directory / "paired_summary.json").read_text()):
            if (row.get("task") == "compensation" and row["budget"] == 10
                    and row["metric"] in {"compensation_hit_rate", "selected_mean_recovery"}
                    and row["margin"] in {0.0, 0.5} and row["control"] in controls):
                statistics.append(row)
    (OUTPUT / "sall_statistics.json").write_text(json.dumps(statistics, indent=2))
    missing = sum(len(matrix["missing_method_evaluations"]) for matrix in matrices)
    report = {"complete": missing == 0, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "run_path": str(OUTPUT), "method_evaluations_complete": missing == 0,
              "matrices": matrices, "selected_original_comparisons": len(selected),
              "independently_verified_sall_manifests": verified,
              "external_training_policy": "One existing default ESM-MSR source run; additional grid training stopped; existing artifacts retained."}
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({**{key: value for key, value in report.items() if key != "matrices"},
                      "matrix_counts": [{key: matrix[key] for key in ["matrix", "expected_method_evaluations", "completed_method_evaluations"]}
                                        for matrix in matrices]}), flush=True)
    assert missing == 0, "Method evaluations still require prediction or evaluation work"


if __name__ == "__main__":
    main()
