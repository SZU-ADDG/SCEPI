import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paper_full_target_evidence as evidence

RUN = ROOT / "runs/key_ablations_20261015"
BASE = ROOT / "runs/sall_full_target/paper_evidence"
ADDED = RUN / "evidence"
OUTPUT = RUN / "paper_evidence"
CONTROLS = {"direct_residual", "no_pair_measured"}
FIELDS = ["threshold", "task", "margin", "unit", "scoring", "budget", "metric", "reference", "control"]


def verify_reference():
    for name in ["phenotype", "residual", "selection"]:
        original = pd.read_parquet(BASE / (name + "_per_protein.parquet"))
        original = original.loc[original.arm.eq("interaction")]
        extension = pd.read_parquet(ADDED / (name + "_per_protein.parquet"))
        extension = extension.loc[extension.arm.eq("interaction")]
        keys = ["threshold", "seed", "fold", "protein_id", "family_group"]
        keys += ["margin"] if name == "residual" else ["task", "margin", "unit", "scoring", "budget"] if name == "selection" else []
        columns = sorted(set(original.select_dtypes(include="number")) & set(extension.select_dtypes(include="number")) - set(keys))
        left, right = original.set_index(keys).sort_index(), extension.set_index(keys).sort_index()
        assert left.index.equals(right.index)
        np.testing.assert_allclose(left[columns], right[columns], rtol=0, atol=1e-12, equal_nan=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    final_path = RUN / "final_manifest.json"
    if args.wait:
        while not final_path.exists():
            time.sleep(30)
    final = json.loads(final_path.read_text())
    assert final["complete"] and final["new_source_fits"] == 100
    assert json.loads((RUN / "matched_conditions_verification.json").read_text())["complete"]
    verify_reference()
    base_jobs = json.loads((ROOT / "runs/sall_full_target/jobs.json").read_text())
    waiting = [job for job in base_jobs if job["arm"] == "esm_msr"]
    jobs = [job for job in base_jobs if job["arm"] != "esm_msr"]
    assert len(jobs) == 1150 and len(waiting) == 50
    added = [job for job in json.loads((RUN / "evaluation_jobs.json").read_text()) if job["arm"] in CONTROLS]
    assert len(added) == 100
    jobs += added
    roots = {job["arm"]: ADDED if job["arm"] in CONTROLS else BASE for job in jobs}
    evidence.OUTPUT = OUTPUT
    evidence.aggregate(jobs, waiting, include_stabilityarc=True, evaluation_roots=roots)
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert manifest["completed_evaluations"] == 1300 and manifest["expected_evaluations"] == 1350
    assert len(manifest["completed_grids"]) == 52
    manifest.update({"base_evidence": str(BASE), "extension_evidence": str(ADDED),
                     "extension_protocol": str(RUN / "protocol.json"), "extension_final_manifest": str(RUN / "final_manifest.json"),
                     "source_working_tree_modified": True, "reference_predictions_and_metrics_match_base": True})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    reference = {tuple(row.get(field) for field in FIELDS): row
                 for row in json.loads((OUTPUT / "paired_summary.json").read_text())}
    maximum_difference, checked = 0.0, 0
    for threshold in [30, 25]:
        directory = OUTPUT / f"verified_family{threshold}"
        subprocess.run([sys.executable, "scripts/summarize_full_target_evidence.py", "--input", str(OUTPUT),
                        "--output", str(directory), "--threshold", str(threshold), "--source-run-path", str(OUTPUT)],
                       cwd=ROOT, check=True)
        proof = json.loads((directory / "manifest.json").read_text())
        assert proof["complete"] and proof["verified_grids"] == 650 and len(proof["arms"]) == 26
        for row in json.loads((directory / "paired_summary.json").read_text()):
            original = reference[tuple(row.get(field) for field in FIELDS)]
            for field in ["proteins", "family_groups", "paired_protein_repeats", "complete_grid", "repeats"]:
                assert row[field] == original[field]
            for field in ["reference_mean", "control_mean", "mean_gain"]:
                difference = abs(row[field] - original[field])
                assert difference <= 1e-12
                maximum_difference = max(maximum_difference, difference)
            checked += 1
    assert checked == len(reference)
    result = {"complete": True, "source_commit": manifest["source_commit"], "run_path": str(OUTPUT),
              "completed_arms": 26, "completed_evaluations": 1300, "pending_external_evaluations": 50,
              "paired_comparisons_checked": checked, "maximum_mean_difference": maximum_difference,
              "existing_reference_metrics_preserved": True,
              "source_working_tree_modified": True,
              "full_planned_matrix_complete": False}
    (OUTPUT / "integration_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
