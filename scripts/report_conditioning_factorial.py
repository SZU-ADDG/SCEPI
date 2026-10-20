import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import input_records
from scripts import paper_full_target_evidence as evidence
from scripts.run_conditioning_factorial import CONTROLS, OUTPUT, TAGS
from scripts.run_sall_full_target import paired_summary

ADDED = OUTPUT / "evidence"
DESTINATION = OUTPUT / "paper_evidence"
BASE = ROOT / "runs/key_ablations_20261015/paper_evidence"
CANONICAL = {"final_interaction": "interaction", "final_remove_interaction": "remove_interaction",
             "final_direct_residual": "direct_residual", "final_no_pair_measured": "no_pair_measured",
             "final_values_only": "values_only", "final_mask_only": "mask_only"}
LEGACY = {"interaction": "joint_objective", "remove_interaction": "joint_remove_interaction",
          "direct_residual": "joint_direct_residual", "no_pair_measured": "joint_no_pair_measured"}


def initialize():
    evidence.OUTPUT = ADDED
    evidence.initialize()


def verify_match(row):
    run, reference = ROOT / row["run"], ROOT / row["reference_run"]
    fitted = json.loads((run / "manifest.json").read_text())
    original = json.loads((reference / "manifest.json").read_text())
    assert fitted["complete"] and not fitted["smoke"]
    assert fitted["trainable_parameters"] == original["trainable_parameters"]
    assert fitted["single_run"] == original["single_run"]
    assert json.loads((run / "permissions.json").read_text()) == json.loads((reference / "permissions.json").read_text())
    for field in ["projection", "learning_rate", "lambda_y", "representation", "geometry", "flexible", "joint",
                  "objective", "batch_size", "threshold", "fold", "seed"]:
        assert fitted["config"][field] == original["config"][field], (row, field)
    expected = CONTROLS[row["arm"]]
    assert fitted["config"]["pair_form"] == expected["form"]
    assert fitted["config"]["pair_single_input"] == expected["single"]
    assert fitted["config"].get("pair_mask_input", "auto") == expected["mask"]
    for name in ["training_quartets.parquet", "training_observations.parquet", "validation_quartets.parquet"]:
        pd.testing.assert_frame_equal(pd.read_parquet(run / name), pd.read_parquet(reference / name))


def integrate(jobs):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in ["phenotype", "residual", "selection"]:
        base = pd.read_parquet(BASE / (name + "_per_protein.parquet"))
        base["arm"] = base.arm.replace(LEGACY)
        chunks = [base]
        for job in jobs:
            chunk = pd.read_parquet(evidence.directory(job) / (name + ".parquet"))
            chunk["arm"] = chunk.arm.map(CANONICAL)
            assert chunk.arm.notna().all()
            chunks.append(chunk)
        table = pd.concat(chunks, ignore_index=True)
        if name == "phenotype":
            table = table.loc[table.phenotype_count.ge(30)].copy()
        keys = ["threshold", "seed", "protein_id", "family_group", "arm"]
        keys += ["margin"] if name == "residual" else ["task", "margin", "unit", "scoring", "budget"] if name == "selection" else []
        assert not table.duplicated(keys).any()
        table.to_parquet(DESTINATION / (name + "_per_protein.parquet"), index=False)
        tables[name] = table
    grid = tables["phenotype"][["threshold", "arm", "seed", "fold"]].drop_duplicates()
    coverage = Counter((int(row.threshold), row.arm) for row in grid.itertuples())
    assert len(grid) == 1600 and len(set(grid.arm)) == 32
    assert len(coverage) == 64 and all(count == 25 for count in coverage.values())
    summary = []
    for metric in ["compensation_hit_rate", "selected_mean_recovery", "selected_mean_stability", "precision_lower", "precision_upper"]:
        summary += paired_summary(tables["selection"], ["threshold", "task", "margin", "unit", "scoring", "budget"], metric, True, coverage, tie_tolerance=1e-12)
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        summary += paired_summary(tables["residual"], ["threshold", "margin"], metric, False, coverage, tie_tolerance=1e-12)
    for metric, higher in [("phenotype_rmse", False), ("phenotype_spearman", True), ("phenotype_pearson", True)]:
        summary += paired_summary(tables["phenotype"], ["threshold"], metric, higher, coverage, tie_tolerance=1e-12)
    evidence.bootstrap(summary, tables)
    (DESTINATION / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"complete": True, "available_jobs_complete": True, "expected_evaluations": 1600,
                "completed_evaluations": 1600, "waiting_evaluations": 0,
                "scope": "32 complete S-all arms; discontinued ESM-MSR extensions remain outside this scope.",
                "source_commit": source_commit, "source_working_tree_modified": True,
                "run_path": str(DESTINATION), "base_evidence": str(BASE), "extension_evidence": str(ADDED),
                "source_selection": str(OUTPUT / "source_selection.json"),
                "canonical_arms": CANONICAL, "preserved_joint_recipe_arms": LEGACY,
                "completed_grids": [{"threshold": threshold, "arm": arm, "evaluations": count}
                                    for (threshold, arm), count in sorted(coverage.items())],
                "paired_comparisons": len(summary), "win_tie_tolerance": 1e-12}
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2))
    maximum, checked = 0.0, 0
    fields = ["threshold", "task", "margin", "unit", "scoring", "budget", "metric", "reference", "control"]
    indexed = {tuple(row.get(key) for key in fields): row for row in summary}
    for threshold in [30, 25]:
        destination = DESTINATION / f"verified_family{threshold}"
        with (OUTPUT / f"summary_family{threshold}.log").open("w") as stream:
            subprocess.run([sys.executable, "scripts/summarize_full_target_evidence.py", "--input", str(DESTINATION),
                            "--output", str(destination), "--threshold", str(threshold),
                            "--source-run-path", str(DESTINATION), "--source-commit", source_commit],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        rows = json.loads((destination / "paired_summary.json").read_text())
        for row in rows:
            first = indexed[tuple(row.get(key) for key in fields)]
            for field in ["reference_mean", "control_mean", "mean_gain"]:
                difference = abs(first[field] - row[field])
                assert difference <= 1e-12
                maximum = max(maximum, difference)
            for field in ["proteins", "family_groups", "repeats", "paired_protein_repeats", "complete_grid"]:
                assert first[field] == row[field]
            checked += 1
    assert checked == len(summary)
    (DESTINATION / "integration_verification.json").write_text(json.dumps(
        {"complete": True, "paired_comparisons_checked": checked, "maximum_mean_difference": maximum,
         "completed_arms": 32, "all_original_results_preserved": True}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        path = OUTPUT / "report_controller.json"
        assert not path.exists(), "Inspect the recorded report controller before restarting."
        with (OUTPUT / "report.controller.log").open("a") as stream:
            process = subprocess.Popen([sys.executable, "scripts/report_conditioning_factorial.py", "--watch"], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        path.write_text(json.dumps({"pid": process.pid}, indent=2))
        print(json.dumps({"launched_report_pid": process.pid}))
        return
    initialize()
    jobs = [{"arm": arm, "threshold": threshold, "fold": fold, "seed": seed, "condition": "Sall"}
            for arm in TAGS for threshold in [30, 25] for fold in range(5) for seed in [11, 23, 47, 89, 131]]
    assert len(jobs) == 300
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    matched = {(row["arm"], row["threshold"], row["fold"], row["seed"]): row for row in protocol["matched_runs"]}
    verified = set()
    while True:
        ready, finished = [], []
        for job in jobs:
            if (evidence.directory(job) / "manifest.json").exists():
                assert json.loads((evidence.directory(job) / "manifest.json").read_text())["complete"]
                finished.append(job)
                continue
            truth = evidence.TRUTH[(job["threshold"], job["fold"])]
            _, missing = input_records(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", truth)
            if missing:
                continue
            if job["arm"] in CONTROLS:
                key = tuple(job[name] for name in ["arm", "threshold", "fold", "seed"])
                verify_match(matched[key])
                verified.add(key)
            ready.append(job)
        if ready:
            with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
                list(pool.map(evidence.evaluate, ready))
            finished += ready
        source_complete = sum((ROOT / row["run"] / "manifest.json").exists() for row in protocol["matched_runs"])
        state = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
                 "completed_control_fits": source_complete, "expected_control_fits": 200,
                 "new_fits_completed": source_complete - protocol["existing_control_fits_reused"],
                 "new_fits_required": protocol["new_source_fits_required_at_preparation"],
                 "completed_evaluations": len(finished), "expected_evaluations": len(jobs)}
        (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        if len(finished) == len(jobs):
            for row in protocol["matched_runs"]:
                verify_match(row)
            (OUTPUT / "matched_conditions_verification.json").write_text(json.dumps(
                {"complete": True, "matched_control_fits": 200, "all_four_factorial_cells_complete": True,
                 "identical_training_and_validation_rows": True, "identical_parameter_counts": True}, indent=2))
            integrate(jobs)
            subprocess.run([sys.executable, "scripts/analyze_conditioning_factorial.py"], cwd=ROOT, check=True)
            state.update(complete=True, source_selection=str(OUTPUT / "source_selection.json"),
                         evidence_manifest=str(DESTINATION / "manifest.json"))
            (OUTPUT / "final_manifest.json").write_text(json.dumps(state, indent=2))
            (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
            print(json.dumps(state), flush=True)
            return
        if not args.watch:
            return
        for controller in json.loads((OUTPUT / "launch.json").read_text())["processes"]:
            complete = OUTPUT / f"queue{controller['queue']}/manifest.json"
            assert complete.exists() or psutil.pid_exists(controller["pid"]), controller
        time.sleep(30)


if __name__ == "__main__":
    main()
