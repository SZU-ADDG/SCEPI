import argparse
from concurrent.futures import ProcessPoolExecutor
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
from epistasis.compensation import input_records
from epistasis.jobs import run_jobs
from scripts import paper_full_target_evidence as evidence

OUTPUT = ROOT / "runs/key_ablations_20261015"
PYTHON = "/data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python"
SEEDS = [11, 23, 47, 89, 131]
CONTROLS = {
    "direct_residual": {"suffix": "direct", "option": "--pair-form", "value": "direct"},
    "no_pair_measured": {"suffix": "nopairmeasured", "option": "--pair-single-input", "value": "predicted"},
}


def prepare():
    if (OUTPUT / "protocol.json").exists():
        protocol = json.loads((OUTPUT / "protocol.json").read_text())
        assert protocol["expected_source_fits"] == 100 and protocol["hyperparameter_search"] is False
        return
    verification = json.loads((OUTPUT / "implementation_verification.json").read_text())
    assert verification["complete"] and not verification["performance_evaluation"]
    evaluation_jobs, matched_runs = [], []
    for arm, control in CONTROLS.items():
        jobs, selected = [], {}
        for threshold in [30, 25]:
            selection_path = ROOT / "runs/e3" / f"pair_repeats_full_g0_f1_j1_family{threshold}/selected_hyperparameters.json"
            selection = json.loads(selection_path.read_text())
            for fold in range(5):
                key = f"family{threshold}/fold{fold}"
                chosen = ROOT / selection[key]["run"]
                selected_name = chosen.name.replace("_combined_seed", f"_{control['suffix']}_combined_seed")
                assert selected_name != chosen.name
                selected[key] = {"run": str(chosen.with_name(selected_name).relative_to(ROOT)),
                                 "matched_full_model": str(chosen.relative_to(ROOT)),
                                 "selection": "Reuse full-model source-validation hyperparameters; no control-specific search."}
                for seed in SEEDS:
                    reference = chosen.with_name(chosen.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
                    original = json.loads((reference / "manifest.json").read_text())
                    assert original["complete"] and not original["smoke"]
                    config = original["config"]
                    assert config["joint"] and config["flexible"] and config["objective"] == "combined"
                    assert config["representation"] == "full" and not config["geometry"] and not config["no_interaction"]
                    single_run = original["single_run"]
                    assert json.loads((ROOT / single_run / "manifest.json").read_text())["complete"]
                    name = reference.name.replace("_combined_seed", f"_{control['suffix']}_combined_seed")
                    run = reference.with_name(name)
                    relative = str(run.relative_to(ROOT))
                    command = [PYTHON, "scripts/train_pair.py", "--threshold", str(threshold), "--fold", str(fold),
                               "--seed", str(seed), "--projection", str(config["projection"]),
                               "--learning-rate", str(config["learning_rate"]), "--lambda-y", str(config["lambda_y"]),
                               "--representation", "full", "--objective", "combined", "--joint", "--flexible",
                               "--batch-size", str(config["batch_size"]), "--single-run", single_run,
                               control["option"], control["value"]]
                    stages = [{"name": "pair_training", "command": command,
                               "manifest": relative + "/manifest.json", "gpu_memory_gib": 2}]
                    for partition in ["validation", "test"]:
                        stages.append({"name": "evaluation_" + partition,
                                       "command": [PYTHON, "scripts/evaluate_family.py", "--run", relative,
                                                   "--partition", partition, "--batch-size", str(config["batch_size"])],
                                       "manifest": relative + f"/evaluation_{partition}/manifest.json", "gpu_memory_gib": 2})
                    jobs.append({"name": key + "/" + name, "stages": stages})
                    matched_runs.append({"arm": arm, "threshold": threshold, "fold": fold, "seed": seed,
                                         "run": relative, "reference_run": str(reference.relative_to(ROOT)),
                                         "single_run": single_run, "reference_parameters": original["trainable_parameters"]})
                    evaluation_jobs.append({"arm": arm, "threshold": threshold, "seed": seed,
                                            "fold": fold, "condition": "Sall"})
        assert len(jobs) == 50
        directory = ROOT / "runs/e3" / f"pair_repeats_full_g0_f1_j1_{control['suffix']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "selected_hyperparameters.json").write_text(json.dumps(selected, indent=2))
        (OUTPUT / f"source_jobs_{arm}.json").write_text(json.dumps(jobs, indent=2))
    evaluation_jobs += [{"arm": "interaction", "threshold": threshold, "seed": seed, "fold": fold, "condition": "Sall"}
                        for threshold in [30, 25] for fold in range(5) for seed in SEEDS]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_path": str(OUTPUT),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_working_tree_modified": True, "controls": CONTROLS, "hyperparameter_search": False,
        "expected_source_fits": 100, "expected_matched_evaluations": 150,
        "thresholds": [30, 25], "folds": list(range(5)), "seeds": SEEDS,
        "direct_residual": "Use F(a,b) directly in place of its four-state contrast. Keep the symmetric network, all inputs and parameter shapes unchanged.",
        "no_pair_measured": "Keep the external calibrated single-effect sum and its permission rule. Feed uncalibrated model-predicted single effects and zero observation indicators to F. Remove both measured-value and availability access from F.",
        "calibration_scope": "The external calibration formula, source-validation selection procedure, allowed single labels and shared single-head training are unchanged. Learned weights are independently fitted under the matched recipe.",
        "matching": "Reuse each full-model fold's selected projection, learning rate and phenotype weight; use its seed-specific pretrained single checkpoint, the same random initialization, family partitions, training rows, sampler, optimizer, losses and checkpoint metric.",
        "optimization": "AdamW weight decay 1e-4, batch size 1024, at most 100 epochs and patience 10. Same update budget and early-stopping rule; realized epoch counts may differ.",
        "target_labels": "S-all references and permitted singles only; zero target double labels for fitting, selection or calibration.",
        "evaluation": "Unchanged full-target S-all truth, eligibility, candidate budgets 1/3/10/30, native and common-main scoring, and margins 0/0.25/0.5. Average matched repeats within protein, then weight proteins equally.",
        "decision": "Use paired mean improvements in residual RMSE and candidate selection. Report repeat direction and protein wins. Retain no independent performance claim for a component without positive matched evidence.",
        "existing_tasks": "Existing trainers, evaluators, weights and logs are preserved.",
        "matched_runs": matched_runs,
    }
    (OUTPUT / "evaluation_jobs.json").write_text(json.dumps(evaluation_jobs, indent=2))
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (OUTPUT / "manifest.json").write_text(json.dumps({"complete": False, "stage": "prepared",
        "expected_source_fits": 100, "completed_source_fits": 0, "expected_matched_evaluations": 150}, indent=2))


def initialize_evidence():
    evidence.OUTPUT = OUTPUT / "evidence"
    evidence.initialize()


def verify_matched_run(row):
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
    for filename in ["training_quartets.parquet", "training_observations.parquet", "validation_quartets.parquet"]:
        pd.testing.assert_frame_equal(pd.read_parquet(run / filename), pd.read_parquet(reference / filename))
    return True


def report(once):
    initialize_evidence()
    jobs = json.loads((OUTPUT / "evaluation_jobs.json").read_text())
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    matched = {tuple(row[field] for field in ["arm", "threshold", "fold", "seed"]): row for row in protocol["matched_runs"]}
    verified = set()
    while True:
        ready, waiting, complete = [], [], []
        for job in jobs:
            if (evidence.directory(job) / "manifest.json").exists():
                assert json.loads((evidence.directory(job) / "manifest.json").read_text())["complete"]
                complete.append(job)
                continue
            truth = evidence.TRUTH[(job["threshold"], job["fold"])]
            _, missing = input_records(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", truth)
            if missing:
                waiting.append(job)
                continue
            if job["arm"] != "interaction":
                key = tuple(job[field] for field in ["arm", "threshold", "fold", "seed"])
                if key not in verified:
                    verify_matched_run(matched[key])
                    verified.add(key)
            ready.append(job)
        if ready:
            with ProcessPoolExecutor(max_workers=4, initializer=initialize_evidence) as pool:
                list(pool.map(evidence.evaluate, ready))
            complete += ready
        source_complete = sum((ROOT / row["run"] / "manifest.json").exists() for row in protocol["matched_runs"])
        all_ready = len(complete) == len(jobs)
        state = {"complete": False, "evaluations_ready": all_ready, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "expected_source_fits": 100, "completed_source_fits": source_complete,
                 "expected_matched_evaluations": len(jobs), "completed_matched_evaluations": len(complete),
                 "waiting_evaluations": len(waiting), "protocol": str(OUTPUT / "protocol.json")}
        (OUTPUT / "manifest.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        if all_ready:
            for row in protocol["matched_runs"]:
                verify_matched_run(row)
            evidence.aggregate(jobs, [])
            (OUTPUT / "matched_conditions_verification.json").write_text(json.dumps({"complete": True,
                "matched_source_fits": 100, "identical_training_and_validation_rows": True,
                "identical_parameter_counts": True, "identical_pretrained_singles": True,
                "same_hyperparameters_and_selection_rule": True}, indent=2))
            state["complete"] = True
            state["evidence_manifest"] = str(evidence.OUTPUT / "manifest.json")
            (OUTPUT / "manifest.json").write_text(json.dumps(state, indent=2))
            return
        if once:
            return
        time.sleep(30)


def launch():
    assert Path(sys.executable).resolve() == Path(PYTHON).resolve()
    assert not (OUTPUT / "launch.json").exists(), "Launch already recorded; inspect existing controllers before restarting."
    processes = []
    for name, options in [(arm, ["--arm", arm]) for arm in CONTROLS] + [("reporting", ["--report"])]:
        log = OUTPUT / f"{name}.controller.log"
        with log.open("a") as stream:
            process = subprocess.Popen([PYTHON, "scripts/run_key_ablations.py", *options], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append({"stage": name, "pid": process.pid, "log": str(log.relative_to(ROOT))})
    (OUTPUT / "launch.json").write_text(json.dumps({"processes": processes}, indent=2))
    print(json.dumps({"launched": processes}, indent=2))


def summarize_results():
    (OUTPUT / "statistics_controller.json").write_text(json.dumps({"pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv}, indent=2))
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    controllers = json.loads((OUTPUT / "launch.json").read_text())["processes"]
    report_pid = next(item["pid"] for item in controllers if item["stage"] == "reporting")
    manifest = OUTPUT / "manifest.json"
    while not json.loads(manifest.read_text())["complete"]:
        assert psutil.pid_exists(report_pid), "Reporting controller exited before completion; inspect its log."
        for row in protocol["matched_runs"]:
            for stage in ["pair_training", "evaluation_validation", "evaluation_test"]:
                path = ROOT / row["run"] / (stage + ".capacity.json")
                if path.exists():
                    assert json.loads(path.read_text())["state"] != "failed", path
        time.sleep(30)
    source = OUTPUT / "evidence"
    assert json.loads((source / "manifest.json").read_text())["complete"]
    assert json.loads((OUTPUT / "matched_conditions_verification.json").read_text())["matched_source_fits"] == 100
    fields = ["threshold", "task", "margin", "unit", "scoring", "budget", "metric", "reference", "control"]
    original_rows = json.loads((source / "paired_summary.json").read_text())
    indexed = {tuple(row.get(field) for field in fields): row for row in original_rows}
    assert len(indexed) == len(original_rows)
    summaries, checked = [], 0
    maximum_difference = 0.0
    for threshold in [30, 25]:
        destination = source / f"verified_family{threshold}"
        subprocess.run([PYTHON, "scripts/summarize_full_target_evidence.py", "--threshold", str(threshold),
                        "--input", str(source), "--output", str(destination), "--source-run-path", str(source)],
                       cwd=ROOT, check=True)
        proof = json.loads((destination / "manifest.json").read_text())
        assert proof["complete"] and proof["verified_grids"] == 75
        assert set(proof["arms"]) == {"interaction", *CONTROLS}
        for row in json.loads((destination / "paired_summary.json").read_text()):
            original = indexed[tuple(row.get(field) for field in fields)]
            for field in ["proteins", "family_groups", "paired_protein_repeats", "complete_grid", "repeats"]:
                assert row[field] == original[field]
            for field in ["reference_mean", "control_mean", "mean_gain"]:
                difference = abs(row[field] - original[field])
                assert difference <= 1e-12
                maximum_difference = max(maximum_difference, difference)
            assert row["complete_grid"] and row["repeats"] == 5
            assert row["wins"] + row["ties"] + row["losses"] == row["proteins"]
            checked += 1
            residual = row["metric"] == "epsilon_rmse" and row["margin"] in [0, 0.5]
            selection = (row.get("task") == "compensation" and row["budget"] == 10 and row["margin"] in [0, 0.5]
                         and row["metric"] in ["compensation_hit_rate", "selected_mean_recovery"])
            if residual or selection:
                summaries.append(row)
    assert checked == len(original_rows)
    (OUTPUT / "primary_statistics.json").write_text(json.dumps(summaries, indent=2))
    result = {"complete": True, "source_commit": json.loads((OUTPUT / "protocol.json").read_text())["source_commit"],
              "run_path": str(OUTPUT), "source_working_tree_modified": True,
              "new_source_fits": 100, "matched_reference_evaluations": 50, "matched_evaluations": 150,
              "independently_checked_paired_comparisons": checked, "maximum_mean_difference": maximum_difference,
              "outputs": ["primary_statistics.json", "evidence/verified_family30/paper_table.csv",
                          "evidence/verified_family25/paper_table.csv", "matched_conditions_verification.json"],
              "new_results_written_to_manuscript": False}
    (OUTPUT / "final_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--arm", choices=list(CONTROLS))
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    prepare()
    if args.arm:
        jobs = json.loads((OUTPUT / f"source_jobs_{args.arm}.json").read_text())
        run_jobs(ROOT, jobs, OUTPUT / args.arm, gpu_memory_gib=2)
    elif args.report:
        report(args.once)
    elif args.launch:
        launch()
    elif args.summarize:
        summarize_results()


if __name__ == "__main__":
    main()
