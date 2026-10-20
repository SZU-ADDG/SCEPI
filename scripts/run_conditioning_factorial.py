import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs

OUTPUT = ROOT / "runs/conditioning_factorial_20261021"
PYTHON = "/data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python"
CONTROLS = {
    "final_no_pair_measured": {"suffix": "nopairmeasured", "single": "predicted", "mask": "auto", "form": "contrast"},
    "final_values_only": {"suffix": "nomask", "single": "calibrated", "mask": "zero", "form": "contrast"},
    "final_mask_only": {"suffix": "predictedvalues", "single": "predicted", "mask": "observed", "form": "contrast"},
    "final_direct_residual": {"suffix": "direct", "single": "calibrated", "mask": "auto", "form": "direct"},
}
TAGS = {"final_interaction": "source_selected", "final_remove_interaction": "source_selected"}
TAGS.update({arm: "source_selected_" + config["suffix"] for arm, config in CONTROLS.items()})


def prepare():
    selection = json.loads((OUTPUT / "source_selection.json").read_text())
    assert selection["complete"] and not selection["target_predictions_or_metrics_read"]
    jobs, matches = [], []
    selected = {arm: {} for arm in TAGS}
    reused = 0
    for key, chosen in selection["selected"].items():
        for arm in TAGS:
            reference = Path(chosen["run"])
            if arm in CONTROLS:
                suffix = CONTROLS[arm]["suffix"]
                name = reference.name.replace(f"_{chosen['objective']}_seed", f"_{suffix}_{chosen['objective']}_seed")
                assert name != reference.name
                reference = reference.with_name(name)
            selected[arm][key] = {"run": str(reference), "matched_full_model": chosen["run"],
                                  "objective": chosen["objective"], "selection": "Reuse the full model's source-selected recipe; independently refit weights."}
        for repetition in chosen["repeats"]:
            reference = ROOT / repetition["run"]
            original = json.loads((reference / "manifest.json").read_text())
            config = original["config"]
            stages = [{"name": "evaluation_" + partition,
                       "command": [PYTHON, "scripts/evaluate_family.py", "--run", repetition["run"], "--partition", partition],
                       "manifest": repetition["run"] + f"/evaluation_{partition}/manifest.json", "gpu_memory_gib": 2}
                      for partition in ["validation", "test"]]
            jobs.append({"name": "final_interaction/" + key + f"/seed{repetition['seed']}", "stages": stages})
            for arm, control in CONTROLS.items():
                run = reference.with_name(reference.name.replace(f"_{config['objective']}_seed", f"_{control['suffix']}_{config['objective']}_seed"))
                relative = str(run.relative_to(ROOT))
                command = [PYTHON, "scripts/train_pair.py", "--threshold", str(config["threshold"]), "--fold", str(config["fold"]),
                           "--seed", str(config["seed"]), "--projection", str(config["projection"]),
                           "--learning-rate", str(config["learning_rate"]), "--lambda-y", str(config["lambda_y"]),
                           "--representation", "full", "--objective", config["objective"], "--joint", "--flexible",
                           "--batch-size", str(config["batch_size"]), "--single-run", original["single_run"],
                           "--pair-form", control["form"], "--pair-single-input", control["single"],
                           "--pair-mask-input", control["mask"]]
                existing = run / "manifest.json"
                if existing.exists():
                    saved = json.loads(existing.read_text())
                    assert saved["complete"] and not saved["smoke"]
                    assert saved["single_run"] == original["single_run"]
                    assert saved["trainable_parameters"] == original["trainable_parameters"]
                    for field in ["projection", "learning_rate", "lambda_y", "objective", "batch_size", "threshold", "fold", "seed"]:
                        assert saved["config"][field] == config[field]
                    reused += 1
                stages = [{"name": "pair_training", "command": command, "manifest": relative + "/manifest.json", "gpu_memory_gib": 2}]
                stages += [{"name": "evaluation_" + partition,
                            "command": [PYTHON, "scripts/evaluate_family.py", "--run", relative, "--partition", partition],
                            "manifest": relative + f"/evaluation_{partition}/manifest.json", "gpu_memory_gib": 2}
                           for partition in ["validation", "test"]]
                jobs.append({"name": arm + "/" + key + f"/seed{config['seed']}", "stages": stages})
                matches.append({"arm": arm, "threshold": config["threshold"], "fold": config["fold"], "seed": config["seed"],
                                "run": relative, "reference_run": str(reference.relative_to(ROOT)), "single_run": original["single_run"]})
    assert len(jobs) == 250 and len(matches) == 200
    for arm, tag in TAGS.items():
        path = ROOT / "runs/e3" / ("pair_repeats_" + tag)
        path.mkdir(parents=True, exist_ok=True)
        destination = path / "selected_hyperparameters.json"
        if destination.exists():
            assert json.loads(destination.read_text()) == selected[arm]
        else:
            destination.write_text(json.dumps(selected[arm], indent=2))
    jobs_path = OUTPUT / "source_jobs.json"
    if jobs_path.exists():
        assert json.loads(jobs_path.read_text()) == jobs
    else:
        jobs_path.write_text(json.dumps(jobs, indent=2))
        protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "source_working_tree_modified": True, "source_selection": str((OUTPUT / "source_selection.json").relative_to(ROOT)),
                    "source_fits_in_control_matrix": len(matches), "new_source_fits_required_at_preparation": len(matches) - reused,
                    "existing_control_fits_reused": reused, "reference_fits_reused": 50,
                    "full_factorial": {"final_interaction": ["calibrated", "observed"],
                                       "final_values_only": ["calibrated", "zero"],
                                       "final_mask_only": ["predicted", "observed"],
                                       "final_no_pair_measured": ["predicted", "zero"]},
                    "matching": "Same selected objective, hyperparameters, single initialization, parameter shapes, rows, random seed, training recipe, validation criterion and stopping rules; only the specified pair input or cycle parameterization changes.",
                    "WT_states": "Every state retains the original active gating for values and availability.",
                    "calibration": "External calibration and its source-validation omega rule remain unchanged; each model is independently fitted.",
                    "target_double_labels_for_fitting_or_selection": 0,
                    "matched_runs": matches}
        (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--queue", type=int)
    parser.add_argument("--queues", type=int, default=3)
    args = parser.parse_args()
    assert args.queues > 0
    jobs = prepare()
    if args.prepare_only:
        protocol = json.loads((OUTPUT / "protocol.json").read_text())
        print(json.dumps({key: protocol[key] for key in ["source_fits_in_control_matrix", "new_source_fits_required_at_preparation",
                                                        "existing_control_fits_reused", "reference_fits_reused"]}, indent=2))
        return
    verification = json.loads((OUTPUT / "implementation_verification.json").read_text())
    assert verification["complete"] and not verification["performance_evaluation"]
    assert Path(sys.executable).resolve() == Path(PYTHON).resolve()
    if args.launch:
        path = OUTPUT / "launch.json"
        assert not path.exists(), "Inspect existing controllers before starting a new launch."
        processes = []
        for queue in range(args.queues):
            log = OUTPUT / f"queue{queue}.controller.log"
            with log.open("a") as stream:
                process = subprocess.Popen([PYTHON, "scripts/run_conditioning_factorial.py", "--queue", str(queue), "--queues", str(args.queues)],
                                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append({"queue": queue, "pid": process.pid, "log": str(log.relative_to(ROOT))})
        path.write_text(json.dumps({"processes": processes, "queues": args.queues}, indent=2))
        print(json.dumps({"launched": processes}, indent=2))
        return
    assert args.queue is not None and 0 <= args.queue < args.queues
    current = [job for index, job in enumerate(jobs) if index % args.queues == args.queue]
    run_jobs(ROOT, current, OUTPUT / f"queue{args.queue}", gpu_memory_gib=2)


if __name__ == "__main__":
    main()
