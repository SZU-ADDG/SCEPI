import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs

OUTPUT = ROOT / "runs/key_ablations_20261015"
ARMS = ["direct_residual", "no_pair_measured"]


def verify_completed_reuse():
    records = []
    for arm in ARMS:
        job = json.loads((OUTPUT / f"source_jobs_{arm}.json").read_text())[0]
        stage = job["stages"][0]
        run = ROOT / Path(stage["manifest"]).parent
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["complete"] and not manifest["smoke"]
        paths = [run / name for name in ["best.pt", "manifest.json", "history.json", "config.json", "permissions.json",
                 "training_quartets.parquet", "training_observations.parquet", "validation_quartets.parquet"]]
        before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
        subprocess.run(stage["command"], cwd=ROOT, check=True)
        after = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
        assert before == after
        assert json.loads((run / "manifest.json").read_text()) == manifest
        records.append({"arm": arm, "run": str(run.relative_to(ROOT)), "completed_training_reused": True,
                        "existing_artifacts_unchanged": True})
    result = {"complete": True, "real_completed_runs": records,
              "training_or_evaluation_repeated": False}
    (OUTPUT / "scheduling_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--middle-folds", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_completed_reuse()
        return
    assert json.loads((OUTPUT / "scheduling_verification.json").read_text())["complete"]
    suffix = "_folds12" if args.middle_folds else ""
    folds = [1, 2] if args.middle_folds else [3, 4]
    controller = OUTPUT / f"accelerator{suffix}_controller.json"
    assert not controller.exists(), "Inspect the recorded accelerator before starting another one."
    jobs = []
    for arm in ARMS:
        for job in json.loads((OUTPUT / f"source_jobs_{arm}.json").read_text()):
            if job["name"].startswith(tuple(f"family25/fold{fold}/" for fold in folds)):
                stage = {**job["stages"][0], "name": f"pair_training_accelerator{suffix}"}
                jobs.append({**job, "stages": [stage]})
    assert len(jobs) == 20
    jobs.reverse()
    controller.write_text(json.dumps({"pid": os.getpid(), "jobs": len(jobs), "folds": folds, "new_experiment_configurations": 0,
        "source_fits_in_original_protocol": 100, "only_source_training": True,
        "evaluation": "The original controllers retain validation and test evaluation stages.",
        "run_lock": "The training process serializes duplicate claims and reuses completed matching manifests."}, indent=2))
    run_jobs(ROOT, jobs, OUTPUT / f"accelerator{suffix}", gpu_memory_gib=2)


if __name__ == "__main__":
    main()
