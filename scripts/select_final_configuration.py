import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 23, 47, 89, 131]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/conditioning_factorial_20261021")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    assert output.is_relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    permissions_by_fold = {}
    for objective in ["combined", "phenotype"]:
        suffix = "" if objective == "combined" else "_phenotype"
        directory = ROOT / "runs/e3" / ("pair_grid_full_g0_f1_j1" + suffix)
        jobs = json.loads((directory / "jobs.json").read_text())
        assert len(jobs) == 120
        for job in jobs:
            run = ROOT / job["run"]
            manifest = json.loads((run / "manifest.json").read_text())
            config = manifest["config"]
            assert manifest["complete"] and not manifest["smoke"]
            assert config["objective"] == objective and config["seed"] == 11
            assert config["representation"] == "full" and config["joint"] and config["flexible"]
            assert not config["geometry"] and not config["no_interaction"]
            key = f"family{config['threshold']}/fold{config['fold']}"
            permissions = json.loads((run / "permissions.json").read_text())
            groups = [set(permissions[name]) for name in ["train", "validation", "test"]]
            assert all(groups) and not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            if key in permissions_by_fold:
                assert permissions_by_fold[key] == permissions
            permissions_by_fold[key] = permissions
            history = json.loads((run / "history.json").read_text())
            best = min(history, key=lambda row: (row["validation_score"], row["epoch"]))
            assert best["epoch"] == manifest["best_epoch"]
            scores = best["validation_conditions"]
            omega = min([0.0, 0.5, 1.0], key=lambda value: (scores[f"S128_w{value:g}"], value))
            score = 0.5 * scores["Z"] + 0.25 * scores[f"S128_w{omega:g}"] + 0.25 * scores[f"Sall_w{omega:g}"]
            assert math.isclose(score, manifest["validation_score"], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(best["validation_score"], score, rel_tol=0, abs_tol=1e-12)
            assert omega == manifest["omega"] == best["omega"]
            candidates.append({"key": key, "threshold": config["threshold"], "fold": config["fold"],
                               "objective": objective, "run": job["run"], "validation_score": score,
                               "validation_Z": scores["Z"], "validation_S128": scores[f"S128_w{omega:g}"],
                               "validation_Sall": scores[f"Sall_w{omega:g}"], "omega": omega,
                               "projection": config["projection"], "learning_rate": config["learning_rate"],
                               "lambda_y": config["lambda_y"], "best_epoch": best["epoch"],
                               "training_proteins": len(groups[0]), "validation_proteins": len(groups[1]),
                               "excluded_target_proteins": len(groups[2])})
    selected = {}
    for key in sorted(permissions_by_fold):
        current = [row for row in candidates if row["key"] == key]
        assert len(current) == 24
        by_objective = {objective: min([row for row in current if row["objective"] == objective],
                                      key=lambda row: (row["validation_score"], row["run"]))
                        for objective in ["combined", "phenotype"]}
        winner = min(by_objective.values(), key=lambda row: (row["validation_score"], row["run"]))
        repeats = []
        for seed in SEEDS:
            run = Path(winner["run"])
            run = run.with_name(run.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
            manifest = json.loads((ROOT / run / "manifest.json").read_text())
            assert manifest["complete"] and not manifest["smoke"]
            assert manifest["config"]["objective"] == winner["objective"]
            assert json.loads((ROOT / run / "permissions.json").read_text()) == permissions_by_fold[key]
            repeats.append({"seed": seed, "run": str(run), "single_run": manifest["single_run"]})
        selected[key] = {**winner, "objective_candidates": by_objective, "repeats": repeats}
    assert len(selected) == 10
    destination = output / "source_selection.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous["selected"] == selected
    else:
        result = {"complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_working_tree_modified": True, "selection_seed": 11, "repetition_seeds": SEEDS,
                  "criterion": "Per outer fold, minimize 0.5*Z+0.25*S128+0.25*Sall source-validation residual RMSE; omega minimizes source-validation S128 RMSE.",
                  "candidate_objectives": ["combined", "phenotype"], "candidate_fits_read": len(candidates),
                  "target_predictions_or_metrics_read": False,
                  "target_labels_for_selection": 0, "selection_scope": "Independent within each outer partition; no pooled cross-fold objective selection.",
                  "selected": selected}
        destination.write_text(json.dumps(result, indent=2))
        pd.DataFrame(candidates).to_csv(output / "source_validation_candidates.csv", index=False)
    print(json.dumps({"complete": True, "candidate_fits": len(candidates),
                      "selected_objectives": dict(Counter(row["objective"] for row in selected.values())),
                      "folds": [{"key": key, "objective": row["objective"],
                                 "combined_validation": row["objective_candidates"]["combined"]["validation_score"],
                                 "phenotype_validation": row["objective_candidates"]["phenotype"]["validation_score"]}
                                for key, row in selected.items()]}, indent=2))


if __name__ == "__main__":
    main()
