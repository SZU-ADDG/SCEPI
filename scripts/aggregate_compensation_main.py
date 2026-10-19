import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import OUTPUT

METRICS = ["precision_lower", "precision_upper", "selected_mean_stability", "selected_best_stability",
           "selected_mean_recovery", "selected_best_recovery", "compensation_hit_rate"]
SETTINGS = ["threshold", "condition", "task", "margin", "unit", "scoring", "budget"]


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def comparison(wide, reference, control, metric, lower_better, bootstrap, grid_complete):
    if reference not in wide or control not in wide:
        return None
    paired = wide[[reference, control]].dropna()
    if not len(paired):
        return None
    gain = paired[control] - paired[reference] if lower_better else paired[reference] - paired[control]
    by_protein = gain.groupby(level=["protein_id", "family_group"]).mean()
    by_seed = gain.groupby(level="seed").mean()
    result = {"reference": reference, "control": control, "metric": metric, "proteins": len(by_protein),
              "family_groups": by_protein.index.get_level_values("family_group").nunique(),
              "paired_protein_repeats": len(gain), "evaluated_repeats": len(by_seed),
              "evaluation_grid_complete": bool(grid_complete),
              "reference_mean": float(paired[reference].groupby(level="protein_id").mean().mean()),
              "control_mean": float(paired[control].groupby(level="protein_id").mean().mean()),
              "mean_gain": float(by_protein.mean()), "positive_repeats": int(by_seed.gt(0).sum()),
              "per_repeat_mean_gain": {str(seed): float(value) for seed, value in by_seed.items()},
              "wins": int(by_protein.gt(0).sum()), "losses": int(by_protein.lt(0).sum()), "ties": int(by_protein.eq(0).sum())}
    if lower_better:
        result["relative_error_reduction"] = result["mean_gain"] / result["control_mean"] if result["control_mean"] else None
    if bootstrap and grid_complete and result["family_groups"] >= 2:
        family_sum = by_protein.groupby(level="family_group").sum().to_numpy()
        family_count = by_protein.groupby(level="family_group").size().to_numpy()
        rng = np.random.default_rng(20261010)
        weights = rng.multinomial(len(family_sum), np.full(len(family_sum), 1 / len(family_sum)), size=10000)
        estimates = (weights @ family_sum) / (weights @ family_count)
        result["family_bootstrap_95_interval"] = np.quantile(estimates, [0.025, 0.975]).tolist()
        result["bootstrap_positive_fraction"] = float(np.mean(estimates > 0))
        result["bootstrap_replicates"] = 10000
    return result


def main():
    jobs = json.loads((OUTPUT / "jobs.json").read_text())
    selection_chunks, residual_chunks, completed = [], [], []
    coverage = []
    for job in jobs:
        path = ROOT / job["stages"][0]["manifest"]
        if not path.exists():
            continue
        information = json.loads(path.read_text())
        assert information["complete"] and not information["smoke"]
        directory = path.parent
        selected = pd.read_parquet(directory / "selection_per_decision.parquet",
                                   columns=SETTINGS + ["protein_id", "family_group", "arm", "seed", "fold", "pool_size"] + METRICS)
        if len(selected):
            selected = selected.groupby(SETTINGS + ["protein_id", "family_group", "arm", "seed", "fold"], as_index=False)[METRICS].mean()
            selection_chunks.append(selected)
        residual = pd.read_parquet(directory / "residual_per_protein.parquet")
        if len(residual):
            residual_chunks.append(residual)
        completed.append(job["name"])
        coverage.append({key: information[key] for key in ["arm", "threshold", "seed", "fold", "condition", "primary_proteins", "primary_selected_candidates", "primary_compensation_hits", "excluded_proteins"]})
    selection = pd.concat(selection_chunks, ignore_index=True) if selection_chunks else pd.DataFrame()
    residual = pd.concat(residual_chunks, ignore_index=True) if residual_chunks else pd.DataFrame()
    annotation = pd.concat([
        pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet", columns=["protein_id", "pfam_annotated"]).assign(threshold=threshold)
        for threshold in [30, 25]
    ], ignore_index=True)
    if len(selection):
        selection = selection.merge(annotation, on=["threshold", "protein_id"], validate="many_to_one")
    if len(residual):
        residual = residual.merge(annotation, on=["threshold", "protein_id"], validate="many_to_one")
    selection.to_parquet(OUTPUT / "selection_per_protein.parquet", index=False)
    residual.to_parquet(OUTPUT / "residual_per_protein.parquet", index=False)
    finished_grids = pd.DataFrame(coverage).groupby(["threshold", "condition", "arm"]).size() if coverage else pd.Series(dtype=int)
    comparisons = []
    for table, keys, metrics, lower_better in [
        (selection, SETTINGS, METRICS, False),
        (residual, ["threshold", "condition", "margin"], ["epsilon_rmse", "pair_macro_epsilon_rmse"], True),
    ]:
        if not len(table):
            continue
        groups = [("all_groups", values, group) for values, group in table.groupby(keys, sort=True)]
        groups += [("annotated_families", values, group) for values, group in table.loc[table.pfam_annotated].groupby(keys, sort=True)]
        for family_scope, values, group in groups:
            configuration = {key: native(value) for key, value in zip(keys, values)}
            configuration["family_scope"] = family_scope
            references = ["interaction", "geometry"] if configuration["condition"] == "Sall" else ["interaction_C", "interaction_target_only_C"]
            present = set(group.arm)
            if not present.intersection(references):
                continue
            panels = {metric: group.pivot(index=["seed", "protein_id", "family_group"], columns="arm", values=metric)
                      for metric in metrics}
            for reference in references:
                if reference not in present:
                    continue
                controls = present - {reference}
                if reference == "interaction_target_only_C":
                    controls &= {"mochi_order1_C", "mochi_order2_C", "mean_ridge_target_C", "mean_mlp_target_C", "kermut_C"}
                for control in sorted(controls):
                    for metric in metrics:
                        bootstrap = (not lower_better and configuration["task"] == "compensation"
                                     and configuration["margin"] in [0.0, 0.5] and configuration["unit"] == "protein"
                                     and configuration["scoring"] == "observed_main" and configuration["budget"] == 10
                                     and metric == "compensation_hit_rate")
                        grid_complete = (finished_grids.get((configuration["threshold"], configuration["condition"], reference), 0) == 25
                                         and finished_grids.get((configuration["threshold"], configuration["condition"], control), 0) == 25)
                        item = comparison(panels[metric], reference, control, metric, lower_better, bootstrap, grid_complete)
                        if item is not None:
                            comparisons.append({**configuration, **item})
    (OUTPUT / "paired_summary.json").write_text(json.dumps(comparisons, indent=2))
    (OUTPUT / "evaluation_coverage.json").write_text(json.dumps(coverage, indent=2))
    information = {"complete": len(completed) == len(jobs), "state": "complete" if len(completed) == len(jobs) else "running",
                   "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "protocol": "runs/compensation_main/protocol.json", "run_path": str(OUTPUT),
                   "expected_evaluations": len(jobs), "completed_evaluations": len(completed),
                   "remaining_evaluations": len(jobs) - len(completed), "paired_comparisons": len(comparisons),
                   "selection_protein_repeats": len(selection), "residual_protein_repeats": len(residual),
                   "statistics": "Equal-protein paired gains after averaging repeats; 10000 family-cluster bootstrap samples for primary budget10 compensation and margin0.5; intervals descriptive.",
                   "outputs": ["selection_per_protein.parquet", "residual_per_protein.parquet", "paired_summary.json", "evaluation_coverage.json"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(information, indent=2))
    print(json.dumps(information), flush=True)


if __name__ == "__main__":
    main()
