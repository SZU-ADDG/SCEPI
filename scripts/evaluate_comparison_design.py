import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from train_single import ROOT


ARMS = ("additive", "global_nonlinear", "interaction", "remove_interaction")
CONDITIONS = ("Z", "S128", "Sall")
COLUMNS = [
    "variant_id", "protein_id", "mutation_key", "family_group", "pair_id",
    "main", "margin", "epsilon", "relative_point", "relative_lower", "relative_upper",
    "wt_point", "wt_all_point", "wt_point_in_range", "a_id", "b_id", "wt_variant_id",
    "prediction_mu", "prediction_a", "prediction_b", "prediction_wt", "prediction_epsilon",
    "observed_a", "observed_b", "budget_eligible",
]


def load_predictions(selected):
    split = pd.read_parquet(
        ROOT / "data/splits/combination_splits.parquet",
        columns=["variant_id", "protein_id", "seed", "scheme", "set"],
    )
    pair_test = split.loc[
        split.seed.eq(11) & split.scheme.eq("unseen_pair") & split["set"].eq("test")
    ]
    assert pair_test.variant_id.is_unique
    chunks = []
    for fold in range(5):
        for condition in CONDITIONS:
            merged = None
            for arm in ARMS:
                parent_arm = "interaction" if arm == "remove_interaction" else arm
                evaluation = ROOT / selected[parent_arm][str(fold)]["run"] / "evaluation_test"
                manifest = json.loads((evaluation / "manifest.json").read_text())
                assert manifest["complete"] and manifest["fold"] == fold and manifest["seed"] == 11
                permissions = pd.read_parquet(evaluation / "target_permissions.parquet")
                allowed_ids = set(pair_test.loc[pair_test.protein_id.isin(permissions.index), "variant_id"])
                method = "remove_interaction" if arm == "remove_interaction" else "shared"
                frame = pd.read_parquet(evaluation / f"{method}_{condition}.parquet", columns=COLUMNS)
                frame = frame.loc[frame.variant_id.isin(allowed_ids)].copy()
                assert frame.variant_id.is_unique and set(frame.variant_id) == allowed_ids
                if arm == "additive":
                    merged = frame.drop(columns=["prediction_mu", "prediction_epsilon"]).copy()
                    merged["common_predicted_main"] = frame.prediction_a + frame.prediction_b - frame.prediction_wt
                else:
                    metadata = frame[["variant_id", "protein_id", "family_group", "pair_id", "mutation_key"]]
                    check = merged[metadata.columns].merge(metadata, on="variant_id", validate="one_to_one", suffixes=("_left", "_right"))
                    for name in metadata.columns.drop("variant_id"):
                        assert check[f"{name}_left"].equals(check[f"{name}_right"])
                values = frame[["variant_id", "prediction_mu", "prediction_epsilon"]].rename(
                    columns={"prediction_mu": f"native_{arm}", "prediction_epsilon": f"epsilon_{arm}"}
                )
                merged = merged.merge(values, on="variant_id", validate="one_to_one")
            merged["fold"] = fold
            merged["condition"] = condition
            chunks.append(merged)
    result = pd.concat(chunks, ignore_index=True)
    assert not result.duplicated(["condition", "variant_id"]).any()
    return result


def residual_metrics(frame):
    rows = []
    for (condition, protein), group in frame.groupby(["condition", "protein_id"], sort=True):
        main = group.loc[group.main.fillna(False) & group.budget_eligible]
        if len(main) < 30:
            continue
        for arm in ARMS:
            squared = (main[f"epsilon_{arm}"] - main.epsilon) ** 2
            rows.append({
                "condition": condition, "protein_id": protein,
                "family_group": group.family_group.iloc[0], "arm": arm,
                "epsilon_rmse": float(np.sqrt(squared.mean())),
                "pair_macro_epsilon_rmse": float(squared.groupby(main.pair_id).mean().pow(0.5).mean()),
            })
    return pd.DataFrame(rows)


def evaluate_pool(frame, condition, task, unit, scoring):
    keys = ["protein_id", "pair_id"] if unit == "pair" else ["protein_id"]
    rows = []
    for _, group in frame.groupby(keys, sort=True):
        if len(group) < 100:
            continue
        for arm in ARMS:
            if scoring == "native":
                score = group[f"native_{arm}"]
            else:
                main = group.common_observed_main if scoring == "observed_main" else group.common_predicted_main
                score = main + group[f"epsilon_{arm}"]
            ranked = group.assign(score=score).sort_values(["score", "mutation_key"], ascending=[False, True], kind="stable")
            for budget in (1, 3, 10, 30):
                chosen = ranked.head(budget)
                entry = {
                    "condition": condition, "task": task, "unit": unit, "scoring": scoring,
                    "protein_id": group.protein_id.iloc[0], "family_group": group.family_group.iloc[0],
                    "pair_id": group.pair_id.iloc[0] if unit == "pair" else "all",
                    "arm": arm, "budget": budget, "pool_size": len(group),
                    "certain_beneficial_pool": int(group.relative_lower.ge(0.5).sum()),
                    "possible_beneficial_pool": int(group.relative_upper.ge(0.5).sum()),
                    "precision_lower": float(chosen.relative_lower.ge(0.5).mean()),
                    "precision_upper": float(chosen.relative_upper.ge(0.5).mean()),
                    "selected_variant_ids": chosen.variant_id.tolist(),
                    "selected_scores": chosen.score.tolist(),
                    "selected_relative_point": chosen.relative_point.tolist(),
                }
                if task != "original":
                    assert chosen.relative_point.notna().all()
                    entry["selected_mean_stability"] = float(chosen.relative_point.mean())
                    entry["selected_best_stability"] = float(chosen.relative_point.max())
                if task.startswith("compensation"):
                    recovery = chosen.relative_point - chosen[["single_a", "single_b"]].max(axis=1)
                    entry["selected_mean_recovery"] = float(recovery.mean())
                    entry["compensation_hit_rate"] = float(recovery.ge(0.5).mean())
                    entry["selected_recovery"] = recovery.tolist()
                rows.append(entry)
    return rows


def paired_comparisons(table, keys, metrics, lower_better=False):
    results = []
    for settings, group in table.groupby(keys, sort=True):
        settings = settings if isinstance(settings, tuple) else (settings,)
        for metric in metrics:
            wide = group.pivot(index=["protein_id", "family_group"], columns="arm", values=metric).dropna()
            if not len(wide):
                continue
            for control in ("additive", "global_nonlinear", "remove_interaction"):
                benefit = wide[control] - wide.interaction if lower_better else wide.interaction - wide[control]
                record = dict(zip(keys, settings))
                if "budget" in record:
                    record["budget"] = int(record["budget"])
                record.update({
                    "metric": metric, "control": control, "proteins": len(wide),
                    "control_mean": float(wide[control].mean()), "interaction_mean": float(wide.interaction.mean()),
                    "mean_gain": float(benefit.mean()), "wins": int(benefit.gt(0).sum()),
                    "losses": int(benefit.lt(0).sum()), "ties": int(benefit.eq(0).sum()),
                })
                if lower_better:
                    record["relative_error_reduction"] = float(benefit.mean() / wide[control].mean())
                results.append(record)
    return results


def main():
    output = ROOT / "runs/comparison_design_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((ROOT / "runs/motivation_gate/selected_runs.json").read_text())["models"]
    frame = load_predictions(selected)
    variants = pd.read_parquet(
        ROOT / "data/processed/tsuboyama/variants.parquet", columns=["variant_id", "point"]
    ).set_index("variant_id")
    assert variants.index.is_unique
    frame["single_a"] = frame.a_id.map(variants.point) - frame.wt_point
    frame["single_b"] = frame.b_id.map(variants.point) - frame.wt_point
    frame["common_observed_main"] = frame.single_a + frame.single_b
    measured = frame.loc[frame.main.fillna(False)]
    np.testing.assert_allclose(measured.common_observed_main + measured.epsilon, measured.relative_point, atol=1e-6)
    residual = residual_metrics(frame)
    residual.to_parquet(output / "residual_per_protein.parquet", index=False)
    residual_comparisons = paired_comparisons(
        residual, ["condition"], ["epsilon_rmse", "pair_macro_epsilon_rmse"], lower_better=True
    )
    selection_rows = []
    for condition, subset in frame.groupby("condition", sort=True):
        pool = subset.loc[
            subset.budget_eligible & subset.wt_all_point & subset.wt_point_in_range & subset.wt_point.le(4.5)
        ].copy()
        point = pool.loc[pool.main.fillna(False)].copy()
        tasks = {"original": pool, "point": point}
        scoring_modes = ["native", "predicted_main"]
        if condition == "Sall":
            assert point.observed_a.all() and point.observed_b.all()
            tasks["compensation"] = point.loc[point.single_a.le(-0.5) & point.single_b.le(-0.5)]
            tasks["compensation_margin050"] = tasks["compensation"].loc[tasks["compensation"].margin.ge(0.5)]
            scoring_modes.append("observed_main")
        for task, candidates in tasks.items():
            for scoring in scoring_modes:
                if scoring == "observed_main" and task == "original":
                    continue
                for unit in ("protein", "pair"):
                    selection_rows.extend(evaluate_pool(candidates, condition, task, unit, scoring))
    selection = pd.DataFrame(selection_rows)
    selection.to_parquet(output / "selection_per_decision.parquet", index=False)
    settings = ["condition", "task", "unit", "scoring", "budget"]
    metric_names = [
        "precision_lower", "precision_upper", "selected_mean_stability", "selected_best_stability",
        "selected_mean_recovery", "compensation_hit_rate",
    ]
    macro = selection.groupby(settings + ["protein_id", "family_group", "arm"], as_index=False)[metric_names].mean()
    comparisons = paired_comparisons(macro, settings, metric_names)
    original = selection.loc[
        selection.condition.eq("S128") & selection.task.eq("original") & selection.unit.eq("protein")
        & selection.scoring.eq("native") & selection.arm.eq("interaction") & selection.budget.eq(10)
    ]
    coverage = {
        "original_S128_proteins": len(original),
        "original_S128_no_possible_beneficial": int(original.possible_beneficial_pool.eq(0).sum()),
        "original_S128_no_certain_beneficial": int(original.certain_beneficial_pool.eq(0).sum()),
    }
    summary = {"coverage": coverage, "residual_comparisons": residual_comparisons, "selection_comparisons": comparisons}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "complete": True, "source_commit": source_commit, "script": "scripts/evaluate_comparison_design.py",
        "source_revision": "Working-tree script recorded alongside the cited base commit.",
        "selected_runs": "runs/motivation_gate/selected_runs.json",
        "inputs": ["data/splits/combination_splits.parquet", "data/processed/tsuboyama/variants.parquet"],
        "run_path": str(output), "scope": "Seed11 unseen_pair test intersected with family30 held-out proteins.",
        "status": "Exploratory comparison-design diagnostic; model selection unchanged and training unchanged.",
        "conditions": list(CONDITIONS), "model_selection": "Existing source-validation choices from motivation_gate.",
        "candidate_minimum": 100, "budgets": [1, 3, 10, 30], "stabilization_threshold": 0.5,
        "compensation_pool": "Sall permitted single effects each <= -0.5 kcal/mol; all four endpoints in measurement range.",
        "compensation_hit": "Measured double stability >= better measured single stability + 0.5 kcal/mol.",
        "measurement_boundary": "Compensation task also evaluated with quartet margin >= 0.5 kcal/mol.",
        "outputs": ["summary.json", "residual_per_protein.parquet", "selection_per_decision.parquet"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    concise = [
        row for row in comparisons if row["condition"] in ("S128", "Sall") and row["budget"] == 10
        and row["unit"] == "protein" and row["control"] == "global_nonlinear"
        and row["metric"] in ("precision_lower", "selected_mean_stability", "compensation_hit_rate")
    ]
    print(json.dumps({"coverage": coverage, "residual_comparisons": residual_comparisons, "selection": concise}, indent=2))


if __name__ == "__main__":
    main()
