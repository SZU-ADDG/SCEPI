import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import frozen_table, input_records, normalized_predictions, residual_metrics, selection_metrics
from scripts import paper_full_target_evidence as evidence_tools
from scripts.run_sall_full_target import paired_summary
OUTPUT = ROOT / "runs/conditioning_factorial_20261021"
EVIDENCE = OUTPUT / "paper_evidence"
CELLS = ["interaction", "values_only", "mask_only", "no_pair_measured"]
CONTRASTS = [
    ("joint_input", "interaction", "no_pair_measured"),
    ("values_with_mask", "interaction", "mask_only"),
    ("values_without_mask", "values_only", "no_pair_measured"),
    ("mask_with_values", "interaction", "values_only"),
    ("mask_without_values", "mask_only", "no_pair_measured"),
]


def describe(gain, rng, draws):
    assert np.isfinite(gain.to_numpy()).all()
    sums = gain.groupby("family_group", observed=True).sum()
    sizes = gain.groupby("family_group", observed=True).size().reindex(sums.index)
    key = tuple(zip(sums.index.astype(str), sizes.astype(int)))
    if key not in draws:
        draws[key] = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=10000)
    weights = draws[key]
    bootstrap = (weights @ sums.to_numpy()) / (weights @ sizes.to_numpy())
    group_means = sums / sizes
    largest = sizes.sort_values(ascending=False, kind="stable").index[0]
    leave_one = (gain.sum() - sums) / (len(gain) - sizes)
    return {"proteins": len(gain), "family_groups": len(sums), "mean_gain": float(gain.mean()),
            "gain_interval_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "bootstrap_positive_fraction": float((bootstrap > 0).mean()),
            "equal_group_mean_gain": float(group_means.mean()),
            "largest_group": str(largest), "largest_group_proteins": int(sizes.loc[largest]),
            "without_largest_group_gain": float(leave_one.loc[largest]),
            "leave_one_group_out_minimum": float(leave_one.min()),
            "leave_one_group_out_maximum": float(leave_one.max()),
            "wins": int(gain.gt(1e-12).sum()), "losses": int(gain.lt(-1e-12).sum()),
            "ties": int(gain.abs().le(1e-12).sum())}


def panels(tables):
    for threshold in [30, 25]:
        for margin in [0.0, 0.5]:
            subset = tables["residual"].loc[tables["residual"].threshold.eq(threshold) & tables["residual"].margin.eq(margin)]
            yield {"threshold": threshold, "margin": margin, "metric": "epsilon_rmse"}, subset, "epsilon_rmse", False
            for scoring in ["native", "observed_main"]:
                subset = tables["selection"].loc[
                    tables["selection"].threshold.eq(threshold) & tables["selection"].margin.eq(margin)
                    & tables["selection"].task.eq("compensation") & tables["selection"].unit.eq("protein")
                    & tables["selection"].budget.eq(10) & tables["selection"].scoring.eq(scoring)]
                for metric in ["compensation_hit_rate", "selected_mean_recovery"]:
                    yield {"threshold": threshold, "margin": margin, "scoring": scoring, "budget": 10, "metric": metric}, subset, metric, True


def coverage(cohort_evidence=EVIDENCE):
    features = ROOT / "data/features/training/tsuboyama"
    singles = pd.read_parquet(features / "singles/metadata.parquet")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    assert order.groupby("variant_id").seed.nunique().eq(5).all()
    known = set(order.variant_id)
    singles["available"] = singles.variant_id.isin(known)
    counts = singles.groupby("protein_id").agg(accepted_singles=("variant_id", "size"), available_singles=("available", "sum"))
    frames = []
    for threshold in [30, 25]:
        family = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet")
        table = family.merge(counts, on="protein_id", validate="one_to_one")
        table["threshold"] = threshold
        table["S32_coverage"] = table.available_singles.clip(upper=32) / table.accepted_singles
        table["S128_coverage"] = table.available_singles.clip(upper=128) / table.accepted_singles
        table["Sall_coverage"] = table.available_singles / table.accepted_singles
        frames.append(table)
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(OUTPUT / "single_label_coverage.parquet", index=False)
    summary = []
    for threshold in [30, 25]:
        for fold in range(5):
            truth = pd.read_parquet(ROOT / "runs/sall_full_target/truth" / f"family{threshold}/fold{fold}.parquet")
            assert truth.loc[truth.main.fillna(False), ["known_single_a", "known_single_b"]].all().all()
        for cohort, name in [("residual", "residual"), ("selection", "selection")]:
            cohort_table = pd.read_parquet(cohort_evidence / (name + "_per_protein.parquet"))
            selected = cohort_table.loc[cohort_table.threshold.eq(threshold) & cohort_table.arm.eq("interaction") & cohort_table.margin.eq(0)]
            if cohort == "selection":
                selected = selected.loc[selected.task.eq("compensation") & selected.scoring.eq("native") & selected.budget.eq(10)]
            proteins = set(selected.protein_id)
            current = table.loc[table.threshold.eq(threshold) & table.protein_id.isin(proteins)]
            assert len(current) == len(proteins)
            summary.append({"threshold": threshold, "cohort": cohort, "proteins": len(current),
                            "accepted_singles_median": float(current.accepted_singles.median()),
                            "available_singles_median": float(current.available_singles.median()),
                            "S32_fraction_median": float(current.S32_coverage.median()),
                            "S128_fraction_median": float(current.S128_coverage.median()),
                            "Sall_fraction_median": float(current.Sall_coverage.median()),
                            "Sall_fraction_minimum": float(current.Sall_coverage.min())})
    return summary


def single_label_budgets():
    selection = json.loads((OUTPUT / "source_selection.json").read_text())
    chunks = []
    for row in selection["selected"].values():
        for repeat in row["repeats"]:
            table = pd.read_parquet(ROOT / repeat["run"] / "evaluation_test/per_protein.parquet")
            table = table.loc[table.method.eq("shared") & table.main_eligible].copy()
            table["threshold"] = row["threshold"]
            chunks.append(table)
    table = pd.concat(chunks, ignore_index=True)
    assert not table.duplicated(["threshold", "condition", "protein_id", "seed"]).any()
    assert table.groupby(["threshold", "condition", "protein_id"]).seed.nunique().eq(5).all()
    records = []
    for (threshold, condition), group in table.groupby(["threshold", "condition"], sort=True):
        per_protein = group.groupby("protein_id").epsilon_rmse.mean()
        assert len(per_protein) == 155
        records.append({"threshold": int(threshold), "condition": condition, "proteins": len(per_protein),
                        "protein_repetition_records": len(group), "residual_rmse": float(per_protein.mean())})
    primary = pd.read_parquet(EVIDENCE / "residual_per_protein.parquet")
    for threshold in [30, 25]:
        group = primary.loc[primary.arm.eq("interaction") & primary.threshold.eq(threshold) & primary.margin.eq(0)]
        score = group.groupby("protein_id").epsilon_rmse.mean().mean()
        result = next(row["residual_rmse"] for row in records if row["threshold"] == threshold and row["condition"] == "Sall")
        np.testing.assert_allclose(result, score, rtol=0, atol=1e-12)
    table.to_parquet(OUTPUT / "single_label_budget_per_protein.parquet", index=False)
    (OUTPUT / "single_label_budgets.json").write_text(json.dumps(records, indent=2))
    return records


def group_sizes(tables):
    records = []
    for threshold in [30, 25]:
        for name in ["residual", "selection"]:
            table = tables[name]
            selected = table.loc[table.arm.eq("interaction") & table.threshold.eq(threshold) & table.margin.eq(0)]
            if name == "selection":
                selected = selected.loc[selected.task.eq("compensation") & selected.scoring.eq("native") & selected.budget.eq(10)]
            proteins = selected[["protein_id", "family_group"]].drop_duplicates()
            assert not proteins.protein_id.duplicated().any()
            sizes = proteins.groupby("family_group", observed=True).size()
            records.append({"threshold": threshold, "cohort": name, "proteins": len(proteins), "groups": len(sizes),
                            "minimum": int(sizes.min()), "median": float(sizes.median()), "maximum": int(sizes.max()),
                            "quartiles": sizes.quantile([0.25, 0.75]).tolist(),
                            "sizes": {str(key): int(value) for key, value in sizes.items()}})
    (OUTPUT / "homology_group_sizes.json").write_text(json.dumps(records, indent=2))
    return records


def site_pair_subset():
    chunks = {"residual": [], "selection": []}
    sources = {"interaction": "final_interaction", "joint_objective": "interaction",
               "thermo_epistatic": "thermo_epistatic", "kermut": "kermut",
               "global_nonlinear": "global_nonlinear", "additive": "additive"}
    for arm, source in sources.items():
        for seed in [11, 23, 47, 89, 131]:
            for fold in range(5):
                truth = frozen_table(30, seed, fold)
                entries, missing = input_records(source, 30, seed, fold, "Sall", truth)
                assert not missing, missing
                frame, excluded, _ = normalized_predictions(source, 30, seed, fold, "Sall", truth, entries)
                assert not excluded
                for name, table in [("residual", residual_metrics(frame)), ("selection", selection_metrics(frame, units=("protein",)))]:
                    for key, value in {"arm": arm, "threshold": 30, "seed": seed, "fold": fold, "condition": "Sall"}.items():
                        table[key] = value
                    chunks[name].append(table)
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in chunks.items()}
    coverage_counts = {(30, arm): 25 for arm in sources}
    rows = []
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        rows += paired_summary(tables["residual"], ["threshold", "margin"], metric, False, coverage_counts)
    for metric in ["compensation_hit_rate", "selected_mean_recovery"]:
        rows += paired_summary(tables["selection"], ["threshold", "task", "margin", "unit", "scoring", "budget"], metric, True, coverage_counts)
    evidence_tools.bootstrap(rows, tables)
    directory = OUTPUT / "site_pair_subset"
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(directory / (name + "_per_protein.parquet"), index=False)
    (directory / "paired_summary.json").write_text(json.dumps(rows, indent=2))
    residual = tables["residual"].loc[tables["residual"].margin.eq(0)]
    native = tables["selection"].loc[tables["selection"].task.eq("compensation") & tables["selection"].margin.eq(0)
                                    & tables["selection"].scoring.eq("native") & tables["selection"].budget.eq(10)]
    means = pd.concat([residual.groupby(["arm", "protein_id"]).epsilon_rmse.mean().groupby("arm").mean(),
                       native.groupby(["arm", "protein_id"])[["compensation_hit_rate", "selected_mean_recovery"]].mean().groupby("arm").mean()], axis=1)
    means.to_csv(directory / "paper_table.csv")
    assert abs(means.loc["joint_objective", "epsilon_rmse"] - 0.549) < 0.0005
    assert abs(100 * means.loc["joint_objective", "compensation_hit_rate"] - 5.75) < 0.005
    result = {"complete": True, "arms": list(sources), "matched_evaluations": 150,
              "source_selected_final_model": True, "original_joint_recipe_reconstruction_matches": True,
              "target_double_training_labels": 0,
              "residual_proteins": residual.loc[residual.arm.eq("interaction")].protein_id.nunique(),
              "selection_proteins": native.loc[native.arm.eq("interaction")].protein_id.nunique(),
              "residual_protein_repetitions": len(residual.loc[residual.arm.eq("interaction")]),
              "selection_protein_repetitions": len(native.loc[native.arm.eq("interaction")]),
              "bootstrap": {"unit": "homology group", "draws": 10000, "seed": 20261013}}
    assert result["residual_proteins"] == 70 and result["selection_proteins"] == 63
    (directory / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert manifest["complete"] and json.loads((EVIDENCE / "integration_verification.json").read_text())["complete"]
    tables = {name: pd.read_parquet(EVIDENCE / (name + "_per_protein.parquet")) for name in ["selection", "residual"]}
    rng, draws = np.random.default_rng(20261021), {}
    factorial, sensitivity, individual = [], [], []
    controls = ["no_pair_measured", "values_only", "mask_only", "direct_residual", "remove_interaction",
                "joint_objective", "phenotype_only", "mean_ridge", "thermo_epistatic", "kermut", "stabilityarc_kermut"]
    for settings, subset, metric, higher in panels(tables):
        wide = subset.pivot(index=["protein_id", "family_group", "seed"], columns="arm", values=metric)
        assert all(cell in wide for cell in CELLS)
        repeated = wide[CELLS].dropna()
        assert repeated.groupby(["protein_id", "family_group"]).size().eq(5).all()
        means = repeated.groupby(["protein_id", "family_group"]).mean()
        for contrast, reference, control in CONTRASTS:
            gain = means[reference] - means[control] if higher else means[control] - means[reference]
            repeat_gain = (repeated[reference] - repeated[control] if higher else repeated[control] - repeated[reference]).groupby("seed").mean()
            factorial.append({**settings, "contrast": contrast, "reference": reference, "control": control,
                              "reference_mean": float(means[reference].mean()), "control_mean": float(means[control].mean()),
                              **describe(gain, rng, draws), "repeat_gains": {str(seed): float(value) for seed, value in repeat_gain.items()}})
            records = gain.rename("gain").reset_index()
            for key, value in {**settings, "contrast": contrast}.items():
                records[key] = value
            individual.append(records)
        if higher:
            interaction = means.interaction - means.mask_only - means.values_only + means.no_pair_measured
        else:
            interaction = means.mask_only - means.interaction - means.no_pair_measured + means.values_only
        factorial.append({**settings, "contrast": "difference_of_conditional_gains", **describe(interaction, rng, draws)})
        for control in controls:
            paired = wide[["interaction", control]].dropna()
            assert paired.groupby(["protein_id", "family_group"]).size().eq(5).all()
            current = paired.groupby(["protein_id", "family_group"]).mean()
            gain = current.interaction - current[control] if higher else current[control] - current.interaction
            sensitivity.append({**settings, "reference": "interaction", "control": control,
                                "reference_mean": float(current.interaction.mean()), "control_mean": float(current[control].mean()),
                                **describe(gain, rng, draws)})
    (OUTPUT / "factorial_summary.json").write_text(json.dumps(factorial, indent=2))
    pd.DataFrame(factorial).to_csv(OUTPUT / "factorial_summary.csv", index=False)
    pd.concat(individual, ignore_index=True).to_parquet(OUTPUT / "factorial_per_protein.parquet", index=False)
    (OUTPUT / "group_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    group_distribution = group_sizes(tables)
    single_coverage = coverage()
    (OUTPUT / "single_label_coverage.json").write_text(json.dumps(single_coverage, indent=2))
    budgets = single_label_budgets()
    subset = site_pair_subset()
    command = [sys.executable, "scripts/analyze_random_enrichment.py", "--evidence", str(EVIDENCE),
               "--output", str(OUTPUT / "random_enrichment")]
    with (OUTPUT / "random_enrichment.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = {"complete": True, "factorial_contrasts": len(factorial), "group_sensitivity_comparisons": len(sensitivity),
              "source_selection": str(OUTPUT / "source_selection.json"), "single_label_coverage": single_coverage,
              "homology_group_sizes": group_distribution,
              "source_selected_label_budgets": budgets, "source_selected_site_pair_subset": subset,
              "aggregation": "Average five matched fits within each protein, then average proteins equally; additional equal-group and leave-one-group-out sensitivity.",
              "bootstrap": {"unit": "homology group", "draws": 10000, "seed": 20261021},
              "new_training_in_this_analysis": False}
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
