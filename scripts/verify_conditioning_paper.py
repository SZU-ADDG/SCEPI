import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/conditioning_factorial_20261021"
SOURCE = STUDY / "paper_evidence"
FIELDS = ["threshold", "task", "margin", "unit", "scoring", "budget", "metric", "reference", "control"]
LABELS = {
    "interaction": "SCEPI", "additive": "Additive", "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D", "thermo_additive": "ThermoMPNN additive", "kermut": "Kermut",
    "mean_mlp": "Mean MLP", "mean_ridge": "Mean ridge", "esm2_additive": "ESM-2 additive",
    "esm2_context": "ESM-2 context", "remove_interaction": "Pair removed", "no_cross": "Local context",
    "reference_context": "Wild-type context", "identity_features": "Identity features",
    "phenotype_only": "Phenotype loss", "residual_only": "Residual loss", "frozen_main": "Frozen singles",
    "identity_mapping": "Identity map", "geometry": "Structure",
    "geometry_permuted_inference": r"Permuted distances\textsuperscript{I}",
    "geometry_permuted_retrained": r"Permuted distances\textsuperscript{R}",
    "geometry_rewired_inference": r"Rewired distances\textsuperscript{I}",
    "geometry_rewired_retrained": r"Rewired distances\textsuperscript{R}",
    "stabilityarc_kermut": "StabilityArc + Kermut", "direct_residual": "Direct residual head",
    "no_pair_measured": "Predicted singles", "values_only": "Values only", "mask_only": "Indicators only",
    "joint_objective": "Joint objective", "joint_remove_interaction": "Joint pair removed",
    "joint_direct_residual": "Joint direct head", "joint_no_pair_measured": "Joint predicted singles",
}


def table_text(manuscript, label):
    marker = "\\label{" + label + "}"
    assert manuscript.count(marker) == 1
    return manuscript.split(marker, 1)[1].split(r"\end{table}", 1)[0]


def number(value, precision):
    value = 0.0 if abs(value) < 1e-12 else value
    result = f"{value:.{precision}f}"
    return "$" + result + "$" if value < 0 else result


def interval(row, scale, precision):
    lower, upper = row["gain_interval_95"]
    return "[" + number(scale * lower, precision) + ", " + number(scale * upper, precision) + "]"


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["complete"] and manifest["completed_evaluations"] == 1600
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    records = json.loads((SOURCE / "paired_summary.json").read_text())
    indexed = {tuple(row.get(field) for field in FIELDS): row for row in records}
    assert len(indexed) == len(records) == manifest["paired_comparisons"]
    means_checked, table_rows, conditioning_rows = 0, 0, 0
    maximum = 0.0
    tables = {}
    for threshold, label in [(30, "tab:allcomponents"), (25, "tab:threshold25")]:
        directory = SOURCE / f"verified_family{threshold}"
        independent = json.loads((directory / "paired_summary.json").read_text())
        for row in independent:
            original = indexed[tuple(row.get(field) for field in FIELDS)]
            for field in ["reference_mean", "control_mean", "mean_gain"]:
                difference = abs(row[field] - original[field])
                assert difference < 1e-12
                maximum = max(maximum, difference)
            for field in ["proteins", "family_groups", "paired_protein_repeats", "repeats", "wins", "ties", "losses"]:
                assert row[field] == original[field]
            for seed, value in row["repeat_gains"].items():
                np.testing.assert_allclose(value, original["repeat_gains"][seed], rtol=0, atol=1e-12)
            means_checked += 1
        tables[threshold] = pd.read_csv(directory / "paper_table.csv", index_col="arm")
        assert set(tables[threshold].index) == set(LABELS)
        fragment = table_text(manuscript, label)
        for arm, values in tables[threshold].iterrows():
            expected = (f"{LABELS[arm]} & {values.phenotype_rmse:.3f} & {values.epsilon_rmse:.3f}"
                        f" & {100 * values.compensation_hit_rate:.2f} & ${values.selected_mean_recovery:.3f}$")
            assert expected in fragment, (threshold, expected)
            table_rows += 1
    assert means_checked == len(records)
    fragment = table_text(manuscript, "tab:conditioning")
    for arm in ["interaction", "values_only", "mask_only", "no_pair_measured", "direct_residual"]:
        first, second = tables[30].loc[arm], tables[25].loc[arm]
        expected = (f"{LABELS[arm]} & {first.epsilon_rmse:.3f} & {second.epsilon_rmse:.3f}"
                    f" & {100 * first.compensation_hit_rate:.2f} & {100 * second.compensation_hit_rate:.2f}")
        assert expected in fragment, expected
        conditioning_rows += 1
    interval_rows = 0
    fragment = table_text(manuscript, "tab:pairfunctionuncertainty")
    for threshold in [30, 25]:
        for control in ["no_pair_measured", "values_only", "mask_only", "direct_residual"]:
            residual = next(row for row in records if row["threshold"] == threshold and row["control"] == control
                            and row["metric"] == "epsilon_rmse" and row["margin"] == 0)
            native = next(row for row in records if row["threshold"] == threshold and row["control"] == control
                          and row["metric"] == "compensation_hit_rate" and row["margin"] == 0
                          and row["task"] == "compensation" and row["budget"] == 10
                          and row["scoring"] == "native" and row["unit"] == "protein")
            precision = 5 if threshold == 30 and control == "values_only" else 3
            expected = (f"{threshold}\\% & {LABELS[control]} & {number(residual['mean_gain'], 3)} {interval(residual, 1, precision)}"
                        f" & {number(100 * native['mean_gain'], 2)} {interval(native, 100, 2)}")
            assert expected in fragment, expected
            interval_rows += 1
    for label, metric, margin, scoring in [
        ("tab:nativeuncertainty", "compensation_hit_rate", 0, "native"),
        ("tab:marginuncertainty", "compensation_hit_rate", 0.5, "observed_main"),
        ("tab:recoveryuncertainty", "selected_mean_recovery", 0, "native"),
    ]:
        fragment = table_text(manuscript, label)
        scale, precision = (1, 3) if metric == "selected_mean_recovery" else (100, 2)
        for control in ["additive", "global_nonlinear", "thermo_epistatic", "kermut", "stabilityarc_kermut", "mean_mlp", "mean_ridge"]:
            row = next(row for row in records if row["threshold"] == 30 and row["control"] == control
                       and row["metric"] == metric and row["margin"] == margin
                       and row["task"] == "compensation" and row["budget"] == 10
                       and row["scoring"] == scoring and row["unit"] == "protein")
            expected = f"{LABELS[control]} & {number(scale * row['mean_gain'], precision)} & {interval(row, scale, precision)}"
            if label == "tab:nativeuncertainty":
                expected += f" & {row['bootstrap_positive_fraction']:.4f}"
            elif label == "tab:recoveryuncertainty":
                expected += f" & {row['wins']}"
            assert expected in fragment, expected
            interval_rows += 1
    primary_arms = {"interaction", "additive", "global_nonlinear", "thermo_epistatic", "kermut",
                    "stabilityarc_kermut", "mean_mlp", "mean_ridge"}
    for name in ["phenotype", "residual", "selection"]:
        full = pd.read_parquet(SOURCE / (name + "_per_protein.parquet"))
        primary = pd.read_parquet(STUDY / "primary_evidence" / (name + "_per_protein.parquet"))
        current = full.loc[full.arm.isin(primary_arms), primary.columns]
        keys = ["threshold", "arm", "protein_id", "seed"]
        keys += ["margin"] if name == "residual" else ["task", "margin", "scoring", "budget"] if name == "selection" else []
        pd.testing.assert_frame_equal(current.sort_values(keys).reset_index(drop=True),
                                      primary.sort_values(keys).reset_index(drop=True), check_dtype=False)
    for name in ["method_overview", "compensation_discovery", "prediction_comparison", "component_analysis"]:
        fragment = (ROOT / "paper/figures" / (name + ".tikz")).read_text().strip()
        assert manuscript.count(fragment) == 1, name
    assert manuscript.count(r"\begin{tikzpicture}") == manuscript.count(r"\end{tikzpicture}") == 4
    budgets = json.loads((STUDY / "single_label_budgets.json").read_text())
    fragment = table_text(manuscript, "tab:singlebudgets")
    for row in budgets:
        if row["threshold"] == 30:
            condition = row["condition"].replace("Sall", "S-all")
            assert f"{condition} & {row['residual_rmse']:.3f} & " in fragment
    subset = pd.read_csv(STUDY / "site_pair_subset/paper_table.csv", index_col="arm")
    fragment = table_text(manuscript, "tab:pairsubset")
    for arm in ["interaction", "thermo_epistatic", "kermut", "global_nonlinear", "additive"]:
        row = subset.loc[arm]
        expected = f"{LABELS[arm]} & {row.epsilon_rmse:.3f} & {100 * row.compensation_hit_rate:.2f} & ${row.selected_mean_recovery:.3f}$"
        assert expected in fragment, expected
    result = {"complete": True, "source_commit": manifest["source_commit"], "source_run_path": str(SOURCE),
              "paired_comparisons_checked": means_checked, "maximum_mean_difference": maximum,
              "manuscript_table_rows_checked": table_rows, "main_conditioning_table_rows_checked": conditioning_rows,
              "paired_interval_table_rows_checked": interval_rows,
              "primary_figure_snapshot_matches_final_evidence": True, "native_figure_fragments_embedded": 4,
              "source_selected_budget_table_verified": True, "source_selected_site_pair_subset_verified": True}
    (STUDY / "primary_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
