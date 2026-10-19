import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "work/sall_full_target"
FIELDS = ["threshold", "task", "margin", "unit", "scoring", "budget", "metric", "reference", "control"]
LABELS = {
    "interaction": "SCEPI", "additive": "Additive", "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D", "thermo_additive": "ThermoMPNN additive", "kermut": "Kermut",
    "mean_mlp": "Mean MLP", "mean_ridge": "Mean ridge", "esm2_additive": "ESM-2 additive",
    "esm2_context": "ESM-2 context", "remove_interaction": "Pair removed",
    "no_cross": "Local context", "reference_context": "Wild-type context",
    "identity_features": "Identity features", "phenotype_only": "Phenotype loss",
    "residual_only": "Residual loss", "frozen_main": "Frozen singles",
    "identity_mapping": "Identity map", "geometry": "Structure",
    "geometry_permuted_inference": "Permuted distances\\textsuperscript{I}",
    "geometry_permuted_retrained": "Permuted distances\\textsuperscript{R}",
    "geometry_rewired_inference": "Rewired distances\\textsuperscript{I}",
    "geometry_rewired_retrained": "Rewired distances\\textsuperscript{R}",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "direct_residual": "Direct residual head",
    "no_pair_measured": "Predicted singles",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=EVIDENCE / "paper_evidence")
    parser.add_argument("--family30", type=Path, default=EVIDENCE / "paper_verified")
    parser.add_argument("--family25", type=Path, default=EVIDENCE / "family25_verified")
    parser.add_argument("--original", type=Path, default=ROOT / "work/method_reporting_20261015")
    args = parser.parse_args()
    evidence_manifest = json.loads((args.input / "manifest.json").read_text())
    remote = json.loads((args.input / "paired_summary.json").read_text())
    reference = {tuple(row.get(field) for field in FIELDS): row for row in remote}
    assert len(reference) == len(remote) == evidence_manifest["paired_comparisons"]
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    maximum_difference, comparisons, table_rows, interval_rows, summaries = 0.0, 0, 0, 0, []
    tables = {}
    for threshold, directory in [(30, args.family30), (25, args.family25)]:
        rows = json.loads((directory / "paired_summary.json").read_text())
        assert len(rows) == sum(row["threshold"] == threshold for row in remote)
        for row in rows:
            original = reference[tuple(row.get(field) for field in FIELDS)]
            for field in ["proteins", "family_groups", "paired_protein_repeats", "complete_grid", "repeats"]:
                assert row[field] == original[field], (threshold, field, row["metric"], row["control"])
            for field in ["reference_mean", "control_mean", "mean_gain"]:
                difference = abs(row[field] - original[field])
                assert difference <= 1e-12, (threshold, field, difference)
                maximum_difference = max(maximum_difference, difference)
            for seed, value in row["repeat_gains"].items():
                assert abs(value - original["repeat_gains"][seed]) <= 1e-12
            assert row["wins"] + row["ties"] + row["losses"] == row["proteins"]
            comparisons += 1
        table = pd.read_csv(directory / "paper_table.csv", index_col="arm")
        tables[threshold] = table
        declared = {row["arm"] for row in evidence_manifest["completed_grids"]
                    if row["threshold"] == threshold and row["evaluations"] == 25}
        assert set(table.index) == declared and declared <= set(LABELS)
        for arm, values in table.iterrows():
            displayed = (f"{LABELS[arm]} & {values.phenotype_rmse:.3f} & {values.epsilon_rmse:.3f}"
                         f" & {100 * values.compensation_hit_rate:.2f} & ${values.selected_mean_recovery:.3f}$")
            assert displayed in manuscript, (threshold, displayed)
            table_rows += 1
        for arm, label in {"direct_residual": "Direct residual", "no_pair_measured": "Predicted singles"}.items():
            if arm not in declared:
                continue
            residual = [row for row in rows if row["control"] == arm
                        and row["metric"] == "epsilon_rmse" and row["margin"] == 0]
            selection = [row for row in rows if row["control"] == arm
                         and row.get("task") == "compensation" and row.get("unit") == "protein"
                         and row.get("scoring") == "native" and row.get("budget") == 10
                         and row["metric"] == "compensation_hit_rate" and row["margin"] == 0]
            assert len(residual) == len(selection) == 1
            displayed = f"{threshold}\\% & {label}"
            for row, scale, digits in [(residual[0], 1, 3), (selection[0], 100, 2)]:
                values = [scale * row["mean_gain"]] + [scale * value for value in row["gain_interval_95"]]
                numbers = [f"${value:.{digits}f}$" if value < 0 else f"{value:.{digits}f}" for value in values]
                displayed += f" & {numbers[0]} [{numbers[1]}, {numbers[2]}]"
            assert displayed in manuscript, (threshold, displayed)
            interval_rows += 1
        summaries.append({"threshold": threshold, "grids": 25 * len(table), "comparisons": len(rows), "table_rows": len(table)})
    conditioning_rows = 0
    for arm, label, emphasized in [("interaction", "SCEPI", {0, 1, 2}),
                                    ("no_pair_measured", "Predicted singles", set()),
                                    ("direct_residual", "Direct residual head", {3})]:
        primary, stricter = tables[30].loc[arm], tables[25].loc[arm]
        values = [primary.epsilon_rmse, stricter.epsilon_rmse,
                  100 * primary.compensation_hit_rate, 100 * stricter.compensation_hit_rate]
        numbers = [f"{value:.{3 if index < 2 else 2}f}" for index, value in enumerate(values)]
        numbers = ["\\textbf{" + value + "}" if index in emphasized else value
                   for index, value in enumerate(numbers)]
        displayed = label + " & " + " & ".join(numbers)
        assert displayed in manuscript, displayed
        conditioning_rows += 1
    conditioning_reductions = [100 * (table.loc["no_pair_measured", "epsilon_rmse"]
                                     - table.loc["interaction", "epsilon_rmse"])
                               / table.loc["no_pair_measured", "epsilon_rmse"]
                               for table in [tables[30], tables[25]]]
    for value in conditioning_reductions:
        assert f"{value:.1f}\\%" in manuscript
    for figure in ["method_overview", "compensation_discovery", "prediction_comparison", "component_analysis"]:
        fragment = (ROOT / "paper/figures" / (figure + ".tikz")).read_text().strip()
        assert manuscript.count(fragment) == 1, figure
    assert manuscript.count("\\begin{tikzpicture}") == manuscript.count("\\end{tikzpicture}") == 4
    for name, proteins, repetitions in [("phenotype", 159, 795), ("residual", 155, 775), ("selection", 120, 600)]:
        table = pd.read_parquet(args.input / (name + "_per_protein.parquet"))
        table = table.loc[table.arm.eq("interaction") & table.threshold.eq(30)]
        if name == "residual":
            table = table.loc[table.margin.eq(0)]
        if name == "selection":
            table = table.loc[table.task.eq("compensation") & table.margin.eq(0) & table.budget.eq(10)
                              & table.scoring.eq("native")]
        assert table.protein_id.nunique() == proteins and len(table) == repetitions
        assert table.groupby("protein_id").seed.nunique().eq(5).all()
    original = json.loads((args.original / "original_comparisons.json").read_text())
    indexed = {(row["scheme"], row["condition"], row["control"]): row for row in original}
    for condition, label in [("Z", "Z"), ("S32", "S32"), ("S128", "S128"), ("Sall", "S-all")]:
        controls = ["additive", "global_nonlinear", "thermo_epistatic"]
        values = [indexed[("family", condition, arm)] for arm in controls]
        assert all(abs(row["reference_mean"] - values[0]["reference_mean"]) < 1e-12 for row in values)
        line = f"{label} & {values[0]['reference_mean']:.3f}" + "".join(f" & {row['control_mean']:.3f}" for row in values)
        line += " & ---" if condition == "Z" else f" & {indexed[('family', condition, 'kermut')]['control_mean']:.3f}"
        assert line in manuscript, line
    conditions = ["C128", "C512", "C2048", "Call"]
    reference_rows = [indexed[("unseen_pair", condition, "interaction_C_r0")] for condition in conditions]
    coverage = "Coverage" + "".join(f" & {row['proteins']} / {row['paired_protein_repeats']}" for row in reference_rows)
    assert coverage in manuscript, coverage
    line = "SCEPI adaptation\\textsuperscript{S}" + "".join(f" & {row['reference_mean']:.3f}" for row in reference_rows)
    assert line in manuscript, line
    for arm, label in [("interaction_C_r0", "Pair removed\\textsuperscript{S}"),
                       ("thermo_epistatic_C", "ThermoMPNN-D"), ("mochi_order2_C", "MoCHI\\textsuperscript{2}")]:
        values = [indexed[("unseen_pair", condition, arm)] for condition in conditions]
        assert all(abs(row["reference_mean"] - reference["reference_mean"]) < 1e-12
                   for row, reference in zip(values, reference_rows))
        line = label + "".join(f" & {row['control_mean']:.3f}" for row in values)
        assert line in manuscript, line
    report = {"complete": True, "source_commit": evidence_manifest["source_commit"],
              "source_run_path": str(args.input),
              "paired_comparisons_checked": comparisons, "manuscript_table_rows_checked": table_rows,
              "matched_control_interval_rows_checked": interval_rows,
              "main_conditioning_table_rows_checked": conditioning_rows,
              "conditioning_relative_residual_reductions_percent": conditioning_reductions,
              "maximum_mean_difference": maximum_difference, "floating_tie_tolerance": 1e-12,
              "native_figure_fragments_embedded": 4, "thresholds": summaries,
              "original_budget_table_rows_checked": 8,
              "runtime": {"numpy": np.__version__, "pandas": pd.__version__},
              "full_planned_matrix_complete": False}
    (args.input / "manuscript_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
