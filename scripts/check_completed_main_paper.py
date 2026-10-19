import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["C128", "C512", "C2048", "Call"]
STABILITY = {
    "interaction_C": r"SCEPI adaptation\textsuperscript{S}",
    "interaction_C_r0": r"Pair removed\textsuperscript{S}",
    "interaction_target_only_C": r"SCEPI adaptation\textsuperscript{T}",
    "interaction_target_only_C_r0": r"Pair removed\textsuperscript{T}",
    "original_geometry_C": r"SCEPI structure\textsuperscript{S}",
    "original_geometry_target_only_C": r"SCEPI structure\textsuperscript{T}",
    "thermo_epistatic_C": "ThermoMPNN-D",
    "thermo_additive_C": "ThermoMPNN additive",
    "kermut_C": "Kermut",
    "mochi_order1_C": r"MoCHI\textsuperscript{1}",
    "mochi_order2_C": r"MoCHI\textsuperscript{2}",
    "mean_ridge_target_C": r"Mean ridge\textsuperscript{T}",
    "mean_mlp_target_C": r"Mean MLP\textsuperscript{T}",
    "mean_ridge_source_C": r"Mean ridge\textsuperscript{S}",
    "mean_mlp_source_C": r"Mean MLP\textsuperscript{S}",
}
FUNCTIONAL = {
    "transfer_shared": r"SCEPI adaptation\textsuperscript{S}",
    "transfer_remove_interaction": r"Pair removed\textsuperscript{S}",
    "transfer_independent_no_r": r"Independent no-pair\textsuperscript{S}",
    "target_only_shared": r"SCEPI adaptation\textsuperscript{T}",
    "target_only_remove_interaction": r"Pair removed\textsuperscript{T}",
    "target_only_independent_no_r": r"Independent no-pair\textsuperscript{T}",
    "mochi_order1": r"MoCHI\textsuperscript{1}",
    "mochi_order2": r"MoCHI\textsuperscript{2}",
    "mavenn_order1": r"MAVE-NN\textsuperscript{1}",
    "mavenn_order2": r"MAVE-NN\textsuperscript{2}",
    "rfa_order2": "RFA",
    "mean_ridge": "Mean ridge",
    "mean_mlp": "Mean MLP",
}
MSR = {"interaction": "SCEPI", "esm_msr": "ESM-MSR", "thermo_epistatic": "ThermoMPNN-D",
       "global_nonlinear": "Global nonlinear", "additive": "Additive", "kermut": "Kermut"}


def number(value, bold=False):
    if pd.isna(value):
        return "---"
    result = f"{value:.3f}"
    result = f"${result}$" if value < 0 else result
    return "\\textbf{" + result + "}" if bold else result


def check_rows(manuscript, frame, labels, columns, value, emphasize):
    panel = frame.pivot(index="arm", columns=["scheme", "condition"], values=value).reindex(columns=columns)
    assert not panel.isna().all(axis=0).any()
    total = 0
    for arm, label in labels.items():
        assert arm in panel.index, arm
        values = panel.loc[arm]
        line = label + "".join(" & " + number(value, (arm, column) in emphasize)
                                 for column, value in values.items())
        assert line + r" \\" in manuscript, line
        total += int(values.notna().sum())
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "runs/final_main_reporting_20261018")
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["complete"] and manifest["published_evaluations"] == manifest["authorized_evaluations"] == 12622
    frame = pd.read_csv(args.input / "common_cohort_summary.csv")
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    cells = 0
    for scheme in ["random", "unseen_pair", "two_pair_secondary"]:
        selected = frame.loc[frame.dataset.eq("tsuboyama") & frame.metric.eq("epsilon_rmse") & frame.scheme.eq(scheme)]
        assert selected.mutation_order.eq("all").all()
        budgets = ["C128", "Call"] if scheme == "two_pair_secondary" else CONDITIONS
        labels = {key: value for key, value in STABILITY.items() if key in selected.arm.unique()}
        assert len(labels) == (11 if scheme == "two_pair_secondary" else 15)
        bold = {("original_geometry_C", (scheme, condition)) for condition in budgets}
        cells += check_rows(manuscript, selected, labels, [(scheme, condition) for condition in budgets], "mean", bold)
        coverage = selected.groupby("condition")[["proteins", "protein_repeat_records"]].first()
        line = "Coverage" + "".join(f" & {coverage.loc[key, 'proteins']} / {coverage.loc[key, 'protein_repeat_records']}" for key in budgets)
        assert line + r" \\" in manuscript
    for metric in ["epsilon_rmse_training_scale", "phenotype_spearman"]:
        selected = frame.loc[frame.dataset.eq("functional_doubles") & frame.metric.eq(metric)
                             & frame.mutation_order.eq("quartet" if metric.startswith("epsilon_") else "all")]
        assert selected.compared_arms.eq(13).all() and selected.repeats.eq(5).all()
        expected_assays = selected.assays.eq(5).all() if metric.startswith("epsilon_") else selected.assays.isin([5, 6]).all()
        assert expected_assays
        columns = [(scheme, condition) for scheme in ["random", "unseen_pair"] for condition in CONDITIONS]
        cells += check_rows(manuscript, selected, FUNCTIONAL, columns, "assay_mean", set())
    for schemes in [["trpb_one_to_many", "trpb_two_to_many", "trpb_by_position", "amylase_one_to_many", "ired_two_to_many"],
                    ["hydro_three_to_many", "hydro_to_P01053", "hydro_to_P06241", "hydro_to_P0A9X9"]]:
        selected = frame.loc[frame.dataset.eq("flip_primary") & frame.metric.eq("phenotype_spearman")
                             & frame.mutation_order.eq("all") & frame.scheme.isin(schemes)]
        assert selected.repeats.eq(5).all()
        assert selected.loc[~selected.scheme.str.startswith("hydro_to_")].compared_arms.eq(13).all()
        assert selected.loc[selected.scheme.str.startswith("hydro_to_")].compared_arms.eq(8).all()
        cells += check_rows(manuscript, selected, FUNCTIONAL, [(scheme, "official") for scheme in schemes], "mean", set())
    msr = pd.read_csv(args.input / "esm_msr_matched_summary.csv")
    assert msr.esm_msr_checkpoint_epoch.eq(10).all()
    assert msr.loc[msr.metric.eq("epsilon_rmse"), "proteins"].eq(61).all()
    assert msr.loc[~msr.metric.eq("epsilon_rmse"), "proteins"].eq(65).all()
    panel = msr.pivot(index="arm", columns=["condition", "metric"], values="mean")
    metrics = ["epsilon_rmse", "phenotype_rmse", "phenotype_spearman"]
    for arm, label in MSR.items():
        line = label
        for condition in ["Z", "Sall"]:
            for metric in metrics:
                value = panel.loc[arm, (condition, metric)]
                best = panel[(condition, metric)].max() if metric == "phenotype_spearman" else panel[(condition, metric)].min()
                line += " & " + number(value, bool(np.isfinite(value) and value == best))
                cells += int(np.isfinite(value))
        assert line + r" \\" in manuscript, line
    rfa = pd.read_csv(args.input / "rfa_target_provenance.csv")
    assert len(rfa) == 380 and rfa.evaluated.sum() == 350
    assert rfa.configuration.eq("fixed_default").sum() == 284
    assert rfa.configuration.eq("validation_selected").sum() == 66
    assert rfa.fixed_new_fits.sum() == 264 and rfa.fixed_reused_models.sum() == 20
    assert manifest["rfa_undefined_spearman_records"] == 12
    assert "284 use the fixed configuration and 66 use validation-selected penalties" in manuscript
    assert "350 eligible fits, comprising 310 functional-assay fits and 40 FLIP2 fits" in manuscript
    source = pd.read_parquet(ROOT / "runs/original_main/per_protein.parquet",
                             filters=[("dataset", "==", "functional_doubles"), ("scheme", "==", "unseen_pair"),
                                      ("condition", "==", "C128"), ("mutation_order", "==", "all"),
                                      ("arm", "in", ["transfer_shared", "transfer_remove_interaction"])])
    assays = pd.read_parquet(ROOT / "data/processed/functional_doubles/proteins.parquet").set_index("protein_id").assay_id
    full = source.groupby(["arm", "protein_id"]).phenotype_spearman.mean().reset_index()
    full["assay"] = full.protein_id.map(assays)
    full_means = full.groupby(["arm", "assay"]).phenotype_spearman.mean().groupby(level="arm").mean()
    assert (f"Spearman correlation {full_means['transfer_shared']:.3f}, compared with "
            f"{full_means['transfer_remove_interaction']:.3f} after pair removal") in manuscript
    report = {"complete": True, "source_commit": manifest["source_commit"], "source_run": manifest["source_run"],
              "source_manifest": str(args.input / "manifest.json"), "checked_numeric_cells": cells,
              "authorized_evaluations": 12622, "esm_msr_checkpoint_epoch": 10,
              "rfa_evaluated_targets": 350, "rfa_undefined_rank_correlations": 12,
              "functional_aggregation": "Equal assay means after equal condition and repetition averaging."}
    (args.input / "manuscript_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
