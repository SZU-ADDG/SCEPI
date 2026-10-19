import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import PREDICTION_COLUMNS, residual_metrics, selection_metrics
from epistasis.metrics import per_protein_metrics
from scripts.paper_full_target_evidence import bootstrap
from scripts.run_sall_full_target import paired_summary
from scripts.train_single import family_permissions


def main():
    output = ROOT / "runs/stabilityarc_full_target"
    output.mkdir(parents=True, exist_ok=True)
    chunks = {key: [] for key in ["residual", "selection", "phenotype"]}
    coverage, manifests = {}, []
    for threshold in [30, 25]:
        for fold in range(5):
            truth = pd.read_parquet(ROOT / "runs/sall_full_target/truth" / f"family{threshold}/fold{fold}.parquet")
            truth = truth.loc[truth.structure_available].copy()
            assert set(truth.protein_id) <= set(family_permissions(threshold, fold)["test"])
            for seed in [11, 23, 47, 89, 131]:
                directory = ROOT / "runs/stabilityarc_target" / f"family{threshold}/fold{fold}/seed{seed}"
                complete = json.loads((directory / "manifest.json").read_text())
                assert complete["complete"]
                predictions, excluded = [], []
                for protein in sorted(truth.protein_id.unique()):
                    target = directory / protein / "Sall"
                    manifest = json.loads((target / "manifest.json").read_text())
                    assert manifest["complete"] and manifest["target_combination_labels"] == manifest["target_test_labels_used"] == 0
                    assert manifest["seed"] == seed and manifest["fold"] == fold and manifest["threshold"] == threshold
                    source = ROOT / manifest["source_run"]
                    assert json.loads((source / "permissions.json").read_text()) == family_permissions(threshold, fold)
                    if not manifest["evaluated"]:
                        excluded.append(protein)
                        continue
                    predictions.append(pd.read_parquet(target / "predictions.parquet", columns=PREDICTION_COLUMNS))
                    manifests.append(str((target / "manifest.json").relative_to(ROOT)))
                frame = truth.loc[~truth.protein_id.isin(excluded)].merge(
                    pd.concat(predictions, ignore_index=True), on="variant_id", validate="one_to_one")
                assert len(frame) == len(truth.loc[~truth.protein_id.isin(excluded)])
                np.testing.assert_allclose(frame.prediction_epsilon,
                                           frame.prediction_mu - frame.prediction_a - frame.prediction_b,
                                           rtol=1e-5, atol=1e-5)
                tables = {"residual": residual_metrics(frame), "selection": selection_metrics(frame, units=("protein",)),
                          "phenotype": per_protein_metrics(frame)}
                tables["phenotype"] = tables["phenotype"].merge(frame[["protein_id", "family_group"]].drop_duplicates(),
                                                                 on="protein_id", validate="one_to_one")
                for key, table in tables.items():
                    table = table.drop(columns=[column for column in table if column.startswith("selected_")
                                               and table[column].dtype == object])
                    table["arm"], table["threshold"], table["fold"], table["seed"] = "stabilityarc_kermut", threshold, fold, seed
                    chunks[key].append(table)
                coverage[(threshold, "stabilityarc_kermut")] = coverage.get((threshold, "stabilityarc_kermut"), 0) + 1
    primary = ROOT / "runs/sall_full_target/paper_evidence"
    tables = {}
    for key, current in chunks.items():
        reference = pd.read_parquet(primary / f"{key}_per_protein.parquet")
        reference = reference.loc[reference.arm.eq("interaction")].copy()
        tables[key] = pd.concat([reference, *current], ignore_index=True)
        for threshold in [30, 25]:
            assert coverage[(threshold, "stabilityarc_kermut")] == 25
            coverage[(threshold, "interaction")] = 25
    tables["phenotype"] = tables["phenotype"].loc[tables["phenotype"].phenotype_count.ge(30)]
    summary = []
    for metric in ["compensation_hit_rate", "selected_mean_recovery", "selected_mean_stability", "precision_lower", "precision_upper"]:
        summary.extend(paired_summary(tables["selection"], ["threshold", "task", "margin", "unit", "scoring", "budget"], metric, True, coverage))
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        summary.extend(paired_summary(tables["residual"], ["threshold", "margin"], metric, False, coverage))
    for metric, higher in [("phenotype_rmse", False), ("phenotype_spearman", True), ("phenotype_pearson", True)]:
        summary.extend(paired_summary(tables["phenotype"], ["threshold"], metric, higher, coverage))
    bootstrap(summary, tables)
    for key, table in tables.items():
        table.to_parquet(output / f"{key}_per_protein.parquet", index=False)
    (output / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    manifest = {"complete": True, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "evaluations": 50, "thresholds": [30, 25], "folds": list(range(5)), "seeds": [11, 23, 47, 89, 131],
                "target_prediction_manifests": manifests, "compared_to": str(primary),
                "paired_comparisons": len(summary), "frozen_full_target_truth_verified": True,
                "target_double_training_labels": 0, "scoring": "Native Kermut posterior phenotype and its own four-prediction contrast"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "target_prediction_manifests"}), flush=True)


if __name__ == "__main__":
    main()
