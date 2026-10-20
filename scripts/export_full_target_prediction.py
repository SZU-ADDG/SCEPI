import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/sall_full_target/paper_evidence"
OUTPUT = ROOT / "runs/sall_full_target/paper_prediction"
ARMS = [
    "interaction",
    "additive",
    "global_nonlinear",
    "thermo_epistatic",
    "kermut",
    "stabilityarc_kermut",
    "mean_mlp",
    "mean_ridge",
]
KEYS = ["arm", "threshold", "seed", "fold", "protein_id"]


def main():
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    SOURCE, OUTPUT = args.evidence, args.output
    evidence_manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert evidence_manifest["available_jobs_complete"]
    phenotype = pd.read_parquet(SOURCE / "phenotype_per_protein.parquet")
    table = phenotype.loc[
        phenotype.threshold.eq(30) & phenotype.arm.isin(ARMS)
    ].copy()
    assert table.loc[~table.arm.eq("stabilityarc_kermut"), "condition"].eq("Sall").all()
    assert table.loc[table.arm.eq("stabilityarc_kermut"), "condition"].isna().all()
    table["condition"] = "Sall"
    assert set(table.arm.unique()) == set(ARMS)
    assert not table.duplicated(KEYS).any()
    assert table.groupby("arm").size().eq(795).all()
    assert table.groupby(["arm", "seed", "fold"]).size().sum() == len(table)
    assert table.groupby("arm").protein_id.nunique().eq(159).all()
    assert set(table.seed.unique()) == {11, 23, 47, 89, 131}
    assert set(table.fold.unique()) == set(range(5))
    assert table.loc[table.main_eligible].groupby("arm").protein_id.nunique().eq(155).all()

    residual = pd.read_parquet(SOURCE / "residual_per_protein.parquet")
    residual = residual.loc[
        residual.threshold.eq(30)
        & residual.margin.eq(0)
        & residual.arm.isin(ARMS),
        KEYS + ["condition", "epsilon_rmse", "pair_macro_epsilon_rmse"],
    ]
    assert residual.loc[~residual.arm.eq("stabilityarc_kermut"), "condition"].eq("Sall").all()
    assert residual.loc[residual.arm.eq("stabilityarc_kermut"), "condition"].isna().all()
    residual["condition"] = "Sall"
    assert not residual.duplicated(KEYS).any()
    check = table.loc[table.main_eligible, KEYS + ["epsilon_rmse", "pair_macro_epsilon_rmse"]].merge(
        residual,
        on=KEYS,
        suffixes=("_phenotype", "_residual"),
        validate="one_to_one",
    )
    assert len(check) == int(table.main_eligible.sum())
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        np.testing.assert_allclose(
            check[f"{metric}_phenotype"],
            check[f"{metric}_residual"],
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT / "per_protein.parquet", index=False)
    summary = {
        "source_manifest": evidence_manifest,
        "prediction_grid_complete": True,
        "source_tables": [
            str(SOURCE / "phenotype_per_protein.parquet"),
            str(SOURCE / "residual_per_protein.parquet"),
        ],
        "scope": "Threshold-30 S-all; merged full-target evidence table; five seeds and five held-out folds",
        "arms": ARMS,
        "prediction_records": int(len(table)),
        "protein_records_per_arm": 795,
        "phenotype_proteins_per_arm": 159,
        "residual_proteins_per_arm": 155,
        "residual_margin": 0.0,
        "residual_metrics_reconciled": True,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
