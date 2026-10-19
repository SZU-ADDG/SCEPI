import json
from pathlib import Path

import numpy as np
import pandas as pd


root = Path("/data3/doomx/RECOMB-3-Epistasis")
run = root / "runs/compensation_main"
summary = json.loads((run / "paired_summary.json").read_text())
manifest = json.loads((run / "manifest.json").read_text())
controls = {
    "additive", "global_nonlinear", "kermut", "thermo_epistatic",
    "esm_msr", "remove_interaction", "no_cross", "reference_context",
    "identity_features", "phenotype_only", "residual_only",
}
common = {
    "threshold": 30, "condition": "Sall", "family_scope": "all_groups",
    "reference": "interaction",
}
selection_rows = [
    row for row in summary
    if all(row.get(key) == value for key, value in common.items())
    and row.get("task") == "compensation" and row.get("unit") == "protein"
    and row.get("budget") == 10 and row.get("margin") in [0.0, 0.5]
    and row.get("scoring") in ["native", "observed_main"]
    and row.get("control") in controls
    and row.get("metric") in ["compensation_hit_rate", "selected_mean_recovery"]
]
residual_rows = [
    row for row in summary
    if all(row.get(key) == value for key, value in common.items())
    and row.get("metric") == "epsilon_rmse" and row.get("margin") == 0
    and row.get("control") in controls
]

# 从逐蛋白记录独立重算正文所用汇总，保持原有重复权重。
selection = pd.read_parquet(
    run / "selection_per_protein.parquet",
    columns=["threshold", "condition", "task", "unit", "budget", "margin",
             "scoring", "arm", "seed", "protein_id", "family_group",
             "compensation_hit_rate", "selected_mean_recovery"],
    filters=[("threshold", "==", 30), ("condition", "==", "Sall"),
             ("task", "==", "compensation"), ("unit", "==", "protein"),
             ("budget", "==", 10)],
)
residual = pd.read_parquet(
    run / "residual_per_protein.parquet",
    columns=["threshold", "condition", "margin", "arm", "seed",
             "protein_id", "family_group", "epsilon_rmse"],
    filters=[("threshold", "==", 30), ("condition", "==", "Sall"),
             ("margin", "==", 0)],
)

for rows, table, settings, lower_better in [
    (selection_rows, selection, ["task", "unit", "budget", "margin", "scoring"], False),
    (residual_rows, residual, ["margin"], True),
]:
    for row in rows:
        group = table
        for key in settings:
            group = group.loc[group[key].eq(row[key])]
        metric, control = row["metric"], row["control"]
        wide = group.pivot(
            index=["seed", "protein_id", "family_group"], columns="arm", values=metric
        )[["interaction", control]].dropna()
        gain = wide[control] - wide.interaction if lower_better else wide.interaction - wide[control]
        protein_gain = gain.groupby(level=["protein_id", "family_group"]).mean()
        observed = [wide.interaction.groupby(level="protein_id").mean().mean(),
                    wide[control].groupby(level="protein_id").mean().mean(), protein_gain.mean()]
        expected = [row["reference_mean"], row["control_mean"], row["mean_gain"]]
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
        assert len(wide) == row["paired_protein_repeats"]
        assert len(protein_gain) == row["proteins"]
        assert int(protein_gain.gt(0).sum()) == row["wins"]
        assert int(protein_gain.lt(0).sum()) == row["losses"]
        assert int(protein_gain.eq(0).sum()) == row["ties"]

output = {
    "source_manifest": manifest,
    "verified_from": [str(run / "selection_per_protein.parquet"),
                      str(run / "residual_per_protein.parquet")],
    "comparisons_recomputed": len(selection_rows) + len(residual_rows),
    "selection": selection_rows,
    "residual": residual_rows,
}
(root / "work/paper_method_evidence.json").write_text(json.dumps(output, indent=2))
fields = ["scoring", "margin", "control", "metric", "proteins",
          "evaluation_grid_complete", "reference_mean", "control_mean",
          "mean_gain", "positive_repeats", "wins", "losses", "ties"]
print(json.dumps({"comparisons_recomputed": output["comparisons_recomputed"],
                  "selection": [{key: row[key] for key in fields}
                                for row in selection_rows if row["margin"] == 0],
                  "residual": residual_rows}, indent=2))
