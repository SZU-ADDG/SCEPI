from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "work/sall_full_target/paper_evidence"
OUTPUT = ROOT / "work/sall_full_target/paper_verified"
SEEDS = {11, 23, 47, 89, 131}
BOOTSTRAP_SEED = 20261013
TIE_TOLERANCE = 1e-12


def summarize(table, keys, metric, higher, coverage, rng, draws):
    rows = []
    for values, group in table.groupby(keys, observed=True, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        settings = {key: value.item() if isinstance(value, np.generic) else value
                    for key, value in zip(keys, values)}
        wide = group.pivot(index=["seed", "protein_id", "family_group"], columns="arm", values=metric)
        for control in sorted(set(wide.columns) - {"interaction"}):
            paired = wide[["interaction", control]].dropna()
            assert len(paired)
            protein = paired.groupby(["protein_id", "family_group"], observed=True).mean()
            gain = protein.interaction - protein[control] if higher else protein[control] - protein.interaction
            repeat_gain = (paired.interaction - paired[control] if higher else paired[control] - paired.interaction).groupby("seed").mean()
            assert set(repeat_gain.index) == SEEDS
            assert paired.groupby("protein_id").size().eq(5).all()
            row = {**settings, "metric": metric, "reference": "interaction", "control": control,
                   "proteins": len(protein), "family_groups": int(protein.index.get_level_values("family_group").nunique()),
                   "paired_protein_repeats": len(paired), "complete_grid": all(coverage[(settings["threshold"], arm)] == 25 for arm in ["interaction", control]),
                   "reference_mean": float(protein.interaction.mean()), "control_mean": float(protein[control].mean()),
                   "mean_gain": float(gain.mean()), "wins": int(gain.gt(TIE_TOLERANCE).sum()), "losses": int(gain.lt(-TIE_TOLERANCE).sum()),
                   "ties": int(gain.abs().le(TIE_TOLERANCE).sum()), "positive_repeats": int(repeat_gain.gt(TIE_TOLERANCE).sum()), "repeats": len(repeat_gain),
                   "repeat_gains": {str(int(seed)): float(value) for seed, value in repeat_gain.items()}}
            use_bootstrap = (metric in ["epsilon_rmse", "pair_macro_epsilon_rmse", "phenotype_rmse", "phenotype_spearman"]
                             or settings.get("task") == "compensation" and settings["budget"] == 10
                             and metric in ["compensation_hit_rate", "selected_mean_recovery"])
            if use_bootstrap:
                sums = gain.groupby("family_group", observed=True).sum()
                sizes = gain.groupby("family_group", observed=True).size().reindex(sums.index)
                key = tuple(zip(sums.index.astype(str), sizes.astype(int)))
                if key not in draws:
                    draws[key] = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=10000)
                counts = draws[key]
                bootstrap_values = (counts @ sums.to_numpy()) / (counts @ sizes.to_numpy())
                row.update(gain_interval_95=np.quantile(bootstrap_values, [0.025, 0.975]).tolist(),
                           bootstrap_positive_fraction=float((bootstrap_values > 0).mean()), bootstrap_resamples=10000)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-run-path", default="/data3/doomx/RECOMB-3-Epistasis/runs/sall_full_target/paper_evidence/family30_snapshot")
    args = parser.parse_args()
    tables = {name: pd.read_parquet(args.input / (name + "_per_protein.parquet"))
              for name in ["phenotype", "residual", "selection"]}
    tables = {name: table.loc[table.threshold.eq(args.threshold)].copy() for name, table in tables.items()}
    grids = tables["phenotype"][["threshold", "arm", "seed", "fold"]].drop_duplicates()
    coverage = Counter((int(row.threshold), row.arm) for row in grids.itertuples())
    arms = set(grids.arm)
    assert "interaction" in arms and len(grids) == 25 * len(arms)
    assert set(grids.threshold) == {args.threshold}
    assert all(value == 25 for value in coverage.values())
    assert set(grids.seed) == SEEDS and set(grids.fold) == set(range(5))
    assert tables["selection"].unit.eq("protein").all()
    rng, draws, rows = np.random.default_rng(BOOTSTRAP_SEED), {}, []
    specifications = [("selection", ["threshold", "task", "margin", "unit", "scoring", "budget"],
                       [(name, True) for name in ["compensation_hit_rate", "selected_mean_recovery", "selected_mean_stability", "precision_lower", "precision_upper"]]),
                      ("residual", ["threshold", "margin"], [("epsilon_rmse", False), ("pair_macro_epsilon_rmse", False)]),
                      ("phenotype", ["threshold"], [("phenotype_rmse", False), ("phenotype_spearman", True), ("phenotype_pearson", True)])]
    for name, keys, metrics in specifications:
        table = tables[name]
        assert not table.duplicated(keys + ["seed", "protein_id", "family_group", "arm"]).any()
        for metric, higher in metrics:
            measured = table.loc[table[metric].notna()]
            rows.extend(summarize(measured, keys, metric, higher, coverage, rng, draws))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "paired_summary.json").write_text(json.dumps(rows, indent=2))
    native = tables["selection"].loc[tables["selection"].task.eq("compensation") & tables["selection"].margin.eq(0)
                                    & tables["selection"].budget.eq(10) & tables["selection"].scoring.eq("native")]
    common = tables["selection"].loc[tables["selection"].task.eq("compensation") & tables["selection"].margin.eq(0)
                                    & tables["selection"].budget.eq(10) & tables["selection"].scoring.eq("observed_main")]
    metrics = [(tables["phenotype"], ["phenotype_rmse", "phenotype_spearman"]),
               (tables["residual"].loc[tables["residual"].margin.eq(0)], ["epsilon_rmse"]),
               (native, ["compensation_hit_rate", "selected_mean_recovery"])]
    overview = pd.concat([table.groupby(["arm", "protein_id"], observed=True)[names].mean().groupby("arm").mean()
                          for table, names in metrics], axis=1)
    overview["common_main_hit_rate"] = common.groupby(["arm", "protein_id"]).compensation_hit_rate.mean().groupby("arm").mean()
    overview.to_csv(args.output / "paper_table.csv")
    manifest = {"complete": True, "scope": f"All {len(arms)} available family{args.threshold} S-all arms; each five-fold, five-repeat grid is complete.",
                "full_planned_matrix_complete": False, "source_commit": "875d04d6df818110001f3889758cf998321a96f6",
                "source_run_path": args.source_run_path,
                "source_tables": [name + "_per_protein.parquet" for name in tables],
                "verified_grids": len(grids), "arms": sorted(grids.arm.unique()), "paired_comparisons": len(rows),
                "win_tie_tolerance": TIE_TOLERANCE,
                "aggregation": "Matched five-repeat mean within protein followed by equal-protein mean.",
                "bootstrap": {"unit": "homology group", "resamples": 10000, "seed": BOOTSTRAP_SEED,
                              "cohort_draws_shared_across_comparisons": True}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    primary = [row for row in rows if row["control"] in ["additive", "global_nonlinear", "thermo_epistatic", "kermut", "mean_ridge", "mean_mlp"]
               and row.get("task") == "compensation" and row["budget"] == 10 and row["margin"] in [0, 0.5]
               and row["metric"] in ["compensation_hit_rate", "selected_mean_recovery"]]
    print(json.dumps({"manifest": manifest, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
