import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.main_evaluation import OUTPUT

SETTINGS = ["dataset", "threshold", "scheme", "condition", "mutation_order"]
LOWER_BETTER = {"epsilon_rmse", "pair_macro_epsilon_rmse", "epsilon_rmse_training_scale", "phenotype_rmse",
                "phenotype_rmse_training_scale", "interval_distance_rmse"}
METRICS = ["epsilon_rmse", "pair_macro_epsilon_rmse", "epsilon_spearman", "epsilon_rmse_training_scale",
           "phenotype_rmse", "phenotype_rmse_training_scale", "phenotype_spearman", "phenotype_pearson",
           "interval_distance_rmse", "precision10_lower", "precision10_upper", "precision30_lower", "precision30_upper"]


def bootstrap(differences, family_test):
    rng = np.random.default_rng(20261009)
    if not family_test:
        values = differences.to_numpy()
        draws = rng.integers(len(values), size=(10000, len(values)))
        estimates = values[draws].mean(axis=1)
    else:
        groups = [group.to_numpy() for _, group in differences.groupby(level="family_group", sort=True)]
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10000)
        sums, sizes = np.zeros(10000), np.zeros(10000)
        for column, values in enumerate(groups):
            frequency = counts[:, column]
            sizes += frequency * len(values)
            for repetitions in np.unique(frequency):
                if repetitions == 0:
                    continue
                rows = np.flatnonzero(frequency == repetitions)
                draws = rng.integers(len(values), size=(len(rows), int(repetitions) * len(values)))
                sums[rows] += values[draws].sum(axis=1)
        assert np.all(sizes > 0)
        estimates = sums / sizes
    return {"paired_bootstrap_95_interval": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "bootstrap_positive_fraction": float(np.mean(estimates > 0)), "bootstrap_replicates": 10000,
            "bootstrap_unit": "family_then_protein" if family_test else "protein"}


def paired_result(panel, reference, control, metric, complete, primary, family_test):
    if reference not in panel or control not in panel:
        return None
    paired = panel[[reference, control]].dropna()
    if not len(paired):
        return None
    gains = paired[control] - paired[reference] if metric in LOWER_BETTER else paired[reference] - paired[control]
    by_protein = gains.groupby(level=["protein_id", "family_group"]).mean()
    by_seed = gains.groupby(level="seed").mean()
    result = {"reference": reference, "control": control, "metric": metric,
              "evaluation_grid_complete": bool(complete), "proteins": len(by_protein),
              "family_groups": by_protein.index.get_level_values("family_group").nunique(),
              "paired_protein_repeats": len(gains), "evaluated_repeats": len(by_seed),
              "reference_mean": float(paired[reference].groupby(level="protein_id").mean().mean()),
              "control_mean": float(paired[control].groupby(level="protein_id").mean().mean()),
              "mean_gain": float(by_protein.mean()), "wins": int(by_protein.gt(0).sum()),
              "losses": int(by_protein.lt(0).sum()), "ties": int(by_protein.eq(0).sum()),
              "positive_repeats": int(by_seed.gt(0).sum()),
              "per_repeat_mean_gain": {str(seed): float(value) for seed, value in by_seed.items()}}
    if metric in LOWER_BETTER:
        result["relative_error_reduction"] = result["mean_gain"] / result["control_mean"] if result["control_mean"] else None
    if primary and complete and len(by_protein) >= 2:
        result.update(bootstrap(by_protein, family_test))
        result["paired_wilcoxon_pvalue"] = float(wilcoxon(by_protein, zero_method="zsplit").pvalue)
    return result


def main():
    jobs = json.loads((OUTPUT / "jobs.json").read_text())
    chunks, strata_chunks, coverage = [], [], []
    for job in jobs:
        path = ROOT / job["stages"][0]["manifest"]
        if not path.exists():
            continue
        information = json.loads(path.read_text())
        assert information["complete"] and not information["smoke"]
        summary = pd.read_parquet(path.parent / "per_protein.parquet")
        if len(summary):
            if "mutation_order" not in summary:
                summary["mutation_order"] = "all"
            chunks.append(summary)
        strata = path.parent / "strata_per_protein.parquet"
        if strata.exists():
            table = pd.read_parquet(strata)
            if len(table):
                strata_chunks.append(table)
        coverage.append({**{key: job[key] for key in ["dataset", "arm", "threshold", "scheme", "condition", "seed", "fold"]},
                         "normalized_variants": information["normalized_variants"], "excluded_proteins": information["excluded_proteins"]})
    summary = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    strata = pd.concat(strata_chunks, ignore_index=True) if strata_chunks else pd.DataFrame()
    summary.to_parquet(OUTPUT / "per_protein.parquet", index=False)
    strata.to_parquet(OUTPUT / "strata_per_protein.parquet", index=False)
    counts = pd.DataFrame(coverage).groupby(["dataset", "threshold", "scheme", "condition", "arm"]).size() if coverage else pd.Series(dtype=int)
    expected = pd.DataFrame(jobs).groupby(["dataset", "threshold", "scheme", "condition", "arm"]).size()
    comparisons = []
    for table, settings, metrics in [(summary, SETTINGS, METRICS),
                                     (strata, SETTINGS[:-1] + ["stratum", "stratum_value"], ["epsilon_rmse", "pair_macro_epsilon_rmse", "epsilon_spearman"])]:
        if not len(table):
            continue
        groups = [("all_groups", key, group) for key, group in table.groupby(settings, sort=True)]
        groups += [("annotated_families", key, group) for key, group in table.loc[table.pfam_annotated].groupby(settings, sort=True)]
        for family_scope, key, group in groups:
            configuration = {field: value.item() if isinstance(value, np.generic) else value for field, value in zip(settings, key)}
            configuration["family_scope"] = family_scope
            if configuration["dataset"] != "tsuboyama":
                references = ["transfer_shared", "target_only_shared"]
            elif configuration["scheme"] == "family":
                references = ["original_interaction", "interaction", "original_geometry", "geometry"]
            else:
                references = ["interaction_C", "interaction_target_only_C", "original_geometry_C", "original_geometry_target_only_C"]
            if not set(group.arm).intersection(references):
                continue
            panels = {}
            for metric in metrics:
                if metric not in group:
                    continue
                valid = group
                if metric.startswith(("epsilon_", "pair_macro_epsilon")):
                    valid = group.loc[group.main_eligible.fillna(False)]
                panels[metric] = valid.pivot(index=["seed", "protein_id", "family_group"], columns="arm", values=metric)
            base = tuple(configuration[field] for field in ["dataset", "threshold", "scheme", "condition"])
            for reference in references:
                if reference not in set(group.arm):
                    continue
                controls = set(group.arm) - {reference}
                if reference == "interaction_target_only_C":
                    controls &= {"interaction_target_only_C_r0", "mochi_order1_C", "mochi_order2_C", "mean_ridge_target_C", "mean_mlp_target_C", "kermut_C"}
                if reference == "target_only_shared":
                    controls = {arm for arm in controls if not arm.startswith("transfer_")}
                for control in sorted(controls):
                    complete = (counts.get(base + (reference,), 0) == expected[base + (reference,)]
                                and counts.get(base + (control,), 0) == expected[base + (control,)])
                    for metric, panel in panels.items():
                        primary = (table is summary and family_scope == "all_groups" and configuration["dataset"] == "tsuboyama"
                                   and configuration["threshold"] == 30 and configuration["mutation_order"] == "all"
                                   and ((configuration["scheme"] == "family" and configuration["condition"] in ["Z", "S128"]
                                         and reference in ["original_interaction", "interaction"] and metric == "epsilon_rmse")
                                        or (configuration["scheme"] == "family" and configuration["condition"] == "S128"
                                            and reference in ["original_interaction", "interaction"] and metric == "precision10_lower")
                                        or (configuration["scheme"] == "unseen_pair" and configuration["condition"] == "C128"
                                            and reference == "interaction_C" and metric == "epsilon_rmse")))
                        result = paired_result(panel, reference, control, metric, complete, primary, configuration["scheme"] == "family")
                        if result is not None:
                            comparisons.append({**configuration, **result})
    tested = [index for index, item in enumerate(comparisons) if "paired_wilcoxon_pvalue" in item]
    if tested:
        _, adjusted, _, _ = multipletests([comparisons[index]["paired_wilcoxon_pvalue"] for index in tested], method="holm")
        for index, value in zip(tested, adjusted):
            comparisons[index]["holm_pvalue_all_reported_primary_comparisons"] = float(value)
    (OUTPUT / "paired_summary.json").write_text(json.dumps(comparisons, indent=2))
    (OUTPUT / "evaluation_coverage.json").write_text(json.dumps(coverage, indent=2))
    information = {"complete": len(coverage) == len(jobs), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "protocol": "runs/original_main/protocol.json", "run_path": str(OUTPUT),
                   "expected_evaluations": len(jobs), "completed_evaluations": len(coverage), "remaining_evaluations": len(jobs) - len(coverage),
                   "paired_comparisons": len(comparisons), "protein_repeat_rows": len(summary), "stratum_protein_repeat_rows": len(strata),
                   "bootstrap_completed_primary_comparisons": len(tested), "outputs": ["per_protein.parquet", "strata_per_protein.parquet", "paired_summary.json", "evaluation_coverage.json"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(information, indent=2))
    print(json.dumps(information), flush=True)


if __name__ == "__main__":
    main()
