import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.evaluation_scope import authorized_jobs

SETTINGS = ["dataset", "threshold", "scheme", "condition", "mutation_order"]
METRICS = ["epsilon_rmse", "epsilon_rmse_training_scale", "phenotype_rmse", "phenotype_rmse_training_scale", "phenotype_spearman"]
STABILITY_ARMS = ["interaction_C", "interaction_C_r0", "interaction_target_only_C", "interaction_target_only_C_r0",
                  "original_geometry_C", "original_geometry_target_only_C",
                  "thermo_epistatic_C", "thermo_additive_C", "mochi_order1_C", "mochi_order2_C", "kermut_C",
                  "mean_ridge_target_C", "mean_mlp_target_C", "mean_ridge_source_C", "mean_mlp_source_C"]
FUNCTIONAL_ARMS = ["transfer_shared", "transfer_remove_interaction", "transfer_independent_no_r",
                   "target_only_shared", "target_only_remove_interaction", "target_only_independent_no_r",
                   "mochi_order1", "mochi_order2", "mavenn_order1", "mavenn_order2", "rfa_order2", "mean_ridge", "mean_mlp"]


def eligible(group, metric):
    selected = group.loc[group[metric].notna()]
    if metric.startswith("epsilon_"):
        selected = selected.loc[selected.main_eligible.eq(True)]
    return selected


def summarize(frame, expected, completed):
    rows = []
    for key, group in frame.groupby(SETTINGS + ["arm"], sort=True):
        configuration = dict(zip(SETTINGS + ["arm"], key))
        grid_key = tuple(configuration[field] for field in ["dataset", "threshold", "scheme", "condition", "arm"])
        for metric in METRICS:
            selected = eligible(group, metric)
            if not len(selected):
                continue
            protein_means = selected.groupby("protein_id")[metric].mean()
            repeat_means = selected.groupby("seed")[metric].mean()
            rows.append({**configuration, "metric": metric, "mean": float(protein_means.mean()),
                         "proteins": len(protein_means), "protein_repeat_records": len(selected),
                         "repeats": len(repeat_means), "repeat_std": float(repeat_means.std(ddof=1)) if len(repeat_means) > 1 else np.nan,
                         "expected_evaluations": int(expected[grid_key]), "completed_evaluations": int(completed.get(grid_key, 0)),
                         "grid_complete": completed.get(grid_key, 0) == expected[grid_key]})
    return pd.DataFrame(rows)


def paired_checks(frame, published):
    maximum = 0.0
    count = 0
    rows = []
    references = {"functional_doubles": ["transfer_shared", "target_only_shared"],
                  "flip_primary": ["transfer_shared", "target_only_shared"],
                  "tsuboyama": ["interaction_C", "interaction_target_only_C"]}
    indexed = {tuple(row.get(field) for field in SETTINGS + ["reference", "control", "metric"]): row
               for row in published if row["family_scope"] == "all_groups"}
    for key, group in frame.groupby(SETTINGS, sort=True):
        configuration = dict(zip(SETTINGS, key))
        for metric in METRICS:
            selected = eligible(group, metric)
            if not len(selected):
                continue
            panel = selected.pivot(index=["protein_id", "seed"], columns="arm", values=metric)
            for reference in references[configuration["dataset"]]:
                if reference not in panel:
                    continue
                for control in panel.columns:
                    match = tuple(key) + (reference, control, metric)
                    if match not in indexed:
                        continue
                    values = panel[[reference, control]].dropna()
                    if not len(values):
                        continue
                    protein = values.groupby(level="protein_id").mean()
                    direction = 1 if metric == "phenotype_spearman" else -1
                    gains = direction * (values[reference] - values[control])
                    by_protein = gains.groupby(level="protein_id").mean()
                    original = indexed[match]
                    actual = {"reference_mean": float(protein[reference].mean()), "control_mean": float(protein[control].mean()),
                              "mean_gain": float(by_protein.mean())}
                    for field, value in actual.items():
                        difference = abs(value - original[field])
                        assert difference <= 1e-12, (match, field, value, original[field])
                        maximum = max(maximum, difference)
                    assert len(values) == original["paired_protein_repeats"]
                    assert len(protein) == original["proteins"]
                    count += 1
                    rows.append({**configuration, "reference": reference, "control": control, "metric": metric,
                                 **actual, "proteins": len(protein), "protein_repeat_records": len(values),
                                 "wins": int(by_protein.gt(1e-12).sum()), "ties": int(by_protein.abs().le(1e-12).sum()),
                                 "losses": int(by_protein.lt(-1e-12).sum()),
                                 "positive_repeats": int(gains.groupby(level="seed").mean().gt(0).sum()),
                                 "evaluation_grid_complete": original["evaluation_grid_complete"]})
    assert count > 0
    return pd.DataFrame(rows), count, maximum


def common_cohorts(frame):
    rows = []
    assays = pd.read_parquet(ROOT / "data/processed/functional_doubles/proteins.parquet").set_index("protein_id").assay_id
    for key, group in frame.groupby(SETTINGS, sort=True):
        configuration = dict(zip(SETTINGS, key))
        arms = STABILITY_ARMS if configuration["dataset"] == "tsuboyama" else FUNCTIONAL_ARMS
        if configuration["dataset"] == "tsuboyama" and configuration["scheme"] == "family":
            continue
        for metric in METRICS:
            selected = eligible(group.loc[group.arm.isin(arms)], metric)
            if not len(selected):
                continue
            panel = selected.pivot(index=["protein_id", "seed"], columns="arm", values=metric)
            available = [arm for arm in arms if arm in panel and panel[arm].notna().any()]
            common = panel[available].dropna()
            if not len(common):
                continue
            means = common.groupby(level="protein_id").mean()
            assay_means = means.assign(assay_id=means.index.map(assays)).groupby("assay_id")[available].mean() if configuration["dataset"] == "functional_doubles" else None
            if assay_means is not None:
                replicate_assays = common.reset_index().assign(assay_id=lambda values: values.protein_id.map(assays))
                replicate_means = replicate_assays.groupby(["seed", "assay_id"])[available].mean().groupby(level="seed").mean()
            else:
                replicate_means = common.groupby(level="seed").mean()
            for arm in available:
                rows.append({**configuration, "arm": arm, "metric": metric, "mean": float(means[arm].mean()),
                             "proteins": len(means), "protein_repeat_records": len(common),
                             "repeats": common.index.get_level_values("seed").nunique(),
                             "compared_arms": len(available), "available_arms": ";".join(available),
                             "assay_mean": float(assay_means[arm].mean()) if assay_means is not None else np.nan,
                             "repeat_std": float(replicate_means[arm].std(ddof=1)) if len(replicate_means) > 1 else np.nan,
                             "assays": len(assay_means) if assay_means is not None else 0})
    return pd.DataFrame(rows)


def rfa_provenance():
    records = []
    for dataset in ["functional_doubles", "flip_primary"]:
        tasks = pd.read_parquet(ROOT / "runs/original_main/functional_truth" / (dataset + "_local_tasks.parquet"),
                                columns=["protein_id", "seed", "scheme", "condition"])
        assert not tasks.duplicated().any()
        for task in tasks.to_dict("records"):
            relative = Path(dataset) / "order2" / task["protein_id"] / f"seed{task['seed']}" / task["scheme"] / task["condition"]
            candidates = [ROOT / "runs" / family / relative / "manifest.json"
                          for family in ["rfa", "rfa_accelerated", "rfa_bounded", "rfa_fixed"]]
            existing = [path for path in candidates if path.exists()]
            assert existing, (dataset, task)
            manifest = existing[0]
            information = json.loads(manifest.read_text())
            assert information["complete"]
            evaluated = information["evaluated"]
            records.append({"dataset": dataset, **task, "manifest": str(manifest.relative_to(ROOT)),
                            "evaluated": evaluated, "configuration": "fixed_default" if information.get("fixed_default", False) else "validation_selected" if evaluated else "ineligible",
                            "candidate_configurations": int(information["trials"]) if evaluated else 0,
                            "source_commit": information["source_commit"],
                            "reason": information.get("reason"),
                            "optimization_policy": information.get("optimization_policy", {}).get("version"),
                            "fixed_new_fits": information["optimization_budget"]["newly_fitted_trials"] if evaluated and information.get("fixed_default", False) else 0,
                            "fixed_reused_models": information["optimization_budget"]["retained_trials"] if evaluated and information.get("fixed_default", False) else 0})
    return pd.DataFrame(records)


def esm_subset(frame):
    rows = []
    subset = frame.loc[frame.dataset.eq("tsuboyama") & frame.threshold.eq(30) & frame.fold.eq(0)
                       & frame.seed.eq(11) & frame.scheme.eq("family") & frame.mutation_order.eq("all")]
    for condition in ["Z", "Sall"]:
        group = subset.loc[subset.condition.eq(condition)]
        arms = ["interaction", "esm_msr", "thermo_epistatic", "global_nonlinear", "additive"]
        if condition == "Sall":
            arms.append("kermut")
        for metric in ["epsilon_rmse", "phenotype_rmse", "phenotype_spearman"]:
            selected = eligible(group.loc[group.arm.isin(arms)], metric)
            panel = selected.pivot(index="protein_id", columns="arm", values=metric)[arms].dropna()
            assert len(panel)
            for arm in arms:
                rows.append({"condition": condition, "metric": metric, "arm": arm,
                             "mean": float(panel[arm].mean()), "proteins": len(panel), "fold": 0, "seed": 11,
                             "esm_msr_checkpoint_epoch": 10})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    source = ROOT / "runs/original_main"
    output = ROOT / "runs/final_main_reporting_20261018"
    output.mkdir(exist_ok=True)
    state = json.loads((source / "current.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if args.require_complete:
        assert state["active_scope_complete"] and state["reporting_complete"]
        assert state["completed_evaluations"] == state["authorized_evaluations"] == manifest["completed_evaluations"]
    jobs = json.loads((source / "jobs.json").read_text())
    authorized, discontinued = authorized_jobs(ROOT, "original_main", jobs)
    coverage = json.loads((source / "evaluation_coverage.json").read_text())
    assert len(coverage) == manifest["completed_evaluations"]
    fields = ["dataset", "threshold", "scheme", "condition", "arm"]
    expected = pd.DataFrame(authorized).groupby(fields).size()
    completed = pd.DataFrame(coverage).groupby(fields).size()
    frame = pd.read_parquet(source / "per_protein.parquet")
    assert not frame.duplicated(SETTINGS + ["arm", "protein_id", "seed", "fold"]).any()
    arm_summary = summarize(frame, expected, completed)
    paired, comparisons, maximum = paired_checks(frame, json.loads((source / "paired_summary.json").read_text()))
    common = common_cohorts(frame)
    arm_summary.to_csv(output / "arm_summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    common.to_csv(output / "common_cohort_summary.csv", index=False)
    frame.loc[frame.arm.eq("esm_msr")].to_csv(output / "esm_msr_available_subset.csv", index=False)
    esm_subset(frame).to_csv(output / "esm_msr_matched_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "evaluation_coverage.csv", index=False)
    if args.require_complete:
        rfa = rfa_provenance()
        rfa.to_csv(output / "rfa_target_provenance.csv", index=False)
        undefined = frame.loc[frame.dataset.eq("functional_doubles") & frame.arm.eq("rfa_order2")
                              & frame.mutation_order.eq("all") & frame.phenotype_spearman.isna()]
        undefined.to_csv(output / "rfa_undefined_spearman_records.csv", index=False)
    complete = bool(state["active_scope_complete"] and state.get("reporting_complete", False)
                    and manifest["completed_evaluations"] == len(authorized))
    record = {"complete": complete, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_run": "runs/original_main", "authorized_evaluations": len(authorized),
              "discontinued_evaluations": len(discontinued), "published_evaluations": manifest["completed_evaluations"],
              "arm_metric_rows": len(arm_summary), "paired_comparisons_checked": comparisons,
              "maximum_paired_mean_difference": maximum, "common_cohort_rows": len(common),
              "aggregation": "Average matched repetitions within protein, then average proteins equally. Functional assay means additionally average KRAS conditions within their assay before weighting assays equally.",
              "outputs": ["arm_summary.csv", "paired_comparisons.csv", "common_cohort_summary.csv", "esm_msr_available_subset.csv", "esm_msr_matched_summary.csv", "evaluation_coverage.csv"]}
    if args.require_complete:
        record["rfa_target_configurations"] = rfa.groupby(["dataset", "configuration"]).size().reset_index(name="targets").to_dict("records")
        record["rfa_fixed_new_fits"] = int(rfa.fixed_new_fits.sum())
        record["rfa_fixed_reused_models"] = int(rfa.fixed_reused_models.sum())
        record["rfa_undefined_spearman_records"] = len(undefined)
        record["outputs"].append("rfa_target_provenance.csv")
        record["outputs"].append("rfa_undefined_spearman_records.csv")
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
