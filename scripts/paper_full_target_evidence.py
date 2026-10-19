import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import definitions, input_records, normalized_predictions, residual_metrics, selection_metrics
from epistasis.metrics import per_protein_metrics
from scripts.run_sall_full_target import paired_summary

RUN = ROOT / "runs/sall_full_target"
OUTPUT = RUN / "paper_evidence"


def initialize():
    global TRUTH
    TRUTH = {(threshold, fold): pd.read_parquet(RUN / "truth" / f"family{threshold}/fold{fold}.parquet")
             for threshold in [30, 25] for fold in range(5)}


def directory(job):
    return OUTPUT / "evaluations" / job["arm"] / f"family{job['threshold']}" / f"seed{job['seed']}" / f"fold{job['fold']}"


def evaluate(job):
    target = directory(job)
    path = target / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text())["complete"]
        return job
    truth = TRUTH[(job["threshold"], job["fold"])]
    if definitions()[job["arm"]]["requires_structure"]:
        truth = truth.loc[truth.structure_available]
    entries, missing = input_records(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", truth)
    assert not missing, missing
    frame, excluded, provenance = normalized_predictions(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", truth, entries)
    tables = {"residual": residual_metrics(frame), "selection": selection_metrics(frame, units=("protein",)),
              "phenotype": per_protein_metrics(frame)}
    if len(tables["phenotype"]):
        tables["phenotype"] = tables["phenotype"].merge(frame[["protein_id", "family_group"]].drop_duplicates(),
                                                       on="protein_id", validate="one_to_one")
    verified = []
    legacy = RUN / "evaluations" / job["arm"] / f"family{job['threshold']}" / f"seed{job['seed']}" / f"fold{job['fold']}"
    if (legacy / "manifest.json").exists():
        for name, filename in [("residual", "residual_per_protein.parquet"),
                               ("selection", "selection_per_decision.parquet"),
                               ("phenotype", "phenotype_per_protein.parquet")]:
            original = pd.read_parquet(legacy / filename)
            if name == "selection":
                original = original.loc[original.unit.eq("protein")]
            current = tables[name]
            keys = ["protein_id"] + (["margin"] if name == "residual" else
                    ["task", "margin", "unit", "scoring", "budget"] if name == "selection" else [])
            numeric = [column for column in current.select_dtypes(include="number") if column not in keys]
            first = current.set_index(keys).sort_index()[numeric]
            second = original.set_index(keys).sort_index()[numeric]
            assert first.index.equals(second.index)
            np.testing.assert_allclose(first, second, rtol=0, atol=1e-12, equal_nan=True)
            verified.append(filename)
    target.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table = table.drop(columns=[column for column in table if column.startswith("selected_")
                                   and table[column].dtype == object])
        for key, value in job.items():
            table[key] = value
        table.to_parquet(target / (name + ".parquet"), index=False)
    manifest = {"complete": True, **job, "full_target_coverage_verified": True,
                "four_prediction_contrast_verified": True, "normalized_variants": len(frame),
                "excluded_proteins": excluded, "input_predictions": provenance,
                "matched_full_evaluation_tables": verified,
                "scope": "All protein-level endpoints from the unchanged full-target protocol; site-pair decisions remain in the full evaluation."}
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**job, "complete": True, "variants": len(frame), "independent_checks": len(verified)}), flush=True)
    return job


def bootstrap(summary, tables):
    rng = np.random.default_rng(20261013)
    panels = {}
    for row in summary:
        metric = row["metric"]
        if metric.startswith("phenotype_"):
            table, settings, higher = tables["phenotype"], ["threshold"], metric != "phenotype_rmse"
        elif metric.endswith("epsilon_rmse"):
            table, settings, higher = tables["residual"], ["threshold", "margin"], False
        else:
            if row["task"] != "compensation" or row["budget"] != 10:
                continue
            table, settings, higher = tables["selection"], ["threshold", "task", "margin", "unit", "scoring", "budget"], True
        panel_key = (metric, tuple((key, row[key]) for key in settings))
        if panel_key not in panels:
            for key in settings:
                table = table.loc[table[key].eq(row[key])]
            panels[panel_key] = table.pivot(index=["seed", "protein_id", "family_group"], columns="arm", values=metric)
        paired = panels[panel_key][["interaction", row["control"]]].dropna()
        gain = paired.interaction - paired[row["control"]] if higher else paired[row["control"]] - paired.interaction
        gain = gain.groupby(["protein_id", "family_group"]).mean()
        sums = gain.groupby("family_group").sum()
        sizes = gain.groupby("family_group").size().reindex(sums.index)
        counts = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=10000)
        values = (counts @ sums.to_numpy()) / (counts @ sizes.to_numpy())
        row["gain_interval_95"] = np.quantile(values, [0.025, 0.975]).tolist()
        row["bootstrap_positive_fraction"] = float(np.mean(values > 0))
        row["bootstrap_resamples"] = 10000


def aggregate(jobs, waiting, summary_name=None, include_stabilityarc=False, evaluation_roots=None):
    destination = OUTPUT / summary_name if summary_name else OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    coverage, details = Counter(), []
    chunks = {name: [] for name in ["selection", "residual", "phenotype"]}
    for job in jobs:
        source_directory = directory(job) if evaluation_roots is None else (
            evaluation_roots[job["arm"]] / "evaluations" / job["arm"]
            / f"family{job['threshold']}" / f"seed{job['seed']}" / f"fold{job['fold']}")
        manifest = json.loads((source_directory / "manifest.json").read_text())
        assert manifest["complete"]
        details.append({**job, "normalized_variants": manifest["normalized_variants"],
                        "matched_full_evaluation_tables": manifest["matched_full_evaluation_tables"]})
        coverage[(job["threshold"], job["arm"])] += 1
        for name in chunks:
            chunks[name].append(pd.read_parquet(source_directory / (name + ".parquet")))
    external_evaluations = 0
    external_sources = []
    if include_stabilityarc:
        external = ROOT / "runs/stabilityarc_full_target"
        source = json.loads((external / "manifest.json").read_text())
        assert source["complete"] and source["target_double_training_labels"] == 0
        thresholds = {job["threshold"] for job in jobs}
        for name in chunks:
            table = pd.read_parquet(external / (name + "_per_protein.parquet"))
            table = table.loc[table.arm.eq("stabilityarc_kermut") & table.threshold.isin(thresholds)].copy()
            grid = table[["threshold", "fold", "seed"]].drop_duplicates()
            assert len(grid) == 25 * len(thresholds)
            assert set(grid.seed) == {11, 23, 47, 89, 131}
            assert set(grid.fold) == set(range(5))
            chunks[name].append(table)
        for threshold in sorted(thresholds):
            coverage[(threshold, "stabilityarc_kermut")] = 25
            for fold in range(5):
                for seed in [11, 23, 47, 89, 131]:
                    details.append({"threshold": threshold, "fold": fold, "seed": seed,
                                    "arm": "stabilityarc_kermut", "source_manifest": str(external / "manifest.json")})
        external_evaluations = 25 * len(thresholds)
        external_sources.append(str(external / "manifest.json"))
    tables = {name: pd.concat(chunks[name], ignore_index=True) for name in chunks}
    tables["phenotype"] = tables["phenotype"].loc[tables["phenotype"].phenotype_count.ge(30)]
    summary = []
    for metric in ["compensation_hit_rate", "selected_mean_recovery", "selected_mean_stability", "precision_lower", "precision_upper"]:
        summary.extend(paired_summary(tables["selection"], ["threshold", "task", "margin", "unit", "scoring", "budget"], metric, True, coverage))
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        summary.extend(paired_summary(tables["residual"], ["threshold", "margin"], metric, False, coverage))
    for metric, higher in [("phenotype_rmse", False), ("phenotype_spearman", True), ("phenotype_pearson", True)]:
        summary.extend(paired_summary(tables["phenotype"], ["threshold"], metric, higher, coverage))
    bootstrap(summary, tables)
    for name, table in tables.items():
        table.to_parquet(destination / (name + "_per_protein.parquet"), index=False)
    (destination / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    (destination / "evaluation_coverage.json").write_text(json.dumps(details, indent=2))
    manifest = {"complete": not waiting, "available_jobs_complete": True,
                "expected_evaluations": len(jobs) + len(waiting) + external_evaluations,
                "completed_evaluations": len(jobs) + external_evaluations,
                "waiting_evaluations": len(waiting), "waiting_jobs": waiting,
                "additional_external_evaluations": external_evaluations, "external_sources": external_sources,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "run_path": str(destination), "protocol": str(RUN / "protocol.json"),
                "completed_grids": [{"threshold": key[0], "arm": key[1], "evaluations": count} for key, count in sorted(coverage.items())],
                "protein_level_metrics_match_full_protocol": True, "paired_comparisons": len(summary),
                "outputs": ["selection_per_protein.parquet", "residual_per_protein.parquet", "phenotype_per_protein.parquet", "paired_summary.json"]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key not in ["waiting_jobs", "completed_grids"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[30, 25], default=[30, 25])
    parser.add_argument("--summary-name")
    parser.add_argument("--include-stabilityarc", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initialize()
    jobs, waiting = [], []
    for job in json.loads((RUN / "jobs.json").read_text()):
        if job["threshold"] not in args.thresholds:
            continue
        truth = TRUTH[(job["threshold"], job["fold"])]
        if definitions()[job["arm"]]["requires_structure"]:
            truth = truth.loc[truth.structure_available]
        _, missing = input_records(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", truth)
        if missing:
            waiting.append(job)
        else:
            jobs.append(job)
    if args.aggregate_only:
        assert args.summary_name
        assert all((directory(job) / "manifest.json").exists() for job in jobs), "Selected evaluations are still running"
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
            list(pool.map(evaluate, jobs))
    aggregate(jobs, waiting, args.summary_name, args.include_stabilityarc)


if __name__ == "__main__":
    main()
