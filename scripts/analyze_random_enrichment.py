import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [1, 3, 10, 30]
MARGINS = [0.0, 0.25, 0.5]
SEEDS = [11, 23, 47, 89, 131]


def candidate_expectations(threshold, output):
    rows = []
    for fold in range(5):
        path = ROOT / "runs/sall_full_target/truth" / f"family{threshold}/fold{fold}.parquet"
        truth = pd.read_parquet(path)
        assert truth.variant_id.is_unique and truth.outer_group.eq(fold).all()
        for margin in MARGINS:
            candidates = truth.loc[
                truth.selection_pool & truth.main.fillna(False)
                & truth.compensation_pool & truth.margin.ge(margin)
            ].copy()
            counts = candidates.groupby("protein_id").size()
            candidates = candidates.loc[candidates.protein_id.isin(counts.index[counts.ge(100)])].copy()
            candidates["recovery"] = candidates.relative_point - candidates[["single_a", "single_b"]].max(axis=1)
            candidates["hit"] = candidates.recovery.ge(0.5)
            assert candidates.known_single_a.all() and candidates.known_single_b.all()
            assert candidates.single_a.le(-0.5).all() and candidates.single_b.le(-0.5).all()
            assert np.isfinite(candidates.recovery).all()
            saved = candidates[["variant_id", "protein_id", "family_group", "recovery", "hit"]]
            saved.to_parquet(output / f"candidates_family{threshold}_fold{fold}_margin{margin:g}.parquet", index=False)
            for protein, group in candidates.groupby("protein_id", sort=True):
                size, hits = len(group), int(group.hit.sum())
                rate = hits / size
                for budget in BUDGETS:
                    expected_hits = budget * rate
                    hit_rate_variance = rate * (1 - rate) * (size - budget) / (budget * (size - 1))
                    if 0 < hits < size:
                        np.testing.assert_allclose(expected_hits, hypergeom.mean(size, hits, budget), atol=1e-14, rtol=0)
                        np.testing.assert_allclose(
                            hit_rate_variance, hypergeom.var(size, hits, budget) / budget**2,
                            atol=1e-15, rtol=0,
                        )
                    else:
                        assert expected_hits in [0, budget] and hit_rate_variance == 0
                    rows.append({
                        "threshold": threshold, "fold": fold, "margin": margin,
                        "protein_id": protein, "family_group": group.family_group.iloc[0],
                        "budget": budget, "pool_size": size, "pool_hits": hits,
                        "random_hit_rate": rate, "random_expected_hits": expected_hits,
                        "random_hit_rate_variance": hit_rate_variance,
                        "random_mean_recovery": float(group.recovery.mean()),
                    })
    result = pd.DataFrame(rows)
    assert not result.duplicated(["threshold", "margin", "protein_id", "budget"]).any()
    for margin, expected in [(0.0, 120), (0.25, 120), (0.5, 115)]:
        assert result.loc[result.margin.eq(margin)].groupby("budget").protein_id.nunique().eq(expected).all()
    return result


def paired_analysis(selection, random, draws):
    keys = ["threshold", "fold", "margin", "protein_id", "family_group", "budget"]
    selected = selection.loc[
        selection.task.eq("compensation") & selection.unit.eq("protein")
        & selection.budget.isin(BUDGETS) & selection.margin.isin(MARGINS)
    ].copy()
    assert set(selected.seed) == set(SEEDS)
    assert selected.groupby([*keys, "arm", "scoring"]).seed.nunique().eq(5).all()
    paired = selected.merge(random, on=keys, how="left", suffixes=("", "_random"), validate="many_to_one")
    assert paired.random_hit_rate.notna().all()
    assert paired.pool_size.eq(paired.pool_size_random).all()
    protein_keys = [*keys, "arm", "scoring"]
    protein = paired.groupby(protein_keys, as_index=False).agg(
        model_hit_rate=("compensation_hit_rate", "mean"),
        model_mean_recovery=("selected_mean_recovery", "mean"),
        random_hit_rate=("random_hit_rate", "first"),
        random_mean_recovery=("random_mean_recovery", "first"),
        pool_size=("pool_size", "first"), pool_hits=("pool_hits", "first"),
    )
    protein["hit_rate_gain"] = protein.model_hit_rate - protein.random_hit_rate
    protein["recovery_gain"] = protein.model_mean_recovery - protein.random_mean_recovery
    rng = np.random.default_rng(20261019)
    resampling = {}
    summary = []
    for (threshold, margin, scoring, budget, arm), frame in protein.groupby(
        ["threshold", "margin", "scoring", "budget", "arm"], sort=True
    ):
        frame = frame.sort_values(["family_group", "protein_id"])
        groups = frame.groupby("family_group", sort=True)
        sizes = groups.size().to_numpy()
        group_names = tuple(groups.size().index)
        key = (int(threshold), float(margin), tuple(frame.protein_id), group_names)
        if key not in resampling:
            resampling[key] = rng.multinomial(len(sizes), np.full(len(sizes), 1 / len(sizes)), size=draws)
        counts = resampling[key]
        columns = ["model_hit_rate", "random_hit_rate", "model_mean_recovery", "random_mean_recovery"]
        sums = groups[columns].sum().to_numpy()
        boot = (counts @ sums) / (counts @ sizes)[:, None]
        means = frame[columns].mean()
        assert means.random_hit_rate > 0 and np.all(boot[:, 1] > 0)
        hit_gain = boot[:, 0] - boot[:, 1]
        enrichment = boot[:, 0] / boot[:, 1]
        recovery_gain = boot[:, 2] - boot[:, 3]
        record = {
            "threshold": int(threshold), "margin": float(margin), "scoring": scoring,
            "budget": int(budget), "arm": arm, "proteins": len(frame), "family_groups": len(sizes),
            "candidates": int(frame.pool_size.sum()), "pool_hits": int(frame.pool_hits.sum()),
            "model_hit_rate": float(means.model_hit_rate), "random_hit_rate": float(means.random_hit_rate),
            "hit_rate_gain": float(means.model_hit_rate - means.random_hit_rate),
            "enrichment": float(means.model_hit_rate / means.random_hit_rate),
            "model_mean_recovery": float(means.model_mean_recovery),
            "random_mean_recovery": float(means.random_mean_recovery),
            "recovery_gain": float(means.model_mean_recovery - means.random_mean_recovery),
            "hit_gain_interval_95": np.quantile(hit_gain, [0.025, 0.975]).tolist(),
            "enrichment_interval_95": np.quantile(enrichment, [0.025, 0.975]).tolist(),
            "recovery_gain_interval_95": np.quantile(recovery_gain, [0.025, 0.975]).tolist(),
            "bootstrap_positive_fraction": float(np.mean(hit_gain > 0)), "bootstrap_resamples": draws,
        }
        summary.append(record)
    return protein, summary, len(paired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=ROOT / "runs/key_ablations_20261015/paper_evidence")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reviewer_revision_20261019/random_enrichment")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    assert args.bootstrap_draws >= 10000
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.evidence / "manifest.json").read_text())
    integration = json.loads((args.evidence / "integration_verification.json").read_text())
    assert source["available_jobs_complete"] and integration["complete"]
    assert len(source["completed_grids"]) >= 16 and len(source["completed_grids"]) % 2 == 0
    assert integration["completed_arms"] * 2 == len(source["completed_grids"])
    assert source["completed_evaluations"] == sum(row["evaluations"] for row in source["completed_grids"])
    assert all(row["evaluations"] == 25 for row in source["completed_grids"])
    random = pd.concat([candidate_expectations(threshold, args.output) for threshold in [30, 25]], ignore_index=True)
    selection = pd.read_parquet(args.evidence / "selection_per_protein.parquet")
    figure_arms = ["interaction", "additive", "global_nonlinear", "thermo_epistatic", "kermut",
                   "stabilityarc_kermut", "mean_mlp", "mean_ridge"]
    figure_selection = selection.loc[
        selection.threshold.eq(30) & selection.task.eq("compensation") & selection.unit.eq("protein")
        & selection.margin.eq(0.0) & selection.scoring.eq("native") & selection.arm.isin(figure_arms)
    ]
    figure_selection.to_parquet(args.output / "figure_selection.parquet", index=False)
    protein, summary, paired_count = paired_analysis(selection, random, args.bootstrap_draws)
    random.to_parquet(args.output / "random_per_protein.parquet", index=False)
    random.to_csv(args.output / "random_per_protein.csv", index=False)
    protein.to_parquet(args.output / "paired_per_protein.parquet", index=False)
    protein.to_csv(args.output / "paired_per_protein.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "complete": True, "source_commit": source_commit, "analysis_script": "scripts/analyze_random_enrichment.py",
        "source_evidence": str(args.evidence.relative_to(ROOT)), "run_path": str(args.output.relative_to(ROOT)),
        "target_combination_labels_used_for_fitting": 0, "analysis_type": "Post-review retrospective random-selection analysis",
        "random_selection": "Uniform sampling without replacement from each unchanged eligible protein candidate pool.",
        "random_expectation": "Exact pool hit fraction and pool mean recovery; no Monte Carlo random rankings.",
        "enrichment": "Ratio of equal-protein mean model hit rate to equal-protein mean random hit rate; not a mean of per-protein ratios.",
        "aggregation": "Average five matched fitting repetitions within protein, then give proteins equal weight.",
        "uncertainty": "10000 paired homology-group bootstrap draws; both numerator and denominator resampled together.",
        "bootstrap_seed": 20261019, "thresholds": [30, 25], "margins": MARGINS, "budgets": BUDGETS,
        "random_protein_budget_records": len(random), "paired_repetition_records_checked": paired_count,
        "paired_protein_records": len(protein), "summary_rows": len(summary),
        "checks": ["candidate eligibility", "five matched repetitions", "pool-size equality with original evaluation", "hypergeometric mean and variance"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    primary = [row for row in summary if row["arm"] == "interaction" and row["scoring"] == "native" and row["budget"] == 10]
    print(json.dumps({"complete": True, "summary_rows": len(summary), "SCEPI_top10": primary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
