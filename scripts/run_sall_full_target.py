import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import BUDGETS, MARGINS, SEEDS, definitions, input_records, normalized_predictions, residual_metrics, selection_metrics
from epistasis.metrics import per_protein_metrics
from epistasis.evaluation_scope import authorized_jobs, scope_status

OUTPUT = ROOT / "runs/sall_full_target"
ARMS = {name: item for name, item in definitions().items() if item["conditions"] == ["Sall"]}


def commit():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def directory(job):
    return OUTPUT / "evaluations" / job["arm"] / f"family{job['threshold']}" / f"seed{job['seed']}" / f"fold{job['fold']}"


def prepare():
    if (OUTPUT / "protocol.json").exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet", columns=["variant_id", "point"]).set_index("variant_id")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet", columns=["variant_id", "seed"])
    available = {seed: set(order.loc[order.seed.eq(seed), "variant_id"]) for seed in SEEDS}
    assert all(available[seed] == available[SEEDS[0]] for seed in SEEDS)
    truth, coverage, jobs = {}, [], []
    metadata = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet", columns=["variant_id", "protein_id"])
    for threshold in [30, 25]:
        for fold in range(5):
            source = ROOT / "runs/original_main/truth" / f"family{threshold}" / f"fold{fold}.parquet"
            frame = pd.read_parquet(source)
            assert frame.variant_id.is_unique and frame.outer_group.eq(fold).all()
            expected = metadata.loc[metadata.protein_id.isin(frame.protein_id.unique()), "variant_id"]
            assert set(frame.variant_id) == set(expected)
            frame["single_a"] = frame.a_id.map(variants.point) - frame.wt_point
            frame["single_b"] = frame.b_id.map(variants.point) - frame.wt_point
            frame["common_observed_main"] = frame.single_a + frame.single_b
            frame["known_single_a"] = frame.a_id.isin(available[SEEDS[0]])
            frame["known_single_b"] = frame.b_id.isin(available[SEEDS[0]])
            frame["selection_pool"] = frame.wt_all_point & frame.wt_point_in_range & frame.wt_point.le(4.5)
            frame["compensation_pool"] = (frame.selection_pool & frame.main.fillna(False) & frame.known_single_a & frame.known_single_b
                                            & frame.single_a.le(-0.5) & frame.single_b.le(-0.5))
            point = frame.loc[frame.main.fillna(False)]
            np.testing.assert_allclose(point.common_observed_main + point.epsilon, point.relative_point, atol=1e-6)
            target = OUTPUT / "truth" / f"family{threshold}" / f"fold{fold}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.sort_values("variant_id").to_parquet(target, index=False)
            truth[(threshold, fold)] = frame
            counts = frame.loc[frame.compensation_pool].groupby("protein_id").size()
            eligible = frame.loc[frame.compensation_pool & frame.protein_id.isin(counts.index[counts.ge(100)])]
            coverage.append({"threshold": threshold, "fold": fold, "variants": len(frame), "proteins": frame.protein_id.nunique(),
                             "continuous_quartets": int(frame.main.sum()), "compensation_proteins": eligible.protein_id.nunique(),
                             "compensation_groups": eligible.family_group.nunique(), "compensation_candidates": len(eligible),
                             "truth": str(target.relative_to(ROOT))})
            for seed in SEEDS:
                for arm, specification in ARMS.items():
                    assert threshold in specification["thresholds"]
                    jobs.append({"arm": arm, "threshold": threshold, "seed": seed, "fold": fold, "condition": "Sall"})
    protocol = {"source_commit": commit(), "run_path": str(OUTPUT), "condition": "Sall", "target_double_training_labels": 0,
                "scientific_claim": "A shared constrained interaction predictor improves full-target double-mutant prediction and compensatory discovery in held-out homology groups.",
                "evaluation_population": "Every accepted target double from the frozen feature metadata. No target site-pair test mask is applied.",
                "source_holdout": "Existing outer homology groups and source-validation selections are retained. Target double labels do not enter source training or target adaptation.",
                "target_inputs": "All permitted target single-mutant measurements. Seed-specific source fits and target fits remain separate.",
                "primary_comparison": "Native phenotype predictions on the same eligible proteins and variants for each paired comparison.",
                "compensation": "Both measured single effects <= -0.5 kcal/mol; complete point-valued quartets; reference <= 4.5 kcal/mol; minimum 100 candidates per protein; hit recovery >= 0.5 kcal/mol.",
                "candidate_status": "Measurement status and boundary margins define retrospective evaluation strata. Target double outcomes never select checkpoints or prediction scores.",
                "component_comparison": "Common observed single sum plus the four-prediction residual.",
                "aggregation": "Paired protein means after averaging matched source/target repeats. Family groups define the bootstrap resampling unit.",
                "budgets": BUDGETS, "margins": MARGINS, "seeds": SEEDS, "thresholds": [30, 25], "folds": list(range(5)),
                "arms": ARMS, "expected_evaluations": len(jobs), "legacy_subset": "runs/compensation_main"}
    (OUTPUT / "candidate_coverage.json").write_text(json.dumps(coverage, indent=2))
    (OUTPUT / "jobs.json").write_text(json.dumps(jobs, indent=2))
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))


def evaluate(job, truth, entries):
    target = directory(job)
    target.mkdir(parents=True, exist_ok=True)
    assert not (target / "manifest.json").exists()
    frame, excluded, provenance = normalized_predictions(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", truth, entries)
    residual = residual_metrics(frame)
    selection = selection_metrics(frame)
    phenotype = per_protein_metrics(frame)
    if len(phenotype):
        phenotype = phenotype.merge(frame[["protein_id", "family_group", "pfam_annotated", "origin"]].drop_duplicates(), on="protein_id", validate="one_to_one")
    for name, table in [("residual_per_protein", residual), ("selection_per_decision", selection), ("phenotype_per_protein", phenotype)]:
        for key, value in job.items():
            table[key] = value
        table.to_parquet(target / (name + ".parquet"), index=False)
    manifest = {"complete": True, "smoke": False, "source_commit": commit(), **job,
                "protocol": str((OUTPUT / "protocol.json").relative_to(ROOT)), "input_predictions": provenance,
                "excluded_proteins": excluded, "normalized_variants": len(frame), "proteins": frame.protein_id.nunique(),
                "full_target_coverage_verified": True, "four_prediction_contrast_verified": True,
                "outputs": ["residual_per_protein.parquet", "selection_per_decision.parquet", "phenotype_per_protein.parquet"]}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"complete": True, **job, "variants": len(frame)}), flush=True)
    return job


def paired_summary(table, keys, metric, higher, coverage):
    rows = []
    if not len(table):
        return rows
    for values, group in table.groupby(keys, dropna=False, observed=True, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        settings = {key: value.item() if isinstance(value, np.generic) else value
                    for key, value in zip(keys, values)}
        pivot = group.pivot(index=["seed", "protein_id", "family_group"], columns="arm", values=metric)
        if "interaction" not in pivot:
            continue
        for control in pivot.columns:
            if control == "interaction":
                continue
            paired = pivot[["interaction", control]].dropna()
            if not len(paired):
                continue
            first = paired.interaction
            second = paired[control]
            difference = first - second if higher else second - first
            means = paired.groupby(["protein_id", "family_group"]).mean()
            gains = difference.groupby(["protein_id", "family_group"]).mean()
            repeats = difference.groupby("seed").mean()
            threshold = settings["threshold"]
            complete = all(coverage.get((threshold, arm), 0) == 25 for arm in ["interaction", control])
            rows.append({**settings, "metric": metric, "reference": "interaction", "control": control,
                         "proteins": len(means), "family_groups": means.index.get_level_values("family_group").nunique(),
                         "paired_protein_repeats": len(paired), "complete_grid": complete,
                         "reference_mean": float(means.interaction.mean()), "control_mean": float(means[control].mean()),
                         "mean_gain": float(gains.mean()), "wins": int(gains.gt(0).sum()), "losses": int(gains.lt(0).sum()),
                         "ties": int(gains.eq(0).sum()), "repeat_gains": {str(int(seed)): float(value) for seed, value in repeats.items()},
                         "positive_repeats": int(repeats.gt(0).sum()), "repeats": len(repeats)})
    return rows


def aggregate(jobs):
    tables = {name: [] for name in ["residual_per_protein", "selection_per_decision", "phenotype_per_protein"]}
    coverage = Counter()
    detailed = []
    for job in jobs:
        path = directory(job) / "manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        assert manifest["complete"] and manifest["full_target_coverage_verified"]
        coverage[(job["threshold"], job["arm"])] += 1
        detailed.append({**job, "normalized_variants": manifest["normalized_variants"], "excluded_proteins": manifest["excluded_proteins"]})
        for name in tables:
            table = pd.read_parquet(path.parent / (name + ".parquet"))
            if name == "selection_per_decision":
                numeric = [column for column in table if column.startswith("selected_") and column not in ["selected_variant_ids", "selected_scores", "selected_relative_point", "selected_recovery"]]
                numeric += ["precision_lower", "precision_upper", "compensation_hit_rate", "compensation_hits"]
                table = table.groupby(["arm", "threshold", "seed", "fold", "condition", "protein_id", "family_group", "task", "margin", "unit", "scoring", "budget"], dropna=False, observed=True)[numeric].mean().reset_index()
            tables[name].append(table)
    joined = {}
    for name, chunks in tables.items():
        joined[name] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        joined[name].to_parquet(OUTPUT / (name + ".parquet"), index=False)
    summaries = []
    selection = joined["selection_per_decision"]
    selection_keys = ["threshold", "task", "margin", "unit", "scoring", "budget"]
    for metric in ["compensation_hit_rate", "selected_mean_recovery", "selected_mean_stability", "selected_best_recovery", "precision_lower", "precision_upper"]:
        summaries.extend(paired_summary(selection, selection_keys, metric, True, coverage))
    residual = joined["residual_per_protein"]
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        summaries.extend(paired_summary(residual, ["threshold", "margin"], metric, False, coverage))
    phenotype = joined["phenotype_per_protein"]
    if len(phenotype):
        phenotype = phenotype.loc[phenotype.phenotype_count.ge(30)]
    for metric, higher in [("phenotype_rmse", False), ("phenotype_spearman", True), ("phenotype_pearson", True)]:
        summaries.extend(paired_summary(phenotype, ["threshold"], metric, higher, coverage))
    (OUTPUT / "paired_summary.json").write_text(json.dumps(summaries, indent=2))
    (OUTPUT / "evaluation_coverage.json").write_text(json.dumps(detailed, indent=2))
    manifest = {"complete": len(detailed) == len(jobs), "source_commit": commit(), "run_path": str(OUTPUT),
                "protocol": str((OUTPUT / "protocol.json").relative_to(ROOT)), "expected_evaluations": len(jobs),
                "completed_evaluations": len(detailed), "remaining_evaluations": len(jobs) - len(detailed),
                "completed_grids": [{"threshold": key[0], "arm": key[1], "evaluations": value} for key, value in sorted(coverage.items())],
                "paired_comparisons": len(summaries), "statistics": "Paired equal-protein means after matched repeat averaging; positive mean gain is the effect criterion.",
                "outputs": ["candidate_coverage.json", "evaluation_coverage.json", "phenotype_per_protein.parquet", "residual_per_protein.parquet", "selection_per_decision.parquet", "paired_summary.json"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "completed_grids"}), flush=True)
    return len(detailed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    prepare()
    all_jobs = json.loads((OUTPUT / "jobs.json").read_text())
    jobs, discontinued = authorized_jobs(ROOT, "sall_full_target", all_jobs)
    if args.aggregate_only:
        aggregate(all_jobs)
        return
    truth = {(threshold, fold): pd.read_parquet(OUTPUT / "truth" / f"family{threshold}" / f"fold{fold}.parquet")
             for threshold in [30, 25] for fold in range(5)}
    aggregated = json.loads((OUTPUT / "manifest.json").read_text())["completed_evaluations"] if (OUTPUT / "manifest.json").exists() else 0
    while True:
        ready, waiting, finished = [], [], 0
        for job in jobs:
            path = directory(job) / "manifest.json"
            if path.exists():
                assert json.loads(path.read_text())["complete"]
                finished += 1
                continue
            current = truth[(job["threshold"], job["fold"])]
            if ARMS[job["arm"]]["requires_structure"]:
                current = current.loc[current.structure_available]
            entries, missing = input_records(job["arm"], job["threshold"], job["seed"], job["fold"], "Sall", current)
            if missing:
                waiting.append({**job, "dependencies": missing})
            else:
                ready.append((job, current, entries))
        state = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **scope_status(len(all_jobs), discontinued, finished),
                 "completed_evaluations": finished, "ready_evaluations": len(ready), "waiting_evaluations": len(waiting),
                 "waiting_dependencies": waiting}
        (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
        print(json.dumps({key: value for key, value in state.items() if key != "waiting_dependencies"}), flush=True)
        if ready:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: evaluate(*item), ready))
        completed = finished + len(ready)
        if completed != aggregated:
            aggregated = aggregate(all_jobs)
        if completed == len(jobs) or args.once:
            state.update(**scope_status(len(all_jobs), discontinued, completed),
                         completed_evaluations=completed, ready_evaluations=0,
                         state="authorized_scope_complete" if completed == len(jobs) else "waiting_for_predictions")
            (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
