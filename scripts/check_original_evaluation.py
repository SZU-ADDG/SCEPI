import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.main_evaluation import OUTPUT, frozen_truth, functional_entries, functional_metrics, normalized_stability, stability_definitions, stability_entries, stability_metrics


def main():
    jobs = json.loads((OUTPUT / "jobs.json").read_text())
    cases = [("tsuboyama", "original_interaction", "Z"), ("tsuboyama", "original_interaction", "S128"),
             ("tsuboyama", "interaction", "Sall"), ("tsuboyama", "esm2_context", "S128"),
             ("tsuboyama", "mean_ridge", "Z"), ("tsuboyama", "kermut", "S128"),
             ("tsuboyama", "mean_ridge", "Sall"), ("tsuboyama", "interaction_C", "C128"),
             ("tsuboyama", "interaction_C_r0", "C128"), ("tsuboyama", "interaction_target_only_C", "C128"),
             ("tsuboyama", "mochi_order2_C", "C128"), ("functional_doubles", "transfer_shared", "C128"),
             ("functional_doubles", "mochi_order2", "C128"), ("flip_primary", "transfer_shared", "official"),
             ("flip_primary", "mean_ridge", "official")]
    records = []
    for dataset, arm, condition in cases:
        candidates = [job for job in jobs if job["dataset"] == dataset and job["arm"] == arm and job["condition"] == condition
                      and job["threshold"] in [0, 30] and job["seed"] == 11 and job["scheme"] in ["family", "unseen_pair", "amylase_one_to_many"]]
        assert candidates, (dataset, arm, condition)
        chosen, entries, truth = None, None, None
        for job in sorted(candidates, key=lambda item: item["name"]):
            current = frozen_truth(job)
            values, missing = stability_entries(job, sorted(current.protein_id.unique())) if dataset == "tsuboyama" else functional_entries(job)
            if not missing:
                chosen, entries, truth = job, values, current
                break
        assert chosen is not None, (dataset, arm, condition)
        comparisons = []
        if dataset == "tsuboyama":
            frame, excluded, provenance = normalized_stability(chosen, truth, entries)
            summary, strata, selections = stability_metrics(frame)
            for entry in entries:
                if not entry["information"].get("evaluated", True):
                    continue
                previous = pd.read_parquet(entry["manifest"].parent / "per_protein.parquet")
                if "condition" in previous:
                    previous = previous.loc[previous.condition.eq(condition)]
                if "method" in previous:
                    specification = stability_definitions()[arm]
                    method = specification["prediction_method"] if specification["kind"] == "pair" else arm
                    previous = previous.loc[previous.method.eq(method)]
                if "partition" in previous:
                    partition = "family_pool" if arm == "kermut" and chosen["scheme"] == "family" else "test"
                    previous = previous.loc[previous.partition.eq(partition)]
                merged = summary.merge(previous[["protein_id", "epsilon_rmse"]], on="protein_id", suffixes=("", "_saved"), validate="one_to_one")
                if arm == "interaction_C_r0":
                    independent = frame.loc[frame.main].groupby("protein_id").epsilon.apply(lambda values: float(np.sqrt(np.mean(values ** 2))))
                    np.testing.assert_allclose(summary.set_index("protein_id").epsilon_rmse, independent.reindex(summary.protein_id), rtol=1e-6, atol=1e-6)
                    comparisons.append(len(independent))
                else:
                    np.testing.assert_allclose(merged.epsilon_rmse, merged.epsilon_rmse_saved, rtol=1e-5, atol=1e-5, equal_nan=True)
                    comparisons.append(len(merged))
            assert sum(comparisons) > 0, chosen["name"]
            records.append({"job": chosen["name"], "normalized_variants": len(frame), "existing_protein_metrics_matched": sum(comparisons),
                            "stratum_rows": len(strata), "selected_candidate_decisions": len(selections), "label_provenance_records": len(provenance)})
        else:
            summary, excluded, provenance = functional_metrics(chosen, truth, entries)
            for entry in entries:
                if not entry["information"].get("evaluated", True):
                    continue
                saved = pd.read_csv(entry["prediction_directory"] / "metrics.csv")
                saved = saved.loc[saved.partition.eq("test") & saved.mutation_order.eq("all")]
                current = summary.loc[summary.mutation_order.eq("all")]
                if entry["protein"] is not None:
                    current = current.loc[current.protein_id.eq(entry["protein"])]
                if "protein_id" in saved:
                    merged = current.merge(saved[["protein_id", "phenotype_spearman"]], on="protein_id", suffixes=("", "_saved"), validate="one_to_one")
                    np.testing.assert_allclose(merged.phenotype_spearman, merged.phenotype_spearman_saved, rtol=1e-6, atol=1e-6, equal_nan=True)
                else:
                    assert len(current) == len(saved) == 1
                    np.testing.assert_allclose(current.phenotype_spearman, saved.phenotype_spearman, rtol=1e-6, atol=1e-6, equal_nan=True)
                comparisons.append(len(current))
            assert sum(comparisons) > 0
            records.append({"job": chosen["name"], "existing_protein_metrics_matched": sum(comparisons),
                            "functional_quartets": int(summary.loc[summary.mutation_order.eq("quartet"), "main_quartets"].sum()),
                            "label_provenance_records": len(provenance)})
        print(json.dumps(records[-1]), flush=True)
    directory = ROOT / "work/original_checks"
    directory.mkdir(parents=True, exist_ok=True)
    information = {"complete": True, "real_prediction_cases": len(records), "checks": records}
    (directory / "metric_verification.json").write_text(json.dumps(information, indent=2))
    print(json.dumps(information), flush=True)


if __name__ == "__main__":
    main()
