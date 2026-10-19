import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/e1_esm_msr"


def correlation(x, y, function):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(function(x, y).statistic)


def main():
    reference = pd.read_parquet(ROOT / "data/processed/esm_msr_benchmark/variants.parquet").sort_values("sample_id").reset_index(drop=True)
    ranks = [json.loads((OUTPUT / f"manifest_rank{rank}.json").read_text()) for rank in range(8)]
    assert all(item["complete"] and not item["smoke"] for item in ranks)
    assert sum(item["variants_per_checkpoint"] for item in ranks) == len(reference)
    frames = []
    for seed in [1, 2, 3]:
        paths = sorted(OUTPUT.glob(f"seed{seed}_*.parquet"))
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        assert len(frame) == len(reference) and frame.sample_id.is_unique
        for column in ["sample_id", "partition", "code", "mut_type", "aa_seq"]:
            assert frame[column].equals(reference[column]), column
        np.testing.assert_allclose(frame.ddG_ML, reference.ddG_ML, rtol=0, atol=0)
        prediction_columns = ["wt_lora_pred", "mt_lora_pred", "combined_pred", "combined_dddg_pred"]
        assert np.isfinite(frame[prediction_columns]).all().all()
        frames.append(frame)
    ensemble = frames[0].copy()
    for column in prediction_columns:
        ensemble[column] = np.mean([frame[column].to_numpy() for frame in frames], axis=0)
    ensemble["checkpoint_seed"] = "ensemble_1_2_3"
    ensemble["prediction_stability"] = -ensemble.combined_pred
    ensemble["observed_stability"] = -ensemble.ddG_ML
    ensemble["predicted_epistasis_stability"] = -ensemble.combined_dddg_pred
    ensemble["observed_epistasis_stability"] = -ensemble.dddG_ML
    ensemble.to_parquet(OUTPUT / "ensemble_predictions.parquet", index=False)
    rows = []
    for label, frame in [(str(seed), value) for seed, value in zip([1, 2, 3], frames)] + [("ensemble", ensemble)]:
        for partition, subset in frame.groupby("partition", sort=True):
            for mutation_order, current in [("all", subset), ("single", subset.loc[subset.pos2.isna()]), ("double", subset.loc[subset.pos2.notna()])]:
                for endpoint, target, prediction in [
                    ("ddG", "ddG_ML", "combined_pred"), ("ddG_WT_branch", "ddG_ML", "wt_lora_pred"),
                    ("ddG_MT_branch", "ddG_ML", "mt_lora_pred"),
                    ("author_residual", "dddG_ML", "combined_dddg_pred"),
                ]:
                    valid = current.loc[current[target].notna()]
                    if valid.empty:
                        continue
                    for protein, group in [("__pooled__", valid), *list(valid.groupby("code", sort=True))]:
                        observed = group[target].to_numpy()
                        predicted = group[prediction].to_numpy()
                        rows.append({
                            "checkpoint": label, "partition": partition, "mutation_order": mutation_order,
                            "endpoint": endpoint, "protein": protein, "n": len(group),
                            "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                            "spearman": correlation(observed, predicted, spearmanr),
                            "pearson": correlation(observed, predicted, pearsonr),
                        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    macro = metrics.loc[metrics.protein.ne("__pooled__")].groupby(
        ["checkpoint", "partition", "mutation_order", "endpoint"], as_index=False
    ).agg(proteins=("protein", "size"), n=("n", "sum"), macro_protein_rmse=("rmse", "mean"),
          macro_protein_spearman=("spearman", "mean"), macro_protein_pearson=("pearson", "mean"))
    macro.to_csv(OUTPUT / "macro_metrics.csv", index=False)
    manifest = {
        "complete": True, "protocol": "author_hyperopt_splits_validation_and_test",
        "variants": len(reference), "counts": reference.groupby("partition").size().to_dict(),
        "proteins": reference.groupby("partition").code.nunique().to_dict(),
        "checkpoint_seeds": [1, 2, 3], "raw_predictions": len(reference) * 3,
        "scope": "Original author preprocessing, fixed public LoRA weights and original partitions. New family-isolation claims require independent retraining.",
        "directions": "Author ddG is destabilizing-positive; physical stability and residual columns are negated.",
        "author_residual": "Uses the author's combined_dddg_pred output and original dddG_ML labels.",
        "rank_manifests": [f"manifest_rank{rank}.json" for rank in range(8)],
        "source_commits": sorted({item["source_commit"] for item in ranks}),
        "evaluation_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "gpus": [item["gpu"] for item in ranks],
        "outputs": ["ensemble_predictions.parquet", "metrics.csv", "macro_metrics.csv"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(macro.loc[macro.checkpoint.eq("ensemble") & macro.endpoint.isin(["ddG", "author_residual"])].to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
