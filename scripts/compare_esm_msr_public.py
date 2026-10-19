import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "runs/e1_esm_msr/public_comparison"
    output.mkdir(parents=True, exist_ok=True)
    records, named_records, unmatched = [], [], []
    for partition in ["val", "test"]:
        for seed in [1, 2, 3]:
            public_path = ROOT / "data/extracted/esm_msr/predictions" / f"hyperopt_splits-{partition}" / "msr_dual_ensemble_small_large_release" / f"lora_seed{seed}.safetensors_epsilon1_unmasked_predictions.csv"
            published = pd.read_csv(public_path, low_memory=False)
            named_checkpoint = published
            if seed == 1:
                public_path = public_path.with_name("lora_seed1_last.safetensors_epsilon1_unmasked_predictions.csv")
                published = pd.read_csv(public_path, low_memory=False)
            reproduced = pd.concat([pd.read_parquet(path) for path in sorted((ROOT / "runs/e1_esm_msr").glob(f"seed{seed}_{partition}_*.parquet"))], ignore_index=True)
            assert not published.uid.duplicated().any() and not reproduced.uid.duplicated().any()
            join = reproduced.merge(published, on="uid", how="outer", suffixes=("_reproduction", "_public"), indicator=True, validate="one_to_one")
            missing = join.loc[join._merge.ne("both"), ["uid", "_merge"]].copy()
            if len(missing):
                unmatched.append(missing.assign(partition=partition, checkpoint_seed=seed))
            common = join.loc[join._merge.eq("both")]
            assert len(common) == len(reproduced), "Public release does not cover the complete reproduced benchmark"
            assert common.aa_seq_reproduction.eq(common.aa_seq_public).all()
            assert common.mut_type_reproduction.eq(common.mut_type_public).all()
            for field in ["ddG_ML", "ddG_additive_ML", "dddG_ML"]:
                np.testing.assert_allclose(common[field + "_reproduction"], common[field + "_public"], rtol=0, atol=1e-10, equal_nan=True)
            for field in ["wt_lora_pred", "mt_lora_pred", "combined_pred", "combined_dddg_pred", "combined_pred_additive"]:
                a = common[field + "_reproduction"].to_numpy(dtype=np.float64)
                b = common[field + "_public"].to_numpy(dtype=np.float64)
                assert np.array_equal(np.isfinite(a), np.isfinite(b))
                use = np.isfinite(a)
                difference = a[use] - b[use]
                assert np.max(np.abs(difference)) < 1e-4, (partition, seed, field)
                named_values = named_checkpoint.set_index("uid")[field].reindex(common.uid).to_numpy(dtype=np.float64)
                named_difference = a[use] - named_values[use]
                named_records.append({"partition": partition, "checkpoint_seed": seed, "field": field,
                                      "compared": int(use.sum()),
                                      "maximum_absolute_difference": float(np.max(np.abs(named_difference))),
                                      "mean_absolute_difference": float(np.mean(np.abs(named_difference)))})
                records.append({"partition": partition, "checkpoint_seed": seed, "field": field,
                                "compared": int(use.sum()), "public_rows": len(published), "reproduced_rows": len(reproduced),
                                "maximum_absolute_difference": float(np.max(np.abs(difference))),
                                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                                "root_mean_square_difference": float(np.sqrt(np.mean(difference ** 2))),
                                "pearson": float(np.corrcoef(a[use], b[use])[0, 1]),
                                "public_file": str(public_path.relative_to(ROOT))})
            common[["uid", "code_reproduction", "combined_pred_reproduction", "combined_pred_public",
                    "combined_dddg_pred_reproduction", "combined_dddg_pred_public"]].to_parquet(
                        output / f"{partition}_seed{seed}.parquet", index=False)
    pd.DataFrame(records).to_csv(output / "numerical_comparison.csv", index=False)
    if unmatched:
        pd.concat(unmatched, ignore_index=True).to_csv(output / "unmatched_public_rows.csv", index=False)
    pd.DataFrame(named_records).to_csv(output / "named_checkpoint_comparison.csv", index=False)
    manifest = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "public_release": "Zenodo 21539278",
                "weight_prediction_correspondence": {"seed1": "Released weights reproduce the author prediction file labeled lora_seed1_last.", "seed2": "lora_seed2", "seed3": "lora_seed3"},
                "inference": {"epsilon": 1, "mask_strategy": None,
                "skip_reverse": False, "skip_additive": False, "checkpoint_seeds": [1, 2, 3]},
                "complete_reproduced_benchmark_matched": True, "metrics": records,
                "reproduction_manifest": "runs/e1_esm_msr/manifest.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(records).query("field == 'combined_pred'").drop(columns="public_file").to_string(index=False))


if __name__ == "__main__":
    main()
