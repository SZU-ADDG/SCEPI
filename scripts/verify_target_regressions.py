import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.higher import HigherFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = (ROOT / args.run_root).resolve()
    assert root.is_relative_to(ROOT)
    manifests = sorted(root.glob("*/order*/*/seed*/*/*/manifest.json"))
    assert manifests
    current_key, features, rows = None, None, []
    for path in manifests:
        information = json.loads(path.read_text())
        assert information["complete"]
        if not information["evaluated"]:
            continue
        key = (information["dataset"], information["protein_id"])
        if key != current_key:
            features = HigherFeatures(ROOT / "data/features/higher" / key[0] / key[1])
            current_key = key
        arrays = np.load(path.parent / "predictions.npz")
        assert arrays["variants"].shape == (len(features.metadata),)
        assert arrays["singles"].shape == (len(features.single_metadata),)
        assert np.isfinite(arrays["variants"]).all() and np.isfinite(arrays["singles"]).all()
        assert np.isfinite(arrays["reference"]).all()
        by_mutation = pd.Series(np.arange(len(features.single_metadata)), index=features.single_metadata.mutation_key)
        singles = features.metadata.loc[features.metadata.n_mutations.eq(1)]
        direct = arrays["variants"][singles.index.to_numpy()]
        counterfactual = arrays["singles"][singles.mutation_key.map(by_mutation).to_numpy(dtype=np.int64)]
        np.testing.assert_allclose(direct, counterfactual, rtol=1e-6, atol=1e-5)
        reference_rows = features.metadata.index[features.metadata.n_mutations.eq(0)]
        if len(reference_rows):
            np.testing.assert_allclose(arrays["variants"][reference_rows], arrays["reference"], rtol=1e-6, atol=1e-5)
        double_path = path.parent / "double_predictions.parquet"
        if double_path.exists():
            doubles = pd.read_parquet(double_path)
            computed = doubles.prediction_mu - doubles.prediction_a - doubles.prediction_b + doubles.prediction_wt
            np.testing.assert_allclose(computed, doubles.prediction_epsilon, rtol=0, atol=1e-10)
        ledger = pd.read_parquet(path.parent / "label_permissions.parquet")
        train = set(ledger.loc[ledger.role.eq("train"), "variant_id"])
        validation = set(ledger.loc[ledger.role.eq("validation"), "variant_id"])
        test = pd.read_parquet(path.parent / "predictions.parquet").query("set == 'test'")
        assert not train & (validation | set(test.variant_id))
        rows.append({"manifest": str(path.relative_to(ROOT)), "single_counterfactuals": len(singles),
                     "reference_counterfactuals": len(reference_rows),
                     "maximum_single_difference": float(np.max(np.abs(direct - counterfactual))) if len(singles) else 0.0,
                     "test_variants": len(test), "label_disjointness": True})
    assert rows
    output = root / "counterfactual_verification"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "checks.csv", index=False)
    result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "verified_fits": len(rows), "single_counterfactuals": sum(row["single_counterfactuals"] for row in rows),
              "reference_counterfactuals": sum(row["reference_counterfactuals"] for row in rows),
              "maximum_single_difference": max(row["maximum_single_difference"] for row in rows),
              "tolerance": {"absolute": 1e-5, "relative": 1e-6}, "label_disjointness": True}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
