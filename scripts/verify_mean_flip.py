import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.mean_flip import MeanFlipRegression, jobs_for
from epistasis.mean_regression import MeanMLP

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", required=True)
args = parser.parse_args()
root = (ROOT / args.run_root).resolve()
assert root.is_relative_to(ROOT)
official = {(job["scheme"], job["condition"], job["seed"]): job for job in jobs_for()}
truth = pd.read_parquet(ROOT / "data/processed/flip_primary/variants.parquet").set_index("variant_id")
paths = sorted(root.glob("*/*/*/seed*/manifest.json"))
assert paths
records = []
for path in paths:
    info = json.loads(path.read_text())
    assert info["complete"]
    if not info["evaluated"]:
        continue
    assert info["source_labels_used"] == 0
    job = official[(info["scheme"], info["condition"], info["seed"])]
    permissions = pd.read_parquet(path.parent / "permissions.parquet")
    assert permissions.variant_id.is_unique
    permitted = {label: permissions.loc[permissions.partition.eq(label), "variant_id"].tolist()
                 for label in ["train", "validation", "test"]}
    assert set(permitted["train"]) <= set(job["train_ids"])
    for label in ["validation", "test"]:
        assert set(permitted[label]) == set(job[label + "_ids"])
    if not info["smoke"]:
        assert set(permitted["train"]) == set(job["train_ids"])
    proteins = sorted(truth.loc[permissions.variant_id].protein_id.unique())
    regression = MeanFlipRegression(proteins, info["method"], "cpu")
    state = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
    train, validation = state["train_indices"], state["validation_indices"]
    assert set(regression.metadata.iloc[train].variant_id) == set(permitted["train"])
    assert set(regression.metadata.iloc[validation].variant_id) == set(permitted["validation"])
    training = regression.values[train].astype(np.float64)
    expected_scale = training.std(axis=0)
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_array_equal(state["x_center"], training.mean(axis=0))
    np.testing.assert_array_equal(state["x_scale"], expected_scale)
    values = truth.loc[regression.metadata.iloc[train].variant_id].point.to_numpy()
    assert state["y_center"] == float(np.mean(values))
    assert state["y_scale"] == float(np.std(values))
    normalized = (regression.values.astype(np.float64) - state["x_center"]) / state["x_scale"]
    if info["method"] == "mean_ridge":
        estimator = Ridge(alpha=state["selected"]["ridge_alpha"], fit_intercept=False, solver="cholesky")
        estimator.fit(normalized[train], (values - state["y_center"]) / state["y_scale"])
        np.testing.assert_allclose(estimator.coef_, state["coefficient"].numpy(), rtol=1e-5, atol=1e-5)
        raw = estimator.predict(normalized) * state["y_scale"] + state["y_center"]
        coefficient_difference = float(np.max(np.abs(estimator.coef_ - state["coefficient"].numpy())))
    else:
        model = MeanMLP(state["selected"]["width"]).eval()
        model.load_state_dict(state["model"])
        with torch.inference_mode():
            raw = np.concatenate([model(torch.tensor(normalized[start:start + 4096], dtype=torch.float32)).numpy()
                                  for start in range(0, len(normalized), 4096)]) * state["y_scale"] + state["y_center"]
        coefficient_difference = None
    all_predictions = pd.read_parquet(path.parent / "all_variant_predictions.parquet")
    counterfactual = pd.read_parquet(path.parent / "counterfactual_predictions.parquet")
    assert all_predictions.variant_id.tolist() == regression.metadata.variant_id.tolist()
    assert counterfactual.row.tolist() == regression.counterfactuals.row.tolist()
    saved = np.concatenate([all_predictions.prediction_absolute.to_numpy(), counterfactual.prediction_absolute.to_numpy()])
    np.testing.assert_allclose(saved, raw, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(saved, saved[regression.prediction_alias])
    predictions = pd.read_parquet(path.parent / "predictions.parquet")
    for label in ["validation", "test"]:
        assert set(predictions.loc[predictions["set"].eq(label), "variant_id"]) == set(permitted[label])
    references = counterfactual.loc[counterfactual.kind.eq("reference")].set_index("protein_id").prediction_absolute
    np.testing.assert_allclose(predictions.prediction_mu, predictions.prediction_absolute - predictions.protein_id.map(references), rtol=0, atol=0)
    quartets = pd.read_parquet(path.parent / "quartet_predictions.parquet")
    if len(quartets):
        expected = (truth.loc[quartets.variant_id].point.to_numpy() - truth.loc[quartets.a_id].point.to_numpy()
                    - truth.loc[quartets.b_id].point.to_numpy() + truth.loc[quartets.wt_id].point.to_numpy())
        np.testing.assert_allclose(quartets.epsilon, expected, rtol=0, atol=0)
        np.testing.assert_allclose(quartets.prediction_epsilon,
                                   quartets.prediction_ab - quartets.prediction_a - quartets.prediction_b + quartets.prediction_wt,
                                   rtol=0, atol=0)
        predicted_lookup = all_predictions.set_index("variant_id").prediction_absolute
        for label in ["a", "b", "wt"]:
            np.testing.assert_array_equal(quartets["prediction_" + label], predicted_lookup.loc[quartets[label + "_id"]].to_numpy())
        for column in ["a_id", "b_id", "wt_id"]:
            assert np.array_equal(truth.loc[quartets[column]].protein_id.to_numpy(), quartets.protein_id.to_numpy())
    training_backgrounds = sorted(truth.loc[permitted["train"]].protein_id.unique())
    assert training_backgrounds == job["training_proteins"]
    row = {"manifest": str(path.relative_to(ROOT)), "method": info["method"], "replayed_genotypes": len(saved),
           "evaluated_variants": len(predictions), "complete_quartets": len(quartets),
           "maximum_prediction_difference": float(np.max(np.abs(saved - raw))),
           "sklearn_coefficient_difference": coefficient_difference, "training_backgrounds": training_backgrounds,
           "test_backgrounds": job["test_proteins"], "official_test_set_preserved": True,
           "training_only_normalization": True, "label_permissions_verified": True}
    records.append(row)
    print(json.dumps(row), flush=True)
assert records
output = root / "verification"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_json(output / "checks.json", orient="records", indent=2)
result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "verified_fits": len(records), "replayed_genotypes": sum(row["replayed_genotypes"] for row in records),
          "maximum_prediction_difference": max(row["maximum_prediction_difference"] for row in records),
          "all_permitted_training_backgrounds": True, "official_test_sets_preserved": True,
          "ridge_reference": "Independent scikit-learn Ridge fit on actual permitted training labels.",
          "mlp_reference": "Independent CPU checkpoint replay with training-only normalization.",
          "quartet_reference": "Original FLIP measurements and model predictions of each quartet genotype."}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
