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
from epistasis.higher import HigherFeatures
from epistasis.mean_regression import MeanMLP, MeanRegression

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", required=True)
args = parser.parse_args()
root = (ROOT / args.run_root).resolve()
assert root.is_relative_to(ROOT)
records = []
for path in sorted(root.glob("*/order*/*/seed*/*/*/manifest.json")):
    information = json.loads(path.read_text())
    assert information["complete"]
    if not information["evaluated"]:
        continue
    features = HigherFeatures(ROOT / "data/features/higher" / information["dataset"] / information["protein_id"])
    method = information["method"]
    regression = MeanRegression(features, information["dataset"], method, "cpu")
    state = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
    train, validation = state["train_indices"], state["validation_indices"]
    assert not set(train) & set(validation)
    x = (regression.values.astype(np.float64) - state["x_center"]) / state["x_scale"]
    rows = np.unique(np.linspace(0, len(x) - 1, min(512, len(x)), dtype=int))
    if method == "mean_ridge":
        y = (features.metadata.point.to_numpy()[train] - state["y_center"]) / state["y_scale"]
        reference = Ridge(alpha=state["selected"]["ridge_alpha"], fit_intercept=False, solver="cholesky")
        reference.fit(x[train], y)
        np.testing.assert_allclose(reference.coef_, state["coefficient"].numpy(), rtol=1e-5, atol=1e-5)
        normalized = reference.predict(x[rows])
        coefficient_difference = float(np.max(np.abs(reference.coef_ - state["coefficient"].numpy())))
    else:
        model = MeanMLP(state["selected"]["width"]).eval()
        model.load_state_dict(state["model"])
        with torch.inference_mode():
            normalized = model(torch.tensor(x[rows], dtype=torch.float32)).numpy()
        coefficient_difference = None
    saved = np.load(path.parent / "predictions.npz")
    observed = np.concatenate([saved["variants"], saved["reference"].reshape(1), saved["singles"]])
    observed = (observed[rows] - state["y_center"]) / state["y_scale"]
    np.testing.assert_allclose(observed, normalized, rtol=1e-5, atol=1e-5)
    records.append({"manifest": str(path.relative_to(ROOT)), "method": method, "replayed_genotypes": len(rows),
                    "maximum_normalized_prediction_difference": float(np.max(np.abs(observed - normalized))),
                    "sklearn_coefficient_difference": coefficient_difference,
                    "training_validation_disjoint": True})
assert records
output = root / "implementation_verification"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(output / "checks.csv", index=False)
result = {"complete": True, "source_commit": subprocess.check_output(
              ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "verified_fits": len(records), "replayed_genotypes": sum(row["replayed_genotypes"] for row in records),
          "maximum_normalized_prediction_difference": max(row["maximum_normalized_prediction_difference"] for row in records),
          "ridge_reference": "scikit-learn Ridge, same training-only normalization and selected regularization.",
          "mlp_reference": "Independent CPU replay of the saved model and preprocessing."}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
