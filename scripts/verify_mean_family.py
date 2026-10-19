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
from epistasis.mean_family import MeanFamilyCache, MeanFamilyEvaluation, predict
from epistasis.mean_regression import MeanMLP

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
args = parser.parse_args()
directory = (ROOT / args.run).resolve()
assert directory.is_relative_to(ROOT)
information = json.loads((directory / "manifest.json").read_text())
assert information["complete"]
permissions = json.loads((directory / "permissions.json").read_text())
state = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
training = pd.read_parquet(directory / "training_variants.parquet")
validation = pd.read_parquet(directory / "validation_variants.parquet")
assert set(training.protein_id) <= set(permissions["train"])
assert set(validation.protein_id) <= set(permissions["validation"])
assert not set(training.variant_id) & set(validation.variant_id)
cache = MeanFamilyCache()
index = cache.table.set_index("variant_id").row
rows = index.reindex(training.variant_id).to_numpy(dtype=int)
x = (cache.batch(rows).astype(np.float64) - state["x_center"]) / state["x_scale"]
y = (training.point.to_numpy() - state["y_center"]) / state["y_scale"]
np.testing.assert_allclose(np.mean(x, axis=0), 0, atol=1e-9)
np.testing.assert_allclose(np.std(x, axis=0), 1, atol=1e-8)
model = None
coefficient_difference = None
if state["method"] == "mean_ridge":
    reference = Ridge(alpha=state["ridge_alpha"], fit_intercept=False, solver="cholesky").fit(x, y)
    np.testing.assert_allclose(reference.coef_, state["coefficient"].numpy(), rtol=1e-5, atol=1e-5)
    coefficient_difference = float(np.max(np.abs(reference.coef_ - state["coefficient"].numpy())))
else:
    model = MeanMLP(state["width"]).eval()
    model.load_state_dict(state["model"])
evaluator = MeanFamilyEvaluation(cache, permissions["validation"], information["smoke"])
values = predict(cache, evaluator.rows, state, model, device="cpu")
score, prediction, metrics = evaluator.evaluate(values)
saved = pd.read_parquet(directory / "validation_predictions.parquet").set_index("variant_id")
current = prediction.set_index("variant_id").reindex(saved.index)
difference = float(np.max(np.abs(current.prediction_epsilon - saved.prediction_epsilon)))
np.testing.assert_allclose(current.prediction_epsilon, saved.prediction_epsilon, rtol=1e-5, atol=1e-5)
manifest = {"complete": True, "method": state["method"], "training_variants": len(training),
            "validation_quartets": len(prediction), "cpu_replay_maximum_epsilon_difference": difference,
            "sklearn_coefficient_difference": coefficient_difference, "training_validation_disjoint": True,
            "training_only_normalization": True,
            "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
(directory / "verification.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
