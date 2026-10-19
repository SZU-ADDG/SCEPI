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
from epistasis.mean_family import MeanFamilyCache, MeanFamilyEvaluation
from epistasis.mean_regression import MeanMLP


def cpu_predictions(cache, evaluator, state):
    values = (cache.batch(evaluator.rows).astype(np.float64) - state["x_center"]) / state["x_scale"]
    if state["method"] == "mean_ridge":
        predicted = values @ state["coefficient"].numpy()
    else:
        model = MeanMLP(state["width"]).eval()
        model.load_state_dict(state["model"])
        with torch.inference_mode():
            predicted = np.concatenate([model(torch.tensor(values[start:start + 512], dtype=torch.float32)).numpy()
                                        for start in range(0, len(values), 512)])
    predicted = predicted * state["y_scale"] + state["y_center"]
    parts = predicted[np.searchsorted(evaluator.rows, evaluator.components)]
    frame = pd.DataFrame(index=evaluator.table.variant_id)
    frame["prediction_mu"] = parts[:, 3] - parts[:, 0]
    frame["prediction_a"] = parts[:, 1] - parts[:, 0]
    frame["prediction_b"] = parts[:, 2] - parts[:, 0]
    frame["prediction_wt"] = 0.0
    frame["prediction_epsilon"] = parts[:, 3] - parts[:, 1] - parts[:, 2] + parts[:, 0]
    return frame


parser = argparse.ArgumentParser()
parser.add_argument("--run-root", default="runs/mean_family_target_smoke")
args = parser.parse_args()
root = (ROOT / args.run_root).resolve()
assert root.is_relative_to(ROOT)
paths = sorted(root.glob("family*/*/[SC]/*/seed*/*/*/manifest.json"))
assert paths
cache = MeanFamilyCache()
lookup = cache.table.set_index("variant_id").row
records = []
columns = ["prediction_mu", "prediction_a", "prediction_b", "prediction_wt", "prediction_epsilon"]
for path in paths:
    information = json.loads(path.read_text())
    assert information["complete"]
    if not information["evaluated"]:
        continue
    training = pd.read_parquet(path.parent / "training_variants.parquet")
    saved = pd.read_parquet(path.parent / "predictions.parquet").set_index("variant_id")
    source = ROOT / information["source_manifest"]
    source_training = pd.read_parquet(source.parent / "training_variants.parquet")
    permission = json.loads((source.parent / "permissions.json").read_text())
    family = pd.read_parquet(ROOT / "data/splits" / f"families_{information['threshold']}.parquet").set_index("protein_id")
    relatives = set(family.index[family.family_group.eq(family.loc[information["protein_id"], "family_group"])])
    assert relatives <= set(permission["test"])
    assert not relatives & (set(permission["train"]) | set(permission["validation"]))
    assert not set(training.variant_id) & (set(source_training.variant_id) | set(saved.index))
    assert set(source_training.protein_id) <= set(permission["train"])
    if information["mode"] == "S":
        assert training.n_mutations.le(1).all()
    state = torch.load(path.parent / "best.pt", map_location="cpu", weights_only=False)
    source_state = torch.load(source.parent / "best.pt", map_location="cpu", weights_only=False)
    evaluator = MeanFamilyEvaluation(cache, [information["protein_id"]])
    predicted = cpu_predictions(cache, evaluator, state).loc[saved.index]
    np.testing.assert_allclose(predicted[columns], saved[columns], rtol=1e-5, atol=1e-5)
    difference = float(np.max(np.abs(predicted[columns].to_numpy() - saved[columns].to_numpy())))
    initial = pd.read_parquet(path.parent / "initial_predictions.parquet").set_index("variant_id")
    independent_initial = cpu_predictions(cache, evaluator, source_state).loc[initial.index]
    np.testing.assert_allclose(independent_initial[columns], initial[columns], rtol=1e-5, atol=1e-5)
    coefficient_difference = None
    if state["method"] == "mean_ridge":
        combined = pd.concat([source_training, training], ignore_index=True)
        assert combined.variant_id.is_unique
        x = cache.batch(lookup.loc[combined.variant_id].to_numpy(dtype=int)).astype(np.float64)
        y = combined.point.to_numpy(dtype=np.float64)
        np.testing.assert_allclose(x.mean(0), state["x_center"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x.std(0), state["x_scale"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose([y.mean(), y.std()], [state["y_center"], state["y_scale"]], rtol=1e-8, atol=1e-8)
        reference = Ridge(alpha=state["ridge_alpha"], fit_intercept=False, solver="cholesky")
        reference.fit((x - state["x_center"]) / state["x_scale"], (y - state["y_center"]) / state["y_scale"])
        np.testing.assert_allclose(reference.coef_, state["coefficient"].numpy(), rtol=1e-5, atol=1e-5)
        coefficient_difference = float(np.max(np.abs(reference.coef_ - state["coefficient"].numpy())))
    records.append({"manifest": str(path.relative_to(ROOT)), "method": information["method"], "mode": information["mode"],
                    "replayed_variants": len(saved), "maximum_prediction_difference": difference,
                    "sklearn_coefficient_difference": coefficient_difference, "label_disjointness": True,
                    "source_family_exclusion": True, "source_initialization_verified": True})
    print(json.dumps(records[-1]), flush=True)
assert records
output = root / "verification"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(output / "checks.csv", index=False)
result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "verified_fits": len(records), "replayed_variants": sum(row["replayed_variants"] for row in records),
          "maximum_prediction_difference": max(row["maximum_prediction_difference"] for row in records),
          "methods": sorted({row["method"] for row in records}), "modes": sorted({row["mode"] for row in records}),
          "label_disjointness": True, "source_family_exclusion": True, "source_initialization_verified": True,
          "ridge_reference": "Independent scikit-learn fit on the concatenated source and permitted target labels.",
          "mlp_reference": "Independent CPU replay of source and adapted checkpoints."}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
