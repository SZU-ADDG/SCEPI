import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
root = ROOT / "runs" / ("flip_supplementary_smoke" if args.smoke else "flip_supplementary")
for rank in range(8):
    assert json.loads((root / f"manifest_rank{rank}.json").read_text())["complete"]
paths = sorted(root.glob("*/*/*/*/seed*/manifest.json"))
assert len(paths) == (12 if args.smoke else 60)
records = pd.read_parquet(ROOT / "data/processed/flip_supplementary/records.parquet")
results = []
for path in paths:
    information = json.loads(path.read_text())
    assert information["complete"] and information["smoke"] == args.smoke
    directory = path.parent
    state = torch.load(directory / "model.pt", map_location="cpu")
    table = records.loc[records.dataset.eq(information["dataset"]) & records.official_split.eq(information["split"])].sort_values("source_row").reset_index(drop=True)
    features = ROOT / "data/features/mean/flip_supplementary"
    variants = pd.read_parquet(features / (information["dataset"] + "_metadata.parquet"))
    matrix = np.load(features / (information["dataset"] + "_mean.npy"), mmap_mode="r")
    index = variants.set_index("variant_id").matrix_row
    extra = variants.loc[~variants.variant_id.isin(table.variant_id)]
    ids = pd.concat([table.variant_id, extra.variant_id], ignore_index=True)
    values = matrix[ids.map(index).to_numpy(dtype=int)].astype(np.float64)
    train, validation = np.asarray(state["train_indices"]), np.asarray(state["validation_indices"])
    column = information["column"]
    assert table.loc[train, column].eq("train").all()
    np.testing.assert_array_equal(np.sort(validation), table.index[table[column].eq("validation")])
    if not args.smoke:
        np.testing.assert_array_equal(np.sort(train), table.index[table[column].eq("train")])
    assert not set(table.loc[train, "variant_id"]) & set(table.loc[table[column].isin(["validation", "test"]), "variant_id"])
    ledger = pd.read_parquet(directory / "permissions.parquet")
    assert ledger.record_id.is_unique
    for label, indices in [("train", train), ("validation", validation), ("test", table.index[table[column].eq("test")])]:
        assert set(ledger.loc[ledger.permission.eq(label), "record_id"]) == set(table.loc[indices, "record_id"])
    center, scale = values[train].mean(axis=0), values[train].std(axis=0)
    scale[scale == 0] = 1.
    np.testing.assert_array_equal(center, state["x_center"])
    np.testing.assert_array_equal(scale, state["x_scale"])
    normalized = (values - center) / scale
    method = information["method"]
    if method == "mean_logistic":
        probability = np.zeros((len(values), 4), dtype=np.float64)
        probability[:, state["classes"]] = softmax(normalized @ state["coefficient"].T + state["intercept"], axis=1)
        output = probability
    elif method == "mean_ridge":
        y_center, y_scale = table.target.iloc[train].mean(), table.target.iloc[train].std(ddof=0)
        np.testing.assert_allclose([y_center, y_scale], [state["y_center"], state["y_scale"]], rtol=0, atol=1e-12)
        alpha = state["selected"]["ridge_alpha"]
        reference = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky").fit(normalized[train], (table.target.iloc[train] - y_center) / y_scale)
        np.testing.assert_allclose(reference.coef_, state["coefficient"].numpy(), atol=1e-7, rtol=1e-6)
        output = normalized @ state["coefficient"].numpy() * y_scale + y_center
    else:
        parameters = state["model"]
        outputs = []
        for start in range(0, len(normalized), 257):
            hidden = torch.tensor(normalized[start:start + 257], dtype=torch.float32)
            with torch.inference_mode():
                hidden = F.gelu(F.linear(hidden, parameters["layers.0.weight"], parameters["layers.0.bias"]))
                hidden = F.gelu(F.linear(hidden, parameters["layers.2.weight"], parameters["layers.2.bias"]))
                hidden = F.linear(hidden, parameters["layers.4.weight"], parameters["layers.4.bias"])
                if method == "mean_classifier_mlp":
                    hidden = hidden.softmax(-1)
            outputs.append(hidden.numpy())
        output = np.concatenate(outputs)
        if method == "mean_mlp":
            output = output[:, 0] * state["y_scale"] + state["y_center"]
    first = pd.Series(np.arange(len(ids)), index=ids).groupby(level=0, sort=False).first()
    output = output[ids.map(first).to_numpy(dtype=int)]
    saved = pd.read_parquet(directory / "all_record_predictions.parquet")
    np.testing.assert_array_equal(saved.record_id, table.record_id)
    np.testing.assert_array_equal(saved.target, table.target)
    if information["dataset"] == "nucb":
        observed = saved[[f"probability_{label}" for label in range(4)]].to_numpy()
        np.testing.assert_allclose(output[:len(table)], observed, atol=1e-5, rtol=1e-6)
        np.testing.assert_array_equal(observed.argmax(axis=1), saved.prediction_class)
        np.testing.assert_array_equal(state["training_classes"], np.unique(table.target.iloc[train].to_numpy(dtype=int)))
        metrics = pd.read_csv(directory / "metrics.csv")
        for label in ["validation", "test"]:
            chosen = saved.loc[saved[column].eq(label)]
            f1 = f1_score(chosen.target, chosen.prediction_class, labels=np.arange(4), average="macro", zero_division=0)
            logged = metrics.loc[metrics.partition.eq(label) & metrics.mutation_order.eq("all"), "macro_f1"].item()
            np.testing.assert_allclose(f1, logged, atol=1e-14, rtol=0)
    else:
        observed = saved.prediction_absolute.to_numpy()
        tolerance = 1e-5 * max(1., float(state["y_scale"]))
        np.testing.assert_allclose(output[:len(table)], observed, atol=tolerance, rtol=1e-6)
        other = pd.read_parquet(directory / "counterfactual_predictions.parquet")
        np.testing.assert_array_equal(other.variant_id, extra.variant_id)
        np.testing.assert_allclose(output[len(table):], other.prediction_absolute, atol=tolerance, rtol=1e-6)
        if information["dataset"] == "pdz3":
            quartets = pd.read_parquet(directory / "quartet_predictions.parquet")
            predictions = pd.Series(output, index=ids)
            epsilon = (quartets.variant_id.map(predictions) - quartets.a_id.map(predictions)
                       - quartets.b_id.map(predictions) + quartets.wt_id.map(predictions))
            np.testing.assert_allclose(epsilon, quartets.prediction_epsilon, atol=4 * tolerance, rtol=1e-6)
    evaluated = pd.read_parquet(directory / "predictions.parquet")
    assert set(evaluated.loc[evaluated.partition.eq("test"), "record_id"]) == set(table.loc[table.effective_set.eq("test"), "record_id"])
    difference = float(np.max(np.abs(output[:len(table)] - observed)))
    results.append({"manifest": str(path.relative_to(ROOT)), "dataset": information["dataset"], "method": method,
                    "split": information["split"], "protocol": information["protocol"], "records": len(table),
                    "maximum_prediction_difference": difference, "training_and_validation_permissions_verified": True,
                    "all_original_test_records_verified": True, "independent_cpu_prediction_replay": True})
    print(json.dumps(results[-1]), flush=True)
destination = root / "verification"
destination.mkdir(exist_ok=True)
result = {"complete": True, "smoke": args.smoke, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "verified_fits": len(results), "replayed_records": sum(record["records"] for record in results),
          "maximum_prediction_difference": max(record["maximum_prediction_difference"] for record in results),
          "datasets": sorted({record["dataset"] for record in results}), "results": results}
(destination / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
