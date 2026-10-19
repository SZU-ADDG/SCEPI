import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.higher import HigherFeatures
from epistasis.mean_regression import MeanMLP, MeanRegression
from run_mochi_targets import save_results

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", required=True)
args = parser.parse_args()
root = (ROOT / args.run_root).resolve()
assert root.is_relative_to(ROOT)
commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
for path in sorted(root.glob("*/order*/*/seed*/*/*/manifest.json")):
    information = json.loads(path.read_text())
    assert information["complete"]
    if not information["evaluated"]:
        continue
    method = information["method"]
    assert method in ["mean_mlp", "mean_ridge"]
    features = HigherFeatures(ROOT / "data/features/higher" / information["dataset"] / information["protein_id"])
    regression = MeanRegression(features, information["dataset"], method, "cuda")
    state = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
    x = (regression.values.astype(np.float64) - state["x_center"]) / state["x_scale"]
    if method == "mean_mlp":
        model = MeanMLP(state["selected"]["width"]).cuda().eval()
        model.load_state_dict(state["model"])
    else:
        coefficient = state["coefficient"].cuda()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            current = torch.tensor(x[start:start + 4096], dtype=torch.float32 if method == "mean_mlp" else torch.float64, device="cuda")
            values = model(current) if method == "mean_mlp" else current @ coefficient
            predictions.append(values.cpu().numpy() * state["y_scale"] + state["y_center"])
    predictions = np.concatenate(predictions)[regression.prediction_alias]
    assert np.isfinite(predictions).all()
    previous = pd.read_parquet(path.parent / "predictions.parquet")
    job = {"validation_ids": previous.loc[previous["set"].eq("validation"), "variant_id"].tolist(),
           "test_ids": previous.loc[previous["set"].eq("test"), "variant_id"].tolist()}
    np.savez_compressed(path.parent / "predictions.npz", variants=predictions[:regression.count],
                        reference=predictions[regression.count], singles=predictions[regression.count + 1:])
    information["metrics"] = save_results(features, job, path.parent, information["dataset"])
    information["prediction_commit"] = commit
    information["duplicate_genotypes"] = "Identical full-genotype embeddings share a single prediction for all observed and counterfactual aliases."
    path.write_text(json.dumps(information, indent=2))
    print(json.dumps({"complete": True, "manifest": str(path.relative_to(ROOT)), "genotypes": len(predictions)}), flush=True)
