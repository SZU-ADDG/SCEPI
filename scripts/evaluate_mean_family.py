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
from epistasis.mean_family import MeanFamilyCache, MeanFamilyEvaluation, predict
from epistasis.mean_regression import MeanMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--partition", choices=["validation", "test"], required=True)
    args = parser.parse_args()
    directory = (ROOT / args.run).resolve()
    assert directory.is_relative_to(ROOT)
    information = json.loads((directory / "manifest.json").read_text())
    assert information["complete"] and not information["smoke"]
    output = directory / ("evaluation_" + args.partition)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists()
    torch.set_float32_matmul_precision("highest")
    permissions = json.loads((directory / "permissions.json").read_text())
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model = None
    if state["method"] == "mean_mlp":
        model = MeanMLP(state["width"]).cuda().eval()
        model.load_state_dict(state["model"])
    cache = MeanFamilyCache()
    evaluator = MeanFamilyEvaluation(cache, permissions[args.partition])
    values = predict(cache, evaluator.rows, state, model)
    score, table, metrics = evaluator.evaluate(values)
    replay_difference = None
    if args.partition == "validation":
        original = pd.read_parquet(directory / "validation_predictions.parquet").set_index("variant_id")
        current = table.set_index("variant_id").reindex(original.index)
        np.testing.assert_allclose(current.prediction_epsilon, original.prediction_epsilon, rtol=1e-5, atol=1e-5)
        replay_difference = float(np.max(np.abs(current.prediction_epsilon - original.prediction_epsilon)))
    table["mode"], table["condition"] = "Z", "Z"
    metrics["mode"], metrics["condition"] = "Z", "Z"
    table.to_parquet(output / "predictions.parquet", index=False)
    metrics.to_parquet(output / "per_protein.parquet", index=False)
    cache.table.iloc[evaluator.rows][["variant_id", "protein_id"]].assign(prediction=values).to_parquet(
        output / "all_genotype_predictions.parquet", index=False)
    manifest = {"complete": True, "smoke": False, "method": state["method"],
                "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "training_manifest": str((directory / "manifest.json").relative_to(ROOT)),
                "partition": args.partition, "mode": "Z", "target_labels_used": 0,
                "variants": len(table), "proteins": table.protein_id.nunique(), "macro_epsilon_rmse": score,
                "checkpoint_validation_replay_difference": replay_difference, "four_model_predictions": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
