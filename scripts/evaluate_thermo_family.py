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
from epistasis.thermo_training import DoubleEvaluation, FEATURES, ThermoCache, new_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    directory = (ROOT / args.run).resolve()
    assert directory.is_relative_to(ROOT)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"] and not manifest["smoke"]
    config = manifest["config"]
    permissions = json.loads((directory / "permissions.json").read_text())
    output = directory / ("evaluation_" + args.partition)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists()
    model = new_head(config["mode"], config["seed"])
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    cache = ThermoCache(FEATURES / "direct", config["mode"])
    evaluator = DoubleEvaluation(cache, permissions[args.partition])
    score, phenotype_mse, predictions, metrics = evaluator.evaluate(model)
    predictions["mode"], predictions["condition"] = "Z", "Z"
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics["mode"], metrics["condition"] = "Z", "Z"
    metrics.to_parquet(output / "per_protein.parquet", index=False)
    families = pd.read_parquet(ROOT / "data/splits" / f"families_{config['threshold']}.parquet")
    assert set(predictions.protein_id) <= set(permissions[args.partition])
    assert not set(predictions.protein_id) & set(permissions["train"])
    assert np.isfinite(predictions.prediction_epsilon).all()
    record = {"complete": True, "smoke": False, "source_commit": subprocess.check_output(
                  ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "training_manifest": str((directory / "manifest.json").relative_to(ROOT)),
              "partition": args.partition, "mode": "Z", "conditions": ["Z"],
              "target_labels_used": 0, "variants": len(predictions), "proteins": predictions.protein_id.nunique(),
              "macro_epsilon_rmse": score, "phenotype_mse": phenotype_mse,
              "counterfactual_definition": "Each double, both component singles, and the empty genotype evaluated with the same trained head.",
              "sign": "Negative author destabilizing DDG; relative predictions subtract the same model's empty-genotype output."}
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
