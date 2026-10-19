import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from esm_msr.training import ESM3EpistasisLightningModule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.msr import FEATURES, QuartetEvaluation, evaluation_precision, make_loader, offline_model_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--partition", choices=["validation", "test"], required=True)
    parser.add_argument("--verify-smoke", action="store_true")
    args = parser.parse_args()
    directory = (ROOT / args.run).resolve()
    assert directory.is_relative_to(ROOT)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"] and (not manifest["smoke"] or (args.verify_smoke and args.partition == "validation"))
    configuration = json.loads((directory / "config.json").read_text())["author"]
    permissions = json.loads((directory / "permissions.json").read_text())
    output = directory / ("evaluation_" + args.partition)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists()
    metadata = pd.read_parquet(FEATURES / "variants.parquet")
    evaluator = QuartetEvaluation(metadata, permissions[args.partition], smoke=manifest["smoke"])
    logging.getLogger().setLevel(logging.WARNING)
    loader = make_loader(evaluator.inputs, False, batch_size=16)
    offline_model_directory()
    module = ESM3EpistasisLightningModule(**configuration).cuda().eval()
    state = torch.load(directory / "best.ckpt", map_location="cpu", weights_only=True)
    missing, unexpected = module.load_state_dict(state["state_dict"], strict=False)
    assert not unexpected
    assert not [name for name in missing if "lora" in name.lower() or "calibration" in name.lower()]
    identifiers, predictions = [], []
    assert configuration["evaluation_precision"] == "fp32"
    with torch.inference_mode(), evaluation_precision():
        for batch in loader:
            identifiers.extend(batch["variant_id"])
            batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            predictions.extend(module.model.forward_batch(batch, mask_strategy=configuration["mask_strategy"])[
                "combined_pred"].cpu().float().numpy().tolist())
    score, table, metrics = evaluator.evaluate(pd.Series(predictions, index=identifiers))
    replay_difference = None
    if args.partition == "validation":
        previous = pd.read_parquet(directory / "validation_predictions.parquet").set_index("variant_id")
        current = table.set_index("variant_id").reindex(previous.index)
        np.testing.assert_allclose(current.prediction_mu, previous.prediction_mu, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(current.prediction_epsilon, previous.prediction_epsilon, rtol=1e-5, atol=1e-5)
        replay_difference = float(np.max(np.abs(current.prediction_epsilon - previous.prediction_epsilon)))
    table["mode"], table["condition"] = "Z", "Z"
    metrics["mode"], metrics["condition"] = "Z", "Z"
    table.to_parquet(output / "predictions.parquet", index=False)
    metrics.to_parquet(output / "per_protein.parquet", index=False)
    pd.DataFrame({"variant_id": identifiers, "native_prediction": predictions}).to_parquet(
        output / "all_genotype_predictions.parquet", index=False)
    result = {"complete": True, "smoke": manifest["smoke"], "partition": args.partition, "mode": "Z",
              "training_manifest": str((directory / "manifest.json").relative_to(ROOT)),
              "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "variants": len(table), "proteins": table.protein_id.nunique(), "target_labels_used": 0,
              "macro_epsilon_rmse": score, "checkpoint_validation_replay_difference": replay_difference,
              "training_precision": configuration["precision"], "evaluation_precision": "fp32",
              "both_branches": True,
              "four_model_predictions": True, "label_permissions_verified": True}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
