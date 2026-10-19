import argparse
import gc
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
from epistasis.msr import DirectDataset, FEATURES, make_loader, offline_model_directory

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", required=True)
args = parser.parse_args()
root = (ROOT / args.run_root).resolve()
assert root.is_relative_to(ROOT)
paths = sorted(root.glob("family*/*/*/seed*/*/*/manifest.json"))
assert paths
metadata = pd.read_parquet(FEATURES / "variants.parquet")
lookup = metadata.set_index("variant_id")
variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet").set_index("variant_id")
logging.getLogger().setLevel(logging.WARNING)
module, records = None, []
for path in paths:
    info = json.loads(path.read_text())
    assert info["complete"]
    if not info["evaluated"]:
        continue
    permissions = pd.read_parquet(path.parent / "permissions.parquet")
    assert permissions.variant_id.is_unique
    allowed = set(permissions.loc[permissions.partition.eq("train"), "variant_id"])
    evaluated = set(permissions.loc[permissions.partition.ne("train"), "variant_id"])
    assert not allowed & evaluated
    training = pd.read_parquet(path.parent / "training_variants.parquet")
    assert set(training.variant_id) <= allowed and training.training_point_eligible.all()
    reference = pd.read_parquet(path.parent / "training_reference.parquet")
    assert len(reference) == 1 and set(reference.variant_id) <= allowed and reference.point_in_range.all()
    np.testing.assert_allclose(training.native_ddg, reference.point.iloc[0] - variants.loc[training.variant_id].point.to_numpy(), rtol=1e-6, atol=1e-6)
    sampled = pd.read_parquet(path.parent / "sampled_training_variants.parquet")
    assert set(sampled.variant_id) <= set(training.variant_id)
    assert len(sampled) == info["sampled_training_variants"] and len(sampled)
    if info["mode"] == "S":
        assert training.n_mutations.eq(1).all()
    data = DirectDataset(training, set(training.variant_id))
    for row, item in zip(training.itertuples(), data.data):
        if row.n_mutations == 2:
            for key, identifier in [("ddG_A", row.single_a_id), ("ddG_B", row.single_b_id)]:
                if identifier not in set(training.variant_id):
                    assert np.isnan(item[key])
    source = ROOT / info["source_manifest"]
    source_info = json.loads(source.read_text())
    source_permissions = json.loads((source.parent / "permissions.json").read_text())
    families = pd.read_parquet(ROOT / "data/splits" / f"families_{info['threshold']}.parquet").set_index("protein_id")
    relatives = set(families.index[families.family_group.eq(families.loc[info["protein_id"], "family_group"])])
    assert relatives <= set(source_permissions["test"])
    assert not relatives & (set(source_permissions["train"]) | set(source_permissions["validation"]))
    if info["validation_score"] is None:
        assert info["epochs"] == min(info["max_epochs"], source_info["best_epoch"] + 1)
    assert any("wt_adapter" in name for name in info["nonzero_gradient_tensors"])
    assert any("mt_adapter" in name for name in info["nonzero_gradient_tensors"])
    configuration = json.loads((path.parent / "config.json").read_text())["author"]
    original = json.loads((source.parent / "config.json").read_text())["author"]
    for key in ["learning_rate", "weight_decay", "lambda_rank_wt", "lambda_rank_combined", "lambda_reg_wt",
                "lambda_reg_combined", "lambda_epi_combined", "rank_loss", "reg_loss", "lr_warmup_steps", "lora_mode"]:
        assert configuration[key] == original[key]
    offline_model_directory()
    if module is None:
        module = ESM3EpistasisLightningModule(**configuration).cuda().eval()
    state = torch.load(path.parent / "best.ckpt", map_location="cpu", weights_only=True)
    missing, unexpected = module.load_state_dict(state["state_dict"], strict=False)
    assert not unexpected
    assert not [name for name in missing if "lora" in name.lower() or "calibration" in name.lower()]
    previous = pd.read_parquet(path.parent / "all_genotype_predictions.parquet").set_index("variant_id")
    inputs = metadata.loc[metadata.variant_id.isin(previous.index)]
    assert set(inputs.variant_id) == set(previous.index)
    loader = make_loader(inputs, False, batch_size=8)
    predictions, identifiers = [], []
    assert configuration["evaluation_precision"] == "fp32" and info["evaluation_precision"] == "fp32"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32
    with torch.inference_mode():
        for batch in loader:
            identifiers.extend(batch["variant_id"])
            batch = {name: value.cuda() if torch.is_tensor(value) else value for name, value in batch.items()}
            values = module.model.forward_batch(batch, mask_strategy=configuration["mask_strategy"])["combined_pred"]
            predictions.extend(values.float().cpu().numpy().tolist())
    native = pd.Series(predictions, index=identifiers)
    np.testing.assert_allclose(native.loc[previous.index], previous.native_prediction, rtol=1e-5, atol=1e-5)
    saved = pd.read_parquet(path.parent / "predictions.parquet")
    assert set(saved.variant_id) == evaluated
    ids = lookup.loc[saved.variant_id]
    wt = native.loc[("msr_reference_" + saved.protein_id).to_numpy()].to_numpy()
    predicted_mu = -native.loc[saved.variant_id].to_numpy() + wt
    predicted_a = -native.loc[ids.single_a_id].to_numpy() + wt
    predicted_b = -native.loc[ids.single_b_id].to_numpy() + wt
    expected = np.column_stack([predicted_mu, predicted_a, predicted_b, predicted_mu - predicted_a - predicted_b])
    observed = saved[["prediction_mu", "prediction_a", "prediction_b", "prediction_epsilon"]].to_numpy()
    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)
    assert saved.prediction_wt.eq(0).all()
    row = {"manifest": str(path.relative_to(ROOT)), "mode": info["mode"], "replayed_genotypes": len(native),
           "evaluated_variants": len(saved), "maximum_native_difference": float(np.max(np.abs(native.loc[previous.index] - previous.native_prediction))),
           "maximum_prediction_difference": float(np.max(np.abs(observed - expected))),
           "family_exclusion": True, "target_label_permissions": True, "both_branch_gradients": True,
           "source_selected_settings": True, "original_model_replay": True}
    records.append(row)
    print(json.dumps(row), flush=True)
    del data, loader, state
    gc.collect()
    torch.cuda.empty_cache()
assert records
output = root / "verification"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(output / "checks.csv", index=False)
result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "verified_fits": len(records), "modes": sorted({row["mode"] for row in records}),
          "replayed_genotypes": sum(row["replayed_genotypes"] for row in records),
          "maximum_prediction_difference": max(row["maximum_prediction_difference"] for row in records),
          "original_model_replay": "Original author model, saved adapted checkpoint, independent batches of eight.",
          "four_model_predictions": True, "family_exclusion_and_label_permissions": True}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
