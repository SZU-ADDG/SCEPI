import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from esm_msr.training import ESM3EpistasisLightningModule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.msr import FEATURES, make_loader, offline_model_directory
from epistasis.msr_target import TargetQuartets

directory = ROOT / "runs/msr_target_batch_diagnostic/checkpoint_source"
output = ROOT / "runs/msr_target_batch_diagnostic"
output.mkdir(exist_ok=True)
configuration = json.loads((directory / "config.json").read_text())["author"]
metadata = pd.read_parquet(FEATURES / "variants.parquet")
doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
previous = pd.read_parquet(directory / "validation_predictions.parquet")
saved_native = pd.read_parquet(directory / "all_genotype_predictions.parquet").set_index("variant_id").native_prediction
evaluator = TargetQuartets(metadata, doubles.loc[doubles.variant_id.isin(previous.variant_id)])
all_inputs = metadata.loc[metadata.variant_id.isin(saved_native.index)]
logging.getLogger().setLevel(logging.WARNING)
offline_model_directory()
module = ESM3EpistasisLightningModule(**configuration).cuda().eval()
checkpoint = torch.load(directory / "best.ckpt", map_location="cpu", weights_only=True)
missing, unexpected = module.load_state_dict(checkpoint["state_dict"], strict=False)
assert not unexpected and not [name for name in missing if "lora" in name.lower() or "calibration" in name.lower()]
assert all(value.dtype == torch.float32 for value in module.parameters())
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
results, records = {}, []
for precision, table_name, batch_size in [
    ("bf16", "validation", 16), ("bf16", "all", 16), ("bf16", "validation", 8),
    ("fp32", "validation", 16), ("fp32", "all", 16), ("fp32", "validation", 8),
]:
    table = evaluator.inputs if table_name == "validation" else all_inputs
    rows = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        for batch_index, batch in enumerate(make_loader(table, False, batch_size=batch_size)):
            ids = batch["variant_id"]
            # 保留真实批次位置和原始分支输出，用于区分批次差异与检查点差异。
            batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            values = module.model.forward_batch(batch, mask_strategy=configuration["mask_strategy"])
            arrays = {name: value.float().cpu().numpy() for name, value in values.items()}
            for index, identifier in enumerate(ids):
                rows.append({"variant_id": identifier, "batch_index": batch_index, "batch_count": len(ids),
                             "maximum_mutations": batch["mut_pos"].shape[1],
                             **{name: float(value[index]) for name, value in arrays.items()}})
    key = f"{precision}_{table_name}_{batch_size}"
    result = pd.DataFrame(rows).set_index("variant_id")
    result.to_parquet(output / (key + ".parquet"))
    results[key] = result
    score, predicted, _ = evaluator.evaluate(result.combined_pred.loc[list(evaluator.ids)])
    native_difference = np.abs(result.combined_pred - saved_native.loc[result.index])
    row = {"condition": key, "genotypes": len(result), "score": score,
           "saved_native_maximum_difference": float(native_difference.max()),
           "saved_validation_mu_maximum_difference": float(np.max(np.abs(
               predicted.set_index("variant_id").prediction_mu - previous.set_index("variant_id").prediction_mu)))}
    records.append(row)
    print(json.dumps(row), flush=True)
comparisons = []
for precision in ["bf16", "fp32"]:
    first = results[f"{precision}_validation_16"]
    for other in [f"{precision}_all_16", f"{precision}_validation_8"]:
        second = results[other].loc[first.index]
        for column in ["wt_lora_raw", "mt_lora_raw", "combined_pred"]:
            difference = np.abs(first[column] - second[column])
            affected = first.loc[difference.gt(1e-5), ["batch_index", "batch_count", "maximum_mutations"]].copy()
            affected["difference"] = difference.loc[affected.index]
            affected.to_csv(output / f"{precision}_{other}_{column}_differences.csv")
            comparisons.append({"precision": precision, "other": other, "column": column,
                                "maximum_difference": float(difference.max()), "affected": len(affected),
                                "first_batch_sizes": sorted(affected.batch_count.unique().tolist())})
information = {"complete": True, "conditions": records, "comparisons": comparisons,
               "checkpoint": str((directory / "best.ckpt").relative_to(ROOT)),
               "genotypes_are_measured_or_model_counterfactual": True,
               "tf32_enabled": torch.backends.cuda.matmul.allow_tf32}
(output / "manifest.json").write_text(json.dumps(information, indent=2))
print(json.dumps(information, indent=2))
