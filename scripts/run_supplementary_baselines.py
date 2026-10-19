import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.supplementary import SupplementaryClassifier, SupplementaryRegression, save_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--world-size", type=int, default=8)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
assert 0 <= args.rank < args.world_size
torch.set_float32_matmul_precision("highest")
records = pd.read_parquet(ROOT / "data/processed/flip_supplementary/records.parquet")
jobs = []
for (dataset, split), group in records.groupby(["dataset", "official_split"], sort=True):
    equivalent = group.effective_set.eq(group.strict_set).all()
    for protocol, column in [("official", "effective_set")] + ([] if equivalent else [("strict", "strict_set")]):
        for method in (["mean_logistic", "mean_classifier_mlp"] if dataset == "nucb" else ["mean_ridge", "mean_mlp"]):
            for seed in ([11] if args.smoke else [11, 23, 47, 89, 131]):
                jobs.append({"dataset": dataset, "split": split, "protocol": protocol, "column": column,
                             "method": method, "seed": seed, "protocol_equivalence": ["official", "strict"] if equivalent else [protocol]})
root = ROOT / "runs" / ("flip_supplementary_smoke" if args.smoke else "flip_supplementary")
root.mkdir(parents=True, exist_ok=True)
commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
outputs = []
for job in jobs[args.rank::args.world_size]:
    directory = root / job["dataset"] / job["split"] / job["protocol"] / job["method"] / f"seed{job['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text())["complete"]
        outputs.append(str(path.relative_to(ROOT)))
        continue
    started = time.monotonic()
    constructor = SupplementaryClassifier if job["dataset"] == "nucb" else SupplementaryRegression
    model = constructor(job["dataset"], job["split"], job["method"], "cuda")
    model.partition_column = job["column"]
    metadata = model.metadata
    train = metadata.index[metadata[job["column"]].eq("train")].to_numpy(dtype=int)
    val = metadata.index[metadata[job["column"]].eq("validation")].to_numpy(dtype=int)
    test = metadata.index[metadata[job["column"]].eq("test")].to_numpy(dtype=int)
    full_training = len(train)
    if args.smoke:
        rng = np.random.default_rng(job["seed"])
        if job["dataset"] == "nucb":
            train = np.concatenate([rng.choice(group.index, max(1, int(np.ceil(len(group) / 100))), replace=False)
                                    for _, group in metadata.loc[train].groupby("target", sort=True)])
        else:
            train = rng.choice(train, max(2, int(np.ceil(len(train) / 100))), replace=False)
    assert min(len(train), len(val), len(test)) >= 2
    assert not set(metadata.loc[train, "variant_id"]) & set(metadata.loc[np.concatenate([val, test]), "variant_id"])
    assert not set(metadata.loc[val, "variant_id"]) & set(metadata.loc[test, "variant_id"])
    ledger = pd.concat([metadata.loc[rows].assign(permission=label) for label, rows in [("train", train), ("validation", val), ("test", test)]])
    ledger.to_parquet(directory / "permissions.parquet", index=False)
    result = model.fit(train, val, metadata.target.to_numpy(), job["seed"], directory, max_epochs=2 if args.smoke else 100)
    metrics = save_evaluation(model, directory)
    information = {"complete": True, "evaluated": True, "source_commit": commit, "smoke": args.smoke,
                   **job, **result, "metrics": metrics, "training_records": len(train), "original_training_records": full_training,
                   "validation_records": len(val), "test_records": len(test), "training_record_fraction": len(train) / full_training,
                   "feature_width": model.values.shape[1], "source_labels_used": 0, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                   "selection_metric": "Validation macro F1, then log loss." if job["dataset"] == "nucb" else "Validation Spearman, then RMSE.",
                   "input_representation": "Separate frozen ESM-2 chain means and measured sequence lengths." if job["dataset"] == "pdz3" else "Frozen ESM-2 full-sequence mean.",
                   "official_test_records_preserved": True, "elapsed_seconds": time.monotonic() - started}
    path.write_text(json.dumps(information, indent=2))
    outputs.append(str(path.relative_to(ROOT)))
    print(json.dumps({**job, "training_records": len(train), "test_records": len(test), "complete": True}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
(root / f"manifest_rank{args.rank}.json").write_text(json.dumps(
    {"complete": True, "smoke": args.smoke, "rank": args.rank, "world_size": args.world_size, "outputs": outputs}, indent=2))
