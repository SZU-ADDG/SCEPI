import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.mean_flip import MeanFlipRegression, jobs_for, save_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mean_ridge", "mean_mlp"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert 0 <= args.rank < args.world_size
    torch.set_float32_matmul_precision("highest")
    jobs = jobs_for()
    variants = pd.read_parquet(ROOT / "data/processed/flip_primary/variants.parquet")
    if args.smoke:
        jobs = [job for job in jobs if job["scheme"] == "hydro_to_P06241" and job["condition"] == "official" and job["seed"] == 11]
        assert len(jobs) == 1
        job = jobs[0]
        original = variants.loc[variants.variant_id.isin(job["train_ids"])]
        rng = np.random.default_rng(job["seed"])
        chosen = []
        for protein, group in original.groupby("protein_id", sort=True):
            chosen.extend(rng.choice(group.variant_id.to_numpy(), size=max(2, int(np.ceil(len(group) / 100))), replace=False).tolist())
        job["original_training_variants"] = len(job["train_ids"])
        job["train_ids"] = chosen
    output = ROOT / "runs" / ("mean_flip_smoke" if args.smoke else "mean_flip") / args.method
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifests, regression, cached_proteins = [], None, None
    for job in jobs[args.rank::args.world_size]:
        directory = output / job["scheme"] / job["condition"] / f"seed{job['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        if path.exists():
            assert json.loads(path.read_text())["complete"]
            manifests.append(str(path.relative_to(ROOT)))
            continue
        started = time.monotonic()
        relevant = set(job["train_ids"]) | set(job["validation_ids"]) | set(job["test_ids"])
        metadata = variants.loc[variants.variant_id.isin(relevant)]
        assert set(metadata.variant_id) == relevant
        proteins = tuple(sorted(metadata.protein_id.unique()))
        information = {"complete": True, "source_commit": commit, "dataset": "flip_primary",
                       "method": args.method, "smoke": args.smoke,
                       **{key: value for key, value in job.items() if not key.endswith("_ids")}}
        pd.DataFrame([{"variant_id": identifier, "partition": label} for label in ["train", "validation", "test"]
                      for identifier in job[label + "_ids"]]).to_parquet(directory / "permissions.parquet", index=False)
        training = metadata.loc[metadata.variant_id.isin(job["train_ids"])]
        validation = metadata.loc[metadata.variant_id.isin(job["validation_ids"])]
        assert training.point.notna().all() and validation.point.notna().all()
        if len(training) < 2 or len(validation) < 2 or training.point.std(ddof=0) == 0:
            information.update({"evaluated": False, "reason": "permitted_training_or_validation_unavailable"})
        else:
            if cached_proteins != proteins:
                del regression
                gc.collect()
                regression = MeanFlipRegression(proteins, args.method, "cuda")
                cached_proteins = proteins
            lookup = pd.Series(np.arange(len(regression.metadata)), index=regression.metadata.variant_id)
            train = lookup.loc[job["train_ids"]].to_numpy(dtype=int)
            val = lookup.loc[job["validation_ids"]].to_numpy(dtype=int)
            result = regression.fit(train, val, regression.metadata.point.to_numpy(), job["seed"], directory,
                                    max_epochs=2 if args.smoke else 100)
            evaluation = save_evaluation(regression, job, directory)
            information.update({"evaluated": True, **result, "metrics": evaluation,
                                "selection_metric": "Macro-average validation Spearman across proteins; macro RMSE breaks ties.",
                                "official_training_set_preserved": not args.smoke,
                                "training_variants": len(train), "validation_variants": len(val),
                                "test_variants": len(job["test_ids"]), "source_labels_used": 0,
                                "max_epochs": 2 if args.smoke else 100,
                                "counterfactual_backgrounds": sorted(regression.counterfactuals.protein_id.unique())})
        information["elapsed_seconds"] = time.monotonic() - started
        path.write_text(json.dumps(information, indent=2))
        manifests.append(str(path.relative_to(ROOT)))
        print(json.dumps({"method": args.method, "scheme": job["scheme"], "condition": job["condition"],
                          "seed": job["seed"], "evaluated": information["evaluated"]}), flush=True)
    (output / f"manifest_rank{args.rank}.json").write_text(json.dumps(
        {"complete": True, "rank": args.rank, "world_size": args.world_size, "outputs": manifests}, indent=2))


if __name__ == "__main__":
    main()
