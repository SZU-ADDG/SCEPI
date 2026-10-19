import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from epistasis.mean_family import MeanFamilyCache
from epistasis.mean_target import fit_target
from epistasis.metrics import per_protein_metrics
from run_mochi_targets import tasks_for


def selected_source(method, threshold, fold, seed):
    selected = []
    key = f"family{threshold}/fold{fold}"
    for path in (ROOT / "runs/mean_family").glob(f"repeats_{method}_*/selected_hyperparameters.json"):
        records = json.loads(path.read_text())
        if key in records:
            selected.append(records[key]["run"])
    assert selected and len(set(selected)) == 1, (method, key, selected)
    source = ROOT / selected[0]
    return source.with_name(source.name.rsplit("_seed", 1)[0] + f"_seed{seed}")


def single_tasks(variants, doubles, seeds):
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    jobs = []
    for protein in sorted(doubles.protein_id.unique()):
        reference = variants.loc[variants.protein_id.eq(protein) & variants.n_mutations.eq(0)]
        assert len(reference) == 1
        for seed in seeds:
            available = order.loc[order.protein_id.eq(protein) & order.seed.eq(seed)].sort_values("rank")
            for condition, budget in [("S32", 32), ("S128", 128), ("Sall", -1)]:
                selected = available if budget < 0 else available.iloc[:budget]
                jobs.append({"protein_id": protein, "seed": seed, "scheme": "family", "condition": condition,
                             "train_ids": reference.variant_id.tolist() + selected.variant_id.tolist(), "validation_ids": [],
                             "test_ids": doubles.loc[doubles.protein_id.eq(protein), "variant_id"].tolist(),
                             "target_single_labels": len(selected), "target_combination_labels": 0,
                             "budget_eligible": budget < 0 or len(available) >= budget})
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mean_ridge", "mean_mlp"], required=True)
    parser.add_argument("--mode", choices=["S", "C"], required=True)
    parser.add_argument("--threshold", type=int, choices=[30, 25], default=30)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--source-run")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.source_run is None or args.smoke
    assert args.mode != "C" or args.threshold == 30
    torch.set_float32_matmul_precision("highest")
    family = pd.read_parquet(ROOT / "data/splits" / f"families_{args.threshold}.parquet").set_index("protein_id")
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet")
    doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
    jobs = tasks_for("tsuboyama", "C") if args.mode == "C" else single_tasks(variants, doubles, args.seeds)
    jobs = [job for job in jobs if job["seed"] in args.seeds
            and int(family.loc[job["protein_id"], "outer_group"]) in args.folds]
    if args.smoke:
        jobs = jobs[::100]
    jobs = jobs[args.rank::args.world_size]
    root = ROOT / "runs" / ("mean_family_target_smoke" if args.smoke else "mean_family_target")
    root = root / f"family{args.threshold}" / args.method / args.mode
    root.mkdir(parents=True, exist_ok=True)
    cache = MeanFamilyCache() if jobs else None
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    outputs = []
    for job in jobs:
        protein, seed = job["protein_id"], job["seed"]
        directory = root / protein / f"seed{seed}" / job["scheme"] / job["condition"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        if path.exists():
            assert json.loads(path.read_text())["complete"]
            outputs.append(str(path.relative_to(ROOT)))
            continue
        allowed = set(job["train_ids"])
        assert not allowed & (set(job["validation_ids"]) | set(job["test_ids"]))
        training = cache.table.loc[cache.table.variant_id.isin(allowed) & cache.table.point_in_range].copy()
        assert training.variant_id.is_unique and set(training.protein_id) <= {protein}
        if args.mode == "S":
            assert training.n_mutations.le(1).all()
        training[["variant_id", "protein_id", "mutation_key", "n_mutations", "point", "row"]].to_parquet(
            directory / "training_variants.parquet", index=False)
        record = {"complete": True, "smoke": args.smoke, "dataset": "tsuboyama", "method": args.method,
                  "threshold": args.threshold, "mode": args.mode, "source_commit": commit,
                  **{key: value for key, value in job.items() if not key.endswith("_ids")},
                  "target_point_labels_used": len(training), "target_labels_permitted": len(allowed)}
        if len(training) < 2 or (args.mode == "C" and not job["budget_eligible"]):
            record.update(evaluated=False, reason="insufficient_permitted_point_labels" if len(training) < 2 else "combination_budget_unavailable")
        else:
            fold = int(family.loc[protein, "outer_group"])
            source = (ROOT / args.source_run).resolve() if args.source_run else selected_source(args.method, args.threshold, fold, seed)
            assert source.is_relative_to(ROOT)
            information = json.loads((source / "manifest.json").read_text())
            assert information["complete"] and (args.smoke or not information["smoke"])
            config = information["config"]
            assert config["method"] == args.method and config["fold"] == fold and config["threshold"] == args.threshold and config["seed"] == seed
            permission = json.loads((source / "permissions.json").read_text())
            relatives = set(family.index[family.family_group.eq(family.loc[protein, "family_group"])])
            assert relatives <= set(permission["test"])
            assert not relatives & (set(permission["train"]) | set(permission["validation"]))
            started = time.monotonic()
            predicted, details = fit_target(source, cache, training, job["validation_ids"], directory, seed)
            partition = {identifier: label for label in ["validation", "test"] for identifier in job[label + "_ids"]}
            predicted["set"] = predicted.variant_id.map(partition)
            predicted = predicted.loc[predicted["set"].notna()].copy()
            assert set(predicted.variant_id) == set(partition)
            predicted["condition"], predicted["mode"] = job["condition"], args.mode
            predicted.to_parquet(directory / "predictions.parquet", index=False)
            metrics = [per_protein_metrics(group).assign(partition=label, condition=job["condition"], mode=args.mode)
                       for label, group in predicted.groupby("set")]
            pd.concat(metrics, ignore_index=True).to_parquet(directory / "per_protein.parquet", index=False)
            record.update(evaluated=True, **details, source_manifest=str((source / "manifest.json").relative_to(ROOT)),
                          source_family_permissions_verified=True, target_label_permissions_verified=True,
                          elapsed_seconds=time.monotonic() - started)
            gc.collect()
            torch.cuda.empty_cache()
        path.write_text(json.dumps(record, indent=2))
        outputs.append(str(path.relative_to(ROOT)))
        print(json.dumps({"protein": protein, "seed": seed, "condition": job["condition"], "evaluated": record["evaluated"]}), flush=True)
    partition = "_folds" + "_".join(map(str, args.folds)) + "_seeds" + "_".join(map(str, args.seeds))
    (root / f"manifest{partition}_rank{args.rank}.json").write_text(json.dumps(
        {"complete": True, "config": vars(args), "outputs": outputs, "source_commit": commit,
         "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))


if __name__ == "__main__":
    main()
