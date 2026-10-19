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
sys.path.insert(0, str(ROOT / "scripts"))
from epistasis.metrics import per_protein_metrics
from epistasis.thermo_training import FEATURES, ThermoCache
from epistasis.thermo_target import fit_target
from run_mochi_targets import tasks_for


def selected_source(head, threshold, fold, seed):
    key = f"family{threshold}/fold{fold}"
    selected = []
    for path in (ROOT / "runs/thermo_family").glob(f"repeats_{head}_*/selected_hyperparameters.json"):
        entries = json.loads(path.read_text())
        if key in entries:
            selected.append(entries[key]["run"])
    assert selected and len(set(selected)) == 1, (head, key, selected)
    run = ROOT / selected[0]
    return run.with_name(run.name.rsplit("_seed", 1)[0] + f"_seed{seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=["epistatic", "additive"], required=True)
    parser.add_argument("--mode", choices=["S", "C"], required=True)
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source-run")
    args = parser.parse_args()
    assert args.source_run is None or args.smoke
    assert args.mode != "C" or args.threshold == 30
    torch.set_float32_matmul_precision("highest")
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet")
    family = pd.read_parquet(ROOT / "data/splits" / f"families_{args.threshold}.parquet").set_index("protein_id")
    doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
    single_order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    jobs = []
    if args.mode == "C":
        jobs = tasks_for("tsuboyama", "C")
    else:
        for protein in sorted(doubles.protein_id.unique()):
            wt = variants.loc[variants.protein_id.eq(protein) & variants.n_mutations.eq(0)]
            assert len(wt) == 1
            for seed in args.seeds:
                order = single_order.loc[single_order.protein_id.eq(protein) & single_order.seed.eq(seed)].sort_values("rank")
                for condition, budget in [("S32", 32), ("S128", 128), ("Sall", -1)]:
                    allowed = order if budget < 0 else order.iloc[:budget]
                    jobs.append({"protein_id": protein, "seed": seed, "scheme": "family", "condition": condition,
                                 "train_ids": wt.variant_id.tolist() + allowed.variant_id.tolist(),
                                 "validation_ids": [], "test_ids": doubles.loc[doubles.protein_id.eq(protein), "variant_id"].tolist(),
                                 "budget_eligible": budget < 0 or len(order) >= budget,
                                 "target_single_labels": len(allowed), "target_combination_labels": 0})
    jobs = [job for job in jobs if job["seed"] in args.seeds
            and int(family.loc[job["protein_id"], "outer_group"]) in args.folds]
    if args.smoke:
        jobs = jobs[::100]
    jobs = jobs[args.rank::args.world_size]
    output = ROOT / "runs" / ("thermo_target_smoke" if args.smoke else "thermo_target")
    output = output / f"family{args.threshold}" / args.head / args.mode
    output.mkdir(parents=True, exist_ok=True)
    direct = pd.read_parquet(FEATURES / "direct_index.parquet") if jobs else None
    cache = ThermoCache(FEATURES / "direct", args.head) if jobs else None
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    outputs = []
    for job in jobs:
        protein, seed = job["protein_id"], job["seed"]
        directory = output / protein / f"seed{seed}" / job["scheme"] / job["condition"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        if path.exists():
            assert json.loads(path.read_text())["complete"]
            outputs.append(str(path.relative_to(ROOT)))
            continue
        allowed = set(job["train_ids"])
        assert not allowed & (set(job["validation_ids"]) | set(job["test_ids"]))
        training = direct.loc[direct.variant_id.isin(allowed) & direct.training_point_eligible].copy()
        wt = variants.loc[variants.variant_id.isin(allowed) & variants.n_mutations.eq(0) & variants.point_in_range].copy()
        if len(wt):
            assert len(wt) == 1
            wt["native_ddg"] = 0.0
            training = pd.concat([wt, training], ignore_index=True)
        assert set(training.variant_id) <= allowed
        if args.mode == "S":
            assert training.n_mutations.le(1).all()
        training.to_parquet(directory / "training_variants.parquet", index=False)
        record = {"complete": True, "smoke": args.smoke, "dataset": "tsuboyama", "method": "thermo_" + args.head,
                  "threshold": args.threshold, "mode": args.mode, "source_commit": commit,
                  **{key: value for key, value in job.items() if not key.endswith("_ids")},
                  "target_point_labels_used": len(training), "target_labels_permitted": len(allowed)}
        if len(training) < 2 or (args.mode == "C" and not job["budget_eligible"]):
            record.update(evaluated=False, reason="insufficient_permitted_point_labels" if len(training) < 2 else "combination_budget_unavailable")
        else:
            fold = int(family.loc[protein, "outer_group"])
            source = (ROOT / args.source_run).resolve() if args.source_run else selected_source(args.head, args.threshold, fold, seed)
            assert source.is_relative_to(ROOT)
            source_info = json.loads((source / "manifest.json").read_text())
            assert source_info["complete"] and (args.smoke or not source_info["smoke"])
            assert source_info["config"]["threshold"] == args.threshold
            assert source_info["config"]["fold"] == fold and source_info["config"]["seed"] == seed
            permission = json.loads((source / "permissions.json").read_text())
            relatives = set(family.index[family.family_group.eq(family.loc[protein, "family_group"])])
            assert relatives <= set(permission["test"])
            assert not relatives & (set(permission["train"]) | set(permission["validation"]))
            started = time.monotonic()
            prediction, details = fit_target(source, cache, training, job["validation_ids"], directory, seed)
            sets = {identifier: label for label in ["validation", "test"] for identifier in job[label + "_ids"]}
            prediction["set"] = prediction.variant_id.map(sets)
            prediction = prediction.loc[prediction["set"].notna()].copy()
            assert set(prediction.variant_id) == set(sets)
            prediction["condition"], prediction["mode"] = job["condition"], args.mode
            prediction.to_parquet(directory / "predictions.parquet", index=False)
            summaries = [per_protein_metrics(group).assign(partition=partition, condition=job["condition"], mode=args.mode)
                         for partition, group in prediction.groupby("set")]
            pd.concat(summaries, ignore_index=True).to_parquet(directory / "per_protein.parquet", index=False)
            record.update(evaluated=True, **details, source_manifest=str((source / "manifest.json").relative_to(ROOT)),
                          source_family_permissions_verified=True, target_label_permissions_verified=True,
                          elapsed_seconds=time.monotonic() - started)
            gc.collect()
            torch.cuda.empty_cache()
        path.write_text(json.dumps(record, indent=2))
        outputs.append(str(path.relative_to(ROOT)))
        print(json.dumps({"protein": protein, "seed": seed, "condition": job["condition"],
                          "evaluated": record["evaluated"]}), flush=True)
    partition = "_folds" + "_".join(map(str, args.folds)) + "_seeds" + "_".join(map(str, args.seeds))
    (output / f"manifest{partition}_rank{args.rank}.json").write_text(json.dumps(
        {"complete": True, "config": vars(args), "outputs": outputs, "source_commit": commit,
         "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))


if __name__ == "__main__":
    main()
