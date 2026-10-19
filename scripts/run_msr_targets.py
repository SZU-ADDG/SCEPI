import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from epistasis.metrics import per_protein_metrics
from epistasis.msr import FEATURES
from epistasis.msr_target import fit_target
from run_mean_family_targets import single_tasks
from run_mochi_targets import tasks_for


def selected_source(threshold, fold, seed):
    key = f"family{threshold}/fold{fold}"
    selected = []
    for path in (ROOT / "runs/msr_family").glob("repeats_*/selected_hyperparameters.json"):
        records = json.loads(path.read_text())
        if key in records:
            selected.append(records[key]["run"])
    assert selected and len(set(selected)) == 1, (key, selected)
    source = ROOT / selected[0]
    return source.with_name(source.name.rsplit("_seed", 1)[0] + f"_seed{seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["S", "C"], required=True)
    parser.add_argument("--threshold", type=int, choices=[30, 25], default=30)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--source-run")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.source_run is not None:
        assert len(args.folds) == len(args.seeds) == 1
    assert args.mode != "C" or args.threshold == 30
    family = pd.read_parquet(ROOT / "data/splits" / f"families_{args.threshold}.parquet").set_index("protein_id")
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet")
    doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
    metadata = pd.read_parquet(FEATURES / "variants.parquet")
    jobs = tasks_for("tsuboyama", "C") if args.mode == "C" else single_tasks(variants, doubles, args.seeds)
    jobs = [job for job in jobs if job["seed"] in args.seeds
            and int(family.loc[job["protein_id"], "outer_group"]) in args.folds
            and (args.conditions is None or job["condition"] in args.conditions)]
    if args.smoke:
        continuous = set(metadata.loc[metadata.training_point_eligible, "variant_id"])
        references = set(variants.loc[variants.n_mutations.eq(0) & variants.point_in_range, "variant_id"])
        jobs = [job for job in jobs if job["budget_eligible"]
                and len(set(job["train_ids"]) & continuous) >= 16
                and len(set(job["train_ids"]) & references) == 1]
        jobs = jobs[::100]
    jobs = jobs[args.rank::args.world_size]
    output = ROOT / "runs" / ("msr_target_smoke" if args.smoke else "msr_target") / f"family{args.threshold}" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifests, module = [], None
    for job in jobs:
        protein, seed = job["protein_id"], job["seed"]
        directory = output / protein / f"seed{seed}" / job["scheme"] / job["condition"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        if path.exists():
            assert json.loads(path.read_text())["complete"]
            manifests.append(str(path.relative_to(ROOT)))
            continue
        allowed = set(job["train_ids"])
        assert not allowed & (set(job["validation_ids"]) | set(job["test_ids"]))
        reference = variants.loc[variants.variant_id.isin(allowed) & variants.n_mutations.eq(0) & variants.point_in_range]
        training = metadata.loc[metadata.variant_id.isin(allowed) & metadata.training_point_eligible].copy()
        assert set(training.protein_id) <= {protein}
        if args.mode == "S":
            assert training.n_mutations.eq(1).all()
        training.to_parquet(directory / "training_variants.parquet", index=False)
        reference.to_parquet(directory / "training_reference.parquet", index=False)
        pd.DataFrame([{"variant_id": identifier, "partition": label} for label in ["train", "validation", "test"]
                      for identifier in job[label + "_ids"]]).to_parquet(directory / "permissions.parquet", index=False)
        information = {"complete": True, "smoke": args.smoke, "source_commit": commit, "method": "esm_msr",
                       "dataset": "tsuboyama", "threshold": args.threshold, "mode": args.mode,
                       **{key: value for key, value in job.items() if not key.endswith("_ids")},
                       "target_point_labels_used": len(training) + len(reference), "target_labels_permitted": len(allowed)}
        if len(reference) != 1 or len(training) < 16 or not job["budget_eligible"]:
            reason = "combination_or_single_budget_unavailable" if not job["budget_eligible"] else "insufficient_continuous_target_labels_for_original_ListMLE"
            information.update(evaluated=False, reason=reason)
        else:
            observed = variants.set_index("variant_id").loc[training.variant_id]
            np.testing.assert_allclose(training.native_ddg, reference.point.iloc[0] - observed.point.to_numpy(), rtol=1e-6, atol=1e-6)
            fold = int(family.loc[protein, "outer_group"])
            source = (ROOT / args.source_run).resolve() if args.source_run else selected_source(args.threshold, fold, seed)
            assert source.is_relative_to(ROOT)
            source_info = json.loads((source / "manifest.json").read_text())
            assert source_info["complete"] and (args.smoke or not source_info["smoke"])
            config = source_info["config"]
            assert config["fold"] == fold and config["threshold"] == args.threshold and config["seed"] == seed
            permissions = json.loads((source / "permissions.json").read_text())
            relatives = set(family.index[family.family_group.eq(family.loc[protein, "family_group"])])
            assert relatives <= set(permissions["test"])
            assert not relatives & (set(permissions["train"]) | set(permissions["validation"]))
            started = time.monotonic()
            prediction, details, module = fit_target(source, metadata, doubles, training, job, directory, args.smoke, module)
            partition = {identifier: label for label in ["validation", "test"] for identifier in job[label + "_ids"]}
            prediction["set"] = prediction.variant_id.map(partition)
            assert set(prediction.variant_id) == set(partition) and prediction["set"].notna().all()
            prediction["condition"], prediction["mode"] = job["condition"], args.mode
            prediction.to_parquet(directory / "predictions.parquet", index=False)
            metrics = [per_protein_metrics(group).assign(partition=label, condition=job["condition"], mode=args.mode)
                       for label, group in prediction.groupby("set")]
            pd.concat(metrics, ignore_index=True).to_parquet(directory / "per_protein.parquet", index=False)
            information.update(evaluated=True, **details, source_manifest=str((source / "manifest.json").relative_to(ROOT)),
                               source_family_permissions_verified=True, target_label_permissions_verified=True,
                               elapsed_seconds=time.monotonic() - started)
        path.write_text(json.dumps(information, indent=2))
        manifests.append(str(path.relative_to(ROOT)))
        print(json.dumps({"protein": protein, "seed": seed, "condition": job["condition"], "evaluated": information["evaluated"]}), flush=True)
    partition = "_folds" + "_".join(map(str, args.folds)) + "_seeds" + "_".join(map(str, args.seeds))
    if args.conditions:
        partition += "_conditions" + "_".join(args.conditions)
    (output / f"manifest{partition}_rank{args.rank}.json").write_text(json.dumps(
        {"complete": True, "config": vars(args), "source_commit": commit, "outputs": manifests,
         "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))


if __name__ == "__main__":
    main()
