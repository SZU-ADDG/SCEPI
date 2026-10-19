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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.higher import HigherFeatures
from epistasis.metrics import correlation, per_protein_metrics
from epistasis.mochi import MochiRegression
from epistasis.mean_regression import MeanRegression
from epistasis.rfa import RFARegression


def tasks_for(dataset, mode):
    root = ROOT / "data/processed" / dataset
    variants = pd.read_parquet(root / "variants.parquet")
    jobs = []
    if mode == "flip":
        split = pd.read_parquet(root / "splits.parquet")
        for source_file, partition in split.groupby("source_file", sort=True):
            equivalent = partition.effective_set.eq(partition.strict_set).all()
            for protocol, column in [("official", "effective_set")] + ([] if equivalent else [("strict", "strict_set")]):
                for protein in sorted(partition.loc[partition[column].eq("test"), "protein_id"].unique()):
                    current = partition.loc[partition.protein_id.eq(protein)]
                    for seed in [11, 23, 47, 89, 131]:
                        jobs.append({"protein_id": protein, "seed": seed, "scheme": Path(source_file).parent.name + "_" + current.official_split.iloc[0],
                                     "condition": protocol, "protocol_equivalence": ["official", "strict"] if equivalent else [protocol],
                                     "train_ids": current.loc[current[column].eq("train"), "variant_id"].tolist(),
                                     "validation_ids": current.loc[current[column].eq("validation"), "variant_id"].tolist(),
                                     "test_ids": current.loc[current[column].eq("test"), "variant_id"].tolist(),
                                     "budget_eligible": True, "budget": None})
    else:
        split_root = ROOT / "data/splits" / ("functional" if dataset == "functional_doubles" else "")
        splits = pd.read_parquet(split_root / "combination_splits.parquet")
        orders = pd.read_parquet(split_root / "single_label_order.parquet")
        singles_by_key = {key: group.variant_id.tolist() for key, group in orders.groupby(["protein_id", "seed"], sort=False)}
        references = {protein: group.variant_id.tolist() for protein, group in variants.loc[variants.n_mutations.eq(0)].groupby("protein_id", sort=False)}
        for (protein, seed, scheme), partition in splits.groupby(["protein_id", "seed", "scheme"], sort=True):
            single_ids = singles_by_key.get((protein, seed), [])
            wt_ids = references.get(protein, [])
            candidate = partition.loc[partition["set"].eq("train")].sort_values("train_rank")
            for condition in ["C128", "C512", "C2048", "Call"]:
                budget = len(candidate) if condition == "Call" else int(condition[1:])
                selected = candidate.iloc[:budget]
                train_ids = single_ids + wt_ids + selected.variant_id.tolist()
                validation_ids = partition.loc[partition["set"].eq("validation"), "variant_id"].tolist()
                test_ids = partition.loc[partition["set"].eq("test"), "variant_id"].tolist()
                assert not set(train_ids) & (set(validation_ids) | set(test_ids))
                if scheme != "random":
                    assert not set(selected.pair_id) & set(partition.loc[partition["set"].isin(["validation", "test"]), "pair_id"])
                jobs.append({"protein_id": protein, "seed": int(seed), "scheme": scheme, "condition": condition,
                             "train_ids": train_ids, "validation_ids": validation_ids, "test_ids": test_ids,
                             "budget": budget, "budget_eligible": len(candidate) >= budget and budget > 0,
                             "target_single_labels": len(single_ids), "target_combination_labels": len(selected),
                             "target_reference_variants": len(wt_ids)})
    return jobs


def save_results(features, job, directory, dataset, absolute_identifiable=True):
    arrays = np.load(directory / "predictions.npz")
    metadata = features.metadata.copy()
    metadata["prediction_absolute"] = arrays["variants"]
    metadata["prediction_mu"] = arrays["variants"] - float(arrays["reference"])
    metadata["prediction_wt"] = 0.0
    sets = {identifier: label for label, identifiers in [("validation", job["validation_ids"]), ("test", job["test_ids"])] for identifier in identifiers}
    metadata["set"] = metadata.variant_id.map(sets)
    metadata = metadata.loc[metadata["set"].notna()].copy()
    metadata.to_parquet(directory / "predictions.parquet", index=False)
    summary = []
    for partition, group in metadata.groupby("set"):
        for order, selected in [("all", group)] + [(str(value), group.loc[group.n_mutations.eq(value)]) for value in [1, 2, 3, 4]] + [("5plus", group.loc[group.n_mutations.ge(5)])]:
            point = selected.loc[selected.point.notna()]
            summary.append({"partition": partition, "mutation_order": order, "variants": len(selected),
                            "point_variants": len(point), "phenotype_spearman": correlation(point.point, point.prediction_absolute, "spearman"),
                            "phenotype_rmse": float(np.sqrt(np.mean((point.point - point.prediction_absolute) ** 2))) if len(point) and absolute_identifiable else None})
    if dataset == "tsuboyama":
        doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
        doubles = doubles.loc[doubles.variant_id.isin(metadata.variant_id)].copy()
        row_index = pd.Series(np.arange(len(features.metadata)), index=features.metadata.variant_id)
        allele_map = {row.variant_id: features.pair_alleles[features.pair_incidence[row_index[row.variant_id]].indices.item()] for row in doubles.itertuples()}
        predicted = metadata.set_index("variant_id")
        doubles["prediction_mu"] = doubles.variant_id.map(predicted.prediction_mu)
        doubles["prediction_a"] = [arrays["singles"][allele_map[key][0]] - float(arrays["reference"]) for key in doubles.variant_id]
        doubles["prediction_b"] = [arrays["singles"][allele_map[key][1]] - float(arrays["reference"]) for key in doubles.variant_id]
        doubles["prediction_wt"] = 0.0
        doubles["prediction_epsilon"] = doubles.prediction_mu - doubles.prediction_a - doubles.prediction_b
        doubles["set"] = doubles.variant_id.map(sets)
        doubles.to_parquet(directory / "double_predictions.parquet", index=False)
        records = [per_protein_metrics(group).assign(partition=partition) for partition, group in doubles.groupby("set")]
        pd.concat(records, ignore_index=True).to_parquet(directory / "per_protein.parquet", index=False)
    elif dataset == "functional_doubles":
        quartets = pd.read_parquet(ROOT / "data/processed/functional/quartets.parquet")
        quartets = quartets.loc[quartets.ab_id.isin(metadata.variant_id)].copy()
        predicted = pd.Series(arrays["variants"], index=features.metadata.variant_id)
        quartets["prediction_epsilon"] = (quartets.ab_id.map(predicted) - quartets.a_id.map(predicted) -
                                          quartets.b_id.map(predicted) + quartets.wt_id.map(predicted))
        assert quartets.prediction_epsilon.notna().all()
        quartets["set"] = quartets.ab_id.map(sets)
        quartets.to_parquet(directory / "quartet_predictions.parquet", index=False)
        for partition, group in quartets.groupby("set"):
            summary.append({"partition": partition, "mutation_order": "quartet", "variants": len(group),
                            "epsilon_rmse": float(np.sqrt(np.mean((group.prediction_epsilon - group.epsilon) ** 2))),
                            "epsilon_spearman": correlation(group.epsilon, group.prediction_epsilon, "spearman")})
    pd.DataFrame(summary).to_csv(directory / "metrics.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tsuboyama", "functional_doubles", "flip_primary"], required=True)
    parser.add_argument("--mode", choices=["C", "flip"], required=True)
    parser.add_argument("--order", type=int, choices=[1, 2], required=True)
    parser.add_argument("--method", choices=["mochi", "rfa", "mavenn", "mean_ridge", "mean_mlp"], default="mochi")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--proteins", nargs="+")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-file")
    parser.add_argument("--rfa-accelerated", action="store_true")
    parser.add_argument("--rfa-policy")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.method != "rfa" or args.order == 2
    assert not args.rfa_accelerated or (args.method == "rfa" and not args.smoke)
    assert args.rfa_policy is None or args.rfa_accelerated
    rfa_policy = json.loads((ROOT / args.rfa_policy).read_text()) if args.rfa_policy is not None else None
    fixed_default = rfa_policy is not None and rfa_policy["penalty_candidates"] == 1
    if args.method == "mavenn":
        from epistasis.mavenn import MavennRegression
    if args.task_file is not None:
        assert args.rank == 0 and args.world_size == 1 and not args.smoke and args.proteins is None
        payload = json.loads((ROOT / args.task_file).read_text())
        assert (payload["dataset"], payload["mode"], payload["index"]) == (args.dataset, args.mode, args.task_index)
        jobs = [payload["job"]]
    else:
        jobs = tasks_for(args.dataset, args.mode)
        if args.proteins is not None:
            jobs = [job for job in jobs if job["protein_id"] in args.proteins]
        if args.smoke:
            jobs = jobs[::100]
        if args.task_index is not None:
            assert args.rank == 0 and args.world_size == 1 and not args.smoke
            assert 0 <= args.task_index < len(jobs)
            jobs = [jobs[args.task_index]]
        else:
            jobs = jobs[args.rank::args.world_size]
    method = args.method + ("_smoke" if args.smoke else "")
    if args.rfa_accelerated:
        method = "rfa_bounded" if args.rfa_policy is not None else "rfa_accelerated"
        if fixed_default:
            method = "rfa_fixed"
    output = ROOT / "runs" / method / args.dataset / f"order{args.order}"
    output.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    outputs = []
    features, current_protein = None, None
    for job in jobs:
        directory = output / job["protein_id"] / f"seed{job['seed']}" / job["scheme"] / job["condition"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        if path.exists():
            assert json.loads(path.read_text())["complete"]
            outputs.append(str(path.relative_to(ROOT)))
            continue
        if current_protein != job["protein_id"]:
            features = HigherFeatures(ROOT / "data/features/higher" / args.dataset / job["protein_id"])
            current_protein = job["protein_id"]
        table = features.metadata
        index = pd.Series(np.arange(len(table)), index=table.variant_id)
        permitted_train = index.loc[job["train_ids"]].to_numpy()
        permitted_validation = index.loc[job["validation_ids"]].to_numpy()
        points = table.point.to_numpy(dtype=np.float64)
        train = permitted_train[np.isfinite(points[permitted_train])]
        validation = permitted_validation[np.isfinite(points[permitted_validation])]
        ledger = pd.concat([table.iloc[permitted_train].assign(role="train"), table.iloc[permitted_validation].assign(role="validation")])
        ledger["point_likelihood_used"] = ledger.point.notna()
        ledger.to_parquet(directory / "label_permissions.parquet", index=False)
        manifest = {"complete": True, "smoke": args.smoke, "method": f"{args.method}_order{args.order}" if args.method != "rfa" else "rfa",
                    "source_commit": source_commit, "dataset": args.dataset, **{key: value for key, value in job.items() if not key.endswith("_ids")},
                    "permitted_training_variants": len(permitted_train), "permitted_validation_variants": len(permitted_validation),
                    "point_training_variants": len(train), "point_validation_variants": len(validation), "test_variants": len(job["test_ids"]),
                    "point_likelihood": "Available biological-variant point estimates; interval labels are recorded and never imputed."}
        if args.rfa_accelerated:
            manifest.update(accelerated=True,
                            acceleration_validation="runs/rfa_acceleration_20261018/validation.json")
        if args.rfa_policy is not None:
            manifest.update(bounded_optimization=True,
                            fixed_default=fixed_default,
                            budget_validation=f"runs/{method}_20261018/validation.json",
                            optimization_policy=rfa_policy)
        if not job["budget_eligible"] or len(train) < 2 or len(validation) < 2 or np.std(points[train]) == 0:
            manifest.update({"evaluated": False, "reason": "budget_unavailable" if not job["budget_eligible"] else "permitted_training_or_validation_unavailable"})
        else:
            started = time.monotonic()
            if args.method == "mochi":
                regression = MochiRegression(features, args.order, train, args.device)
            elif args.method == "rfa":
                regression = RFARegression(features, args.dataset, accelerated=args.rfa_accelerated, optimizer_policy=args.rfa_policy)
            elif args.method == "mavenn":
                regression = MavennRegression(features, args.dataset, args.order, args.device)
            else:
                regression = MeanRegression(features, args.dataset, args.method, args.device)
            epsilon_components = None
            if args.dataset == "tsuboyama":
                quartets = pd.read_parquet(ROOT / "data/processed/tsuboyama/quartets.parquet")
                quartets = quartets.loc[quartets.main & quartets.ab_id.isin(job["validation_ids"])]
                if len(quartets):
                    assert set(quartets.a_id) | set(quartets.b_id) | set(quartets.wt_id) <= set(job["train_ids"])
                    component_rows = np.stack([quartets[part + "_id"].map(index).to_numpy() for part in ["wt", "a", "b", "ab"]], axis=1)
                    epsilon_components = (component_rows, quartets.epsilon.to_numpy())
            details = regression.fit(train, validation, points, job["seed"], directory, epsilon_components=epsilon_components)
            metrics = save_results(features, job, directory, args.dataset)
            manifest.update({"evaluated": True, **details, "metrics": metrics, "elapsed_seconds": time.monotonic() - started})
            del regression
            gc.collect()
        pending = path.with_suffix(".pending.json")
        pending.write_text(json.dumps(manifest, indent=2))
        pending.replace(path)
        outputs.append(str(path.relative_to(ROOT)))
        print(json.dumps({"protein": job["protein_id"], "seed": job["seed"], "scheme": job["scheme"],
                          "condition": job["condition"], "order": args.order, "evaluated": manifest["evaluated"]}), flush=True)
    partition = "_proteins_" + "_".join(args.proteins) if args.proteins is not None else ""
    if args.task_index is not None:
        partition += f"_task{args.task_index}"
    (output / f"manifest{partition}_rank{args.rank}.json").write_text(json.dumps({"complete": True, "smoke": args.smoke,
        "config": vars(args), "outputs": outputs, "source_commit": source_commit, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}, indent=2))


if __name__ == "__main__":
    main()
