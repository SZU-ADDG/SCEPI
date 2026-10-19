import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from train_single import ROOT, FEATURES, family_permissions
from epistasis.metrics import per_protein_metrics

PENALTIES = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def target_scales(singles, order, score, proteins, budget, prior, penalty):
    result = pd.Series(prior, index=pd.Index(proteins, name="protein_id"), dtype=float)
    if budget == 0:
        return result
    allowed = order if budget < 0 else order.loc[order["rank"].lt(budget)]
    targets = singles.loc[singles.protein_id.isin(proteins)].merge(allowed[["variant_id"]], on="variant_id", validate="one_to_one")
    targets["score"] = score.reindex(targets.variant_id).to_numpy()
    assert np.isfinite(targets[["score", "relative_point"]]).all().all()
    targets["xx"] = targets.score ** 2
    targets["xy"] = targets.score * targets.relative_point
    sums = targets.groupby("protein_id")[["xx", "xy"]].sum()
    assert sums.xx.gt(0).all()
    result.loc[sums.index] = (sums.xy + penalty * prior) / (sums.xx + penalty)
    return result


def predict(table, scales, score_ab, endpoint, additive=False):
    frame = table.copy()
    slope = frame.protein_id.map(scales).to_numpy()
    rows = frame.matrix_row.to_numpy()
    frame["prediction_mu"] = score_ab.reindex(frame.variant_id).to_numpy() * slope
    frame["prediction_a"] = endpoint[rows, 0] * slope
    frame["prediction_b"] = endpoint[rows, 1] * slope
    frame["prediction_wt"] = 0.0
    frame["prediction_epsilon"] = frame.prediction_mu - frame.prediction_a - frame.prediction_b
    if additive:
        frame["prediction_mu"] = frame.prediction_a + frame.prediction_b
        frame["prediction_epsilon"] = 0.0
    frame["calibration_slope"] = slope
    return frame


def macro_rmse(frame):
    main = frame.loc[frame.main.fillna(False)].copy()
    counts = main.groupby("protein_id").size()
    main = main.loc[main.protein_id.isin(counts.index[counts.ge(30)])]
    error = (main.prediction_epsilon - main.epsilon) ** 2
    return float(error.groupby(main.protein_id).mean().pow(0.5).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[25, 30], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=[11, 23, 47, 89, 131], required=True)
    args = parser.parse_args()
    output = ROOT / "runs/e3" / f"family{args.threshold}" / f"fold{args.fold}" / "esm_scores" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists()
    started = time.monotonic()
    singles = pd.read_parquet(FEATURES / "singles/metadata.parquet")
    doubles = pd.read_parquet(FEATURES / "doubles/metadata.parquet")
    endpoint = np.load(FEATURES / "doubles/score.npy", mmap_mode="r")
    mean_metadata = pd.read_parquet(ROOT / "data/features/mean/tsuboyama/metadata.parquet", columns=["variant_id"])
    scores = np.load(ROOT / "data/features/mean/tsuboyama/scores.npy")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    order = order.loc[order.seed.eq(args.seed)].copy()
    target_count = order.groupby("protein_id").size()
    permissions = family_permissions(args.threshold, args.fold)
    family = pd.read_parquet(ROOT / "data/splits" / f"families_{args.threshold}.parquet")
    family["available_single_labels"] = family.protein_id.map(target_count).fillna(0).astype(int)
    source = pd.concat([singles, doubles], ignore_index=True)
    source = source.loc[source.protein_id.isin(permissions["train"]) & source.point_in_range & source.wt_point_in_range].copy()
    assert np.isfinite(source.relative_point).all()
    source[["variant_id", "protein_id"]].to_parquet(output / "source_calibration_labels.parquet", index=False)
    (output / "permissions.json").write_text(json.dumps(permissions, indent=2))
    configurations, metrics = [], []
    for index, method in enumerate(["esm2_additive", "esm2_context"]):
        score_ab = pd.Series(scores[:, index], index=mean_metadata.variant_id)
        single_score = pd.Series(scores[:, 0], index=mean_metadata.variant_id)
        x = score_ab.reindex(source.variant_id).to_numpy()
        y = source.relative_point.to_numpy()
        prior = float(np.dot(x, y) / np.dot(x, x))
        validation = doubles.loc[doubles.protein_id.isin(permissions["validation"])]
        trials = []
        for penalty in PENALTIES:
            scores_by_condition = {}
            for condition, budget in [("Z", 0), ("S128", 128), ("Sall", -1)]:
                slopes = target_scales(singles, order, single_score, permissions["validation"], budget, prior, penalty)
                scores_by_condition[condition] = macro_rmse(predict(validation, slopes, score_ab, endpoint, additive=index == 0))
            value = 0.5 * scores_by_condition["Z"] + 0.25 * scores_by_condition["S128"] + 0.25 * scores_by_condition["Sall"]
            assert np.isfinite(value)
            trials.append({"penalty": penalty, "validation_score": value, "conditions": scores_by_condition})
        selected = min(trials, key=lambda trial: (trial["validation_score"], trial["penalty"]))
        configurations.append({"method": method, "source_slope": prior, "selected": selected, "trials": trials})
        for partition in ["validation", "test"]:
            table = doubles.loc[doubles.protein_id.isin(permissions[partition])]
            for condition, budget in [("Z", 0), ("S32", 32), ("S128", 128), ("Sall", -1)]:
                slopes = target_scales(singles, order, single_score, permissions[partition], budget, prior, selected["penalty"])
                frame = predict(table, slopes, score_ab, endpoint, additive=index == 0)
                frame = frame.merge(family, on="protein_id", validate="many_to_one")
                frame["method"], frame["condition"], frame["seed"] = method, condition, args.seed
                frame["budget_eligible"] = True if budget <= 0 else frame.available_single_labels.ge(budget)
                frame["target_single_labels"] = 0 if budget == 0 else (frame.available_single_labels if budget < 0 else frame.available_single_labels.clip(upper=budget))
                frame["target_wt_labels"] = int(budget != 0)
                frame.to_parquet(output / f"{partition}_{method}_{condition}.parquet", index=False)
                current = per_protein_metrics(frame)
                current = current.merge(family, on="protein_id", validate="one_to_one")
                current["method"], current["condition"], current["partition"] = method, condition, partition
                current["seed"], current["fold"] = args.seed, args.fold
                current["budget_eligible"] = True if budget <= 0 else current.available_single_labels.ge(budget)
                metrics.append(current)
        print(json.dumps({"method": method, "source_slope": prior, "selected": selected}), flush=True)
    pd.concat(metrics, ignore_index=True).to_parquet(output / "per_protein.parquet", index=False)
    manifest = {
        "complete": True, "config": vars(args), "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_calibration_variants": len(source), "calibration": "Through-origin least squares on uncensored source variants. Target S uses only its fixed WT and single-label budget, with ridge penalty toward the source slope selected on source validation.",
        "source_calibration_censored_labels_used": 0, "selection_scope": "Validation families only; candidate selection metrics never select a model.",
        "methods": configurations, "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"complete": True, "output": str(output.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    main()
