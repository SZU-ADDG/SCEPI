import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/compensation_main"
SEEDS = [11, 23, 47, 89, 131]
BUDGETS = [1, 3, 10, 30]
MARGINS = [0.0, 0.25, 0.5]
PREDICTION_COLUMNS = ["variant_id", "prediction_mu", "prediction_a", "prediction_b", "prediction_wt", "prediction_epsilon"]


def definitions():
    pair = {
        "interaction": "full_g0_f1_j1",
        "direct_residual": "full_g0_f1_j1_direct",
        "no_pair_measured": "full_g0_f1_j1_nopairmeasured",
        "final_interaction": "source_selected",
        "final_remove_interaction": "source_selected",
        "final_direct_residual": "source_selected_direct",
        "final_no_pair_measured": "source_selected_nopairmeasured",
        "final_values_only": "source_selected_nomask",
        "final_mask_only": "source_selected_predictedvalues",
        "additive": "full_g0_f0_j1_n1",
        "global_nonlinear": "full_g0_f1_j1_n1",
        "remove_interaction": "full_g0_f1_j1",
        "frozen_main": "full_g0_f1_j0",
        "identity_mapping": "full_g0_f0_j1",
        "reference_context": "wt_g0_f1_j1",
        "identity_features": "identity_g0_f1_j1",
        "no_cross": "no_cross_g0_f1_j1",
        "residual_only": "full_g0_f1_j1_residual",
        "phenotype_only": "full_g0_f1_j1_phenotype",
        "geometry": "full_g1_f1_j1",
        "geometry_permuted_retrained": "full_g1_f1_j1_geopermuted",
        "geometry_rewired_retrained": "full_g1_f1_j1_georewired",
        "geometry_permuted_inference": "full_g1_f1_j1",
        "geometry_rewired_inference": "full_g1_f1_j1",
    }
    result = {}
    for arm, tag in pair.items():
        result[arm] = {"kind": "pair", "tag": tag, "conditions": ["Sall"], "thresholds": [30, 25],
                       "requires_structure": arm.startswith("geometry"),
                       "prediction_method": "remove_interaction" if arm in {"remove_interaction", "final_remove_interaction"} else "shared"}
    for arm in ["esm2_additive", "esm2_context"]:
        result[arm] = {"kind": "esm_scores", "conditions": ["Sall"], "thresholds": [30, 25], "requires_structure": False}
    for arm, template in {
        "mean_ridge": "runs/mean_family_target/family{threshold}/mean_ridge/S/{protein}/seed{seed}/family/Sall",
        "mean_mlp": "runs/mean_family_target/family{threshold}/mean_mlp/S/{protein}/seed{seed}/family/Sall",
        "kermut": "runs/kermut_target/S/{protein}/seed{seed}/family/Sall",
        "thermo_additive": "runs/thermo_target/family{threshold}/additive/S/{protein}/seed{seed}/family/Sall",
        "thermo_epistatic": "runs/thermo_target/family{threshold}/epistatic/S/{protein}/seed{seed}/family/Sall",
        "esm_msr": "runs/msr_target/family{threshold}/S/{protein}/seed{seed}/family/Sall",
    }.items():
        result[arm] = {"kind": "target", "template": template, "conditions": ["Sall"], "thresholds": [30, 25],
                       "requires_structure": arm in ["kermut", "thermo_additive", "thermo_epistatic"],
                       "prediction_file": "predictions.parquet"}
    auxiliary = {
        "interaction_C": "runs/shared_target/{protein}/seed{seed}/unseen_pair/{condition}",
        "interaction_target_only_C": "runs/shared_target_only/{protein}/seed{seed}/unseen_pair/{condition}",
        "mean_ridge_source_C": "runs/mean_family_target/family{threshold}/mean_ridge/C/{protein}/seed{seed}/unseen_pair/{condition}",
        "mean_mlp_source_C": "runs/mean_family_target/family{threshold}/mean_mlp/C/{protein}/seed{seed}/unseen_pair/{condition}",
        "kermut_C": "runs/kermut_target/C/{protein}/seed{seed}/unseen_pair/{condition}",
        "thermo_additive_C": "runs/thermo_target/family{threshold}/additive/C/{protein}/seed{seed}/unseen_pair/{condition}",
        "thermo_epistatic_C": "runs/thermo_target/family{threshold}/epistatic/C/{protein}/seed{seed}/unseen_pair/{condition}",
        "esm_msr_C": "runs/msr_target/family{threshold}/C/{protein}/seed{seed}/unseen_pair/{condition}",
        "mochi_order1_C": "runs/mochi/tsuboyama/order1/{protein}/seed{seed}/unseen_pair/{condition}",
        "mochi_order2_C": "runs/mochi/tsuboyama/order2/{protein}/seed{seed}/unseen_pair/{condition}",
        "mean_ridge_target_C": "runs/mean_ridge/tsuboyama/order2/{protein}/seed{seed}/unseen_pair/{condition}",
        "mean_mlp_target_C": "runs/mean_mlp/tsuboyama/order2/{protein}/seed{seed}/unseen_pair/{condition}",
    }
    for arm, template in auxiliary.items():
        result[arm] = {"kind": "target", "template": template, "conditions": ["C128", "C512", "C2048", "Call"],
                       "thresholds": [30], "requires_structure": arm in ["kermut_C", "thermo_additive_C", "thermo_epistatic_C"],
                       "prediction_file": "double_predictions.parquet" if arm.startswith(("mochi_", "mean_ridge_target", "mean_mlp_target")) else "predictions.parquet"}
    return result


def destination(arm, threshold, seed, fold, condition):
    return OUTPUT / "evaluations" / arm / f"family{threshold}" / f"seed{seed}" / f"fold{fold}" / condition


def frozen_table(threshold, seed, fold):
    return pd.read_parquet(OUTPUT / "candidates" / f"family{threshold}" / f"seed{seed}" / f"fold{fold}.parquet")


def input_records(arm, threshold, seed, fold, condition, truth):
    specification = definitions()[arm]
    missing, entries = [], []
    if specification["kind"] == "pair":
        key, tag = f"family{threshold}/fold{fold}", specification["tag"]
        selections = []
        for directory in ["pair_repeats_" + tag, "pair_repeats_" + tag + f"_family{threshold}"]:
            path = ROOT / "runs/e3" / directory / "selected_hyperparameters.json"
            if path.exists():
                selected = json.loads(path.read_text())
                if key in selected:
                    selections.append(selected[key]["run"])
        if not selections:
            return [], ["source-validation selection for " + tag + "/" + key]
        assert len(set(selections)) == 1
        selected = ROOT / selections[0]
        source = selected.with_name(selected.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
        suffix = ""
        if arm.endswith("_inference"):
            suffix = "_" + ("permuted" if "permuted" in arm else "rewired")
        directory = source / ("evaluation_test" + suffix)
        entries = [{"manifest": directory / "manifest.json", "prediction": directory / f"{specification['prediction_method']}_Sall.parquet",
                    "source": source, "protein": None}]
    elif specification["kind"] == "esm_scores":
        directory = ROOT / "runs/e3" / f"family{threshold}" / f"fold{fold}" / "esm_scores" / f"seed{seed}"
        entries = [{"manifest": directory / "manifest.json", "prediction": directory / f"test_{arm}_Sall.parquet",
                    "source": directory, "protein": None}]
    else:
        for protein in sorted(truth.protein_id.unique()):
            directory = ROOT / specification["template"].format(threshold=threshold, seed=seed, protein=protein, condition=condition)
            entries.append({"manifest": directory / "manifest.json", "prediction": directory / specification["prediction_file"],
                            "source": None, "protein": protein})
    for entry in entries:
        path = entry["manifest"]
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        information = json.loads(path.read_text())
        assert information["complete"] and not information.get("smoke", False), path
        entry["information"] = information
        if information.get("evaluated", True) and not entry["prediction"].exists():
            raise FileNotFoundError(entry["prediction"])
    return entries, missing


def check_family(source, threshold, fold, seed, truth):
    permissions = json.loads((source / "permissions.json").read_text())
    assert set(truth.protein_id) <= set(permissions["test"])
    assert not set(truth.protein_id) & (set(permissions["train"]) | set(permissions["validation"]))
    information = json.loads((source / "manifest.json").read_text())
    config = information["config"]
    assert config["threshold"] == threshold and config["fold"] == fold and config["seed"] == seed


def normalized_predictions(arm, threshold, seed, fold, condition, truth, entries):
    chunks, excluded, provenance = [], [], []
    for entry in entries:
        information = entry["information"]
        if not information.get("evaluated", True):
            excluded.append({"protein_id": entry["protein"], "reason": information["reason"]})
            continue
        current = truth if entry["protein"] is None else truth.loc[truth.protein_id.eq(entry["protein"])]
        if entry["source"] is not None:
            check_family(entry["source"], threshold, fold, seed, current)
        for field in ["source_manifest", "source_run"]:
            if field in information:
                source = ROOT / information[field]
                source = source.parent if field == "source_manifest" else source
                check_family(source, threshold, fold, seed, current)
        frame = pd.read_parquet(entry["prediction"], columns=PREDICTION_COLUMNS)
        assert frame.variant_id.is_unique
        assert set(current.variant_id) <= set(frame.variant_id), entry["prediction"]
        selected = frame.loc[frame.variant_id.isin(current.variant_id)].copy()
        assert np.isfinite(selected[PREDICTION_COLUMNS[1:]].to_numpy()).all()
        np.testing.assert_allclose(selected.prediction_epsilon,
                                   selected.prediction_mu - selected.prediction_a - selected.prediction_b + selected.prediction_wt,
                                   rtol=1e-5, atol=1e-5)
        if information.get("mode") == "S":
            assert information["target_combination_labels"] == 0
        if "condition" in information:
            assert information["condition"] == condition
        if "seed" in information:
            assert information["seed"] == seed
        chunks.append(selected)
        provenance.append({"manifest": str(entry["manifest"].relative_to(ROOT)),
                           "predictions": str(entry["prediction"].relative_to(ROOT)),
                           "source_commit": information["source_commit"],
                           "label_usage": {key: information[key] for key in [
                               "target_single_labels", "target_combination_labels", "target_point_labels_used",
                               "target_labels_permitted", "point_training_variants", "point_validation_variants",
                               "source_training_variants", "source_labels_used", "supervised_source_weights_used",
                               "target_test_labels_used", "source_calibration_variants",
                           ] if key in information}})
    permitted = truth.loc[~truth.protein_id.isin([item["protein_id"] for item in excluded])].copy()
    values = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=PREDICTION_COLUMNS)
    assert values.variant_id.is_unique and set(values.variant_id) == set(permitted.variant_id)
    return permitted.merge(values, on="variant_id", validate="one_to_one"), excluded, provenance


def residual_metrics(frame):
    rows = []
    for protein, group in frame.groupby("protein_id", sort=True):
        for margin in MARGINS:
            point = group.loc[group.main.fillna(False) & group.margin.ge(margin)]
            if len(point) < 30:
                continue
            squared = (point.prediction_epsilon - point.epsilon) ** 2
            rows.append({"protein_id": protein, "family_group": group.family_group.iloc[0], "margin": margin,
                         "variants": len(point), "epsilon_rmse": float(np.sqrt(squared.mean())),
                         "pair_macro_epsilon_rmse": float(squared.groupby(point.pair_id).mean().pow(0.5).mean())})
    return pd.DataFrame(rows, columns=["protein_id", "family_group", "margin", "variants", "epsilon_rmse", "pair_macro_epsilon_rmse"])


def selection_metrics(frame, units=("protein", "pair")):
    rows = []
    original = frame.loc[frame.selection_pool]
    tasks = [("original", 0.0, original, ["native"])]
    for margin in MARGINS:
        point = original.loc[original.main.fillna(False) & original.margin.ge(margin)]
        tasks += [("point", margin, point, ["native", "observed_main"]),
                  ("compensation", margin, point.loc[point.compensation_pool], ["native", "observed_main"])]
    for task, margin, candidates, scorings in tasks:
        for unit in units:
            keys = ["protein_id"] if unit == "protein" else ["protein_id", "pair_id"]
            for _, group in candidates.groupby(keys, sort=True):
                if len(group) < 100:
                    continue
                for scoring in scorings:
                    if scoring == "observed_main":
                        assert group.known_single_a.all() and group.known_single_b.all()
                    score = group.prediction_mu if scoring == "native" else group.common_observed_main + group.prediction_epsilon
                    assert score.notna().all()
                    ranked = group.assign(score=score).sort_values(["score", "mutation_key"], ascending=[False, True], kind="stable")
                    for budget in BUDGETS:
                        chosen = ranked.head(budget)
                        entry = {"protein_id": group.protein_id.iloc[0], "family_group": group.family_group.iloc[0],
                                 "pair_id": group.pair_id.iloc[0] if unit == "pair" else "all", "task": task, "margin": margin,
                                 "unit": unit, "scoring": scoring, "budget": budget, "pool_size": len(group),
                                 "selected_variant_ids": chosen.variant_id.tolist(), "selected_scores": chosen.score.tolist(),
                                 "precision_lower": float(chosen.relative_lower.ge(0.5).mean()),
                                 "precision_upper": float(chosen.relative_upper.ge(0.5).mean())}
                        if task != "original":
                            recovery = chosen.relative_point - chosen[["single_a", "single_b"]].max(axis=1)
                            entry.update(selected_relative_point=chosen.relative_point.tolist(), selected_recovery=recovery.tolist(),
                                         selected_mean_stability=float(chosen.relative_point.mean()), selected_best_stability=float(chosen.relative_point.max()),
                                         selected_mean_recovery=float(recovery.mean()), selected_best_recovery=float(recovery.max()))
                        if task == "compensation":
                            entry.update(compensation_hits=int(recovery.ge(0.5).sum()), compensation_hit_rate=float(recovery.ge(0.5).mean()))
                        rows.append(entry)
    columns = ["protein_id", "family_group", "pair_id", "task", "margin", "unit", "scoring", "budget", "pool_size",
               "selected_variant_ids", "selected_scores", "precision_lower", "precision_upper", "selected_relative_point", "selected_recovery",
               "selected_mean_stability", "selected_best_stability", "selected_mean_recovery", "selected_best_recovery", "compensation_hits", "compensation_hit_rate"]
    return pd.DataFrame(rows, columns=columns)
