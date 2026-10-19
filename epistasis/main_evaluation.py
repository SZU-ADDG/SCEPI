import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from epistasis.compensation import PREDICTION_COLUMNS, SEEDS, check_family, definitions as compensation_definitions
from epistasis.metrics import correlation, per_protein_metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/original_main"
_SOURCE_SELECTION_CACHE = {}
_FUNCTIONAL_TASK_CACHE = {}
CONDITIONS = ["Z", "S32", "S128", "Sall"]
COMBINATION_CONDITIONS = ["C128", "C512", "C2048", "Call"]
SCHEMES = ["unseen_pair", "random", "two_pair_secondary"]


@lru_cache(maxsize=1)
def stability_definitions():
    result = {}
    for arm, item in compensation_definitions().items():
        if item["kind"] in ["pair", "esm_scores"]:
            result[arm] = {**item, "conditions": CONDITIONS, "scope": "family"}
        elif item["conditions"] == ["Sall"]:
            result[arm] = {**item, "conditions": CONDITIONS if arm != "kermut" else CONDITIONS[1:],
                           "scope": "family", "template": item["template"].replace("/Sall", "/{condition}")}
        else:
            result[arm] = {**item, "scope": "combination",
                           "template": item["template"].replace("/unseen_pair/", "/{scheme}/")}
    original = {
        "original_interaction": "full_g0_f0_j0", "original_remove_interaction": "full_g0_f0_j0",
        "original_no_cross": "no_cross_g0_f0_j0", "original_reference_context": "wt_g0_f0_j0",
        "original_identity_features": "identity_g0_f0_j0", "original_residual_only": "full_g0_f0_j0_residual",
        "original_phenotype_only": "full_g0_f0_j0_phenotype", "original_geometry": "full_g1_f0_j0",
        "original_geometry_permuted_retrained": "full_g1_f0_j0_geopermuted",
        "original_geometry_rewired_retrained": "full_g1_f0_j0_georewired",
        "original_geometry_permuted_inference": "full_g1_f0_j0",
        "original_geometry_rewired_inference": "full_g1_f0_j0",
    }
    for arm, tag in original.items():
        result[arm] = {"kind": "pair", "tag": tag, "scope": "family", "conditions": CONDITIONS,
                       "thresholds": [30, 25], "requires_structure": "geometry" in arm,
                       "prediction_method": "remove_interaction" if arm.endswith("remove_interaction") else "shared"}
    for arm, root, remove in [
        ("interaction_C_r0", "shared_target", True),
        ("interaction_target_only_C_r0", "shared_target_only", True),
        ("original_geometry_C", "shared_target_geometry", False),
        ("original_geometry_target_only_C", "shared_target_only_geometry", False),
    ]:
        result[arm] = {"kind": "target", "scope": "combination", "conditions": COMBINATION_CONDITIONS,
                       "thresholds": [30], "requires_structure": "geometry" in arm, "remove_identity_interaction": remove,
                       "template": "runs/" + root + "/{protein}/seed{seed}/{scheme}/{condition}",
                       "prediction_file": "predictions.parquet"}
    return result


def functional_definitions(dataset):
    result = {}
    for supervision in ["transfer", "target_only"]:
        for method in ["shared", "remove_interaction", "independent_no_r"]:
            arm = supervision + "_" + method
            result[arm] = {"template": "runs/functional_adaptation/{dataset}/" + supervision + "/{protein}/seed{seed}/{scheme}/{condition}",
                           "subdirectory": method, "permissions": "shared"}
    for method, orders in [("mochi", [1, 2]), ("mavenn", [1, 2]), ("rfa", [2])]:
        for order in orders:
            result[method + f"_order{order}"] = {"template": "runs/" + method + "/{dataset}/order" + str(order) + "/{protein}/seed{seed}/{scheme}/{condition}",
                                                "subdirectory": "", "permissions": "local"}
    for method in ["mean_ridge", "mean_mlp"]:
        if dataset == "flip_primary":
            result[method] = {"template": "runs/mean_flip/" + method + "/{scheme}/{condition}/seed{seed}",
                              "subdirectory": "", "permissions": "shared", "global_predictions": True}
        else:
            result[method] = {"template": "runs/" + method + "/{dataset}/order2/{protein}/seed{seed}/{scheme}/{condition}",
                              "subdirectory": "", "permissions": "local"}
    return result


def evaluation_directory(job):
    return OUTPUT / "evaluations" / job["dataset"] / job["arm"] / f"family{job['threshold']}" / f"seed{job['seed']}" / job["scheme"] / f"fold{job['fold']}" / job["condition"]


def selected_source(root, pattern, threshold, fold, seed):
    key = f"family{threshold}/fold{fold}"
    expressions = tuple([pattern] if isinstance(pattern, str) else pattern)
    cache_key = (root, expressions, threshold, fold, seed)
    if cache_key in _SOURCE_SELECTION_CACHE:
        return _SOURCE_SELECTION_CACHE[cache_key]
    choices = []
    for expression in expressions:
        for path in (ROOT / root).glob(expression):
            selected = json.loads(path.read_text())
            if key in selected:
                choices.append(selected[key]["run"])
    if not choices:
        return None
    assert len(set(choices)) == 1, (key, choices)
    source = ROOT / choices[0]
    selected_source_path = source.with_name(source.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
    _SOURCE_SELECTION_CACHE[cache_key] = selected_source_path
    return selected_source_path


def stability_entries(job, proteins):
    item = stability_definitions()[job["arm"]]
    condition, threshold, fold, seed = (job[key] for key in ["condition", "threshold", "fold", "seed"])
    source = None
    if item["kind"] == "pair":
        source = selected_source("runs/e3", ["pair_repeats_" + item["tag"] + "/selected_hyperparameters.json",
                                            "pair_repeats_" + item["tag"] + f"_family{threshold}/selected_hyperparameters.json"], threshold, fold, seed)
        suffix = ""
        if job["arm"].endswith("_inference"):
            suffix = "_permuted" if "permuted" in job["arm"] else "_rewired"
        if source is None:
            return [], [f"Source-validation selection {item['tag']}/family{threshold}/fold{fold}"]
        directory = source / ("evaluation_test" + suffix)
        filename = item["prediction_method"] + "_" + condition + ".parquet"
    elif item["kind"] == "esm_scores":
        directory = ROOT / "runs/e3" / f"family{threshold}/fold{fold}/esm_scores/seed{seed}"
        source = directory
        filename = "test_" + job["arm"] + "_" + condition + ".parquet"
    elif condition == "Z":
        if job["arm"].startswith("mean_"):
            source = selected_source("runs/mean_family", "repeats_" + job["arm"] + "_*/selected_hyperparameters.json", threshold, fold, seed)
        elif job["arm"].startswith("thermo_"):
            mode = "additive" if job["arm"] == "thermo_additive" else "epistatic"
            source = selected_source("runs/thermo_family", "repeats_" + mode + "_*/selected_hyperparameters.json", threshold, fold, seed)
        else:
            assert job["arm"] == "esm_msr"
            source = selected_source("runs/msr_family", "repeats_*/selected_hyperparameters.json", threshold, fold, seed)
        if source is None:
            return [], [f"Source-validation selection {job['arm']}/family{threshold}/fold{fold}"]
        directory, filename = source / "evaluation_test", "predictions.parquet"
    else:
        entries = []
        for protein in proteins:
            directory = ROOT / item["template"].format(protein=protein, **job)
            entries.append({"manifest": directory / "manifest.json", "prediction": directory / item["prediction_file"],
                            "source": None, "protein": protein})
        return inspect_entries(entries)
    return inspect_entries([{"manifest": directory / "manifest.json", "prediction": directory / filename,
                             "source": source, "protein": None}])


def inspect_entries(entries):
    missing = []
    for entry in entries:
        path = entry["manifest"]
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        information = json.loads(path.read_text())
        assert information["complete"] and not information.get("smoke", False), path
        entry["information"] = information
        if information.get("evaluated", True):
            assert entry["prediction"].exists(), entry["prediction"]
    return entries, missing


def frozen_truth(job):
    if job["dataset"] != "tsuboyama":
        return pd.read_parquet(OUTPUT / "functional_truth" / (job["dataset"] + ".parquet"))
    path = OUTPUT / "truth" / f"family{job['threshold']}" / f"fold{job['fold']}.parquet"
    truth = pd.read_parquet(path)
    item = stability_definitions()[job["arm"]]
    if item["requires_structure"]:
        truth = truth.loc[truth.structure_available].copy()
    if job["scheme"] != "family":
        split = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet", columns=["variant_id", "seed", "scheme", "set"])
        ids = split.loc[split.seed.eq(job["seed"]) & split.scheme.eq(job["scheme"]) & split["set"].eq("test"), "variant_id"]
        truth = truth.loc[truth.variant_id.isin(ids)].copy()
    return truth


def normalized_stability(job, truth, entries):
    frames, excluded, provenance = [], [], []
    specification = stability_definitions()[job["arm"]]
    target_tasks = stability_tasks(job, truth) if specification["kind"] == "target" and job["condition"] != "Z" else {}
    for entry in entries:
        information = entry["information"]
        if not information.get("evaluated", True):
            excluded.append({"protein_id": entry["protein"], "reason": information["reason"]})
            continue
        expected = truth if entry["protein"] is None else truth.loc[truth.protein_id.eq(entry["protein"])]
        ledgers = verify_target_permissions(entry["manifest"].parent, [target_tasks[entry["protein"]]]) if target_tasks else []
        if entry["source"] is not None:
            check_family(entry["source"], job["threshold"], job["fold"], job["seed"], expected)
        for field in ["source_manifest", "source_run"]:
            if field in information:
                source = ROOT / information[field]
                source = source.parent if field == "source_manifest" else source
                check_family(source, job["threshold"], job["fold"], job["seed"], expected)
                if specification.get("remove_identity_interaction", False):
                    assert not json.loads((source / "manifest.json").read_text())["config"]["flexible"]
        original = pd.read_parquet(entry["prediction"])
        assert original.variant_id.is_unique
        assert set(expected.variant_id) <= set(original.variant_id), entry["prediction"]
        if "budget_eligible" in original:
            unavailable = original.loc[~original.budget_eligible, "protein_id"].unique()
            expected = expected.loc[~expected.protein_id.isin(unavailable)]
            excluded.extend({"protein_id": protein, "reason": "single_label_budget_unavailable"} for protein in unavailable)
        predictions = original.loc[original.variant_id.isin(expected.variant_id), PREDICTION_COLUMNS].copy()
        if "set" in original:
            partition = "family_pool" if job["arm"] == "kermut" and job["scheme"] == "family" else "test"
            assert original.loc[original.variant_id.isin(expected.variant_id), "set"].eq(partition).all(), entry["prediction"]
        assert np.isfinite(predictions[PREDICTION_COLUMNS[1:]].to_numpy()).all()
        np.testing.assert_allclose(predictions.prediction_epsilon,
                                   predictions.prediction_mu - predictions.prediction_a - predictions.prediction_b + predictions.prediction_wt,
                                   rtol=1e-5, atol=1e-5)
        if specification.get("remove_identity_interaction", False):
            predictions["prediction_mu"] = predictions.prediction_a + predictions.prediction_b - predictions.prediction_wt
            predictions["prediction_epsilon"] = 0.0
        frames.append(expected.merge(predictions, on="variant_id", validate="one_to_one"))
        provenance.append({"manifest": str(entry["manifest"].relative_to(ROOT)), "predictions": str(entry["prediction"].relative_to(ROOT)),
                           "source_commit": information["source_commit"],
                           "label_ledgers": ledgers,
                           "label_usage": {key: information[key] for key in ["target_single_labels", "target_combination_labels",
                               "target_test_labels_used", "source_labels_used", "supervised_source_weights_used"] if key in information}})
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(truth.columns) + PREDICTION_COLUMNS[1:])
    assert frame.variant_id.is_unique
    permitted = truth.loc[~truth.protein_id.isin([item["protein_id"] for item in excluded])]
    assert set(frame.variant_id) == set(permitted.variant_id)
    return frame, excluded, provenance


def stability_tasks(job, truth):
    if job["scheme"] != "family":
        table = pd.read_parquet(OUTPUT / "tsuboyama_C_tasks.parquet", filters=[("seed", "==", job["seed"]),
                                ("scheme", "==", job["scheme"]), ("condition", "==", job["condition"]),
                                ("protein_id", "in", sorted(truth.protein_id.unique()))])
        return {task["protein_id"]: task for task in table.to_dict("records")}
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet", columns=["variant_id", "protein_id", "rank"],
                            filters=[("seed", "==", job["seed"]), ("protein_id", "in", sorted(truth.protein_id.unique()))])
    tasks = {}
    for protein, group in truth.groupby("protein_id", sort=True):
        labels = order.loc[order.protein_id.eq(protein)].sort_values("rank")
        if job["condition"] != "Sall":
            labels = labels.head(int(job["condition"][1:]))
        references = group.wt_variant_id.unique().tolist()
        tasks[protein] = {"train_ids": references + labels.variant_id.tolist(), "validation_ids": [], "test_ids": group.variant_id.tolist()}
    return tasks


def stability_metrics(frame):
    if not len(frame):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = per_protein_metrics(frame)
    annotation = frame[["protein_id", "family_group", "pfam_annotated", "origin"]].drop_duplicates()
    summary = summary.merge(annotation, on="protein_id", validate="one_to_one")
    strata, selections = [], []
    for protein, group in frame.groupby("protein_id", sort=True):
        partitions = [("measurement", "main", group.loc[group.main.fillna(False)]),
                      ("measurement", "all_point", group.loc[group.epsilon.notna()]),
                      ("measurement", "margin025", group.loc[group.main.fillna(False) & group.margin.ge(0.25)]),
                      ("measurement", "margin050", group.loc[group.main.fillna(False) & group.margin.ge(0.5)])]
        point = group.loc[group.main.fillna(False)]
        for field in ["distance_stratum", "sequence_gap_stratum", "precision_stratum", "reference_stratum", "additive_stratum", "origin"]:
            partitions.extend((field, str(value), subset) for value, subset in point.groupby(field, observed=True, sort=True))
        for field, value, subset in partitions:
            squared = (subset.prediction_epsilon - subset.epsilon) ** 2
            strata.append({"protein_id": protein, "family_group": group.family_group.iloc[0],
                           "pfam_annotated": bool(group.pfam_annotated.iloc[0]), "stratum": field, "stratum_value": value,
                           "quartets": len(subset), "main_eligible": len(subset) >= 30,
                           "epsilon_rmse": float(np.sqrt(squared.mean())) if len(subset) else np.nan,
                           "pair_macro_epsilon_rmse": float(squared.groupby(subset.pair_id).mean().pow(0.5).mean()) if len(subset) else np.nan,
                           "epsilon_spearman": correlation(subset.epsilon, subset.prediction_epsilon, "spearman")})
        pool = group.loc[group.wt_all_point & group.wt_point_in_range & group.wt_point.le(4.5)]
        if len(pool) >= 100:
            ranked = pool.sort_values(["prediction_mu", "mutation_key"], ascending=[False, True], kind="stable")
            for budget in [10, 30]:
                chosen = ranked.head(budget)
                selections.append({"protein_id": protein, "family_group": group.family_group.iloc[0],
                                   "budget": budget, "pool_size": len(pool), "variant_ids": chosen.variant_id.tolist(),
                                   "precision_lower": float(chosen.relative_lower.ge(0.5).mean()),
                                   "precision_upper": float(chosen.relative_upper.ge(0.5).mean()),
                                   "possible_beneficial_count": int(pool.relative_upper.ge(0.5).sum())})
    return summary, pd.DataFrame(strata), pd.DataFrame(selections)


def functional_tasks(job, permissions):
    key = (job["dataset"], permissions, job["seed"], job["scheme"], job["condition"])
    if key not in _FUNCTIONAL_TASK_CACHE:
        tasks = pd.read_parquet(OUTPUT / "functional_truth" / (job["dataset"] + "_" + permissions + "_tasks.parquet"),
                                filters=[("seed", "==", job["seed"]), ("scheme", "==", job["scheme"]), ("condition", "==", job["condition"])])
        _FUNCTIONAL_TASK_CACHE[key] = tuple(tasks.to_dict("records"))
    return [dict(task) for task in _FUNCTIONAL_TASK_CACHE[key]]


def functional_entries(job):
    specification = functional_definitions(job["dataset"])[job["arm"]]
    tasks = functional_tasks(job, specification["permissions"])
    assert tasks
    entries = []
    for task in tasks:
        directory = ROOT / specification["template"].format(protein=task["protein_id"], **job)
        if job["arm"] == "rfa_order2" and not (directory / "manifest.json").is_file():
            accelerated = ROOT / "runs/rfa_accelerated" / directory.relative_to(ROOT / "runs/rfa")
            if (accelerated / "manifest.json").is_file():
                information = json.loads((accelerated / "manifest.json").read_text())
                validation = json.loads((ROOT / information["acceleration_validation"]).read_text())
                assert information["complete"] and information["accelerated"] and validation["complete"]
                directory = accelerated
            else:
                bounded = ROOT / "runs/rfa_bounded" / directory.relative_to(ROOT / "runs/rfa")
                if (bounded / "manifest.json").is_file():
                    information = json.loads((bounded / "manifest.json").read_text())
                    validation = json.loads((ROOT / information["budget_validation"]).read_text())
                    assert information["complete"] and information["bounded_optimization"] and validation["complete"]
                    directory = bounded
                else:
                    fixed = ROOT / "runs/rfa_fixed" / directory.relative_to(ROOT / "runs/rfa")
                    if (fixed / "manifest.json").is_file():
                        information = json.loads((fixed / "manifest.json").read_text())
                        validation = json.loads((ROOT / information["budget_validation"]).read_text())
                        assert information["complete"] and information["fixed_default"] and validation["complete"]
                        directory = fixed
        prediction_directory = directory / specification["subdirectory"]
        entries.append({"manifest": directory / "manifest.json", "prediction": prediction_directory / "predictions.parquet",
                        "prediction_directory": prediction_directory, "source": None,
                        "protein": None if specification.get("global_predictions", False) else task["protein_id"], "tasks": [task]})
    if specification.get("global_predictions", False):
        assert len({str(entry["manifest"]) for entry in entries}) == 1
        entries = [dict(entries[0], tasks=tasks)]
    return inspect_entries(entries)


def verify_target_permissions(directory, tasks):
    permitted = set().union(*(set(task["train_ids"]) for task in tasks))
    validation = set().union(*(set(task["validation_ids"]) for task in tasks))
    testing = set().union(*(set(task["test_ids"]) for task in tasks))
    assert not permitted & (validation | testing)
    ledgers = []
    for filename in ["training_permissions.parquet", "training_observations.parquet", "training_variants.parquet", "allowed_labels.parquet", "label_permissions.parquet", "permissions.parquet"]:
        path = directory / filename
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        if "role" in table:
            selected = table.loc[table.role.eq("train")]
            assert set(table.loc[table.role.eq("validation"), "variant_id"]) <= validation
        elif "partition" in table:
            selected = table.loc[table.partition.eq("train")]
            assert set(table.loc[table.partition.eq("validation"), "variant_id"]) <= validation
            assert set(table.loc[table.partition.eq("test"), "variant_id"]) <= testing
        else:
            selected = table
        assert set(selected.variant_id) <= permitted, path
        assert not set(selected.variant_id) & testing
        ledgers.append(str(path.relative_to(ROOT)))
    assert ledgers, directory
    return ledgers


def functional_metrics(job, variants, entries):
    quartets = pd.read_parquet(OUTPUT / "functional_truth" / (job["dataset"] + "_quartets.parquet"))
    indexed = variants.set_index("variant_id")
    results, excluded, provenance = [], [], []
    for entry in entries:
        information = entry["information"]
        tasks = entry["tasks"]
        if not information.get("evaluated", True):
            excluded.extend({"protein_id": task["protein_id"], "reason": information["reason"]} for task in tasks)
            continue
        ledgers = verify_target_permissions(entry["manifest"].parent, tasks)
        if "source_checkpoints" in information:
            permissions = pd.read_parquet(ROOT / "data/processed/external_transfer/source_permissions.parquet")
            related = permissions.loc[permissions.threshold.eq(30) & permissions.dataset.eq(job["dataset"])
                                      & permissions.protein_id.isin([task["protein_id"] for task in tasks]), "related_source_proteins"]
            prohibited = set().union(*(set(values) for values in related))
            for checkpoint in information["source_checkpoints"]:
                permission = json.loads((ROOT / checkpoint).with_name("permissions.json").read_text())
                assert not prohibited & (set(permission["train"]) | set(permission["validation"]))
        prediction = pd.read_parquet(entry["prediction"])
        assert prediction.variant_id.is_unique
        expected = set().union(*(set(task["test_ids"]) for task in tasks))
        assert expected <= set(prediction.variant_id)
        assert prediction.loc[prediction.variant_id.isin(expected), "set"].eq("test").all()
        observed = prediction.set_index("variant_id")
        for task in tasks:
            protein = task["protein_id"]
            testing = indexed.loc[task["test_ids"]].copy()
            assert testing.protein_id.eq(protein).all()
            testing["prediction_absolute"] = observed.prediction_absolute.reindex(testing.index)
            assert np.isfinite(testing.prediction_absolute).all()
            training = indexed.loc[task["train_ids"]].point.dropna()
            scale = float(training.std(ddof=0))
            assert len(training) >= 2 and scale > 0
            identifiable = information.get("absolute_prediction_identifiable", True)
            for order, selected in [("all", testing)] + [(str(value), testing.loc[testing.n_mutations.eq(value)]) for value in [1, 2, 3, 4]] + [("5plus", testing.loc[testing.n_mutations.ge(5)])]:
                point = selected.loc[selected.point.notna()]
                error = float(np.sqrt(np.mean((point.prediction_absolute - point.point) ** 2))) if len(point) and identifiable else np.nan
                results.append({"protein_id": protein, "family_group": protein, "pfam_annotated": False,
                                "mutation_order": order, "variants": len(selected), "point_variants": len(point),
                                "phenotype_spearman": correlation(point.point, point.prediction_absolute, "spearman"),
                                "phenotype_rmse": error, "phenotype_rmse_training_scale": error / scale,
                                "training_scale": scale, "absolute_identifiable": identifiable,
                                "training_variants": len(task["train_ids"]), "validation_variants": len(task["validation_ids"])})
            current = quartets.loc[quartets.protein_id.eq(protein) & quartets.ab_id.isin(task["test_ids"])].copy()
            if len(current):
                directory = entry["prediction_directory"]
                if (directory / "predictions.npz").exists():
                    metadata = pd.read_parquet(ROOT / "data/features/higher" / job["dataset"] / protein / "variants.parquet", columns=["variant_id"])
                    arrays = np.load(directory / "predictions.npz")
                    assert len(metadata) == len(arrays["variants"])
                    values = pd.Series(arrays["variants"], index=metadata.variant_id)
                    components = np.column_stack([current[field].map(values) for field in ["ab_id", "a_id", "b_id", "wt_id"]])
                    assert np.isfinite(components).all()
                    current["prediction_epsilon"] = components[:, 0] - components[:, 1] - components[:, 2] + components[:, 3]
                else:
                    assert job["arm"] in ["mean_ridge", "mean_mlp"] and job["dataset"] == "flip_primary"
                    returned = pd.read_parquet(directory / "quartet_predictions.parquet").set_index("variant_id")
                    assert set(current.ab_id) <= set(returned.index)
                    values = returned.loc[current.ab_id]
                    np.testing.assert_allclose(values.epsilon, current.epsilon, atol=1e-8)
                    np.testing.assert_allclose(values.prediction_epsilon, values.prediction_ab - values.prediction_a - values.prediction_b + values.prediction_wt, atol=1e-5)
                    current["prediction_epsilon"] = values.prediction_epsilon.to_numpy()
                squared = (current.prediction_epsilon - current.epsilon) ** 2
                residual = float(np.sqrt(squared.mean()))
                pair_error = float(squared.groupby(current.pair_id).mean().pow(0.5).mean())
            else:
                residual, pair_error = np.nan, np.nan
            results.append({"protein_id": protein, "family_group": protein, "pfam_annotated": False,
                            "mutation_order": "quartet", "main_quartets": len(current), "main_eligible": len(current) >= 30,
                            "epsilon_rmse": residual, "pair_macro_epsilon_rmse": pair_error,
                            "epsilon_rmse_training_scale": residual / scale,
                            "epsilon_spearman": correlation(current.epsilon, current.prediction_epsilon, "spearman") if len(current) else np.nan,
                            "training_scale": scale})
        provenance.append({"manifest": str(entry["manifest"].relative_to(ROOT)), "predictions": str(entry["prediction"].relative_to(ROOT)),
                           "source_commit": information["source_commit"], "label_ledgers": ledgers,
                           "training_point_variants": information.get("point_training_variants", information.get("training_variants")),
                           "supervision": information.get("supervision", "target_only")})
    return pd.DataFrame(results), excluded, provenance
