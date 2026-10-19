import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from evaluate_family import load_model
from run_mochi_targets import tasks_for, save_results
from train_single import ROOT
from epistasis.functional import FunctionalData, FunctionalModel, fit_stage
from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect


def jobs_for(dataset):
    if dataset == "functional_doubles":
        return tasks_for(dataset, "C")
    split = pd.read_parquet(ROOT / "data/processed/flip_primary/splits.parquet")
    jobs = []
    for source_file, partition in split.groupby("source_file", sort=True):
        equivalent = partition.effective_set.eq(partition.strict_set).all()
        protocols = [("official", "effective_set")] + ([] if equivalent else [("strict", "strict_set")])
        for protocol, column in protocols:
            for protein in sorted(partition.loc[partition[column].eq("test"), "protein_id"].unique()):
                for seed in [11, 23, 47, 89, 131]:
                    jobs.append({"protein_id": protein, "seed": seed, "scheme": Path(source_file).parent.name + "_" + partition.official_split.iloc[0],
                                 "condition": protocol, "budget_eligible": True,
                                 "protocol_equivalence": ["official", "strict"] if equivalent else [protocol],
                                 "train_ids": partition.loc[partition[column].eq("train"), "variant_id"].tolist(),
                                 "validation_ids": partition.loc[partition[column].eq("validation"), "variant_id"].tolist(),
                                 "test_ids": partition.loc[partition[column].eq("test") & partition.protein_id.eq(protein), "variant_id"].tolist()})
    return jobs


def original_observations(dataset):
    if dataset == "functional_doubles":
        return pd.read_parquet(ROOT / "data/processed/functional/observations.parquet")
    root = ROOT / "data/processed/flip_primary"
    variants = pd.read_parquet(root / "variants.parquet")
    records = []
    for name in ["ired", "hydro"]:
        source = pd.read_parquet(root / f"original_observations_{name}.parquet")
        source = source[["variant_id", "protein_id", "fitness", "source_row"]].copy()
        source["observation_id"] = name + "_" + source.source_row.astype(str)
        source["point"] = source.fitness
        source["lower"], source["upper"], source["measurement_status"] = source.point, source.point, "point"
        source["source_file"] = str((root / f"original_observations_{name}.parquet").relative_to(ROOT))
        records.append(source)
    official = variants.loc[~variants.dataset.isin(["ired", "hydro"])].copy()
    official["observation_id"] = "official_" + official.variant_id
    official["measurement_status"] = "point"
    official["source_file"] = "data/processed/flip_primary/variants.parquet"
    records.append(official)
    result = pd.concat(records, ignore_index=True)
    assert result.observation_id.is_unique
    return result


def source_runs(dataset, protein, seed):
    permission = pd.read_parquet(ROOT / "data/processed/external_transfer/source_permissions.parquet")
    permission = permission.loc[permission.threshold.eq(30) & permission.dataset.eq(dataset) & permission.protein_id.eq(protein)]
    assert len(permission) == 1 and not permission.requires_custom_source_exclusion.item()
    row = permission.iloc[0]
    selection = json.loads((ROOT / "runs/e3/pair_repeats_full_g0_f0_j0_family30/selected_hyperparameters.json").read_text())
    runs = []
    for fold in row.eligible_source_test_folds:
        chosen = ROOT / selection[f"family30/fold{fold}"]["run"]
        chosen = chosen.with_name(chosen.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
        information = json.loads((chosen / "permissions.json").read_text())
        assert not set(row.related_source_proteins) & (set(information["train"]) | set(information["validation"]))
        runs.append(chosen)
    return runs


def export(model, data, protein, job, directory, dataset, remove_interaction):
    features = data.features[protein]
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        effects, interactions = [], []
        for source in model.sources:
            if remove_interaction:
                current = torch.cat([source.single(features.batch("single", np.arange(start, min(start + 1024, len(features.single_metadata))), device))
                                     for start in range(0, len(features.single_metadata), 1024)]).cpu().numpy()
                effects.append(current)
            else:
                current, pairs = features.infer_terms(source)
                effects.append(current)
                interactions.append(pairs)
        effects = np.mean(np.stack(effects), axis=0)
        interaction = np.zeros(len(features.pair_metadata), dtype=np.float32) if remove_interaction else np.mean(np.stack(interactions), axis=0)
        additive, pair = features.combine(effects, interaction)
        index = model.protein_index[protein]
        reference = float((model.anchors[index] + model.offset_masks[index] * model.offsets[index]).cpu()) * model.scale + model.center
        def mapped(values):
            return np.concatenate([model.mapping(torch.tensor(values[start:start + 4096], device=device, dtype=torch.float32)).cpu().numpy()
                                   for start in range(0, len(values), 4096)]) * model.scale + reference
        variants, singles = mapped(additive + pair), mapped(effects)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "predictions.npz", variants=variants, reference=reference, singles=singles)
    metrics = save_results(features, job, directory, dataset, absolute_identifiable=model.identifiable[protein])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["functional_doubles", "flip_primary"], required=True)
    parser.add_argument("--supervision", choices=["transfer", "target_only"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    jobs = jobs_for(args.dataset)
    if args.smoke:
        jobs = jobs[::100]
    jobs = jobs[args.rank::args.world_size]
    variants = pd.read_parquet(ROOT / "data/processed" / args.dataset / "variants.parquet")
    observations = original_observations(args.dataset)
    output = ROOT / "runs" / ("functional_adaptation_smoke" if args.smoke else "functional_adaptation") / args.dataset / args.supervision
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    outputs, cached_data, cached_proteins = [], None, None
    for job in jobs:
        directory = output / job["protein_id"] / f"seed{job['seed']}" / job["scheme"] / job["condition"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        if path.exists():
            assert json.loads(path.read_text())["complete"]
            outputs.append(str(path.relative_to(ROOT)))
            continue
        manifest = {"complete": True, "source_commit": commit, "dataset": args.dataset, "supervision": args.supervision,
                    "smoke": args.smoke, **{key: value for key, value in job.items() if not key.endswith("_ids")}}
        allowed = set(job["train_ids"])
        assert not allowed & (set(job["validation_ids"]) | set(job["test_ids"]))
        relevant = allowed | set(job["validation_ids"]) | set(job["test_ids"])
        proteins = tuple(sorted(variants.loc[variants.variant_id.isin(relevant), "protein_id"].unique()))
        if cached_proteins != proteins:
            cached_data = FunctionalData(ROOT, args.dataset, proteins)
            cached_proteins = proteins
        data = cached_data
        train = data.lookup.loc[job["train_ids"]].to_numpy()
        validation = data.lookup.loc[job["validation_ids"]].to_numpy()
        validation = validation[data.metadata.iloc[validation].point.notna().to_numpy()]
        raw = observations.loc[observations.variant_id.isin(allowed)].copy()
        raw.to_parquet(directory / "training_observations.parquet", index=False)
        data.metadata.iloc[validation][["variant_id", "protein_id", "point"]].to_parquet(directory / "validation_permissions.parquet", index=False)
        values = data.metadata.iloc[train].point.dropna().to_numpy()
        if not job["budget_eligible"] or len(values) < 2 or len(validation) < 2 or np.std(values) == 0:
            manifest.update({"evaluated": False, "reason": "budget_unavailable" if not job["budget_eligible"] else "permitted_training_or_validation_unavailable"})
        else:
            started = time.monotonic()
            center, scale = float(np.mean(values)), float(np.std(values))
            quartets = data.quartets(train)
            quartets.to_parquet(directory / "training_quartets.parquet", index=False)
            runs = source_runs(args.dataset, job["protein_id"], job["seed"]) if args.supervision == "transfer" else []
            source_templates = [load_model(run).cpu() for run in runs]
            projections = [None] if runs else [64, 128]
            phase_records, best_main, best_full = [], None, None
            best_main_score = best_full_score = (float("inf"), float("inf"))
            for projection in projections:
                for learning_rate in [1e-4, 3e-4, 1e-3]:
                    torch.manual_seed(job["seed"])
                    if runs:
                        sources = deepcopy(source_templates)
                    else:
                        configuration = ModelConfig(projection=projection)
                        sources = [EpistasisModel(SingleEffect(configuration), PairEffect(configuration), PhenotypeMap(False))]
                    model = FunctionalModel(sources, data, train, center, scale).cuda()
                    main_fit = fit_stage(model, data, raw, validation, quartets, learning_rate, 1.0, job["seed"], "main")
                    name = f"p{projection}_lr{learning_rate}"
                    pd.DataFrame(main_fit["history"]).to_csv(directory / f"{name}_main.csv", index=False)
                    phase_records.append({key: value for key, value in main_fit.items() if key not in ["model", "noise", "history"]} | {"projection": projection})
                    if main_fit["score"] < best_main_score:
                        best_main_score, best_main = main_fit["score"], deepcopy(model).cpu()
                        torch.save({"model": main_fit["model"], "noise": main_fit["noise"], "projection": projection,
                                    "center": center, "scale": scale, "source_runs": [str(run.relative_to(ROOT)) for run in runs]},
                                   directory / "independent_no_r.pt")
                    main_model = deepcopy(model).cpu()
                    del model
                    for lambda_y in [0.1, 1.0]:
                        model = deepcopy(main_model).cuda()
                        pair_fit = fit_stage(model, data, raw, validation, quartets, learning_rate, lambda_y, job["seed"],
                                             "interaction", main_fit["noise"])
                        pd.DataFrame(pair_fit["history"]).to_csv(directory / f"{name}_ly{lambda_y}_interaction.csv", index=False)
                        phase_records.append({key: value for key, value in pair_fit.items() if key not in ["model", "noise", "history"]} | {"projection": projection})
                        if pair_fit["score"] < best_full_score:
                            best_full_score, best_full = pair_fit["score"], deepcopy(model).cpu()
                            selected = {"projection": projection, "learning_rate": learning_rate, "lambda_y": lambda_y,
                                        "main_epoch": main_fit["best_epoch"], "interaction_epoch": pair_fit["best_epoch"],
                                        "validation_score": pair_fit["score"]}
                            torch.save({"model": pair_fit["model"], "noise": pair_fit["noise"], "selected": selected,
                                        "center": center, "scale": scale, "source_runs": [str(run.relative_to(ROOT)) for run in runs]},
                                       directory / "best.pt")
                        print(json.dumps({"protein": job["protein_id"], "phase": "interaction", "configuration": name,
                                          "lambda_y": lambda_y, "complete": True}), flush=True)
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    del main_model
            assert best_full is not None and best_main is not None
            metrics = {}
            for method, model, remove in [("shared", best_full, False), ("remove_interaction", best_full, True), ("independent_no_r", best_main, True)]:
                model.cuda()
                metrics[method] = export(model, data, job["protein_id"], job, directory / method, args.dataset, remove)
                model.cpu()
            pd.DataFrame(phase_records).to_json(directory / "selection.json", orient="records", indent=2)
            manifest.update({"evaluated": True, "selected": selected, "independent_no_r_validation_score": best_main_score,
                             "training_observations": len(raw), "training_variants": len(train), "validation_variants": len(validation),
                             "test_variants": len(job["test_ids"]), "training_complete_quartets": len(quartets),
                             "source_checkpoints": [str((run / "best.pt").relative_to(ROOT)) for run in runs],
                             "source_ensemble": "Equal mean of physical-scale allele and interaction terms across all eligible source folds." if runs else "One model initialized without supervised source weights.",
                             "source_labels_used": "Permitted source-fold training labels" if runs else 0,
                             "target_phenotypes_in_physical_effect_inputs": False, "stages": "Main effect and monotonic map; then freeze both and fit pair effects.",
                             "normalization": {"training_mean": center, "training_std": scale},
                             "absolute_prediction_identifiable": best_full.identifiable[job["protein_id"]],
                             "reference_policy": "Observed training WT anchors are fixed. Missing WT offsets are learned only from allowed training observations; held-out backgrounds without training labels receive relative predictions and no absolute RMSE.",
                             "metrics": metrics, "elapsed_seconds": time.monotonic() - started})
            del best_full, best_main, source_templates
            gc.collect()
            torch.cuda.empty_cache()
        path.write_text(json.dumps(manifest, indent=2))
        outputs.append(str(path.relative_to(ROOT)))
        print(json.dumps({"protein": job["protein_id"], "seed": job["seed"], "scheme": job["scheme"],
                          "condition": job["condition"], "evaluated": manifest["evaluated"]}), flush=True)
    (output / f"manifest_rank{args.rank}.json").write_text(json.dumps({"complete": True, "config": vars(args), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
