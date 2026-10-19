import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from train_single import ROOT
from epistasis.metrics import correlation
from epistasis.model import ObservationModel, PhenotypeMap


def optimize(x, y, validation_x, validation_y, learning_rate, epochs, seed, select):
    torch.manual_seed(seed)
    mapping, noise = PhenotypeMap(True), ObservationModel()
    optimizer = torch.optim.AdamW([*mapping.parameters(), *noise.parameters()], lr=learning_rate, weight_decay=1e-4)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    vx, vy = torch.tensor(validation_x, dtype=torch.float32), torch.tensor(validation_y, dtype=torch.float32)
    history, best, best_epoch, state = [], float("inf"), -1, None
    for epoch in range(epochs):
        predicted = mapping(x)
        loss = noise.nll(predicted, y, y, torch.ones(len(y), dtype=torch.bool)).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(torch.isfinite(value.grad).all() for value in [*mapping.parameters(), *noise.parameters()])
        optimizer.step()
        with torch.no_grad():
            value = float(noise.nll(mapping(vx), vy, vy, torch.ones(len(vy), dtype=torch.bool)).mean()) if select else float(loss)
        history.append({"epoch": epoch, "training_nll": float(loss.detach()), "validation_nll": value if select else None})
        if not select or value < best:
            best, best_epoch = value, epoch
            state = (deepcopy(mapping.state_dict()), deepcopy(noise.state_dict()))
        if select and epoch - best_epoch >= 10:
            break
    mapping.load_state_dict(state[0])
    return mapping, {"best_epoch": best_epoch, "validation_nll": best if select else None, "history": history,
                     "noise": state[1], "learning_rate": learning_rate}


def calibrate(allowed, raw, reference, directory, seed):
    validation_ids = set(allowed.iloc[::5].variant_id)
    training_ids = set(allowed.variant_id) - validation_ids
    assert training_ids and validation_ids
    training = raw.loc[raw.variant_id.isin(training_ids) | raw.n_mutations.eq(0)]
    validation = raw.loc[raw.variant_id.isin(validation_ids)]
    assert len(training) and len(validation)
    scale = float(allowed.loc[allowed.variant_id.isin(training_ids), "point"].std(ddof=0))
    assert np.isfinite(scale) and scale > 0
    trials = []
    for learning_rate in [1e-4, 3e-4, 1e-3]:
        _, details = optimize(training.latent.to_numpy(), (training.point.to_numpy() - reference) / scale,
                              validation.latent.to_numpy(), (validation.point.to_numpy() - reference) / scale,
                              learning_rate, 100, seed, True)
        trials.append(details)
    selected = min(trials, key=lambda trial: (trial["validation_nll"], trial["learning_rate"]))
    full_scale = float(allowed.point.std(ddof=0))
    assert full_scale > 0 and np.isfinite(full_scale)
    mapping, details = optimize(raw.latent.to_numpy(), (raw.point.to_numpy() - reference) / full_scale,
                                np.array([], dtype=np.float32), np.array([], dtype=np.float32),
                                selected["learning_rate"], selected["best_epoch"] + 1, seed, False)
    torch.save({"mapping": mapping.state_dict(), "noise": details["noise"], "scale": full_scale,
                "reference": reference, "seed": seed}, directory / "mapping.pt")
    pd.concat([pd.DataFrame(trial["history"]).assign(learning_rate=trial["learning_rate"]) for trial in trials]).to_csv(directory / "calibration_trials.csv", index=False)
    pd.DataFrame(details["history"]).to_csv(directory / "refit_history.csv", index=False)
    return mapping, full_scale, {"selected_learning_rate": selected["learning_rate"], "selected_epochs": selected["best_epoch"] + 1,
                                 "validation_single_labels": len(validation_ids), "scale": full_scale,
                                 "selection": "Held-out permitted single-observation Gaussian likelihood; refit all permitted observations for the selected epoch count."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    args = parser.parse_args()
    torch.set_num_threads(4)
    proteins = pd.read_parquet(ROOT / "data/processed/functional_doubles/proteins.parquet").sort_values("protein_id")
    variants = pd.read_parquet(ROOT / "data/processed/functional_doubles/variants.parquet")
    observations = pd.read_parquet(ROOT / "data/processed/functional/observations.parquet")
    quartets = pd.read_parquet(ROOT / "data/processed/functional/quartets.parquet")
    output = ROOT / "runs/functional_transfer"
    output.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifests = []
    for protein in proteins.iloc[args.rank::args.world_size].itertuples():
        table = variants.loc[variants.protein_id.eq(protein.protein_id)].copy()
        lookup = table.set_index("variant_id")
        for seed in args.seeds:
            cache = ROOT / "data/features/transferred_effects/functional_doubles" / protein.protein_id / f"seed{seed}"
            assert json.loads((cache / "manifest.json").read_text())["complete"]
            cached = pd.read_parquet(cache / "variants.parquet")
            np.testing.assert_array_equal(cached.variant_id, table.variant_id)
            terms = np.load(cache / "terms.npz")
            table["latent"] = terms["latent"]
            table["additive"] = terms["additive"]
            available = table.loc[table.n_mutations.eq(1) & table.all_point & table.point.notna()].sort_values("mutation_key")
            available = available.iloc[np.random.default_rng(seed).permutation(len(available))]
            wt = table.loc[table.n_mutations.eq(0) & table.all_point & table.point.notna()]
            for condition, budget in [("Z", 0), ("S32", 32), ("S128", 128), ("Sall", -1)]:
                directory = output / protein.protein_id / f"seed{seed}" / condition
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "manifest.json"
                if path.exists():
                    assert json.loads(path.read_text())["complete"]
                    manifests.append(str(path.relative_to(ROOT)))
                    continue
                allowed = available.iloc[:budget] if budget > 0 else (available if budget < 0 else available.iloc[:0])
                manifest = {"complete": True, "protein_id": protein.protein_id, "assay_id": protein.assay_id,
                            "condition_id": protein.condition_id, "condition": condition, "seed": seed,
                            "source_commit": source_commit, "physical_functions_frozen": True,
                            "source_effects": str((cache / "manifest.json").relative_to(ROOT)),
                            "target_single_labels": len(allowed), "target_combination_labels": 0,
                            "target_reference_variants": int(len(wt)) if budget != 0 else 0,
                            "budget_eligible": budget < 0 or len(available) >= budget,
                            "target_units": protein.units, "available_singles": len(available)}
                if budget and (len(wt) != 1 or len(allowed) < 5 or allowed.point.nunique() < 2):
                    manifest.update({"evaluated": False, "reason": "reference_or_calibration_single_labels_unavailable"})
                    path.write_text(json.dumps(manifest, indent=2))
                    manifests.append(str(path.relative_to(ROOT)))
                    continue
                allowed_ids = set(allowed.variant_id) | (set(wt.variant_id) if budget else set())
                raw = observations.loc[observations.variant_id.isin(allowed_ids) & observations.point.notna()].copy()
                assert raw.protein_id.eq(protein.protein_id).all()
                raw["latent"] = raw.variant_id.map(table.set_index("variant_id").latent)
                raw[["observation_id", "variant_id", "protein_id", "point", "n_mutations"]].to_parquet(directory / "training_permissions.parquet", index=False)
                if budget:
                    mapping, scale, details = calibrate(allowed, raw, float(wt.point.iloc[0]), directory, seed)
                    transform = lambda value: mapping(torch.tensor(value, dtype=torch.float32)).detach().numpy() * scale
                    manifest.update(details)
                    manifest["prediction_units"] = protein.units
                else:
                    transform = lambda value: np.asarray(value)
                    manifest["prediction_units"] = "Transferred stability latent scale; phenotype ranking only."
                records, metrics = [], []
                for method, latent in [("shared", table.latent.to_numpy()), ("remove_interaction", table.additive.to_numpy())]:
                    prediction = table[["variant_id", "protein_id", "mutation_key", "n_mutations", "point"]].copy()
                    prediction["prediction_relative"] = transform(latent)
                    prediction["method"] = method
                    records.append(prediction)
                    evaluated = prediction.loc[prediction.n_mutations.eq(2)]
                    entry = {"method": method, "doubles": len(evaluated),
                             "phenotype_spearman": correlation(evaluated.point, evaluated.prediction_relative, "spearman")}
                    if budget:
                        values = pd.Series(prediction.prediction_relative.to_numpy(), index=prediction.variant_id)
                        units = quartets.loc[quartets.protein_id.eq(protein.protein_id) & quartets.ab_id.isin(values.index)].copy()
                        units["prediction_epsilon"] = (units.ab_id.map(values) - units.a_id.map(values) -
                                                       units.b_id.map(values) + units.wt_id.map(values))
                        assert units.prediction_epsilon.notna().all()
                        units["method"] = method
                        units.to_parquet(directory / f"quartets_{method}.parquet", index=False)
                        entry["complete_quartets"] = len(units)
                        entry["epsilon_rmse"] = float(np.sqrt(np.mean((units.prediction_epsilon - units.epsilon) ** 2))) if len(units) else None
                        entry["epsilon_spearman"] = correlation(units.epsilon, units.prediction_epsilon, "spearman")
                    metrics.append(entry)
                pd.concat(records, ignore_index=True).to_parquet(directory / "predictions.parquet", index=False)
                pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
                manifest.update({"evaluated": True, "raw_training_observations": len(raw), "metrics": metrics})
                path.write_text(json.dumps(manifest, indent=2))
                manifests.append(str(path.relative_to(ROOT)))
                print(json.dumps({"protein": protein.protein_id, "seed": seed, "condition": condition, "complete": True}), flush=True)
    (output / ("manifest_seeds" + "_".join(str(value) for value in args.seeds) + f"_rank{args.rank}.json")).write_text(json.dumps({
        "complete": True, "config": vars(args), "outputs": manifests, "source_commit": source_commit}, indent=2))


if __name__ == "__main__":
    main()
