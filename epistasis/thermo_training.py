import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epistasis.metrics import per_protein_metrics
from epistasis.thermo import ALPHABET, CachedThermo, new_model

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/features/thermo/tsuboyama"


class ThermoCache:
    def __init__(self, directory, mode, augmentation=False, smoke=False):
        self.mode = mode
        self.directory = directory / ("smoke" if smoke else "full") / mode
        columns = ["variant_id", "protein_id", "n_mutations", "wt_residues", "mut_residues",
                   "native_ddg", "matrix_row", "cache_row"]
        columns += ["source_a_id", "source_b_id"] if augmentation else ["mutation_key", "observed_variant", "training_point_eligible"]
        parts, self.arrays = [], []
        for rank in range(8):
            manifest = json.loads((self.directory / f"manifest_rank{rank}.json").read_text())
            assert manifest["complete"] and manifest["mode"] == mode and manifest["smoke"] == smoke
            table = pd.read_parquet(self.directory / f"metadata_rank{rank}.parquet", columns=columns)
            table["cache_rank"] = rank
            values = np.load(self.directory / f"features_rank{rank}.npy", mmap_mode="r")
            assert len(values) == len(table) == manifest["rows"] and values.dtype == np.float32
            parts.append(table)
            self.arrays.append(values)
        self.table = pd.concat(parts, ignore_index=True).sort_values("matrix_row").reset_index(drop=True)
        assert self.table.variant_id.is_unique and self.table.matrix_row.is_unique
        self.table["row"] = np.arange(len(self.table))
        self.rank = self.table.cache_rank.to_numpy()
        self.row = self.table.cache_row.to_numpy()
        self.counts = self.table.n_mutations.to_numpy(dtype=np.int64)
        self.wild = np.zeros((len(self.table), 2), dtype=np.int64)
        self.mutant = np.zeros((len(self.table), 2), dtype=np.int64)
        for index, (wild, mutant) in enumerate(zip(self.table.wt_residues, self.table.mut_residues)):
            assert len(wild) == len(mutant) == self.counts[index]
            self.wild[index, :len(wild)] = [ALPHABET.index(value) for value in wild]
            self.mutant[index, :len(mutant)] = [ALPHABET.index(value) for value in mutant]

    def batch(self, rows):
        shape = (len(rows), 2, 512) if self.mode == "epistatic" else (len(rows), 384)
        values = np.empty(shape, dtype=np.float32)
        for rank in np.unique(self.rank[rows]):
            selected = self.rank[rows] == rank
            values[selected] = self.arrays[rank][self.row[rows[selected]]]
        return (torch.from_numpy(values).cuda(), torch.from_numpy(self.counts[rows]).cuda(),
                torch.from_numpy(self.wild[rows]).cuda(), torch.from_numpy(self.mutant[rows]).cuda())

    def predict(self, model, rows):
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(rows), 2048):
                values.append(model(*self.batch(rows[start:start + 2048]))[0].cpu().numpy())
        result = np.concatenate(values)
        assert len(result) == len(rows) and np.isfinite(result).all()
        return result


def new_head(mode, seed):
    original, configuration = new_model(mode, seed)
    model = CachedThermo(original, mode).cuda()
    return model


def reference_prediction(model):
    model.eval()
    with torch.inference_mode():
        if model.mode == "additive":
            return 0.0
        values = torch.zeros((1, 2, 512), device="cuda")
        count = torch.zeros(1, dtype=torch.long, device="cuda")
        residues = torch.zeros((1, 2), dtype=torch.long, device="cuda")
        result = model(values, count, residues, residues)[0]
    return float(result.item())


class DoubleEvaluation:
    def __init__(self, cache, proteins, smoke=False):
        self.cache = cache
        table = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
        table = table.loc[table.protein_id.isin(proteins)].copy()
        if smoke:
            table = table.iloc[::100].copy()
        lookup = cache.table.set_index(["protein_id", "mutation_key"]).row
        self.table = table.reset_index(drop=True)
        single_a, single_b, direct = [], [], []
        for row in self.table.itertuples():
            keys = [f"{wild}{int(position) + 1}{mutant}"
                    for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)]
            assert len(keys) == 2
            single_a.append(lookup.loc[(row.protein_id, keys[0])])
            single_b.append(lookup.loc[(row.protein_id, keys[1])])
            if cache.mode == "epistatic":
                direct.append(lookup.loc[(row.protein_id, row.mutation_key)])
        self.a, self.b = np.asarray(single_a, dtype=int), np.asarray(single_b, dtype=int)
        self.direct = np.asarray(direct, dtype=int)
        self.rows = np.unique(np.concatenate([self.a, self.b, self.direct]))
        assert len(self.table) > 0 and len(self.rows) > 0

    def predict(self, model):
        values = self.cache.predict(model, self.rows)
        a, b = values[np.searchsorted(self.rows, self.a)], values[np.searchsorted(self.rows, self.b)]
        direct = values[np.searchsorted(self.rows, self.direct)] if model.mode == "epistatic" else a + b
        reference = reference_prediction(model)
        result = self.table.copy()
        result["prediction_mu"] = -direct + reference
        result["prediction_a"] = -a + reference
        result["prediction_b"] = -b + reference
        result["prediction_wt"] = 0.0
        result["prediction_epsilon"] = result.prediction_mu - result.prediction_a - result.prediction_b
        if model.mode == "additive":
            np.testing.assert_allclose(result.prediction_epsilon, 0, atol=1e-5)
        return result

    def evaluate(self, model, smoke=False):
        table = self.predict(model)
        metrics = per_protein_metrics(table)
        selected = metrics.loc[metrics.main_quartets.gt(0) if smoke else metrics.main_eligible]
        assert len(selected) > 0 and np.isfinite(selected.epsilon_rmse).all()
        point = table.loc[table.relative_point.notna()]
        phenotype_mse = float(np.mean((point.prediction_mu - point.relative_point) ** 2))
        return float(selected.epsilon_rmse.mean()), phenotype_mse, table, metrics


def fit_source(mode, cache, augmentation, permissions, seed, learning_rate, weight_decay, directory, smoke):
    direct = cache.table.loc[cache.table.protein_id.isin(permissions["train"])
                             & cache.table.training_point_eligible
                             & cache.table.n_mutations.eq(2 if mode == "epistatic" else 1)].copy()
    if smoke:
        direct = direct.iloc[::100].copy()
    parts = [direct.assign(cache_source=0)]
    if mode == "epistatic":
        assert augmentation is not None
        assert augmentation.table.protein_id.isin(permissions["train"]).all()
        parts.append(augmentation.table.assign(cache_source=1))
    training = pd.concat(parts, ignore_index=True)
    assert len(training) > 0 and np.isfinite(training.native_ddg).all()
    assert not set(training.protein_id) & (set(permissions["validation"]) | set(permissions["test"]))
    training[["variant_id", "protein_id", "native_ddg", "cache_source"] +
             (["source_a_id", "source_b_id"] if mode == "epistatic" else [])].to_parquet(
                 directory / "training_variants.parquet", index=False)
    if mode == "epistatic":
        evaluator = DoubleEvaluation(cache, permissions["validation"], smoke=smoke)
        evaluator.table[["variant_id", "protein_id"]].to_parquet(directory / "validation_variants.parquet", index=False)
    else:
        validation = cache.table.loc[cache.table.protein_id.isin(permissions["validation"])
                                     & cache.table.training_point_eligible].copy()
        if smoke:
            validation = validation.iloc[::100].copy()
        assert len(validation) > 0
        validation[["variant_id", "protein_id"]].to_parquet(directory / "validation_variants.parquet", index=False)
    model = new_head(mode, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5)
    rng, history = np.random.default_rng(seed), []
    best, best_epoch = float("inf"), -1
    source_caches = [cache, augmentation]
    for epoch in range(2 if smoke else 100):
        model.train()
        shuffled = rng.permutation(len(training))
        seen, loss_sum = 0, 0.0
        for start in range(0, len(shuffled), 256):
            current = training.iloc[shuffled[start:start + 256]]
            pieces = [[], [], [], []]
            targets = []
            for source, group in current.groupby("cache_source", sort=True):
                batch = source_caches[source].batch(group.row.to_numpy(dtype=int))
                for slot, values in zip(pieces, batch):
                    slot.append(values)
                targets.append(group.native_ddg.to_numpy(dtype=np.float32))
            batch = [torch.cat(piece, dim=0) for piece in pieces]
            target = torch.tensor(np.concatenate(targets), device="cuda")
            predicted, asymmetry = model(*batch)
            loss = (predicted - target).square().mean() + asymmetry.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(current)
            seen += len(current)
        assert seen == len(training)
        if mode == "epistatic":
            score, phenotype_mse, prediction, metrics = evaluator.evaluate(model, smoke=smoke)
        else:
            prediction = validation[["variant_id", "protein_id", "native_ddg"]].copy()
            prediction["prediction_native_ddg"] = cache.predict(model, validation.row.to_numpy(dtype=int))
            prediction["squared_error"] = (prediction.prediction_native_ddg - prediction.native_ddg) ** 2
            metrics = prediction.groupby("protein_id").squared_error.mean().pow(0.5).rename("rmse").reset_index()
            score = float(metrics.rmse.mean())
            phenotype_mse = float(prediction.squared_error.mean())
        assert np.isfinite(score) and np.isfinite(phenotype_mse)
        scheduler.step(phenotype_mse)
        entry = {"epoch": epoch, "training_loss": loss_sum / seen, "validation_score": score,
                 "validation_phenotype_mse": phenotype_mse, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(entry)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(json.dumps(entry), flush=True)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "mode": mode, "seed": seed, "best_epoch": best_epoch, "validation_score": score},
                       directory / "best.pt")
            prediction.to_parquet(directory / "validation_predictions.parquet", index=False)
            metrics.to_parquet(directory / "validation_per_protein.parquet", index=False)
        if epoch - best_epoch >= 10:
            break
    assert np.isfinite(best)
    return {"validation_score": best, "best_epoch": best_epoch, "epochs": len(history),
            "direct_training_variants": len(direct), "augmented_training_transitions": len(training) - len(direct),
            "training_proteins": training.protein_id.nunique(),
            "source_label_permissions_verified": True,
            "training_loss": "Author mean squared destabilizing free-energy error and original symmetry penalty.",
            "optimizer": "AdamW and ReduceLROnPlateau(factor=0.5); original batch size 256.",
            "point_likelihood": "Accepted continuous point estimates with measured uncensored reference; no interval imputation.",
            "selection_metric": "macro_epsilon_rmse" if mode == "epistatic" else "macro_single_rmse",
            "direct_cache": str(cache.directory.relative_to(ROOT)),
            "augmentation_cache": str(augmentation.directory.relative_to(ROOT)) if augmentation is not None else None}
