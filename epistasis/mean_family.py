from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epistasis.mean_regression import MeanMLP
from epistasis.metrics import per_protein_metrics

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/features/mean/tsuboyama"


class MeanFamilyCache:
    def __init__(self):
        assert json.loads((FEATURES / "counterfactuals/manifest.json").read_text())["complete"]
        self.table = pd.read_parquet(FEATURES / "metadata.parquet")
        self.values = np.load(FEATURES / "mean.npy", mmap_mode="r")
        self.original_count = len(self.table)
        assert len(self.values) == self.original_count
        doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
        existing = set(zip(self.table.protein_id, self.table.mutation_key))
        extras, records = [], []
        for protein, group in doubles.groupby("protein_id", sort=True):
            directory = FEATURES / "counterfactuals" / protein
            single_table = pd.read_parquet(directory / "metadata.parquet")
            single_lookup = pd.Series(single_table.index, index=single_table.mutation_key)
            means = np.load(directory / "single_mean.npy", mmap_mode="r")
            needed = {(int(position), wild, mutant) for row in group.itertuples()
                      for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)}
            for position, wild, mutant in sorted(needed):
                key = f"{wild}{position + 1}{mutant}"
                if (protein, key) in existing:
                    continue
                records.append({"variant_id": f"mean_cf_{protein}_{key}", "protein_id": protein,
                                "mutation_key": key, "n_mutations": 1, "point": np.nan,
                                "point_in_range": False, "all_point": False})
                extras.append(means[single_lookup[key]].astype(np.float32))
                existing.add((protein, key))
        self.extra_values = np.asarray(extras, dtype=np.float32).reshape(-1, 1280)
        self.table = pd.concat([self.table, pd.DataFrame(records)], ignore_index=True)
        self.table["row"] = np.arange(len(self.table))
        assert self.table.variant_id.is_unique
        self.lookup = self.table.set_index(["protein_id", "mutation_key"]).row

    def batch(self, rows):
        rows = np.asarray(rows, dtype=int)
        values = np.empty((len(rows), 1280), dtype=np.float32)
        direct = rows < self.original_count
        values[direct] = self.values[rows[direct]].astype(np.float32)
        values[~direct] = self.extra_values[rows[~direct] - self.original_count]
        return values


class MeanFamilyEvaluation:
    def __init__(self, cache, proteins, smoke=False):
        self.cache = cache
        self.table = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
        self.table = self.table.loc[self.table.protein_id.isin(proteins)].copy()
        if smoke:
            self.table = self.table.iloc[::100].copy()
        components = []
        for row in self.table.itertuples():
            keys = [f"{wild}{int(position) + 1}{mutant}"
                    for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)]
            components.append([cache.lookup[(row.protein_id, key)] for key in ["", *keys, row.mutation_key]])
        self.components = np.asarray(components, dtype=int)
        self.rows = np.unique(self.components)
        self.smoke = smoke
        assert self.components.shape == (len(self.table), 4) and len(self.table)

    def evaluate(self, values):
        assert len(values) == len(self.rows) and np.isfinite(values).all()
        component = values[np.searchsorted(self.rows, self.components)]
        table = self.table.copy()
        table["prediction_mu"] = component[:, 3] - component[:, 0]
        table["prediction_a"] = component[:, 1] - component[:, 0]
        table["prediction_b"] = component[:, 2] - component[:, 0]
        table["prediction_wt"] = 0.0
        table["prediction_epsilon"] = table.prediction_mu - table.prediction_a - table.prediction_b
        metrics = per_protein_metrics(table)
        selected = metrics.loc[metrics.main_quartets.gt(0) if self.smoke else metrics.main_eligible]
        assert len(selected) and np.isfinite(selected.epsilon_rmse).all()
        return float(selected.epsilon_rmse.mean()), table, metrics


def training_normalization(cache, rows, targets):
    total = np.zeros(1280, dtype=np.float64)
    squares = np.zeros(1280, dtype=np.float64)
    for start in range(0, len(rows), 4096):
        values = cache.batch(rows[start:start + 4096]).astype(np.float64)
        total += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
    center = total / len(rows)
    variance = squares / len(rows) - center ** 2
    assert variance.min() >= -1e-12
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale == 0] = 1
    target_scale = float(np.std(targets))
    assert target_scale > 0
    return {"x_center": center, "x_scale": scale, "y_center": float(np.mean(targets)), "y_scale": target_scale}


def predict(cache, rows, state, model=None, device="cuda"):
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), 4096):
            values = (cache.batch(rows[start:start + 4096]).astype(np.float64) - state["x_center"]) / state["x_scale"]
            if state["method"] == "mean_ridge":
                coefficient = state["coefficient"].to(device)
                predicted = torch.tensor(values, dtype=torch.float64, device=device) @ coefficient
            else:
                assert model is not None
                predicted = model(torch.tensor(values, dtype=torch.float32, device=device))
            records.append(predicted.cpu().numpy() * state["y_scale"] + state["y_center"])
    return np.concatenate(records)


def fit(cache, permissions, directory, method, seed, learning_rate, width, weight_decay, ridge_penalty, smoke):
    training = cache.table.loc[cache.table.protein_id.isin(permissions["train"]) & cache.table.point_in_range].copy()
    total_count = len(training)
    if smoke:
        training = training.iloc[::100].copy()
    assert len(training) and np.isfinite(training.point).all()
    assert not set(training.protein_id) & (set(permissions["validation"]) | set(permissions["test"]))
    rows, target = training.row.to_numpy(dtype=int), training.point.to_numpy(dtype=np.float64)
    training[["variant_id", "protein_id", "point"]].to_parquet(directory / "training_variants.parquet", index=False)
    evaluator = MeanFamilyEvaluation(cache, permissions["validation"], smoke)
    cache.table.iloc[evaluator.rows][["variant_id", "protein_id"]].to_parquet(
        directory / "validation_variants.parquet", index=False)
    normalization = training_normalization(cache, rows, target)
    np.savez_compressed(directory / "normalization.npz", **normalization)
    normalized_y = (target - normalization["y_center"]) / normalization["y_scale"]
    state = {**normalization, "method": method}
    history = []
    if method == "mean_ridge":
        gram = torch.zeros((1280, 1280), dtype=torch.float64, device="cuda")
        rhs = torch.zeros(1280, dtype=torch.float64, device="cuda")
        for start in range(0, len(rows), 4096):
            values = (cache.batch(rows[start:start + 4096]).astype(np.float64) - normalization["x_center"]) / normalization["x_scale"]
            x = torch.tensor(values, dtype=torch.float64, device="cuda")
            y = torch.tensor(normalized_y[start:start + 4096], dtype=torch.float64, device="cuda")
            gram += x.T @ x
            rhs += x.T @ y
        alpha = ridge_penalty * len(rows)
        factor = torch.linalg.cholesky(gram + alpha * torch.eye(1280, dtype=torch.float64, device="cuda"))
        state["coefficient"] = torch.cholesky_solve(rhs[:, None], factor).flatten().cpu()
        state["ridge_alpha"], state["ridge_penalty_per_observation"] = alpha, ridge_penalty
        assert torch.isfinite(state["coefficient"]).all()
        score, predictions, metrics = evaluator.evaluate(predict(cache, evaluator.rows, state))
        best_epoch = 0
        history.append({"epoch": 0, "validation_score": score})
        best = score
        torch.save(state, directory / "best.pt")
        predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
        metrics.to_parquet(directory / "validation_metrics.parquet", index=False)
    else:
        torch.manual_seed(seed)
        model = MeanMLP(width).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        rng = np.random.default_rng(seed)
        best, best_epoch = float("inf"), -1
        state.update(width=width, learning_rate=learning_rate, weight_decay=weight_decay)
        for epoch in range(2 if smoke else 100):
            model.train()
            order = rng.permutation(len(rows))
            count, loss_sum = 0, 0.0
            for start in range(0, len(order), 512):
                selected = order[start:start + 512]
                values = (cache.batch(rows[selected]).astype(np.float64) - normalization["x_center"]) / normalization["x_scale"]
                x = torch.tensor(values, dtype=torch.float32, device="cuda")
                y = torch.tensor(normalized_y[selected], dtype=torch.float32, device="cuda")
                loss = (model(x) - y).square().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
                optimizer.step()
                count += len(selected)
                loss_sum += float(loss.detach()) * len(selected)
            assert count == len(rows)
            model.eval()
            score, predictions, metrics = evaluator.evaluate(predict(cache, evaluator.rows, state, model))
            entry = {"epoch": epoch, "validation_score": score, "training_mse": loss_sum / count}
            history.append(entry)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({**state, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                           directory / "best.pt")
                predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
                metrics.to_parquet(directory / "validation_metrics.parquet", index=False)
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            print(json.dumps(entry), flush=True)
            if epoch - best_epoch >= 10:
                break
        assert best_epoch >= 0
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    return {"validation_score": best, "best_epoch": best_epoch, "epochs": len(history),
            "training_variants": len(training), "full_training_variants": total_count,
            "training_fraction": len(training) / total_count, "validation_quartets": len(evaluator.table),
            "label_permissions_verified": True, "target_labels_used": 0,
            "normalization": "Source-training embedding and phenotype statistics only.",
            "selection": "Source-validation macro epsilon RMSE from four model predictions.",
            "features": "Mean-pooled ESM-2 650M embeddings of complete observed and counterfactual sequences."}
