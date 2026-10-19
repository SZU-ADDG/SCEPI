from copy import deepcopy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from epistasis.metrics import correlation

ROOT = Path(__file__).resolve().parents[1]


class MeanMLP(nn.Module):
    def __init__(self, width, input_dim=1280):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, 128),
                                    nn.GELU(), nn.Linear(128, 1))

    def forward(self, values):
        return self.layers(values).flatten()


class MeanRegression:
    def __init__(self, features, dataset, method, device):
        assert method in ["mean_ridge", "mean_mlp"]
        self.features, self.method, self.device = features, method, device
        directory = ROOT / "data/features/mean" / dataset
        assert json.loads((directory / "counterfactuals/manifest.json").read_text())["complete"]
        metadata = pd.read_parquet(directory / "metadata.parquet", columns=["variant_id", "matrix_row", "protein_id"])
        index = metadata.set_index("variant_id").matrix_row
        matrix = np.load(directory / "mean.npy", mmap_mode="r")
        protein = features.metadata.protein_id.unique()
        assert len(protein) == 1
        counterfactual = directory / "counterfactuals" / protein[0]
        single_metadata = pd.read_parquet(counterfactual / "metadata.parquet")
        single_lookup = pd.Series(np.arange(len(single_metadata)), index=single_metadata.mutation_key)
        single_rows = single_lookup.reindex(features.single_metadata.mutation_key).to_numpy()
        assert np.isfinite(single_rows).all()
        singles = np.load(counterfactual / "single_mean.npy")[single_rows.astype(int)]
        reference = np.load(counterfactual / "reference_mean.npy")
        self.values = np.concatenate([matrix[index.reindex(features.metadata.variant_id).to_numpy(dtype=int)].astype(np.float32),
                                      reference[None], singles], axis=0)
        assert np.isfinite(self.values).all()
        self.count = len(features.metadata)
        self.prediction_alias = np.arange(len(self.values))
        single_index = pd.Series(np.arange(len(features.single_metadata)), index=features.single_metadata.mutation_key)
        observed = features.metadata.loc[features.metadata.n_mutations.eq(1)]
        aliases = self.count + 1 + single_index.reindex(observed.mutation_key).to_numpy(dtype=int)
        self.prediction_alias[aliases] = observed.index.to_numpy()
        reference_rows = features.metadata.index[features.metadata.n_mutations.eq(0)]
        assert len(reference_rows) <= 1
        if len(reference_rows):
            self.prediction_alias[self.count] = reference_rows[0]
        repeated = np.flatnonzero(self.prediction_alias != np.arange(len(self.values)))
        np.testing.assert_array_equal(self.values[repeated], self.values[self.prediction_alias[repeated]])

    def score(self, predicted, validation, points, epsilon_components):
        rho = correlation(points[validation], predicted[validation], "spearman")
        rmse = float(np.sqrt(np.mean((points[validation] - predicted[validation]) ** 2)))
        if epsilon_components is None:
            return (-rho if np.isfinite(rho) else float("inf"), rmse), rho
        rows, truth = epsilon_components
        values = predicted[rows]
        epsilon = values[:, 3] - values[:, 1] - values[:, 2] + values[:, 0]
        return (float(np.sqrt(np.mean((epsilon - truth) ** 2))), rmse), rho

    def save_predictions(self, predictions, directory):
        np.savez_compressed(directory / "predictions.npz", variants=predictions[:self.count],
                            reference=predictions[self.count], singles=predictions[self.count + 1:])

    def fit(self, train_indices, validation_indices, points, seed, directory, epsilon_components=None, max_epochs=100):
        assert 1 <= max_epochs <= 100
        assert not set(train_indices) & set(validation_indices)
        assert np.isfinite(points[train_indices]).all() and np.isfinite(points[validation_indices]).all()
        x_train = self.values[train_indices].astype(np.float64)
        x_center, x_scale = x_train.mean(axis=0), x_train.std(axis=0)
        constant = x_scale == 0
        x_scale[constant] = 1.0
        center, scale = float(np.mean(points[train_indices])), float(np.std(points[train_indices]))
        assert scale > 0
        scaled = (points[train_indices] - center) / scale
        best, selected, trials = (float("inf"), float("inf")), None, []
        np.savez_compressed(directory / "normalization.npz", x_center=x_center, x_scale=x_scale,
                            y_center=center, y_scale=scale, constant_training_features=constant)
        if self.method == "mean_ridge":
            x = torch.tensor((x_train - x_center) / x_scale, dtype=torch.float64, device=self.device)
            y = torch.tensor(scaled, dtype=torch.float64, device=self.device)
            gram, rhs = x.T @ x, x.T @ y
            identity = torch.eye(x.shape[1], dtype=torch.float64, device=self.device)
            for penalty in np.logspace(-6, 5, 12):
                alpha = float(penalty) * len(train_indices)
                factor = torch.linalg.cholesky(gram + alpha * identity)
                coefficient = torch.cholesky_solve(rhs[:, None], factor).flatten()
                assert torch.isfinite(coefficient).all()
                values = []
                for start in range(0, len(self.values), 4096):
                    current = torch.tensor((self.values[start:start + 4096].astype(np.float64) - x_center) / x_scale,
                                           device=self.device, dtype=torch.float64)
                    values.append((current @ coefficient).cpu().numpy() * scale + center)
                predicted = np.concatenate(values)
                score, rho = self.score(predicted, validation_indices, points, epsilon_components)
                trial = {"ridge_penalty_per_observation": float(penalty), "ridge_alpha": alpha,
                         "primary_score": score[0], "secondary_score": score[1], "validation_spearman": rho}
                trials.append(trial)
                if score < best:
                    best, selected = score, trial
                    best_predictions = predicted
                    torch.save({"coefficient": coefficient.cpu(), "x_center": x_center, "x_scale": x_scale,
                                "y_center": center, "y_scale": scale, "selected": selected,
                                "train_indices": train_indices, "validation_indices": validation_indices}, directory / "model.pt")
                pd.DataFrame(trials).to_csv(directory / "trials.csv", index=False)
                print({"mean_trial": trial}, flush=True)
        else:
            normalized = ((self.values.astype(np.float64) - x_center) / x_scale).astype(np.float32)
            y = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            for learning_rate, width, weight_decay in itertools.product([1e-4, 3e-4, 1e-3], [128, 256], [1e-4, 0.01]):
                torch.manual_seed(seed)
                model = MeanMLP(width, input_dim=self.values.shape[1]).to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                local_best, best_epoch, state = (float("inf"), float("inf")), -1, None
                rng, history = np.random.default_rng(seed), []
                eval_rows = validation_indices if epsilon_components is None else np.unique(np.concatenate([validation_indices, epsilon_components[0].ravel()]))
                for epoch in range(max_epochs):
                    model.train()
                    order = rng.permutation(len(train_indices))
                    seen, loss_sum = 0, 0.0
                    for start in range(0, len(order), 512):
                        rows = order[start:start + 512]
                        current = torch.tensor(normalized[train_indices[rows]], device=self.device)
                        prediction = model(current)
                        loss = (prediction - y[torch.as_tensor(rows, device=self.device)]).square().mean()
                        assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        assert all(torch.isfinite(value.grad).all() for value in model.parameters())
                        optimizer.step()
                        seen += len(rows)
                        loss_sum += float(loss.detach()) * len(rows)
                    assert seen == len(train_indices)
                    predicted = np.full(len(self.values), np.nan)
                    model.eval()
                    with torch.inference_mode():
                        for start in range(0, len(eval_rows), 4096):
                            rows = eval_rows[start:start + 4096]
                            predicted[rows] = model(torch.tensor(normalized[rows], device=self.device)).cpu().numpy() * scale + center
                    score, rho = self.score(predicted, validation_indices, points, epsilon_components)
                    history.append({"epoch": epoch, "training_mse": loss_sum / seen,
                                    "primary_score": score[0], "secondary_score": score[1], "validation_spearman": rho})
                    if score < local_best:
                        local_best, best_epoch, state = score, epoch, deepcopy(model.state_dict())
                    if epoch - best_epoch >= 10:
                        break
                assert state is not None
                trial = {"learning_rate": learning_rate, "width": width, "weight_decay": weight_decay,
                         "best_epoch": best_epoch, "epochs": len(history),
                         "primary_score": local_best[0], "secondary_score": local_best[1]}
                trials.append(trial)
                pd.DataFrame(history).to_csv(directory / f"history_lr{learning_rate}_w{width}_wd{weight_decay}.csv", index=False)
                if local_best < best:
                    best, selected = local_best, trial
                    model.load_state_dict(state)
                    model.eval()
                    values = []
                    with torch.inference_mode():
                        for start in range(0, len(normalized), 4096):
                            values.append(model(torch.tensor(normalized[start:start + 4096], device=self.device)).cpu().numpy() * scale + center)
                    best_predictions = np.concatenate(values)
                    torch.save({"model": {key: value.detach().cpu() for key, value in state.items()},
                                "x_center": x_center, "x_scale": x_scale, "y_center": center, "y_scale": scale,
                                "selected": selected, "train_indices": train_indices, "validation_indices": validation_indices}, directory / "model.pt")
                pd.DataFrame(trials).to_csv(directory / "trials.csv", index=False)
                print({"mean_trial": trial}, flush=True)
                del model, optimizer
        assert selected is not None and np.isfinite(best_predictions).all()
        best_predictions = best_predictions[self.prediction_alias]
        self.save_predictions(best_predictions, directory)
        return {"method": self.method, "selected": selected, "trials": len(trials), "source_labels_used": 0,
                "target_labels_used": len(train_indices), "validation_labels_used": len(validation_indices),
                "features": "Mean-pooled frozen ESM-2 650M embeddings of the complete genotype.",
                "normalization_labels": "Training means and variances only, for embeddings and phenotype.",
                "selection_metric": "epsilon_rmse" if epsilon_components is not None else "phenotype_spearman",
                "objective": "Squared error with ridge regularization." if self.method == "mean_ridge" else "Squared error; AdamW.",
                "counterfactuals": "The same fitted direct regression evaluates each variant, its reference, and all component singles.",
                "device": str(self.device)}
