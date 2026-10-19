from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from pymochi.models import MochiModel, MochiWeightedL1Loss

from epistasis.metrics import correlation


class MochiRegression:
    def __init__(self, features, order, train_indices, device):
        self.features = features
        self.device = device
        parts = [sparse.csr_matrix(np.ones((len(features.metadata), 1), dtype=np.float32)),
                 features.single_incidence]
        if order == 2:
            parts.append(features.pair_incidence)
        self.matrix = sparse.hstack(parts, format="csr", dtype=np.float32)
        observed = np.asarray(self.matrix[train_indices].sum(axis=0)).ravel() > 0
        assert observed[0]
        self.mask = torch.tensor(observed[None, None, :], dtype=torch.float32, device=device)
        self.design = pd.DataFrame({"phenotype": [1], "transformation": ["SumOfSigmoids"], "trait": [[1]]})
        self.order = order
        self.model = None
        self.center, self.scale = None, None

    def new_model(self, seed, y):
        torch.manual_seed(seed)
        model = MochiModel(input_shape=self.matrix.shape[1], mask=self.mask,
                           model_design=self.design.copy(deep=True), custom_transformations={},
                           sos_architecture=[20], sos_outputlinear=False).to(self.device)
        quantiles = np.quantile(y, [0.1, 0.9])
        model.weights_init_fill(linear_weight=[float(quantiles[1] - quantiles[0])],
                                linear_bias=[float(quantiles[0])])
        return model

    def forward(self, model, matrix):
        values = torch.tensor(matrix.toarray(), dtype=torch.float32, device=self.device)
        select = torch.ones((len(values), 1), device=self.device)
        return model(select, values, self.mask).flatten()

    def predict_matrix(self, model, matrix, batch_size=512):
        predictions = []
        model.eval()
        with torch.no_grad():
            for start in range(0, matrix.shape[0], batch_size):
                predictions.append(self.forward(model, matrix[start:start + batch_size]).cpu().numpy())
        result = np.concatenate(predictions)
        assert np.isfinite(result).all()
        return result

    def fit(self, train_indices, validation_indices, points, seed, directory, epsilon_components=None):
        assert not set(train_indices) & set(validation_indices)
        assert len(train_indices) >= 2 and len(validation_indices) >= 2
        y_train, y_valid = points[train_indices], points[validation_indices]
        assert np.isfinite(y_train).all() and np.isfinite(y_valid).all()
        self.center, self.scale = float(np.mean(y_train)), float(np.std(y_train))
        assert self.scale > 0
        scaled = (points - self.center) / self.scale
        loss_function = MochiWeightedL1Loss()
        if epsilon_components is not None:
            component_rows, epsilon_truth = epsilon_components
            assert set(component_rows[:, 3]) <= set(validation_indices)
            unique_rows = np.unique(component_rows)
            component_index = np.searchsorted(unique_rows, component_rows)
        summaries, all_history, selected_model = [], [], None
        best_trial = (float("inf"), float("inf"), float("inf"), float("inf"))
        for learning_rate in [0.005, 0.01, 0.05]:
            for penalty in [0.0, 1e-6, 1e-5, 1e-4]:
                model = self.new_model(seed, scaled[train_indices])
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                rng = np.random.default_rng(seed)
                best, best_epoch, state = (float("inf"), float("inf")), -1, None
                history = []
                for epoch in range(100):
                    model.train()
                    count, total = 0, 0.0
                    shuffled = rng.permutation(train_indices)
                    for start in range(0, len(shuffled), 512):
                        indices = shuffled[start:start + 512]
                        predicted = self.forward(model, self.matrix[indices])
                        target = torch.tensor(scaled[indices], dtype=torch.float32, device=self.device)
                        loss = loss_function(predicted, target, torch.ones_like(target)).mean()
                        regularization, _ = model.calculate_l1l2_norm()
                        loss = loss + penalty * regularization
                        assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
                        optimizer.step()
                        count += len(indices)
                        total += float(loss.detach()) * len(indices)
                    assert count == len(train_indices)
                    predicted = self.predict_matrix(model, self.matrix[validation_indices])
                    rho = correlation(scaled[validation_indices], predicted, "spearman")
                    rmse = float(np.sqrt(np.mean((predicted - scaled[validation_indices]) ** 2)))
                    epsilon_rmse = None
                    if epsilon_components is not None:
                        values = self.predict_matrix(model, self.matrix[unique_rows]) * self.scale + self.center
                        components = values[component_index]
                        predicted_epsilon = components[:, 3] - components[:, 1] - components[:, 2] + components[:, 0]
                        epsilon_rmse = float(np.sqrt(np.mean((predicted_epsilon - epsilon_truth) ** 2)))
                    score = (epsilon_rmse, rmse) if epsilon_rmse is not None else (-rho if np.isfinite(rho) else float("inf"), rmse)
                    history.append({"epoch": epoch, "training_loss": total / count,
                                    "validation_spearman": rho, "validation_scaled_rmse": rmse,
                                    "validation_epsilon_rmse": epsilon_rmse,
                                    "learning_rate": learning_rate, "l1_penalty": penalty})
                    if score < best:
                        best, best_epoch = score, epoch
                        state = deepcopy(model.state_dict())
                    if epoch - best_epoch >= 10:
                        break
                assert state is not None
                all_history.extend(history)
                trial = (*best, learning_rate, penalty)
                summaries.append({"learning_rate": learning_rate, "l1_penalty": penalty, "best_epoch": best_epoch,
                                  "validation_spearman": history[best_epoch]["validation_spearman"],
                                  "validation_epsilon_rmse": history[best_epoch]["validation_epsilon_rmse"],
                                  "selection_metric": "epsilon_rmse" if epsilon_components is not None else "phenotype_spearman",
                                  "validation_scaled_rmse": best[1], "epochs": len(history)})
                if trial < best_trial:
                    best_trial = trial
                    model.load_state_dict(state)
                    selected_model = deepcopy(model)
                    selected = summaries[-1]
                del model, optimizer
        self.model = selected_model
        torch.save({"model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                    "mask": self.mask.cpu(), "target_mean": self.center, "target_std": self.scale,
                    "selected": selected, "order": self.order, "seed": seed,
                    "training_indices": train_indices, "validation_indices": validation_indices}, directory / "model.pt")
        pd.DataFrame(all_history).to_csv(directory / "optimization.csv", index=False)
        pd.DataFrame(summaries).to_csv(directory / "trials.csv", index=False)
        values = self.predict_matrix(self.model, self.matrix) * self.scale + self.center
        count = len(self.features.single_metadata)
        reference_rows = np.zeros(count + 1, dtype=int)
        reference_columns = np.zeros(count + 1, dtype=int)
        reference_rows[1:] = np.arange(1, count + 1)
        base = sparse.coo_matrix((np.ones(count + 1), (reference_rows, reference_columns)),
                                 shape=(count + 1, self.matrix.shape[1])).tocsr()
        singles = sparse.coo_matrix((np.ones(count), (np.arange(1, count + 1), np.arange(1, count + 1))),
                                    shape=base.shape).tocsr()
        counterfactual = self.predict_matrix(self.model, base + singles) * self.scale + self.center
        np.savez_compressed(directory / "predictions.npz", variants=values, reference=counterfactual[0], singles=counterfactual[1:])
        return {"selected": selected, "trials": len(summaries), "feature_order": self.order,
                "feature_columns": self.matrix.shape[1], "fitted_columns": int(self.mask.sum()),
                "initialization_labels": "Training observations only.",
                "training_loss": "Author MochiWeightedL1Loss with uniform loss weights; no fabricated measurement uncertainty.",
                "variant_encoding": "Reference-coded observed amino-acid substitutions and unordered pair indicators.",
                "minimum_observations_per_fitted_term": 1, "target_labels_used": len(train_indices),
                "validation_labels_used": len(validation_indices), "model_class": "pymochi.models.MochiModel",
                "vendor_version": "v1.1", "vendor_commit": "b71c73806c2b7049bc05c499258c1cea6013189f",
                "source_labels_used": 0, "device": str(self.device)}
