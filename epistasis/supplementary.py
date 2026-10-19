from copy import deepcopy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
import torch
from torch import nn

from epistasis.mean_regression import MeanRegression
from epistasis.metrics import correlation

ROOT = Path(__file__).resolve().parents[1]


def classification_metrics(labels, probability):
    predicted = probability.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, labels=np.arange(4), average="macro", zero_division=0)),
            "class_absolute_error": float(np.mean(np.abs(labels - predicted))),
            "log_loss": float(log_loss(labels, probability, labels=np.arange(4)))}


class SupplementaryMLP(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, 128),
                                    nn.GELU(), nn.Linear(128, 4))

    def forward(self, values):
        return self.layers(values)


class SupplementaryRegression(MeanRegression):
    def __init__(self, dataset, split, method, device):
        self.method, self.device, self.dataset = method, device, dataset
        source = ROOT / "data/processed/flip_supplementary"
        records = pd.read_parquet(source / "records.parquet")
        self.metadata = records.loc[records.dataset.eq(dataset) & records.official_split.eq(split)].sort_values("source_row").reset_index(drop=True)
        directory = ROOT / "data/features/mean/flip_supplementary"
        assert json.loads((directory / "manifest.json").read_text())["complete"]
        variants = pd.read_parquet(directory / (dataset + "_metadata.parquet"))
        index = variants.set_index("variant_id").matrix_row
        matrix = np.load(directory / (dataset + "_mean.npy"), mmap_mode="r")
        selected = self.metadata.variant_id.map(index).to_numpy(dtype=int)
        self.count = len(selected)
        self.counterfactuals = variants.loc[~variants.variant_id.isin(self.metadata.variant_id)].copy().reset_index(drop=True)
        self.counterfactuals["row"] = np.arange(self.count, self.count + len(self.counterfactuals))
        self.values = np.concatenate([matrix[selected], matrix[self.counterfactuals.matrix_row.to_numpy(dtype=int)]]).astype(np.float32)
        assert np.isfinite(self.values).all()
        ids = pd.concat([self.metadata.variant_id, self.counterfactuals.variant_id], ignore_index=True)
        first = pd.Series(np.arange(len(ids)), index=ids).groupby(level=0, sort=False).first()
        self.prediction_alias = ids.map(first).to_numpy(dtype=int)
        np.testing.assert_array_equal(self.values, self.values[self.prediction_alias])

    def save_predictions(self, predictions, directory):
        self.metadata.assign(prediction_absolute=predictions[:self.count]).to_parquet(directory / "all_record_predictions.parquet", index=False)
        self.counterfactuals.assign(prediction_absolute=predictions[self.counterfactuals.row.to_numpy(dtype=int)]).to_parquet(
            directory / "counterfactual_predictions.parquet", index=False)


class SupplementaryClassifier(SupplementaryRegression):
    def fit(self, train_indices, validation_indices, points, seed, directory, max_epochs=100):
        assert self.dataset == "nucb" and self.method in ["mean_logistic", "mean_classifier_mlp"]
        assert not set(train_indices) & set(validation_indices)
        labels = points.astype(np.int64)
        np.testing.assert_array_equal(labels, points)
        assert set(labels) <= set(range(4))
        x = self.values.astype(np.float64)
        center, scale = x[train_indices].mean(axis=0), x[train_indices].std(axis=0)
        scale[scale == 0] = 1.
        normalized = (x - center) / scale
        classes = np.unique(labels[train_indices])
        assert len(classes) >= 3
        best, selected, trials = (float("inf"), float("inf")), None, []
        parameters = list(np.logspace(-6, 5, 12)) if self.method == "mean_logistic" else list(
            itertools.product([1e-4, 3e-4, 1e-3], [128, 256], [1e-4, 0.01]))
        for parameter in parameters:
            if self.method == "mean_logistic":
                model = LogisticRegression(C=float(parameter), solver="lbfgs", max_iter=100000, tol=1e-6, random_state=seed)
                model.fit(normalized[train_indices], labels[train_indices])
                assert np.max(model.n_iter_) < model.max_iter, model.n_iter_
                probability = np.zeros((len(self.values), 4), dtype=np.float64)
                probability[:, model.classes_] = model.predict_proba(normalized)
                metrics = classification_metrics(labels[validation_indices], probability[validation_indices])
                score = (-metrics["macro_f1"], metrics["log_loss"])
                trial = {"C": float(parameter), "iterations": int(model.n_iter_.max()), **metrics}
                state = {"coefficient": model.coef_, "intercept": model.intercept_, "classes": model.classes_}
            else:
                learning_rate, width, weight_decay = parameter
                torch.manual_seed(seed)
                model = SupplementaryMLP(normalized.shape[1], width).to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                y = torch.tensor(labels, device=self.device)
                local_best, best_epoch, local_state = (float("inf"), float("inf")), -1, None
                rng, history = np.random.default_rng(seed), []
                for epoch in range(max_epochs):
                    model.train()
                    total, seen = 0., 0
                    for indices in np.array_split(rng.permutation(train_indices), max(1, int(np.ceil(len(train_indices) / 512)))):
                        values = torch.tensor(normalized[indices], dtype=torch.float32, device=self.device)
                        loss = nn.functional.cross_entropy(model(values), y[indices])
                        assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
                        optimizer.step()
                        total += float(loss.detach()) * len(indices)
                        seen += len(indices)
                    assert seen == len(train_indices)
                    model.eval()
                    with torch.inference_mode():
                        probability = torch.cat([model(torch.tensor(normalized[indices], dtype=torch.float32, device=self.device)).softmax(-1).cpu()
                                                 for indices in np.array_split(validation_indices, max(1, int(np.ceil(len(validation_indices) / 4096))))]).numpy()
                    metrics = classification_metrics(labels[validation_indices], probability)
                    score = (-metrics["macro_f1"], metrics["log_loss"])
                    history.append({"epoch": epoch, "training_cross_entropy": total / seen, **metrics})
                    if score < local_best:
                        local_best, best_epoch, local_state = score, epoch, deepcopy(model.state_dict())
                    if epoch - best_epoch >= 10:
                        break
                assert local_state is not None
                model.load_state_dict(local_state)
                model.eval()
                with torch.inference_mode():
                    probability = torch.cat([model(torch.tensor(normalized[start:start + 4096], dtype=torch.float32, device=self.device)).softmax(-1).cpu()
                                             for start in range(0, len(normalized), 4096)]).numpy()
                metrics = classification_metrics(labels[validation_indices], probability[validation_indices])
                score = (-metrics["macro_f1"], metrics["log_loss"])
                trial = {"learning_rate": learning_rate, "width": width, "weight_decay": weight_decay,
                         "best_epoch": best_epoch, "epochs": len(history), **metrics}
                state = {"model": {key: value.detach().cpu() for key, value in local_state.items()}}
                pd.DataFrame(history).to_csv(directory / f"history_lr{learning_rate}_w{width}_wd{weight_decay}.csv", index=False)
            assert np.isfinite(probability).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-6, rtol=0)
            trials.append(trial)
            if score < best:
                best, selected = score, trial
                best_probability = probability.copy()
                torch.save({**state, "x_center": center, "x_scale": scale, "train_indices": train_indices,
                            "validation_indices": validation_indices, "selected": trial,
                            "method": self.method, "training_classes": classes}, directory / "model.pt")
            pd.DataFrame(trials).to_csv(directory / "trials.csv", index=False)
        assert selected is not None
        probability = best_probability[self.prediction_alias]
        frame = self.metadata.copy()
        for label in range(4):
            frame[f"probability_{label}"] = probability[:self.count, label]
        frame["prediction_class"] = probability[:self.count].argmax(axis=1)
        frame.to_parquet(directory / "all_record_predictions.parquet", index=False)
        return {"method": self.method, "selected": selected, "trials": len(trials),
                "target_labels_used": len(train_indices), "validation_labels_used": len(validation_indices),
                "source_labels_used": 0, "training_classes": classes.tolist(),
                "absent_training_classes": sorted(set(range(4)) - set(classes)),
                "class_ontology": [0, 1, 2, 3], "selection_metric": "Validation macro F1 over four declared classes; log loss breaks ties.",
                "class_weighting": "Unweighted maximum likelihood; no synthetic or duplicated class observations."}


def save_evaluation(regression, directory):
    predictions = pd.read_parquet(directory / "all_record_predictions.parquet")
    predictions["partition"] = predictions[regression.partition_column]
    selected = predictions.loc[predictions.partition.isin(["validation", "test"])].copy()
    selected.to_parquet(directory / "predictions.parquet", index=False)
    summaries = []
    for partition, group in selected.groupby("partition", sort=True):
        strata = [("all", group)] + [(str(order), group.loc[group.n_mutations.eq(order)]) for order in [1, 2, 3, 4]]
        strata += [("5plus", group.loc[group.n_mutations.ge(5)])]
        for order, subset in strata:
            if len(subset) == 0:
                continue
            result = {"partition": partition, "mutation_order": order, "records": len(subset),
                      "unique_inputs": subset.variant_id.nunique()}
            if regression.dataset == "nucb":
                result.update(classification_metrics(subset.target.to_numpy(dtype=int),
                                                     subset[[f"probability_{value}" for value in range(4)]].to_numpy()))
            else:
                result.update(phenotype_spearman=correlation(subset.target, subset.prediction_absolute, "spearman"),
                              phenotype_rmse=float(np.sqrt(np.mean((subset.target - subset.prediction_absolute) ** 2))),
                              phenotype_mae=float(np.mean(np.abs(subset.target - subset.prediction_absolute))))
            summaries.append(result)
    pd.DataFrame(summaries).to_csv(directory / "metrics.csv", index=False)
    quartet_count = 0
    if regression.dataset == "pdz3":
        extras = pd.read_parquet(directory / "counterfactual_predictions.parquet")
        unique = pd.concat([predictions[["variant_id", "prediction_absolute"]], extras[["variant_id", "prediction_absolute"]]])
        assert unique.variant_id.is_unique
        values = unique.set_index("variant_id").prediction_absolute
        quartets = pd.read_parquet(ROOT / "data/processed/flip_supplementary/pdz3_quartets.parquet")
        quartets = quartets.loc[quartets.variant_id.isin(selected.variant_id)].copy()
        quartets["partition"] = quartets.variant_id.map(selected.set_index("variant_id").partition)
        for name, key in [("ab", "variant_id"), ("a", "a_id"), ("b", "b_id"), ("wt", "wt_id")]:
            quartets["prediction_" + name] = quartets[key].map(values)
        assert quartets.filter(like="prediction_").notna().all().all()
        quartets["prediction_epsilon"] = quartets.prediction_ab - quartets.prediction_a - quartets.prediction_b + quartets.prediction_wt
        quartets.to_parquet(directory / "quartet_predictions.parquet", index=False)
        quartet_count = len(quartets)
        metrics = [{"partition": label, "quartets": len(group),
                    "epsilon_rmse": float(np.sqrt(np.mean((group.prediction_epsilon - group.epsilon) ** 2))),
                    "epsilon_spearman": correlation(group.epsilon, group.prediction_epsilon, "spearman")}
                   for label, group in quartets.groupby("partition")]
        pd.DataFrame(metrics).to_csv(directory / "quartet_metrics.csv", index=False)
    return {"evaluation_records": len(selected), "test_records": int(selected.partition.eq("test").sum()),
            "complete_evaluated_quartets": quartet_count, "official_test_records_preserved": True}
