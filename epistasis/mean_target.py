from copy import deepcopy
import fcntl
import json

import numpy as np
import pandas as pd
import torch

from epistasis.mean_family import MeanFamilyEvaluation, predict
from epistasis.mean_regression import MeanMLP


def moments(cache, rows, targets):
    total = torch.zeros(1280, dtype=torch.float64, device="cuda")
    gram = torch.zeros((1280, 1280), dtype=torch.float64, device="cuda")
    rhs = torch.zeros(1280, dtype=torch.float64, device="cuda")
    for start in range(0, len(rows), 4096):
        x = torch.tensor(cache.batch(rows[start:start + 4096]), dtype=torch.float64, device="cuda")
        y = torch.tensor(targets[start:start + 4096], dtype=torch.float64, device="cuda")
        total += x.sum(0)
        gram += x.T @ x
        rhs += x.T @ y
    return {"count": len(rows), "sum_x": total.cpu(), "sum_xx": gram.cpu(), "sum_xy": rhs.cpu(),
            "sum_y": float(targets.sum()), "sum_yy": float(np.square(targets).sum())}


def source_moments(cache, source):
    training = pd.read_parquet(source / "training_variants.parquet")
    indices = cache.table.set_index("variant_id").row.loc[training.variant_id].to_numpy(dtype=int)
    identifiers = training.variant_id.tolist()
    output = source.parent.parent / "source_ridge_moments.pt"
    with (output.parent / "source_ridge_moments.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            result = torch.load(output, map_location="cpu", weights_only=False)
            assert result["variant_ids"] == identifiers
        else:
            result = moments(cache, indices, training.point.to_numpy(dtype=np.float64))
            result["variant_ids"] = identifiers
            torch.save(result, output)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return result


def combined_ridge(cache, source, training, source_state):
    original = source_moments(cache, source)
    current = moments(cache, training.row.to_numpy(dtype=int), training.point.to_numpy(dtype=np.float64))
    count = original["count"] + current["count"]
    total = {key: original[key] + current[key] for key in ["sum_x", "sum_xx", "sum_xy", "sum_y", "sum_yy"]}
    center = total["sum_x"].numpy() / count
    variance = total["sum_xx"].diagonal().numpy() / count - center ** 2
    assert variance.min() >= -1e-12
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale == 0] = 1
    y_center = total["sum_y"] / count
    y_scale = float(np.sqrt(total["sum_yy"] / count - y_center ** 2))
    assert np.isfinite(y_scale) and y_scale > 0
    x_center = torch.tensor(center, dtype=torch.float64)
    x_scale = torch.tensor(scale, dtype=torch.float64)
    gram = (total["sum_xx"] - count * x_center[:, None] * x_center[None, :]) / (x_scale[:, None] * x_scale[None, :])
    rhs = (total["sum_xy"] - count * x_center * y_center) / (x_scale * y_scale)
    alpha = source_state["ridge_penalty_per_observation"] * count
    factor = torch.linalg.cholesky(gram.cuda() + alpha * torch.eye(1280, dtype=torch.float64, device="cuda"))
    coefficient = torch.cholesky_solve(rhs.cuda()[:, None], factor).flatten().cpu()
    assert torch.isfinite(coefficient).all()
    return {"method": "mean_ridge", "x_center": center, "x_scale": scale, "y_center": y_center, "y_scale": y_scale,
            "coefficient": coefficient, "ridge_alpha": alpha,
            "ridge_penalty_per_observation": source_state["ridge_penalty_per_observation"],
            "source_training_variants": original["count"], "target_training_variants": current["count"]}


def prediction_table(evaluator, state, model=None):
    values = predict(evaluator.cache, evaluator.rows, state, model)
    components = values[np.searchsorted(evaluator.rows, evaluator.components)]
    table = evaluator.table.copy()
    table["prediction_mu"] = components[:, 3] - components[:, 0]
    table["prediction_a"] = components[:, 1] - components[:, 0]
    table["prediction_b"] = components[:, 2] - components[:, 0]
    table["prediction_wt"] = 0.0
    table["prediction_epsilon"] = table.prediction_mu - table.prediction_a - table.prediction_b
    assert np.isfinite(table[["prediction_mu", "prediction_a", "prediction_b", "prediction_epsilon"]]).all().all()
    return table


def fit_target(source, cache, training, validation_ids, directory, seed):
    manifest = json.loads((source / "manifest.json").read_text())
    state = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    method = state["method"]
    evaluator = MeanFamilyEvaluation(cache, training.protein_id.unique())
    model = None
    if method == "mean_mlp":
        torch.manual_seed(seed)
        model = MeanMLP(state["width"]).cuda().eval()
        model.load_state_dict(state["model"])
    initial = prediction_table(evaluator, state, model)
    initial.to_parquet(directory / "initial_predictions.parquet", index=False)
    validation = evaluator.table.variant_id.isin(validation_ids).to_numpy()
    main = validation & evaluator.table.main.to_numpy()
    point = validation & evaluator.table.relative_point.notna().to_numpy()
    use_validation = main.any() or point.any()

    def score(table):
        epsilon = float(np.sqrt(np.mean((table.loc[main, "prediction_epsilon"] - table.loc[main, "epsilon"]) ** 2))) if main.any() else 0.0
        phenotype = float(np.sqrt(np.mean((table.loc[point, "prediction_mu"] - table.loc[point, "relative_point"]) ** 2))) if point.any() else 0.0
        return epsilon, phenotype

    history = []
    if method == "mean_ridge":
        state = combined_ridge(cache, source, training, state)
        predicted = prediction_table(evaluator, state)
        best, best_epoch = score(predicted), 0
        history.append({"epoch": 0, "validation_epsilon_rmse": best[0] if main.any() else None})
        selection = "Exact ridge refit on source and permitted target labels; source-validation selected penalty."
    else:
        rows = training.row.to_numpy(dtype=int)
        x = (cache.batch(rows).astype(np.float64) - state["x_center"]) / state["x_scale"]
        y = (training.point.to_numpy(dtype=np.float64) - state["y_center"]) / state["y_scale"]
        x = torch.tensor(x, dtype=torch.float32, device="cuda")
        y = torch.tensor(y, dtype=torch.float32, device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=state["learning_rate"], weight_decay=state["weight_decay"])
        best, best_epoch = score(initial), -1
        best_state = deepcopy(model.state_dict())
        rng = np.random.default_rng(seed)
        maximum_epochs = 100 if use_validation else max(1, min(100, manifest["best_epoch"] + 1))
        for epoch in range(maximum_epochs):
            model.train()
            permutation = rng.permutation(len(rows))
            total, count = 0.0, 0
            for start in range(0, len(rows), 512):
                index = torch.as_tensor(permutation[start:start + 512], device="cuda")
                loss = (model(x[index]) - y[index]).square().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
                optimizer.step()
                total += float(loss.detach()) * len(index)
                count += len(index)
            assert count == len(rows)
            model.eval()
            current = score(prediction_table(evaluator, state, model)) if use_validation else (0.0, 0.0)
            if not use_validation or current < best:
                best, best_epoch, best_state = current, epoch, deepcopy(model.state_dict())
            history.append({"epoch": epoch, "training_mse": total / count,
                            "validation_epsilon_rmse": current[0] if main.any() else None,
                            "validation_phenotype_rmse": current[1] if point.any() else None})
            if use_validation and epoch - best_epoch >= 10:
                break
        model.load_state_dict(best_state)
        model.eval()
        state["model"] = {key: value.detach().cpu() for key, value in best_state.items()}
        predicted = prediction_table(evaluator, state, model)
        selection = "Target validation epsilon RMSE with phenotype tie-break." if main.any() else (
            "Target validation phenotype RMSE." if point.any() else "Fixed source-validation selected epoch count.")
    state.update(source_checkpoint=str(source / "best.pt"), best_epoch=best_epoch, seed=seed)
    torch.save(state, directory / "best.pt")
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    return predicted, {"epochs": len(history), "best_epoch": best_epoch, "selection": selection,
                       "source_training_variants": manifest["training_variants"],
                       "target_training_variants": len(training), "target_test_labels_used": 0,
                       "normalization": "Source and permitted target labels for ridge refit; frozen source statistics for MLP adaptation.",
                       "four_model_predictions": True}
