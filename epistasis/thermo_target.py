from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch

from epistasis.thermo_training import DoubleEvaluation, new_head


def predict_training(model, cache, mapping):
    if model.mode == "epistatic":
        result = torch.zeros(len(mapping), device="cuda")
        asymmetry = torch.zeros_like(result)
        observed = mapping[:, 0] >= 0
        if observed.any():
            rows = mapping[observed, 0]
            values, penalty = model(*cache.batch(rows))
            indices = torch.as_tensor(np.flatnonzero(observed), device="cuda")
            result = result.index_copy(0, indices, values)
            asymmetry = asymmetry.index_copy(0, indices, penalty)
        empty = ~observed
        if empty.any():
            count = int(empty.sum())
            values = torch.zeros((count, 2, 512), device="cuda")
            order = torch.zeros(count, dtype=torch.long, device="cuda")
            residues = torch.zeros((count, 2), dtype=torch.long, device="cuda")
            value, penalty = model(values, order, residues, residues)
            indices = torch.as_tensor(np.flatnonzero(empty), device="cuda")
            result = result.index_copy(0, indices, value)
            asymmetry = asymmetry.index_copy(0, indices, penalty)
        return result, asymmetry
    result = torch.zeros(len(mapping), device="cuda")
    for slot in range(2):
        selected = mapping[:, slot] >= 0
        if selected.any():
            value = model(*cache.batch(mapping[selected, slot]))[0]
            result = result.index_add(0, torch.as_tensor(np.flatnonzero(selected), device="cuda"), value)
    empty = (mapping < 0).all(axis=1)
    if empty.any():
        count = int(empty.sum())
        values = torch.zeros((count, 384), device="cuda")
        order = torch.zeros(count, dtype=torch.long, device="cuda")
        residues = torch.zeros((count, 2), dtype=torch.long, device="cuda")
        result = result.index_add(0, torch.as_tensor(np.flatnonzero(empty), device="cuda"), model(values, order, residues, residues)[0])
    return result, torch.zeros_like(result)


def training_mapping(table, cache):
    lookup = cache.table.set_index(["protein_id", "mutation_key"]).row
    mapping = np.full((len(table), 2), -1, dtype=int)
    for index, row in enumerate(table.itertuples()):
        if row.n_mutations == 0:
            continue
        if cache.mode == "epistatic":
            mapping[index, 0] = lookup[(row.protein_id, row.mutation_key)]
        else:
            for slot, (position, wild, mutant) in enumerate(zip(row.positions, row.wt_residues, row.mut_residues)):
                mapping[index, slot] = lookup[(row.protein_id, f"{wild}{int(position) + 1}{mutant}")]
    return mapping


def fit_target(source, cache, training, validation_ids, directory, seed):
    manifest = json.loads((source / "manifest.json").read_text())
    config = manifest["config"]
    model = new_head(cache.mode, seed)
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    mapping = training_mapping(training, cache)
    target = training.native_ddg.to_numpy(dtype=np.float32)
    assert len(target) >= 2 and np.isfinite(target).all()
    protein = training.protein_id.unique()
    assert len(protein) == 1
    evaluator = DoubleEvaluation(cache, protein)
    validation_mask = evaluator.table.variant_id.isin(validation_ids).to_numpy()
    main_mask = validation_mask & evaluator.table.main.to_numpy()
    use_epsilon = cache.mode == "epistatic" and main_mask.any()
    point_mask = validation_mask & evaluator.table.relative_point.notna().to_numpy()
    use_validation = use_epsilon or point_mask.any()

    def evaluate():
        prediction = evaluator.predict(model)
        if use_epsilon:
            primary = float(np.sqrt(np.mean((prediction.loc[main_mask, "prediction_epsilon"] -
                                               prediction.loc[main_mask, "epsilon"]) ** 2)))
        else:
            primary = 0.0
        secondary = float(np.sqrt(np.mean((prediction.loc[point_mask, "prediction_mu"] -
                                           prediction.loc[point_mask, "relative_point"]) ** 2))) if point_mask.any() else 0.0
        return (primary, secondary), prediction

    best, initial = evaluate()
    initial.to_parquet(directory / "initial_predictions.parquet", index=False)
    state = deepcopy(model.state_dict())
    best_epoch = -1
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5)
    rng, history = np.random.default_rng(seed), []
    torch.manual_seed(seed)
    maximum_epochs = 100 if use_validation else max(1, min(100, manifest["best_epoch"] + 1))
    for epoch in range(maximum_epochs):
        model.train()
        order = rng.permutation(len(training))
        seen, loss_sum = 0, 0.0
        for start in range(0, len(order), 256):
            rows = order[start:start + 256]
            prediction, asymmetry = predict_training(model, cache, mapping[rows])
            truth = torch.tensor(target[rows], device="cuda")
            loss = (prediction - truth).square().mean() + asymmetry.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
            optimizer.step()
            seen += len(rows)
            loss_sum += float(loss.detach()) * len(rows)
        assert seen == len(training)
        score = evaluate()[0] if use_validation else (0.0, 0.0)
        if use_validation:
            scheduler.step(score[0] if use_epsilon else score[1])
        if not use_validation or score < best:
            best, best_epoch = score, epoch
            state = deepcopy(model.state_dict())
        entry = {"epoch": epoch, "training_loss": loss_sum / seen,
                 "validation_epsilon_rmse": score[0] if use_epsilon else None,
                 "validation_phenotype_rmse": score[1] if point_mask.any() else None}
        history.append(entry)
        if use_validation and epoch - best_epoch >= 10:
            break
    model.load_state_dict(state)
    prediction = evaluator.predict(model)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    torch.save({"model": {key: value.detach().cpu() for key, value in state.items()},
                "source_checkpoint": str(source / "best.pt"), "mode": cache.mode,
                "seed": seed, "best_epoch": best_epoch}, directory / "best.pt")
    details = {"epochs": len(history), "best_epoch": best_epoch,
               "selection": "Target validation epsilon RMSE with phenotype RMSE tie-break." if use_epsilon else
                            ("Target validation phenotype RMSE." if use_validation else "Fixed source-validation selected epoch count."),
               "source_selected_learning_rate": config["learning_rate"],
               "source_selected_weight_decay": config["weight_decay"],
               "training_variants": len(training), "training_objective": "Author native-DDG MSE and symmetry penalty.",
               "interval_labels_imputed": 0, "counterfactuals": "Each double and both single genotypes use the same adapted head and reference output."}
    return prediction, details
