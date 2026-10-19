from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epistasis.model import EpistasisModel, ModelConfig, ObservationModel, PairEffect, PhenotypeMap, SingleEffect

ROOT = Path(__file__).resolve().parents[1]


def fit_target_main(source, directory, singles, doubles, raw, validation, seed):
    if (directory / "manifest.json").exists():
        information = json.loads((directory / "manifest.json").read_text())
        assert information["complete"] and information["source_training_labels_used"] == 0
        observed = pd.read_parquet(directory / "training_observations.parquet")
        assert set(observed.observation_id) == set(raw.loc[raw.n_mutations.le(1), "observation_id"])
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    source_info = json.loads((source / "manifest.json").read_text())
    single_info = json.loads((ROOT / source_info["single_run"] / "manifest.json").read_text())
    assert not source_info["config"]["flexible"]
    single_config = ModelConfig(**source_info["single_config"])
    assert not single_config.geometry
    pair_config = ModelConfig(**source_info["pair_config"])
    torch.manual_seed(seed)
    model = EpistasisModel(SingleEffect(single_config), PairEffect(pair_config), PhenotypeMap(False)).cuda()
    noise = ObservationModel().cuda()
    model.pair.requires_grad_(False)
    original_pair = deepcopy(model.pair.state_dict())
    initial_single = deepcopy(model.single.state_dict())
    training = raw.loc[raw.n_mutations.le(1)].copy()
    assert len(training) and training.observation_id.is_unique and training.n_mutations.eq(1).any()
    assert not set(training.variant_id) & set(validation.variant_id)
    training[["observation_id", "variant_id", "protein_id", "n_mutations", "lower", "upper"]].to_parquet(
        directory / "training_observations.parquet", index=False)
    validation = validation.loc[validation.relative_point.notna()].copy()
    validation[["variant_id", "protein_id"]].to_parquet(directory / "validation_variants.parquet", index=False)
    use_validation = len(validation) > 0
    maximum_epochs = 100 if use_validation else max(1, min(100, single_info["best_epoch"] + 1))
    optimizer = torch.optim.AdamW([*model.single.parameters(), *noise.parameters()],
                                  lr=single_info["config"]["learning_rate"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_epoch, history, nonzero_gradient_batches = float("inf"), -1, [], 0
    for epoch in range(maximum_epochs):
        model.train()
        permutation = rng.permutation(len(training))
        total, count = 0.0, 0
        for start in range(0, len(training), 1024):
            batch = training.iloc[permutation[start:start + 1024]]
            active = batch.n_mutations.eq(1).to_numpy()
            rows = batch.loc[active, "matrix_row"].to_numpy(dtype=int)
            prediction = torch.zeros(len(batch), device="cuda")
            if len(rows):
                prediction = prediction.index_copy(0, torch.as_tensor(np.flatnonzero(active), device="cuda"),
                                                    model.single(singles.batch(rows)))
            predicted = prediction + torch.tensor(batch.reference_point.to_numpy(), device="cuda")
            loss = noise.nll(predicted, torch.tensor(batch.lower.to_numpy(), device="cuda"),
                             torch.tensor(batch.upper.to_numpy(), device="cuda"),
                             torch.tensor(batch.measurement_status.eq("point").to_numpy(), device="cuda")).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in model.single.parameters() if parameter.grad is not None]
            assert all(torch.isfinite(value).all() for value in gradients)
            nonzero_gradient_batches += int(any(torch.count_nonzero(value) for value in gradients))
            assert noise.sigma_raw.grad is not None and torch.isfinite(noise.sigma_raw.grad)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        assert count == len(training)
        score = None
        if use_validation:
            model.eval()
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(validation), 1024):
                    rows = validation.matrix_row.iloc[start:start + 1024].to_numpy(dtype=int)
                    predictions.append(model.single(doubles.batch(rows)).sum(-1).cpu().numpy())
            values = np.concatenate(predictions)
            score = float(np.sqrt(np.mean((values - validation.relative_point.to_numpy()) ** 2)))
            assert np.isfinite(score)
        if not use_validation or score < best:
            best, best_epoch = score if use_validation else None, epoch
            state = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                     "noise": {name: value.detach().cpu().clone() for name, value in noise.state_dict().items()},
                     "single_config": asdict(single_config), "pair_config": asdict(pair_config), "flexible": False,
                     "omit_pair": False, "epoch": epoch, "validation_score": score, "omega": source_info["omega"],
                     "source_training_labels_used": 0, "source_validation_hyperparameters_used": True}
            torch.save(state, directory / "best.pt")
            if use_validation:
                validation.assign(prediction=values).to_parquet(directory / "validation_predictions.parquet", index=False)
        history.append({"epoch": epoch, "training_nll": total / count, "validation_additive_rmse": score,
                        "raw_observations_seen": count})
        if use_validation and epoch - best_epoch >= 10:
            break
    assert best_epoch >= 0 and nonzero_gradient_batches
    assert any(not torch.equal(value, initial_single[name]) for name, value in model.single.state_dict().items())
    assert all(torch.equal(value, original_pair[name]) for name, value in model.pair.state_dict().items())
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    manifest = {"complete": True, "stage": "target_single_effect_training", "config": source_info["config"],
                "omega": source_info["omega"], "best_epoch": best_epoch,
                "pair_epoch_budget": max(1, min(100, source_info["best_epoch"] + 1)),
                "epochs": len(history), "training_observations": len(training),
                "training_variants": training.variant_id.nunique(), "training_proteins": training.protein_id.unique().tolist(),
                "source_training_labels_used": 0, "supervised_source_weights_used": False,
                "source_validation_hyperparameters_used": True, "hyperparameter_source_run": str(source.relative_to(ROOT)),
                "selection": "Target validation additive phenotype RMSE." if use_validation else "Fixed source-validation selected single-head epoch count.",
                "single_learning_rate": single_info["config"]["learning_rate"], "nonzero_gradient_batches": nonzero_gradient_batches,
                "pair_weights_unchanged": True, "reference_anchor": "Permitted measured target WT in the continuous range."}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return directory
