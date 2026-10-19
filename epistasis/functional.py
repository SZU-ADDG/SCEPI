from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from epistasis.higher import HigherFeatures
from epistasis.metrics import correlation
from epistasis.model import ObservationModel, PhenotypeMap


class FunctionalData:
    def __init__(self, root, dataset, proteins):
        self.features, frames = {}, []
        for protein in sorted(proteins):
            current = HigherFeatures(Path(root) / "data/features/higher" / dataset / protein)
            self.features[protein] = current
            frames.append(current.metadata.assign(local_row=np.arange(len(current.metadata))))
        self.metadata = pd.concat(frames, ignore_index=True)
        assert self.metadata.variant_id.is_unique
        self.lookup = pd.Series(np.arange(len(self.metadata)), index=self.metadata.variant_id)
        self.proteins = sorted(self.features)
        self.protein_index = {protein: index for index, protein in enumerate(self.proteins)}

    def quartets(self, training_indices):
        allowed = self.metadata.iloc[training_indices]
        allowed = allowed.loc[allowed.all_point & allowed.point.notna()]
        records = []
        for protein, table in allowed.groupby("protein_id"):
            wt = table.loc[table.n_mutations.eq(0)]
            if len(wt) != 1:
                continue
            assert not table.mutation_key.duplicated().any()
            by_mutation = table.set_index("mutation_key")
            for row in table.loc[table.n_mutations.eq(2)].itertuples():
                keys = [f"{wild}{position + 1}{mutant}" for position, wild, mutant in
                        zip(row.positions, row.wt_residues, row.mut_residues)]
                if not all(key in by_mutation.index for key in keys):
                    continue
                first, second = by_mutation.loc[keys[0]], by_mutation.loc[keys[1]]
                records.append({"wt": int(wt.index[0]), "a": int(self.lookup[first.variant_id]),
                                "b": int(self.lookup[second.variant_id]), "ab": int(self.lookup[row.variant_id]),
                                "epsilon": row.point - first.point - second.point + float(wt.point.iloc[0]),
                                "protein_id": protein, "pair_id": tuple(row.positions)})
        result = pd.DataFrame(records, columns=["wt", "a", "b", "ab", "epsilon", "protein_id", "pair_id"])
        if len(result):
            pair_size = result.groupby(["protein_id", "pair_id"]).ab.transform("size")
            pair_count = result.groupby("protein_id").pair_id.transform("nunique")
            result["sampling_weight"] = 1.0 / (pair_size * pair_count)
            result.sampling_weight /= result.sampling_weight.sum()
            assert set(result[["wt", "a", "b", "ab"]].to_numpy().ravel()) <= set(training_indices)
        return result


class FunctionalModel(nn.Module):
    def __init__(self, sources, data, train_indices, center, scale):
        super().__init__()
        self.sources = nn.ModuleList(sources)
        self.mapping = PhenotypeMap(True)
        self.center, self.scale = float(center), float(scale)
        self.protein_index = data.protein_index
        training = data.metadata.iloc[train_indices]
        anchors, masks, identifiable = [], [], {}
        for protein in data.proteins:
            current = training.loc[training.protein_id.eq(protein)]
            reference = current.loc[current.n_mutations.eq(0) & current.point.notna()]
            assert len(reference) <= 1
            anchors.append((float(reference.point.iloc[0]) - center) / scale if len(reference) else 0.0)
            masks.append(float(len(reference) == 0 and len(current) > 0))
            identifiable[protein] = bool(len(current) > 0)
        self.offsets = nn.Parameter(torch.zeros(len(anchors)))
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.register_buffer("offset_masks", torch.tensor(masks, dtype=torch.float32))
        self.identifiable = identifiable

    def forward(self, data, indices, remove_interaction=False):
        device = self.anchors.device
        result = {key: torch.zeros(len(indices), device=device) for key in ["phenotype", "relative", "additive", "interaction"]}
        subset = data.metadata.iloc[indices]
        for protein in subset.protein_id.unique():
            positions = np.flatnonzero(subset.protein_id.to_numpy() == protein)
            rows = subset.iloc[positions].local_row.to_numpy()
            terms = [data.features[protein].forward(source, rows, remove_interaction) for source in self.sources]
            additive = torch.stack([entry["additive_latent"] for entry in terms]).mean(0)
            interaction = torch.stack([entry["interaction_latent"] for entry in terms]).mean(0)
            relative = self.mapping(additive + interaction)
            index = self.protein_index[protein]
            anchor = self.anchors[index] + self.offset_masks[index] * self.offsets[index]
            values = {"phenotype": relative + anchor, "relative": relative, "additive": additive, "interaction": interaction}
            for key in result:
                result[key] = result[key].index_copy(0, torch.as_tensor(positions, device=device), values[key])
        return result

    def stage(self, name):
        assert name in ["main", "interaction"]
        for source in self.sources:
            source.single.requires_grad_(name == "main")
            source.pair.requires_grad_(name == "interaction")
        self.mapping.requires_grad_(name == "main")
        self.offsets.requires_grad_(name == "main")

    def fixed_main_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if ".single." in name or name.startswith("mapping.") or name in ["offsets", "anchors", "offset_masks"]}


def batch_size(data, indices, stage):
    maximum = int(data.metadata.iloc[indices].n_mutations.max())
    bound = 2048 // max(1, maximum)
    if stage == "interaction":
        bound = min(bound, 1024 // max(1, math.comb(maximum, 2)))
    return max(1, min(1024, bound))


def predict(model, data, indices, remove_interaction=False):
    model.eval()
    frames = []
    size = batch_size(data, indices, "main" if remove_interaction else "interaction")
    with torch.inference_mode():
        for start in range(0, len(indices), size):
            rows = indices[start:start + size]
            value = model(data, rows, remove_interaction)
            frame = data.metadata.iloc[rows][["variant_id", "protein_id", "point", "n_mutations"]].copy()
            frame["prediction_absolute"] = value["phenotype"].cpu().numpy() * model.scale + model.center
            frame["prediction_relative"] = value["relative"].cpu().numpy() * model.scale
            frame["additive_latent"] = value["additive"].cpu().numpy()
            frame["interaction_latent"] = value["interaction"].cpu().numpy()
            frame["absolute_identifiable"] = frame.protein_id.map(model.identifiable)
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    assert np.isfinite(result[["prediction_absolute", "prediction_relative"]]).all().all()
    return result


def validation_score(model, data, indices, remove_interaction):
    predictions = predict(model, data, indices, remove_interaction)
    values, errors = [], []
    for _, group in predictions.groupby("protein_id"):
        group = group.loc[group.point.notna()]
        rho = correlation(group.point, group.prediction_relative, "spearman")
        if np.isfinite(rho):
            values.append(rho)
        if len(group) and group.absolute_identifiable.all():
            errors.append(float(np.sqrt(np.mean((group.point - group.prediction_absolute) ** 2))))
    assert errors
    return (-float(np.mean(values)) if values else float("inf"), float(np.mean(errors)))


def fit_stage(model, data, raw, validation_indices, quartets, learning_rate, lambda_y, seed, stage, noise_state=None):
    device = next(model.parameters()).device
    model.stage(stage)
    noise = ObservationModel().to(device)
    if noise_state is not None:
        noise.load_state_dict(noise_state)
    parameters = [value for value in model.parameters() if value.requires_grad] + list(noise.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    raw = raw.copy()
    raw["matrix_row"] = raw.variant_id.map(data.lookup)
    assert raw.matrix_row.notna().all() and raw.observation_id.is_unique
    rows = raw.matrix_row.to_numpy(dtype=np.int64)
    size = batch_size(data, rows, stage)
    rng = np.random.default_rng(seed)
    remove = stage == "main"
    best = validation_score(model, data, validation_indices, remove)
    state = ({key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
             deepcopy(noise.state_dict()))
    best_epoch, history = -1, []
    fixed = model.fixed_main_state() if stage == "interaction" else None
    for epoch in range(100):
        model.train()
        permutation = rng.permutation(len(raw))
        count, total_nll, total_huber, steps = 0, 0.0, 0.0, 0
        for start in range(0, len(raw), size):
            batch = raw.iloc[permutation[start:start + size]]
            prediction = model(data, batch.matrix_row.to_numpy(dtype=np.int64), remove)["phenotype"]
            lower = torch.tensor((batch.lower.to_numpy() - model.center) / model.scale, device=device)
            upper = torch.tensor((batch.upper.to_numpy() - model.center) / model.scale, device=device)
            point = torch.tensor(batch.measurement_status.eq("point").to_numpy(), device=device)
            nll = noise.nll(prediction, lower, upper, point).mean()
            huber = torch.zeros((), device=device)
            if stage == "interaction" and len(quartets):
                selection = rng.choice(len(quartets), size=min(size, 1024), replace=True, p=quartets.sampling_weight.to_numpy())
                chosen = quartets.iloc[selection]
                components = chosen[["wt", "a", "b", "ab"]].to_numpy(dtype=np.int64)
                unique = np.unique(components)
                phenotype = model(data, unique)["relative"]
                component = phenotype[torch.tensor(np.searchsorted(unique, components), device=device)]
                epsilon = component[:, 3] - component[:, 1] - component[:, 2] + component[:, 0]
                truth = torch.tensor(chosen.epsilon.to_numpy() / model.scale, dtype=torch.float32, device=device)
                huber = F.huber_loss(epsilon, truth, delta=1.0)
            loss = nll if stage == "main" else huber + lambda_y * nll
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(value.grad).all() for value in parameters if value.grad is not None)
            optimizer.step()
            count += len(batch)
            steps += 1
            total_nll += float(nll.detach())
            total_huber += float(huber.detach())
        assert count == len(raw)
        score = validation_score(model, data, validation_indices, remove)
        history.append({"epoch": epoch, "nll": total_nll / steps, "huber": total_huber / steps,
                        "validation_spearman": -score[0], "validation_rmse": score[1], "observations": count})
        if score < best:
            best, best_epoch = score, epoch
            state = ({key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                     deepcopy(noise.state_dict()))
        if epoch - best_epoch >= 10:
            break
    model.load_state_dict(state[0])
    if fixed is not None:
        assert all(torch.equal(value, model.fixed_main_state()[name]) for name, value in fixed.items())
    return {"model": state[0], "noise": state[1], "best_epoch": best_epoch, "score": best,
            "history": history, "batch_size": size, "learning_rate": learning_rate, "lambda_y": lambda_y,
            "stage": stage, "training_observations_per_epoch": len(raw), "training_quartets": len(quartets)}
