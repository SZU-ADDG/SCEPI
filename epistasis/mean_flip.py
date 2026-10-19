from pathlib import Path

import numpy as np
import pandas as pd

from epistasis.mean_regression import MeanRegression, ROOT
from epistasis.metrics import correlation


class MeanFlipRegression(MeanRegression):
    def __init__(self, proteins, method, device):
        assert method in ["mean_ridge", "mean_mlp"]
        self.method, self.device = method, device
        directory = ROOT / "data/features/mean/flip_primary"
        metadata = pd.read_parquet(directory / "metadata.parquet")
        self.metadata = metadata.loc[metadata.protein_id.isin(proteins)].reset_index(drop=True)
        assert self.metadata.variant_id.is_unique and set(self.metadata.protein_id) == set(proteins)
        matrix = np.load(directory / "mean.npy", mmap_mode="r")
        values = [matrix[self.metadata.matrix_row.to_numpy(dtype=int)].astype(np.float32)]
        self.count = len(self.metadata)
        extra, aliases, row = [], [], self.count
        for protein in sorted(proteins):
            current = self.metadata.loc[self.metadata.protein_id.eq(protein)]
            observed = current.set_index("mutation_key")
            counterfactual = directory / "counterfactuals" / protein
            reference = np.load(counterfactual / "reference_mean.npy").astype(np.float32)
            values.append(reference.reshape(1, 1280))
            extra.append({"row": row, "protein_id": protein, "kind": "reference", "mutation_key": ""})
            reference_rows = current.index[current.n_mutations.eq(0)].tolist()
            assert len(reference_rows) <= 1
            if reference_rows:
                aliases.append((row, reference_rows[0]))
            row += 1
            singles = pd.read_parquet(counterfactual / "metadata.parquet")
            values.append(np.load(counterfactual / "single_mean.npy").astype(np.float32))
            for item in singles.itertuples():
                extra.append({"row": row, "protein_id": protein, "kind": "single", "mutation_key": item.mutation_key})
                if item.mutation_key in observed.index:
                    assert observed.loc[item.mutation_key, "n_mutations"] == 1
                    aliases.append((row, int(current.index[current.mutation_key.eq(item.mutation_key)][0])))
                row += 1
        self.counterfactuals = pd.DataFrame(extra)
        self.values = np.concatenate(values)
        assert len(self.values) == row and np.isfinite(self.values).all()
        self.prediction_alias = np.arange(len(self.values))
        for first, second in aliases:
            np.testing.assert_array_equal(self.values[first], self.values[second])
            self.prediction_alias[first] = second

    def score(self, predicted, validation, points, epsilon_components):
        assert epsilon_components is None
        correlations, errors = [], []
        selected = self.metadata.iloc[validation]
        for protein in selected.protein_id.unique():
            rows = validation[selected.protein_id.eq(protein).to_numpy()]
            rho = correlation(points[rows], predicted[rows], "spearman")
            if np.isfinite(rho):
                correlations.append(rho)
            errors.append(float(np.sqrt(np.mean((points[rows] - predicted[rows]) ** 2))))
        assert errors
        rho = float(np.mean(correlations)) if correlations else np.nan
        return (-rho if np.isfinite(rho) else float("inf"), float(np.mean(errors))), rho

    def save_predictions(self, predictions, directory):
        self.metadata.assign(prediction_absolute=predictions[:self.count]).to_parquet(
            directory / "all_variant_predictions.parquet", index=False)
        self.counterfactuals.assign(prediction_absolute=predictions[self.counterfactuals.row.to_numpy()]).to_parquet(
            directory / "counterfactual_predictions.parquet", index=False)


def jobs_for():
    split = pd.read_parquet(ROOT / "data/processed/flip_primary/splits.parquet")
    jobs = []
    for source_file, partition in split.groupby("source_file", sort=True):
        equivalent = partition.effective_set.eq(partition.strict_set).all()
        for protocol, column in [("official", "effective_set")] + ([] if equivalent else [("strict", "strict_set")]):
            for seed in [11, 23, 47, 89, 131]:
                training = partition.loc[partition[column].eq("train"), "variant_id"].tolist()
                validation = partition.loc[partition[column].eq("validation"), "variant_id"].tolist()
                testing = partition.loc[partition[column].eq("test"), "variant_id"].tolist()
                assert not set(training) & (set(validation) | set(testing))
                assert not set(validation) & set(testing)
                jobs.append({"scheme": Path(source_file).parent.name + "_" + partition.official_split.iloc[0],
                             "condition": protocol, "seed": seed,
                             "protocol_equivalence": ["official", "strict"] if equivalent else [protocol],
                             "train_ids": training, "validation_ids": validation, "test_ids": testing,
                             "training_proteins": sorted(partition.loc[partition[column].eq("train"), "protein_id"].unique()),
                             "validation_proteins": sorted(partition.loc[partition[column].eq("validation"), "protein_id"].unique()),
                             "test_proteins": sorted(partition.loc[partition[column].eq("test"), "protein_id"].unique())})
    return jobs


def save_evaluation(regression, job, directory):
    all_predictions = pd.read_parquet(directory / "all_variant_predictions.parquet")
    counterfactuals = pd.read_parquet(directory / "counterfactual_predictions.parquet")
    references = counterfactuals.loc[counterfactuals.kind.eq("reference")].set_index("protein_id").prediction_absolute
    singles = counterfactuals.loc[counterfactuals.kind.eq("single")].set_index(["protein_id", "mutation_key"]).prediction_absolute
    partition = {identifier: label for label in ["validation", "test"] for identifier in job[label + "_ids"]}
    all_predictions["prediction_mu"] = all_predictions.prediction_absolute - all_predictions.protein_id.map(references)
    all_predictions["set"] = all_predictions.variant_id.map(partition)
    predictions = all_predictions.loc[all_predictions["set"].notna()].copy()
    assert set(predictions.variant_id) == set(partition)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    summaries, quartets = [], []
    for (label, protein), group in predictions.groupby(["set", "protein_id"]):
        for order, selected in [("all", group)] + [(str(value), group.loc[group.n_mutations.eq(value)]) for value in [1, 2, 3, 4]] + [("5plus", group.loc[group.n_mutations.ge(5)])]:
            points = selected.loc[selected.point.notna()]
            summaries.append({"partition": label, "protein_id": protein, "mutation_order": order,
                              "variants": len(selected), "point_variants": len(points),
                              "phenotype_spearman": correlation(points.point, points.prediction_absolute, "spearman"),
                              "phenotype_rmse": float(np.sqrt(np.mean((points.point - points.prediction_absolute) ** 2))) if len(points) else None})
        observed = all_predictions.loc[all_predictions.protein_id.eq(protein)].set_index("mutation_key")
        wt = observed.loc[observed.n_mutations.eq(0) & observed.point.notna()]
        assert len(wt) <= 1
        if not len(wt):
            continue
        for row in group.loc[group.n_mutations.eq(2) & group.point.notna()].itertuples():
            keys = [f"{wild}{int(position) + 1}{mutant}" for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)]
            if not all(key in observed.index and np.isfinite(observed.loc[key, "point"]) for key in keys):
                continue
            a, b = (observed.loc[key] for key in keys)
            true = row.point - a.point - b.point + float(wt.point.iloc[0])
            predicted = row.prediction_absolute - singles[(protein, keys[0])] - singles[(protein, keys[1])] + references[protein]
            quartets.append({"variant_id": row.variant_id, "protein_id": protein, "set": label,
                             "wt_id": wt.variant_id.iloc[0], "a_id": a.variant_id, "b_id": b.variant_id,
                             "epsilon": true, "prediction_epsilon": predicted,
                             "prediction_ab": row.prediction_absolute, "prediction_a": singles[(protein, keys[0])],
                             "prediction_b": singles[(protein, keys[1])], "prediction_wt": references[protein]})
    pd.DataFrame(summaries).to_csv(directory / "metrics.csv", index=False)
    quartet_table = pd.DataFrame(quartets, columns=["variant_id", "protein_id", "set", "wt_id", "a_id", "b_id",
                                                  "epsilon", "prediction_epsilon", "prediction_ab", "prediction_a", "prediction_b", "prediction_wt"])
    quartet_table.to_parquet(directory / "quartet_predictions.parquet", index=False)
    if len(quartet_table):
        metrics = [{"partition": label, "protein_id": protein, "quartets": len(group),
                    "epsilon_rmse": float(np.sqrt(np.mean((group.prediction_epsilon - group.epsilon) ** 2))),
                    "epsilon_spearman": correlation(group.epsilon, group.prediction_epsilon, "spearman")}
                   for (label, protein), group in quartet_table.groupby(["set", "protein_id"])]
        pd.DataFrame(metrics).to_csv(directory / "quartet_metrics.csv", index=False)
    return {"evaluated_variants": len(predictions), "evaluated_proteins": sorted(predictions.protein_id.unique()),
            "complete_quartets": len(quartet_table), "official_test_set_preserved": True,
            "training_backgrounds": job["training_proteins"], "test_backgrounds": job["test_proteins"]}
