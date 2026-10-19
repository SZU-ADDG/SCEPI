from pathlib import Path
import json
import os
import subprocess

from Bio.Seq import MutableSeq
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]


class RFARegression:
    def __init__(self, features, dataset, accelerated=False, optimizer_policy=None):
        self.features = features
        self.accelerated = accelerated
        self.optimizer_policy = Path(optimizer_policy).resolve() if optimizer_policy is not None else None
        self.fixed_default = False
        if self.optimizer_policy is not None:
            assert self.accelerated and self.optimizer_policy.is_relative_to(ROOT)
            self.fixed_default = json.loads(self.optimizer_policy.read_text())["penalty_candidates"] == 1
        proteins = pd.read_parquet(ROOT / "data/processed" / dataset / "proteins.parquet").set_index("protein_id")
        protein = features.metadata.protein_id.unique()
        assert len(protein) == 1
        self.reference = proteins.loc[protein[0], "reference_sequence"]

    def fit(self, train_indices, validation_indices, points, seed, directory, epsilon_components=None):
        assert not set(train_indices) & set(validation_indices)
        assert len(train_indices) >= 2 and len(validation_indices) >= 2
        features = self.features
        sequences = features.metadata.aa_seq.tolist() + [self.reference]
        for row in features.single_metadata.itertuples():
            sequence = MutableSeq(self.reference)
            assert self.reference[row.positions[0]] == row.wt_residues
            sequence[row.positions[0]] = row.mut_residues
            sequences.append(str(sequence))
        positions = sorted({int(row.positions[0]) for row in features.single_metadata.itertuples()})
        residues = np.asarray([[sequence[position] for position in positions] for sequence in sequences])
        encoder = OrdinalEncoder(dtype=np.int64)
        genotypes = encoder.fit_transform(residues) + 1
        assert all(len(category) >= 2 for category in encoder.categories_)
        pd.DataFrame(genotypes).to_csv(directory / "genotypes.csv", header=False, index=False)
        pd.DataFrame({"point": np.concatenate([points, np.full(len(sequences) - len(points), np.nan)])}).to_csv(directory / "points.csv", index=False)
        pd.concat([pd.DataFrame({"row_index": train_indices + 1, "role": "train"}),
                   pd.DataFrame({"row_index": validation_indices + 1, "role": "validation"})]).to_csv(directory / "row_permissions.csv", index=False)
        pd.DataFrame({"sequence_position": positions, "allowed_residues": ["".join(category) for category in encoder.categories_]}).to_csv(directory / "genotype_vocabulary.csv", index=False)
        if epsilon_components is not None:
            indices, truth = epsilon_components
            frame = pd.DataFrame(indices + 1, columns=["wt", "a", "b", "ab"])
            frame["epsilon"] = truth
            frame.to_csv(directory / "validation_quartets.csv", index=False)
        command = ["Rscript", str(ROOT / "scripts/fit_rfa.R"), str(directory), str(seed)]
        if self.optimizer_policy is not None:
            relative = directory.relative_to(ROOT / "runs" / ("rfa_fixed" if self.fixed_default else "rfa_bounded"))
            prefixes = ["rfa_bounded", "rfa_accelerated", "rfa"] if self.fixed_default else ["rfa_accelerated", "rfa"]
            sources = [ROOT / "runs" / prefix / relative for prefix in prefixes]
            (directory / "prior_sources.json").write_text(json.dumps([str(path) for path in sources if path.is_dir()], indent=2))
        with (directory / "author_fit.log").open("a" if self.accelerated else "w") as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env={**os.environ, "RECOMB_RFA_ACCELERATION": "1" if self.accelerated else "0",
                                "RECOMB_RFA_POLICY": str(self.optimizer_policy) if self.optimizer_policy is not None else ""})
        output = pd.read_csv(directory / "raw_predictions.csv")
        np.testing.assert_array_equal(output.row_index, np.arange(1, len(sequences) + 1))
        values = output.prediction.to_numpy()
        assert np.isfinite(values).all()
        count = len(features.metadata)
        np.savez_compressed(directory / "predictions.npz", variants=values[:count],
                            reference=values[count], singles=values[count + 1:])
        summary = pd.read_csv(directory / "fit_summary.csv").iloc[0].to_dict()
        budget = json.loads((directory / "budget_summary.json").read_text()) if self.optimizer_policy is not None else {}
        return {**summary, "model": "RFA second-order effects with sigmoid nonspecific epistasis",
                "author_program": "external/RFA/Scripts/RFA-functions.R",
                "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/RFA"), "rev-parse", "HEAD"], text=True).strip(),
                "reference_free_effects": True, "source_labels_used": 0,
                "normalization_labels": "Training biological-variant point estimates only.",
                "genotype_vocabulary": "Sequence inputs and prediction-only counterfactual genotypes; no phenotype values.",
                "target_labels_used": len(train_indices), "validation_labels_used": len(validation_indices),
                "device": "cpu", "command": command,
                "accelerated": self.accelerated,
                "bounded_optimization": self.optimizer_policy is not None,
                "optimization_budget": budget,
                "budget_validation": f"runs/{'rfa_fixed' if self.fixed_default else 'rfa_bounded'}_20261018/validation.json" if self.optimizer_policy is not None else None,
                "fixed_default": self.fixed_default,
                "implementation": "Same libLBFGS solver with vectorized parameter extraction, cached numeric sparse matrices and crossprod gradients." if self.accelerated else "Original author implementation.",
                "acceleration_validation": "runs/rfa_acceleration_20261018/validation.json" if self.accelerated else None}
