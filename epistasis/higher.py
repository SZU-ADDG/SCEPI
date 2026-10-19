from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch


class HigherFeatures:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = pd.read_parquet(self.directory / "variants.parquet")
        self.single_metadata = pd.read_parquet(self.directory / "single/metadata.parquet")
        self.pair_metadata = pd.read_parquet(self.directory / "pair/metadata.parquet")
        self.single_arrays = self.arrays("single")
        self.pair_arrays = self.arrays("pair")
        self.single_incidence = sparse.load_npz(self.directory / "single_incidence.npz").tocsr()
        self.pair_incidence = sparse.load_npz(self.directory / "pair_incidence.npz").tocsr()
        self.pair_alleles = np.stack(self.pair_metadata.allele_indices).astype(np.int64)
        assert self.single_incidence.shape == (len(self.metadata), len(self.single_metadata))
        assert self.pair_incidence.shape == (len(self.metadata), len(self.pair_metadata))

    def arrays(self, kind):
        return {path.stem: np.load(path, mmap_mode="r") for path in (self.directory / kind).glob("*.npy")}

    def batch(self, kind, indices, device):
        arrays = self.single_arrays if kind == "single" else self.pair_arrays
        return {key: torch.from_numpy(np.array(value[indices], copy=True)).to(device) for key, value in arrays.items()}

    def forward(self, model, indices, remove_interaction=False):
        device = next(model.parameters()).device
        single = self.single_incidence[indices].tocoo()
        pair = self.pair_incidence[indices].tocoo()
        pair_ids = np.unique(pair.col)
        allele_ids = np.union1d(single.col, self.pair_alleles[pair_ids].ravel())
        effects = model.single(self.batch("single", allele_ids, device))
        additive = torch.zeros(len(indices), device=device)
        if len(single.data):
            values = effects[torch.as_tensor(np.searchsorted(allele_ids, single.col), device=device)]
            additive = additive.index_add(0, torch.as_tensor(single.row, device=device), values)
        interaction = torch.zeros_like(additive)
        if len(pair_ids) and not remove_interaction:
            pair_effects = effects[torch.as_tensor(np.searchsorted(allele_ids, self.pair_alleles[pair_ids]), device=device)]
            observed = torch.zeros_like(pair_effects, dtype=torch.bool)
            pair_values = model.pair(self.batch("pair", pair_ids, device), pair_effects, observed)
            values = pair_values[torch.as_tensor(np.searchsorted(pair_ids, pair.col), device=device)]
            interaction = interaction.index_add(0, torch.as_tensor(pair.row, device=device), values)
        latent = additive + interaction
        return {"phenotype": model.mapping(latent), "latent": latent,
                "additive_latent": additive, "interaction_latent": interaction}

    def infer_terms(self, model, batch_size=1024):
        device = next(model.parameters()).device
        model.eval()
        with torch.inference_mode():
            effects = torch.cat([
                model.single(self.batch("single", np.arange(start, min(start + batch_size, len(self.single_metadata))), device))
                for start in range(0, len(self.single_metadata), batch_size)
            ])
            interactions = []
            for start in range(0, len(self.pair_metadata), batch_size):
                indices = np.arange(start, min(start + batch_size, len(self.pair_metadata)))
                pair_effects = effects[torch.as_tensor(self.pair_alleles[indices], device=device)]
                interactions.append(model.pair(self.batch("pair", indices, device), pair_effects,
                                               torch.zeros_like(pair_effects, dtype=torch.bool)).cpu().numpy())
        return effects.cpu().numpy(), np.concatenate(interactions)

    def combine(self, effects, interactions):
        assert effects.shape == (len(self.single_metadata),)
        assert interactions.shape == (len(self.pair_metadata),)
        additive = self.single_incidence @ effects
        interaction = self.pair_incidence @ interactions
        return additive, interaction
