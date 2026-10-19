from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "external/ThermoMPNN-D"
sys.path.insert(0, str(VENDOR))

from thermompnn.datasets.dataset_utils import Mutation
from thermompnn.datasets.v2_datasets import parse_PDB, tied_featurize_mut
from thermompnn.ssm_utils import get_config
from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def new_model(mode, seed):
    torch.manual_seed(seed)
    configuration = get_config(mode)
    configuration.platform.thermompnn_dir = str(VENDOR)
    configuration.project = None
    configuration.model.proteinmpnn_dropout = 0.0
    cls = TransferModelPLv2Siamese if mode == "epistatic" else TransferModelPLv2
    model = cls(configuration).model
    assert all(not value.requires_grad for value in model.prot_mpnn.parameters())
    assert model.prot_mpnn.features.augment_eps == 0
    return model, configuration


@lru_cache(maxsize=512)
def load_structure(path):
    path = ROOT / path
    assert path.is_relative_to(ROOT) and path.is_file()
    if path.suffix == ".pt":
        structure = torch.load(path, map_location="cpu", weights_only=False)
    else:
        structure = parse_PDB(str(path), input_chain_list=["A"])[0]
    assert set(key for key in structure if key.startswith("seq_chain_")) == {"seq_chain_A"}
    return structure


class MeasuredTransitions(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        structure = deepcopy(load_structure(row.structure))
        assert structure["seq"] == row.background_sequence
        positions = [int(value) for value in row.positions]
        assert all(structure["seq"][position] == wild for position, wild in zip(positions, row.wt_residues))
        assert len(positions) == len(row.wt_residues) == len(row.mut_residues) and len(positions) > 0
        structure["mutation"] = Mutation(positions, list(row.wt_residues), list(row.mut_residues),
                                          float("nan"), row.protein_id)
        return structure


def tensor_batch(examples):
    values = tied_featurize_mut(examples, side_chains=False)
    X, S, mask, lengths, chain_M, chain_encoding, residue_idx, positions, wildtype, mutant, measured, atom_mask = values
    return [value.cuda() for value in [X, S, mask, chain_M, residue_idx, chain_encoding,
                                       positions, wildtype, mutant, measured, torch.as_tensor(atom_mask)]]


class CachedThermo(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.mode = mode
        self.light_attention = original.light_attention
        self.ddg_out = original.ddg_out

    def forward(self, features, counts, wildtype, mutant):
        if self.mode == "epistatic":
            assert features.ndim == 3 and features.shape[1:] == (2, 512)
            inputs = features[:, :1] if counts.eq(1).all() else features
            values = self.light_attention(inputs.transpose(0, 1))
            if values.shape[0] == 1:
                values = torch.cat([values, torch.zeros_like(values)], dim=0)
            present = torch.arange(2, device=features.device)[:, None] < counts[None, :]
            values = values * present[..., None]
            first = self.ddg_out(torch.cat([values[0], values[1]], dim=-1)).flatten()
            second = self.ddg_out(torch.cat([values[1], values[0]], dim=-1)).flatten()
            return (first + second) * 0.5, (first - second).abs() * 0.5
        assert features.ndim == 2 and features.shape[1] == 384
        values = self.ddg_out(self.light_attention(features.unsqueeze(-1)))
        native = values.gather(1, mutant[:, :1]) - values.gather(1, wildtype[:, :1])
        return native.flatten(), torch.zeros(len(features), device=features.device)


def capture(original, mode, batch):
    seen = []
    handle = original.light_attention.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
    first, second = original(*batch)
    handle.remove()
    assert len(seen) == 1
    if mode == "epistatic":
        values = seen[0].transpose(0, 1)
        if values.shape[1] == 1:
            values = torch.cat([values, torch.zeros_like(values)], dim=1)
        predicted = (first + second).flatten() * 0.5
    else:
        values = seen[0].squeeze(-1)
        predicted = first.flatten()
    assert torch.isfinite(values).all() and torch.isfinite(predicted).all()
    return values, predicted


def verify_cached_head(original, mode, batch, counts, targets):
    original.eval()
    wrapper = CachedThermo(original, mode).cuda().eval()
    with torch.no_grad():
        features, observed = capture(original, mode, batch)
        predicted, _ = wrapper(features, counts, batch[7], batch[8])
        torch.testing.assert_close(predicted, observed, rtol=1e-6, atol=1e-5)
        repeated_features, repeated = capture(original, mode, [value.clone() for value in batch])
        torch.testing.assert_close(features, repeated_features, rtol=0, atol=1e-5)
        torch.testing.assert_close(observed, repeated, rtol=0, atol=1e-5)
    original.train()
    original.prot_mpnn.eval()
    original.zero_grad(set_to_none=True)
    torch.manual_seed(20261009)
    first, second = original(*[value.clone() for value in batch])
    native = (first + second).flatten() * 0.5 if mode == "epistatic" else first.flatten()
    assert torch.isfinite(targets).all()
    loss = (native - targets).square().mean()
    if mode == "epistatic":
        loss = loss + (first - second).abs().mean() * 0.5
    loss.backward()
    gradients = {name: value.grad.detach().clone() for name, value in wrapper.named_parameters() if value.grad is not None}
    original.zero_grad(set_to_none=True)
    torch.manual_seed(20261009)
    native, asymmetry = wrapper(features, counts, batch[7], batch[8])
    loss_cached = (native - targets).square().mean() + asymmetry.mean()
    loss_cached.backward()
    for name, value in wrapper.named_parameters():
        if name in gradients:
            assert value.grad is not None
            torch.testing.assert_close(value.grad, gradients[name], rtol=1e-5, atol=1e-5)
    original.zero_grad(set_to_none=True)
    original.eval()
    return {"verified": True, "variants": len(counts), "maximum_forward_difference": float((predicted - observed).abs().max()),
            "compared_parameter_gradients": len(gradients), "frozen_encoder_deterministic": True,
            "measurement_argument": "NaN during feature extraction; no phenotype labels enter the encoder."}
