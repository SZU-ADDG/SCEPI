from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def single_features(protein, variants, cache_root):
    with h5py.File(cache_root / f"{protein.protein_id}_singles.h5", "r") as handle:
        assert handle.attrs["reference_sequence"] == protein.reference_sequence
        keys = handle["keys"].asstr()[:].tolist()
        index = {key: i for i, key in enumerate(keys)}
        wt = handle["wt"][:].astype(np.float32)
        single = handle["single"][:]
        log_probs = handle["wt_masked_log_probs"][:]
    positions = np.array([row.positions[0] for row in variants.itertuples()], dtype=np.int64)
    mutant_indices = np.array([index[key] for key in variants.mutation_key], dtype=np.int64)
    wt_aa = np.array([ALPHABET.index(protein.reference_sequence[position]) for position in positions], dtype=np.int64)
    mut_aa = np.array([ALPHABET.index(aa) for aa in variants.mut_residues], dtype=np.int64)
    return {
        "wt": wt[positions], "delta": single[mutant_indices, positions].astype(np.float32) - wt[positions],
        "wt_aa": wt_aa, "mut_aa": mut_aa,
        "score": log_probs[positions, mut_aa] - log_probs[positions, wt_aa],
    }


def pair_features(protein, variants, cache_root, geometry_root=None):
    with h5py.File(cache_root / f"{protein.protein_id}_singles.h5", "r") as handle:
        assert handle.attrs["reference_sequence"] == protein.reference_sequence
        keys = handle["keys"].asstr()[:].tolist()
        index = {key: i for i, key in enumerate(keys)}
        wt = handle["wt"][:].astype(np.float32)
        single = handle["single"][:]
        log_probs = handle["wt_masked_log_probs"][:]
    positions = np.array([row.positions for row in variants.itertuples()], dtype=np.int64)
    assert positions.shape == (len(variants), 2)
    mutant_indices = np.array([[index[key] for key in mutation.split(":")] for mutation in variants.mutation_key], dtype=np.int64)
    wt_aa = np.array([[ALPHABET.index(protein.reference_sequence[position]) for position in pair] for pair in positions], dtype=np.int64)
    mut_aa = np.array([[ALPHABET.index(aa) for aa in row.mut_residues] for row in variants.itertuples()], dtype=np.int64)
    result = {
        "wt": wt[positions],
        "delta": single[mutant_indices, positions].astype(np.float32) - wt[positions],
        "cross": single[mutant_indices[:, ::-1], positions].astype(np.float32) - wt[positions],
        "wt_aa": wt_aa, "mut_aa": mut_aa,
        "score": log_probs[positions, mut_aa] - log_probs[positions, wt_aa],
        "sequence_gap": np.diff(positions, axis=1)[:, 0].astype(np.float32),
        "sequence_length": np.full(len(variants), len(protein.reference_sequence), dtype=np.float32),
    }
    if geometry_root is not None:
        geometry = np.load(geometry_root / f"{protein.protein_id}.npz")
        coordinates = geometry["coordinates"][positions]
        result["ca_distance"] = np.linalg.norm(coordinates[:, 0] - coordinates[:, 1], axis=-1)
        result["confidence"] = geometry["confidence"][positions]
        assert np.isfinite(result["ca_distance"]).all() and np.isfinite(result["confidence"]).all()
    return result


def to_torch(data, device="cpu"):
    return {key: torch.as_tensor(value, device=device) for key, value in data.items()}
