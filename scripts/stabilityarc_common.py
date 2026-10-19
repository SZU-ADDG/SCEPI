from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/stabilityarc"))
from stabilityarc.model import AMINO_ACID_TO_INDEX, StabilityArc

CACHE = ROOT / "data/features/stabilityarc/tsuboyama"
DATA = ROOT / "data/processed/tsuboyama"


def mutation_arrays(frame, sequence, device):
    positions = np.zeros((len(frame), 2), dtype=np.int64)
    residues = np.zeros_like(positions)
    counts = frame.n_mutations.to_numpy(dtype=np.int64)
    assert np.isin(counts, [0, 1, 2]).all()
    for index, row in enumerate(frame.itertuples()):
        count = int(row.n_mutations)
        assert len(row.positions) == len(row.mut_residues) == len(row.wt_residues) == count
        for endpoint, (position, wildtype, mutant) in enumerate(zip(row.positions, row.wt_residues, row.mut_residues)):
            assert sequence[int(position)] == wildtype and wildtype != mutant
            positions[index, endpoint] = int(position)
            residues[index, endpoint] = AMINO_ACID_TO_INDEX[mutant]
    return tuple(torch.as_tensor(value, device=device) for value in [positions, residues, counts])


def score_arrays(model, embedding, contacts, arrays):
    positions, residues, counts = arrays
    hidden, scores = model.backbone(embedding.unsqueeze(0))
    scores = scores.squeeze(0)
    site = model.site_projection(hidden.squeeze(0))
    prediction = scores[positions[:, 0], residues[:, 0]] * counts.ge(1)
    prediction = prediction + scores[positions[:, 1], residues[:, 1]] * counts.eq(2)
    active = counts.eq(2).nonzero(as_tuple=True)[0]
    if len(active):
        left_position, right_position = positions[active, 0], positions[active, 1]
        left = site[left_position] + model.pair_residue(residues[active, 0])
        right = site[right_position] + model.pair_residue(residues[active, 1])
        gap = (left_position - right_position).abs().to(left.dtype).unsqueeze(1) / max(len(site) - 1, 1)
        contact = contacts[left_position, right_position].unsqueeze(1)
        features = torch.cat([left + right, (left - right).abs(), left * right, gap, contact], dim=1)
        prediction = prediction.index_add(0, active, model.pair_head(features).squeeze(1))
    return prediction, scores


def source_frames():
    variants = pd.read_parquet(DATA / "variants.parquet", columns=[
        "variant_id", "protein_id", "n_mutations", "positions", "wt_residues", "mut_residues",
        "point", "all_point", "point_in_range",
    ])
    references = variants.loc[variants.n_mutations.eq(0)].set_index("protein_id")
    assert references.index.is_unique
    valid_references = references.index[references.all_point & references.point_in_range]
    frame = variants.loc[variants.protein_id.isin(valid_references)
                         & variants.n_mutations.isin([1, 2]) & variants.all_point & variants.point_in_range].copy()
    frame["relative_point"] = frame.point - frame.protein_id.map(references.point)
    assert np.isfinite(frame.relative_point).all()
    return {protein: group.reset_index(drop=True) for protein, group in frame.groupby("protein_id", sort=True)}


def native_training_targets(frame):
    targets = frame.relative_point.to_numpy(dtype=np.float32).copy()
    singles = targets[frame.n_mutations.eq(1).to_numpy()]
    assert len(singles) >= 2
    median = float(np.median(singles))
    mad = float(np.median(np.abs(singles - median)))
    assert np.isfinite(mad) and mad > 0
    doubles = frame.n_mutations.eq(2).to_numpy()
    targets[doubles] = (targets[doubles] - median) / mad
    return targets, {"single_median": median, "single_mad": mad}


def load_features(protein, sequence, device):
    payload = torch.load(CACHE / f"{protein}.pt", map_location="cpu", weights_only=True)
    assert payload["reference_sequence"] == sequence
    embedding, contacts = payload["embedding"], payload["contacts"]
    assert embedding.shape == (len(sequence), 1152) and contacts.shape == (len(sequence), len(sequence))
    assert torch.isfinite(embedding).all() and torch.isfinite(contacts).all()
    torch.testing.assert_close(contacts, contacts.T, rtol=0, atol=1e-6)
    assert torch.count_nonzero(contacts.diag()) == 0
    return embedding.to(device), contacts.to(device)
