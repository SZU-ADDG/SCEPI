import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stabilityarc_common import CACHE, DATA, load_features, mutation_arrays, native_training_targets, score_arrays, source_frames
from scripts.train_single import family_permissions
from epistasis.gradient_validation import finite_nonzero_gradients
from stabilityarc.model import MutationTask, load_stabilityarc


def main():
    assert json.loads((CACHE / "manifest.json").read_text())["complete"]
    frames = source_frames()
    normalizations = {protein: native_training_targets(frame)[1] for protein, frame in frames.items()}
    for threshold in [30, 25]:
        test_sets = []
        for fold in range(5):
            permissions = {key: set(value) for key, value in family_permissions(threshold, fold).items()}
            assert not permissions["train"] & permissions["validation"]
            assert not permissions["train"] & permissions["test"]
            assert not permissions["validation"] & permissions["test"]
            test_sets.append(permissions["test"])
        assert sum(map(len, test_sets)) == len(set.union(*test_sets))
    proteins = pd.read_parquet(DATA / "proteins.parquet").set_index("protein_id")
    protein = next(protein for protein, frame in frames.items() if frame.n_mutations.eq(2).any())
    frame = frames[protein]
    sample = pd.concat([frame.loc[frame.n_mutations.eq(1)].head(12),
                        frame.loc[frame.n_mutations.eq(2)].head(12)], ignore_index=True)
    sequence = proteins.loc[protein, "reference_sequence"]
    embedding, contacts = load_features(protein, sequence, "cpu")
    arrays = mutation_arrays(sample, sequence, "cpu")
    model, _ = load_stabilityarc(ROOT / "external/stabilityarc/checkpoints/stabilityarc.pt", "cpu")
    model.eval()
    vectorized, _ = score_arrays(model, embedding, contacts, arrays)
    task = MutationTask(embedding, arrays[0], arrays[1])
    scores, site = model.score_task_components(task, torch.device("cpu"))
    edits = [[(int(position), int(residue)) for position, residue in zip(positions[:int(count)], residues[:int(count)])]
             for positions, residues, count in zip(*arrays)]
    native = model.score_edit_sets(scores, site, contacts, edits, torch.device("cpu"))
    torch.testing.assert_close(vectorized, native, rtol=1e-5, atol=1e-5)
    vectorized.square().mean().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    native.square().mean().backward()
    maximum_gradient_difference = 0.0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=5e-5, atol=2e-5)
        maximum_gradient_difference = max(maximum_gradient_difference, float((parameter.grad - gradients[name]).abs().max()))
    assert any(torch.count_nonzero(gradient) for name, gradient in gradients.items() if "pair_head" in name)
    assert any(torch.count_nonzero(gradient) for name, gradient in gradients.items() if "backbone" in name)
    legacy_nonzero = {name for name, parameter in model.named_parameters()
                      if parameter.grad is not None and torch.count_nonzero(parameter.grad)}
    assert finite_nonzero_gradients(model.named_parameters()) == legacy_nonzero
    assert len(sample) == 24 and np.isfinite(list(item["single_mad"] for item in normalizations.values())).all()
    output = ROOT / "runs/stabilityarc_validation"
    output.mkdir(parents=True, exist_ok=True)
    record = {"complete": True, "real_variants_checked": len(sample), "protein_id": protein,
              "forward_maximum_difference": float((vectorized - native).abs().max().detach()),
              "gradient_maximum_difference": maximum_gradient_difference,
              "source_assay_normalizations_checked": len(normalizations), "family_partitions_checked": 10,
              "batched_gradient_check_matches_scalar_checks": True,
              "reference_and_contact_cache": str((CACHE / "manifest.json").relative_to(ROOT)),
              "scope": "Actual measured variants, author checkpoint, exact vectorized forward and gradients, disjoint source permissions"}
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
