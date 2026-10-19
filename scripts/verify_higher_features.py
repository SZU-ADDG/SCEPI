import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.higher import HigherFeatures
from evaluate_family import load_model


def main():
    torch.set_float32_matmul_precision("highest")
    selection = json.loads((ROOT / "runs/e3/pair_repeats_full_g0_f0_j0_family30/selected_hyperparameters.json").read_text())
    run = ROOT / selection["family30/fold0"]["run"]
    model = load_model(run)
    output = ROOT / "runs/verification/higher_features"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    identifiers = []
    for dataset in ["functional_doubles", "flip_primary"]:
        for directory in sorted((ROOT / "data/features/higher" / dataset).glob("*")):
            if not directory.is_dir():
                continue
            data = HigherFeatures(directory)
            rows = np.concatenate([
                np.flatnonzero(data.metadata.n_mutations.eq(order))[:128]
                for order in [0, 1, 2, 3, 4, 5]
            ])
            assert len(rows)
            effects, pairs = data.infer_terms(model)
            additive, interaction = data.combine(effects, pairs)
            with torch.no_grad():
                expected = model.mapping(torch.from_numpy((additive + interaction)[rows]).cuda()).cpu().numpy()
                direct = data.forward(model, rows)["phenotype"].cpu().numpy()
            np.testing.assert_allclose(direct, expected, rtol=1e-5, atol=2e-5)
            doubles = np.flatnonzero(data.metadata.n_mutations.eq(2))[:128]
            pair_ids = [data.pair_incidence[index].indices.item() for index in doubles]
            with torch.no_grad():
                features = data.batch("pair", pair_ids, "cuda")
                shape = features["score"].shape
                exact = model(features, torch.zeros(shape, device="cuda"),
                              torch.zeros(shape, dtype=torch.bool, device="cuda"))
            np.testing.assert_allclose(exact["phenotype"].cpu().numpy(),
                                       (additive + interaction)[doubles], rtol=1e-5, atol=2e-5)
            np.testing.assert_allclose(exact["epsilon"].cpu().numpy(), interaction[doubles], rtol=1e-5, atol=2e-5)
            wt = data.metadata.n_mutations.eq(0).to_numpy()
            assert np.all(additive[wt] == 0) and np.all(interaction[wt] == 0)
            assert np.all(interaction[data.metadata.n_mutations.le(1)] == 0)
            model.zero_grad(set_to_none=True)
            prediction = data.forward(model, doubles[:16])["phenotype"]
            prediction.square().mean().backward()
            gradients = [parameter.grad for parameter in model.pair.parameters()]
            assert any(value is not None and torch.any(value != 0) for value in gradients)
            assert all(torch.isfinite(value).all() for value in gradients if value is not None)
            identifiers.append(data.metadata.iloc[rows][["variant_id", "protein_id", "n_mutations"]].assign(dataset=dataset))
            results.append({"dataset": dataset, "protein": directory.name, "validated_variants": len(rows),
                            "validated_double_variants": len(doubles), "maximum_difference": float(np.max(np.abs(expected - direct)))})
    pd.concat(identifiers, ignore_index=True).to_parquet(output / "actual_variants.parquet", index=False)
    manifest = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "model_checkpoint": str((run / "best.pt").relative_to(ROOT)), "results": results,
                "checks": ["Actual-variant sparse incidence versus differentiable batches",
                           "Two-mutation specialization versus original EpistasisModel",
                           "Reference and single-mutation interaction zeros", "Finite nonzero pair gradients"],
                "experimental_labels_used": 0, "purpose": "Numerical implementation verification; no performance claim."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
