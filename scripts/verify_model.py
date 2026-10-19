import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.features import pair_features, to_torch
from epistasis.model import EpistasisModel, ModelConfig, ObservationModel, PairEffect, PhenotypeMap, SingleEffect

DATA = ROOT / "data/processed/tsuboyama"
CACHE = ROOT / "data/features/esm2/tsuboyama/full"
GEOMETRY = ROOT / "data/features/structures/tsuboyama"


def real_batch():
    proteins = pd.read_parquet(DATA / "proteins.parquet").set_index("protein_id")
    variants = pd.read_parquet(DATA / "variants.parquet").set_index("variant_id")
    quartets = pd.read_parquet(DATA / "quartets.parquet")
    for protein_id, group in quartets.loc[quartets.main].groupby("protein_id"):
        if (CACHE / f"{protein_id}_singles.h5").exists() and (GEOMETRY / f"{protein_id}.npz").exists():
            selected = group.iloc[:128]
            protein = proteins.loc[[protein_id]].reset_index().iloc[0]
            ab = variants.loc[selected.ab_id].reset_index()
            features = to_torch(pair_features(protein, ab, CACHE, GEOMETRY))
            observed = torch.from_numpy(np.stack([
                variants.loc[selected.a_id].point.to_numpy() - variants.loc[selected.wt_id].point.to_numpy(),
                variants.loc[selected.b_id].point.to_numpy() - variants.loc[selected.wt_id].point.to_numpy(),
            ], axis=1).astype(np.float32))
            return features, observed, selected.quartet_id.tolist()
    raise AssertionError("No completed real quartet feature batch")


def verify_algebra(features, observed):
    results = []
    for representation in ["full", "no_cross", "wt", "identity"]:
        for geometry in [False, True]:
            torch.manual_seed(11)
            config = ModelConfig(representation=representation, geometry=geometry)
            for flexible in [False, True]:
                model = EpistasisModel(SingleEffect(config), PairEffect(config), PhenotypeMap(flexible))
                mask = torch.ones_like(observed, dtype=torch.bool)
                output = model(features, observed, mask, omega=1)
                swapped = {key: value.flip(1) if value.ndim >= 2 else value for key, value in features.items()}
                reverse = model(swapped, observed.flip(1), mask.flip(1), omega=1)
                torch.testing.assert_close(output["phenotype"], reverse["phenotype"], rtol=0, atol=1e-5)
                torch.testing.assert_close(output["epsilon"], output["epsilon_global"] + output["epsilon_specific"], rtol=0, atol=1e-5)
                if not flexible:
                    torch.testing.assert_close(output["epsilon"], output["interaction"], rtol=0, atol=1e-5)
                hidden = torch.zeros_like(mask)
                masked = model(features, observed, hidden, omega=1)
                hidden_values = torch.full_like(observed, float("nan"))
                masked_other = model(features, hidden_values, hidden, omega=1)
                torch.testing.assert_close(masked["phenotype"], masked_other["phenotype"], rtol=0, atol=0)
                for endpoint in [0, 1]:
                    reference = {key: value.clone() for key, value in features.items()}
                    reference["mut_aa"][:, endpoint] = reference["wt_aa"][:, endpoint]
                    zero = model(reference, observed, mask, omega=1)
                    torch.testing.assert_close(zero["interaction"], torch.zeros_like(zero["interaction"]), rtol=0, atol=1e-5)
                    torch.testing.assert_close(zero["single"][:, endpoint], torch.zeros_like(zero["single"][:, endpoint]), rtol=0, atol=1e-5)
                z = observed.sum(dim=-1).detach().requires_grad_(True)
                transformed = model.mapping.normalized(z)
                derivative = torch.autograd.grad(transformed.sum(), z)[0]
                assert (derivative > 0).all()
                origin = torch.zeros_like(z, requires_grad=True)
                origin_value = model.mapping.normalized(origin)
                origin_derivative = torch.autograd.grad(origin_value.sum(), origin)[0]
                torch.testing.assert_close(origin_value, torch.zeros_like(origin_value), rtol=0, atol=1e-5)
                torch.testing.assert_close(origin_derivative, torch.ones_like(origin_derivative), rtol=0, atol=1e-5)
                output["phenotype"].square().mean().backward()
                assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
                results.append({"representation": representation, "geometry": geometry, "flexible_mapping": flexible, "passed": True})
    return results


def verify_likelihood():
    observations = pd.read_parquet(DATA / "observations.parquet")
    accepted = observations.loc[observations.accepted]
    selected = accepted.groupby("measurement_status", sort=True).head(128)
    lower = torch.from_numpy(selected.lower.to_numpy())
    upper = torch.from_numpy(selected.upper.to_numpy())
    point = torch.from_numpy(selected.measurement_status.eq("point").to_numpy())
    prediction = torch.from_numpy(selected.deltaG.to_numpy()).requires_grad_(True)
    assert torch.isfinite(prediction).all()
    model = ObservationModel()
    loss = model.nll(prediction, lower, upper, point)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(prediction.grad).all(), "Censored likelihood produced non-finite mean gradients"
    assert torch.isfinite(model.sigma_raw.grad).all(), "Censored likelihood produced non-finite noise gradient"
    return {"point_rows": int(point.sum()), "interval_rows": int((~point).sum()), "finite_forward_and_gradients": True}


def main():
    features, observed, ids = real_batch()
    likelihood = verify_likelihood()
    algebra = verify_algebra(features, observed)
    result = {"real_quartets": ids, "algebra": algebra, "likelihood": likelihood, "atol": 1e-5}
    output = ROOT / "runs/model_constraints"
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"quartets": len(ids), "configurations": len(algebra), "likelihood": likelihood}))


if __name__ == "__main__":
    main()
