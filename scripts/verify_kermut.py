import json
from pathlib import Path
import sys

import gpytorch
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/kermut"))
from src.data.data_utils import Tokenizer
from src.model.kernel import Kermut
from epistasis.kermut import WildTypeKermut
from src.model.gp import ExactGPKermut


def main():
    torch.manual_seed(37)
    device = torch.device("cuda")
    files = sorted((ROOT / "data/features/proteinmpnn/tsuboyama/smoke").glob("tsub_*/manifest.json"))
    assert files
    info = json.loads(files[0].read_text())
    data = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet")
    group = data.loc[data.protein_id.eq(info["protein_id"])]
    measured = group.loc[group.n_mutations.gt(0) & group.point_in_range].head(32)
    assert len(measured) == 32
    wild_type = group.loc[group.n_mutations.eq(0)]
    assert len(wild_type) == 1
    tokenizer = Tokenizer(flatten=True)
    wt = tokenizer([info["reference_sequence"]]).float().to(device)
    mutants = tokenizer(measured.aa_seq.tolist()).float().to(device)
    kwargs = {
        "conditional_probs": torch.tensor(np.load(ROOT / info["probabilities"])),
        "wt_sequence": wt[0].cpu(), "coords": torch.tensor(np.load(ROOT / info["coordinates"])["coordinates"]),
        "h_scale": 1.0, "h_lengthscale": 0.1, "d_lengthscale": 0.1, "p_lengthscale": 0.1,
    }
    original = Kermut(**kwargs).to(device)
    adapted = WildTypeKermut(**kwargs).to(device)
    adapted.load_state_dict(original.state_dict())
    expected = original(mutants).to_dense()
    actual = adapted(mutants).to_dense()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    for (name, first), (other_name, second) in zip(original.named_parameters(), adapted.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-6, atol=1e-6)
    augmented = torch.cat([mutants, wt], dim=0)
    covariance = adapted(augmented).to_dense()
    assert covariance.shape == (33, 33)
    torch.testing.assert_close(covariance[:-1, :-1], expected, rtol=0, atol=0)
    assert torch.count_nonzero(covariance[-1]) == 0 and torch.count_nonzero(covariance[:, -1]) == 0
    assert torch.count_nonzero(adapted(wt).to_dense()) == 0
    assert adapted(wt, mutants).to_dense().shape == (1, 32)
    torch.testing.assert_close(adapted(augmented, diag=True), covariance.diagonal(), rtol=0, atol=0)
    mean_root = ROOT / "data/features/mean/tsuboyama"
    metadata = pd.read_parquet(mean_root / "metadata.parquet")
    rows = metadata.set_index("variant_id").matrix_row.reindex(measured.variant_id).to_numpy()
    wt_row = int(metadata.set_index("variant_id").loc[wild_type.variant_id.iloc[0], "matrix_row"])
    mean = np.load(mean_root / "mean.npy", mmap_mode="r")
    scores = np.load(mean_root / "scores.npy", mmap_mode="r")
    features = torch.cat([mutants, torch.tensor(scores[rows, 0:1], device=device),
                          torch.tensor(mean[rows].astype(np.float32), device=device)], dim=1)
    test_features = torch.cat([wt, torch.tensor(scores[wt_row:wt_row+1, 0:1], device=device),
                               torch.tensor(mean[wt_row:wt_row+1].astype(np.float32), device=device)], dim=1)
    targets = torch.tensor(measured.point.to_numpy(np.float32), device=device)
    targets = (targets - targets.mean()) / targets.std()
    config = OmegaConf.load(ROOT / "external/kermut/configs/gp/kermut.yaml")
    config.mutation_kernel.model._target_ = "epistasis.kermut.WildTypeKermut"
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.HalfCauchyPrior(0.1)).to(device)
    model = ExactGPKermut(features, targets, likelihood, config.mutation_kernel,
                         use_zero_shot=True, use_global_kernel=True, **{key: value for key, value in kwargs.items() if key in ["conditional_probs", "wt_sequence", "coords"]}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    likelihood.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = -objective(model(features), targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        optimizer.step()
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        prediction = model(test_features)
        assert torch.isfinite(prediction.mean).all() and (prediction.variance > 0).all()
    output = ROOT / "runs/kermut_checks"
    output.mkdir(parents=True, exist_ok=True)
    measured[["variant_id", "protein_id", "aa_seq", "point"]].to_parquet(output / "real_training_records.parquet", index=False)
    result = {"complete": True, "real_mutants": len(measured), "wild_type": wild_type.variant_id.iloc[0],
              "mutant_covariance_and_gradients_match_author": True, "wild_type_structural_covariance_zero": True,
              "full_gp_wild_type_prediction_finite": True, "diag_and_cross_covariance": True}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
