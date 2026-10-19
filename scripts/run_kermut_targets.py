import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import gpytorch
import h5py
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/kermut"))
from epistasis.kermut import WildTypeKermut
from epistasis.metrics import per_protein_metrics
from src.data.data_utils import Tokenizer
from src.model.gp import ExactGPKermut

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def build_inputs(protein, variants, doubles):
    mean_root = ROOT / "data/features/mean/tsuboyama"
    metadata = pd.read_parquet(mean_root / "metadata.parquet", columns=["protein_id", "variant_id", "matrix_row"])
    rows = metadata.set_index("variant_id").matrix_row.reindex(variants.variant_id).to_numpy()
    mean = np.load(mean_root / "mean.npy", mmap_mode="r")
    scores = np.load(mean_root / "scores.npy", mmap_mode="r")
    sequences = variants.aa_seq.tolist()
    keys = variants.mutation_key.tolist()
    embeddings = [value for value in mean[rows].astype(np.float32)]
    zero_shot = scores[rows, 0].tolist()
    index = {key: i for i, key in enumerate(keys)}
    assert len(index) == len(variants)
    components = {}
    for record in doubles.itertuples():
        components.update({key: (int(position), mutant) for key, position, mutant in zip(record.mutation_key.split(":"), record.positions, record.mut_residues)})
    with h5py.File(ROOT / "data/features/esm2/tsuboyama/full" / f"{protein.protein_id}_singles.h5", "r") as handle:
        assert handle.attrs["reference_sequence"] == protein.reference_sequence
        cache_index = {key: i for i, key in enumerate(handle["keys"].asstr()[:])}
        log_probs = handle["wt_masked_log_probs"][:]
        for key, (position, mutant) in sorted(components.items()):
            if key in index:
                continue
            sequence = list(protein.reference_sequence)
            sequence[position] = mutant
            sequences.append("".join(sequence))
            embeddings.append(handle["single"][cache_index[key]].astype(np.float32).mean(axis=0))
            zero_shot.append(float(log_probs[position, ALPHABET.index(mutant)] - log_probs[position, ALPHABET.index(protein.reference_sequence[position])]))
            index[key] = len(keys)
            keys.append(key)
    tokenizer = Tokenizer(flatten=True)
    one_hot = tokenizer(sequences).float()
    inputs = torch.cat([one_hot, torch.tensor(zero_shot, dtype=torch.float32)[:, None],
                        torch.tensor(np.stack(embeddings))], dim=1)
    assert torch.isfinite(inputs).all()
    assert keys[index[""]] == "" and sequences[index[""]] == protein.reference_sequence
    return inputs.cuda(), index, tokenizer([protein.reference_sequence]).float()[0]


def fit_and_predict(inputs, train_indices, targets, wt_encoding, structural, seed, directory):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = inputs[train_indices]
    train_y = torch.tensor(targets, dtype=torch.float32, device="cuda")
    center, scale = train_y.mean(), train_y.std()
    assert torch.isfinite(train_y).all() and scale > 0
    train_y = (train_y - center) / scale
    config = OmegaConf.load(ROOT / "external/kermut/configs/gp/kermut.yaml")
    config.mutation_kernel.model._target_ = "epistasis.kermut.WildTypeKermut"
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.HalfCauchyPrior(0.1))
    model = ExactGPKermut(
        train_x, train_y, likelihood, config.mutation_kernel, use_zero_shot=True, use_global_kernel=True,
        conditional_probs=torch.tensor(np.load(ROOT / structural["probabilities"])),
        wt_sequence=wt_encoding, coords=torch.tensor(np.load(ROOT / structural["coordinates"])["coordinates"]),
    ).cuda()
    likelihood = likelihood.cuda()
    model.train()
    likelihood.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    history = []
    started = time.monotonic()
    for step in range(150):
        optimizer.zero_grad()
        loss = -objective(model(train_x), train_y)
        assert torch.isfinite(loss), (step, directory)
        loss.backward()
        assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
        optimizer.step()
        history.append({"step": step, "negative_log_marginal_likelihood": float(loss.detach())})
    model.eval()
    likelihood.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(inputs), 256):
            predicted = likelihood(model(inputs[offset:offset + 256])).mean * scale + center
            assert torch.isfinite(predicted).all()
            predictions.append(predicted.cpu().numpy())
    values = np.concatenate(predictions)
    torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "label_mean": float(center), "label_std": float(scale), "seed": seed,
                "training_indices": train_indices}, directory / "model.pt")
    pd.DataFrame(history).to_csv(directory / "optimization.csv", index=False)
    del model, likelihood, optimizer, objective
    gc.collect()
    torch.cuda.empty_cache()
    return values, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--mode", choices=["S", "C"], default="S")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "runs" / ("kermut_target_smoke" if args.smoke else "kermut_target") / args.mode
    destination.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_parquet(ROOT / "data/processed/tsuboyama/proteins.parquet")
    variants_all = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet")
    doubles_all = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    proteins = proteins.loc[proteins.protein_id.isin(doubles_all.protein_id)].sort_values("protein_id")
    if args.smoke:
        proteins = proteins.loc[proteins.protein_id.isin(order.protein_id.unique())]
        proteins = proteins.iloc[:max(1, round(len(proteins) * 0.01))]
    proteins = proteins.iloc[args.rank::args.world_size]
    combination = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet") if args.mode == "C" else None
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifests = []
    started = time.monotonic()
    for protein in proteins.itertuples():
        variants = variants_all.loc[variants_all.protein_id.eq(protein.protein_id)].reset_index(drop=True)
        doubles = doubles_all.loc[doubles_all.protein_id.eq(protein.protein_id)].copy()
        structure_path = ROOT / "data/features/proteinmpnn/tsuboyama/full" / protein.protein_id / "manifest.json"
        structural = json.loads(structure_path.read_text())
        assert structural["complete"] and structural["reference_sequence"] == protein.reference_sequence
        wild_type = variants.loc[variants.n_mutations.eq(0)]
        assert len(wild_type) == 1
        inputs, index, wt_encoding = build_inputs(protein, variants, doubles)
        for seed in [11, 23, 47, 89, 131]:
            single_order = order.loc[order.seed.eq(seed) & order.protein_id.eq(protein.protein_id)].sort_values("rank")
            if args.mode == "S":
                settings = [("family", condition, budget, None) for condition, budget in [("S32", 32), ("S128", 128), ("Sall", -1)]]
            else:
                split = combination.loc[combination.seed.eq(seed) & combination.protein_id.eq(protein.protein_id)]
                settings = [(scheme, condition, budget, subset) for scheme, subset in split.groupby("scheme", sort=True)
                            for condition, budget in [("C128", 128), ("C512", 512), ("C2048", 2048), ("Call", -1)]]
            if args.conditions:
                settings = [item for item in settings if item[1] in args.conditions]
                assert settings, args.conditions
            for scheme, condition, budget, split in settings:
                directory = destination / protein.protein_id / f"seed{seed}" / scheme / condition
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "manifest.json"
                if path.exists():
                    previous = json.loads(path.read_text())
                    assert previous["complete"]
                    manifests.append(str(path.relative_to(ROOT)))
                    continue
                allowed_singles = single_order if args.mode == "C" or budget < 0 else single_order.iloc[:budget]
                allowed_ids = set(allowed_singles.variant_id) | set(wild_type.variant_id)
                if args.mode == "C":
                    candidate = split.loc[split["set"].eq("train")].sort_values("train_rank")
                    chosen = candidate if budget < 0 else candidate.iloc[:budget]
                    allowed_ids |= set(chosen.variant_id)
                    eligible = budget < 0 or len(candidate) >= budget
                    evaluation = doubles.merge(split[["variant_id", "set"]], on="variant_id", validate="one_to_one")
                    evaluation = evaluation.loc[evaluation["set"].isin(["validation", "test"])].copy()
                    assert not set(chosen.variant_id) & set(evaluation.variant_id)
                    if scheme != "random":
                        assert not set(chosen.pair_id) & set(evaluation.pair_id.dropna())
                else:
                    chosen = variants.iloc[:0]
                    eligible = budget < 0 or len(single_order) >= budget
                    evaluation = doubles.copy()
                    evaluation["set"] = "family_pool"
                allowed = variants.loc[variants.variant_id.isin(allowed_ids)].copy()
                allowed["used_by_gaussian_likelihood"] = allowed.point.notna()
                allowed.to_parquet(directory / "allowed_labels.parquet", index=False)
                training = allowed.loc[allowed.used_by_gaussian_likelihood]
                manifest = {
                    "complete": True, "smoke": args.smoke, "mode": args.mode, "scheme": scheme,
                    "condition": condition, "seed": seed, "protein_id": protein.protein_id,
                    "source_commit": source_commit, "source_labels_used": 0,
                    "target_single_labels": len(allowed_singles), "target_wt_variants": len(wild_type),
                    "target_combination_labels": len(chosen), "gaussian_point_labels_used": len(training),
                    "interval_labels_not_used": int((~allowed.used_by_gaussian_likelihood).sum()),
                    "available_single_labels": len(single_order), "budget_eligible": eligible,
                    "likelihood": "Author Gaussian likelihood uses available point labels; interval labels remain recorded and are never imputed.",
                    "optimization": {"steps": 150, "optimizer": "AdamW", "lr": 0.1, "weight_decay": 0.01, "restarts": 1},
                    "compound_model": "Original reproduce structure kernel plus ESM2 mean-embedding RBF and learned ESM2 zero-shot mean.",
                    "structural_manifest": str(structure_path.relative_to(ROOT)),
                }
                if len(training) < 2 or (args.mode == "C" and not eligible):
                    manifest["evaluated"] = False
                    manifest["reason"] = "insufficient_permitted_point_labels" if len(training) < 2 else "combination_budget_unavailable"
                    path.write_text(json.dumps(manifest, indent=2))
                    manifests.append(str(path.relative_to(ROOT)))
                    continue
                train_indices = [index[key] for key in training.mutation_key]
                values, elapsed = fit_and_predict(inputs, train_indices, training.point.to_numpy(), wt_encoding, structural, seed, directory)
                prediction_wt = float(values[index[""]])
                evaluation["prediction_mu"] = [float(values[index[key]]) - prediction_wt for key in evaluation.mutation_key]
                component_indices = np.array([[index[key] for key in mutation.split(":")] for mutation in evaluation.mutation_key])
                evaluation["prediction_a"] = values[component_indices[:, 0]] - prediction_wt
                evaluation["prediction_b"] = values[component_indices[:, 1]] - prediction_wt
                evaluation["prediction_wt"] = 0.0
                evaluation["prediction_epsilon"] = evaluation.prediction_mu - evaluation.prediction_a - evaluation.prediction_b
                evaluation["model_wt_absolute"] = prediction_wt
                evaluation["method"], evaluation["condition"], evaluation["seed"] = "kermut", condition, seed
                evaluation["budget_eligible"] = eligible
                evaluation["target_single_labels"], evaluation["target_combination_labels"] = len(allowed_singles), len(chosen)
                evaluation.to_parquet(directory / "predictions.parquet", index=False)
                metrics = []
                for partition, group in evaluation.groupby("set", sort=True):
                    current = per_protein_metrics(group)
                    current["partition"], current["method"], current["condition"] = partition, "kermut", condition
                    current["seed"], current["scheme"], current["budget_eligible"] = seed, scheme, eligible
                    metrics.append(current)
                pd.concat(metrics, ignore_index=True).to_parquet(directory / "per_protein.parquet", index=False)
                manifest.update({"evaluated": True, "evaluated_variants": len(evaluation), "elapsed_seconds": elapsed,
                                 "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "peak_vram_bytes": torch.cuda.max_memory_allocated()})
                path.write_text(json.dumps(manifest, indent=2))
                manifests.append(str(path.relative_to(ROOT)))
                print(json.dumps({"protein": protein.protein_id, "seed": seed, "scheme": scheme,
                                  "condition": condition, "complete": True, "training": len(training), "seconds": elapsed}), flush=True)
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {"complete": True, "config": vars(args), "source_commit": source_commit,
                "proteins": len(proteins), "outputs": manifests, "elapsed_seconds": time.monotonic() - started,
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "source_supervision": "none"}
    (destination / f"manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
