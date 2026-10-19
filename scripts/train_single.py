import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.model import ModelConfig, ObservationModel, SingleEffect

FEATURES = ROOT / "data/features/training/tsuboyama"
DATA = ROOT / "data/processed/tsuboyama"


class FeatureArrays:
    def __init__(self, kind, geometry_intervention="none"):
        directory = FEATURES / kind
        self.metadata = pd.read_parquet(directory / "metadata.parquet")
        self.arrays = {path.stem: np.load(path, mmap_mode="r") for path in directory.glob("*.npy")}
        if geometry_intervention != "none":
            assert kind == "doubles"
            control = FEATURES / ("geometry_" + geometry_intervention)
            assert json.loads((control / "manifest.json").read_text())["complete"]
            for key in ["ca_distance", "confidence"]:
                self.arrays[key] = np.load(control / f"{key}.npy", mmap_mode="r")
        assert all(len(array) == len(self.metadata) for array in self.arrays.values())

    def batch(self, indices, device="cuda"):
        return {key: torch.from_numpy(np.array(array[indices], copy=True)).to(device) for key, array in self.arrays.items()}


def family_permissions(threshold, fold):
    table = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet")
    validation = (fold + 1) % 5
    train = table.loc[~table.outer_group.isin([fold, validation]), "protein_id"].tolist()
    valid = table.loc[table.outer_group.eq(validation), "protein_id"].tolist()
    test = table.loc[table.outer_group.eq(fold), "protein_id"].tolist()
    assert not set(train) & set(valid) and not set(train) & set(test)
    return {"train": train, "validation": valid, "test": test}


def parameter_config(projection, representation):
    config = ModelConfig(projection=projection, representation=representation)
    if representation == "identity":
        target = sum(p.numel() for p in SingleEffect(ModelConfig(projection=projection)).parameters())
        hidden = min(range(1, 4097), key=lambda width: abs((169 * width + 257) - target))
        config = ModelConfig(projection=projection, representation=representation, hidden=hidden)
    return config


def evaluate(model, features, table, batch_size):
    model.eval()
    values = []
    with torch.inference_mode():
        for offset in range(0, len(table), batch_size):
            group = table.iloc[offset:offset + batch_size]
            values.append(model(features.batch(group.matrix_row.to_numpy())).cpu().numpy())
    predictions = table[["variant_id", "protein_id", "relative_point"]].copy()
    predictions["prediction"] = np.concatenate(values)
    predictions["squared_error"] = (predictions.prediction - predictions.relative_point) ** 2
    per_protein = predictions.groupby("protein_id").squared_error.mean().pow(0.5)
    assert np.isfinite(per_protein).all()
    return float(per_protein.mean()), predictions, per_protein


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--projection", type=int, choices=[64, 128], default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--representation", choices=["full", "no_cross", "wt", "identity"], default="full")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    config = parameter_config(args.projection, args.representation)
    run_name = f"{args.representation}_p{args.projection}_lr{args.learning_rate:g}_seed{args.seed}"
    run = ROOT / "runs/e3" / f"family{args.threshold}" / f"fold{args.fold}" / ("single_smoke" if args.smoke else "single") / run_name
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "manifest.json").exists(), f"Completed run already exists: {run}"
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    permissions = family_permissions(args.threshold, args.fold)
    (run / "permissions.json").write_text(json.dumps(permissions, indent=2))
    (run / "config.json").write_text(json.dumps({**vars(args), "model": asdict(config)}, indent=2))
    features = FeatureArrays("singles")
    observations = pd.read_parquet(FEATURES / "observation_targets.parquet")
    references = pd.read_parquet(FEATURES / "reference_targets.parquet").set_index("protein_id")
    usable_references = set(references.index[references.point_in_range])
    training = observations.loc[
        observations.protein_id.isin(set(permissions["train"]) & usable_references)
        & observations.n_mutations.isin([0, 1])
    ].copy()
    training["matrix_row"] = training.variant_id.map(features.metadata.set_index("variant_id").matrix_row).fillna(-1).astype(int)
    assert training.loc[training.n_mutations.eq(1), "matrix_row"].ge(0).all()
    training["reference_point"] = training.protein_id.map(references.point)
    validation = features.metadata.loc[
        features.metadata.protein_id.isin(permissions["validation"])
        & features.metadata.point_in_range & features.metadata.wt_point_in_range
    ].copy()
    if args.smoke:
        training = training.iloc[::100].copy()
        validation = validation.iloc[::100].copy()
    assert len(training) > 0 and len(validation) > 0
    assert not set(training.protein_id) & set(permissions["test"])
    training[["observation_id", "variant_id", "protein_id", "n_mutations"]].to_parquet(run / "training_observations.parquet", index=False)
    validation[["variant_id", "protein_id"]].to_parquet(run / "validation_variants.parquet", index=False)
    model = SingleEffect(config).cuda()
    noise = ObservationModel().cuda()
    optimizer = torch.optim.AdamW([*model.parameters(), *noise.parameters()], lr=args.learning_rate, weight_decay=1e-4)
    started = time.monotonic()
    history = []
    best = float("inf")
    best_epoch = -1
    epochs = 2 if args.smoke else 100
    rng = np.random.default_rng(args.seed)
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(training))
        count, total_loss = 0, 0.0
        for offset in range(0, len(training), args.batch_size):
            batch = training.iloc[permutation[offset:offset + args.batch_size]]
            rows = batch.matrix_row.to_numpy()
            active = rows >= 0
            prediction = torch.zeros(len(batch), device="cuda")
            prediction[torch.from_numpy(active).cuda()] = model(features.batch(rows[active]))
            reference = torch.from_numpy(batch.reference_point.to_numpy()).cuda()
            lower = torch.from_numpy(batch.lower.to_numpy()).cuda()
            upper = torch.from_numpy(batch.upper.to_numpy()).cuda()
            point = torch.from_numpy(batch.measurement_status.eq("point").to_numpy()).cuda()
            likelihood = noise.nll(prediction + reference, lower, upper, point)
            loss = likelihood.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            assert torch.isfinite(noise.sigma_raw.grad)
            optimizer.step()
            count += len(batch)
            total_loss += float(likelihood.detach().sum())
        assert count == len(training)
        score, predictions, per_protein = evaluate(model, features, validation, args.batch_size)
        entry = {"epoch": epoch, "training_nll": total_loss / count, "validation_macro_rmse": score, "sigma": float(noise.sigma.detach()), "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        (run / "history.json").write_text(json.dumps(history, indent=2))
        if score < best:
            best, best_epoch = score, epoch
            torch.save({
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "noise": {key: value.detach().cpu() for key, value in noise.state_dict().items()},
                "model_config": asdict(config), "epoch": epoch, "validation_macro_rmse": score,
            }, run / "best.pt")
            predictions.to_parquet(run / "validation_predictions.parquet", index=False)
            per_protein.rename("rmse").to_csv(run / "validation_per_protein.csv")
        if epoch - best_epoch >= 10:
            break
    assert np.isfinite(best)
    manifest = {
        "stage": "source_single_effect_training", "complete": True, "smoke": args.smoke,
        "source_commit": source_commit, "config": vars(args), "model_config": asdict(config),
        "validation_macro_rmse": best, "best_epoch": best_epoch, "epochs": len(history),
        "training_observations": len(training), "training_biological_variants": training.variant_id.nunique(),
        "training_proteins": training.protein_id.nunique(), "validation_variants": len(validation),
        "validation_proteins": validation.protein_id.nunique(), "reference_anchor": "mean of quality-passing WT observations; all WT observations must be uncensored and strictly inside the assay range",
        "training_likelihood": "Each eligible raw WT/single observation occurs once per epoch. Point and interval observations use Gaussian density/CDF respectively.",
        "source_features": str((FEATURES / "manifest.json").relative_to(ROOT)),
        "permissions": str((run / "permissions.json").relative_to(ROOT)), "checkpoint": str((run / "best.pt").relative_to(ROOT)),
        "seed": args.seed, "torch": torch.__version__, "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(),
        "elapsed_seconds": time.monotonic() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
