import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from train_single import DATA, FEATURES, ROOT, FeatureArrays, family_permissions
from epistasis.model import EpistasisModel, ModelConfig, ObservationModel, PairEffect, PhenotypeMap, SingleEffect


class Labels:
    def __init__(self, singles, doubles, seed):
        order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
        order = order.loc[order.seed.eq(seed)].set_index("variant_id")
        single_table = singles.metadata.set_index("variant_id")
        self.single_value = singles.metadata.relative_point.to_numpy(dtype=np.float32)
        self.single_rank = singles.metadata.variant_id.map(order["rank"]).fillna(-1).to_numpy(dtype=np.int64)
        lookup = singles.metadata.set_index(["protein_id", "mutation_key"]).variant_id
        parts = doubles.metadata.mutation_key.str.split(":")
        identifiers = []
        for endpoint in [0, 1]:
            index = pd.MultiIndex.from_arrays([doubles.metadata.protein_id, parts.str[endpoint]], names=["protein_id", "mutation_key"])
            identifiers.append(lookup.reindex(index).to_numpy())
        self.pair_value = np.stack([pd.Series(ids).map(single_table.relative_point).to_numpy(dtype=np.float32) for ids in identifiers], axis=1)
        self.pair_rank = np.stack([pd.Series(ids).map(order["rank"]).fillna(-1).to_numpy(dtype=np.int64) for ids in identifiers], axis=1)

    def values(self, kind, rows, modes):
        values = self.pair_value[rows] if kind == "doubles" else self.single_value[rows]
        ranks = self.pair_rank[rows] if kind == "doubles" else self.single_rank[rows]
        if kind == "doubles":
            modes = modes[:, None]
        allowed = (ranks >= 0) & ((modes == 3) | ((modes == 2) & (ranks < 128)))
        assert np.isfinite(values[allowed]).all()
        return torch.from_numpy(values).cuda(), torch.from_numpy(allowed).cuda()


def training_modes(proteins, rng):
    unique = sorted(set(proteins))
    modes = dict(zip(unique, rng.integers(0, 4, size=len(unique))))
    return np.array([modes[protein] for protein in proteins], dtype=np.int64)


def select_single(args):
    if args.single_run is not None:
        directory = (ROOT / args.single_run).resolve()
        assert directory.is_relative_to(ROOT)
        return directory
    representation = "full" if args.representation == "no_cross" else args.representation
    directory = ROOT / "runs/e3" / f"family{args.threshold}" / f"fold{args.fold}" / "single"
    candidates = []
    for projection in [64, 128]:
        for learning_rate in [1e-4, 3e-4, 1e-3]:
            path = directory / f"{representation}_p{projection}_lr{learning_rate:g}_seed11"
            manifest = json.loads((path / "manifest.json").read_text())
            assert manifest["complete"] and not manifest["smoke"]
            candidates.append((manifest["validation_macro_rmse"], str(path), path))
    return min(candidates)[2]


def pair_config(args):
    config = ModelConfig(projection=args.projection, representation=args.representation, geometry=args.geometry,
                         pair_form=args.pair_form, pair_single_input=args.pair_single_input,
                         pair_mask_input=args.pair_mask_input)
    if args.representation == "identity":
        target = sum(parameter.numel() for parameter in PairEffect(ModelConfig(projection=args.projection, geometry=args.geometry)).parameters())
        base = sum(parameter.numel() for parameter in PairEffect(ModelConfig(projection=args.projection, representation="identity", geometry=args.geometry, hidden=1)).parameters())
        increment = base - 257
        hidden = min(range(1, 8193), key=lambda width: abs(increment * width + 257 - target))
        config = ModelConfig(projection=args.projection, representation=args.representation, geometry=args.geometry,
                             hidden=hidden, pair_form=args.pair_form, pair_single_input=args.pair_single_input,
                             pair_mask_input=args.pair_mask_input)
    return config


def evaluate(model, doubles, labels, validation, batch_size):
    model.eval()
    conditions = [("Z", 0, 0.0)] + [(f"S128_w{omega:g}", 2, omega) for omega in [0.0, 0.5, 1.0]] + [(f"Sall_w{omega:g}", 3, omega) for omega in [0.0, 0.5, 1.0]]
    scores, frames = {}, {}
    with torch.inference_mode():
        for name, mode, omega in conditions:
            results = []
            for offset in range(0, len(validation), batch_size):
                frame = validation.iloc[offset:offset + batch_size]
                rows = frame.matrix_row.to_numpy()
                values, allowed = labels.values("doubles", rows, np.full(len(rows), mode, dtype=np.int64))
                prediction = model(doubles.batch(rows), values, allowed, omega=omega)
                output = frame[["variant_id", "quartet_id", "protein_id", "pair_id", "epsilon", "relative_point"]].copy()
                output["prediction_epsilon"] = prediction["epsilon"].cpu().numpy()
                output["prediction_mu"] = prediction["phenotype"].cpu().numpy()
                output["prediction_a"] = prediction["single"][:, 0].cpu().numpy()
                output["prediction_b"] = prediction["single"][:, 1].cpu().numpy()
                output["prediction_wt"] = 0.0
                results.append(output)
            prediction = pd.concat(results, ignore_index=True)
            difference = (prediction.prediction_mu - prediction.relative_point) if model.omit_pair and not model.mapping.flexible else (prediction.prediction_epsilon - prediction.epsilon)
            macro = difference.pow(2).groupby(prediction.protein_id).mean().pow(0.5).mean()
            assert np.isfinite(macro)
            scores[name] = float(macro)
            frames[name] = prediction
    omega = min([0.0, 0.5, 1.0], key=lambda value: (scores[f"S128_w{value:g}"], value))
    combined = 0.5 * scores["Z"] + 0.25 * scores[f"S128_w{omega:g}"] + 0.25 * scores[f"Sall_w{omega:g}"]
    return combined, omega, scores, frames


def phenotype_predictions(model, singles, doubles, labels, batch, rng, omega=1.0):
    prediction = torch.zeros(len(batch), device="cuda")
    modes = training_modes(batch.protein_id.tolist(), rng)
    for number, kind, features in [(1, "singles", singles), (2, "doubles", doubles)]:
        selected = batch.n_mutations.to_numpy() == number
        rows = batch.loc[selected, "matrix_row"].to_numpy(dtype=np.int64)
        if len(rows) == 0:
            continue
        data = features.batch(rows)
        values, allowed = labels.values(kind, rows, modes[selected])
        if number == 1:
            latent = model.calibrated_effects(data, values, allowed, omega)
            current = model.mapping(latent)
        else:
            current = model(data, values, allowed, omega=omega)["phenotype"]
        prediction[torch.from_numpy(selected).cuda()] = current
    return prediction + torch.from_numpy(batch.reference_point.to_numpy()).cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=[11, 23, 47, 89, 131], default=11)
    parser.add_argument("--projection", type=int, choices=[64, 128], default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-y", type=float, choices=[0.1, 1.0], default=0.1)
    parser.add_argument("--representation", choices=["full", "no_cross", "wt", "identity"], default="full")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--flexible", action="store_true")
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--geometry-intervention", choices=["none", "permuted", "rewired"], default="none")
    parser.add_argument("--objective", choices=["combined", "residual", "phenotype"], default="combined")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--single-run")
    parser.add_argument("--pair-form", choices=["contrast", "direct"], default="contrast")
    parser.add_argument("--pair-single-input", choices=["calibrated", "predicted"], default="calibrated")
    parser.add_argument("--pair-mask-input", choices=["auto", "observed", "zero"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.geometry or args.geometry_intervention == "none"
    assert not args.no_interaction or args.joint or args.flexible
    assert not (args.no_interaction and not args.flexible and args.objective == "residual")
    control_tag = (f"_geo{args.geometry_intervention}" if args.geometry_intervention != "none" else "") + ("_n1" if args.no_interaction else "")
    control_tag += "_direct" if args.pair_form == "direct" else ""
    if args.pair_single_input == "predicted":
        control_tag += "_predictedvalues" if args.pair_mask_input == "observed" else "_nopairmeasured"
    elif args.pair_mask_input == "zero":
        control_tag += "_nomask"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    name = f"{args.representation}_p{args.projection}_lr{args.learning_rate:g}_ly{args.lambda_y:g}_g{int(args.geometry)}_f{int(args.flexible)}_j{int(args.joint)}{control_tag}_{args.objective}_seed{args.seed}"
    run = ROOT / "runs/e3" / f"family{args.threshold}" / f"fold{args.fold}" / ("pair_smoke" if args.smoke else "pair") / name
    run.mkdir(parents=True, exist_ok=True)
    if not args.smoke and (args.pair_form == "direct" or args.pair_single_input == "predicted" or args.pair_mask_input == "zero"):
        # 两个队列可以认领同一消融配置，文件锁保证实际训练只执行一次。
        source_lock = (run / "pair_training.lock").open("a")
        fcntl.flock(source_lock, fcntl.LOCK_EX)
        if (run / "manifest.json").exists():
            completed = json.loads((run / "manifest.json").read_text())
            assert completed["complete"] and not completed["smoke"]
            completed_config = {"pair_mask_input": "auto", **completed["config"]}
            assert completed_config == vars(args)
            print(json.dumps({"complete": True, "reused_completed_training": str(run.relative_to(ROOT))}), flush=True)
            return
    assert not (run / "manifest.json").exists(), f"Completed run already exists: {run}"
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    single_run = select_single(args)
    checkpoint = torch.load(single_run / "best.pt", map_location="cpu", weights_only=True)
    single_config = ModelConfig(**checkpoint["model_config"])
    single = SingleEffect(single_config)
    single.load_state_dict(checkpoint["model"])
    config = pair_config(args)
    model = EpistasisModel(single, PairEffect(config), PhenotypeMap(args.flexible), omit_pair=args.no_interaction).cuda()
    if args.no_interaction:
        model.pair.requires_grad_(False)
    if not args.joint:
        model.freeze_single()
    noise = ObservationModel().cuda()
    noise.load_state_dict(checkpoint["noise"])
    singles, doubles = FeatureArrays("singles"), FeatureArrays("doubles", args.geometry_intervention)
    labels = Labels(singles, doubles, args.seed)
    permissions = family_permissions(args.threshold, args.fold)
    (run / "permissions.json").write_text(json.dumps(permissions, indent=2))
    (run / "config.json").write_text(json.dumps({**vars(args), "single_run": str(single_run.relative_to(ROOT)), "model": asdict(config)}, indent=2))
    references = pd.read_parquet(FEATURES / "reference_targets.parquet").set_index("protein_id")
    usable = set(references.index[references.point_in_range])
    training = doubles.metadata.loc[doubles.metadata.protein_id.isin(set(permissions["train"]) & usable) & doubles.metadata.main].copy()
    validation = doubles.metadata.loc[doubles.metadata.protein_id.isin(permissions["validation"]) & doubles.metadata.main].copy()
    if args.geometry:
        training = training.loc[training.structure_available]
        validation = validation.loc[validation.structure_available]
    eligible = validation.groupby("protein_id").size()
    validation = validation.loc[validation.protein_id.isin(eligible.index[eligible >= 30])]
    observations = pd.read_parquet(FEATURES / "observation_targets.parquet")
    observations = observations.loc[observations.protein_id.isin(set(permissions["train"]) & usable) & observations.n_mutations.isin([0, 1, 2])].copy()
    index = pd.concat([singles.metadata[["variant_id", "matrix_row"]], doubles.metadata[["variant_id", "matrix_row"]]]).set_index("variant_id").matrix_row
    observations["matrix_row"] = observations.variant_id.map(index).fillna(-1).astype(int)
    assert observations.loc[observations.n_mutations.gt(0), "matrix_row"].ge(0).all()
    observations["reference_point"] = observations.protein_id.map(references.point)
    if args.geometry:
        valid_double_ids = set(doubles.metadata.loc[doubles.metadata.structure_available, "variant_id"])
        observations = observations.loc[observations.n_mutations.ne(2) | observations.variant_id.isin(valid_double_ids)]
    if args.smoke:
        training = training.iloc[::100].copy()
        validation = validation.iloc[::100].copy()
        observations = observations.iloc[::100].copy()
    assert len(training) > 0 and len(validation) > 0 and len(observations) > 0
    assert not set(training.protein_id) & set(permissions["test"])
    training[["variant_id", "quartet_id", "protein_id", "pair_id"]].to_parquet(run / "training_quartets.parquet", index=False)
    observations[["observation_id", "variant_id", "protein_id"]].to_parquet(run / "training_observations.parquet", index=False)
    validation[["variant_id", "quartet_id", "protein_id"]].to_parquet(run / "validation_quartets.parquet", index=False)
    group_size = training.groupby(["protein_id", "pair_id"]).variant_id.transform("size")
    pair_count = training.groupby("protein_id").pair_id.transform("nunique")
    weights = (1.0 / group_size / pair_count).to_numpy()
    weights /= weights.sum()
    train_rows = training.matrix_row.to_numpy()
    truth = doubles.metadata.epsilon.to_numpy(dtype=np.float32)
    optimizer = torch.optim.AdamW([parameter for parameter in [*model.parameters(), *noise.parameters()] if parameter.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, best_epoch, best_omega = float("inf"), -1, None
    history = []
    started = time.monotonic()
    epochs = 2 if args.smoke else 100
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(observations))
        total_residual, total_phenotype, seen, steps = 0.0, 0.0, 0, 0
        for offset in range(0, len(observations), args.batch_size):
            batch = observations.iloc[permutation[offset:offset + args.batch_size]]
            residual = torch.zeros((), device="cuda")
            phenotype = torch.zeros((), device="cuda")
            if args.objective != "phenotype":
                rows = rng.choice(train_rows, size=len(batch), replace=True, p=weights)
                modes = training_modes(doubles.metadata.iloc[rows].protein_id.tolist(), rng)
                values, allowed = labels.values("doubles", rows, modes)
                prediction = model(doubles.batch(rows), values, allowed, omega=1.0)
                residual = F.huber_loss(prediction["epsilon"], torch.from_numpy(truth[rows]).cuda(), delta=1.0)
            if args.objective != "residual":
                prediction_y = phenotype_predictions(model, singles, doubles, labels, batch, rng)
                phenotype = noise.nll(prediction_y, torch.from_numpy(batch.lower.to_numpy()).cuda(), torch.from_numpy(batch.upper.to_numpy()).cuda(), torch.from_numpy(batch.measurement_status.eq("point").to_numpy()).cuda()).mean()
            loss = residual + args.lambda_y * phenotype if args.objective == "combined" else (residual if args.objective == "residual" else phenotype)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            assert noise.sigma_raw.grad is None or torch.isfinite(noise.sigma_raw.grad)
            optimizer.step()
            total_residual += float(residual.detach())
            total_phenotype += float(phenotype.detach())
            seen += len(batch)
            steps += 1
        assert seen == len(observations)
        score, omega, scores, frames = evaluate(model, doubles, labels, validation, args.batch_size)
        entry = {"epoch": epoch, "training_huber": total_residual / steps, "training_nll": total_phenotype / steps, "validation_score": score, "omega": omega, "validation_conditions": scores, "sigma": float(noise.sigma.detach()), "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        (run / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)
        if score < best:
            best, best_epoch, best_omega = score, epoch, omega
            torch.save({
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "noise": {key: value.detach().cpu() for key, value in noise.state_dict().items()},
                "single_config": asdict(single_config), "pair_config": asdict(config), "flexible": args.flexible,
                "epoch": epoch, "validation_score": score, "omega": omega, "omit_pair": args.no_interaction,
            }, run / "best.pt")
            for condition in ["Z", f"S128_w{omega:g}", f"Sall_w{omega:g}"]:
                frames[condition].to_parquet(run / f"validation_{condition}.parquet", index=False)
        if epoch - best_epoch >= 10:
            break
    assert np.isfinite(best)
    manifest = {
        "stage": "shared_pair_effect_training", "complete": True, "smoke": args.smoke,
        "source_commit": source_commit, "config": vars(args), "single_run": str(single_run.relative_to(ROOT)),
        "single_config": asdict(single_config), "pair_config": asdict(config),
        "best_epoch": best_epoch, "epochs": len(history), "validation_score": best, "omega": best_omega,
        "validation_selection": "Select omega by pseudo-target S128; rank by 0.5*Z+0.25*S128+0.25*Sall. The identity no-interaction control uses phenotype RMSE because its residual is identically zero; other models use epsilon RMSE.",
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "training_calibration": 1.0, "training_input_permissions": {"Z": 0.5, "S128": 0.25, "Sall": 0.25},
        "residual_sampling": "Equal proteins, then equal site pairs, then equal quartets; Huber delta=1 kcal/mol.",
        "phenotype_sampling": "Every selected raw observation exactly once per epoch.",
        "training_quartets": len(training), "training_proteins": training.protein_id.nunique(),
        "training_observations": len(observations), "validation_quartets": len(validation),
        "permissions": str((run / "permissions.json").relative_to(ROOT)), "checkpoint": str((run / "best.pt").relative_to(ROOT)),
        "source_features": str((FEATURES / "manifest.json").relative_to(ROOT)),
        "seed": args.seed, "torch": torch.__version__, "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(),
        "elapsed_seconds": time.monotonic() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
