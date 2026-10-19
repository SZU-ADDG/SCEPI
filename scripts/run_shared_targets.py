import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from evaluate_family import load_model
from train_pair import Labels
from train_single import ROOT, FEATURES, FeatureArrays
from epistasis.metrics import per_protein_metrics
from epistasis.model import ObservationModel
from epistasis.physical_target import fit_target_main


def predict(model, doubles, labels, table, omega):
    frames = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(table), 1024):
            group = table.iloc[offset:offset + 1024].copy()
            rows = group.matrix_row.to_numpy()
            values, allowed = labels.values("doubles", rows, np.full(len(rows), 3))
            result = model(doubles.batch(rows), values, allowed, omega=omega)
            for name, value in [("mu", result["phenotype"]), ("a", result["single"][:, 0]),
                                ("b", result["single"][:, 1]), ("epsilon", result["epsilon"])]:
                group["prediction_" + name] = value.cpu().numpy()
            group["prediction_wt"] = 0.0
            frames.append(group)
    return pd.concat(frames, ignore_index=True)


def raw_prediction(model, singles, doubles, labels, table, omega):
    prediction = torch.zeros(len(table), device="cuda")
    for number, kind, features in [(1, "singles", singles), (2, "doubles", doubles)]:
        selected = table.n_mutations.to_numpy() == number
        rows = table.loc[selected, "matrix_row"].to_numpy(dtype=np.int64)
        if not len(rows):
            continue
        values, allowed = labels.values(kind, rows, np.full(len(rows), 3))
        data = features.batch(rows)
        value = model.mapping(model.calibrated_effects(data, values, allowed, omega)) if number == 1 else model(data, values, allowed, omega=omega)["phenotype"]
        prediction[torch.from_numpy(selected).cuda()] = value
    return prediction + torch.tensor(table.reference_point.to_numpy(), device="cuda")


def train(source, directory, singles, doubles, labels, observations, training, validation, evaluation, seed):
    configuration = json.loads((source / "manifest.json").read_text())
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    omega = configuration["omega"]
    model = load_model(source)
    model.freeze_single()
    fixed_single = {name: value.detach().cpu().clone() for name, value in model.single.state_dict().items()}
    noise = ObservationModel().cuda()
    noise.load_state_dict(checkpoint["noise"])
    optimizer = torch.optim.AdamW([*model.pair.parameters(), *noise.parameters()],
                                  lr=configuration["config"]["learning_rate"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    validation = validation.loc[validation.main].copy()
    use_validation = len(validation) > 0
    fixed_epochs = configuration["pair_epoch_budget"] if "pair_epoch_budget" in configuration else configuration["best_epoch"] + 1
    maximum_epochs = 100 if use_validation else min(100, fixed_epochs)
    reference = observations.reference_point.unique()
    assert len(reference) == 1 and np.isfinite(reference).all()
    main = training.loc[training.main].copy()
    if len(main):
        weights = 1.0 / main.groupby("pair_id").variant_id.transform("size").to_numpy()
        weights /= weights.sum()
    best, best_epoch = float("inf"), -1
    history = []
    started = time.monotonic()

    def save(epoch, score):
        torch.save({**checkpoint, "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "noise": {name: value.detach().cpu() for name, value in noise.state_dict().items()},
                    "target_epoch": epoch, "target_validation_rmse": score}, directory / "best.pt")

    if use_validation:
        initial = predict(model, doubles, labels, validation, omega)
        best = float(np.sqrt(np.mean((initial.prediction_epsilon - initial.epsilon) ** 2)))
        save(-1, best)
    for epoch in range(maximum_epochs):
        model.train()
        permutation = rng.permutation(len(observations))
        total_nll, total_huber, count, steps = 0.0, 0.0, 0, 0
        for start in range(0, len(observations), 1024):
            batch = observations.iloc[permutation[start:start + 1024]]
            residual = torch.zeros((), device="cuda")
            if len(main):
                rows = rng.choice(main.matrix_row.to_numpy(), size=len(batch), replace=True, p=weights)
                values, allowed = labels.values("doubles", rows, np.full(len(rows), 3))
                predicted = model(doubles.batch(rows), values, allowed, omega=omega)["epsilon"]
                target = torch.tensor(doubles.metadata.iloc[rows].epsilon.to_numpy(dtype=np.float32), device="cuda")
                residual = F.huber_loss(predicted, target, delta=1.0)
            predicted_y = raw_prediction(model, singles, doubles, labels, batch, omega)
            nll = noise.nll(predicted_y, torch.tensor(batch.lower.to_numpy(), device="cuda"),
                            torch.tensor(batch.upper.to_numpy(), device="cuda"),
                            torch.tensor(batch.measurement_status.eq("point").to_numpy(), device="cuda")).mean()
            loss = residual + configuration["config"]["lambda_y"] * nll
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
            assert noise.sigma_raw.grad is not None and torch.isfinite(noise.sigma_raw.grad)
            optimizer.step()
            count += len(batch)
            steps += 1
            total_nll += float(nll.detach())
            total_huber += float(residual.detach())
        assert count == len(observations)
        score = None
        if use_validation:
            predicted = predict(model, doubles, labels, validation, omega)
            score = float(np.sqrt(np.mean((predicted.prediction_epsilon - predicted.epsilon) ** 2)))
            assert np.isfinite(score)
            if score < best:
                best, best_epoch = score, epoch
                save(epoch, score)
        else:
            best_epoch = epoch
            save(epoch, None)
        history.append({"epoch": epoch, "nll": total_nll / steps, "huber": total_huber / steps,
                        "validation_epsilon_rmse": score, "sigma": float(noise.sigma.detach())})
        if use_validation and epoch - best_epoch >= 10:
            break
    pd.DataFrame(history).to_csv(directory / "optimization.csv", index=False)
    fitted = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(fitted["model"])
    assert all(torch.equal(value.detach().cpu(), fixed_single[name]) for name, value in model.single.state_dict().items())
    result = predict(model, doubles, labels, evaluation, omega)
    result.to_parquet(directory / "predictions.parquet", index=False)
    summaries = []
    for partition, frame in result.groupby("set"):
        summary = per_protein_metrics(frame)
        summary["partition"] = partition
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_parquet(directory / "per_protein.parquet", index=False)
    details = {"epochs": len(history), "selected_epoch": best_epoch,
               "validation_epsilon_rmse": best if use_validation else None,
               "early_stopping": "target_validation_epsilon_rmse" if use_validation else "fixed_source_selected_epoch",
               "validation_complete_quartets": len(validation), "training_complete_quartets": len(main),
               "training_observations": len(observations), "elapsed_seconds": time.monotonic() - started,
               "source_hyperparameters": configuration["config"], "omega": omega}
    del model, noise, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, choices=[11, 23, 47, 89, 131], default=[11, 23, 47, 89, 131])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--supervision", choices=["transfer", "target_only"], default="transfer")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--schemes", nargs="+", choices=["unseen_pair", "random", "two_pair_secondary"],
                        default=["unseen_pair", "random", "two_pair_secondary"])
    parser.add_argument("--conditions", nargs="+", choices=["C128", "C512", "C2048", "Call"],
                        default=["C128", "C512", "C2048", "Call"])
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    name = "shared_target" if args.supervision == "transfer" else "shared_target_only"
    name += "_geometry" if args.geometry else ""
    output = ROOT / "runs" / (name + ("_smoke" if args.smoke else ""))
    output.mkdir(parents=True, exist_ok=True)
    singles, doubles = FeatureArrays("singles"), FeatureArrays("doubles")
    observations = pd.read_parquet(FEATURES / "observation_targets.parquet")
    references = pd.read_parquet(FEATURES / "reference_targets.parquet").set_index("protein_id")
    single_order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    split = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet")
    family = pd.read_parquet(ROOT / "data/splits/families_30.parquet").set_index("protein_id")
    selected_root = "pair_repeats_full_g1_f0_j0" if args.geometry else "pair_repeats_full_g0_f0_j0_family30"
    selection = json.loads((ROOT / "runs/e3" / selected_root / "selected_hyperparameters.json").read_text())
    if args.geometry:
        assert doubles.metadata.structure_available.all()
    all_proteins = sorted(doubles.metadata.protein_id.unique())
    if args.smoke:
        eligible = sorted(set(split.loc[split.scheme.eq("unseen_pair"), "protein_id"]) &
                          set(references.index[references.point_in_range]))
        all_proteins = eligible[:max(1, round(len(eligible) * 0.01))]
    proteins = all_proteins[args.rank::args.world_size]
    index = pd.concat([singles.metadata[["variant_id", "matrix_row"]], doubles.metadata[["variant_id", "matrix_row"]]]).set_index("variant_id").matrix_row
    observations["matrix_row"] = observations.variant_id.map(index).fillna(-1).astype(int)
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    outputs = []
    for seed in args.seeds:
        labels = Labels(singles, doubles, seed)
        for protein in proteins:
            fold = int(family.loc[protein, "outer_group"])
            selected = ROOT / selection[f"family30/fold{fold}"]["run"]
            source = selected.with_name(selected.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
            source_manifest = json.loads((source / "manifest.json").read_text())
            assert source_manifest["complete"] and not source_manifest["smoke"]
            assert source_manifest["config"]["geometry"] == args.geometry
            permissions = json.loads((source / "permissions.json").read_text())
            relatives = set(family.index[family.family_group.eq(family.loc[protein, "family_group"])])
            assert relatives <= set(permissions["test"])
            assert not relatives & (set(permissions["train"]) | set(permissions["validation"]))
            current = split.loc[split.protein_id.eq(protein) & split.seed.eq(seed)]
            target_singles = single_order.loc[single_order.protein_id.eq(protein) & single_order.seed.eq(seed)]
            wt_ids = set(observations.loc[observations.protein_id.eq(protein) & observations.n_mutations.eq(0), "variant_id"])
            for scheme in args.schemes:
                partition = current.loc[current.scheme.eq(scheme)]
                if not len(partition):
                    continue
                candidate = partition.loc[partition["set"].eq("train")].sort_values("train_rank")
                for condition in args.conditions:
                    directory = output / protein / f"seed{seed}" / scheme / condition
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "manifest.json"
                    if path.exists():
                        assert json.loads(path.read_text())["complete"]
                        outputs.append(str(path.relative_to(ROOT)))
                        continue
                    budget = len(candidate) if condition == "Call" else int(condition[1:])
                    chosen = candidate.iloc[:budget]
                    allowed_ids = set(target_singles.variant_id) | set(chosen.variant_id) | wt_ids
                    evaluation = doubles.metadata.merge(partition[["variant_id", "set"]], on="variant_id", validate="one_to_one")
                    evaluation = evaluation.loc[evaluation["set"].isin(["validation", "test"])].copy()
                    assert not allowed_ids & set(evaluation.variant_id)
                    if scheme != "random":
                        assert not set(chosen.pair_id) & set(evaluation.pair_id)
                    raw = observations.loc[observations.variant_id.isin(allowed_ids)].copy()
                    raw["reference_point"] = float(references.loc[protein, "point"])
                    raw[["observation_id", "variant_id", "protein_id", "n_mutations", "lower", "upper"]].to_parquet(directory / "training_permissions.parquet", index=False)
                    manifest = {"complete": True, "smoke": args.smoke, "method": "shared" if args.supervision == "transfer" else "shared_target_only", "mode": "C",
                                "supervision": args.supervision, "supervised_source_weights_used": args.supervision == "transfer",
                                "source_validation_hyperparameters_used": True, "geometry": args.geometry,
                                "condition": condition, "scheme": scheme, "seed": seed, "protein_id": protein,
                                "source_commit": source_commit, "source_run": str(source.relative_to(ROOT)),
                                "source_family_exclusion_verified": True, "source_fold": fold,
                                "target_single_labels": len(target_singles), "target_wt_variants": len(wt_ids),
                                "target_combination_labels": len(chosen), "available_combination_labels": len(candidate),
                                "target_test_labels_used": 0, "budget_eligible": len(candidate) >= budget}
                    if len(candidate) < budget or not references.loc[protein, "point_in_range"] or not len(chosen):
                        manifest.update({"evaluated": False, "reason": "combination_budget_unavailable" if len(candidate) < budget or not len(chosen) else "reference_outside_continuous_range"})
                    else:
                        training = doubles.metadata.loc[doubles.metadata.variant_id.isin(chosen.variant_id)].copy()
                        assert raw.loc[raw.n_mutations.gt(0), "matrix_row"].ge(0).all()
                        initial_run = source
                        if args.supervision == "target_only":
                            initial_run = fit_target_main(source, directory.parent / "target_main", singles, doubles, raw,
                                                          evaluation.loc[evaluation["set"].eq("validation")], seed)
                            manifest["source_training_labels_used"] = 0
                            manifest["target_main_run"] = str(initial_run.relative_to(ROOT))
                        details = train(initial_run, directory, singles, doubles, labels, raw, training,
                                        evaluation.loc[evaluation["set"].eq("validation")], evaluation, seed)
                        manifest.update({"evaluated": True, **details, "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
                    path.write_text(json.dumps(manifest, indent=2))
                    outputs.append(str(path.relative_to(ROOT)))
                    print(json.dumps({key: manifest[key] for key in ["protein_id", "seed", "scheme", "condition", "evaluated"]}), flush=True)
    name = "_".join(args.schemes) + "_" + "_".join(args.conditions)
    name += "_seeds" + "_".join(str(value) for value in args.seeds)
    (output / f"manifest_{name}_rank{args.rank}.json").write_text(json.dumps({
        "complete": True, "config": vars(args), "source_commit": source_commit, "proteins": len(proteins),
        "outputs": outputs, "smoke": args.smoke,
    }, indent=2))


if __name__ == "__main__":
    main()
