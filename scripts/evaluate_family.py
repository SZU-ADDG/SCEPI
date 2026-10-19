import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from train_pair import Labels
from train_single import ROOT, FeatureArrays, family_permissions
from epistasis.metrics import per_protein_metrics
from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect


def load_model(run):
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = EpistasisModel(
        SingleEffect(ModelConfig(**checkpoint["single_config"])),
        PairEffect(ModelConfig(**checkpoint["pair_config"])),
        PhenotypeMap(checkpoint["flexible"]), omit_pair=checkpoint.get("omit_pair", False),
    )
    model.load_state_dict(checkpoint["model"])
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--partition", choices=["test", "validation"], default="test")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--geometry-intervention", choices=["none", "permuted", "rewired"], default="none")
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    assert run.is_relative_to(ROOT)
    source = json.loads((run / "manifest.json").read_text())
    assert source["complete"] and not source["smoke"]
    config = source["config"]
    suffix = "_" + args.geometry_intervention if args.geometry_intervention != "none" else ""
    output = run / ("evaluation_" + args.partition + suffix)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists()
    model = load_model(run)
    intervention = args.geometry_intervention if args.geometry_intervention != "none" else config.get("geometry_intervention", "none")
    assert config["geometry"] or intervention == "none"
    singles, doubles = FeatureArrays("singles"), FeatureArrays("doubles", intervention)
    labels = Labels(singles, doubles, source["seed"])
    permissions = family_permissions(config["threshold"], config["fold"])
    saved = json.loads((run / "permissions.json").read_text())
    assert permissions == saved
    proteins = permissions[args.partition]
    assert not set(proteins) & set(permissions["train"])
    table = doubles.metadata.loc[doubles.metadata.protein_id.isin(proteins)].copy()
    if config["geometry"]:
        table = table.loc[table.structure_available]
    family = pd.read_parquet(ROOT / "data/splits" / f"families_{config['threshold']}.parquet")
    target_counts = singles.metadata.assign(available=labels.single_rank >= 0).groupby("protein_id").available.sum()
    metadata = family.set_index("protein_id")
    metadata["available_single_labels"] = target_counts.reindex(metadata.index, fill_value=0)
    metadata = metadata.loc[metadata.index.isin(proteins)].copy()
    metadata.to_parquet(output / "target_permissions.parquet")
    started = time.monotonic()
    records = []
    summaries = []
    for condition, budget in [("Z", 0), ("S32", 32), ("S128", 128), ("Sall", -1)]:
        results = {"shared": [], "remove_interaction": []}
        omega = 0.0 if budget == 0 else source["omega"]
        with torch.inference_mode():
            for offset in range(0, len(table), args.batch_size):
                group = table.iloc[offset:offset + args.batch_size]
                rows = group.matrix_row.to_numpy()
                ranks = labels.pair_rank[rows]
                allowed = (ranks >= 0) & ((ranks < budget) if budget >= 0 else True)
                values = torch.from_numpy(labels.pair_value[rows]).cuda()
                mask = torch.from_numpy(allowed).cuda()
                prediction = model(doubles.batch(rows), values, mask, omega=omega)
                for method in results:
                    current = group.copy()
                    mu = prediction["phenotype"] if method == "shared" else model.mapping(model.calibrated_effects(doubles.batch(rows), values, mask, omega).sum(-1))
                    a, b = prediction["single"][:, 0], prediction["single"][:, 1]
                    current["prediction_mu"] = mu.cpu().numpy()
                    current["prediction_a"] = a.cpu().numpy()
                    current["prediction_b"] = b.cpu().numpy()
                    current["prediction_wt"] = 0.0
                    current["prediction_epsilon"] = (mu - a - b).cpu().numpy()
                    if method == "remove_interaction" and not model.mapping.flexible:
                        current["prediction_epsilon"] = 0.0
                    current["observed_a"] = allowed[:, 0]
                    current["observed_b"] = allowed[:, 1]
                    current["condition"] = condition
                    current["method"] = method
                    current["omega"] = omega
                    current["seed"] = source["seed"]
                    results[method].append(current)
        for method, chunks in results.items():
            frame = pd.concat(chunks, ignore_index=True)
            assert len(frame) == len(table) and set(frame.variant_id) == set(table.variant_id)
            frame = frame.merge(metadata.reset_index(), on="protein_id", validate="many_to_one")
            frame["budget_eligible"] = True if budget <= 0 else frame.available_single_labels.ge(budget)
            frame["target_single_labels"] = 0 if budget == 0 else (frame.available_single_labels if budget < 0 else frame.available_single_labels.clip(upper=budget))
            frame["target_wt_labels"] = int(budget != 0)
            frame.to_parquet(output / f"{method}_{condition}.parquet", index=False)
            per_protein = per_protein_metrics(frame)
            per_protein = per_protein.merge(metadata.reset_index(), on="protein_id", validate="one_to_one")
            per_protein["budget_eligible"] = True if budget <= 0 else per_protein.available_single_labels.ge(budget)
            per_protein["condition"], per_protein["method"] = condition, method
            per_protein["seed"], per_protein["fold"] = source["seed"], config["fold"]
            records.append(per_protein)
            eligible = per_protein.loc[per_protein.main_eligible & per_protein.budget_eligible]
            selection = per_protein.loc[per_protein.selection_eligible & per_protein.budget_eligible]
            summaries.append({
                "condition": condition, "method": method, "proteins": len(eligible),
                "annotated_family_groups": eligible.loc[eligible.pfam_annotated, "family_group"].nunique(),
                "macro_epsilon_rmse": float(eligible.epsilon_rmse.mean()),
                "pair_macro_epsilon_rmse": float(eligible.pair_macro_epsilon_rmse.mean()),
                "selection_proteins": len(selection),
                "macro_precision10_lower": float(selection.precision10_lower.mean()),
                "macro_precision10_upper": float(selection.precision10_upper.mean()),
            })
    pd.concat(records, ignore_index=True).to_parquet(output / "per_protein.parquet", index=False)
    manifest = {
        "complete": True, "partition": args.partition, "training_run": str(run.relative_to(ROOT)),
        "geometry_intervention": intervention,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "threshold": config["threshold"], "fold": config["fold"], "seed": source["seed"], "variants": len(table),
        "selection_scope": "Checkpoint and omega were fixed using source validation only before this evaluation.",
        "predictions": "All accepted double variants in this partition; metrics retain main, margin, interval and selection flags.",
        "summary": summaries, "elapsed_seconds": time.monotonic() - started,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
