import argparse
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from evaluate_family import load_model
from train_single import ROOT
from epistasis.higher import HigherFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    permissions = pd.read_parquet(ROOT / "data/processed/external_transfer/source_permissions.parquet")
    permissions = permissions.loc[permissions.threshold.eq(30)].sort_values(["dataset", "protein_id"])
    selection = json.loads((ROOT / "runs/e3/pair_repeats_full_g0_f0_j0_family30/selected_hyperparameters.json").read_text())
    output = ROOT / "data/features/transferred_effects"
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(row, seed) for row in permissions.itertuples() for seed in args.seeds]
    manifests = []
    for permission, seed in tasks[args.rank::args.world_size]:
        folder = output / permission.dataset / permission.protein_id / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text())["complete"]
            manifests.append(str(manifest_path.relative_to(ROOT)))
            continue
        assert not permission.requires_custom_source_exclusion
        data = HigherFeatures(ROOT / "data/features/higher" / permission.dataset / permission.protein_id)
        effects, interactions, checkpoints = [], [], []
        for fold in permission.eligible_source_test_folds:
            chosen = ROOT / selection[f"family30/fold{fold}"]["run"]
            source = chosen.with_name(chosen.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
            information = json.loads((source / "manifest.json").read_text())
            assert information["complete"] and not information["config"]["flexible"] and not information["config"]["geometry"]
            source_permissions = json.loads((source / "permissions.json").read_text())
            assert not set(permission.related_source_proteins) & (set(source_permissions["train"]) | set(source_permissions["validation"]))
            model = load_model(source)
            main_effect, pair_effect = data.infer_terms(model)
            effects.append(main_effect)
            interactions.append(pair_effect)
            checkpoints.append(str((source / "best.pt").relative_to(ROOT)))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        effects, interactions = np.mean(np.stack(effects), axis=0), np.mean(np.stack(interactions), axis=0)
        additive, interaction = data.combine(effects, interactions)
        assert np.isfinite(additive).all() and np.isfinite(interaction).all()
        np.savez_compressed(folder / "terms.npz", single=effects, pair=interactions,
                            additive=additive, interaction=interaction, latent=additive + interaction)
        data.metadata[["variant_id", "protein_id", "mutation_key", "n_mutations"]].to_parquet(folder / "variants.parquet", index=False)
        manifest = {"complete": True, "dataset": permission.dataset, "protein_id": permission.protein_id,
                    "seed": seed, "variants": len(data.metadata), "source_checkpoints": checkpoints,
                    "source_folds": [int(value) for value in permission.eligible_source_test_folds],
                    "ensemble": "Equal average of physical-scale allele and pair effects from all eligible source folds.",
                    "target_labels_used": 0, "geometry": False,
                    "family_exclusion_manifest": "data/processed/external_transfer/manifest.json",
                    "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifests.append(str(manifest_path.relative_to(ROOT)))
        print(json.dumps({"protein": permission.protein_id, "seed": seed, "folds": manifest["source_folds"],
                          "variants": len(data.metadata), "complete": True}), flush=True)
    (output / ("manifest_seeds" + "_".join(str(value) for value in args.seeds) + f"_rank{args.rank}.json")).write_text(json.dumps({
        "complete": True, "config": vars(args), "outputs": manifests}, indent=2))


if __name__ == "__main__":
    main()
