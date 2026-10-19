import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stabilityarc_common import CACHE, DATA, StabilityArc, load_features, mutation_arrays, native_training_targets, score_arrays, source_frames
from scripts.train_single import family_permissions
from epistasis.gradient_validation import finite_nonzero_gradients


def source_directory(threshold, fold, seed):
    return ROOT / "runs/stabilityarc_source" / f"family{threshold}/fold{fold}/seed{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[30, 25], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=[11, 23, 47, 89, 131], required=True)
    args = parser.parse_args()
    assert json.loads((CACHE / "manifest.json").read_text())["complete"]
    assert json.loads((ROOT / "runs/stabilityarc_validation/manifest.json").read_text())["complete"]
    directory = source_directory(args.threshold, args.fold, args.seed)
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "manifest.json").exists()
    permissions = family_permissions(args.threshold, args.fold)
    (directory / "permissions.json").write_text(json.dumps(permissions, indent=2))
    frames = source_frames()
    proteins = pd.read_parquet(DATA / "proteins.parquet").set_index("protein_id")
    train_ids = sorted(set(permissions["train"]) & set(frames))
    validation_ids = sorted(protein for protein in set(permissions["validation"]) & set(frames)
                            if frames[protein].n_mutations.eq(2).sum() >= 30)
    assert train_ids and validation_ids and not set(train_ids) & (set(permissions["validation"]) | set(permissions["test"]))
    training = pd.concat([frames[protein][["variant_id", "protein_id", "n_mutations", "relative_point"]]
                          for protein in train_ids], ignore_index=True)
    training.to_parquet(directory / "permitted_training_variants.parquet", index=False)
    normalizations = {protein: native_training_targets(frames[protein])[1] for protein in train_ids}
    (directory / "normalizations.json").write_text(json.dumps(normalizations, indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    rng = np.random.default_rng(args.seed)
    model = StabilityArc(model_dim=256, depth=4, num_heads=8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    history, seen_ids = [], set()
    best, best_epoch = -float("inf"), -1
    started = time.monotonic()
    for epoch in range(40):
        model.train()
        total_loss = 0.0
        for protein in rng.permutation(train_ids):
            frame = frames[protein]
            target, _ = native_training_targets(frame)
            if len(frame) > 40000:
                chosen = rng.choice(len(frame), 40000, replace=False)
                target = target[chosen]
                frame = frame.iloc[chosen].reset_index(drop=True)
            seen_ids.update(frame.variant_id)
            sequence = proteins.loc[protein, "reference_sequence"]
            embedding, contacts = load_features(protein, sequence, "cuda")
            arrays = mutation_arrays(frame, sequence, "cuda")
            prediction, _ = score_arrays(model, embedding, contacts, arrays)
            targets = torch.as_tensor(target, device="cuda")
            huber = F.huber_loss(prediction, targets, delta=1.0)
            pairs = torch.as_tensor(rng.integers(len(frame), size=(1024, 2)), device="cuda")
            measured = torch.as_tensor(frame.relative_point.to_numpy(dtype=np.float32), device="cuda")
            signs = torch.sign(measured[pairs[:, 0]] - measured[pairs[:, 1]])
            active = signs.ne(0)
            assert active.any()
            ranking = F.softplus(-signs[active] * (prediction[pairs[active, 0]] - prediction[pairs[active, 1]])).mean()
            loss = huber + ranking
            assert torch.isfinite(loss), (protein, epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert finite_nonzero_gradients(model.named_parameters())
            optimizer.step()
            total_loss += float(loss.detach())
        model.eval()
        correlations = []
        with torch.inference_mode():
            for protein in validation_ids:
                frame = frames[protein].loc[frames[protein].n_mutations.eq(2)].reset_index(drop=True)
                sequence = proteins.loc[protein, "reference_sequence"]
                embedding, contacts = load_features(protein, sequence, "cuda")
                prediction, _ = score_arrays(model, embedding, contacts, mutation_arrays(frame, sequence, "cuda"))
                correlation = float(spearmanr(prediction.cpu().numpy(), frame.relative_point.to_numpy()).statistic)
                assert np.isfinite(correlation), protein
                correlations.append(correlation)
        score = float(np.mean(correlations))
        entry = {"epoch": epoch, "training_loss": total_loss / len(train_ids),
                 "validation_macro_double_spearman": score, "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        (directory / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "epoch": epoch, "config": vars(args)}, directory / "best.pt")
    assert np.isfinite(best) and len(history) == 40
    training.loc[training.variant_id.isin(seen_ids)].to_parquet(directory / "sampled_training_variants.parquet", index=False)
    manifest = {"complete": True, "smoke": False, "config": vars(args), "epochs": 40,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "vendor_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "external/stabilityarc", text=True).strip(),
                "architecture": "Unmodified author StabilityArc: 4-layer 256-dimensional 8-head RoPE backbone and full contact-aware symmetric residual",
                "source_labels_used": len(training), "source_training_variants": len(training),
                "sampled_training_variants": len(seen_ids), "training_proteins": len(train_ids),
                "validation_proteins": len(validation_ids), "target_labels_used": 0,
                "label_permissions_verified": True, "validation_score": best, "best_epoch": best_epoch,
                "optimization": {"epochs": 40, "optimizer": "AdamW", "learning_rate": 1e-4,
                                 "weight_decay": 1e-3, "ranking_pairs_per_assay": 1024,
                                 "maximum_measured_variants_per_assay": 40000, "huber_delta": 1.0,
                                 "huber_weight": 1.0, "ranking_weight": 1.0},
                "training_targets": "Point-valued in-range stability records. Singles use WT-relative kcal/mol; doubles are centered and scaled by source-assay single median and MAD. Ranking uses measured WT-relative ordering.",
                "reimplementation_scope": "Author architecture and published optimization protocol on matched project source partitions; unpublished training code and held-out checkpoints are unavailable. Equal loss weights and source-validation epoch selection are explicitly specified here.",
                "selection": "Best source-validation double-mutant macro Spearman among all 40 completed epochs",
                "checkpoint": str((directory / "best.pt").relative_to(ROOT)),
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "elapsed_seconds": time.monotonic() - started}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
