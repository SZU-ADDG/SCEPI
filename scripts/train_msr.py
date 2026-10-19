import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.msr import FEATURES, QuartetEvaluation, SourceTraining, make_loader, offline_model_directory, training_configuration
from scripts.train_single import family_permissions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[25, 30], default=30)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    name = f"lr{args.learning_rate:g}_wd{args.weight_decay:g}_seed{args.seed}"
    output = ROOT / "runs" / ("msr_family_smoke" if args.smoke else "msr_family")
    directory = output / f"family{args.threshold}" / f"fold{args.fold}" / name
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "manifest.json").exists()
    permissions = family_permissions(args.threshold, args.fold)
    metadata = pd.read_parquet(FEATURES / "variants.parquet")
    training = metadata.loc[metadata.protein_id.isin(permissions["train"]) & metadata.training_point_eligible].copy()
    total_training = len(training)
    if args.smoke:
        training = training.iloc[::100].copy()
    assert training.n_mutations.isin([1, 2]).all() and np.isfinite(training.native_ddg).all()
    assert not set(training.protein_id) & (set(permissions["validation"]) | set(permissions["test"]))
    labels = set(training.variant_id)
    evaluator = QuartetEvaluation(metadata, permissions["validation"], smoke=args.smoke)
    assert not labels & evaluator.ids
    training[["variant_id", "protein_id", "native_ddg", "single_a_id", "single_b_id"]].to_parquet(
        directory / "training_variants.parquet", index=False)
    evaluator.inputs[["variant_id", "protein_id"]].to_parquet(directory / "validation_variants.parquet", index=False)
    (directory / "permissions.json").write_text(json.dumps(permissions, indent=2))
    config = training_configuration(args.seed, args.learning_rate, args.weight_decay, directory)
    if args.smoke:
        config["batch_size"] = config["subset_size"]
    (directory / "config.json").write_text(json.dumps({"request": vars(args), "author": config}, indent=2))
    pl.seed_everything(args.seed, workers=True)
    logging.getLogger().setLevel(logging.WARNING)
    train_loader = make_loader(training, True, args.seed, config["batch_size"], allowed_labels=labels)
    assert len(train_loader) > 0
    validation_loader = make_loader(evaluator.inputs, False, batch_size=16)
    offline_model_directory()
    module = SourceTraining(**config)
    module.attach_evaluation(evaluator, directory)
    trainable = {name: parameter.numel() for name, parameter in module.model.named_parameters() if parameter.requires_grad}
    assert trainable and all("lora" in name.lower() or "calibration" in name.lower() for name in trainable)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=config["precision"],
        max_epochs=2 if args.smoke else 100, num_sanity_val_steps=0,
        callbacks=[EarlyStopping(monitor="validation_macro_epsilon_rmse", patience=10, mode="min", check_finite=True)],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        logger=CSVLogger(str(directory), name="logs"), log_every_n_steps=50,
        default_root_dir=str(directory), deterministic=False)
    started = time.monotonic()
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    assert module.gradient_checks and module.nonzero_gradients and (directory / "best.ckpt").is_file()
    assert module.seen_training_ids <= labels
    training.loc[training.variant_id.isin(module.seen_training_ids)].to_parquet(
        directory / "sampled_training_variants.parquet", index=False)
    assert any("wt_adapter" in name for name in module.nonzero_gradients)
    assert any("mt_adapter" in name for name in module.nonzero_gradients)
    manifest = {
        "complete": True, "smoke": args.smoke, "config": vars(args),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/esm-msr"), "rev-parse", "HEAD"], text=True).strip(),
        "training_variants": len(training), "full_training_variants": total_training,
        "training_fraction": len(training) / total_training,
        "sampled_training_variants": len(module.seen_training_ids),
        "training_presentations": module.training_presentations,
        "sampler": "Original balanced protein batches; training_variants records permitted inputs, sampled_training_variants records actually sampled genotypes.",
        "validation_variants": len(evaluator.inputs), "validation_quartets": len(evaluator.table),
        "epochs": len(module.history), "best_epoch": module.best_epoch, "validation_score": module.best_score,
        "trainable_parameters": sum(trainable.values()), "trainable_tensors": len(trainable),
        "gradient_checks": module.gradient_checks, "nonzero_gradient_tensors": sorted(module.nonzero_gradients),
        "label_permissions_verified": True,
        "pretraining": "Frozen ESM3 base; fresh WT and MT LoRA adapters and calibration heads for the current family split.",
        "target_labels_used": 0, "selection": "Source-validation macro epsilon RMSE from four model predictions.",
        "training_precision": config["precision"], "evaluation_precision": "fp32",
        "author_training": "Original streaming ListMLE and MSE losses, two-branch teacher forcing, optimizer, warmup and plateau scheduler.",
        "checkpoint": str((directory / "best.ckpt").relative_to(ROOT)),
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.monotonic() - started}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
