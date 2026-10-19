import gc
import json
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
import torch

from epistasis.metrics import per_protein_metrics
from epistasis.msr import ROOT, SourceTraining, evaluation_precision, make_loader, offline_model_directory


class TargetQuartets:
    def __init__(self, metadata, table):
        self.table = table.reset_index(drop=True)
        assert len(self.table) and self.table.protein_id.nunique() == 1
        direct = metadata.set_index("variant_id")
        self.a = direct.loc[self.table.variant_id, "single_a_id"].to_numpy()
        self.b = direct.loc[self.table.variant_id, "single_b_id"].to_numpy()
        self.wt = ("msr_reference_" + self.table.protein_id).to_numpy()
        self.ids = set(self.table.variant_id) | set(self.a) | set(self.b) | set(self.wt)
        self.inputs = metadata.loc[metadata.variant_id.isin(self.ids)].copy()
        assert len(self.inputs) == len(self.ids)

    def evaluate(self, native):
        assert native.index.is_unique and set(native.index) == self.ids and np.isfinite(native).all()
        table = self.table.copy()
        reference = native.loc[self.wt].to_numpy()
        table["prediction_mu"] = -native.loc[table.variant_id].to_numpy() + reference
        table["prediction_a"] = -native.loc[self.a].to_numpy() + reference
        table["prediction_b"] = -native.loc[self.b].to_numpy() + reference
        table["prediction_wt"] = 0.0
        table["prediction_epsilon"] = table.prediction_mu - table.prediction_a - table.prediction_b
        metrics = per_protein_metrics(table)
        eligible = table.loc[table.main.fillna(False)]
        score = float(np.sqrt(np.mean((eligible.prediction_epsilon - eligible.epsilon) ** 2))) if len(eligible) else None
        return score, table, metrics


class TargetTraining(SourceTraining):
    def on_train_epoch_end(self):
        if self.evaluator is not None:
            return
        assert self.nonzero_gradients
        self.best_epoch = self.current_epoch
        self.history.append({"epoch": self.current_epoch, "steps": self.global_step,
                             "gradient_checks": self.gradient_checks, "validation_score": None})
        pd.DataFrame(self.history).to_csv(self.directory / "history.csv", index=False)
        state = {"state_dict": self.state_dict()}
        self.on_save_checkpoint(state)
        torch.save(state, self.directory / "best.ckpt")


def infer_native(module, evaluator, configuration):
    loader = make_loader(evaluator.inputs, False, batch_size=16)
    module.cuda().eval()
    identifiers, predictions = [], []
    assert configuration["evaluation_precision"] == "fp32"
    with torch.inference_mode(), evaluation_precision():
        for batch in loader:
            identifiers.extend(batch["variant_id"])
            batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            predictions.extend(module.model.forward_batch(batch, mask_strategy=configuration["mask_strategy"])[
                "combined_pred"].cpu().float().numpy().tolist())
    return pd.Series(predictions, index=identifiers, name="native_prediction")


def restore_adapters(module, checkpoint):
    missing, unexpected = module.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not unexpected and checkpoint["state_dict"]
    assert not [name for name in missing if "lora" in name.lower() or "calibration" in name.lower()]
    module.on_load_checkpoint(checkpoint)
    assert not module.peft_manager.has_transitioned
    module.peft_manager.apply_baseline_requires_grad()
    current = module.state_dict()
    assert all(torch.equal(current[name].detach().cpu(), value.cpu()) for name, value in checkpoint["state_dict"].items())


def fit_target(source, metadata, doubles, training, job, directory, smoke, module=None):
    information = json.loads((source / "manifest.json").read_text())
    configuration = json.loads((source / "config.json").read_text())["author"]
    configuration.update(checkpoint_path=str(directory), log_dir=str(directory),
                         experiment_name=directory.name, val_dataloader_names=["target_validation"])
    assert configuration["evaluation_precision"] == "fp32"
    assert configuration["subset_size"] == 16 and configuration["micro_batch_size"] >= 16
    assert len(training) >= 16 and np.isfinite(training.native_ddg).all()
    configuration["batch_size"] = min(configuration["batch_size"], max(16, len(training) // 16 * 16))
    configuration["batch_size"] = 16 if smoke else configuration["batch_size"]
    selection = doubles.loc[doubles.variant_id.isin(job["validation_ids"]) & doubles.main.fillna(False)].copy()
    evaluator = TargetQuartets(metadata, selection) if len(selection) else None
    allowed = set(training.variant_id)
    assert not allowed & (set(job["validation_ids"]) | set(job["test_ids"]))
    (directory / "config.json").write_text(json.dumps({"author": configuration}, indent=2))
    pl.seed_everything(job["seed"], workers=True)
    logging.getLogger().setLevel(logging.WARNING)
    offline_model_directory()
    if module is None:
        module = TargetTraining(**configuration)
    else:
        architecture = ["adapter_mode", "lora_mode", "lora_rank_wt", "lora_rank_mt", "lora_alpha_wt", "lora_alpha_mt"]
        assert all(module.hparams[key] == configuration[key] for key in architecture)
        module.hparams.update(configuration)
        module.val_dataloader_names = configuration["val_dataloader_names"]
    source_state = torch.load(source / "best.ckpt", map_location="cpu", weights_only=True)
    restore_adapters(module, source_state)
    module.validation_step_outputs.clear()
    module.attach_evaluation(evaluator, directory)
    train_loader = make_loader(training, True, job["seed"], configuration["batch_size"], allowed_labels=allowed)
    assert len(train_loader) > 0
    epochs = 100 if evaluator is not None else min(100, max(1, information["best_epoch"] + 1))
    epochs = min(epochs, 2) if smoke else epochs
    callbacks = [EarlyStopping(monitor="validation_macro_epsilon_rmse", patience=10, mode="min", check_finite=True)] if evaluator is not None else []
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=configuration["precision"],
                         max_epochs=epochs, num_sanity_val_steps=0, callbacks=callbacks,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         logger=CSVLogger(str(directory), name="logs"), log_every_n_steps=1,
                         default_root_dir=str(directory), deterministic=False)
    validation_loader = make_loader(evaluator.inputs, False, batch_size=16) if evaluator is not None else None
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    assert module.gradient_checks and module.seen_training_ids <= allowed
    assert any("wt_adapter" in name for name in module.nonzero_gradients)
    assert any("mt_adapter" in name for name in module.nonzero_gradients)
    state = torch.load(directory / "best.ckpt", map_location="cpu", weights_only=True)
    assert any(not torch.equal(value, source_state["state_dict"][name]) for name, value in state["state_dict"].items())
    restore_adapters(module, state)
    training.loc[training.variant_id.isin(module.seen_training_ids)].to_parquet(directory / "sampled_training_variants.parquet", index=False)
    target = TargetQuartets(metadata, doubles.loc[doubles.variant_id.isin(set(job["validation_ids"]) | set(job["test_ids"]))])
    native = infer_native(module, target, configuration)
    native.rename_axis("variant_id").reset_index().to_parquet(directory / "all_genotype_predictions.parquet", index=False)
    _, predicted, _ = target.evaluate(native)
    replay_difference = None
    if evaluator is not None:
        previous = pd.read_parquet(directory / "validation_predictions.parquet").set_index("variant_id")
        current = predicted.set_index("variant_id").loc[previous.index]
        np.testing.assert_allclose(current.prediction_mu, previous.prediction_mu, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(current.prediction_epsilon, previous.prediction_epsilon, rtol=1e-5, atol=1e-5)
        replay_difference = float(np.max(np.abs(current.prediction_epsilon - previous.prediction_epsilon)))
    details = {"source_training_variants": information["training_variants"], "target_training_variants": len(training),
               "sampled_training_variants": len(module.seen_training_ids), "training_presentations": module.training_presentations,
               "epochs": len(module.history), "best_epoch": module.best_epoch, "max_epochs": epochs,
               "validation_score": module.best_score if evaluator is not None else None,
               "gradient_checks": module.gradient_checks, "nonzero_gradient_tensors": sorted(module.nonzero_gradients),
               "source_checkpoint_initialization_verified": True, "source_selected_learning_rate": configuration["learning_rate"],
               "source_selected_weight_decay": configuration["weight_decay"], "batch_size": configuration["batch_size"],
               "selection": "Target validation epsilon RMSE." if evaluator is not None else "Fixed source-validation selected epoch count.",
               "checkpoint_validation_replay_difference": replay_difference,
               "training_precision": configuration["precision"], "evaluation_precision": "fp32",
               "training_objective": "Original two-branch streaming ListMLE and MSE, teacher forcing on permitted component labels only.",
               "scheduler": "Original warmup; validation plateau updates only when target validation labels are available.",
               "both_branches": True, "four_model_predictions": True}
    del trainer, train_loader, validation_loader, source_state, state
    gc.collect()
    torch.cuda.empty_cache()
    return predicted, details, module
