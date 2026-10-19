import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm_msr.config import parse_arguments
from esm_msr.data import ProteinCyclingBatchSampler, ProteinStructureMutationEpistasisDataset, collate_fn_twopass
from esm_msr.training import ESM3EpistasisLightningModule

from epistasis.metrics import per_protein_metrics

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/features/esm_msr/tsuboyama"


@contextmanager
def evaluation_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type="cuda", enabled=False), torch.backends.cudnn.flags(allow_tf32=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


def offline_model_directory():
    os.environ["INFRA_PROVIDER"] = "1"
    os.chdir(ROOT / "models/esm3-sm-open-v1")
    assert Path("data/weights/esm3_sm_open_v1.pth").is_file()


def training_configuration(seed, learning_rate, weight_decay, directory):
    original = sys.argv
    sys.argv = ["esm-msr", "--raw_data_file", str(FEATURES / "variants.parquet"),
                "--af_model_folder", str(FEATURES / "structures"),
                "--experiment_name", directory.name]
    config = vars(parse_arguments())
    sys.argv = original
    released = yaml.safe_load((ROOT / "models/esm-msr/hparams.yaml").read_text())
    keys = [
        "adapter_mode", "lora_mode", "shared_scale_init", "shared_bias_init",
        "rank_loss", "reg_loss", "huber_delta", "lambda_rank_wt", "lambda_rank_combined",
        "lambda_reg_wt", "lambda_reg_combined", "lambda_epi_combined", "mt_reg_mask",
        "double_weight", "reversion_weight", "mut_ctx_weight", "residual_wd",
        "calib_lr_mult", "residual_lr_mult", "detach_ensemble_input", "detach_calibration",
        "detach_regression", "zero_epistasis_for_singles", "subset_size", "invert_list_loss",
        "mask_strategy", "premask_coords", "batch_size", "micro_batch_size",
        "lr_warmup_steps", "mt_lora_delay_steps", "calib_delay_steps", "grad_clip_norm",
        "precision", "use_plddt", "loader_strategy", "subset_caps",
        "incl_singles", "incl_doubles", "incl_mut_ctx", "incl_reversions",
    ]
    for branch in ["wt", "mt"]:
        keys.extend(name + "_" + branch for name in [
            "lora_rank", "lora_alpha", "lora_dropout", "incl_structure_encoder",
            "incl_sequence_head", "last_n_layers", "target_mode", "unfreeze_layernorms", "use_dora"])
    for key in keys:
        if key in released:
            config[key] = released[key]
    config = {key: value for key, value in config.items() if not key.startswith("comet_")}
    config.update(seed=seed, learning_rate=learning_rate, weight_decay=weight_decay,
                  num_epochs=100, freeze_wt_after_epoch=1000, freeze_wt_on_convergence=False,
                  freeze_wt_adapter=False, freeze_mt_adapter=False, load_lora_checkpoint=None,
                  offline_model=True, checkpoint_path=str(directory), log_dir=str(directory),
                  cache_path=str(FEATURES), benchmark_data_path=str(FEATURES),
                  mut_structures_root=str(ROOT / "data/extracted/thermompnn/FINAL_results"),
                  num_workers=0, val_dataloader_names=["source_validation"], evaluation_precision="fp32")
    assert config["incl_singles"] and config["incl_doubles"]
    assert not config["incl_mut_ctx"] and not config["incl_reversions"]
    assert config["micro_batch_size"] >= config["subset_size"]
    return config


class DirectDataset(ProteinStructureMutationEpistasisDataset):
    def __init__(self, table, allowed_labels=None):
        assert table.protein_id.nunique() == 1
        self.dms_name = table.protein_id.iloc[0]
        self.tokenizer = EsmSequenceTokenizer()
        self.vocab = self.tokenizer.get_vocab()
        structure = torch.load(FEATURES / "structures" / (self.dms_name + ".pt"),
                               map_location="cpu", weights_only=True)
        static = {key: value.numpy() for key, value in structure.items() if torch.is_tensor(value)}
        self.data = []
        for row in table.itertuples():
            assert row.background_sequence == structure["sequence"]
            mutations = [(wild, int(position) + 1, mutant)
                         for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)]
            ddg_a, ddg_b = row.native_single_a, row.native_single_b
            if allowed_labels is not None and row.n_mutations == 2:
                if row.single_a_id not in allowed_labels:
                    ddg_a = np.nan
                if row.single_b_id not in allowed_labels:
                    ddg_b = np.nan
            additive = ddg_a + ddg_b if row.n_mutations == 2 else row.native_ddg
            epsilon = row.native_ddg - additive if row.n_mutations == 2 else np.nan
            item = self._create_data_item(
                mutations, row.native_ddg, epsilon, additive, ddg_a, ddg_b,
                self.dms_name, row.background_sequence, row.aa_seq,
                structure["coords_orig"], structure["plddt"], structure["structure_tokens_orig"],
                structure["residue_index"], "double" if row.n_mutations == 2 else "single", "wt")
            # 同一蛋白质共享只读结构，避免为每个变体复制坐标。
            item.update(static)
            item["variant_id"] = row.variant_id
            self.data.append(item)
        self._extract_scalars()
        self.ground_truth_arr = np.asarray([item["ddG"] for item in self.data], dtype=np.float32)


def collate_direct(items):
    batch = collate_fn_twopass(items)
    # 空基因型使用显式掩码，填充值不代表突变。
    for key in ["mut_pos", "wt_id", "mt_id"]:
        if batch[key].shape[1] == 0:
            batch[key] = F.pad(batch[key], (0, 1), value=0)
        assert batch[key].shape == batch["mut_mask"].shape
    batch["variant_id"] = [item["variant_id"] for item in items]
    return batch


def make_loader(table, training, seed=11, batch_size=256, allowed_labels=None):
    datasets = [DirectDataset(group, allowed_labels) for _, group in table.groupby("protein_id", sort=True)]
    combined = ConcatDataset(datasets)
    if training:
        sampler = ProteinCyclingBatchSampler(
            datasets, batch_size, [dataset.dms_name for dataset in datasets],
            strategy="all", subset_caps={"single": None, "double": None, "mut_ctx": 0, "reversion": 0},
            rng_seed=seed, verbose=False)
        return DataLoader(combined, batch_sampler=sampler, collate_fn=collate_direct, num_workers=0)
    batches, offset = [], 0
    for dataset in datasets:
        batches.extend(list(range(offset + start, offset + min(start + batch_size, len(dataset))))
                       for start in range(0, len(dataset), batch_size))
        offset += len(dataset)
    return DataLoader(combined, batch_sampler=batches, collate_fn=collate_direct, num_workers=0)


class QuartetEvaluation:
    def __init__(self, metadata, proteins, smoke=False):
        table = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
        table = table.loc[table.protein_id.isin(proteins)].copy()
        if smoke:
            table = table.iloc[::100].copy()
        self.table = table.reset_index(drop=True)
        direct = metadata.set_index("variant_id")
        self.a = direct.loc[self.table.variant_id, "single_a_id"].to_numpy()
        self.b = direct.loc[self.table.variant_id, "single_b_id"].to_numpy()
        self.wt = ("msr_reference_" + self.table.protein_id).to_numpy()
        self.ids = set(self.table.variant_id) | set(self.a) | set(self.b) | set(self.wt)
        self.inputs = metadata.loc[metadata.variant_id.isin(self.ids)].copy()
        assert len(self.ids) == len(self.inputs) and len(self.table)
        self.smoke = smoke

    def evaluate(self, native):
        assert native.index.is_unique and set(native.index) == self.ids
        assert np.isfinite(native).all()
        result = self.table.copy()
        reference = native.loc[self.wt].to_numpy()
        result["prediction_mu"] = -native.loc[result.variant_id].to_numpy() + reference
        result["prediction_a"] = -native.loc[self.a].to_numpy() + reference
        result["prediction_b"] = -native.loc[self.b].to_numpy() + reference
        result["prediction_wt"] = 0.0
        result["prediction_epsilon"] = result.prediction_mu - result.prediction_a - result.prediction_b
        metrics = per_protein_metrics(result)
        selected = metrics.loc[metrics.main_quartets.gt(0) if self.smoke else metrics.main_eligible]
        assert len(selected) and np.isfinite(selected.epsilon_rmse).all()
        return float(selected.epsilon_rmse.mean()), result, metrics


class SourceTraining(ESM3EpistasisLightningModule):
    def attach_evaluation(self, evaluator, directory):
        self.evaluator = evaluator
        self.directory = directory
        self.history = []
        self.best_score = float("inf")
        self.best_epoch = -1
        self.gradient_checks = 0
        self.nonzero_gradients = set()
        self.seen_training_ids = set()
        self.training_presentations = 0

    def training_step(self, batch, batch_idx):
        identifiers = batch["variant_id"]
        assert len(identifiers) == len(set(identifiers))
        self.seen_training_ids.update(identifiers)
        self.training_presentations += len(identifiers)
        return super().training_step(batch, batch_idx)

    def on_after_backward(self):
        for name, value in self.model.named_parameters():
            if value.grad is not None:
                assert torch.isfinite(value.grad).all(), name
                if torch.count_nonzero(value.grad):
                    self.nonzero_gradients.add(name)
        self.gradient_checks += 1

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with evaluation_precision():
            super().validation_step(batch, batch_idx, dataloader_idx)
        self.validation_step_outputs[dataloader_idx][-1]["variant_id"] = batch["variant_id"]

    def on_validation_epoch_end(self):
        parts = self.validation_step_outputs[0]
        native = pd.Series(np.concatenate([part["comb_scores"] for part in parts]),
                           index=[value for part in parts for value in part["variant_id"]])
        score, prediction, metrics = self.evaluator.evaluate(native)
        self.log("validation_macro_epsilon_rmse", score, on_epoch=True, prog_bar=True, batch_size=1)
        entry = {"epoch": self.current_epoch, "validation_score": score, "steps": self.global_step,
                 "gradient_checks": self.gradient_checks}
        self.history.append(entry)
        pd.DataFrame(self.history).to_csv(self.directory / "history.csv", index=False)
        if score < self.best_score:
            self.best_score, self.best_epoch = score, self.current_epoch
            state = {"state_dict": self.state_dict()}
            self.on_save_checkpoint(state)
            assert state["state_dict"] and self.nonzero_gradients
            torch.save(state, self.directory / "best.ckpt")
            prediction.to_parquet(self.directory / "validation_predictions.parquet", index=False)
            metrics.to_parquet(self.directory / "validation_metrics.parquet", index=False)
        print(json.dumps(entry), flush=True)
        # 作者表型指标只接收实测标签；四项预测指标保留全部反事实输入。
        for part in parts:
            observed = np.isfinite(part["ground_truths"])
            for key in ["wt_scores", "comb_scores", "epi_scores", "ground_truths", "dddG_truths"]:
                part[key] = part[key][observed]
            for key in ["pdb", "mutations", "variant_id"]:
                part[key] = [value for value, keep in zip(part[key], observed) if keep]
        assert sum(len(part["ground_truths"]) for part in parts) > 0
        super().on_validation_epoch_end()
