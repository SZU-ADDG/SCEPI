import argparse
import gc
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
from scripts.stabilityarc_common import AMINO_ACID_TO_INDEX, DATA, StabilityArc, load_features, mutation_arrays, score_arrays
from scripts.train_single import family_permissions
from scripts.train_stabilityarc import source_directory
from scripts.run_kermut_targets import build_inputs, fit_and_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, choices=[30, 25], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=[11, 23, 47, 89, 131], required=True)
    args = parser.parse_args()
    source = source_directory(args.threshold, args.fold, args.seed)
    source_manifest = json.loads((source / "manifest.json").read_text())
    assert source_manifest["complete"] and source_manifest["target_labels_used"] == 0
    permissions = family_permissions(args.threshold, args.fold)
    assert json.loads((source / "permissions.json").read_text()) == permissions
    payload = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    model = StabilityArc(model_dim=256, depth=4, num_heads=8).cuda().eval()
    model.load_state_dict(payload["model"], strict=True)
    variants = pd.read_parquet(DATA / "variants.parquet")
    doubles_all = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    proteins = pd.read_parquet(DATA / "proteins.parquet")
    proteins = proteins.loc[proteins.protein_id.isin(permissions["test"])
                            & proteins.protein_id.isin(doubles_all.protein_id)].sort_values("protein_id")
    destination = ROOT / "runs/stabilityarc_target" / f"family{args.threshold}/fold{args.fold}/seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / "manifest.json").exists()
    outputs = []
    started = time.monotonic()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for protein in proteins.itertuples():
        output = destination / protein.protein_id / "Sall"
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text())["complete"]
            outputs.append(str(manifest_path.relative_to(ROOT)))
            continue
        frame = variants.loc[variants.protein_id.eq(protein.protein_id)].reset_index(drop=True)
        doubles = doubles_all.loc[doubles_all.protein_id.eq(protein.protein_id)].copy()
        wt = frame.loc[frame.n_mutations.eq(0)]
        assert len(wt) == 1
        allowed_singles = order.loc[order.seed.eq(args.seed) & order.protein_id.eq(protein.protein_id)]
        allowed_ids = set(allowed_singles.variant_id) | set(wt.variant_id)
        training = frame.loc[frame.variant_id.isin(allowed_ids) & frame.point.notna()].copy()
        assert training.n_mutations.isin([0, 1]).all() and not set(training.variant_id) & set(doubles.variant_id)
        training.to_parquet(output / "allowed_labels.parquet", index=False)
        structural_path = ROOT / "data/features/proteinmpnn/tsuboyama/full" / protein.protein_id / "manifest.json"
        structural = json.loads(structural_path.read_text())
        assert structural["complete"] and structural["reference_sequence"] == protein.reference_sequence
        manifest = {"complete": True, "source_commit": commit, "source_run": str(source.relative_to(ROOT)),
                    "mode": "S", "condition": "Sall", "method": "stabilityarc_kermut", "seed": args.seed,
                    "threshold": args.threshold, "fold": args.fold, "protein_id": protein.protein_id,
                    "source_labels_used": source_manifest["source_labels_used"],
                    "target_single_labels": len(allowed_singles), "target_combination_labels": 0,
                    "target_point_labels_used": len(training), "target_test_labels_used": 0,
                    "structural_manifest": str(structural_path.relative_to(ROOT)),
                    "compound_model": "Original Kermut composite covariance and Gaussian likelihood; only the mean input is replaced by the source-held-out full StabilityArc score.",
                    "optimization": {"steps": 150, "optimizer": "AdamW", "lr": 0.1, "weight_decay": 0.01}}
        if len(training) < 2:
            manifest.update(evaluated=False, reason="insufficient_permitted_point_labels")
            manifest_path.write_text(json.dumps(manifest, indent=2))
            outputs.append(str(manifest_path.relative_to(ROOT)))
            continue
        inputs, index, wt_encoding = build_inputs(protein, frame, doubles)
        embedding, contacts = load_features(protein.protein_id, protein.reference_sequence, "cuda")
        with torch.inference_mode():
            raw, scores = score_arrays(model, embedding, contacts, mutation_arrays(frame, protein.reference_sequence, "cuda"))
            priors = dict(zip(frame.mutation_key, raw.cpu().numpy().tolist()))
            for row in doubles.itertuples():
                for key, position, mutant in zip(row.mutation_key.split(":"), row.positions, row.mut_residues):
                    if key not in priors:
                        priors[key] = float(scores[int(position), AMINO_ACID_TO_INDEX[mutant]])
        assert set(priors) == set(index) and priors[""] == 0.0
        prior = np.empty(len(inputs), dtype=np.float32)
        for key, row_index in index.items():
            prior[row_index] = priors[key]
        assert np.isfinite(prior).all()
        inputs[:, len(wt_encoding)] = torch.from_numpy(prior).cuda()
        pd.DataFrame({"mutation_key": list(index), "raw_stabilityarc_prior": [priors[key] for key in index]}).to_parquet(
            output / "prior.parquet", index=False)
        values, elapsed = fit_and_predict(inputs, [index[key] for key in training.mutation_key],
                                         training.point.to_numpy(), wt_encoding, structural, args.seed, output)
        prediction_wt = float(values[index[""]])
        evaluation = doubles.copy()
        evaluation["prediction_mu"] = [float(values[index[key]]) - prediction_wt for key in evaluation.mutation_key]
        endpoints = np.array([[index[key] for key in mutation.split(":")] for mutation in evaluation.mutation_key])
        evaluation["prediction_a"] = values[endpoints[:, 0]] - prediction_wt
        evaluation["prediction_b"] = values[endpoints[:, 1]] - prediction_wt
        evaluation["prediction_wt"] = 0.0
        evaluation["prediction_epsilon"] = evaluation.prediction_mu - evaluation.prediction_a - evaluation.prediction_b
        assert np.isfinite(evaluation[["prediction_mu", "prediction_a", "prediction_b", "prediction_epsilon"]]).all().all()
        evaluation.to_parquet(output / "predictions.parquet", index=False)
        manifest.update(evaluated=True, evaluated_variants=len(evaluation), elapsed_seconds=elapsed,
                        gpu=os.environ["CUDA_VISIBLE_DEVICES"], peak_vram_bytes=torch.cuda.max_memory_allocated())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        outputs.append(str(manifest_path.relative_to(ROOT)))
        print(json.dumps({"protein_id": protein.protein_id, "threshold": args.threshold, "fold": args.fold,
                          "seed": args.seed, "complete": True, "target_training_labels": len(training)}), flush=True)
        del inputs, embedding, contacts
        gc.collect()
        torch.cuda.empty_cache()
    (destination / "manifest.json").write_text(json.dumps({"complete": True, "config": vars(args),
                                                          "source_commit": commit, "outputs": outputs,
                                                          "elapsed_seconds": time.monotonic() - started}, indent=2))


if __name__ == "__main__":
    main()
