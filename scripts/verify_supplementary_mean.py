import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
source = ROOT / "data/features/mean" / ("flip_supplementary_smoke" if args.smoke else "flip_supplementary")
information = json.loads((source / "manifest.json").read_text())
assert information["complete"] and information["smoke"] == args.smoke
metadata = pd.read_parquet(source / "sequence_metadata.parquet")
matrix = np.load(source / "sequence_mean.npy", mmap_mode="r")
assert metadata.sequence.is_unique and matrix.shape == (len(metadata), 1280)
np.testing.assert_array_equal(metadata.matrix_row, np.arange(len(metadata)))
model_path = ROOT / "models/esm2_t33_650M_UR50D"
assert json.loads((model_path / "download_manifest.json").read_text())["revision"] == information["model_revision"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = EsmForMaskedLM.from_pretrained(model_path, local_files_only=True).cuda().eval()
maximum, checked, covered = 0., 0, []
for rank in range(8):
    part = pd.read_parquet(source / f"rank{rank}" / "metadata.parquet")
    data = np.load(source / f"rank{rank}" / "mean.npy", mmap_mode="r")
    np.testing.assert_array_equal(data, matrix[part.matrix_row.to_numpy(dtype=int)])
    np.testing.assert_array_equal(part.sequence, metadata.iloc[part.matrix_row].sequence)
    for length, group in part.groupby("length", sort=True):
        selected = group if args.smoke else group.iloc[:16]
        for start in range(0, len(selected), 16):
            batch = selected.iloc[start:start + 16]
            tokens = tokenizer(batch.sequence.tolist(), padding=True, return_tensors="pt")
            assert tokens.input_ids.shape == (len(batch), int(length) + 2)
            assert tokens.attention_mask.eq(1).all()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = model(**{key: value.cuda() for key, value in tokens.items()}, output_hidden_states=True)
            residue_states = output.hidden_states[-1][:, 1:-1].float().cpu().numpy()
            expected = residue_states.mean(axis=1).astype(np.float16)
            observed = data[batch.local_row.to_numpy(dtype=int)]
            difference = float(np.max(np.abs(expected.astype(np.float32) - observed.astype(np.float32))))
            maximum = max(maximum, difference)
            np.testing.assert_allclose(expected, observed, atol=1e-5, rtol=0)
            checked += len(batch)
            covered.append(int(length))
datasets = []
if not args.smoke:
    canonical = pd.read_parquet(ROOT / "data/processed/flip_supplementary/variants.parquet")
    lookup = metadata.set_index("sequence").matrix_row
    for dataset, group in canonical.groupby("dataset", sort=True):
        artifact = pd.read_parquet(source / (dataset + "_metadata.parquet"))
        values = np.load(source / (dataset + "_mean.npy"), mmap_mode="r")
        np.testing.assert_array_equal(artifact.variant_id, group.variant_id)
        first = matrix[artifact.sequence_a.map(lookup).to_numpy(dtype=int)]
        np.testing.assert_array_equal(values[:, :1280], first)
        if dataset == "pdz3":
            for index, row in enumerate(artifact.itertuples()):
                if row.sequence_b:
                    np.testing.assert_array_equal(values[index, 1280:2560], matrix[lookup[row.sequence_b]])
                else:
                    np.testing.assert_array_equal(values[index, 1280:2560], 0.)
                np.testing.assert_array_equal(values[index, -2:], [len(row.sequence_a), len(row.sequence_b)])
        else:
            assert values.shape[1] == 1280
        datasets.append({"dataset": dataset, "inputs": len(artifact), "width": values.shape[1]})
destination = ROOT / "runs" / ("supplementary_features_smoke_verification" if args.smoke else "supplementary_features_verification")
destination.mkdir(parents=True, exist_ok=True)
result = {"complete": True, "smoke": args.smoke, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "cache_sequences_verified": len(metadata), "independent_model_forward_sequences": checked,
          "maximum_stored_mean_difference": maximum, "tested_lengths": sorted(set(covered)),
          "datasets": datasets, "separate_chain_and_empty_peptide_encoding_verified": not args.smoke}
(destination / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
