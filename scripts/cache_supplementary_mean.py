import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from cache_esm2 import Encoder, REVISION

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int)
parser.add_argument("--world-size", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--assemble", action="store_true")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
assert args.world_size == 8
source = ROOT / "data/processed/flip_supplementary"
assert json.loads((source / "manifest.json").read_text())["complete"]
variants = pd.read_parquet(source / "variants.parquet")
sequences = sorted((set(variants.sequence_a) | set(variants.sequence_b)) - {""}, key=lambda value: (len(value), value))
full_count = len(sequences)
if args.smoke:
    sequences = sequences[::100]
metadata = pd.DataFrame({"sequence": sequences, "length": [len(sequence) for sequence in sequences]})
metadata["matrix_row"] = np.arange(len(metadata))
output = ROOT / "data/features/mean" / ("flip_supplementary_smoke" if args.smoke else "flip_supplementary")
output.mkdir(parents=True, exist_ok=True)
commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
if not args.assemble:
    assert args.rank is not None and 0 <= args.rank < 8
    directory = output / f"rank{args.rank}"
    directory.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        info = json.loads(manifest_path.read_text())
        assert info["complete"] and info["smoke"] == args.smoke and info["model_revision"] == REVISION
    else:
        current = metadata.iloc[args.rank::8].copy()
        assert len(current) > 0
        encoder = Encoder(args.batch_size)
        matrix = np.lib.format.open_memmap(directory / "mean.npy.part", mode="w+", dtype=np.float16, shape=(len(current), 1280))
        current["local_row"] = np.arange(len(current))
        filled = np.zeros(len(current), dtype=bool)
        started = time.monotonic()
        for length, group in current.groupby("length", sort=True):
            for start in range(0, len(group), args.batch_size):
                batch = group.iloc[start:start + args.batch_size]
                assert batch.sequence.str.len().eq(length).all() and length > 0
                hidden = encoder.embeddings(batch.sequence.tolist())
                assert hidden.shape == (len(batch), length, 1280)
                rows = batch.local_row.to_numpy(dtype=int)
                assert not filled[rows].any()
                matrix[rows] = hidden.mean(axis=1).astype(np.float16)
                filled[rows] = True
            print(json.dumps({"rank": args.rank, "length": int(length), "completed_sequences": int(filled.sum())}), flush=True)
        assert filled.all() and np.isfinite(matrix).all()
        matrix.flush()
        del matrix
        os.replace(directory / "mean.npy.part", directory / "mean.npy")
        current.to_parquet(directory / "metadata.parquet", index=False)
        information = {"complete": True, "smoke": args.smoke, "rank": args.rank, "world_size": 8,
                       "source_commit": commit, "sequences": len(current), "full_unique_sequences": full_count,
                       "model_revision": REVISION, "model": "ESM-2 650M", "batch_size": args.batch_size,
                       "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                       "precision": "Float16 autocast; residue means accumulated in float32; stored float16.",
                       "elapsed_seconds": time.monotonic() - started}
        manifest_path.write_text(json.dumps(information, indent=2))
        print(json.dumps(information), flush=True)
else:
    mean = np.empty((len(metadata), 1280), dtype=np.float16)
    filled = np.zeros(len(metadata), dtype=bool)
    for rank in range(8):
        directory = output / f"rank{rank}"
        info = json.loads((directory / "manifest.json").read_text())
        assert info["complete"] and info["smoke"] == args.smoke and info["model_revision"] == REVISION
        part = pd.read_parquet(directory / "metadata.parquet")
        rows = part.matrix_row.to_numpy(dtype=int)
        np.testing.assert_array_equal(part.sequence, metadata.loc[rows, "sequence"])
        assert not filled[rows].any()
        values = np.load(directory / "mean.npy", mmap_mode="r")
        assert values.shape == (len(rows), 1280)
        mean[rows] = values
        filled[rows] = True
    assert filled.all() and np.isfinite(mean).all()
    np.save(output / "sequence_mean.npy", mean)
    metadata.to_parquet(output / "sequence_metadata.parquet", index=False)
    artifacts = []
    if not args.smoke:
        lookup = metadata.set_index("sequence").matrix_row
        for dataset, group in variants.groupby("dataset", sort=True):
            group = group.copy().reset_index(drop=True)
            group["matrix_row"] = np.arange(len(group))
            group["embedding_a_row"] = group.sequence_a.map(lookup)
            assert group.embedding_a_row.notna().all()
            first = mean[group.embedding_a_row.to_numpy(dtype=int)]
            if dataset == "pdz3":
                group["embedding_b_row"] = group.sequence_b.map(lookup)
                present = group.sequence_b.ne("").to_numpy()
                assert group.loc[present, "embedding_b_row"].notna().all()
                second = np.zeros_like(first)
                second[present] = mean[group.loc[present, "embedding_b_row"].to_numpy(dtype=int)]
                # 空肽使用零向量，并通过真实链长明确表示该组分缺失。
                lengths = np.column_stack([group.sequence_a.str.len(), group.sequence_b.str.len()]).astype(np.float16)
                matrix = np.concatenate([first, second, lengths], axis=1)
                assert (matrix[~present, 1280:2560] == 0).all() and (matrix[~present, -1] == 0).all()
            else:
                matrix = first
            assert np.isfinite(matrix).all()
            np.save(output / (dataset + "_mean.npy"), matrix)
            group.to_parquet(output / (dataset + "_metadata.parquet"), index=False)
            artifacts.append({"dataset": dataset, "inputs": len(group), "width": matrix.shape[1],
                              "official_inputs": int(group.official_input.sum())})
    manifest = {"complete": True, "smoke": args.smoke, "source_commit": commit, "model_revision": REVISION,
                "encoded_sequences": len(metadata), "full_unique_sequences": full_count,
                "fraction": len(metadata) / full_count, "gpus": list(range(8)), "datasets": artifacts,
                "label_inputs_used": 0, "separate_component_encoding": True, "counterfactual_inputs_included": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
