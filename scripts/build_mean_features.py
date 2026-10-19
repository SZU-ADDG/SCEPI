import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tsuboyama")
    args = parser.parse_args()
    source = ROOT / "data/processed" / args.dataset
    cache = ROOT / "data/features/esm2" / args.dataset / "full"
    output = ROOT / "data/features/mean" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    variants = pd.read_parquet(source / "variants.parquet").sort_values("variant_id").reset_index(drop=True)
    proteins = pd.read_parquet(source / "proteins.parquet").sort_values("protein_id").reset_index(drop=True)
    variants["matrix_row"] = np.arange(len(variants))
    protein_index = pd.Series(proteins.index.to_numpy(), index=proteins.protein_id)
    variants["reference_index"] = variants.protein_id.map(protein_index)
    assert variants.reference_index.notna().all()
    matrix = np.lib.format.open_memmap(output / "mean.npy.part", mode="w+", dtype=np.float16, shape=(len(variants), 1280))
    reference_matrix = np.empty((len(proteins), 1280), dtype=np.float16)
    scores = np.empty((len(variants), 2), dtype=np.float32)
    completed = np.zeros(len(variants), dtype=bool)
    for protein in proteins.itertuples():
        group = variants.loc[variants.protein_id.eq(protein.protein_id)]
        path = cache / f"{protein.protein_id}_variants.h5"
        with h5py.File(path, "r") as handle:
            assert handle.attrs["reference_sequence"] == protein.reference_sequence
            identifiers = handle["variant_ids"].asstr()[:]
            assert set(identifiers) == set(group.variant_id)
            rows = group.set_index("variant_id").matrix_row.reindex(identifiers).to_numpy()
            assert not completed[rows].any()
            matrix[rows] = handle["mean"][:]
            scores[rows, 0] = handle["additive_score"][:]
            scores[rows, 1] = handle["context_score"][:]
            completed[rows] = True
        wild_type = group.loc[group.n_mutations.eq(0)]
        assert len(wild_type) <= 1
        if len(wild_type):
            reference_matrix[protein.Index] = matrix[wild_type.matrix_row.iloc[0]]
        else:
            with h5py.File(cache / f"{protein.protein_id}_singles.h5", "r") as handle:
                reference_matrix[protein.Index] = handle["wt"][:].astype(np.float32).mean(axis=0)
        print(json.dumps({"protein": protein.protein_id, "variants": len(group)}), flush=True)
    assert completed.all()
    assert np.isfinite(matrix).all() and np.isfinite(reference_matrix).all() and np.isfinite(scores).all()
    matrix.flush()
    del matrix
    os.replace(output / "mean.npy.part", output / "mean.npy")
    np.save(output / "reference_mean.npy", reference_matrix)
    np.save(output / "scores.npy", scores)
    variants.to_parquet(output / "metadata.parquet", index=False)
    proteins.to_parquet(output / "references.parquet", index=False)
    manifest = {
        "complete": True, "dataset": args.dataset, "variants": len(variants), "proteins": len(proteins),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_cache": str(cache.relative_to(ROOT)), "score_columns": ["additive_score", "context_score"],
        "embedding_shape": [len(variants), 1280], "embedding_dtype": "float16",
        "reference_embedding": "The matching WT variant embedding where recorded; otherwise the encoded reference sequence stored in the WT cache.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
