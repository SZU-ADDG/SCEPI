import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = ROOT / "data/features/mean" / args.dataset
    assert json.loads((root / "manifest.json").read_text())["complete"]
    metadata = pd.read_parquet(root / "metadata.parquet")
    references = pd.read_parquet(root / "references.parquet")
    values = np.load(root / "mean.npy", mmap_mode="r")
    reference = np.load(root / "reference_mean.npy", mmap_mode="r")
    outputs = []
    for protein in references.itertuples():
        directory = root / "counterfactuals" / protein.protein_id
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text())["complete"]
            outputs.append(str(manifest_path.relative_to(ROOT)))
            continue
        observed = metadata.loc[metadata.protein_id.eq(protein.protein_id) & metadata.n_mutations.eq(1)]
        lookup = observed.set_index("mutation_key").matrix_row
        source = ROOT / "data/features/esm2" / args.dataset / "full" / f"{protein.protein_id}_singles.h5"
        with h5py.File(source, "r") as handle:
            assert handle.attrs["reference_sequence"] == protein.reference_sequence
            keys = handle["keys"].asstr()[:]
            matrix = np.empty((len(keys), 1280), dtype=np.float32)
            present = np.asarray([key in lookup for key in keys])
            for index in np.flatnonzero(~present):
                matrix[index] = handle["single"][index].astype(np.float32).mean(axis=0).astype(np.float16).astype(np.float32)
            matrix[present] = values[lookup.reindex(keys[present]).to_numpy(dtype=int)].astype(np.float32)
        assert np.isfinite(matrix).all()
        np.save(directory / "single_mean.npy", matrix)
        np.save(directory / "reference_mean.npy", reference[protein.Index].astype(np.float32))
        pd.DataFrame({"mutation_key": keys, "observed_variant": present}).to_parquet(directory / "metadata.parquet", index=False)
        result = {"complete": True, "dataset": args.dataset, "protein_id": protein.protein_id,
                  "single_genotypes": len(keys), "measured_single_genotypes": int(present.sum()),
                  "prediction_only_single_genotypes": int((~present).sum()),
                  "feature_policy": "Observed genotypes reuse the exact mean-embedding cache. Unobserved component singles use their previously encoded sequence embeddings.",
                  "source_cache": str(source.relative_to(ROOT)), "phenotype_labels_used": 0,
                  "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
        manifest_path.write_text(json.dumps(result, indent=2))
        outputs.append(str(manifest_path.relative_to(ROOT)))
        print(json.dumps(result), flush=True)
    output = {"complete": True, "dataset": args.dataset, "proteins": len(references), "outputs": outputs}
    (root / "counterfactuals/manifest.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
